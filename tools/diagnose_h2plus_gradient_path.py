from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optax

from h2plus_fci_ground_train5_dense100 import (
    _DEFAULT_SEMILOCAL_XC,
    _TRAIN_SCF_SAFETY_MAX_CYCLE,
    RunLogger,
    build_training_data,
    get_or_build_reference_point,
    parse_args as parse_train_args,
)
from td_graddft import neural_xc
from td_graddft.features import grid_features_for_molecule
from td_graddft.training import (
    GroundStateCoreTrainingConfig,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    ground_state_mse_loss_pointwise_dataset,
    load_params_checkpoint,
    make_ground_state_loss_and_grad,
)
from td_graddft.training.targets import (
    _make_differentiable_scf,
    _predict_ground_state_total_energy_from_molecule,
)


def _tree_norm(tree: Any) -> float:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return 0.0
    total = sum(
        jnp.sum(jnp.square(jnp.asarray(leaf).astype(jnp.float64)))
        for leaf in leaves
    )
    return float(jnp.sqrt(total))


def _tree_abs_max(tree: Any) -> float:
    values = [
        float(jnp.max(jnp.abs(jnp.asarray(leaf))))
        for leaf in jax.tree_util.tree_leaves(tree)
    ]
    return max(values) if values else 0.0


def _top_leaf_norms(tree: Any, limit: int = 8) -> list[dict[str, Any]]:
    rows = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        key = "/".join(str(getattr(part, "key", part)) for part in path)
        arr = jnp.asarray(leaf)
        rows.append(
            {
                "path": key,
                "shape": tuple(int(v) for v in arr.shape),
                "norm": float(jnp.linalg.norm(arr)),
                "abs_max": float(jnp.max(jnp.abs(arr))),
            }
        )
    return sorted(rows, key=lambda row: -row["norm"])[:limit]


def _tree_zeros_like(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda leaf: jnp.zeros_like(jnp.asarray(leaf)), tree)


def _tree_add_scaled(tree: Any, direction: Any, scale: float) -> Any:
    return jax.tree_util.tree_map(
        lambda leaf, delta: jnp.asarray(leaf) + float(scale) * jnp.asarray(delta),
        tree,
        direction,
    )


def _initial_dense_kernel_direction(params: Any) -> Any:
    direction = _tree_zeros_like(params)
    kernel = params["params"]["InitialDense"]["kernel"]
    unit = jnp.ones_like(kernel) / jnp.sqrt(jnp.asarray(kernel.size, dtype=kernel.dtype))
    direction["params"]["InitialDense"]["kernel"] = unit
    return direction


def _tree_dot(lhs: Any, rhs: Any) -> float:
    total = jnp.asarray(0.0, dtype=jnp.float64)
    for left, right in zip(jax.tree_util.tree_leaves(lhs), jax.tree_util.tree_leaves(rhs), strict=True):
        total = total + jnp.sum(jnp.asarray(left, dtype=jnp.float64) * jnp.asarray(right, dtype=jnp.float64))
    return float(total)


def _training_config(args: argparse.Namespace, mode: str) -> GroundStateTrainingConfig:
    coefficient_prior = neural_xc.resolve_coefficient_prior_values(
        tuple(_DEFAULT_SEMILOCAL_XC)
    )
    return GroundStateTrainingConfig.from_parts(
        core=GroundStateCoreTrainingConfig(
            mode="self_consistent",
            energy_mse_weight=float(args.energy_mse_weight),
            energy_mae_weight=float(args.energy_mae_weight),
            energy_normalization=str(args.energy_normalization),
            coefficient_prior_weight=0.0,
            coefficient_prior_values=coefficient_prior,
            scf_max_cycle=(
                _TRAIN_SCF_SAFETY_MAX_CYCLE
                if int(args.train_scf_max_cycle) <= 0
                else int(args.train_scf_max_cycle)
            ),
            scf_damping=float(args.train_scf_damping),
            scf_conv_tol_energy=args.train_scf_conv_tol_energy,
            scf_convergence_metric=str(args.train_scf_convergence_metric),
            scf_conv_tol_density=float(args.train_scf_conv_tol_density),
            scf_vxc_clip=float(args.train_scf_vxc_clip),
            scf_iterate_selection=str(args.scf_iterate_selection),
            scf_gradient_mode=str(mode),
            scf_implicit_diff_tolerance=float(args.scf_implicit_diff_tolerance),
            scf_implicit_diff_regularization=float(args.scf_implicit_diff_regularization),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--reference-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--basis", default="def2-svp")
    parser.add_argument("--grids-level", type=int, default=2)
    parser.add_argument("--train-points", type=int, default=5)
    parser.add_argument("--integral-backend", default="gpu")
    parser.add_argument("--reference-scf-device", default="gpu")
    parser.add_argument("--modes", nargs="+", default=["impl", "expl"])
    parser.add_argument("--point-index", type=int, default=None)
    parser.add_argument("--fd-eps", type=float, default=1e-4)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    train_args = parse_train_args(
        [
            "--basis",
            str(args.basis),
            "--grids-level",
            str(args.grids_level),
            "--train-points",
            str(args.train_points),
            "--dense-points",
            "1",
            "--steps",
            "1",
            "--density-constraint-weight",
            "0.0",
            "--density-matrix-constraint-weight",
            "0.0",
            "--reference-cache",
            str(args.reference_cache),
            "--reference-scf-device",
            str(args.reference_scf_device),
            "--integral-backend",
            str(args.integral_backend),
            "--outdir",
            str(out.parent),
        ]
    )
    logger = RunLogger(out.parent / "grad_diag_ref.log")
    points = [
        get_or_build_reference_point(float(r), args=train_args, logger=logger)
        for r in jnp.linspace(
            float(train_args.r_min),
            float(train_args.r_max),
            int(train_args.train_points),
        )
    ]
    train_data = build_training_data(
        points,
        density_constraint_weight=0.0,
        density_matrix_constraint_weight=0.0,
    )
    if args.point_index is not None:
        point_index = int(args.point_index)
        train_data = (train_data[point_index],)
        points = [points[point_index]]
    functional = neural_xc.Functional(
        semilocal_xc=tuple(_DEFAULT_SEMILOCAL_XC),
        hidden_dims=tuple(int(value) for value in train_args.hidden_dims),
        architecture=str(train_args.network_architecture),
        input_feature_mode=str(train_args.input_feature_mode),
        include_pt2_channel=False,
        name="neural_xc_h2plus_fci_ground",
    )
    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(0),
        points[0].molecule,
        optax.adam(1e-5),
    )
    params_by_label = {"initial": state.params}
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            params_by_label["checkpoint"] = load_params_checkpoint(
                checkpoint_path,
                template=state.params,
            )

    results = []
    mid_molecule = points[len(points) // 2].molecule
    features = grid_features_for_molecule(mid_molecule)
    for mode in tuple(str(value) for value in args.modes):
        cfg = _training_config(train_args, mode)
        loss_grad_fn = make_ground_state_loss_and_grad(
            functional,
            training_config=cfg,
            loss_fn=ground_state_mse_loss_pointwise_dataset,
        )
        for label, params in params_by_label.items():
            loss, metrics, grads = loss_grad_fn(params, train_data)
            scf = _make_differentiable_scf(cfg)
            scf_molecule, scf_info = scf.run(mid_molecule, functional, params)

            def fixed_xc_energy(local_params: Any) -> jnp.ndarray:
                return functional.energy_from_molecule(local_params, mid_molecule)

            def fixed_total_energy(local_params: Any) -> jnp.ndarray:
                return _predict_ground_state_total_energy_from_molecule(
                    local_params,
                    functional,
                    mid_molecule,
                )

            def self_consistent_total_energy(local_params: Any) -> jnp.ndarray:
                local_molecule, _ = scf.run(mid_molecule, functional, local_params)
                return _predict_ground_state_total_energy_from_molecule(
                    local_params,
                    functional,
                    local_molecule,
                )

            def self_consistent_density_sum(local_params: Any) -> jnp.ndarray:
                local_molecule, _ = scf.run(mid_molecule, functional, local_params)
                return jnp.sum(jnp.asarray(local_molecule.rdm1))

            fixed_xc_grads = jax.grad(fixed_xc_energy)(params)
            fixed_total_grads = jax.grad(fixed_total_energy)(params)
            sc_total_grads = jax.grad(self_consistent_total_energy)(params)
            sc_density_grads = jax.grad(self_consistent_density_sum)(params)

            def coeff_sum(local_params: Any) -> jnp.ndarray:
                coeff = functional.channel_coefficients(
                    local_params,
                    features,
                    molecule=mid_molecule,
                )
                return jnp.sum(coeff)

            coeff = functional.channel_coefficients(params, features, molecule=mid_molecule)
            coeff_grads = jax.grad(coeff_sum)(params)
            direction = _initial_dense_kernel_direction(params)
            eps = float(args.fd_eps)
            loss_plus, _ = ground_state_mse_loss_pointwise_dataset(
                _tree_add_scaled(params, direction, eps),
                functional,
                train_data,
                training_config=cfg,
            )
            loss_minus, _ = ground_state_mse_loss_pointwise_dataset(
                _tree_add_scaled(params, direction, -eps),
                functional,
                train_data,
                training_config=cfg,
            )
            fd_loss_directional = float((loss_plus - loss_minus) / (2.0 * eps))
            results.append(
                {
                    "mode": mode,
                    "label": label,
                    "loss": float(loss),
                    "energy_mae_mean_h": float(jnp.mean(metrics["energy_mae"])),
                    "scf_converged_mean": float(jnp.mean(metrics["scf_converged"])),
                    "scf_cycles": [float(v) for v in metrics["scf_cycles"]],
                    "loss_grad_norm": _tree_norm(grads),
                    "loss_grad_abs_max": _tree_abs_max(grads),
                    "nonfinite_grad_fraction": float(metrics["nonfinite_grad_fraction"][0]),
                    "param_norm": _tree_norm(params),
                    "top_loss_grad_leaves": _top_leaf_norms(grads),
                    "fixed_xc_energy_h": float(fixed_xc_energy(params)),
                    "fixed_xc_grad_norm": _tree_norm(fixed_xc_grads),
                    "fixed_xc_grad_abs_max": _tree_abs_max(fixed_xc_grads),
                    "top_fixed_xc_grad_leaves": _top_leaf_norms(fixed_xc_grads),
                    "fixed_total_energy_h": float(fixed_total_energy(params)),
                    "fixed_total_grad_norm": _tree_norm(fixed_total_grads),
                    "fixed_total_grad_abs_max": _tree_abs_max(fixed_total_grads),
                    "top_fixed_total_grad_leaves": _top_leaf_norms(fixed_total_grads),
                    "self_consistent_total_energy_h": float(self_consistent_total_energy(params)),
                    "self_consistent_grad_norm": _tree_norm(sc_total_grads),
                    "self_consistent_grad_abs_max": _tree_abs_max(sc_total_grads),
                    "top_self_consistent_grad_leaves": _top_leaf_norms(sc_total_grads),
                    "self_consistent_density_sum": float(self_consistent_density_sum(params)),
                    "self_consistent_density_grad_norm": _tree_norm(sc_density_grads),
                    "self_consistent_density_grad_abs_max": _tree_abs_max(sc_density_grads),
                    "self_consistent_mid_scf_converged": float(jnp.asarray(scf_info.converged)),
                    "self_consistent_mid_scf_cycles": float(jnp.asarray(scf_info.cycles)),
                    "coeff_mean_mid": [float(v) for v in jnp.mean(coeff, axis=0)],
                    "coeff_grad_norm": _tree_norm(coeff_grads),
                    "coeff_grad_abs_max": _tree_abs_max(coeff_grads),
                    "top_coeff_grad_leaves": _top_leaf_norms(coeff_grads),
                    "initial_dense_kernel_loss_grad_dot": _tree_dot(grads, direction),
                    "initial_dense_kernel_loss_fd_directional": fd_loss_directional,
                    "initial_dense_kernel_fd_abs_error": abs(
                        _tree_dot(grads, direction) - fd_loss_directional
                    ),
                }
            )
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
