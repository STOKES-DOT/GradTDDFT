from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax

from td_graddft.features import restricted_transition_response_features
from td_graddft.features import restricted_grid_features_with_gradients
from td_graddft.tddft.response import (
    build_restricted_tda_matrix,
    refresh_restricted_response_eri_slices,
)
from td_graddft.training import (
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    ground_state_mse_loss,
    make_ground_state_loss_and_grad,
    predict_excitation_energies,
    predict_ground_state_molecule,
)


_H2_PATH = Path(__file__).with_name("h2_s1_tda_train5_dense100_vs_fci.py")
_H2_SPEC = importlib.util.spec_from_file_location("_h2_s1_training_for_memory_profile", _H2_PATH)
if _H2_SPEC is None or _H2_SPEC.loader is None:
    raise RuntimeError(f"Failed to load H2 training helpers from {_H2_PATH}")
_H2 = importlib.util.module_from_spec(_H2_SPEC)
sys.modules[_H2_SPEC.name] = _H2
_H2_SPEC.loader.exec_module(_H2)


class _Logger:
    def log(self, message: str) -> None:
        print(message, flush=True)


class _GpuSampler:
    def __init__(self, gpu_id: str | None, interval_s: float) -> None:
        self.gpu_id = gpu_id
        self.interval_s = float(interval_s)
        self.peak_mib: int | None = None
        self.total_mib: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _query(self) -> tuple[int, int] | None:
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_id}" if self.gpu_id else "--query-gpu=memory.used,memory.total",
            "--query-gpu=memory.used,memory.total" if self.gpu_id else "--format=csv,noheader,nounits",
            "--format=csv,noheader,nounits" if self.gpu_id else "",
        ]
        cmd = [part for part in cmd if part]
        try:
            proc = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        except Exception:
            return None
        line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            return None
        try:
            return int(float(parts[0])), int(float(parts[1]))
        except ValueError:
            return None

    def sample_once(self) -> tuple[int, int] | None:
        sample = self._query()
        if sample is not None:
            used, total = sample
            self.peak_mib = used if self.peak_mib is None else max(self.peak_mib, used)
            self.total_mib = total
        return sample

    def start(self) -> None:
        self.sample_once()

        def run() -> None:
            while not self._stop.wait(self.interval_s):
                self.sample_once()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.sample_once()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def _default_gpu_id() -> str | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return visible.split(",")[0].strip()
    return "0"


def _block_until_ready(value: Any) -> Any:
    try:
        return jax.block_until_ready(value)
    except Exception:
        for leaf in jax.tree_util.tree_leaves(value):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
        return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (jnp.ndarray, np.ndarray)):
        arr = np.asarray(jax.device_get(value))
        return arr.tolist() if arr.ndim else arr.item()
    return value


def _make_h2_args(args: argparse.Namespace) -> argparse.Namespace:
    h2_args = _H2.parse_args([])
    h2_args.basis = str(args.basis)
    h2_args.r_min = float(args.r)
    h2_args.r_max = float(args.r)
    h2_args.train_points = 1
    h2_args.dense_points = 1
    h2_args.steps = 1
    h2_args.grids_level = int(args.grids_level)
    h2_args.integral_backend = str(args.integral_backend)
    h2_args.grid_ao_backend = str(args.grid_ao_backend)
    h2_args.jk_backend = str(args.jk_backend)
    h2_args.reference_scf_backend = str(args.reference_scf_backend)
    h2_args.reference_cache = args.reference_cache
    h2_args.rebuild_reference_cache = bool(args.rebuild_reference_cache)
    h2_args.include_pt2_channel = bool(args.include_pt2_channel)
    h2_args.pt2_channel_mode = str(args.pt2_channel_mode)
    h2_args.response_pt2_mode = str(args.response_pt2_mode)
    h2_args.response_grid_chunk_size = int(args.response_grid_chunk_size)
    h2_args.training_mode = str(args.training_mode)
    h2_args.s1_use_tda = bool(args.s1_use_tda)
    h2_args.energy_mse_weight = 0.0
    h2_args.energy_mae_weight = 0.0
    h2_args.density_constraint_weight = 0.0
    h2_args.s1_weight = 1.0
    h2_args.train_scf_max_cycle = int(args.train_scf_max_cycle)
    h2_args.train_scf_damping = float(args.train_scf_damping)
    h2_args.train_scf_conv_tol_energy = args.train_scf_conv_tol_energy
    h2_args.train_scf_convergence_metric = str(args.train_scf_convergence_metric)
    h2_args.train_scf_conv_tol_density = float(args.train_scf_conv_tol_density)
    h2_args.scf_gradient_mode = str(args.scf_gradient_mode)
    h2_args.scf_implicit_diff_solver = str(args.scf_implicit_diff_solver)
    h2_args.scf_implicit_diff_tolerance = float(args.scf_implicit_diff_tolerance)
    h2_args.scf_implicit_diff_regularization = float(args.scf_implicit_diff_regularization)
    h2_args.scf_implicit_diff_restart = int(args.scf_implicit_diff_restart)
    h2_args.scf_iterate_selection = str(args.scf_iterate_selection)
    h2_args.scf_require_convergence = bool(args.scf_require_convergence)
    h2_args.scf_stop_gradient_on_unconverged = bool(args.scf_stop_gradient_on_unconverged)
    h2_args.scf_stop_gradient_rms_threshold = args.scf_stop_gradient_rms_threshold
    h2_args.jit_train = bool(args.jit_kernels)
    h2_args.jit_eval = bool(args.jit_kernels)
    h2_args.outdir = str(args.outdir)
    return _H2._normalize_args(h2_args)


def _training_config(args: argparse.Namespace, h2_args: argparse.Namespace) -> GroundStateTrainingConfig:
    return GroundStateTrainingConfig(
        mode=str(args.training_mode),
        energy_mse_weight=0.0,
        energy_mae_weight=0.0,
        s1_constraint_use_tda=bool(args.s1_use_tda),
        scf_max_cycle=_H2._HELPERS._resolve_train_scf_max_cycle(args.train_scf_max_cycle),
        scf_damping=float(args.train_scf_damping),
        scf_conv_tol_energy=args.train_scf_conv_tol_energy,
        scf_convergence_metric=str(args.train_scf_convergence_metric),
        scf_conv_tol_density=float(args.train_scf_conv_tol_density),
        scf_vxc_clip=float(h2_args.train_scf_vxc_clip),
        scf_iterate_selection=str(args.scf_iterate_selection),
        scf_require_convergence=bool(args.scf_require_convergence),
        scf_stop_gradient_on_unconverged=bool(args.scf_stop_gradient_on_unconverged),
        scf_stop_gradient_rms_threshold=args.scf_stop_gradient_rms_threshold,
        scf_gradient_mode=str(args.scf_gradient_mode),
        scf_implicit_diff_solver=str(args.scf_implicit_diff_solver),
        scf_implicit_diff_tolerance=float(args.scf_implicit_diff_tolerance),
        scf_implicit_diff_regularization=float(args.scf_implicit_diff_regularization),
        scf_implicit_diff_restart=int(args.scf_implicit_diff_restart),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile GPU memory by phase for one H2 S1/TDA training datum.")
    p.add_argument("--basis", default="def2-svp")
    p.add_argument("--r", type=float, default=0.7)
    p.add_argument("--grids-level", type=int, default=2)
    p.add_argument("--integral-backend", choices=("jax", "cpu", "gpu", "libcint"), default="gpu")
    p.add_argument("--grid-ao-backend", choices=("jax", "pyscf"), default="jax")
    p.add_argument("--jk-backend", choices=("full", "df"), default="full")
    p.add_argument("--reference-scf-backend", choices=("pyscf", "jax_rks"), default="pyscf")
    p.add_argument("--reference-cache", default="outputs/reference_cache/h2_s1_memory_profile.h5")
    p.add_argument("--rebuild-reference-cache", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--include-pt2-channel", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--pt2-channel-mode", choices=("scaled_projected", "local_exact"), default="scaled_projected")
    p.add_argument("--response-pt2-mode", choices=("approx", "strict"), default="approx")
    p.add_argument("--response-grid-chunk-size", type=int, default=1024)
    p.add_argument("--training-mode", choices=("fixed_density", "self_consistent"), default="self_consistent")
    p.add_argument("--s1-use-tda", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--train-scf-max-cycle", type=int, default=0)
    p.add_argument("--train-scf-damping", type=float, default=0.25)
    p.add_argument("--train-scf-conv-tol-energy", type=float, default=1e-6)
    p.add_argument(
        "--train-scf-convergence-metric",
        choices=("energy_and_residual", "energy"),
        default="energy",
    )
    p.add_argument("--train-scf-conv-tol-density", type=float, default=1e-8)
    p.add_argument("--scf-iterate-selection", choices=("final", "best_rms", "first_converged"), default="best_rms")
    p.add_argument("--scf-gradient-mode", choices=("impl", "expl"), default="impl")
    p.add_argument("--scf-implicit-diff-solver", choices=("normal_cg", "gmres", "bicgstab"), default="normal_cg")
    p.add_argument("--scf-implicit-diff-tolerance", type=float, default=1e-6)
    p.add_argument("--scf-implicit-diff-regularization", type=float, default=1e-3)
    p.add_argument("--scf-implicit-diff-restart", type=int, default=12)
    p.add_argument("--scf-require-convergence", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--scf-stop-gradient-on-unconverged", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--scf-stop-gradient-rms-threshold", type=float, default=None)
    p.add_argument("--jit-kernels", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--skip-pregrad-probes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip standalone TDA/S1/loss forward probes and go directly to loss+grad.",
    )
    p.add_argument("--gpu-id", default=None, help="nvidia-smi GPU id/index. Defaults to first CUDA_VISIBLE_DEVICES entry.")
    p.add_argument("--sample-interval", type=float, default=0.2)
    p.add_argument(
        "--profile-bind-components",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Decompose bind_to_molecule_for_response into internal component phases.",
    )
    p.add_argument(
        "--materialize-grid-response-tensor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Explicitly call bound.grid_response_tensor after binding.",
    )
    p.add_argument(
        "--stop-after-bind-components",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop after decomposing bind_to_molecule_for_response into component phases.",
    )
    p.add_argument("--out-json", default=None)
    p.add_argument("--outdir", default="outputs/h2_s1_memory_profile")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    h2_args = _make_h2_args(args)
    logger = _Logger()
    _H2._HELPERS._load_runtime_dependencies(logger)
    gpu_id = str(args.gpu_id) if args.gpu_id is not None else _default_gpu_id()
    phase_rows: list[dict[str, Any]] = []

    def write_report() -> None:
        if args.out_json is None:
            return
        out_path = Path(str(args.out_json))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"args": vars(args), "phases": phase_rows}, default=_json_safe, indent=2),
            encoding="utf-8",
        )

    def print_ranking() -> None:
        ranked = sorted(
            (row for row in phase_rows if row["gpu_mem_peak_delta_mib"] is not None),
            key=lambda row: int(row["gpu_mem_peak_delta_mib"]),
            reverse=True,
        )
        print("[profile] peak delta ranking:", flush=True)
        for row in ranked:
            print(
                f"[profile] {row['phase']}: "
                f"peak_delta={row['gpu_mem_peak_delta_mib']} MiB "
                f"peak={row['gpu_mem_peak_mib']} MiB status={row['status']}",
                flush=True,
            )

    def run_phase(name: str, fn: Callable[[], Any], *, required: bool = True) -> Any:
        sampler = _GpuSampler(gpu_id, float(args.sample_interval))
        before = sampler.sample_once()
        print(f"[profile] phase={name} start gpu_mem={before}", flush=True)
        t0 = time.perf_counter()
        result = None
        status = "ok"
        error = None
        sampler.start()
        try:
            result = fn()
            _block_until_ready(result)
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            sampler.stop()
        after = sampler.sample_once()
        elapsed = time.perf_counter() - t0
        row = {
            "phase": name,
            "status": status,
            "elapsed_s": elapsed,
            "gpu_id": gpu_id,
            "gpu_mem_before_mib": None if before is None else before[0],
            "gpu_mem_after_mib": None if after is None else after[0],
            "gpu_mem_peak_mib": sampler.peak_mib,
            "gpu_mem_total_mib": sampler.total_mib,
            "gpu_mem_peak_delta_mib": (
                None if before is None or sampler.peak_mib is None else sampler.peak_mib - before[0]
            ),
            "error": error,
        }
        phase_rows.append(row)
        print("[profile_json] " + json.dumps(row, default=_json_safe, sort_keys=True), flush=True)
        write_report()
        if status != "ok" and required:
            raise RuntimeError(f"Phase {name!r} failed: {error}")
        return result

    points = run_phase(
        "reference_build_or_load",
        lambda: _H2._get_or_build_reference_curve(
            np.asarray([float(args.r)], dtype=np.float64),
            args=h2_args,
            logger=logger,
            label="train_profile",
        ),
    )
    point = points[0]
    molecule = point.molecule
    print(
        "[profile] reference "
        f"basis={args.basis} R={float(args.r):.6f} A "
        f"ngrids={int(molecule.grid.weights.shape[0])} "
        f"nao={int(molecule.ao.shape[1])} nmo={int(molecule.mo_coeff.shape[-1])}",
        flush=True,
    )

    molecule = run_phase("response_eri_cache", lambda: refresh_restricted_response_eri_slices(molecule))
    point = type(point)(
        r_angstrom=point.r_angstrom,
        atom=point.atom,
        molecule=molecule,
        fci_energy_h=point.fci_energy_h,
        fci_total_energies_h=point.fci_total_energies_h,
        fci_excitation_energies_h=point.fci_excitation_energies_h,
        fci_dm_ao=point.fci_dm_ao,
        fci_density_grid=point.fci_density_grid,
        fci_electron_count=point.fci_electron_count,
    )

    functional = _H2._make_s1_functional(h2_args)
    optimizer = optax.adam(1e-3)
    state = run_phase(
        "init_params",
        lambda: create_train_state_from_molecule(
            functional,
            jax.random.PRNGKey(0),
            molecule,
            optimizer,
        ),
    )
    training_data = run_phase(
        "make_training_datum",
        lambda: _H2.build_s1_training_data(
            [point],
            s1_weight=1.0,
            density_constraint_weight=0.0,
        ),
    )
    datum = training_data[0]
    gs_training = _training_config(args, h2_args)

    if str(args.training_mode) == "self_consistent":
        response_molecule = run_phase(
            "scf_forward_predict_molecule",
            lambda: predict_ground_state_molecule(
                state.params,
                functional,
                molecule,
                training_config=gs_training,
            ),
        )
    else:
        response_molecule = molecule

    if bool(args.profile_bind_components) or bool(args.stop_after_bind_components):
        component_state: dict[str, Any] = {}
        component_state["features"], component_state["total_gradient"] = run_phase(
            "bind_component_features",
            lambda: restricted_grid_features_with_gradients(response_molecule),
            required=False,
        )
        component_state["semilocal_channels"] = run_phase(
            "bind_component_semilocal_channels",
            lambda: functional.semilocal_energy_density_channels(component_state["features"]),
            required=False,
        )
        component_state["semilocal"] = run_phase(
            "bind_component_semilocal_sum",
            lambda: jnp.sum(component_state["semilocal_channels"], axis=-1),
            required=False,
        )
        (
            component_state["hf_projected"],
            component_state["hf_projected_a"],
            component_state["hf_projected_b"],
        ) = run_phase(
            "bind_component_hf_projected",
            lambda: functional.projected_hf_grid_contribution_components(
                response_molecule,
                features=component_state["features"],
            ),
            required=False,
        )

        def hfx_features_fn() -> Any:
            if functional.input_feature_mode == "canonical":
                return functional._canonical_hfx_feature_channels(
                    response_molecule,
                    component_state["features"],
                    hf_energy_density=component_state["hf_projected"],
                    hf_spin_energy_density=(
                        component_state["hf_projected_a"],
                        component_state["hf_projected_b"],
                    ),
                )
            return component_state["hf_projected_a"], component_state["hf_projected_b"]

        (
            component_state["hfx_feature_a"],
            component_state["hfx_feature_b"],
        ) = run_phase("bind_component_hfx_features", hfx_features_fn, required=False)
        component_state["pt2_projected"] = run_phase(
            "bind_component_pt2_projected",
            lambda: (
                functional.projected_pt2_grid_contribution(
                    response_molecule,
                    features=component_state["features"],
                )
                if functional.include_pt2_channel
                else None
            ),
            required=False,
        )
        component_state["coefficient_inputs"] = run_phase(
            "bind_component_coefficient_inputs",
            lambda: functional.coefficient_inputs(
                component_state["features"],
                component_state["semilocal"],
                component_state["hf_projected"],
                pt2_energy_density=component_state["pt2_projected"],
                molecule=response_molecule,
                hf_spin_energy_density=(
                    component_state["hf_projected_a"],
                    component_state["hf_projected_b"],
                ),
            ),
            required=False,
        )
        component_state["coefficients"] = run_phase(
            "bind_component_coefficients",
            lambda: functional.channel_coefficients_from_inputs(
                state.params,
                component_state["coefficient_inputs"],
            ),
            required=False,
        )
        component_state["hf_field"] = run_phase(
            "bind_component_hf_field",
            lambda: functional._local_hf_fraction_from_coefficients(component_state["coefficients"]),
            required=False,
        )
        component_state["strict_payload"] = run_phase(
            "bind_component_strict_payload",
            lambda: functional._strict_response_payload(
                component_state["features"],
                component_state["total_gradient"],
                component_state["hf_projected"],
                pt2_projected=component_state["pt2_projected"],
                hf_spin_energy_density=(
                    component_state["hfx_feature_a"],
                    component_state["hfx_feature_b"],
                ),
            ),
            required=False,
        )
        component_state["projected_tensor"] = run_phase(
            "bind_component_strict_response_tensor",
            lambda: functional._strict_total_response_tensor(
                state.params,
                component_state["features"],
                component_state["total_gradient"],
                component_state["hf_projected"],
                pt2_projected=component_state["pt2_projected"],
                hf_spin_energy_density=(
                    component_state["hfx_feature_a"],
                    component_state["hfx_feature_b"],
                ),
                response_hf_mode=functional.response_hf_mode,
                strict_payload=component_state["strict_payload"],
            ),
            required=False,
        )
        run_phase(
            "bind_component_alpha",
            lambda: jnp.clip(
                jnp.nan_to_num(
                    jnp.tensordot(
                        jnp.asarray(response_molecule.grid.weights),
                        jnp.maximum(component_state["features"].rho, functional.density_floor)
                        * component_state["hf_field"],
                        axes=(0, 0),
                    )
                    / jnp.maximum(
                        jnp.tensordot(
                            jnp.asarray(response_molecule.grid.weights),
                            jnp.maximum(component_state["features"].rho, functional.density_floor),
                            axes=(0, 0),
                        ),
                        functional.density_floor,
                    ),
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                ),
                0.0,
                1.0,
            ),
            required=False,
        )
        if bool(args.stop_after_bind_components):
            write_report()
            print_ranking()
            return 0
    bound = run_phase(
        "bind_response_functional",
        lambda: functional.bind_to_molecule_for_response(state.params, response_molecule),
    )
    if bool(args.materialize_grid_response_tensor) and callable(getattr(bound, "grid_response_tensor", None)):
        run_phase("grid_response_tensor", lambda: bound.grid_response_tensor(response_molecule), required=False)
    feature_kind = str(getattr(bound, "response_feature_kind", "MGGA"))
    run_phase(
        f"transition_response_features_{feature_kind}",
        lambda: restricted_transition_response_features(response_molecule, feature_kind=feature_kind),
        required=False,
    )

    if not bool(args.skip_pregrad_probes):
        def tda_matrix_fn(local_params: Any, local_molecule: Any) -> Any:
            return build_restricted_tda_matrix(local_molecule, functional, xc_params=local_params)

        if bool(args.jit_kernels):
            compiled_tda: dict[str, Any] = {}
            run_phase(
                "compile_tda_matrix",
                lambda: compiled_tda.setdefault(
                    "fn",
                    jax.jit(tda_matrix_fn).lower(state.params, response_molecule).compile(),
                ),
            )
            run_phase("execute_tda_matrix", lambda: compiled_tda["fn"](state.params, response_molecule))
        else:
            run_phase("execute_tda_matrix", lambda: tda_matrix_fn(state.params, response_molecule))

        def s1_forward_fn(local_params: Any, local_molecule: Any) -> Any:
            return predict_excitation_energies(
                local_params,
                functional,
                local_molecule,
                nstates=1,
                use_tda=bool(args.s1_use_tda),
            )

        if bool(args.jit_kernels):
            compiled_s1: dict[str, Any] = {}
            run_phase(
                "compile_s1_forward",
                lambda: compiled_s1.setdefault(
                    "fn",
                    jax.jit(s1_forward_fn).lower(state.params, response_molecule).compile(),
                ),
            )
            run_phase("execute_s1_forward", lambda: compiled_s1["fn"](state.params, response_molecule))
        else:
            run_phase("execute_s1_forward", lambda: s1_forward_fn(state.params, response_molecule))

    def loss_forward(local_params: Any, local_datum: Any) -> Any:
        return ground_state_mse_loss(
            local_params,
            functional,
            local_datum,
            training_config=gs_training,
        )

    loss_and_grad = make_ground_state_loss_and_grad(functional, training_config=gs_training)
    if bool(args.jit_kernels):
        compiled_loss: dict[str, Any] = {}
        compiled_grad: dict[str, Any] = {}
        if not bool(args.skip_pregrad_probes):
            run_phase(
                "compile_loss_forward",
                lambda: compiled_loss.setdefault(
                    "fn",
                    jax.jit(loss_forward).lower(state.params, datum).compile(),
                ),
            )
            run_phase("execute_loss_forward", lambda: compiled_loss["fn"](state.params, datum))
        run_phase(
            "compile_loss_grad",
            lambda: compiled_grad.setdefault(
                "fn",
                jax.jit(loss_and_grad).lower(state.params, datum).compile(),
            ),
        )
        run_phase("execute_loss_grad", lambda: compiled_grad["fn"](state.params, datum))
    else:
        run_phase("execute_loss_forward", lambda: loss_forward(state.params, datum))
        run_phase("execute_loss_grad", lambda: loss_and_grad(state.params, datum))

    if args.out_json is not None:
        write_report()
        print(f"[profile] wrote {Path(str(args.out_json))}", flush=True)

    print_ranking()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
