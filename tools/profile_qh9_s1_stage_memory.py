from __future__ import annotations

import argparse
import csv
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

import optax


_TRAIN_PATH = Path(__file__).with_name("closed_shell_s1_self_consistent_train.py")
_TRAIN_SPEC = importlib.util.spec_from_file_location("_closed_shell_s1_train_for_profile", _TRAIN_PATH)
if _TRAIN_SPEC is None or _TRAIN_SPEC.loader is None:
    raise RuntimeError(f"Failed to load training helpers from {_TRAIN_PATH}")
_TRAIN = importlib.util.module_from_spec(_TRAIN_SPEC)
sys.modules[_TRAIN_SPEC.name] = _TRAIN
_TRAIN_SPEC.loader.exec_module(_TRAIN)


class _Logger:
    def log(self, message: str) -> None:
        print(message, flush=True)


class _GpuSampler:
    def __init__(self, gpu_id: str | None, interval_s: float) -> None:
        self.gpu_id = gpu_id
        self.interval_s = float(interval_s)
        self.peak_mib: int | None = None
        self.total_mib: int | None = None
        self.samples: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _query(self) -> tuple[int, int] | None:
        try:
            if self.gpu_id and self.gpu_id.startswith("GPU-"):
                proc = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=uuid,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
                for line in proc.stdout.strip().splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) >= 3 and parts[0] == self.gpu_id:
                        return int(float(parts[1])), int(float(parts[2]))
                return None
            cmd = [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
            if self.gpu_id:
                cmd.insert(1, f"--id={self.gpu_id}")
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
            now = time.perf_counter()
            self.samples.append((now, used))
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


def _default_gpu_id() -> str:
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


def _phase(
    name: str,
    fn: Callable[[], Any],
    *,
    gpu_id: str,
    sample_interval: float,
) -> tuple[dict[str, Any], Any]:
    sampler = _GpuSampler(gpu_id, sample_interval)
    before = sampler.sample_once()
    t0 = time.perf_counter()
    sampler.start()
    status = "ok"
    error = None
    result = None
    try:
        result = fn()
        _block_until_ready(result)
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    elapsed = time.perf_counter() - t0
    sampler.stop()
    after = sampler.sample_once()
    row = {
        "phase": name,
        "status": status,
        "elapsed_s": elapsed,
        "gpu_before_mib": before[0] if before else None,
        "gpu_after_mib": after[0] if after else None,
        "gpu_peak_mib": sampler.peak_mib,
        "gpu_total_mib": sampler.total_mib,
        "error": error,
    }
    print(json.dumps(row, sort_keys=True), flush=True)
    return row, result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile QH9 closed-shell S1 memory by stage.")
    p.add_argument("--reference-csv", required=True)
    p.add_argument("--reference-cache", required=True)
    p.add_argument("--system", required=True)
    p.add_argument(
        "--init-system",
        default=None,
        help=(
            "Reference system used to initialize the neural functional state. "
            "Defaults to --system. Use the first training molecule to mirror "
            "the streaming training script while profiling a later datum."
        ),
    )
    p.add_argument("--basis", default="def2-svp")
    p.add_argument("--xc", default="b3lyp")
    p.add_argument(
        "--input-feature-mode",
        choices=("enhanced", "canonical", "dm21_original"),
        default=str(_TRAIN.DEFAULT_INPUT_FEATURE_MODE),
    )
    p.add_argument("--grids-level", type=int, default=2)
    p.add_argument("--include-hfx-channel", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--response-hf-mode",
        choices=("approx", "strict"),
        default=str(_TRAIN.DEFAULT_NEURAL_XC_RESPONSE_HF_MODE),
    )
    p.add_argument("--reference-jk-backend", choices=("full", "df"), default="full")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=(64, 64))
    p.add_argument("--scf-hfx-grid-block-size", type=int, default=1024)
    p.add_argument("--s1-weight", type=float, default=1.0)
    p.add_argument("--energy-mse-weight", type=float, default=0.0)
    p.add_argument("--energy-mae-weight", type=float, default=0.0)
    p.add_argument("--density-constraint-weight", type=float, default=0.0)
    p.add_argument("--training-mode", choices=("fixed_density", "self_consistent"), default="self_consistent")
    p.add_argument("--scf-gradient-mode", choices=("unrolled", "implicit_commutator"), default="implicit_commutator")
    p.add_argument("--gpu-id", default=None)
    p.add_argument("--sample-interval", type=float, default=0.2)
    p.add_argument("--outcsv", default=None)
    p.add_argument(
        "--stop-after",
        choices=("cache_load", "datum_build", "state_init", "loss_eval", "loss_grad"),
        default="loss_grad",
    )
    p.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _make_train_args(args: argparse.Namespace) -> argparse.Namespace:
    argv = [
        "--reference-csv",
        str(args.reference_csv),
        "--reference-cache",
        str(args.reference_cache),
        "--basis",
        str(args.basis),
        "--xc",
        str(args.xc),
        "--input-feature-mode",
        str(args.input_feature_mode),
        "--grids-level",
        str(args.grids_level),
        "--reference-jk-backend",
        str(args.reference_jk_backend),
        "--learning-rate",
        str(args.learning_rate),
        "--s1-weight",
        str(args.s1_weight),
        "--energy-mse-weight",
        str(args.energy_mse_weight),
        "--energy-mae-weight",
        str(args.energy_mae_weight),
        "--density-constraint-weight",
        str(args.density_constraint_weight),
        "--training-mode",
        str(args.training_mode),
        "--scf-gradient-mode",
        str(args.scf_gradient_mode),
        "--hidden-dims",
        *[str(dim) for dim in args.hidden_dims],
        "--scf-hfx-grid-block-size",
        str(args.scf_hfx_grid_block_size),
        "--include-hfx-channel" if bool(args.include_hfx_channel) else "--no-include-hfx-channel",
        "--response-hf-mode",
        str(args.response_hf_mode),
        "--stream-train",
        "--skip-initial-eval",
        "--skip-final-evaluation",
    ]
    train_args = _TRAIN.parse_args(argv)
    return train_args


def _append_rows(path: Path | None, rows: list[dict[str, Any]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "system",
        "init_system",
        "phase",
        "status",
        "elapsed_s",
        "gpu_before_mib",
        "gpu_after_mib",
        "gpu_peak_mib",
        "gpu_total_mib",
        "error",
    ]
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def main() -> None:
    args = parse_args()
    gpu_id = str(args.gpu_id or _default_gpu_id())
    train_args = _make_train_args(args)
    logger = _Logger()
    rows = _TRAIN._load_reference_rows(Path(args.reference_csv), basis=str(args.basis))
    selected = [row for row in rows if row.system == str(args.system)]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one row for {args.system!r}, found {len(selected)}.")
    ref_row = selected[0]
    init_system = str(args.init_system or args.system)
    selected_init = [row for row in rows if row.system == init_system]
    if len(selected_init) != 1:
        raise ValueError(f"Expected exactly one row for init system {init_system!r}, found {len(selected_init)}.")
    init_ref_row = selected_init[0]
    out_rows: list[dict[str, Any]] = []

    def record(row: dict[str, Any]) -> dict[str, Any]:
        row = dict(row)
        row["system"] = ref_row.system
        row["init_system"] = init_ref_row.system
        out_rows.append(row)
        _append_rows(Path(args.outcsv) if args.outcsv else None, [row])
        return row

    def stop_if_error(row: dict[str, Any]) -> None:
        if row.get("status") != "ok":
            raise SystemExit(1)

    def stop_if_requested(phase: str) -> None:
        if str(args.stop_after) == phase:
            raise SystemExit(0)

    phase_row, prepared = _phase(
        "cache_load",
        lambda: _TRAIN._prepare_references(
            list(dict.fromkeys([init_ref_row, ref_row])),
            args=train_args,
            logger=logger,
        ),
        gpu_id=gpu_id,
        sample_interval=float(args.sample_interval),
    )
    record(phase_row)
    stop_if_error(phase_row)
    stop_if_requested("cache_load")
    prepared_by_system = {item.row.system: item for item in prepared}
    prepared_ref = prepared_by_system[init_ref_row.system]
    target_ref = prepared_by_system[ref_row.system]

    phase_row, dataset = _phase(
        "datum_build",
        lambda: _TRAIN._build_dataset(
            [target_ref],
            s1_weight=float(train_args.s1_weight),
            density_constraint_weight=float(train_args.density_constraint_weight),
        ),
        gpu_id=gpu_id,
        sample_interval=float(args.sample_interval),
    )
    record(phase_row)
    stop_if_error(phase_row)
    stop_if_requested("datum_build")
    datum = dataset[0]

    functional = _TRAIN.neural_xc.Functional(
        architecture=str(train_args.network_architecture),
        semilocal_xc=tuple(str(name) for name in train_args.semilocal_xc),
        hidden_dims=tuple(int(value) for value in train_args.hidden_dims),
        input_feature_mode=_TRAIN._normalize_input_feature_mode(str(train_args.input_feature_mode)),
        include_pt2_channel=bool(train_args.include_pt2_channel),
        pt2_channel_mode=str(train_args.pt2_channel_mode),
        include_hfx_channel=bool(train_args.include_hfx_channel),
        response_hf_mode=str(train_args.response_hf_mode),
        name=f"profile_qh9_{str(train_args.training_mode)}",
    )
    training_config = _TRAIN._ground_state_training_config(
        mode=str(train_args.training_mode),
        energy_mse_weight=float(train_args.energy_mse_weight),
        energy_mae_weight=float(train_args.energy_mae_weight),
        s1_constraint_use_tda=bool(train_args.s1_use_tda),
        scf_max_cycle=int(train_args.train_scf_max_cycle),
        scf_damping=float(train_args.train_scf_damping),
        scf_conv_tol_density=float(train_args.train_scf_conv_tol_density),
        scf_vxc_clip=float(train_args.train_scf_vxc_clip),
        scf_iterate_selection=str(train_args.scf_iterate_selection),
        scf_require_convergence=bool(train_args.scf_require_convergence),
        scf_gradient_mode=_TRAIN._normalize_scf_gradient_mode(str(train_args.scf_gradient_mode)),
        scf_implicit_diff_solver=str(train_args.scf_implicit_diff_solver),
        scf_implicit_diff_tolerance=float(train_args.scf_implicit_diff_tolerance),
        scf_implicit_diff_regularization=float(train_args.scf_implicit_diff_regularization),
        scf_implicit_diff_restart=int(train_args.scf_implicit_diff_restart),
    )
    optimizer = optax.adam(float(train_args.learning_rate))

    phase_row, state = _phase(
        "state_init",
        lambda: _TRAIN.create_train_state_from_molecule(
            functional,
            jax.random.PRNGKey(int(train_args.seed)),
            prepared_ref.molecule,
            optimizer,
        ),
        gpu_id=gpu_id,
        sample_interval=float(args.sample_interval),
    )
    record(phase_row)
    stop_if_error(phase_row)
    stop_if_requested("state_init")

    eval_kernel = lambda params, item: _TRAIN.ground_state_mse_loss(  # noqa: E731
        params,
        functional,
        item,
        training_config=training_config,
    )
    loss_grad_kernel = _TRAIN.make_ground_state_loss_and_grad(
        functional,
        training_config=training_config,
    )
    if bool(args.jit):
        eval_kernel = jax.jit(eval_kernel)
        loss_grad_kernel = jax.jit(loss_grad_kernel)

    phase_row, _ = _phase(
        "loss_eval",
        lambda: eval_kernel(state.params, datum),
        gpu_id=gpu_id,
        sample_interval=float(args.sample_interval),
    )
    record(phase_row)
    stop_if_error(phase_row)
    stop_if_requested("loss_eval")

    phase_row, _ = _phase(
        "loss_grad",
        lambda: loss_grad_kernel(state.params, datum),
        gpu_id=gpu_id,
        sample_interval=float(args.sample_interval),
    )
    record(phase_row)
    stop_if_error(phase_row)


if __name__ == "__main__":
    main()
