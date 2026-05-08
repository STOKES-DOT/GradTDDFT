from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"[{_timestamp()}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _basis_tag(basis: str) -> str:
    return basis.lower().replace("*", "star").replace("+", "plus").replace(" ", "").replace("-", "")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run the H2 two-stage local workflow: "
            "stage1 self-consistent ground-state training, then "
            "stage2 fixed-density S1 training using the stage1 checkpoint."
        )
    )
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--xc", default="b3lyp")
    p.add_argument("--train-points", type=int, default=5)
    p.add_argument("--dense-points", type=int, default=20)
    p.add_argument("--stage1-steps", type=int, default=2000)
    p.add_argument("--stage2-steps", type=int, default=2000)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--lr-decay-every", type=int, default=0)
    p.add_argument("--lr-decay-factor", type=float, default=0.5)
    p.add_argument("--stage1-outdir", default=None)
    p.add_argument("--stage2-outdir", default=None)
    p.add_argument("--wrapper-log", default=None)
    return p.parse_args()


def _default_stage1_outdir(args: argparse.Namespace) -> str:
    tag = _basis_tag(str(args.basis))
    return f"outputs/h2_stage1_ground_{tag}_local_ep{int(args.stage1_steps)}_log10_driver"


def _default_stage2_outdir(args: argparse.Namespace) -> str:
    tag = _basis_tag(str(args.basis))
    return f"outputs/h2_stage2_s1_fixed_{tag}_local_ep{int(args.stage2_steps)}_log10_driver"


def _run_and_tee(command: Sequence[str], *, env: dict[str, str], logger: RunLogger, stage_name: str) -> None:
    logger.log(f"{stage_name}: exec {' '.join(command)}")
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=Path.cwd(),
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            text = line.rstrip("\n")
            print(text, flush=True)
            with logger.path.open("a", encoding="utf-8") as handle:
                handle.write(f"[{_timestamp()}] [{stage_name}] {text}\n")
    finally:
        process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, list(command))
    logger.log(f"{stage_name}: completed successfully")


def main() -> None:
    args = parse_args()

    stage1_outdir = Path(str(args.stage1_outdir or _default_stage1_outdir(args)))
    stage2_outdir = Path(str(args.stage2_outdir or _default_stage2_outdir(args)))
    wrapper_log = Path(
        str(
            args.wrapper_log
            or stage1_outdir.parent / f"{stage1_outdir.name}_wrapper.log"
        )
    )

    stage1_outdir.mkdir(parents=True, exist_ok=True)
    stage2_outdir.mkdir(parents=True, exist_ok=True)
    stage1_mpl = stage1_outdir.parent / f".mplconfig_{stage1_outdir.name}"
    stage2_mpl = stage2_outdir.parent / f".mplconfig_{stage2_outdir.name}"
    stage1_mpl.mkdir(parents=True, exist_ok=True)
    stage2_mpl.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(wrapper_log)
    logger.log(f"cwd={Path.cwd()}")
    logger.log(f"stage1_outdir={stage1_outdir}")
    logger.log(f"stage2_outdir={stage2_outdir}")

    common_env = os.environ.copy()
    common_env["PYTHONUNBUFFERED"] = "1"
    common_env["JAX_PLATFORMS"] = "cpu"
    common_env["JAX_PLATFORM_NAME"] = "cpu"
    common_env["MPLBACKEND"] = "Agg"

    stage1_cmd = [
        sys.executable,
        "-u",
        "tools/h2_self_consistent_ground_train5_dense100_vs_fci.py",
        "--basis",
        str(args.basis),
        "--xc",
        str(args.xc),
        "--train-points",
        str(int(args.train_points)),
        "--dense-points",
        str(int(args.dense_points)),
        "--steps",
        str(int(args.stage1_steps)),
        "--learning-rate",
        str(float(args.learning_rate)),
        "--lr-decay-every",
        str(int(args.lr_decay_every)),
        "--lr-decay-factor",
        str(float(args.lr_decay_factor)),
        "--training-mode",
        "self_consistent",
        "--include-pt2-channel",
        "--jit-eval",
        "--jit-train",
        "--scf-gradient-mode",
        "implicit_commutator",
        "--outdir",
        str(stage1_outdir),
    ]
    stage1_env = dict(common_env)
    stage1_env["MPLCONFIGDIR"] = str(stage1_mpl)

    stage2_cmd = [
        sys.executable,
        "-u",
        "tools/h2_s1_tda_train5_dense100_vs_fci.py",
        "--basis",
        str(args.basis),
        "--xc",
        str(args.xc),
        "--train-points",
        str(int(args.train_points)),
        "--dense-points",
        str(int(args.dense_points)),
        "--steps",
        str(int(args.stage2_steps)),
        "--learning-rate",
        str(float(args.learning_rate)),
        "--lr-decay-every",
        str(int(args.lr_decay_every)),
        "--lr-decay-factor",
        str(float(args.lr_decay_factor)),
        "--training-mode",
        "fixed_density",
        "--include-pt2-channel",
        "--jit-eval",
        "--no-jit-train",
        "--fixed-density-reference-checkpoint",
        str(stage1_outdir / "neural_xc_params.msgpack"),
        "--init-checkpoint",
        str(stage1_outdir / "neural_xc_params.msgpack"),
        "--outdir",
        str(stage2_outdir),
    ]
    stage2_env = dict(common_env)
    stage2_env["MPLCONFIGDIR"] = str(stage2_mpl)

    _run_and_tee(stage1_cmd, env=stage1_env, logger=logger, stage_name="stage1")
    _run_and_tee(stage2_cmd, env=stage2_env, logger=logger, stage_name="stage2")
    logger.log("two-stage workflow completed")


if __name__ == "__main__":
    main()
