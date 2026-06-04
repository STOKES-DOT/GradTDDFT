from __future__ import annotations

import argparse
import shlex
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Print S1-only species extrapolation commands for molecule and "
            "closed-shell atom reference CSVs."
        )
    )
    p.add_argument("--molecule-reference-csv", required=True)
    p.add_argument("--atom-reference-csv", required=True)
    p.add_argument("--out-root", default="outputs/species_extrapolation_s1_only")
    p.add_argument("--basis", default="def2-svp")
    p.add_argument("--xc", default="b3lyp")
    p.add_argument("--conda-env", default="jax_scf")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--lr-decay-every", type=int, default=500)
    p.add_argument("--lr-decay-factor", type=float, default=0.5)
    p.add_argument("--training-mode", choices=("fixed_density", "self_consistent"), default="self_consistent")
    p.add_argument("--grids-level", type=int, default=2)
    p.add_argument("--stream-train", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-final-evaluation", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args(argv)


def _flag(name: str, value: object) -> list[str]:
    return [name, str(value)]


def _training_command(*, reference_csv: str, outdir: Path, args: argparse.Namespace) -> list[str]:
    cmd = [
        "conda",
        "run",
        "-n",
        args.conda_env,
        "python",
        "-u",
        "tools/closed_shell_s1_self_consistent_train.py",
        *_flag("--reference-csv", reference_csv),
        *_flag("--basis", args.basis),
        *_flag("--xc", args.xc),
        *_flag("--steps", args.steps),
        *_flag("--learning-rate", args.learning_rate),
        *_flag("--lr-decay-every", args.lr_decay_every),
        *_flag("--lr-decay-factor", args.lr_decay_factor),
        *_flag("--training-mode", args.training_mode),
        "--s1-use-tda",
        "--eval-use-tda",
        *_flag("--s1-weight", 1.0),
        *_flag("--energy-mse-weight", 0.0),
        *_flag("--energy-mae-weight", 0.0),
        *_flag("--density-constraint-weight", 0.0),
        *_flag("--grids-level", args.grids_level),
        *_flag("--outdir", outdir),
    ]
    if bool(args.stream_train):
        cmd.append("--stream-train")
    if bool(args.skip_final_evaluation):
        cmd.append("--skip-final-evaluation")
    return [str(part) for part in cmd]


def build_commands(args: argparse.Namespace) -> list[str]:
    out_root = Path(args.out_root)
    commands = [
        _training_command(
            reference_csv=str(args.molecule_reference_csv),
            outdir=out_root / "molecule_s1_only",
            args=args,
        ),
        _training_command(
            reference_csv=str(args.atom_reference_csv),
            outdir=out_root / "closed_shell_atom_s1_only",
            args=args,
        ),
    ]
    return [shlex.join(command) for command in commands]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    for command in build_commands(args):
        print(command)


if __name__ == "__main__":
    main()
