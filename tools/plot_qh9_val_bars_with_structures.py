#!/usr/bin/env python3
"""Plot QH9 validation S1/S2 bars with molecular structure insets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


METHODS = (
    ("EOM-CCSD", "target", "#333333"),
    ("Neural XC", "neural", "#1F77B4"),
    ("B3LYP", "b3lyp", "#D97706"),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_text(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def _prepare_cairo_runtime() -> None:
    homebrew_lib = "/opt/homebrew/lib"
    current = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    parts = [p for p in current.split(":") if p]
    if homebrew_lib not in parts:
        parts.insert(0, homebrew_lib)
    if "/usr/lib" not in parts:
        parts.append("/usr/lib")
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(parts)


def _system_label(system: str) -> str:
    parts = system.split("_", 2)
    if len(parts) == 3:
        return f"{parts[2]}\n{parts[0]}_{parts[1]}"
    return system


def _write_xyz(system: str, atom_block: str, outdir: Path) -> Path:
    atom_lines = [line.strip() for line in atom_block.splitlines() if line.strip()]
    xyz_path = outdir / f"{_safe_text(system)}.xyz"
    with xyz_path.open("w", encoding="utf-8") as f:
        f.write(f"{len(atom_lines)}\n")
        f.write(f"{system}\n")
        for line in atom_lines:
            f.write(f"{line}\n")
    return xyz_path


def _render_structure(
    *,
    system: str,
    atom_block: str,
    outdir: Path,
    xyzrender_src: Path,
) -> Path | None:
    outdir.mkdir(parents=True, exist_ok=True)
    xyz_path = _write_xyz(system, atom_block, outdir)
    png_path = outdir / f"{_safe_text(system)}.png"

    if not xyzrender_src.exists():
        return None

    try:
        _prepare_cairo_runtime()
        src = str(xyzrender_src)
        if src not in sys.path:
            sys.path.insert(0, src)
        from xyzrender import load, render

        mol = load(str(xyz_path))
        render(
            mol,
            output=str(png_path),
            config="flat",
            hy=True,
            canvas_size=640,
            transparent=True,
        )
    except Exception as exc:  # pragma: no cover - depends on local renderer
        print(f"[plot] xyzrender failed for {system}: {exc}", file=sys.stderr)
        return None

    return png_path if png_path.exists() else None


def _parse_atoms(atom_block: str) -> list[tuple[str, float, float, float]]:
    atoms: list[tuple[str, float, float, float]] = []
    for line in atom_block.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        atoms.append((fields[0], float(fields[1]), float(fields[2]), float(fields[3])))
    return atoms


def _plot_structure_fallback(ax: Any, atom_block: str) -> None:
    atoms = _parse_atoms(atom_block)
    if not atoms:
        ax.text(0.5, 0.5, "structure unavailable", ha="center", va="center")
        return
    colors = {
        "H": "#F4F4F4",
        "C": "#333333",
        "N": "#2563EB",
        "O": "#DC2626",
        "F": "#16A34A",
    }
    sizes = {"H": 45, "C": 95, "N": 100, "O": 105, "F": 105}
    xs = [a[1] for a in atoms]
    ys = [a[2] for a in atoms]
    for sym, x, y, _z in atoms:
        ax.scatter(
            x,
            y,
            s=sizes.get(sym, 90),
            c=colors.get(sym, "#888888"),
            edgecolors="#111111",
            linewidths=0.6,
            zorder=3,
        )
        ax.text(x, y, sym, ha="center", va="center", fontsize=8, zorder=4)
    pad = 0.5
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")


def _add_structure_panel(ax: Any, *, image_path: Path | None, atom_block: str, title: str) -> None:
    ax.set_title(title, fontsize=11, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if image_path is not None:
        import matplotlib.image as mpimg

        img = mpimg.imread(image_path)
        ax.imshow(img)
    else:
        _plot_structure_fallback(ax, atom_block)


def _bar_values(row: dict[str, str], state: str) -> list[float]:
    return [
        float(row[f"target_{state}_ev"]),
        float(row[f"neural_{state}_ev"]),
        float(row[f"b3lyp_{state}_ev"]),
    ]


def _write_plot_data(rows: list[dict[str, str]], out_path: Path) -> None:
    fieldnames = [
        "system",
        "state",
        "method",
        "excitation_ev",
        "abs_error_ev",
        "improvement_vs_b3lyp_ev",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for state in ("s1", "s2"):
                for method_label, key, _color in METHODS:
                    abs_err = ""
                    improvement = ""
                    if key == "neural":
                        abs_err = row[f"neural_{state}_abs_err_ev"]
                        improvement = row[f"improvement_{state}_ev"]
                    elif key == "b3lyp":
                        abs_err = row[f"b3lyp_{state}_abs_err_ev"]
                        improvement = "0.0"
                    elif key == "target":
                        abs_err = "0.0"
                    writer.writerow(
                        {
                            "system": row["system"],
                            "state": state.upper(),
                            "method": method_label,
                            "excitation_ev": row[f"{key}_{state}_ev"],
                            "abs_error_ev": abs_err,
                            "improvement_vs_b3lyp_ev": improvement,
                        }
                    )


def _plot(rows: list[dict[str, str]], ref_by_system: dict[str, dict[str, str]], outdir: Path, xyzrender_src: Path) -> dict[str, Any]:
    cache_dir = outdir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib.pyplot as plt
    import numpy as np

    structure_dir = outdir / "structures"
    rendered: dict[str, str | None] = {}
    for row in rows:
        system = row["system"]
        atom_block = ref_by_system[system]["atom"]
        image_path = _render_structure(
            system=system,
            atom_block=atom_block,
            outdir=structure_dir,
            xyzrender_src=xyzrender_src,
        )
        rendered[system] = str(image_path) if image_path is not None else None

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )
    fig = plt.figure(figsize=(8.2, 6.8), constrained_layout=True)
    grid = fig.add_gridspec(3, len(rows), height_ratios=[1.0, 1.2, 1.2])

    for idx, row in enumerate(rows):
        system = row["system"]
        ax = fig.add_subplot(grid[0, idx])
        _add_structure_panel(
            ax,
            image_path=Path(rendered[system]) if rendered[system] else None,
            atom_block=ref_by_system[system]["atom"],
            title=_system_label(system),
        )

    centers = np.arange(len(rows), dtype=float)
    width = 0.22
    labels = [_system_label(row["system"]) for row in rows]

    for panel_idx, state in enumerate(("s1", "s2"), start=1):
        ax = fig.add_subplot(grid[panel_idx, :])
        for method_idx, (method_label, _key, color) in enumerate(METHODS):
            values = [_bar_values(row, state)[method_idx] for row in rows]
            x = centers + (method_idx - 1) * width
            bars = ax.bar(x, values, width=width, label=method_label, color=color)
            ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=2, fontsize=8)
        ax.set_ylabel(f"{state.upper()} energy (eV)")
        ax.set_xticks(centers)
        ax.set_xticklabels(labels)
        ymax = max(max(_bar_values(row, state)) for row in rows)
        ax.set_ylim(0, ymax * 1.18)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
        if panel_idx == 1:
            ax.legend(loc="upper right", frameon=False, ncols=3)

    fig.suptitle("QH9 validation excitation energies: EOM-CCSD vs trained Neural XC vs B3LYP", fontsize=12)
    png_path = outdir / "qh9_val_s1s2_bars_with_structures.png"
    pdf_path = outdir / "qh9_val_s1s2_bars_with_structures.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "png": str(png_path),
        "pdf": str(pdf_path),
        "structures": rendered,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-csv", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--xyzrender-src",
        type=Path,
        default=Path("/Volumes/TF/QH9_db/xyzrender/src"),
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(args.comparison_csv)
    refs = _read_csv(args.reference_csv)
    ref_by_system = {row["system"]: row for row in refs}
    missing = [row["system"] for row in rows if row["system"] not in ref_by_system]
    if missing:
        raise ValueError(f"missing reference rows for: {', '.join(missing)}")

    plot_data_path = args.outdir / "qh9_val_s1s2_bar_plot_data.csv"
    _write_plot_data(rows, plot_data_path)
    outputs = _plot(rows, ref_by_system, args.outdir, args.xyzrender_src)

    manifest = {
        "comparison_csv": str(args.comparison_csv),
        "reference_csv": str(args.reference_csv),
        "plot_data_csv": str(plot_data_path),
        "outputs": outputs,
        "methods": [label for label, _key, _color in METHODS],
        "states": ["S1", "S2"],
        "unit": "eV",
        "note": "Neural XC values are self-consistent predictions from the trained checkpoint; S1 is the optimized target, S2 is an extra nstates=2 inference result.",
    }
    manifest_path = args.outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **outputs}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
