#!/usr/bin/env python3
"""Render a QH9 reference CSV as a grid of molecular structures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from plot_qh9_val_bars_with_structures import (
    _add_structure_panel,
    _read_csv,
    _render_structure,
    _safe_text,
    _system_label,
)


SPLIT_COLORS = {
    "train": "#444444",
    "validation": "#D97706",
    "test": "#1F77B4",
}


def _note_value(notes: str, key: str) -> str:
    match = re.search(rf"(?:^|; ){re.escape(key)}=([^;]+)", notes)
    return match.group(1) if match else ""


def _write_metadata(rows: list[dict[str, str]], rendered: dict[str, str | None], out_path: Path) -> None:
    fieldnames = ["system", "split", "formula", "natoms", "structure_png", "xyz_path"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            system = row["system"]
            structure_png = rendered.get(system) or ""
            writer.writerow(
                {
                    "system": system,
                    "split": row["split"],
                    "formula": _note_value(row.get("notes", ""), "formula"),
                    "natoms": _note_value(row.get("notes", ""), "natoms"),
                    "structure_png": structure_png,
                    "xyz_path": str(Path(structure_png).with_suffix(".xyz")) if structure_png else "",
                }
            )


def _format_title(row: dict[str, str]) -> str:
    formula = _note_value(row.get("notes", ""), "formula")
    title = _system_label(row["system"])
    if formula:
        title = f"{formula}\n{row['system']}"
    return title


def _plot_grid(rows: list[dict[str, str]], outdir: Path, xyzrender_src: Path, columns: int) -> dict[str, Any]:
    cache_dir = outdir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib.pyplot as plt

    structure_dir = outdir / "structures"
    rendered: dict[str, str | None] = {}
    for row in rows:
        image_path = _render_structure(
            system=row["system"],
            atom_block=row["atom"],
            outdir=structure_dir,
            xyzrender_src=xyzrender_src,
        )
        rendered[row["system"]] = str(image_path) if image_path is not None else None

    ncols = min(columns, max(1, len(rows)))
    nrows = math.ceil(len(rows) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.65 * ncols, 2.55 * nrows + 0.6),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle("QH9 10-molecule set for the low-memory S1 run", fontsize=18)

    for idx, row in enumerate(rows):
        ax = axes[idx // ncols][idx % ncols]
        system = row["system"]
        _add_structure_panel(
            ax,
            image_path=Path(rendered[system]) if rendered[system] else None,
            atom_block=row["atom"],
            title=_format_title(row),
        )
        ax.title.set_fontsize(14)
        split = row["split"]
        color = SPLIT_COLORS.get(split, "#777777")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(1.5)
        ax.text(
            0.02,
            0.03,
            split,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12,
            color=color,
            fontweight="bold",
        )

    for idx in range(len(rows), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    png_path = outdir / "qh9_10_molecule_structures_grid.png"
    pdf_path = outdir / "qh9_10_molecule_structures_grid.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    metadata_path = outdir / "qh9_10_molecule_structures.csv"
    _write_metadata(rows, rendered, metadata_path)
    manifest = {
        "reference_csv": "",
        "structures_csv": str(metadata_path),
        "png": str(png_path),
        "pdf": str(pdf_path),
        "structures": rendered,
        "splits": sorted({row["split"] for row in rows}),
        "note": "Rows are ordered as in the reference CSV; this run contains 8 train and 2 validation molecules.",
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument(
        "--xyzrender-src",
        type=Path,
        default=Path("/Volumes/TF/QH9_db/xyzrender/src"),
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(args.reference_csv)
    if not rows:
        raise ValueError(f"empty reference CSV: {args.reference_csv}")

    manifest = _plot_grid(rows, args.outdir, args.xyzrender_src, args.columns)
    manifest["reference_csv"] = str(args.reference_csv)
    manifest_path = args.outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
