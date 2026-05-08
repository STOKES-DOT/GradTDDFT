#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sci(value: Any) -> str:
    number = _float_or_none(value)
    if number is None:
        return "nan"
    return f"{number:.2e}"


def _energy(value: Any) -> str:
    number = _float_or_none(value)
    if number is None:
        return "nan"
    return f"{number:.6f}"


def _log_width(value: Any, *, width: float, min_exp: float = -18.0, max_exp: float = -12.0) -> float:
    number = abs(_float_or_none(value) or 0.0)
    if number <= 0.0:
        exponent = min_exp
    else:
        exponent = max(min(math.log10(number), max_exp), min_exp)
    return width * (exponent - min_exp) / (max_exp - min_exp)


def _rect(x: float, y: float, w: float, h: float, fill: str, *, opacity: float = 1.0, rx: float = 3.0) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0.0):.1f}" height="{h:.1f}" '
        f'rx="{rx:.1f}" fill="{fill}" opacity="{opacity:.3f}"/>'
    )


def _text(
    x: float,
    y: float,
    value: Any,
    *,
    size: int = 12,
    weight: int = 400,
    anchor: str = "start",
    fill: str = "#172026",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="{fill}">{escape(str(value))}</text>'
    )


def _line(x1: float, y1: float, x2: float, y2: float, *, stroke: str = "#d7dee4", width: float = 1.0) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width:.1f}"/>'
    )


def _load_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in data.get("results", [])
        if row.get("status") == "ok"
    ]
    if not rows:
        raise ValueError(f"No successful benchmark rows in {path}.")
    return data, rows


def render_svg(data: dict[str, Any], rows: list[dict[str, Any]], *, title: str) -> str:
    width = 1280
    row_h = 54
    top = 152
    bottom = 78
    height = top + row_h * len(rows) + bottom

    label_x = 42
    energy_x = 270
    energy_w = 290
    error_x = 625
    error_w = 195
    exc_x = 895
    exc_w = 205
    value_x = 1168

    max_abs_energy = max(
        abs(_float_or_none(row.get("e_xc_jax")) or 0.0)
        for row in rows
    )
    max_abs_energy = max(max_abs_energy, 1e-30)

    grid_level = data.get("grid_level")
    points = data.get("grid_points")
    point_selection = data.get("point_selection", "head")
    versions = data.get("versions", {})
    ao_diag = data.get("ao_density_check", {})
    backend = data.get("jax_xc_backend")
    jax_xc_version = data.get("jax_xc_version")

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fb"/>',
        _text(42, 48, title, size=26, weight=700),
        _text(
            42,
            76,
            f"water / {data.get('basis')} / grid_level={grid_level} / points={points} / selection={point_selection}",
            size=14,
            fill="#4c5b66",
        ),
        _text(
            42,
            100,
            f"jax_xc={jax_xc_version} ({backend}), jax={versions.get('jax')}, pyscf={versions.get('pyscf')}",
            size=13,
            fill="#4c5b66",
        ),
        _text(
            42,
            124,
            "AO density check: max |rho_jax - rho_pyscf| = "
            f"{_sci(ao_diag.get('rho_max_abs_diff'))}, rms = {_sci(ao_diag.get('rho_rms_abs_diff'))}",
            size=13,
            fill="#4c5b66",
        ),
        _text(label_x, top - 22, "functional", size=12, weight=700, fill="#33414a"),
        _text(energy_x, top - 22, "E_xc overlay (JAX vs PySCF)", size=12, weight=700, fill="#33414a"),
        _text(error_x, top - 22, "|Delta E_xc|, log10", size=12, weight=700, fill="#33414a"),
        _text(exc_x, top - 22, "max |Delta exc|, log10", size=12, weight=700, fill="#33414a"),
        _text(value_x, top - 22, "E_jax / E_pyscf", size=12, weight=700, anchor="end", fill="#33414a"),
    ]

    for tick_exp in (-18, -16, -14, -12):
        tx = error_x + error_w * ((tick_exp + 18) / 6)
        parts.append(_line(tx, top - 10, tx, height - bottom + 8, stroke="#e0e6eb"))
        parts.append(_text(tx, height - bottom + 30, f"1e{tick_exp}", size=10, anchor="middle", fill="#65727d"))
        tx2 = exc_x + exc_w * ((tick_exp + 18) / 6)
        parts.append(_line(tx2, top - 10, tx2, height - bottom + 8, stroke="#e0e6eb"))
        parts.append(_text(tx2, height - bottom + 30, f"1e{tick_exp}", size=10, anchor="middle", fill="#65727d"))

    parts.extend(
        [
            _rect(42, height - 42, 16, 8, "#197a8a", rx=2),
            _text(64, height - 34, "JAX_XC E_xc", size=11, fill="#4c5b66"),
            _rect(170, height - 42, 16, 8, "#d47a28", rx=2),
            _text(192, height - 34, "PySCF/libxc E_xc", size=11, fill="#4c5b66"),
            _rect(326, height - 42, 16, 8, "#6b5b95", rx=2),
            _text(348, height - 34, "absolute differences", size=11, fill="#4c5b66"),
        ]
    )

    for idx, row in enumerate(rows):
        y = top + idx * row_h
        cy = y + 24
        if idx % 2 == 0:
            parts.append(_rect(28, y - 14, width - 56, row_h - 4, "#ffffff", opacity=0.82, rx=5))
        parts.append(_line(38, y + row_h - 12, width - 38, y + row_h - 12, stroke="#edf1f4"))

        name = row.get("label") or row.get("functional", "")
        raw_name = row.get("functional", "")
        xctype = row.get("xctype", "")
        if row.get("is_hybrid") is True:
            hybrid_coeff = row.get("hybrid_coeff")
            if hybrid_coeff is None:
                xctype = f"HYB-{xctype}"
            else:
                xctype = f"HYB-{xctype}, a={float(hybrid_coeff):.3g}"
        e_jax = _float_or_none(row.get("e_xc_jax")) or 0.0
        e_ref = _float_or_none(row.get("e_xc_pyscf")) or 0.0
        e_jax_w = energy_w * abs(e_jax) / max_abs_energy
        e_ref_w = energy_w * abs(e_ref) / max_abs_energy

        label_meta = xctype if name == raw_name else f"{raw_name} / {xctype}"
        parts.append(_text(label_x, cy - 3, name, size=13, weight=700))
        parts.append(_text(label_x, cy + 16, label_meta, size=9, fill="#7a8791"))
        parts.append(_rect(energy_x, cy - 17, e_ref_w, 11, "#d47a28", opacity=0.78, rx=2))
        parts.append(_rect(energy_x, cy - 1, e_jax_w, 11, "#197a8a", opacity=0.82, rx=2))

        d_e_w = _log_width(row.get("e_abs_diff"), width=error_w)
        d_exc_w = _log_width(row.get("exc_max_abs"), width=exc_w)
        parts.append(_rect(error_x, cy - 9, max(d_e_w, 2.0), 12, "#6b5b95", opacity=0.85, rx=2))
        parts.append(_text(error_x + error_w + 12, cy + 1, _sci(row.get("e_abs_diff")), size=11, fill="#4c5b66"))
        parts.append(_rect(exc_x, cy - 9, max(d_exc_w, 2.0), 12, "#6b5b95", opacity=0.85, rx=2))
        parts.append(_text(exc_x + exc_w + 12, cy + 1, _sci(row.get("exc_max_abs")), size=11, fill="#4c5b66"))
        parts.append(_text(value_x, cy + 1, f"{_energy(e_jax)} / {_energy(e_ref)}", size=11, anchor="end", fill="#33414a"))

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a SVG visual summary for the water jax_xc benchmark.")
    parser.add_argument("json_in", type=Path)
    parser.add_argument("--svg-out", type=Path, required=True)
    parser.add_argument("--title", default="JAX_XC vs PySCF/libxc: water benchmark")
    args = parser.parse_args()

    data, rows = _load_rows(args.json_in)
    svg = render_svg(data, rows, title=args.title)
    args.svg_out.parent.mkdir(parents=True, exist_ok=True)
    args.svg_out.write_text(svg, encoding="utf-8")
    print(f"wrote_svg={args.svg_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
