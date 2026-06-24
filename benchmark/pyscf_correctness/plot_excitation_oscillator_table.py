from __future__ import annotations

import csv
import math
import os
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", str(Path("benchmark") / ".mplconfig"))

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "runs" / "remote_b3lyp_def2svp_singlet_20260527"
TABLE_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MOLECULE_ORDER = ["water", "co", "n2", "ethylene", "formaldehyde", "benzene"]
MOLECULE_LABELS_TEX = {
    "water": r"H$_2$O",
    "co": "CO",
    "n2": r"N$_2$",
    "ethylene": r"C$_2$H$_4$",
    "formaldehyde": r"CH$_2$O",
    "benzene": r"C$_6$H$_6$",
}
MOLECULE_LABELS_PLAIN = {
    "water": "H2O",
    "co": "CO",
    "n2": "N2",
    "ethylene": "C2H4",
    "formaldehyde": "CH2O",
    "benzene": "C6H6",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def as_float(value: str) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(value)
    return x


def fmt_energy(value: float) -> str:
    text = f"{value:.2g}"
    if "." not in text and "e" not in text and value < 10:
        text += ".0"
    return text


def fmt_osc_tex(value: float) -> str:
    if value == 0.0:
        return r"\(0\)"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / (10.0**exponent)
    return rf"\({mantissa:.1f}\times 10^{{{exponent}}}\)"


def fmt_osc_plain(value: float) -> str:
    return f"{value:.1e}"


def summarize(viz: list[dict[str, str]]) -> list[dict[str, float | str]]:
    rows = [
        row
        for row in viz
        if row["support_status"] == "graddft_supported"
        and row["spin_channel"] == "singlet"
        and row["state_index"] == "1"
    ]
    by_key = {(row["molecule"], row["response_solver"]): row for row in rows}
    table_rows: list[dict[str, float | str]] = []
    for molecule in MOLECULE_ORDER:
        for solver in ("tda", "tddft"):
            row = by_key[(molecule, solver)]
            table_rows.append(
                {
                    "molecule": molecule,
                    "molecule_tex": MOLECULE_LABELS_TEX[molecule],
                    "molecule_plain": MOLECULE_LABELS_PLAIN[molecule],
                    "solver": "TDA" if solver == "tda" else "TDDFT",
                    "pyscf_omega_ev": as_float(row["pyscf_excitation_ev"]),
                    "pyscf_f": as_float(row["pyscf_oscillator_strength"]),
                    "graddft_omega_ev": as_float(row["graddft_excitation_ev"]),
                    "graddft_f": as_float(row["graddft_oscillator_strength"]),
                }
            )
    return table_rows


def write_summary_csv(rows: list[dict[str, float | str]]) -> Path:
    path = TABLE_DIR / "b3lyp_s1_excitation_oscillator_summary.csv"
    fields = [
        "molecule",
        "solver",
        "pyscf_omega_ev",
        "pyscf_f",
        "graddft_omega_ev",
        "graddft_f",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
    return path


def result_cell(omega: float, oscillator_strength: float, *, tex: bool) -> str:
    if tex:
        return f"{fmt_energy(omega)} / {fmt_osc_tex(oscillator_strength)}"
    return f"{fmt_energy(omega)} / {fmt_osc_plain(oscillator_strength)}"


def write_latex_table(rows: list[dict[str, float | str]]) -> Path:
    path = TABLE_DIR / "b3lyp_s1_excitation_oscillator_table.tex"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{B3LYP/def2-SVP S$_1$ excitation energies and oscillator strengths from matched PySCF and GradTDDFT calculations. Each entry reports \(\Omega_1\) in eV followed by \(f_1\).}",
        r"\label{tab:b3lyp-s1-excitation-oscillator}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Molecule & Solver & PySCF & GradTDDFT \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    str(row["molecule_tex"]),
                    str(row["solver"]),
                    result_cell(float(row["pyscf_omega_ev"]), float(row["pyscf_f"]), tex=True),
                    result_cell(float(row["graddft_omega_ev"]), float(row["graddft_f"]), tex=True),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines))
    return path


def draw_table(rows: list[dict[str, float | str]]) -> tuple[Path, Path]:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(7.8, 4.6), constrained_layout=True)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    columns = [("Molecule", 0.05), ("Solver", 0.23), ("PySCF", 0.50), ("GradTDDFT", 0.78)]
    ax.hlines([0.90, 0.80, 0.08], 0.04, 0.96, colors="#20242A", linewidths=[1.4, 0.9, 1.4])
    ax.text(
        0.04,
        0.97,
        "B3LYP/def2-SVP S1 excitation energy / oscillator strength",
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    for label, x in columns:
        ax.text(x, 0.84, label, ha="center" if x > 0.1 else "left", va="center", fontsize=9, fontweight="bold")

    y0 = 0.75
    dy = 0.055
    for index, row in enumerate(rows):
        y = y0 - index * dy
        values = [
            str(row["molecule_plain"]),
            str(row["solver"]),
            result_cell(float(row["pyscf_omega_ev"]), float(row["pyscf_f"]), tex=False),
            result_cell(float(row["graddft_omega_ev"]), float(row["graddft_f"]), tex=False),
        ]
        for (_, x), value in zip(columns, values):
            ax.text(x, y, value, ha="center" if x > 0.1 else "left", va="center", fontsize=8.5)

    ax.text(
        0.04,
        0.025,
        "Each cell reports Omega1 (eV) / f1. Values are rounded for manuscript display.",
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#616A76",
    )
    png = FIG_DIR / "b3lyp_s1_excitation_oscillator_table_draft.png"
    pdf = FIG_DIR / "b3lyp_s1_excitation_oscillator_table_draft.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png, pdf


def main() -> None:
    rows = summarize(read_csv(RUN_DIR / "visualization_data.csv"))
    csv_path = write_summary_csv(rows)
    tex_path = write_latex_table(rows)
    png, pdf = draw_table(rows)
    print(csv_path)
    print(tex_path)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
