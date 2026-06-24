#!/usr/bin/env python3
"""Collect manuscript-facing plotting data under benchmark/plot_data.

The script is intentionally conservative: it copies CSV/JSON source
artifacts into a normalized directory tree and writes a manifest. It
does not remove, move, or rewrite the original run directories.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "benchmark" / "plot_data"
RAW_REMOTE_OUTPUTS = "benchmark/loss_curves/raw_remote/home/yjiao/TD-GradDFT/outputs"


@dataclass(frozen=True)
class PlotDataItem:
    category: str
    system: str
    state: str
    task: str
    variant: str
    role: str
    source: str
    dest: str


ITEMS: tuple[PlotDataItem, ...] = (
    PlotDataItem(
        "validation",
        "multi",
        "tddft",
        "pyscf_correctness",
        "summary",
        "summary_table",
        "benchmark/pyscf_correctness/summary.csv",
        "validation/pyscf_correctness/summary.csv",
    ),
    PlotDataItem(
        "validation",
        "multi",
        "tddft",
        "pyscf_correctness",
        "summary",
        "visualization_data",
        "benchmark/pyscf_correctness/visualization_data.csv",
        "validation/pyscf_correctness/visualization_data.csv",
    ),
    PlotDataItem(
        "validation",
        "multi",
        "tddft",
        "pyscf_correctness",
        "summary",
        "task_manifest",
        "benchmark/pyscf_correctness/task_manifest.csv",
        "validation/pyscf_correctness/task_manifest.csv",
    ),
    PlotDataItem(
        "validation",
        "multi",
        "tddft",
        "pyscf_correctness",
        "summary",
        "runs_index",
        "benchmark/pyscf_correctness/runs_index.csv",
        "validation/pyscf_correctness/runs_index.csv",
    ),
    PlotDataItem(
        "validation",
        "multi",
        "tddft",
        "pyscf_correctness",
        "oscillator_strength",
        "summary_table",
        "benchmark/pyscf_correctness/tables/excitation_oscillator_summary.csv",
        "validation/pyscf_correctness/excitation_oscillator_summary.csv",
    ),
    PlotDataItem(
        "validation",
        "benzene",
        "s1",
        "ris_vs_noris_pyscf_tda",
        "b3lyp_def2svp",
        "comparison_table",
        "benchmark/benzene_ris_vs_noris_pyscf_tda/benzene_b3lyp_def2svp_grid0_n3_gpu0_pyscf_lreigh_operator_20260622_145948/benzene_b3lyp_tda_ris_vs_noris_pyscf.csv",
        "validation/pyscf_correctness/benzene_ris_vs_noris_pyscf_tda.csv",
    ),
    PlotDataItem(
        "dissociation",
        "h2plus",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx",
        "dense_curve",
        "outputs/remote_h2_h2plus_ground_hfx_density_grid2_def2svp_2000ep_lr1e3_decay200_20260617_124636/h2plus_ion_ground_hfx/h2plus_ground_dense_curve.csv",
        "dissociation/ground_state/h2plus/implicit_scf_hfx/dense_curve.csv",
    ),
    PlotDataItem(
        "dissociation",
        "h2plus",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx",
        "training_points",
        "outputs/remote_h2_h2plus_ground_hfx_density_grid2_def2svp_2000ep_lr1e3_decay200_20260617_124636/h2plus_ion_ground_hfx/h2plus_reference_points.csv",
        "dissociation/ground_state/h2plus/implicit_scf_hfx/training_points.csv",
    ),
    PlotDataItem(
        "dissociation",
        "h2plus",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx",
        "visualization_data",
        "outputs/h2plus_ground_hfx_mode_comparison_20260617/h2plus_ground_hfx_implicit_scf_dissociation_paper_style_visualization_data.csv",
        "dissociation/ground_state/h2plus/implicit_scf_hfx/visualization_data.csv",
    ),
    PlotDataItem(
        "dissociation",
        "h2plus",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx",
        "summary",
        "outputs/remote_h2_h2plus_ground_hfx_density_grid2_def2svp_2000ep_lr1e3_decay200_20260617_124636/h2plus_ion_ground_hfx/summary.json",
        "dissociation/ground_state/h2plus/implicit_scf_hfx/summary.json",
    ),
    PlotDataItem(
        "dissociation",
        "h2plus",
        "ground",
        "ground_state_dissociation",
        "fixed_density_hfx",
        "dense_curve",
        "outputs/remote_h2plus_ground_hfx_fixed_density_grid2_def2svp_2000ep_lr1e3_decay200_20260617_133509/h2plus_ground_dense_curve.csv",
        "dissociation/ground_state/h2plus/fixed_density_hfx/dense_curve.csv",
    ),
    PlotDataItem(
        "dissociation",
        "h2plus",
        "ground",
        "ground_state_dissociation",
        "fixed_density_hfx",
        "training_points",
        "outputs/remote_h2plus_ground_hfx_fixed_density_grid2_def2svp_2000ep_lr1e3_decay200_20260617_133509/h2plus_reference_points.csv",
        "dissociation/ground_state/h2plus/fixed_density_hfx/training_points.csv",
    ),
    PlotDataItem(
        "dissociation",
        "h2plus",
        "ground",
        "ground_state_dissociation",
        "fixed_density_hfx",
        "visualization_data",
        "outputs/h2plus_ground_hfx_mode_comparison_20260617/h2plus_ground_hfx_fixed_density_dissociation_paper_style_visualization_data.csv",
        "dissociation/ground_state/h2plus/fixed_density_hfx/visualization_data.csv",
    ),
    PlotDataItem(
        "dissociation",
        "h2plus",
        "ground",
        "ground_state_dissociation",
        "fixed_density_hfx",
        "summary",
        "outputs/remote_h2plus_ground_hfx_fixed_density_grid2_def2svp_2000ep_lr1e3_decay200_20260617_133509/summary.json",
        "dissociation/ground_state/h2plus/fixed_density_hfx/summary.json",
    ),
    PlotDataItem(
        "dissociation",
        "h2plus",
        "ground",
        "ground_state_dissociation",
        "implicit_vs_fixed_hfx",
        "metrics",
        "outputs/h2plus_ground_hfx_mode_comparison_20260617/h2plus_ground_hfx_split_paper_style_metrics.json",
        "dissociation/ground_state/h2plus/implicit_vs_fixed_hfx/metrics.json",
    ),
    PlotDataItem(
        "dissociation",
        "h2",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx_pt2",
        "dense_curve",
        "outputs/remote_h2_fci_dense100_pt2_nopt2_20260619/pt2_h2_fci_ground_vs_neural_dense_curve.csv",
        "dissociation/ground_state/h2/implicit_scf_hfx_pt2/dense_curve.csv",
    ),
    PlotDataItem(
        "dissociation",
        "h2",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx_pt2",
        "visualization_data",
        "outputs/remote_h2_fci_dense100_pt2_nopt2_20260619/h2_ground_hfx_pt2_dissociation_paper_style_visualization_data.csv",
        "dissociation/ground_state/h2/implicit_scf_hfx_pt2/visualization_data.csv",
    ),
    PlotDataItem(
        "dissociation",
        "h2",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx_pt2",
        "metrics",
        "outputs/remote_h2_fci_dense100_pt2_nopt2_20260619/h2_ground_hfx_pt2_dissociation_paper_style_metrics.json",
        "dissociation/ground_state/h2/implicit_scf_hfx_pt2/metrics.json",
    ),
    PlotDataItem(
        "dissociation",
        "h2",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx_nopt2",
        "dense_curve",
        "outputs/remote_h2_fci_dense100_pt2_nopt2_20260619/nopt2_h2_fci_ground_vs_neural_dense_curve.csv",
        "dissociation/ground_state/h2/implicit_scf_hfx_nopt2/dense_curve.csv",
    ),
    PlotDataItem(
        "dissociation",
        "h2",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx_nopt2",
        "visualization_data",
        "outputs/remote_h2_fci_dense100_pt2_nopt2_20260619/h2_ground_hfx_nopt2_dissociation_paper_style_visualization_data.csv",
        "dissociation/ground_state/h2/implicit_scf_hfx_nopt2/visualization_data.csv",
    ),
    PlotDataItem(
        "dissociation",
        "h2",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx_nopt2",
        "metrics",
        "outputs/remote_h2_fci_dense100_pt2_nopt2_20260619/h2_ground_hfx_nopt2_dissociation_paper_style_metrics.json",
        "dissociation/ground_state/h2/implicit_scf_hfx_nopt2/metrics.json",
    ),
    PlotDataItem(
        "dissociation",
        "h2",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx_pt2_vs_nopt2",
        "metrics",
        "outputs/remote_h2_fci_dense100_pt2_nopt2_20260619/h2_ground_hfx_pt2_nopt2_dissociation_paper_style_metrics.json",
        "dissociation/ground_state/h2/implicit_scf_hfx_pt2_vs_nopt2/metrics.json",
    ),
    PlotDataItem(
        "dissociation",
        "n2",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx_pt2_train7",
        "predictions",
        "outputs/remote_n2_pt2_train7_all35_eval/n2_ccsdt_ground_predictions.csv",
        "dissociation/ground_state/n2/implicit_scf_hfx_pt2_train7/predictions.csv",
    ),
    PlotDataItem(
        "dissociation",
        "n2",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx_pt2_train7",
        "visualization_data",
        "outputs/remote_n2_pt2_train7_all35_eval/n2_ground_hfx_pt2_dissociation_paper_style_visualization_data.csv",
        "dissociation/ground_state/n2/implicit_scf_hfx_pt2_train7/visualization_data.csv",
    ),
    PlotDataItem(
        "dissociation",
        "n2",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx_pt2_train7",
        "metrics",
        "outputs/remote_n2_pt2_train7_all35_eval/n2_ground_hfx_pt2_dissociation_paper_style_metrics.json",
        "dissociation/ground_state/n2/implicit_scf_hfx_pt2_train7/metrics.json",
    ),
    PlotDataItem(
        "dissociation",
        "n2",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx_pt2_train7",
        "summary",
        "outputs/remote_n2_pt2_train7_all35_eval/summary.json",
        "dissociation/ground_state/n2/implicit_scf_hfx_pt2_train7/summary.json",
    ),
    PlotDataItem(
        "dissociation",
        "n2",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx_nopt2_train7",
        "predictions",
        "outputs/remote_n2_nopt2_train7_all35_eval/n2_ccsdt_ground_predictions.csv",
        "dissociation/ground_state/n2/implicit_scf_hfx_nopt2_train7/predictions.csv",
    ),
    PlotDataItem(
        "dissociation",
        "n2",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx_nopt2_train7",
        "visualization_data",
        "outputs/remote_n2_nopt2_train7_all35_eval/n2_ground_hfx_nopt2_dissociation_paper_style_visualization_data.csv",
        "dissociation/ground_state/n2/implicit_scf_hfx_nopt2_train7/visualization_data.csv",
    ),
    PlotDataItem(
        "dissociation",
        "n2",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx_nopt2_train7",
        "metrics",
        "outputs/remote_n2_nopt2_train7_all35_eval/n2_ground_hfx_nopt2_dissociation_paper_style_metrics.json",
        "dissociation/ground_state/n2/implicit_scf_hfx_nopt2_train7/metrics.json",
    ),
    PlotDataItem(
        "dissociation",
        "n2",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx_nopt2_train7",
        "summary",
        "outputs/remote_n2_nopt2_train7_all35_eval/summary.json",
        "dissociation/ground_state/n2/implicit_scf_hfx_nopt2_train7/summary.json",
    ),
    PlotDataItem(
        "dissociation",
        "n2",
        "ground",
        "ground_state_dissociation",
        "fixed_density_hfx_pt2_all35",
        "predictions",
        "outputs/remote_n2_fixed_density_all35_20260623/n2_fixed_hfx_pt2_all35/n2_ccsdt_ground_predictions.csv",
        "dissociation/ground_state/n2/fixed_density_hfx_pt2_all35/predictions.csv",
    ),
    PlotDataItem(
        "dissociation",
        "n2",
        "ground",
        "ground_state_dissociation",
        "fixed_density_hfx_pt2_all35",
        "visualization_data",
        "outputs/remote_n2_fixed_density_all35_20260623/n2_fixed_hfx_pt2_all35/n2_fixed_density_hfx_pt2_dissociation_paper_style_visualization_data.csv",
        "dissociation/ground_state/n2/fixed_density_hfx_pt2_all35/visualization_data.csv",
    ),
    PlotDataItem(
        "dissociation",
        "n2",
        "ground",
        "ground_state_dissociation",
        "fixed_density_hfx_pt2_all35",
        "metrics",
        "outputs/remote_n2_fixed_density_all35_20260623/n2_fixed_hfx_pt2_all35/n2_fixed_density_hfx_pt2_dissociation_paper_style_metrics.json",
        "dissociation/ground_state/n2/fixed_density_hfx_pt2_all35/metrics.json",
    ),
    PlotDataItem(
        "dissociation",
        "n2",
        "ground",
        "ground_state_dissociation",
        "fixed_density_hfx_nopt2_all35",
        "predictions",
        "outputs/remote_n2_fixed_density_all35_20260623/n2_fixed_hfx_nopt2_all35/n2_ccsdt_ground_predictions.csv",
        "dissociation/ground_state/n2/fixed_density_hfx_nopt2_all35/predictions.csv",
    ),
    PlotDataItem(
        "dissociation",
        "n2",
        "ground",
        "ground_state_dissociation",
        "fixed_density_hfx_nopt2_all35",
        "visualization_data",
        "outputs/remote_n2_fixed_density_all35_20260623/n2_fixed_hfx_nopt2_all35/n2_fixed_density_hfx_nopt2_dissociation_paper_style_visualization_data.csv",
        "dissociation/ground_state/n2/fixed_density_hfx_nopt2_all35/visualization_data.csv",
    ),
    PlotDataItem(
        "dissociation",
        "n2",
        "ground",
        "ground_state_dissociation",
        "fixed_density_hfx_nopt2_all35",
        "metrics",
        "outputs/remote_n2_fixed_density_all35_20260623/n2_fixed_hfx_nopt2_all35/n2_fixed_density_hfx_nopt2_dissociation_paper_style_metrics.json",
        "dissociation/ground_state/n2/fixed_density_hfx_nopt2_all35/metrics.json",
    ),
    PlotDataItem(
        "dissociation",
        "n2",
        "ground",
        "ground_state_dissociation",
        "fixed_density_hfx_pt2_vs_nopt2_all35",
        "metrics",
        "outputs/remote_n2_fixed_density_all35_20260623/n2_fixed_density_hfx_pt2_nopt2_dissociation_paper_style_metrics.json",
        "dissociation/ground_state/n2/fixed_density_hfx_pt2_vs_nopt2_all35/metrics.json",
    ),
    PlotDataItem(
        "dissociation",
        "h2",
        "s1",
        "s1_total_energy_dissociation",
        "tda_hfx_pt2",
        "dense_curve",
        "benchmark/h2_s1_e1total_pt2_strict_20260617/h2_s1_tda_dense_curve.csv",
        "dissociation/excited_state/h2_s1/tda_hfx_pt2/dense_curve.csv",
    ),
    PlotDataItem(
        "dissociation",
        "h2",
        "s1",
        "s1_total_energy_dissociation",
        "tda_hfx_pt2",
        "excited_curve",
        "benchmark/h2_s1_e1total_pt2_strict_20260617/h2_s1_tda_excited_curve.csv",
        "dissociation/excited_state/h2_s1/tda_hfx_pt2/excited_curve.csv",
    ),
    PlotDataItem(
        "dissociation",
        "h2",
        "s1",
        "s1_total_energy_dissociation",
        "tda_hfx_pt2",
        "visualization_data",
        "benchmark/h2_s1_e1total_pt2_strict_20260617/h2_pt2_s1_tda_e1_total_dissociation_paper_style_visualization_data.csv",
        "dissociation/excited_state/h2_s1/tda_hfx_pt2/visualization_data.csv",
    ),
    PlotDataItem(
        "dissociation",
        "h2",
        "s1",
        "s1_total_energy_dissociation",
        "tda_hfx_pt2",
        "metrics",
        "benchmark/h2_s1_e1total_pt2_strict_20260617/h2_pt2_s1_tda_e1_total_dissociation_paper_style_metrics.json",
        "dissociation/excited_state/h2_s1/tda_hfx_pt2/metrics.json",
    ),
    PlotDataItem(
        "dissociation",
        "h2",
        "s1",
        "s1_total_energy_dissociation",
        "tda_hfx_nopt2",
        "dense_curve",
        "benchmark/h2_nopt2_dense100_visualization_20260624/h2_s1_tda_dense_curve.csv",
        "dissociation/excited_state/h2_s1/tda_hfx_nopt2/dense_curve.csv",
    ),
    PlotDataItem(
        "dissociation",
        "h2",
        "s1",
        "s1_total_energy_dissociation",
        "tda_hfx_nopt2",
        "excited_curve",
        "benchmark/h2_nopt2_dense100_visualization_20260624/h2_s1_tda_excited_curve.csv",
        "dissociation/excited_state/h2_s1/tda_hfx_nopt2/excited_curve.csv",
    ),
    PlotDataItem(
        "dissociation",
        "h2",
        "s1",
        "s1_total_energy_dissociation",
        "tda_hfx_nopt2",
        "visualization_data",
        "benchmark/h2_nopt2_dense100_visualization_20260624/h2_nopt2_s1_tda_e1_total_dissociation_paper_style_visualization_data.csv",
        "dissociation/excited_state/h2_s1/tda_hfx_nopt2/visualization_data.csv",
    ),
    PlotDataItem(
        "dissociation",
        "h2",
        "s1",
        "s1_total_energy_dissociation",
        "tda_hfx_nopt2",
        "metrics",
        "benchmark/h2_nopt2_dense100_visualization_20260624/h2_nopt2_s1_tda_e1_total_dissociation_paper_style_metrics.json",
        "dissociation/excited_state/h2_s1/tda_hfx_nopt2/metrics.json",
    ),
    PlotDataItem(
        "dissociation",
        "h2",
        "s1",
        "s1_total_energy_dissociation",
        "tda_hfx_pt2_vs_nopt2",
        "metrics",
        "benchmark/h2_s1_e1total_pt2_strict_20260617/h2_s1_tda_e1_total_pt2_nopt2_dissociation_paper_style_metrics.json",
        "dissociation/excited_state/h2_s1/tda_hfx_pt2_vs_nopt2/metrics.json",
    ),
    PlotDataItem(
        "dissociation",
        "n2",
        "s1",
        "s1_total_energy_dissociation",
        "tda_hfx_pt2_train7",
        "dense_curve",
        "benchmark/n2_hammami_s1total_tda_chunked_eval_skiposc_20260623_094250/pt2strict_dense_curve_35pt_merged.csv",
        "dissociation/excited_state/n2_s1/tda_hfx_pt2_train7/dense_curve.csv",
    ),
    PlotDataItem(
        "dissociation",
        "n2",
        "s1",
        "s1_total_energy_dissociation",
        "tda_hfx_pt2_train7",
        "visualization_data",
        "benchmark/n2_hammami_s1total_tda_chunked_eval_skiposc_20260623_094250/n2_pt2strict_s1_tda_e1_total_dissociation_paper_style_visualization_data.csv",
        "dissociation/excited_state/n2_s1/tda_hfx_pt2_train7/visualization_data.csv",
    ),
    PlotDataItem(
        "dissociation",
        "n2",
        "s1",
        "s1_total_energy_dissociation",
        "tda_hfx_pt2_train7",
        "metrics",
        "benchmark/n2_hammami_s1total_tda_chunked_eval_skiposc_20260623_094250/n2_pt2strict_s1_tda_e1_total_dissociation_paper_style_metrics.json",
        "dissociation/excited_state/n2_s1/tda_hfx_pt2_train7/metrics.json",
    ),
    PlotDataItem(
        "dissociation",
        "n2",
        "s1",
        "s1_total_energy_dissociation",
        "tda_hfx_nopt2_train7",
        "dense_curve",
        "benchmark/n2_hammami_s1total_tda_chunked_eval_skiposc_20260623_094250/nopt2_dense_curve_35pt_merged.csv",
        "dissociation/excited_state/n2_s1/tda_hfx_nopt2_train7/dense_curve.csv",
    ),
    PlotDataItem(
        "dissociation",
        "n2",
        "s1",
        "s1_total_energy_dissociation",
        "tda_hfx_nopt2_train7",
        "visualization_data",
        "benchmark/n2_hammami_s1total_tda_chunked_eval_skiposc_20260623_094250/n2_nopt2_s1_tda_e1_total_dissociation_paper_style_visualization_data.csv",
        "dissociation/excited_state/n2_s1/tda_hfx_nopt2_train7/visualization_data.csv",
    ),
    PlotDataItem(
        "dissociation",
        "n2",
        "s1",
        "s1_total_energy_dissociation",
        "tda_hfx_nopt2_train7",
        "metrics",
        "benchmark/n2_hammami_s1total_tda_chunked_eval_skiposc_20260623_094250/n2_nopt2_s1_tda_e1_total_dissociation_paper_style_metrics.json",
        "dissociation/excited_state/n2_s1/tda_hfx_nopt2_train7/metrics.json",
    ),
    PlotDataItem(
        "dissociation",
        "n2",
        "s1",
        "s1_total_energy_dissociation",
        "tda_hfx_pt2_vs_nopt2_train7",
        "visualization_data",
        "benchmark/n2_hammami_s1total_tda_chunked_eval_skiposc_20260623_094250/n2_pt2_vs_nopt2_s1_total_curve_visualization_data.csv",
        "dissociation/excited_state/n2_s1/tda_hfx_pt2_vs_nopt2_train7/visualization_data.csv",
    ),
    PlotDataItem(
        "dissociation",
        "n2",
        "s1",
        "s1_total_energy_dissociation",
        "tda_hfx_pt2_vs_nopt2_train7",
        "metrics",
        "benchmark/n2_hammami_s1total_tda_chunked_eval_skiposc_20260623_094250/n2_s1_tda_e1_total_pt2_nopt2_dissociation_paper_style_metrics.json",
        "dissociation/excited_state/n2_s1/tda_hfx_pt2_vs_nopt2_train7/metrics.json",
    ),
    PlotDataItem(
        "reference",
        "n2",
        "s0_s1",
        "hammami_2026_large_cas",
        "a1pig",
        "reference_curve",
        "benchmark/reference_curves/n2_hammami_2026_a1pig_s1_reference_large_cas.csv",
        "reference_curves/n2/hammami_2026_a1pig_s1_reference_large_cas.csv",
    ),
    PlotDataItem(
        "reference",
        "n2",
        "s0_s1",
        "hammami_2026_large_cas",
        "s0_s1_a1pig",
        "plot_data",
        "benchmark/reference_curves/n2_hammami_2026_s0_s1_a1pig_plot_data.csv",
        "reference_curves/n2/hammami_2026_s0_s1_a1pig_plot_data.csv",
    ),
    PlotDataItem(
        "reference",
        "n2",
        "s0_s1",
        "hammami_2026_large_cas",
        "s0_s1_a1pig_35pt",
        "plot_data",
        "benchmark/reference_curves/n2_hammami_2026_s0_s1_a1pig_35pt_plot_data.csv",
        "reference_curves/n2/hammami_2026_s0_s1_a1pig_35pt_plot_data.csv",
    ),
)


LOSS_ITEMS: tuple[PlotDataItem, ...] = (
    PlotDataItem(
        "loss_curve",
        "h2plus",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx",
        "training_history",
        "outputs/remote_h2_h2plus_ground_hfx_density_grid2_def2svp_2000ep_lr1e3_decay200_20260617_124636/h2plus_ion_ground_hfx/training_history.csv",
        "loss_curves/dissociation/ground_state/h2plus/implicit_scf_hfx/training_history.csv",
    ),
    PlotDataItem(
        "loss_curve",
        "h2plus",
        "ground",
        "ground_state_dissociation",
        "fixed_density_hfx",
        "training_history",
        "outputs/remote_h2plus_ground_hfx_fixed_density_grid2_def2svp_2000ep_lr1e3_decay200_20260617_133509/training_history.csv",
        "loss_curves/dissociation/ground_state/h2plus/fixed_density_hfx/training_history.csv",
    ),
    PlotDataItem(
        "loss_curve",
        "h2",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx_pt2",
        "training_curve",
        f"{RAW_REMOTE_OUTPUTS}/h2_fci_ground_train7_def2tzvp_grid2_hfx_pt2_vs_nopt2_density1_trainonly_2000ep_lr1e3_decay200_20260619_130646/hfx_pt2/training_curve.csv",
        "loss_curves/dissociation/ground_state/h2/implicit_scf_hfx_pt2/training_curve.csv",
    ),
    PlotDataItem(
        "loss_curve",
        "h2",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx_pt2",
        "training_loss_png",
        f"{RAW_REMOTE_OUTPUTS}/h2_fci_ground_train7_def2tzvp_grid2_hfx_pt2_vs_nopt2_density1_trainonly_2000ep_lr1e3_decay200_20260619_130646/hfx_pt2/training_loss.png",
        "loss_curves/dissociation/ground_state/h2/implicit_scf_hfx_pt2/training_loss.png",
    ),
    PlotDataItem(
        "loss_curve",
        "h2",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx_pt2",
        "summary",
        f"{RAW_REMOTE_OUTPUTS}/h2_fci_ground_train7_def2tzvp_grid2_hfx_pt2_vs_nopt2_density1_trainonly_2000ep_lr1e3_decay200_20260619_130646/hfx_pt2/summary.json",
        "loss_curves/dissociation/ground_state/h2/implicit_scf_hfx_pt2/summary.json",
    ),
    PlotDataItem(
        "loss_curve",
        "h2",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx_nopt2",
        "training_curve",
        f"{RAW_REMOTE_OUTPUTS}/h2_fci_ground_train7_def2tzvp_grid2_hfx_pt2_vs_nopt2_density1_trainonly_2000ep_lr1e3_decay200_20260619_130646/hfx_nopt2/training_curve.csv",
        "loss_curves/dissociation/ground_state/h2/implicit_scf_hfx_nopt2/training_curve.csv",
    ),
    PlotDataItem(
        "loss_curve",
        "h2",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx_nopt2",
        "training_loss_png",
        f"{RAW_REMOTE_OUTPUTS}/h2_fci_ground_train7_def2tzvp_grid2_hfx_pt2_vs_nopt2_density1_trainonly_2000ep_lr1e3_decay200_20260619_130646/hfx_nopt2/training_loss.png",
        "loss_curves/dissociation/ground_state/h2/implicit_scf_hfx_nopt2/training_loss.png",
    ),
    PlotDataItem(
        "loss_curve",
        "h2",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx_nopt2",
        "summary",
        f"{RAW_REMOTE_OUTPUTS}/h2_fci_ground_train7_def2tzvp_grid2_hfx_pt2_vs_nopt2_density1_trainonly_2000ep_lr1e3_decay200_20260619_130646/hfx_nopt2/summary.json",
        "loss_curves/dissociation/ground_state/h2/implicit_scf_hfx_nopt2/summary.json",
    ),
    PlotDataItem(
        "loss_curve",
        "h2",
        "ground",
        "fixed_density_ground_state",
        "fixed_density_hfx_pt2",
        "training_curve",
        f"{RAW_REMOTE_OUTPUTS}/fixed_density_h2_def2tzvp_grid2_hfx_pt2_nopt2_density0_2000ep_lr1e3_decay200_20260623_091823/h2_fixed_hfx_pt2/training_curve.csv",
        "loss_curves/dissociation/ground_state/h2/fixed_density_hfx_pt2/training_curve.csv",
    ),
    PlotDataItem(
        "loss_curve",
        "h2",
        "ground",
        "fixed_density_ground_state",
        "fixed_density_hfx_pt2",
        "training_loss_png",
        f"{RAW_REMOTE_OUTPUTS}/fixed_density_h2_def2tzvp_grid2_hfx_pt2_nopt2_density0_2000ep_lr1e3_decay200_20260623_091823/h2_fixed_hfx_pt2/training_loss.png",
        "loss_curves/dissociation/ground_state/h2/fixed_density_hfx_pt2/training_loss.png",
    ),
    PlotDataItem(
        "loss_curve",
        "h2",
        "ground",
        "fixed_density_ground_state",
        "fixed_density_hfx_pt2",
        "summary",
        f"{RAW_REMOTE_OUTPUTS}/fixed_density_h2_def2tzvp_grid2_hfx_pt2_nopt2_density0_2000ep_lr1e3_decay200_20260623_091823/h2_fixed_hfx_pt2/summary.json",
        "loss_curves/dissociation/ground_state/h2/fixed_density_hfx_pt2/summary.json",
    ),
    PlotDataItem(
        "loss_curve",
        "h2",
        "ground",
        "fixed_density_ground_state",
        "fixed_density_hfx_nopt2",
        "training_curve",
        f"{RAW_REMOTE_OUTPUTS}/fixed_density_h2_def2tzvp_grid2_hfx_pt2_nopt2_density0_2000ep_lr1e3_decay200_20260623_091823/h2_fixed_hfx_nopt2/training_curve.csv",
        "loss_curves/dissociation/ground_state/h2/fixed_density_hfx_nopt2/training_curve.csv",
    ),
    PlotDataItem(
        "loss_curve",
        "h2",
        "ground",
        "fixed_density_ground_state",
        "fixed_density_hfx_nopt2",
        "training_loss_png",
        f"{RAW_REMOTE_OUTPUTS}/fixed_density_h2_def2tzvp_grid2_hfx_pt2_nopt2_density0_2000ep_lr1e3_decay200_20260623_091823/h2_fixed_hfx_nopt2/training_loss.png",
        "loss_curves/dissociation/ground_state/h2/fixed_density_hfx_nopt2/training_loss.png",
    ),
    PlotDataItem(
        "loss_curve",
        "h2",
        "ground",
        "fixed_density_ground_state",
        "fixed_density_hfx_nopt2",
        "summary",
        f"{RAW_REMOTE_OUTPUTS}/fixed_density_h2_def2tzvp_grid2_hfx_pt2_nopt2_density0_2000ep_lr1e3_decay200_20260623_091823/h2_fixed_hfx_nopt2/summary.json",
        "loss_curves/dissociation/ground_state/h2/fixed_density_hfx_nopt2/summary.json",
    ),
    PlotDataItem(
        "loss_curve",
        "n2",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx_pt2_train7",
        "training_history",
        f"{RAW_REMOTE_OUTPUTS}/n2_mrccca_ground_train7_def2tzvp_grid2_energyonly_hfx_pt2_h128_2000ep_lr1e3_decay200_gpu5_20260618_143754/training_history.csv",
        "loss_curves/dissociation/ground_state/n2/implicit_scf_hfx_pt2_train7/training_history.csv",
    ),
    PlotDataItem(
        "loss_curve",
        "n2",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx_pt2_train7",
        "training_loss_png",
        f"{RAW_REMOTE_OUTPUTS}/n2_mrccca_ground_train7_def2tzvp_grid2_energyonly_hfx_pt2_h128_2000ep_lr1e3_decay200_gpu5_20260618_143754/training_loss.png",
        "loss_curves/dissociation/ground_state/n2/implicit_scf_hfx_pt2_train7/training_loss.png",
    ),
    PlotDataItem(
        "loss_curve",
        "n2",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx_pt2_train7",
        "summary",
        f"{RAW_REMOTE_OUTPUTS}/n2_mrccca_ground_train7_def2tzvp_grid2_energyonly_hfx_pt2_h128_2000ep_lr1e3_decay200_gpu5_20260618_143754/summary.json",
        "loss_curves/dissociation/ground_state/n2/implicit_scf_hfx_pt2_train7/summary.json",
    ),
    PlotDataItem(
        "loss_curve",
        "n2",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx_nopt2_train7",
        "training_history",
        f"{RAW_REMOTE_OUTPUTS}/n2_mrccca_ground_train7_def2tzvp_grid2_energyonly_hfx_nopt2_h128_2000ep_lr1e3_decay200_gpu6_20260618_150147/training_history.csv",
        "loss_curves/dissociation/ground_state/n2/implicit_scf_hfx_nopt2_train7/training_history.csv",
    ),
    PlotDataItem(
        "loss_curve",
        "n2",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx_nopt2_train7",
        "training_loss_png",
        f"{RAW_REMOTE_OUTPUTS}/n2_mrccca_ground_train7_def2tzvp_grid2_energyonly_hfx_nopt2_h128_2000ep_lr1e3_decay200_gpu6_20260618_150147/training_loss.png",
        "loss_curves/dissociation/ground_state/n2/implicit_scf_hfx_nopt2_train7/training_loss.png",
    ),
    PlotDataItem(
        "loss_curve",
        "n2",
        "ground",
        "ground_state_dissociation",
        "implicit_scf_hfx_nopt2_train7",
        "summary",
        f"{RAW_REMOTE_OUTPUTS}/n2_mrccca_ground_train7_def2tzvp_grid2_energyonly_hfx_nopt2_h128_2000ep_lr1e3_decay200_gpu6_20260618_150147/summary.json",
        "loss_curves/dissociation/ground_state/n2/implicit_scf_hfx_nopt2_train7/summary.json",
    ),
    PlotDataItem(
        "loss_curve",
        "n2",
        "ground",
        "fixed_density_ground_state",
        "fixed_density_hfx_pt2_all35",
        "training_history",
        "outputs/remote_fixed_density_h2_n2_20260623/n2_fixed_hfx_pt2/training_history.csv",
        "loss_curves/dissociation/ground_state/n2/fixed_density_hfx_pt2_all35/training_history.csv",
    ),
    PlotDataItem(
        "loss_curve",
        "n2",
        "ground",
        "fixed_density_ground_state",
        "fixed_density_hfx_pt2_all35",
        "training_per_point_history",
        "outputs/remote_fixed_density_h2_n2_20260623/n2_fixed_hfx_pt2/training_per_point_history.csv",
        "loss_curves/dissociation/ground_state/n2/fixed_density_hfx_pt2_all35/training_per_point_history.csv",
    ),
    PlotDataItem(
        "loss_curve",
        "n2",
        "ground",
        "fixed_density_ground_state",
        "fixed_density_hfx_pt2_all35",
        "training_loss_png",
        "outputs/remote_fixed_density_h2_n2_20260623/n2_fixed_hfx_pt2/training_loss.png",
        "loss_curves/dissociation/ground_state/n2/fixed_density_hfx_pt2_all35/training_loss.png",
    ),
    PlotDataItem(
        "loss_curve",
        "n2",
        "ground",
        "fixed_density_ground_state",
        "fixed_density_hfx_pt2_all35",
        "summary",
        "outputs/remote_fixed_density_h2_n2_20260623/n2_fixed_hfx_pt2/summary.json",
        "loss_curves/dissociation/ground_state/n2/fixed_density_hfx_pt2_all35/summary.json",
    ),
    PlotDataItem(
        "loss_curve",
        "n2",
        "ground",
        "fixed_density_ground_state",
        "fixed_density_hfx_nopt2_all35",
        "training_history",
        "outputs/remote_fixed_density_h2_n2_20260623/n2_fixed_hfx_nopt2/training_history.csv",
        "loss_curves/dissociation/ground_state/n2/fixed_density_hfx_nopt2_all35/training_history.csv",
    ),
    PlotDataItem(
        "loss_curve",
        "n2",
        "ground",
        "fixed_density_ground_state",
        "fixed_density_hfx_nopt2_all35",
        "training_per_point_history",
        "outputs/remote_fixed_density_h2_n2_20260623/n2_fixed_hfx_nopt2/training_per_point_history.csv",
        "loss_curves/dissociation/ground_state/n2/fixed_density_hfx_nopt2_all35/training_per_point_history.csv",
    ),
    PlotDataItem(
        "loss_curve",
        "n2",
        "ground",
        "fixed_density_ground_state",
        "fixed_density_hfx_nopt2_all35",
        "training_loss_png",
        "outputs/remote_fixed_density_h2_n2_20260623/n2_fixed_hfx_nopt2/training_loss.png",
        "loss_curves/dissociation/ground_state/n2/fixed_density_hfx_nopt2_all35/training_loss.png",
    ),
    PlotDataItem(
        "loss_curve",
        "n2",
        "ground",
        "fixed_density_ground_state",
        "fixed_density_hfx_nopt2_all35",
        "summary",
        "outputs/remote_fixed_density_h2_n2_20260623/n2_fixed_hfx_nopt2/summary.json",
        "loss_curves/dissociation/ground_state/n2/fixed_density_hfx_nopt2_all35/summary.json",
    ),
    PlotDataItem(
        "loss_curve",
        "h2",
        "s1",
        "s1_total_energy_dissociation",
        "tda_hfx_pt2",
        "training_history",
        f"{RAW_REMOTE_OUTPUTS}/remote_h2_s1_e1total_def2svp_grid2_lr1e4_decay400_deferdense_20260617_141349/hf_pt2_strict/training_history.csv",
        "loss_curves/dissociation/excited_state/h2_s1/tda_hfx_pt2/training_history.csv",
    ),
    PlotDataItem(
        "loss_curve",
        "h2",
        "s1",
        "s1_total_energy_dissociation",
        "tda_hfx_pt2",
        "training_loss_png",
        f"{RAW_REMOTE_OUTPUTS}/remote_h2_s1_e1total_def2svp_grid2_lr1e4_decay400_deferdense_20260617_141349/hf_pt2_strict/training_loss.png",
        "loss_curves/dissociation/excited_state/h2_s1/tda_hfx_pt2/training_loss.png",
    ),
    PlotDataItem(
        "loss_curve",
        "h2",
        "s1",
        "s1_total_energy_dissociation",
        "tda_hfx_pt2",
        "summary",
        f"{RAW_REMOTE_OUTPUTS}/remote_h2_s1_e1total_def2svp_grid2_lr1e4_decay400_deferdense_20260617_141349/hf_pt2_strict/summary.json",
        "loss_curves/dissociation/excited_state/h2_s1/tda_hfx_pt2/summary.json",
    ),
    PlotDataItem(
        "loss_curve",
        "h2",
        "s1",
        "s1_total_energy_dissociation",
        "tda_hfx_nopt2",
        "training_history",
        f"{RAW_REMOTE_OUTPUTS}/h2_s1total_tda_train7_hfx_pt2_vs_nopt2_h128_def2tzvp_grid2_trainonly_20260623_150905/hfx_nopt2_dense100_best_pyscfstyle_gpu0_20260624_092133/training_history.csv",
        "loss_curves/dissociation/excited_state/h2_s1/tda_hfx_nopt2/training_history.csv",
    ),
    PlotDataItem(
        "loss_curve",
        "h2",
        "s1",
        "s1_total_energy_dissociation",
        "tda_hfx_nopt2",
        "training_loss_png",
        f"{RAW_REMOTE_OUTPUTS}/h2_s1total_tda_train7_hfx_pt2_vs_nopt2_h128_def2tzvp_grid2_trainonly_20260623_150905/hfx_nopt2_dense100_best_pyscfstyle_gpu0_20260624_092133/training_loss.png",
        "loss_curves/dissociation/excited_state/h2_s1/tda_hfx_nopt2/training_loss.png",
    ),
    PlotDataItem(
        "loss_curve",
        "h2",
        "s1",
        "s1_total_energy_dissociation",
        "tda_hfx_nopt2",
        "summary",
        f"{RAW_REMOTE_OUTPUTS}/h2_s1total_tda_train7_hfx_pt2_vs_nopt2_h128_def2tzvp_grid2_trainonly_20260623_150905/hfx_nopt2_dense100_best_pyscfstyle_gpu0_20260624_092133/summary.json",
        "loss_curves/dissociation/excited_state/h2_s1/tda_hfx_nopt2/summary.json",
    ),
    PlotDataItem(
        "loss_curve",
        "n2",
        "s1",
        "s1_total_energy_dissociation",
        "tda_hfx_pt2_train7",
        "training_history",
        f"{RAW_REMOTE_OUTPUTS}/n2_hammami_s1total_tda_train7_all35_hfx_pt2strict_h128_def2tzvp_grid2_restart_jaxscf_20260622_085508/training_history.csv",
        "loss_curves/dissociation/excited_state/n2_s1/tda_hfx_pt2_train7/training_history.csv",
    ),
    PlotDataItem(
        "loss_curve",
        "n2",
        "s1",
        "s1_total_energy_dissociation",
        "tda_hfx_pt2_train7",
        "training_loss_png",
        f"{RAW_REMOTE_OUTPUTS}/n2_hammami_s1total_tda_train7_all35_hfx_pt2strict_h128_def2tzvp_grid2_restart_jaxscf_20260622_085508/training_loss.png",
        "loss_curves/dissociation/excited_state/n2_s1/tda_hfx_pt2_train7/training_loss.png",
    ),
    PlotDataItem(
        "loss_curve",
        "n2",
        "s1",
        "s1_total_energy_dissociation",
        "tda_hfx_nopt2_train7",
        "training_history",
        f"{RAW_REMOTE_OUTPUTS}/n2_hammami_s1total_tda_train7_all35_hfx_nopt2_h128_def2tzvp_grid2_restart_ckpt_gpu6uuid_20260622_1035/training_history.csv",
        "loss_curves/dissociation/excited_state/n2_s1/tda_hfx_nopt2_train7/training_history.csv",
    ),
    PlotDataItem(
        "loss_curve",
        "n2",
        "s1",
        "s1_total_energy_dissociation",
        "tda_hfx_nopt2_train7",
        "training_loss_png",
        f"{RAW_REMOTE_OUTPUTS}/n2_hammami_s1total_tda_train7_all35_hfx_nopt2_h128_def2tzvp_grid2_restart_ckpt_gpu6uuid_20260622_1035/training_loss.png",
        "loss_curves/dissociation/excited_state/n2_s1/tda_hfx_nopt2_train7/training_loss.png",
    ),
)


SEMILOCAL_PROTOCOL = (
    {
        "variant": "lda_vwn_rpa",
        "semilocal_components": ["lda_x", "lda_c_vwn_rpa"],
        "system": "N2",
        "states": ["ground", "S1"],
        "scf_mode": "implicit_scf",
        "hfx": "enabled",
        "pt2": "disabled",
        "target_curves": ["E0(R)", "Omega1(R)", "E1(R)=E0(R)+Omega1(R)"],
    },
    {
        "variant": "gga_pbe",
        "semilocal_components": ["gga_x_pbe", "gga_c_pbe"],
        "system": "N2",
        "states": ["ground", "S1"],
        "scf_mode": "implicit_scf",
        "hfx": "enabled",
        "pt2": "disabled",
        "target_curves": ["E0(R)", "Omega1(R)", "E1(R)=E0(R)+Omega1(R)"],
    },
    {
        "variant": "mgga_r2scan",
        "semilocal_components": ["mgga_x_r2scan", "mgga_c_r2scan"],
        "system": "N2",
        "states": ["ground", "S1"],
        "scf_mode": "implicit_scf",
        "hfx": "enabled",
        "pt2": "disabled",
        "target_curves": ["E0(R)", "Omega1(R)", "E1(R)=E0(R)+Omega1(R)"],
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_csv_rows(path: Path) -> int | None:
    if path.suffix.lower() != ".csv":
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def copy_item(item: PlotDataItem) -> dict[str, object]:
    source = ROOT / item.source
    dest = OUT_ROOT / item.dest
    status = "copied"
    row_count = None
    size_bytes = None
    sha256 = None
    if source.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        row_count = count_csv_rows(dest)
        size_bytes = dest.stat().st_size
        sha256 = sha256_file(dest)
    else:
        status = "missing"
    return {
        "category": item.category,
        "system": item.system,
        "state": item.state,
        "task": item.task,
        "variant": item.variant,
        "role": item.role,
        "status": status,
        "source_path": item.source,
        "dest_path": str(Path("benchmark") / "plot_data" / item.dest),
        "row_count": row_count,
        "size_bytes": size_bytes,
        "sha256": sha256,
    }


def write_semilocal_protocol() -> list[dict[str, object]]:
    protocol_dir = OUT_ROOT / "semilocal_channel" / "n2_ground_s1_no_pt2_hfx_implicit_scf"
    protocol_dir.mkdir(parents=True, exist_ok=True)

    protocol_json = protocol_dir / "protocol.json"
    protocol_json.write_text(
        json.dumps(SEMILOCAL_PROTOCOL, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    protocol_csv = protocol_dir / "protocol.csv"
    with protocol_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "semilocal_components",
                "system",
                "states",
                "scf_mode",
                "hfx",
                "pt2",
                "target_curves",
            ],
        )
        writer.writeheader()
        for row in SEMILOCAL_PROTOCOL:
            writer.writerow(
                {
                    **row,
                    "semilocal_components": "+".join(row["semilocal_components"]),
                    "states": ";".join(row["states"]),
                    "target_curves": ";".join(row["target_curves"]),
                }
            )

    return [
        {
            "category": "semilocal_channel",
            "system": "n2",
            "state": "ground+s1",
            "task": "semilocal_channel_composition",
            "variant": "lda_vwn_rpa+gga_pbe+mgga_r2scan",
            "role": "protocol",
            "status": "written",
            "source_path": "",
            "dest_path": str(protocol_csv.relative_to(ROOT)),
            "row_count": len(SEMILOCAL_PROTOCOL),
            "size_bytes": protocol_csv.stat().st_size,
            "sha256": sha256_file(protocol_csv),
        },
        {
            "category": "semilocal_channel",
            "system": "n2",
            "state": "ground+s1",
            "task": "semilocal_channel_composition",
            "variant": "lda_vwn_rpa+gga_pbe+mgga_r2scan",
            "role": "protocol_json",
            "status": "written",
            "source_path": "",
            "dest_path": str(protocol_json.relative_to(ROOT)),
            "row_count": None,
            "size_bytes": protocol_json.stat().st_size,
            "sha256": sha256_file(protocol_json),
        },
    ]


def write_manifest(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "category",
        "system",
        "state",
        "task",
        "variant",
        "role",
        "status",
        "source_path",
        "dest_path",
        "row_count",
        "size_bytes",
        "sha256",
    ]
    manifest_csv = OUT_ROOT / "MANIFEST.csv"
    manifest_json = OUT_ROOT / "MANIFEST.json"
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    manifest_json.write_text(
        json.dumps(rows, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def write_readme(rows: list[dict[str, object]]) -> None:
    copied = sum(1 for row in rows if row["status"] in {"copied", "written"})
    missing = [row for row in rows if row["status"] == "missing"]
    readme = f"""# Plot Data Index

This directory contains normalized, manuscript-facing plotting data.
It is a copied data index: original run directories under `outputs/`
and legacy `benchmark/` folders are preserved.

## Layout

- `validation/`: PySCF/GradTDDFT correctness and response-kernel
  validation tables.
- `dissociation/ground_state/`: H2+, H2, and N2 ground-state
  dissociation curves.
- `dissociation/excited_state/`: H2 and N2 first-excited-state
  dissociation curves.
- `loss_curves/`: training loss histories and loss-curve PNGs for the
  dissociation models.
- `reference_curves/`: external/reference curves used by dissociation
  plots.
- `semilocal_channel/`: protocol for the N2 noPT2+HFX implicit-SCF
  semi-local channel-composition tests.

## Naming

Directories follow:

`<category>/<state>/<system>/<mode>/`

where `<mode>` records the SCF treatment, HFX/PT2 setting, and training
grid when needed, for example `implicit_scf_hfx_pt2_train7`.

Each leaf directory uses standardized filenames:

- `dense_curve.csv`, `predictions.csv`, or `visualization_data.csv`
  for plot-ready tables.
- `training_points.csv` for training geometries.
- `metrics.json` and `summary.json` for provenance and numerical
  summaries.

## Manifest

`MANIFEST.csv` and `MANIFEST.json` record source path, destination path,
row count, file size, SHA256 checksum, category, system, state, task,
variant, and role.

Copied/written entries: {copied}
Missing entries: {len(missing)}
"""
    if missing:
        readme += "\n## Missing Sources\n\n"
        for row in missing:
            readme += f"- `{row['source_path']}`\n"
    (OUT_ROOT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = [copy_item(item) for item in (*ITEMS, *LOSS_ITEMS)]
    rows.extend(write_semilocal_protocol())
    write_manifest(rows)
    write_readme(rows)

    copied = sum(1 for row in rows if row["status"] in {"copied", "written"})
    missing = [row for row in rows if row["status"] == "missing"]
    print(f"plot_data_root={OUT_ROOT}")
    print(f"entries={len(rows)} copied_or_written={copied} missing={len(missing)}")
    if missing:
        for row in missing:
            print(f"MISSING {row['source_path']}")


if __name__ == "__main__":
    main()
