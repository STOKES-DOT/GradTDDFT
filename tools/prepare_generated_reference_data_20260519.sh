#!/usr/bin/env bash
set -euo pipefail

cd /home/yjiao/TD-GradDFT

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=

DATA_ROOT="${DATA_ROOT:-datasets/reference_data_20260519/generated}"
mkdir -p "${DATA_ROOT}"

echo "[generated-ref] start=$(date --iso-8601=seconds)"
echo "[generated-ref] cwd=$(pwd)"
echo "[generated-ref] data_root=${DATA_ROOT}"
echo "[generated-ref] threads OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS}"

run_curve() {
  local name="$1"
  shift
  local outdir="${DATA_ROOT}/${name}"

  if [[ -s "${outdir}/points.csv" && -s "${outdir}/states.csv" && -s "${outdir}/manifest.json" ]]; then
    echo "[generated-ref] skip existing ${name}"
    return 0
  fi

  echo "[generated-ref] run ${name}"
  conda run -n jax_scf python -u tools/build_diatomic_reference_curves.py "$@" --outdir "${outdir}"
}

run_curve graddft_h2_h2plus_fci_ccpvqz_80pt \
  --systems H2 H2+ \
  --basis cc-pVQZ \
  --method fci \
  --points 80 \
  --nroots 4 \
  --max-fci-determinants 500000

run_curve excited_small_fci_631pgstar_50pt \
  --systems H2 H2+ LiH \
  --basis '6-31+g*' \
  --method fci \
  --points 50 \
  --nroots 5 \
  --max-fci-determinants 500000

run_curve excited_first_wave_tda_b3lyp_631pgstar_60pt \
  --systems BH HF F2 N2 C2 \
  --basis '6-31+g*' \
  --method tda \
  --xc b3lyp \
  --points 60 \
  --nroots 5

echo "[generated-ref] outputs"
find "${DATA_ROOT}" -maxdepth 2 -type f \( -name 'points.csv' -o -name 'states.csv' -o -name 'manifest.json' \) -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' | sort
echo "[generated-ref] done=$(date --iso-8601=seconds)"
