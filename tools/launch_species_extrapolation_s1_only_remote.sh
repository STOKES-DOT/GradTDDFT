#!/usr/bin/env bash
set -euo pipefail

ROOT="${TDGRADDFT_ROOT:-/home/yjiao/TD-GradDFT}"
ENV_NAME="${TDGRADDFT_ENV_NAME:-jax_scf}"
RUN_ID="${TDGRADDFT_RUN_ID:-species_extrapolation_s1_only_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${TDGRADDFT_OUT_ROOT:-$ROOT/outputs/$RUN_ID}"
BASIS="${TDGRADDFT_BASIS:-def2-svp}"
XC="${TDGRADDFT_XC:-b3lyp}"
GRIDS_LEVEL="${TDGRADDFT_GRIDS_LEVEL:-2}"
STEPS="${TDGRADDFT_STEPS:-4000}"
LR="${TDGRADDFT_LR:-1e-3}"
LR_DECAY_EVERY="${TDGRADDFT_LR_DECAY_EVERY:-500}"
LR_DECAY_FACTOR="${TDGRADDFT_LR_DECAY_FACTOR:-0.5}"
TRAINING_MODE="${TDGRADDFT_TRAINING_MODE:-self_consistent}"
NROOTS="${TDGRADDFT_NROOTS:-3}"
SKIP_FINAL_EVALUATION="${TDGRADDFT_SKIP_FINAL_EVALUATION:-0}"

cd "$ROOT"
mkdir -p "$OUT_ROOT"

export JAX_ENABLE_X64=1
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

bool_is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

show_runtime() {
  conda run -n "$ENV_NAME" python - <<'PY'
import os
import jax
print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
print("jax_default_backend=", jax.default_backend())
print("jax_devices=", jax.devices())
PY
}

MOLECULE_REF="$OUT_ROOT/molecule_s1_references.csv"
ATOM_REF="$OUT_ROOT/atomic_s1_references.csv"

echo "[$(date -Is)] root=$ROOT env=$ENV_NAME out_root=$OUT_ROOT"
echo "[$(date -Is)] basis=$BASIS grids_level=$GRIDS_LEVEL xc=$XC steps=$STEPS"
show_runtime

conda run -n "$ENV_NAME" python -u tools/generate_closed_shell_eomee_s1_csv.py \
  --basis "$BASIS" \
  --nroots "$NROOTS" \
  --outcsv "$MOLECULE_REF" \
  --overwrite

conda run -n "$ENV_NAME" python -u tools/generate_atomic_species_eomee_s1_csv.py \
  --preset closed_shell_s1 \
  --basis "$BASIS" \
  --nroots "$NROOTS" \
  --outcsv "$ATOM_REF" \
  --overwrite

COMMON_TRAIN_ARGS=(
  --basis "$BASIS"
  --xc "$XC"
  --steps "$STEPS"
  --learning-rate "$LR"
  --lr-decay-every "$LR_DECAY_EVERY"
  --lr-decay-factor "$LR_DECAY_FACTOR"
  --training-mode "$TRAINING_MODE"
  --s1-use-tda
  --eval-use-tda
  --s1-weight 1.0
  --energy-mse-weight 0.0
  --energy-mae-weight 0.0
  --density-constraint-weight 0.0
  --grids-level "$GRIDS_LEVEL"
  --stream-train
)

if bool_is_true "$SKIP_FINAL_EVALUATION"; then
  COMMON_TRAIN_ARGS+=(--skip-final-evaluation)
fi

conda run -n "$ENV_NAME" python -u tools/closed_shell_s1_self_consistent_train.py \
  --reference-csv "$MOLECULE_REF" \
  "${COMMON_TRAIN_ARGS[@]}" \
  --outdir "$OUT_ROOT/molecule_s1_only"

conda run -n "$ENV_NAME" python -u tools/closed_shell_s1_self_consistent_train.py \
  --reference-csv "$ATOM_REF" \
  "${COMMON_TRAIN_ARGS[@]}" \
  --outdir "$OUT_ROOT/closed_shell_atom_s1_only"

echo "[$(date -Is)] done out_root=$OUT_ROOT"
