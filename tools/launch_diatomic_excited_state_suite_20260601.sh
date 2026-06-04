#!/usr/bin/env bash
set -euo pipefail

ROOT="${TDGRADDFT_ROOT:-/home/yjiao/TD-GradDFT}"
ENV_NAME="${TDGRADDFT_ENV_NAME:-jax_scf}"
RUN_ID="${TDGRADDFT_RUN_ID:-diatomic_excited_suite_$(date +%Y%m%d_%H%M%S)}"
SUITE_ROOT="${TDGRADDFT_SUITE_ROOT:-$ROOT/outputs/$RUN_ID}"
SYSTEM="${TDGRADDFT_SYSTEM:-both}"                # h2 | h2plus | both
OBJECTIVE="${TDGRADDFT_OBJECTIVE:-joint}"         # e0_only | s1_only | joint | auto
H2PLUS_REFERENCE_EXCITED_METHOD="${TDGRADDFT_H2PLUS_REFERENCE_EXCITED_METHOD:-orbital}"
BASIS="${TDGRADDFT_BASIS:-def2-svp}"
XC="${TDGRADDFT_XC:-b3lyp}"
R_MIN="${TDGRADDFT_R_MIN:-0.4}"
R_MAX="${TDGRADDFT_R_MAX:-6.0}"
TRAIN_POINTS="${TDGRADDFT_TRAIN_POINTS:-5}"
DENSE_POINTS="${TDGRADDFT_DENSE_POINTS:-100}"
STEPS="${TDGRADDFT_STEPS:-2000}"
LR="${TDGRADDFT_LR:-5e-5}"
LR_DECAY_EVERY="${TDGRADDFT_LR_DECAY_EVERY:-4000}"
LR_DECAY_FACTOR="${TDGRADDFT_LR_DECAY_FACTOR:-0.5}"
GRIDS_LEVEL="${TDGRADDFT_GRIDS_LEVEL:-2}"
INTEGRAL_BACKEND="${TDGRADDFT_INTEGRAL_BACKEND:-gpu}"
GPU_BINDING="${TDGRADDFT_GPU_BINDING:-}"
SKIP_INITIAL_EVAL="${TDGRADDFT_SKIP_INITIAL_EVAL:-0}"
SKIP_FINAL_EVALUATION="${TDGRADDFT_SKIP_FINAL_EVALUATION:-0}"
H2_INCLUDE_PT2_CHANNEL="${TDGRADDFT_H2_INCLUDE_PT2_CHANNEL:-1}"
H2_PT2_CHANNEL_MODE="${TDGRADDFT_H2_PT2_CHANNEL_MODE:-scaled_projected}"
H2_RESPONSE_PT2_MODE="${TDGRADDFT_H2_RESPONSE_PT2_MODE:-strict}"
H2PLUS_INCLUDE_PT2_CHANNEL="${TDGRADDFT_H2PLUS_INCLUDE_PT2_CHANNEL:-0}"
H2PLUS_PT2_CHANNEL_MODE="${TDGRADDFT_H2PLUS_PT2_CHANNEL_MODE:-scaled_projected}"
H2PLUS_RESPONSE_PT2_MODE="${TDGRADDFT_H2PLUS_RESPONSE_PT2_MODE:-strict}"

cd "$ROOT"
mkdir -p "$SUITE_ROOT" "$SUITE_ROOT/reference_cache"

export JAX_ENABLE_X64=1
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

if [ -n "$GPU_BINDING" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU_BINDING"
fi

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

run_h2() {
  local outdir="$SUITE_ROOT/h2_${OBJECTIVE}"
  local -a cmd=(
    conda run -n "$ENV_NAME" python -u tools/h2_s1_tda_train5_dense100_vs_fci.py
    --basis "$BASIS"
    --xc "$XC"
    --r-min "$R_MIN"
    --r-max "$R_MAX"
    --train-points "$TRAIN_POINTS"
    --dense-points "$DENSE_POINTS"
    --steps "$STEPS"
    --learning-rate "$LR"
    --lr-decay-every "$LR_DECAY_EVERY"
    --lr-decay-factor "$LR_DECAY_FACTOR"
    --training-mode self_consistent
    --objective "$OBJECTIVE"
    --grids-level "$GRIDS_LEVEL"
    --integral-backend "$INTEGRAL_BACKEND"
    --reference-scf-backend jax_rks
    --train-scf-convergence-metric energy
    --scf-gradient-mode impl
    --jit-train
    --jit-eval
    --reference-cache "$SUITE_ROOT/reference_cache/h2_s1_references.h5"
    --outdir "$outdir"
  )
  if bool_is_true "$H2_INCLUDE_PT2_CHANNEL"; then
    cmd+=(
      --include-pt2-channel
      --pt2-channel-mode "$H2_PT2_CHANNEL_MODE"
      --response-pt2-mode "$H2_RESPONSE_PT2_MODE"
    )
  else
    cmd+=(--no-include-pt2-channel)
  fi
  echo "[$(date -Is)] launching H2 objective=$OBJECTIVE outdir=$outdir"
  echo "[$(date -Is)] H2 response modes: HF=approx PT2=$(
    if bool_is_true "$H2_INCLUDE_PT2_CHANNEL"; then
      printf '%s (%s channel)' "$H2_RESPONSE_PT2_MODE" "$H2_PT2_CHANNEL_MODE"
    else
      printf 'disabled'
    fi
  )"
  "${cmd[@]}"
}

run_h2plus() {
  local outdir="$SUITE_ROOT/h2plus_${OBJECTIVE}"
  local -a cmd=(
    conda run -n "$ENV_NAME" python -u tools/h2plus_s1_tda_train5_dense100.py
    --basis "$BASIS"
    --xc "$XC"
    --r-min "$R_MIN"
    --r-max "$R_MAX"
    --train-points "$TRAIN_POINTS"
    --dense-points "$DENSE_POINTS"
    --steps "$STEPS"
    --learning-rate "$LR"
    --lr-decay-every "$LR_DECAY_EVERY"
    --lr-decay-factor "$LR_DECAY_FACTOR"
    --objective "$OBJECTIVE"
    --reference-excited-method "$H2PLUS_REFERENCE_EXCITED_METHOD"
    --grids-level "$GRIDS_LEVEL"
    --integral-backend "$INTEGRAL_BACKEND"
    --reference-scf-device cpu
    --train-scf-convergence-metric energy
    --scf-gradient-mode impl
    --jit-train
    --jit-eval
    --reference-cache "$SUITE_ROOT/reference_cache/h2plus_s1_references.h5"
    --outdir "$outdir"
  )
  if bool_is_true "$H2PLUS_INCLUDE_PT2_CHANNEL"; then
    cmd+=(
      --include-pt2-channel
      --pt2-channel-mode "$H2PLUS_PT2_CHANNEL_MODE"
      --response-pt2-mode "$H2PLUS_RESPONSE_PT2_MODE"
    )
  else
    cmd+=(--no-include-pt2-channel)
  fi
  if bool_is_true "$SKIP_INITIAL_EVAL"; then
    cmd+=(--skip-initial-eval)
  fi
  if bool_is_true "$SKIP_FINAL_EVALUATION"; then
    cmd+=(--skip-final-evaluation)
  fi
  echo "[$(date -Is)] launching H2+ objective=$OBJECTIVE outdir=$outdir"
  echo "[$(date -Is)] H2+ response modes: HF=approx PT2=$(
    if bool_is_true "$H2PLUS_INCLUDE_PT2_CHANNEL"; then
      printf '%s (%s channel)' "$H2PLUS_RESPONSE_PT2_MODE" "$H2PLUS_PT2_CHANNEL_MODE"
    else
      printf 'disabled'
    fi
  )"
  "${cmd[@]}"
}

echo "[$(date -Is)] suite_root=$SUITE_ROOT system=$SYSTEM objective=$OBJECTIVE"
echo "[$(date -Is)] launcher defaults: H2 HF=approx, H2 PT2=${H2_RESPONSE_PT2_MODE}, H2plus HF=approx, H2plus PT2=${H2PLUS_RESPONSE_PT2_MODE}"
show_runtime

case "$SYSTEM" in
  h2)
    run_h2
    ;;
  h2plus)
    run_h2plus
    ;;
  both)
    run_h2
    run_h2plus
    ;;
  *)
    echo "Unsupported TDGRADDFT_SYSTEM=$SYSTEM; expected h2, h2plus, or both." >&2
    exit 2
    ;;
esac

echo "[$(date -Is)] done suite_root=$SUITE_ROOT"
