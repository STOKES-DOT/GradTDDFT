#!/usr/bin/env bash
set -euo pipefail

ROOT="${TDGRADDFT_ROOT:-/home/yjiao/TD-GradDFT}"
RUN_ID="${TDGRADDFT_RUN_ID:-diatomic_ground_suite_def2svp_grid2_2000ep_lrdecay4000_$(date +%Y%m%d_%H%M%S)}"
SUITE_ROOT="${TDGRADDFT_SUITE_ROOT:-$ROOT/outputs/$RUN_ID}"
ENV_NAME="${TDGRADDFT_ENV_NAME:-jax_scf}"
LR="${TDGRADDFT_LR:-1e-3}"
LR_DECAY_EVERY="${TDGRADDFT_LR_DECAY_EVERY:-4000}"
LR_DECAY_FACTOR="${TDGRADDFT_LR_DECAY_FACTOR:-0.5}"
R_MIN="${TDGRADDFT_R_MIN:-0.4}"
R_MAX="${TDGRADDFT_R_MAX:-6.0}"

cd "$ROOT"
mkdir -p "$SUITE_ROOT" "$SUITE_ROOT/reference_cache"

export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda,cpu
export JAX_PLATFORM_NAME=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

gpu_uuid() {
  local gpu="$1"
  local uuid
  uuid="$(nvidia-smi --id="$gpu" --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)"
  if [ -n "$uuid" ]; then
    printf '%s\n' "$uuid"
  else
    printf '%s\n' "$gpu"
  fi
}

gpu_memory_used_mib() {
  local gpu="$1"
  nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | head -n 1 \
    | awk '{print int($1)}'
}

wait_for_gpu() {
  local gpu="$1"
  local max_mib="${2:-2048}"
  while true; do
    local used
    used="$(gpu_memory_used_mib "$gpu" || true)"
    if [ -n "$used" ] && [ "$used" -lt "$max_mib" ]; then
      return 0
    fi
    echo "[$(date -Is)] waiting for GPU${gpu}; used=${used:-unknown} MiB threshold=${max_mib} MiB"
    sleep 120
  done
}

show_jax_device() {
  conda run -n "$ENV_NAME" python - <<'PY'
import os
import jax
print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
print("jax_default_backend=", jax.default_backend())
print("jax_devices=", jax.devices())
PY
}

run_h2_and_h2plus_on_gpu1() {
  wait_for_gpu 1 2048
  export CUDA_VISIBLE_DEVICES="$(gpu_uuid 1)"
  echo "[$(date -Is)] h2/h2plus worker using physical GPU1 as ${CUDA_VISIBLE_DEVICES}"
  show_jax_device

  conda run -n "$ENV_NAME" python -u tools/h2_self_consistent_ground_train5_dense100_vs_fci.py \
    --basis def2-svp \
    --xc b3lyp \
    --r-min "$R_MIN" \
    --r-max "$R_MAX" \
    --train-points 5 \
    --dense-points 100 \
    --steps 2000 \
    --learning-rate "$LR" \
    --lr-decay-every "$LR_DECAY_EVERY" \
    --lr-decay-factor "$LR_DECAY_FACTOR" \
    --training-mode self_consistent \
    --energy-mse-weight 1.0 \
    --energy-mae-weight 1.0 \
    --density-constraint-weight 1.0 \
    --grids-level 2 \
    --integral-backend gpu \
    --reference-scf-backend jax_rks \
    --train-scf-convergence-metric energy \
    --scf-gradient-mode impl \
    --no-include-pt2-channel \
    --excited-nstates 0 \
    --jit-train \
    --jit-eval \
    --outdir "$SUITE_ROOT/h2_neutral_ground"

  conda run -n "$ENV_NAME" python -u tools/h2plus_fci_ground_train5_dense100.py \
    --basis def2-svp \
    --xc b3lyp \
    --r-min "$R_MIN" \
    --r-max "$R_MAX" \
    --train-points 5 \
    --dense-points 100 \
    --steps 2000 \
    --learning-rate "$LR" \
    --lr-decay-every "$LR_DECAY_EVERY" \
    --lr-decay-factor "$LR_DECAY_FACTOR" \
    --energy-mse-weight 1.0 \
    --energy-mae-weight 1.0 \
    --density-constraint-weight 1.0 \
    --grids-level 2 \
    --integral-backend gpu \
    --reference-scf-device gpu \
    --train-scf-convergence-metric energy \
    --scf-gradient-mode impl \
    --jit-train \
    --jit-eval \
    --reference-cache "$SUITE_ROOT/reference_cache/h2plus_ground_references.h5" \
    --outdir "$SUITE_ROOT/h2plus_ion_ground"
}

run_n2_on_gpu0_when_free() {
  wait_for_gpu 0 2048
  export CUDA_VISIBLE_DEVICES="$(gpu_uuid 0)"
  echo "[$(date -Is)] n2 worker using physical GPU0 as ${CUDA_VISIBLE_DEVICES}"
  show_jax_device

  conda run -n "$ENV_NAME" python -u tools/n2_ccsdt_ground_train5.py \
    --basis def2-svp \
    --xc b3lyp \
    --r-min "$R_MIN" \
    --r-max "$R_MAX" \
    --train-points 5 \
    --steps 2000 \
    --learning-rate "$LR" \
    --lr-decay-every "$LR_DECAY_EVERY" \
    --lr-decay-factor "$LR_DECAY_FACTOR" \
    --energy-mse-weight 1.0 \
    --energy-mae-weight 1.0 \
    --density-constraint-weight 0.0 \
    --density-matrix-constraint-weight 1.0 \
    --grids-level 2 \
    --reference-scf-device gpu \
    --reference-method ccsd_t \
    --train-scf-convergence-metric energy \
    --scf-gradient-mode impl \
    --jit-train \
    --jit-eval \
    --log-every 20 \
    --reference-cache "$SUITE_ROOT/reference_cache/n2_ground_references.h5" \
    --outdir "$SUITE_ROOT/n2_neutral_ground"
}

echo "[$(date -Is)] suite_root=$SUITE_ROOT"
echo "[$(date -Is)] R_MIN=$R_MIN R_MAX=$R_MAX LR=$LR LR_DECAY_EVERY=$LR_DECAY_EVERY LR_DECAY_FACTOR=$LR_DECAY_FACTOR"

run_h2_and_h2plus_on_gpu1 > "$SUITE_ROOT/h2_h2plus_worker.log" 2>&1 &
pid_h="$!"
run_n2_on_gpu0_when_free > "$SUITE_ROOT/n2_worker.log" 2>&1 &
pid_n="$!"

status=0
wait "$pid_h" || status=1
wait "$pid_n" || status=1

if [ "$status" -eq 0 ]; then
  conda run -n "$ENV_NAME" python -u tools/plot_diatomic_training_suite.py --suite-root "$SUITE_ROOT" \
    > "$SUITE_ROOT/uniform_plot.log" 2>&1 || status=1
fi

echo "[$(date -Is)] done status=$status suite_root=$SUITE_ROOT"
exit "$status"
