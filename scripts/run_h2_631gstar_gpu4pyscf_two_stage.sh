#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_h2_631gstar_gpu4pyscf_two_stage.sh no_pt2
  scripts/run_h2_631gstar_gpu4pyscf_two_stage.sh pt2_approx
  scripts/run_h2_631gstar_gpu4pyscf_two_stage.sh pt2_strict

Environment:
  TDGRADDFT_GPU          Physical GPU id to use. If unset, CUDA_VISIBLE_DEVICES is
                         honored; if both are unset, the script waits for a free GPU.
  TDGRADDFT_EXCLUDE_GPUS Comma-separated physical GPU ids excluded from auto-pick.
                         Defaults to 2.
  TDGRADDFT_ROOT         Repository root on the training server.
                         Defaults to /home/yjiao/TD-GradDFT.
  TDGRADDFT_OUTDIR       Full output root. Defaults to outputs/h2_s1_...
  TDGRADDFT_RUN_TAG      Suffix tag for the default output root.
  TDGRADDFT_CONDA_ENV    Conda environment. Defaults to jax_scf.
  TDGRADDFT_STAGE1_STEPS Stage 1 optimization steps. Defaults to 2000.
  TDGRADDFT_STAGE2_STEPS Stage 2 optimization steps. Defaults to 500.
EOF
}

variant="${1:-${TDGRADDFT_VARIANT:-}}"
case "${variant}" in
  no_pt2)
    run_variant="no_pt2"
    stage1_name="stage1_ground_gpu4pyscf_no_pt2"
    stage2_name="stage2_s1_strict_khh_no_pt2"
    stage1_pt2_args=(--no-include-pt2-channel)
    stage2_pt2_args=(--no-include-pt2-channel)
    ;;
  pt2|pt2_approx)
    run_variant="pt2_approx"
    stage1_name="stage1_ground_gpu4pyscf_pt2"
    stage2_name="stage2_s1_pt2_approx"
    stage1_pt2_args=(--include-pt2-channel)
    stage2_pt2_args=(
      --include-pt2-channel
      --pt2-channel-mode scaled_projected
      --response-pt2-mode approx
    )
    ;;
  pt2_strict|strict_pt2|strict_response_pt2|pt2_strict)
    run_variant="pt2_strict"
    stage1_name="stage1_ground_gpu4pyscf_pt2"
    stage2_name="stage2_s1_pt2_strict"
    stage1_pt2_args=(--include-pt2-channel)
    stage2_pt2_args=(
      --include-pt2-channel
      --pt2-channel-mode scaled_projected
      --response-pt2-mode strict
    )
    ;;
  -h|--help|"")
    usage
    exit 0
    ;;
  *)
    echo "Unsupported variant: ${variant}" >&2
    usage >&2
    exit 2
    ;;
esac

root_dir="${TDGRADDFT_ROOT:-/home/yjiao/TD-GradDFT}"
cd "${root_dir}"

timestamp="$(date +%Y%m%d_%H%M%S)"
run_tag="${TDGRADDFT_RUN_TAG:-${timestamp}_stage1lr1e3_d400_stage2lr1e5}"
out_root="${TDGRADDFT_OUTDIR:-outputs/h2_s1_631pgstar_${run_variant}_gpu4pyscf_${run_tag}}"
stage1_out="${out_root}/${stage1_name}"
stage2_out="${out_root}/${stage2_name}"

mkdir -p "${stage1_out}" "${stage2_out}"
: > "${out_root}/train.log"
cp "$0" "${out_root}/run.sh" 2>/dev/null || true
exec > >(tee -a "${out_root}/train.log") 2>&1

exclude_gpus="${TDGRADDFT_EXCLUDE_GPUS:-2}"
free_mem_threshold_mb="${TDGRADDFT_FREE_MEM_THRESHOLD_MB:-1024}"

is_excluded_gpu() {
  local candidate="$1"
  case ",${exclude_gpus}," in
    *",${candidate},"*) return 0 ;;
    *) return 1 ;;
  esac
}

pick_gpu() {
  local line idx mem
  while IFS= read -r line; do
    idx="${line%%,*}"
    mem="${line#*,}"
    idx="${idx//[[:space:]]/}"
    mem="${mem//[[:space:]]/}"
    if [[ "${idx}" =~ ^[0-9]+$ ]] \
      && [[ "${mem}" =~ ^[0-9]+$ ]] \
      && ! is_excluded_gpu "${idx}" \
      && (( mem < free_mem_threshold_mb )); then
      echo "${idx}"
      return 0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true)
}

gpu="${TDGRADDFT_GPU:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "${gpu}" ]]; then
  echo "[wait] start=$(date -Is)"
  echo "[wait] waiting for a free GPU; excluded=${exclude_gpus}; threshold_mb=${free_mem_threshold_mb}"
  while [[ -z "${gpu}" ]]; do
    gpu="$(pick_gpu || true)"
    if [[ -z "${gpu}" ]]; then
      nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
      sleep 120
    fi
  done
fi

export CUDA_VISIBLE_DEVICES="${gpu}"
export JAX_PLATFORMS=cuda,cpu
export JAX_PLATFORM_NAME=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

conda_env="${TDGRADDFT_CONDA_ENV:-jax_scf}"
python_cmd=(conda run -n "${conda_env}" python -u)
python_once=(conda run -n "${conda_env}" python)

echo "[run] start=$(date -Is)"
echo "[run] variant=${run_variant}"
echo "[run] root=${out_root}"
echo "[run] stage1_out=${stage1_out}"
echo "[run] stage2_out=${stage2_out}"
echo "[run] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
"${python_once[@]}" -c "import jax; print('jax_default_backend=', jax.default_backend()); print('jax_devices=', jax.devices())"

common_model_args=(
  --network-architecture graddft_residual
  --input-feature-mode canonical
  --hidden-dims 256 256 256 256 256 256
)

stage1_steps="${TDGRADDFT_STAGE1_STEPS:-2000}"
stage2_steps="${TDGRADDFT_STAGE2_STEPS:-500}"

echo "[stage1] steps=${stage1_steps} lr=1e-3 lr_decay_every=400 lr_decay_factor=0.1"
"${python_cmd[@]}" tools/h2_self_consistent_ground_train5_dense100_vs_fci.py \
  --basis '6-31+g*' \
  --xc b3lyp \
  --train-points 5 \
  --dense-points 100 \
  --steps "${stage1_steps}" \
  --learning-rate 1e-3 \
  --lr-decay-every 400 \
  --lr-decay-factor 0.1 \
  --training-mode self_consistent \
  "${stage1_pt2_args[@]}" \
  --reference-scf-backend gpu4pyscf_rks \
  --scf-runtime-forward-backend gpu4pyscf_rks \
  --implicit-response-backend gpu4pyscf_jk \
  --scf-gradient-mode impl \
  --no-jit-eval \
  --no-jit-train \
  --excited-nstates 1 \
  "${common_model_args[@]}" \
  --outdir "${stage1_out}"

echo "[stage2] steps=${stage2_steps} lr=1e-5 lr_decay_every=0"
"${python_cmd[@]}" tools/h2_s1_tda_train5_dense100_vs_fci.py \
  --basis '6-31+g*' \
  --xc b3lyp \
  --train-points 5 \
  --dense-points 100 \
  --steps "${stage2_steps}" \
  --learning-rate 1e-5 \
  --lr-decay-every 0 \
  --lr-decay-factor 0.1 \
  --training-mode fixed_density \
  "${stage2_pt2_args[@]}" \
  --reference-scf-backend gpu4pyscf_rks \
  --scf-runtime-forward-backend gpu4pyscf_rks \
  --implicit-response-backend gpu4pyscf_jk \
  --scf-gradient-mode impl \
  --jit-eval \
  --no-jit-train \
  --s1-use-tda \
  --eval-use-tda \
  --excited-nstates 1 \
  --equilibrium-spectrum-nstates 1 \
  "${common_model_args[@]}" \
  --fixed-density-reference-checkpoint "${stage1_out}/neural_xc_params.msgpack" \
  --init-checkpoint "${stage1_out}/neural_xc_params.msgpack" \
  --outdir "${stage2_out}"

echo "[run] done=$(date -Is)"
