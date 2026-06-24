#!/usr/bin/env bash
set -uo pipefail

REMOTE_REPO=${REMOTE_REPO:-/home/yjiao/TD-GradDFT}
CONDA_BIN=${CONDA_BIN:-/home/yjiao/opt/miniconda3/bin/conda}
GPU_ID=${GPU_ID:-3}
WAIT_FOR_GPU_FREE=${WAIT_FOR_GPU_FREE:-1}
GPU_WAIT_INTERVAL_SECONDS=${GPU_WAIT_INTERVAL_SECONDS:-120}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
QUEUE_ROOT=${QUEUE_ROOT:-outputs/n2_s1total_ladder_hfx_nopt2_def2tzvp_grid2_train7_dense35_${STAMP}}
STATUS_FILE="${REMOTE_REPO}/${QUEUE_ROOT}/queue_status.tsv"

if [[ "${GPU_ID}" == "2" ]]; then
  echo "Refusing to run on GPU2." >&2
  exit 2
fi

mkdir -p "${REMOTE_REPO}/${QUEUE_ROOT}"
printf "case\tstatus\tstart_time\tend_time\toutdir\n" > "${STATUS_FILE}"

wait_for_gpu_free() {
  if [[ "${WAIT_FOR_GPU_FREE}" != "1" ]]; then
    return 0
  fi
  while true; do
    local pids
    pids=$(
      nvidia-smi -i "${GPU_ID}" \
        --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk 'NF && $1 != "No" {print $1}' \
        | paste -sd, -
    )
    if [[ -z "${pids}" ]]; then
      echo "[queue] GPU${GPU_ID} is free at $(date -Is)."
      return 0
    fi
    echo "[queue] waiting for GPU${GPU_ID}; active compute pids: ${pids}; next check in ${GPU_WAIT_INTERVAL_SECONDS}s."
    sleep "${GPU_WAIT_INTERVAL_SECONDS}"
  done
}

run_case() {
  local label="$1"
  local xc="$2"
  local semilocal="$3"
  local outdir="${QUEUE_ROOT}/${label}"
  local cache="outputs/reference_cache/n2_s1total_${label}_hfx_nopt2_h128_def2tzvp_grid2_train7_dense35_${STAMP}.h5"
  local start_time
  local end_time
  start_time=$(date -Is)
  mkdir -p "${REMOTE_REPO}/${outdir}"
  printf "%s\trunning\t%s\t\t%s\n" "${label}" "${start_time}" "${outdir}" >> "${STATUS_FILE}"

  read -r -a semilocal_args <<< "${semilocal}"

  cd "${REMOTE_REPO}" || exit 2
  echo "[${label}] start ${start_time}"
  echo "[${label}] xc=${xc} semilocal=${semilocal} gpu=${GPU_ID}"

  if CUDA_VISIBLE_DEVICES="${GPU_ID}" \
     XLA_PYTHON_CLIENT_PREALLOCATE=false \
     XLA_PYTHON_CLIENT_ALLOCATOR=platform \
     "${CONDA_BIN}" run -n jax_scf python -u tools/h2_s1_tda_train5_dense100_vs_fci.py \
       --system-label N2 \
       --atom1 N \
       --atom2 N \
       --charge 0 \
       --spin 0 \
       --basis def2-tzvp \
       --xc "${xc}" \
       --r-min 0.8 \
       --r-max 3.0 \
       --train-r-values 0.8 1.1 1.6 2.0 2.2 2.5 3.0 \
       --dense-points 35 \
       --steps 2000 \
       --learning-rate 1e-4 \
       --lr-decay-every 400 \
       --lr-decay-factor 0.5 \
       --training-mode self_consistent \
       --objective s1_only \
       --grids-level 2 \
       --integral-backend cpu \
       --reference-scf-backend pyscf \
       --train-scf-convergence-metric energy \
       --scf-gradient-mode impl \
       --stream-train \
       --skip-initial-eval \
       --defer-dense-eval \
       --include-hfx-channel \
       --response-hf-mode approx \
       --no-include-pt2-channel \
       --hidden-dims 128 128 128 128 \
       --semilocal-xc "${semilocal_args[@]}" \
       --s1-use-tda \
       --eval-use-tda \
       --excited-nstates 1 \
       --equilibrium-spectrum-nstates 1 \
       --skip-equilibrium-spectrum \
       --skip-eval-oscillator-strengths \
       --external-s1-total-csv benchmark/reference_curves/n2_hammami_2026_s0_s1_a1pig_35pt_groundgrid.csv \
       --external-s1-total-column hammami_s1_energy_hartree \
       --external-r-column r_angstrom \
       --external-reference-label Hammami-large-CAS-a1Pi_g \
       --reference-cache "${cache}" \
       --outdir "${outdir}" \
       2>&1 | tee "${REMOTE_REPO}/${outdir}/run.log"; then
    end_time=$(date -Is)
    printf "%s\tcompleted\t%s\t%s\t%s\n" "${label}" "${start_time}" "${end_time}" "${outdir}" >> "${STATUS_FILE}"
    echo "[${label}] completed ${end_time}"
  else
    end_time=$(date -Is)
    printf "%s\tfailed\t%s\t%s\t%s\n" "${label}" "${start_time}" "${end_time}" "${outdir}" >> "${STATUS_FILE}"
    echo "[${label}] failed ${end_time}" >&2
  fi
}

wait_for_gpu_free

run_case "lda_vwn_rpa" "lda_x,lda_c_vwn_rpa" "lda_x lda_c_vwn_rpa"
run_case "gga_pbe" "pbe" "gga_x_pbe gga_c_pbe"
run_case "mgga_r2scan" "r2scan" "mgga_x_r2scan mgga_c_r2scan"

echo "queue_root=${QUEUE_ROOT}" | tee "${REMOTE_REPO}/${QUEUE_ROOT}/queue_done.txt"
