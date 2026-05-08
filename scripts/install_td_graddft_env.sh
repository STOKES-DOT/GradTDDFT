#!/usr/bin/env bash
set -Eeuo pipefail

# Install the Python/CUDA stack needed to run TD-GradDFT.
#
# Default target is the remote jax_scf conda environment. The script is safe to
# rerun: it creates the env if missing, upgrades the package set, and installs
# this repository in editable mode.
#
# Common usage on the remote server:
#   bash scripts/install_td_graddft_env.sh
#
# Useful overrides:
#   ENV_NAME=grad_dft bash scripts/install_td_graddft_env.sh
#   ALLOW_RECREATE_ENV=1 bash scripts/install_td_graddft_env.sh
#   CUDA_MODE=cpu bash scripts/install_td_graddft_env.sh
#   CUDA_MODE=cuda12-local bash scripts/install_td_graddft_env.sh
#   JAX_INSTALL_SPEC='jax[cuda12]==0.6.2' bash scripts/install_td_graddft_env.sh
#   INSTALL_GPU4PYSCF=1 bash scripts/install_td_graddft_env.sh

ENV_NAME="${ENV_NAME:-jax_scf}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA_MODE="${CUDA_MODE:-auto}"
INSTALL_JAX_XC="${INSTALL_JAX_XC:-0}"
INSTALL_GPU4PYSCF="${INSTALL_GPU4PYSCF:-0}"
ALLOW_RECREATE_ENV="${ALLOW_RECREATE_ENV:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

log() {
  printf '[td-graddft-env] %s\n' "$*"
}

warn() {
  printf '[td-graddft-env] WARNING: %s\n' "$*" >&2
}

die() {
  printf '[td-graddft-env] ERROR: %s\n' "$*" >&2
  exit 1
}

find_conda_sh() {
  if command -v conda >/dev/null 2>&1; then
    local base
    base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${base}" && -f "${base}/etc/profile.d/conda.sh" ]]; then
      printf '%s\n' "${base}/etc/profile.d/conda.sh"
      return 0
    fi
  fi

  local candidate
  for candidate in \
    "${HOME}/opt/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh" \
    "/opt/miniconda3/etc/profile.d/conda.sh"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

detect_cuda_major() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local smi
    smi="$(nvidia-smi 2>&1 || true)"
    if grep -q 'Unable to determine the device handle' <<<"${smi}"; then
      warn "nvidia-smi reports at least one unhealthy GPU. Use CUDA_VISIBLE_DEVICES to select a healthy GPU before long runs."
    fi
    local cuda_version
    cuda_version="$(sed -nE 's/.*CUDA Version: ([0-9]+)\..*/\1/p' <<<"${smi}" | head -1)"
    if [[ -n "${cuda_version}" ]]; then
      printf '%s\n' "${cuda_version}"
      return 0
    fi
  fi

  if [[ -f /usr/local/cuda/version.json ]]; then
    python - <<'PY' 2>/dev/null && return 0 || true
import json
with open("/usr/local/cuda/version.json", "r", encoding="utf-8") as handle:
    data = json.load(handle)
version = str(data.get("cuda", {}).get("version", ""))
print(version.split(".", 1)[0])
PY
  fi

  if [[ -f /usr/local/cuda/version.txt ]]; then
    sed -nE 's/.*CUDA Version ([0-9]+)\..*/\1/p' /usr/local/cuda/version.txt | head -1
    return 0
  fi

  return 1
}

resolve_jax_spec() {
  if [[ -n "${JAX_INSTALL_SPEC:-}" ]]; then
    printf '%s\n' "${JAX_INSTALL_SPEC}"
    return 0
  fi

  case "${CUDA_MODE}" in
    cpu)
      printf '%s\n' 'jax[cpu]'
      ;;
    cuda12)
      printf '%s\n' 'jax[cuda12]'
      ;;
    cuda13)
      printf '%s\n' 'jax[cuda13]'
      ;;
    cuda12-local)
      printf '%s\n' 'jax[cuda12-local]'
      ;;
    auto)
      local major
      major="$(detect_cuda_major || true)"
      case "${major}" in
        12)
          printf '%s\n' 'jax[cuda12]'
          ;;
        13)
          printf '%s\n' 'jax[cuda13]'
          ;;
        *)
          warn "No usable NVIDIA CUDA runtime detected; installing CPU JAX."
          printf '%s\n' 'jax[cpu]'
          ;;
      esac
      ;;
    *)
      die "Unsupported CUDA_MODE='${CUDA_MODE}'. Use auto, cpu, cuda12, cuda13, or cuda12-local."
      ;;
  esac
}

configure_cuda_library_paths() {
  case "${JAX_SPEC}" in
    *cuda*-local*)
      local cuda_home="${CUDA_HOME:-/usr/local/cuda}"
      if [[ -d "${cuda_home}" ]]; then
        export CUDA_HOME="${cuda_home}"
        export PATH="${CUDA_HOME}/bin:${PATH}"
        export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
        log "Using local CUDA toolkit from CUDA_HOME=${CUDA_HOME}."
      else
        warn "CUDA_MODE requests local CUDA, but CUDA_HOME='${cuda_home}' does not exist."
      fi
      ;;
    *cuda*)
      if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
        warn "LD_LIBRARY_PATH is set. For jax[cuda12]/jax[cuda13] pip CUDA wheels, this can override JAX's bundled CUDA libraries."
      fi
      ;;
  esac
}

CONDA_SH="$(find_conda_sh)" || die "Cannot find conda.sh. Set PATH or install Miniconda/Anaconda first."
log "Using conda init script: ${CONDA_SH}"
# shellcheck disable=SC1090
source "${CONDA_SH}"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  log "Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION}."
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip setuptools wheel packaging
else
  log "Using existing conda env '${ENV_NAME}'."
fi

conda activate "${ENV_NAME}"
CURRENT_PYTHON_MINOR="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
if [[ "${CURRENT_PYTHON_MINOR}" != "${PYTHON_VERSION}" ]]; then
  if [[ "${ALLOW_RECREATE_ENV}" == "1" ]]; then
    log "Recreating env '${ENV_NAME}' because Python is ${CURRENT_PYTHON_MINOR}, expected ${PYTHON_VERSION}."
    conda deactivate
    conda env remove -y -n "${ENV_NAME}"
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip setuptools wheel packaging
    conda activate "${ENV_NAME}"
  else
    die "Env '${ENV_NAME}' uses Python ${CURRENT_PYTHON_MINOR}, but this stack expects Python ${PYTHON_VERSION}. Re-run with ALLOW_RECREATE_ENV=1 to recreate the empty env, or set PYTHON_VERSION=${CURRENT_PYTHON_MINOR} and downgrade Flax/JAX constraints manually."
  fi
fi
log "Python: $(python -V 2>&1)"
python -m pip install --upgrade pip setuptools wheel packaging

JAX_SPEC="$(resolve_jax_spec)"
configure_cuda_library_paths
log "Installing JAX spec: ${JAX_SPEC}"
python -m pip install --upgrade "${JAX_SPEC}"

FLAX_SPEC="$(python - <<'PY'
import sys
if sys.version_info < (3, 11):
    print("flax>=0.10.7,<0.11")
else:
    print("flax>=0.12.0")
PY
)"

log "Installing TD-GradDFT runtime packages."
python -m pip install --upgrade \
  "numpy>=1.26" \
  "scipy>=1.11" \
  "${FLAX_SPEC}" \
  "optax>=0.2.6" \
  "jaxtyping>=0.2.36" \
  "chex>=0.1.86" \
  "orbax-checkpoint>=0.6.4" \
  "pyscf>=2.6.0" \
  "matplotlib>=3.8" \
  "pytest>=8.0" \
  "basis-set-exchange>=0.10" \
  "h5py>=3.10" \
  "pandas>=2.0" \
  "tqdm>=4.66" \
  "msgpack>=1.0"

if [[ "${INSTALL_JAX_XC}" == "1" ]]; then
  log "Installing optional upstream jax-xc."
  python -m pip install --upgrade "jax-xc>=0.0.12"
else
  log "Skipping optional jax-xc. Set INSTALL_JAX_XC=1 to install it."
fi

if [[ "${INSTALL_GPU4PYSCF}" == "1" ]]; then
  case "${JAX_SPEC}" in
    *cuda12*)
      log "Installing optional GPU4PySCF/CuPy CUDA-12 packages for benchmark tools."
      python -m pip install --upgrade cupy-cuda12x gpu4pyscf-cuda12x
      ;;
    *cuda13*)
      warn "Skipping GPU4PySCF: this script only knows the CUDA-12 GPU4PySCF wheels."
      ;;
    *)
      log "Skipping GPU4PySCF because selected JAX spec is not a CUDA build."
      ;;
  esac
else
  log "Skipping optional GPU4PySCF. Set INSTALL_GPU4PYSCF=1 to install benchmark-only GPU4PySCF/CuPy packages."
fi

if [[ -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
  log "Installing repository in editable mode: ${PROJECT_ROOT}"
  python -m pip install --editable "${PROJECT_ROOT}"
else
  log "PROJECT_ROOT has no pyproject.toml; skipping editable install: ${PROJECT_ROOT}"
fi

log "Running import/device sanity check."
python - <<'PY'
import importlib.metadata as metadata
import os
import sys

packages = [
    "jax",
    "jaxlib",
    "flax",
    "optax",
    "pyscf",
    "numpy",
    "scipy",
    "matplotlib",
    "pytest",
    "jaxtyping",
    "chex",
    "orbax-checkpoint",
]
print("python", sys.version.replace("\n", " "))
for package in packages:
    try:
        print(f"{package}=={metadata.version(package)}")
    except metadata.PackageNotFoundError:
        print(f"{package} MISSING")

try:
    import jax

    print("jax_default_backend", jax.default_backend())
    print("jax_devices", jax.devices())
except Exception as exc:
    print("jax_probe_error", repr(exc))
    raise

try:
    import pyscf
    from pyscf.dft import libxc

    print("pyscf", pyscf.__version__)
    print("libxc_available", bool(libxc))
except Exception as exc:
    print("pyscf_probe_error", repr(exc))
    raise

try:
    import td_graddft

    print("td_graddft_import", td_graddft.__name__)
except Exception as exc:
    print("td_graddft_probe_error", repr(exc))
    raise

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    print("note", "CUDA_VISIBLE_DEVICES is unset. On multi-GPU servers, set it to a healthy GPU id before long runs.")
PY

log "Done."
