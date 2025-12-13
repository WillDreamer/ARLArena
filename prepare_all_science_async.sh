# #!/usr/bin/env bash
# set -Eeuo pipefail
# IFS=$'\n\t'


# log() { printf "\n\033[1;32m[+] %s\033[0m\n" "$*"; }
# warn() { printf "\033[1;33m[!] %s\033[0m\n" "$*"; }
# die() { printf "\033[1;31m[x] %s\033[0m\n" "$*"; exit 1; }
# trap 'die "Error is in Line $LINENO (exit=$?)。"' ERR

# as_root() {
#   if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
#     sudo -H bash -lc "$*"
#   else
#     bash -lc "$*"
#   fi
# }


# log "Run setup game environment "
# # CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || true)}"
# # if [[ -z "${CONDA_BASE}" ]]; then
# #   for p in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/anaconda3"; do
# #     [[ -d "$p" ]] && CONDA_BASE="$p" && break
# #   done
# # fi
# # if [[ -z "${CONDA_BASE}" ]]; then
# #   echo "Conda not found, please confirm it is installed (miniconda or anaconda)." >&2
# #   exit 1
# # fi
# CONDA_BASE=/data1/xw27/miniconda3
# source "${CONDA_BASE}/etc/profile.d/conda.sh"

# conda create -n agentrl_science_async python==3.12 -y
conda activate agentrl_science_async
# python3 -m pip install uv

pip3 install -e .

python3 -m uv pip install transformers==4.51.3
python3 -m uv pip install matplotlib
python3 -m uv pip install gym==0.26.2
python3 -m uv pip install ray==2.45.0
conda install -c conda-forge fire -y
conda install -c conda-forge "numpy<2.0" -y
python3 -m uv pip install "qwen_vl_utils>=0.0.14"
python3 -m uv pip install "qwen_omni_utils"
python3 -m uv pip install func_timeout
python3 -m uv pip install torch==2.8.0
python3 -m uv pip install torchvision==0.21.0
python3 -m uv pip install flash-attn==2.8.1 --no-build-isolation
python3 -m uv pip install "vllm==0.10.0"
python3 -m uv pip install math-verify==0.7.0


python -m pip uninstall -y opentelemetry-sdk opentelemetry-api opentelemetry-exporter-otlp opentelemetry-exporter-otlp-proto-grpc opentelemetry-exporter-otlp-proto-http || true
python -m pip install "opentelemetry-api>=1.30.0" "opentelemetry-sdk>=1.30.0" "opentelemetry-exporter-otlp>=1.30.0"

if command -v conda >/dev/null 2>&1; then
  conda activate agentrl_science_async || true
fi

log "hf auth whoami"
if command -v hf >/dev/null 2>&1; then
  hf auth whoami || die "whoami failed"
else
  huggingface-cli whoami || die "whoami failed"
fi

