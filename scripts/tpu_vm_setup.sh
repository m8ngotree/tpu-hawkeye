#!/usr/bin/env bash
# Run once on a fresh Cloud TPU VM: fetch the repo and install dependencies.
#   bash <(curl -s https://raw.githubusercontent.com/m8ngotree/tpu-hawkeye/main/scripts/tpu_vm_setup.sh)
set -euo pipefail

cd "$HOME"
if [ -d tpu-hawkeye ]; then
  git -C tpu-hawkeye pull --recurse-submodules
else
  git clone --recurse-submodules https://github.com/m8ngotree/tpu-hawkeye.git
fi
cd tpu-hawkeye

python3 -m venv .venv-tpu || { sudo apt-get update -y && sudo apt-get install -y python3-venv && python3 -m venv .venv-tpu; }
source .venv-tpu/bin/activate
pip install -U pip
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install numpy openai

python - <<'PY'
import jax
print(jax.devices())
PY
echo "Setup done. In each new shell: cd ~/tpu-hawkeye && source .venv-tpu/bin/activate"
