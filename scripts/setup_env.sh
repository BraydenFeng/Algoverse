#!/usr/bin/env bash
# one-shot environment setup for Colab / Vast.ai / Lambda Labs
# usage: bash scripts/setup_env.sh
#
# expects HF_TOKEN and ANTHROPIC_API_KEY in env. on Colab, set via:
#   from google.colab import userdata
#   import os
#   os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
#   os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")

set -euo pipefail

echo "[setup] installing python deps"
pip install -q -r requirements.txt

if [[ -z "${HF_TOKEN:-}" ]]; then
	echo "[setup] WARN: HF_TOKEN not set; gated Gemma models will fail to download"
else
	echo "[setup] logging into HF"
	huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
fi

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
	echo "[setup] WARN: ANTHROPIC_API_KEY not set; classifier judge will fail"
fi

echo "[setup] GPU check"
python -c "import torch; print(f'cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"} bf16={torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}')"

echo "[setup] done"
