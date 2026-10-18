#!/usr/bin/env bash
# Environment check for Kaggle 2xT4: GPUs, library versions, and one fp16 VQA-RAD
# forward pass with a CRAFT model, clean vs. misleading-text prompt.
set -euo pipefail
nvidia-smi
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count()); print('transformers', transformers.__version__)"
python -m medvqa.smoke --model "${MODEL:-Qwen/Qwen3-VL-4B-Instruct}" --n "${N:-8}" --out "$OUT_DIR/smoke.json"
