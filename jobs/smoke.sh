#!/usr/bin/env bash
# Environment check for Kaggle 2xT4: GPUs, library versions, and an fp16 VQA-RAD forward pass
# with a CRAFT model, clean vs. misleading-text prompt. Needs mounts:
#   --model qwen-lm/qwen-3-vl/transformers/4b-instruct/1 --dataset phandai/vqa-rad-craft
set -euo pipefail
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch, transformers; print('torch', torch.__version__, 'gpus', torch.cuda.device_count(), 'transformers', transformers.__version__)"
python -m medvqa.smoke --model-variation "${MODEL_VARIATION:-4b-instruct}" --n "${N:-8}" --out "$OUT_DIR/smoke.json"
