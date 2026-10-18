#!/usr/bin/env bash
# Fast diagnostic (~2-3 min): is the runner, GPU, internet and dataset access working,
# and how long does each step take? No model download.
set -uo pipefail
t() { date +%s; }
report="$OUT_DIR/hello.txt"
{
  echo "== GPU"; s=$(t)
  nvidia-smi --query-gpu=name,memory.total --format=csv || echo "NO GPU"
  python -c "import torch; print('torch', torch.__version__, 'cuda available', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"
  echo "gpu_check_seconds $(( $(t) - s ))"

  echo "== Internet: 100 MB download speed"; s=$(t)
  curl -s -o /dev/null -w "http_code %{http_code}, %{speed_download} bytes/s\n" \
    https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/resolve/main/model-00002-of-00002.safetensors \
    --range 0-104857599 || echo "DOWNLOAD FAILED"
  echo "download_seconds $(( $(t) - s ))"

  echo "== VQA-RAD dataset load"; s=$(t)
  python -c "
from datasets import load_dataset
ds = load_dataset('flaviagiammarino/vqa-rad')
print({k: len(v) for k, v in ds.items()})
" || echo "DATASET FAILED"
  echo "dataset_seconds $(( $(t) - s ))"

  echo "== transformers"
  python -c "import transformers; print('transformers', transformers.__version__)"
} 2>&1 | tee "$report"
