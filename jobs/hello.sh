#!/usr/bin/env bash
# Fast diagnostic (~1-2 min): runner, clone, pip and GPU work. Downloads nothing beyond
# the repo and requirements (models/datasets must come from attached Kaggle datasets).
set -uo pipefail
report="$OUT_DIR/hello.txt"
{
  echo "== GPU"
  nvidia-smi --query-gpu=name,memory.total --format=csv || echo "NO GPU"
  python -c "import torch; print('torch', torch.__version__, 'cuda available', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"
  python -c "import torch; x = torch.randn(4096, 4096, device='cuda', dtype=torch.float16); print('fp16 matmul ok', float((x @ x).float().abs().mean()))"
  echo "== versions"
  python -c "import sys, transformers; print('python', sys.version.split()[0], 'transformers', transformers.__version__)"
  echo "== attached inputs"
  ls /kaggle/input 2>/dev/null || echo "(none)"
} 2>&1 | tee "$report"
