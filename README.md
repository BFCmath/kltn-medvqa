# kltn-medvqa

Thesis code: language bias and visual hallucination in medical VQA, starting from a replication of
[CRAFT](https://github.com/zhcz328/CRAFT) (attention heads that let misleading text override the image).

## Layout

| Path | What |
|---|---|
| `third_party/CRAFT/` | upstream CRAFT code as a git submodule (it has no license, so it is referenced, not copied) |
| `medvqa/` | our code (data preparation, smoke test, later our method) |
| `jobs/<name>.sh` | one script per experiment; this is what runs on Kaggle |
| `kaggle/runner/main.py.tmpl` | the Kaggle kernel: clones this repo at a ref and runs `jobs/<name>.sh` |
| `tools/make_kernel.py` | builds a kernel folder under `kaggle/experiments/<name>/` for the control plane |
| `requirements-kaggle.txt` | packages installed on Kaggle on top of the base image |

## Workflow

1. Edit code locally, commit, `git push`.
2. Build a kernel folder for a job:
   ```bash
   python tools/make_kernel.py smoke --env MODEL=Qwen/Qwen3-VL-4B-Instruct
   ```
3. Submit `kaggle/experiments/kltn-smoke` from the Kaggle Team Control Plane dashboard, choosing the owner
   account and accelerator `gpu`. The kernel clones the repo, installs `requirements-kaggle.txt`, runs the job,
   and writes everything to `/kaggle/working/outputs/`, which the control plane downloads.

Jobs receive `OUT_DIR` (downloaded results), `WORK_DIR` (scratch, not downloaded) and `HF_HOME` (model cache).
Keep large files out of `OUT_DIR`.

## Offline runs (internet off, e.g. the competition RTX Pro 6000)

Competition GPUs such as the RTX Pro 6000 (96 GB, bf16) only run with internet off, so the runner
cannot `git clone` or `pip install`. Instead two public kernels provide inputs via `kernel_sources`:

| Kernel | Job | Refresh when |
|---|---|---|
| `hcmuser/kltn-snapshot-code` | `jobs/snapshot_code.sh`: repo + CRAFT code (~23 MB, result JSONs stripped) | after every push you want to run offline |
| `hcmuser/kltn-build-wheels` | `jobs/build_wheels.sh`: wheels for `requirements-kaggle.txt` (Python 3.13 image) | when requirements change |

```bash
python tools/make_kernel.py snapshot_code --public --accelerator cpu   # submit on hcmuser, CPU
python tools/make_kernel.py smoke --rtx --model qwen-lm/qwen-3-vl/transformers/4b-instruct/1 --dataset phandai/vqa-rad-craft
```

`--rtx` = `--offline --machine-shape NvidiaRtxPro6000 --competition arc-prize-2026-arc-agi-3`. The
account must have joined that competition. In the control plane, also pass
`{"accelerator": "gpu", "machine_shape": "NvidiaRtxPro6000"}` as job metadata. Offline runs read code
from the snapshot, not GitHub, so refresh the snapshot first.

## Notes for Kaggle T4

T4 has no bf16: run models in fp16 with eager attention. `jobs/smoke.sh` checks that logits stay finite.
