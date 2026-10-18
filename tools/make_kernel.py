"""Generate a Kaggle kernel folder (main.py + kernel-metadata.json) for one job.

The folder is what the Kaggle Team Control Plane stages and pushes (or what
`kaggle kernels push -p <folder>` uploads directly).

    python tools/make_kernel.py smoke --model qwen-lm/qwen-3-vl/transformers/4b-instruct/1 --dataset phandai/vqa-rad-craft
    python tools/make_kernel.py craft_vqarad --env MODEL_VARIATION=4b-instruct --ref main
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "https://github.com/BFCmath/kltn-medvqa.git"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", help="name of a script in jobs/ (without .sh)")
    ap.add_argument("--name", help="kernel folder / slug name (default: kltn-<job>)")
    ap.add_argument("--ref", default="main", help="git branch or tag to run")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE passed to the job")
    ap.add_argument("--accelerator", default="gpu", choices=("gpu", "cpu"))
    ap.add_argument("--dataset", action="append", default=[],
                    help="Kaggle dataset to mount under /kaggle/input (owner/slug)")
    ap.add_argument("--model", action="append", default=[],
                    help="Kaggle model to mount, e.g. qwen-lm/qwen-3-vl/transformers/4b-instruct/1")
    args = ap.parse_args()

    if not (ROOT / "jobs" / f"{args.job}.sh").exists():
        raise SystemExit(f"jobs/{args.job}.sh not found")
    env = dict(kv.split("=", 1) for kv in args.env)
    name = args.name or f"kltn-{args.job}".replace("_", "-")
    out = ROOT / "kaggle" / "experiments" / name
    out.mkdir(parents=True, exist_ok=True)

    code = (ROOT / "kaggle" / "runner" / "main.py.tmpl").read_text()
    code = (code.replace("__REPO__", args.repo).replace("__REF__", args.ref)
                .replace("__JOB__", args.job).replace("__JOB_ENV__", json.dumps(env)))
    (out / "main.py").write_text(code)
    meta = {
        "id": f"replace-me/{name}",       # the control plane rewrites owner/slug per account
        "title": name,
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": args.accelerator == "gpu",
        "enable_internet": True,          # only for cloning the repo and pip; data/models come from mounts
        "dataset_sources": args.dataset,
        "model_sources": args.model,
        "competition_sources": [],
        "kernel_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
