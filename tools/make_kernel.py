"""Generate a Kaggle kernel folder (main.py + kernel-metadata.json) for one job.

The folder is what the Kaggle Team Control Plane stages and pushes (or what
`kaggle kernels push -p <folder>` uploads directly).

Online (internet on; clones the repo and pip-installs at run time):
    python tools/make_kernel.py smoke --model qwen-lm/qwen-3-vl/transformers/4b-instruct/1 --dataset phandai/vqa-rad-craft

Offline (internet off, e.g. the competition RTX Pro 6000): code and wheels come from two public
kernels whose outputs are mounted via kernel_sources. Refresh the code snapshot after each push:
    python tools/make_kernel.py snapshot_code --public --accelerator cpu   # submit on the owner account
    python tools/make_kernel.py build_wheels  --public --accelerator cpu   # only when requirements change
    python tools/make_kernel.py smoke --rtx --model ... --dataset ...

When submitting through the control plane, pass the same machine shape in the job metadata
({"accelerator": "gpu", "machine_shape": "NvidiaRtxPro6000"}); the kernel-metadata value is
only used for direct `kaggle kernels push`.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "https://github.com/BFCmath/kltn-medvqa.git"
CODE_KERNEL = "hcmuser/kltn-snapshot-code"
WHEELS_KERNEL = "hcmuser/kltn-build-wheels"
RTX_SHAPE = "NvidiaRtxPro6000"
RTX_COMPETITION = "arc-prize-2026-arc-agi-3"   # attaching it is what unlocks the RTX Pro shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", help="name of a script in jobs/ (without .sh)")
    ap.add_argument("--name", help="kernel folder / slug name (default: kltn-<job>)")
    ap.add_argument("--ref", default="main", help="git branch or tag to run (online only)")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE passed to the job")
    ap.add_argument("--accelerator", default="gpu", choices=("gpu", "cpu"))
    ap.add_argument("--dataset", action="append", default=[],
                    help="Kaggle dataset to mount under /kaggle/input (owner/slug)")
    ap.add_argument("--model", action="append", default=[],
                    help="Kaggle model to mount, e.g. qwen-lm/qwen-3-vl/transformers/4b-instruct/1")
    ap.add_argument("--kernel-source", action="append", default=[], help="extra kernel output to mount")
    ap.add_argument("--competition", action="append", default=[], help="competition to attach")
    ap.add_argument("--machine-shape", help="e.g. NvidiaTeslaT4 (default for gpu), NvidiaRtxPro6000")
    ap.add_argument("--offline", action="store_true",
                    help="internet off: code + wheels from mounted kernel outputs")
    ap.add_argument("--rtx", action="store_true",
                    help=f"shortcut for --offline --machine-shape {RTX_SHAPE} --competition {RTX_COMPETITION}")
    ap.add_argument("--code-kernel", default=CODE_KERNEL)
    ap.add_argument("--wheels-kernel", default=WHEELS_KERNEL)
    ap.add_argument("--public", action="store_true", help="make the kernel public (needed for shared sources)")
    args = ap.parse_args()

    if args.rtx:
        args.offline = True
        args.machine_shape = args.machine_shape or RTX_SHAPE
        if RTX_COMPETITION not in args.competition:
            args.competition.append(RTX_COMPETITION)
    if not (ROOT / "jobs" / f"{args.job}.sh").exists():
        raise SystemExit(f"jobs/{args.job}.sh not found")
    env = dict(kv.split("=", 1) for kv in args.env)
    name = args.name or f"kltn-{args.job}".replace("_", "-") + ("-rtx" if args.rtx else "")
    out = ROOT / "kaggle" / "experiments" / name
    out.mkdir(parents=True, exist_ok=True)

    tmpl = "main_offline.py.tmpl" if args.offline else "main.py.tmpl"
    code = (ROOT / "kaggle" / "runner" / tmpl).read_text()
    code = (code.replace("__REPO__", args.repo).replace("__REF__", args.ref)
                .replace("__JOB__", args.job).replace("__JOB_ENV__", json.dumps(env)))
    (out / "main.py").write_text(code)
    kernel_sources = list(args.kernel_source)
    if args.offline:
        kernel_sources += [args.code_kernel, args.wheels_kernel]
    meta = {
        "id": f"replace-me/{name}",       # the control plane rewrites owner/slug per account
        "title": name,
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": not args.public,
        "enable_gpu": args.accelerator == "gpu",
        # online: only for cloning the repo and pip; data/models always come from mounts
        "enable_internet": not args.offline,
        "dataset_sources": args.dataset,
        "model_sources": args.model,
        "competition_sources": args.competition,
        "kernel_sources": kernel_sources,
    }
    if args.machine_shape:
        meta["machine_shape"] = args.machine_shape
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
