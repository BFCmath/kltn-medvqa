#!/usr/bin/env bash
# Download wheels for exactly the packages requirements-kaggle.txt would add or change on top of
# the Kaggle base image, so offline kernels can `pip install --no-index` them.
# Run on CPU with internet: tools/make_kernel.py build_wheels --public --accelerator cpu
# (The runner has already installed them; we resolve against a dry-run report of the original env
#  by asking pip what it would fetch for a clean resolve, restricted to binary wheels.)
set -euo pipefail
DEST="$OUT_DIR/wheels"
mkdir -p "$DEST"
python - "$DEST" <<'EOF'
import json, subprocess, sys
dest = sys.argv[1]
# What pip would install for our requirements in this environment (the runner already installed
# them, so --ignore-installed forces a full report; we then drop packages the base image already
# ships at the same version).
r = subprocess.run([sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed",
                    "--quiet", "--report", "/tmp/report.json", "-r", "requirements-kaggle.txt"],
                   capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-2000:])
report = json.load(open("/tmp/report.json"))
from importlib import metadata
need = []
for item in report["install"]:
    name, ver = item["metadata"]["name"], item["metadata"]["version"]
    try:
        have = metadata.version(name)
    except metadata.PackageNotFoundError:
        have = None
    # keep anything our requirements pinned or that the image lacks / has at another version;
    # skip the big CUDA/torch stack, which must come from the image
    if name.lower().startswith(("torch", "nvidia-", "triton")):
        continue
    need.append(f"{name}=={ver}")
open(f"{dest}/wheels.txt", "w").write("\n".join(need) + "\n")
print(len(need), "packages:", " ".join(need))
d = subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
                    "-d", dest, *need], capture_output=True, text=True)
print(d.stdout[-3000:], d.stderr[-3000:])
if d.returncode:
    sys.exit(d.returncode)
EOF
cp requirements-kaggle.txt "$DEST/"
ls "$DEST" | wc -l
du -sh "$DEST"
