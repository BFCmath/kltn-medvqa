#!/usr/bin/env bash
# Copy the cloned repo (with the CRAFT submodule) into the kernel output so offline kernels
# can mount it via kernel_sources. Run on CPU with internet: tools/make_kernel.py snapshot_code --public --accelerator cpu
set -euo pipefail
DEST="$OUT_DIR/src"
rm -rf "$DEST"
mkdir -p "$DEST"
tar --exclude=.git --exclude='*.xlsx' -cf - . | tar -xf - -C "$DEST"
python - "$DEST" <<'EOF'
import json, subprocess, sys, datetime
dest = sys.argv[1]
info = {
    "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    "craft_commit": subprocess.run(["git", "-C", "third_party/CRAFT", "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
    "created_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
}
json.dump(info, open(f"{dest}/kltn_code_snapshot.json", "w"), indent=2)
print(json.dumps(info))
EOF
du -sh "$DEST"
