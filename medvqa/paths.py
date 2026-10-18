"""Locate inputs mounted by Kaggle under /kaggle/input (models and datasets are never downloaded in runs)."""
import glob
import os
from pathlib import Path

KAGGLE_INPUT = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))


def find_model_dir(variation: str) -> Path:
    """Directory holding config.json for a mounted Kaggle model variation, e.g. '4b-instruct'.

    Kaggle mounts models at /kaggle/input/<model>/<framework>/<variation>/<version>/,
    and some mirrors nest the files one level deeper, so search for config.json.
    """
    pattern = str(KAGGLE_INPUT / "**" / variation / "**" / "config.json")
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        raise FileNotFoundError(f"no mounted model matching '{variation}' under {KAGGLE_INPUT}")
    return Path(hits[0]).parent


def find_dataset_dir(slug: str, marker: str) -> Path:
    """Directory of a mounted dataset that contains `marker` (a file name)."""
    hits = sorted(glob.glob(str(KAGGLE_INPUT / "**" / marker), recursive=True, include_hidden=False))
    if not hits:  # some mounts are symlinks that recursive glob may not descend into
        hits = sorted(str(p) for p in KAGGLE_INPUT.rglob(marker))
    hits = [h for h in hits if slug in h] or hits
    if not hits:
        raise FileNotFoundError(f"no mounted dataset with {marker} under {KAGGLE_INPUT}")
    return Path(hits[0]).parent
