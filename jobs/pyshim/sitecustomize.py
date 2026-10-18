"""Loaded automatically at interpreter start when jobs/pyshim is on PYTHONPATH.

Some CRAFT scripts insert spec-less stub modules for optional video libraries (decord, ffmpeg,
imageio) *before* importing transformers. Newer transformers calls importlib.util.find_spec on
them and crashes ("decord.__spec__ is None"). Importing transformers first avoids that without
editing CRAFT's code.
"""
try:
    import transformers  # noqa: F401
except Exception:
    pass
