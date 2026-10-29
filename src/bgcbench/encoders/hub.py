"""Fetch encoder weights from the Hugging Face Hub at the revisions pinned in bgcbench.MODELS.

With ``BGCBENCH_MODELS=<dir>``, weights are read from ``<dir>/<key>/`` (``<key>`` a key of
``bgcbench.MODELS``, e.g. ``esm2_8m`` for BGC-Prophet) where that file or folder exists, and
from the Hub otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

from .. import MODELS


def model_file(key: str, filename: str) -> str:
    """Local path of one weight file of ``MODELS[key]`` (downloaded on first use)."""
    local = os.environ.get("BGCBENCH_MODELS")
    if local and (Path(local) / key / filename).exists():
        return str(Path(local) / key / filename)
    from huggingface_hub import hf_hub_download

    repo, revision = MODELS[key]
    return hf_hub_download(repo, filename, revision=revision)


def model_dir(key: str, allow_patterns: list[str] | None = None) -> str:
    """Local folder of ``MODELS[key]``, optionally only the files matching ``allow_patterns``."""
    local = os.environ.get("BGCBENCH_MODELS")
    if local and (Path(local) / key).is_dir():
        return str(Path(local) / key)
    from huggingface_hub import snapshot_download

    repo, revision = MODELS[key]
    return snapshot_download(repo, revision=revision, allow_patterns=allow_patterns)
