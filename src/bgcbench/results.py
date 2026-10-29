"""Evaluation results: per-column cells, the run environment and the printed table."""

from __future__ import annotations

import json
import math
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import __version__, data
from .encoders.base import cpu_model, display_name


def environment() -> dict:
    """Package, dataset and library versions and the CPU, as recorded in each result."""
    import scipy
    import sklearn
    import torch
    import pandas

    return {
        "bgcbench": __version__, "dataset": _dataset_source(),
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
        "pandas": pandas.__version__, "torch": torch.__version__,
        "cpu": cpu_model(), "machine": platform.machine(),
    }


def _dataset_source() -> str:
    """``org/repo@revision`` of the main dataset, or the local directory ``BGCBENCH_DATA``."""
    if os.environ.get("BGCBENCH_DATA"):
        return str(data.root())
    return f"{data.DATASET_REPO}@{data.DATASET_REVISION}"


@dataclass
class Result:
    """Scores of one encoder or embedding set on the selected tasks.

    ``cells``: ``{column id: cell}`` in evaluation order. ``tasks``: ``{task: {version,
    metric, official}}``, plus ``dataset`` for task-owned data and ``replaced`` (``{kind:
    source}``) when default vectors were replaced, e.g. other molecule vectors in retrieval.
    :meth:`table` stars tasks outside the official suite and names replaced vectors.
    """

    name: str
    cells: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    tasks: dict = field(default_factory=dict)

    def to_json(self, path: Path) -> None:
        """Write the result as JSON; undefined values (NaN) become ``null``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = _plain({"encoder": self.name, "environment": self.env, "tasks": self.tasks,
                      "cells": self.cells})
        path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")

    def table(self) -> str:
        """The cells as a text table, one line per column."""
        lines = [f"BGC-Bench v{__version__}: {display_name(self.name)}",
                 f"{'column':<14}{'metric':<20}{'value':>8}  95% CI"]
        extra = []
        for col, cell in self.cells.items():
            task = col.rsplit("_", 1)[0]
            info = self.tasks.get(task, {})
            if not info.get("official", True):
                if task not in extra:
                    extra.append(task)
                col += "*"
            metric = info.get("metric", "")
            if cell.get("status") == "na":
                lines.append(f"{col:<14}{metric:<20}{'--':>8}  {cell.get('note', '')}")
                continue
            line = (f"{col:<14}{metric:<20}{cell['point']:>8.4f}  "
                    f"[{cell['ci'][0]:.3f}, {cell['ci'][1]:.3f}]")
            if info.get("replaced"):
                line += "  (" + ", ".join(f"{k}: {v}" for k, v in info["replaced"].items()) + ")"
            if "n_split" in cell and cell.get("n_embedded", cell["n_split"]) < cell["n_split"]:
                line += f"  ({cell['n_embedded']} of {cell['n_split']} split rows embedded)"
            lines.append(line)
        if extra:
            lines.append("* not in the official suite: " + ", ".join(
                f"{t} (v{self.tasks[t]['version']})" for t in extra))
        return "\n".join(lines)


def _plain(o):
    """``o`` with NumPy values as Python ones and non-finite floats as ``None``."""
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [_plain(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    if o is None or isinstance(o, str):
        return o
    if isinstance(o, os.PathLike):
        return os.fspath(o)
    raise TypeError(f"{type(o).__name__} is not JSON serialisable")
