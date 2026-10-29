"""Embedding and evaluation: :func:`run` is :func:`embed_all` followed by :func:`evaluate`.

:func:`evaluate` scores vectors on a list of tasks (default: the official suite), one
column per task split. :func:`embed_all` returns every representation kind an encoder
provides, from the cache when possible; :func:`embed` returns only the BGC and A-domain
vectors.
"""

from __future__ import annotations

import importlib.util
import json
import math
import numbers
import os
import time
from pathlib import Path

from . import SEED, data, limit_threads, set_deterministic
from . import representations as kinds
from .encoders.base import (Embeddings, Encoder, cache_dir, device_name, encoder_class,
                            read_representations)
from .results import Result, environment
from .tasks import is_official
from .tasks.base import Task, select


def evaluate(bgc, adomain=None, chem_embeddings=None, *, reps: dict | None = None,
             name: str = "custom", tasks=None, n_boot: int = 500,
             retrieval_device: str = "cpu", threads: int | None = None,
             log=print) -> Result:
    """Score frozen embeddings on the official columns, or on the ``tasks`` named.

    ``bgc``: BGC vectors (``.npz`` with ``bgc_ids`` + ``embeddings``, a dict, or an
    :class:`Embeddings`). ``adomain``: optional A-domain vectors keyed by ``domain_id``
    (without them the substrate columns are "--"). ``chem_embeddings``: optional molecule
    vectors (``compound_ids`` + ``embeddings``) replacing the shipped MoLFormer vectors in
    retrieval. ``reps``: vectors of further representation kinds, ``{kind: vectors}`` (see
    :mod:`bgcbench.representations`).
    ``tasks``: names or :class:`Task` instances (default: the official suite ``SUITE_V1``).
    ``retrieval_device`` trains the retrieval head (default CPU) and is passed to every
    task as ``device``; ``threads`` overrides the per-task protocol thread counts.
    ``name`` labels the result and selects a task's per-encoder thread count
    (``Task.threads_override``). ``log`` receives one line per scored column.

    Every column runs under :func:`bgcbench.limit_threads` after
    :func:`bgcbench.set_deterministic`.
    """
    _check_options(n_boot, threads)
    selected = select(tasks)
    vectors, replaced = gather(bgc, adomain, chem_embeddings, reps, selected)
    result = Result(name, env={**environment(), "threads": {}, "retrieval_device":
                               device_name(retrieval_device), "n_boot": n_boot})
    for task in selected:
        result.tasks[task.name] = {"version": task.version, "metric": task.metric,
                                   "official": is_official(task)}
        if task.dataset is not None:
            result.tasks[task.name]["dataset"] = task.dataset
        swapped = {k: replaced[k] for k in task.requires if k in replaced}
        if swapped:
            result.tasks[task.name]["replaced"] = swapped
        missing = [k for k in task.requires if k not in vectors]
        for col, split in task.columns():
            if missing:
                result.cells[col] = {"status": "na",
                                     "note": f"no {kinds.get(missing[0]).name} vectors"}
                continue
            n_threads = threads or task.threads_for(name)
            result.env["threads"][task.name] = n_threads
            t0 = time.time()
            try:
                with limit_threads(n_threads):
                    set_deterministic(SEED)
                    cell = task.evaluate({k: vectors[k] for k in task.requires}, split,
                                         n_boot=n_boot, device=retrieval_device)
            except data.CoverageError as exc:
                result.cells[col] = {"status": "na", "note": str(exc)}
                log(f"  {col:<14}--      {exc}")
                continue
            _check_cell(task, col, cell)
            cell["seconds"] = round(time.time() - t0, 1)
            result.cells[col] = cell
            log(f"  {col:<14}{cell['point']:.4f}  [{cell['ci'][0]:.4f}, {cell['ci'][1]:.4f}]"
                f"  ({cell['seconds']} s)")
    return result


def gather(bgc, adomain=None, chem_embeddings=None, reps: dict | None = None,
           tasks=()) -> tuple[dict, dict]:
    """``({kind: Embeddings}, {kind: source})``: the given vectors, plus the default vectors
    of kinds the ``tasks`` need (the shipped MoLFormer molecules); the second dict names the
    given vectors that replace such a default."""
    given = {"bgc": bgc, "adomain": adomain, "molecule": chem_embeddings}
    twice = sorted(k for k in reps or {} if given.get(k) is not None)
    if twice:
        raise ValueError(f"{twice} given both as an argument and in reps")
    supplied = {kind: obj for kind, obj in {**given, **(reps or {})}.items() if obj is not None}
    vectors = {kind: kinds.coerce(kind, obj) for kind, obj in supplied.items()}
    # Kinds whose protocol default (the shipped MoLFormer molecules) the caller replaced.
    replaced = {kind: _source(obj) for kind, obj in supplied.items()
                if kinds.get(kind).default is not None}
    for kind in {k for t in tasks for k in t.requires} - set(vectors):
        default = kinds.get(kind).default
        if default is not None:
            vectors[kind] = kinds.coerce(kind, default())
    return vectors, replaced


def _check_options(n_boot: int, threads: int | None) -> None:
    if n_boot < 0:
        raise ValueError("n_boot must be >= 0")
    if threads is not None and threads < 1:
        raise ValueError("threads must be >= 1")


def _source(obj) -> str:
    """A short name for given vectors: the file stem, or "custom"."""
    return Path(obj).stem if isinstance(obj, (str, os.PathLike)) else "custom"


def _check_cell(task: Task, col: str, cell) -> None:
    """A task returns ``{"point": number, "ci": [lo, hi], ...}``; ``ci`` defaults to NaN."""
    where = f"task {task.name!r} ({col})"
    if not isinstance(cell, dict):
        raise TypeError(f"{where} returned {type(cell).__name__}, expected a dict")
    point = cell.get("point")
    # NaN is allowed: a metric can be undefined on a degenerate test set.
    if (not isinstance(point, numbers.Real) or isinstance(point, bool)
            or math.isinf(point)):
        raise ValueError(f"{where}: 'point' must be a real number (NaN if undefined, not "
                         f"inf), got {point!r}")
    cell.setdefault("ci", [float("nan"), float("nan")])  # n_boot = 0
    ci = cell["ci"]
    if (isinstance(ci, (str, bytes)) or not hasattr(ci, "__len__") or len(ci) != 2
            or not all(isinstance(b, numbers.Real) and not isinstance(b, bool) for b in ci)):
        raise ValueError(f"{where}: 'ci' must be [lo, hi] of real numbers, got {ci!r}")


def embed(encoder, *, device: str | None = None,
          refresh: bool = False) -> tuple[Embeddings, Embeddings | None]:
    """Return cached BGC/A-domain embeddings, computing them on a cache miss."""
    reps = embed_all(encoder, device=device, refresh=refresh)
    return reps["bgc"], reps.get("adomain")


def embed_all(encoder, *, device: str | None = None,
              refresh: bool = False) -> dict[str, Embeddings]:
    """``{kind: Embeddings}`` of every kind the encoder provides (cached after the first call)."""
    if isinstance(encoder, str):
        # a cache hit needs no model, so GPU-environment encoders score in the main env
        cls = encoder_class(encoder)
        hit = None if refresh else read_representations(cache_dir(cls), gpu=cls.gpu)
        if hit is None:
            missing = [m for m in cls.needs if importlib.util.find_spec(m) is None]
            if missing:
                raise ImportError(f"{encoder} needs {', '.join(missing)}, which the GPU "
                                  f"environment has: cd envs/gpu && uv sync --frozen, embed "
                                  f"there, then score here (the embeddings are cached)")
            hit = cls(device=device).representations(refresh=refresh)
        return hit
    if not isinstance(encoder, Encoder):
        raise TypeError("encoder must be a built-in name or an Encoder instance")
    if device is not None:
        raise ValueError("device applies to built-in encoder names; give an Encoder instance "
                         "its device when constructing it")
    return encoder.representations(refresh=refresh)


def run(encoder, *, device: str | None = None, refresh: bool = False, tasks=None,
        **kwargs) -> Result:
    """Embed the dataset with an encoder (built-in name or :class:`Encoder`) and evaluate.

    Other keyword arguments go to :func:`evaluate`.
    """
    # unknown task names and bad options fail before any embedding
    tasks = select(tasks)
    _check_options(kwargs.get("n_boot", 500), kwargs.get("threads"))
    reps = dict(embed_all(encoder, device=device, refresh=refresh))
    owner = encoder_class(encoder) if isinstance(encoder, str) else encoder
    result = evaluate(reps.pop("bgc"), reps=reps, name=owner.name, tasks=tasks, **kwargs)
    return _with_embedding_device(result, owner)


def _with_embedding_device(result: Result, encoder) -> Result:
    """Record which device made the (possibly cached) embeddings: GPU types differ."""
    man = cache_dir(encoder) / "manifest.json"
    if man.exists():
        result.env["embedding_device"] = json.loads(man.read_text())["device"]
    return result
