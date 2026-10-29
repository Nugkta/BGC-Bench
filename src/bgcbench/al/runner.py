"""Active-learning runs: one encoder's learning curves over tasks, strategies and seeds.

:func:`evaluate` takes vectors as :func:`bgcbench.evaluate` does; :func:`run` embeds with an
encoder first. Every (task, strategy, seed) cell is one learning curve, computed on one CPU
thread; ``workers`` runs cells in parallel processes. With ``cache``, each finished cell is
written to ``<cache>/<task>/<strategy>__seed<k>.json`` and reused by a later run with the same
task rows, partition, vectors and package version, so an interrupted run resumes.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import SEED, __version__, data, limit_threads, set_deterministic
from ..encoders.base import device_name, display_name
from ..results import _plain, environment
from ..tasks.base import REGISTRY as TASKS
from ..tasks.base import SUITE_V1, Task, is_official, register_task, select
from . import design as D
from . import stats
from . import strategies as S
from .loop import run_trajectory

THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def has_al(task: Task) -> bool:
    """Whether a task defines an AL protocol (overrides ``Task.al_rows``)."""
    return type(task).al_rows is not Task.al_rows


def al_tasks(tasks=None) -> list[Task]:
    """Tasks with an AL protocol: the named ones, or the official suite."""
    chosen = select(SUITE_V1 if tasks is None else tasks)
    missing = [t.name for t in chosen if not has_al(t)]
    if missing:
        raise ValueError(f"task(s) {missing} have no active-learning protocol (Task.al_rows)")
    return chosen


def list_tasks() -> list[str]:
    """Registered tasks with an AL protocol."""
    return [name for name, task in TASKS.items() if has_al(task)]


def stream(task: Task) -> int:
    return task.al_stream if task.al_stream is not None else D.default_stream(task.name)


# Results
@dataclass
class ALResult:
    """Learning curves of one encoder or embedding set.

    ``curves[task][strategy]``: ``{"status": "ok", "seeds": {seed: {"n", "val", "test"}}}``
    (score lists over the budget grid), or ``{"status": "na", "note": ...}``. ``tasks[task]``
    records the task version, metric, budget grid, partition sizes and chance levels.
    """

    name: str
    protocol: dict
    env: dict = field(default_factory=dict)
    tasks: dict = field(default_factory=dict)
    curves: dict = field(default_factory=dict)

    def scores(self, task: str, strategy: str, which: str = "test", seeds=None
               ) -> tuple[list[int], list[int], np.ndarray]:
        """``(seeds, budgets, scores)``, scores (seeds, budgets), for one task and strategy."""
        cell = self.curves[task][strategy]
        if cell.get("status") != "ok":
            raise KeyError(f"{task}/{strategy}: {cell.get('note', 'no curves')}")
        have = {int(s): c for s, c in cell["seeds"].items()}
        seeds = sorted(have) if seeds is None else [int(s) for s in seeds]
        missing = [s for s in seeds if s not in have]
        if missing:
            raise KeyError(f"{task}/{strategy}: no curves for seeds {missing}")
        ns = have[seeds[0]]["n"]
        return seeds, ns, np.array([have[s][which] for s in seeds], dtype=np.float64)

    def aulc_low(self, task: str, strategy: str, which: str = "test", seeds=None) -> np.ndarray:
        """Per-seed AULC_low (budgets up to 256)."""
        _, ns, Y = self.scores(task, strategy, which, seeds)
        return stats.aulc_low(ns, Y)

    def table(self) -> str:
        """Test AULC_low per task and strategy: seed mean, 95 % seed-bootstrap interval, and
        the seed-paired difference from random acquisition."""
        p = self.protocol
        budget = "full pool" if p["max_budget"] is None else f"<= {p['max_budget']} labels"
        lines = [f"BGC-Bench v{__version__} active learning: {display_name(self.name)}",
                 f"protocol {p['name']}: {len(p['seeds'])} seeds, {budget}; "
                 f"test AULC_low (16-256 labels)",
                 f"{'task':<12}{'strategy':<12}{'AULC_low':>9}  {'95% CI':<16}{'vs random':>10}"]
        extra = []
        for task, cells in self.curves.items():
            info = self.tasks.get(task, {})
            label = task
            if not info.get("official", True):
                label += "*"
                extra.append(task)
            if info.get("status") == "na":
                lines.append(f"{label:<12}{'':<12}{'--':>9}  {info.get('note', '')}")
                continue
            base = self.aulc_low(task, "random") if cells.get("random", {}).get(
                "status") == "ok" else None
            for strategy, cell in cells.items():
                if cell.get("status") != "ok":
                    lines.append(f"{label:<12}{strategy:<12}{'--':>9}  {cell.get('note', '')}")
                    continue
                a = self.aulc_low(task, strategy)
                b = stats.seed_bootstrap(a)
                delta = ""
                if base is not None and strategy != "random" and len(base) == len(a):
                    delta = f"{float(np.mean(a - base)):+.4f}"
                lines.append(f"{label:<12}{strategy:<12}{b['mean']:>9.4f}  "
                             f"[{b['ci'][0]:.3f}, {b['ci'][1]:.3f}]  {delta:>9}")
        if extra:
            lines.append("* not in the official suite: " + ", ".join(
                f"{t} (v{self.tasks[t]['version']})" for t in extra))
        return "\n".join(lines)

    def long_rows(self) -> list[dict]:
        """One dict per (task, strategy, seed, budget): encoder, n, val and test scores."""
        out = []
        for task, cells in self.curves.items():
            for strategy, cell in cells.items():
                for seed, c in (cell.get("seeds") or {}).items():
                    for n, v, t in zip(c["n"], c["val"], c["test"]):
                        out.append({"encoder": self.name, "task": task, "strategy": strategy,
                                    "seed": int(seed), "n": int(n), "val": v, "test": t})
        return out

    def to_json(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = _plain({"encoder": self.name, "protocol": self.protocol, "environment": self.env,
                      "tasks": self.tasks, "curves": self.curves})
        path.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")


def load(path) -> ALResult:
    """An :class:`ALResult` written by :meth:`ALResult.to_json`."""
    d = json.loads(Path(path).read_text())
    return ALResult(d["encoder"], d["protocol"], d.get("environment", {}), d.get("tasks", {}),
                    d.get("curves", {}))


# Cells
def _digest(vectors: dict, kinds) -> str:
    h = hashlib.sha256()
    for kind in sorted(kinds):
        emb = vectors[kind]
        h.update(kind.encode())
        h.update("\x1f".join(emb.ids.tolist()).encode())
        h.update(np.ascontiguousarray(emb.X).tobytes())
    return h.hexdigest()


def _cell_key(task: Task, rows_digest: str, part: D.Partition, vec_digest: str, strategy: str,
              seed: int, device: str) -> str:
    core = {"bgcbench": __version__, "task": task.name, "version": task.version,
            "rows": rows_digest, "partition": part.digest(), "vectors": vec_digest,
            "strategy": strategy, "seed": int(seed), "device": device}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()


def _cell_path(cache: Path, task: str, strategy: str, seed: int) -> Path:
    return Path(cache) / task / f"{strategy}__seed{int(seed)}.json"


def _read_cell(path: Path, key: str) -> dict | None:
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return d["record"] if d.get("key") == key else None


def _write_cell(path: Path, key: str, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_plain({"key": key, "record": record})) + "\n")
    tmp.replace(path)


class _Worker:
    """Builds each task's rows, partition and readout once, then runs cells."""

    def __init__(self, vectors: dict, proto: D.Protocol, device: str) -> None:
        self.vectors, self.proto, self.device = vectors, proto, device
        self._tasks: dict = {}

    def setup(self, task: Task):
        if task.name not in self._tasks:
            rows = task.al_rows()
            if rows is None:
                raise ValueError(f"task {task.name!r} returned no AL rows")
            part = D.partition(rows, stream(task), self.proto)
            with limit_threads(1):
                readout = task.al_readout({k: self.vectors[k] for k in task.requires}, rows,
                                          device=self.device)
            self._tasks[task.name] = (rows, part, readout)
        return self._tasks[task.name]

    def cell(self, task: Task, strategy: S.Strategy, seed: int) -> dict:
        _, part, readout = self.setup(task)
        t0 = time.time()
        with limit_threads(1):
            set_deterministic(SEED)
            record = run_trajectory(readout, part, strategy, seed)
        record["seconds"] = round(time.time() - t0, 2)
        return record


_WORKER: _Worker | None = None


def _init_worker(vector_dir: str, blob: bytes, modules: list[str]) -> None:
    global _WORKER
    import torch
    from threadpoolctl import threadpool_limits

    from .. import representations as kinds

    torch.set_num_threads(1)
    threadpool_limits(1)
    if modules:
        from ..cli import import_module

        for spec in modules:
            import_module(spec)
    vectors = {p.stem: kinds.coerce(p.stem, p) for p in Path(vector_dir).glob("*.npz")}
    proto, device, tasks, strategies = pickle.loads(blob)
    for t in tasks:
        if isinstance(t, Task) and TASKS.get(t.name) is None:
            register_task(t)
    for s in strategies:
        if isinstance(s, S.Strategy) and s.name not in S.REGISTRY:
            S.register_strategy(s.name, s.fn, needs_gradients=s.needs_gradients)
    _WORKER = _Worker(vectors, proto, device)


def _run_cell(task_name: str, strategy_name: str, seed: int) -> tuple[str, str, int, dict]:
    task = TASKS[task_name]
    record = _WORKER.cell(task, S.get_strategy(strategy_name), seed)
    return task_name, strategy_name, seed, record


@contextmanager
def _single_thread_env():
    saved = {k: os.environ.get(k) for k in THREAD_ENV}
    os.environ.update({k: "1" for k in THREAD_ENV})
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _picklable(objs: list) -> list:
    out = []
    for o in objs:
        try:
            pickle.dumps(o)
            out.append(o)
        except Exception:
            out.append(None)
    return out


# Entry points
def evaluate(bgc, adomain=None, chem_embeddings=None, *, reps: dict | None = None,
             name: str = "custom", tasks=None, strategies=None, protocol="quick", seeds=None,
             max_budget="default", workers: int = 1, cache=None,
             retrieval_device: str = "cpu", modules=(), log=print) -> ALResult:
    """Learning curves of frozen vectors on the AL tasks (default: the official five).

    Vectors are given as for :func:`bgcbench.evaluate`. ``strategies``: names (default:
    every registered strategy). ``protocol``: ``"quick"`` (seeds 1-5, budgets up to 256),
    ``"paper"`` (seeds 1-20, the whole pool) or a :class:`Protocol`; ``seeds`` and
    ``max_budget`` override it. ``workers``: parallel processes, each on one CPU thread;
    custom tasks and strategies then need to be importable, or given as files or modules in
    ``modules``. ``cache``: a directory for finished cells. ``retrieval_device`` trains the
    retrieval head.
    """
    proto = D.protocol(protocol, seeds=seeds, max_budget=max_budget)
    chosen = al_tasks(tasks)
    strats = S.select(strategies)
    if workers < 1:
        raise ValueError("workers must be >= 1")
    from ..pipeline import gather

    vectors, replaced = gather(bgc, adomain, chem_embeddings, reps, chosen)
    result = ALResult(name, proto.as_dict(), env={
        **environment(), "threads": 1, "workers": workers,
        "retrieval_device": device_name(retrieval_device)})
    worker = _Worker(vectors, proto, retrieval_device)
    todo, keys = [], {}
    for task in chosen:
        info = {"version": task.version, "metric": task.metric, "official": is_official(task),
                "stream": stream(task)}
        swapped = {k: replaced[k] for k in task.requires if k in replaced}
        if swapped:
            info["replaced"] = swapped
        result.tasks[task.name] = info
        result.curves[task.name] = {}
        missing = [k for k in task.requires if k not in vectors]
        if missing:
            info.update(status="na", note=f"no {missing[0]} vectors")
            log(f"  {task.name:<12}--  no {missing[0]} vectors")
            continue
        try:
            rows, part, readout = worker.setup(task)
        except data.CoverageError as exc:
            info.update(status="na", note=str(exc))
            log(f"  {task.name:<12}--  {exc}")
            continue
        info.update(status="ok", metric=readout.metric or task.metric, grid=part.grid,
                    **part.notes, chance={"val": readout.chance(part.val),
                                          "test": readout.chance(part.test)})
        rows_digest, vec_digest = rows.digest(), _digest(vectors, task.requires)
        for strategy in strats:
            if strategy.needs_gradients and not readout.has_gradients:
                result.curves[task.name][strategy.name] = {
                    "status": "na", "note": "the readout has no gradient embedding"}
                continue
            result.curves[task.name][strategy.name] = {"status": "ok", "seeds": {}}
            for seed in proto.seeds:
                key = _cell_key(task, rows_digest, part, vec_digest, strategy.name, seed,
                                retrieval_device)
                hit = None if cache is None else _read_cell(
                    _cell_path(cache, task.name, strategy.name, seed), key)
                if hit is not None:
                    _store(result, task.name, strategy.name, seed, hit)
                else:
                    keys[(task.name, strategy.name, seed)] = key
                    todo.append((task, strategy, seed))
    n_cached = sum(len(c.get("seeds", {})) for cells in result.curves.values()
                   for c in cells.values())
    log(f"[bgcbench] {name}: {len(todo)} learning curves to compute"
        + (f", {n_cached} from the cache" if n_cached else ""))

    def done(task_name, strategy_name, seed, record):
        _store(result, task_name, strategy_name, seed, record)
        if cache is not None:
            _write_cell(_cell_path(cache, task_name, strategy_name, seed),
                        keys[(task_name, strategy_name, seed)], record)
        last = record["rounds"][-1]
        log(f"  {task_name:<12}{strategy_name:<12}seed {seed:<3} n={last['n']:<5}"
            f"test {last['test']:.4f}  ({record['seconds']} s)")

    if workers == 1 or len(todo) <= 1:
        for task, strategy, seed in todo:
            done(task.name, strategy.name, seed, worker.cell(task, strategy, seed))
    else:
        import multiprocessing as mp
        import tempfile

        from .. import representations as kinds

        blob = pickle.dumps((proto, retrieval_device, _picklable(chosen), _picklable(strats)))
        # Workers read the vectors from files: large arguments to a spawned process travel
        # through a pipe, which blocks if the process dies while starting.
        with tempfile.TemporaryDirectory(prefix="bgcbench_al_") as tmp:
            for kind, emb in vectors.items():
                emb.save(Path(tmp) / f"{kind}.npz", kinds.get(kind).id_key)
            with _single_thread_env(), ProcessPoolExecutor(
                    max_workers=workers, mp_context=mp.get_context("spawn"),
                    initializer=_init_worker,
                    initargs=(tmp, blob, [str(m) for m in modules])) as pool:
                futures = [pool.submit(_run_cell, t.name, s.name, seed) for t, s, seed in todo]
                for fut in as_completed(futures):
                    done(*fut.result())
    _sort(result, proto)
    return result


def _store(result: ALResult, task: str, strategy: str, seed: int, record: dict) -> None:
    rounds = record["rounds"]
    result.curves[task][strategy]["seeds"][int(seed)] = {
        "n": [r["n"] for r in rounds], "val": [r["val"] for r in rounds],
        "test": [r["test"] for r in rounds], "seconds": record.get("seconds")}


def _sort(result: ALResult, proto: D.Protocol) -> None:
    for cells in result.curves.values():
        for cell in cells.values():
            if "seeds" in cell:
                cell["seeds"] = {s: cell["seeds"][s] for s in proto.seeds if s in cell["seeds"]}


def run(encoder, *, device: str | None = None, refresh: bool = False, tasks=None,
        **kwargs) -> ALResult:
    """Embed the dataset with an encoder (built-in name or :class:`bgcbench.Encoder`), then
    run :func:`evaluate`; other keyword arguments go to it."""
    from ..encoders.base import encoder_class
    from ..pipeline import _with_embedding_device, embed_all

    chosen = al_tasks(tasks)
    D.protocol(kwargs.get("protocol", "quick"), seeds=kwargs.get("seeds"),
               max_budget=kwargs.get("max_budget", "default"))
    S.select(kwargs.get("strategies"))
    reps = dict(embed_all(encoder, device=device, refresh=refresh))
    owner = encoder_class(encoder) if isinstance(encoder, str) else encoder
    result = evaluate(reps.pop("bgc"), reps=reps, name=owner.name, tasks=chosen, **kwargs)
    return _with_embedding_device(result, owner)
