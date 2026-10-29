"""Active-learning rows, protocols and the fixed partition every encoder shares.

A task's AL rows (:class:`Rows`) come from its dataset alone: item ids, the test rows, and
optional strata and groups for the validation carve-out. From them :func:`partition`
derives, without any embeddings,

* the validation rows: a fixed 20 % of the training side (``train_test_split`` with seed 0,
  stratified when the task gives strata; with groups, the first fold of
  ``StratifiedGroupKFold(5, shuffle=True, random_state=0)``, so a group stays together);
* the acquisition pool: the remaining training rows;
* the budget grid ``n_k = round(n0 * 2^(k/2))`` up to the pool size or ``max_budget``;
* per seed, a permutation of the pool from ``default_rng([seed, stream])``: its first ``n0``
  entries are the initial labels and the rest is the order of random acquisition.

So every encoder and strategy of a seed starts from the same labels.
"""

from __future__ import annotations

import hashlib
import math
import zlib
from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

N0 = 16
VAL_FRACTION = 0.2
VAL_SEED = 0
AULC_LOW_MAX = 256
BADGE_STREAM = 1000   # BADGE draws: default_rng([seed, BADGE_STREAM + round])


@dataclass
class Rows:
    """A task's AL rows, from its dataset alone.

    ``ids``: item ids in a fixed order (BGC or A-domain ids, matching ``kind``). ``test``:
    True for the held-out test rows; the others are the training side, split into the pool
    and the validation rows. ``strata``: one label per row, stratifying the validation
    carve-out (only training rows are read). ``groups``: one group per row; rows of a group
    stay on the same side of the carve-out. ``table``: per-row data for the readout, in the
    order of ``ids``.
    """

    ids: np.ndarray
    test: np.ndarray
    kind: str = "bgc"
    strata: np.ndarray | None = None
    groups: np.ndarray | None = None
    table: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        self.ids = np.asarray(self.ids).astype(str)
        self.test = np.asarray(self.test, dtype=bool)
        n = len(self.ids)
        if len(set(self.ids.tolist())) != n:
            raise ValueError("AL row ids are not unique")
        for name in ("test", "strata", "groups"):
            value = getattr(self, name)
            if value is not None and len(value) != n:
                raise ValueError(f"{name} has {len(value)} entries for {n} rows")
        if self.table is not None:
            if len(self.table) != n:
                raise ValueError(f"table has {len(self.table)} rows for {n} ids")
            self.table = self.table.reset_index(drop=True)
        if not self.test.any() or self.test.all():
            raise ValueError("AL rows need both test and training rows")

    def features(self, emb) -> np.ndarray:
        """The rows' vectors from :class:`bgcbench.Embeddings`, in row order; every row needs
        one (:class:`bgcbench.data.CoverageError` otherwise)."""
        from .. import data

        X, _, _ = data.align(emb.ids, emb.X, pd.DataFrame({"id": self.ids}), key="id",
                             require_all=True)
        return X

    def digest(self) -> str:
        h = hashlib.sha256()
        for part in (self.ids, self.test.astype(np.uint8),
                     self.strata if self.strata is not None else np.array([], dtype=str),
                     self.groups if self.groups is not None else np.array([], dtype=str)):
            h.update("\x1f".join(np.asarray(part).astype(str).tolist()).encode())
            h.update(b"\x1e")
        return h.hexdigest()


@dataclass(frozen=True)
class Protocol:
    """How long, from where and over which seeds the learning curves run.

    ``seeds``: one learning curve per seed and strategy. ``max_budget``: the largest label
    budget (``None``: the whole pool). ``selection_seeds`` / ``report_seeds``: seeds used to
    choose each encoder's strategy on validation and to report the choice on test (empty:
    no selection). ``n0`` and ``val_fraction`` set the initial labels and the validation
    share. The name records which preset the values come from (``"custom"`` otherwise).
    """

    name: str = "quick"
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    max_budget: int | None = AULC_LOW_MAX
    selection_seeds: tuple[int, ...] = ()
    report_seeds: tuple[int, ...] = ()
    n0: int = N0
    val_fraction: float = VAL_FRACTION

    def __post_init__(self) -> None:
        seeds = tuple(int(s) for s in self.seeds)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "selection_seeds", tuple(int(s) for s in self.selection_seeds))
        object.__setattr__(self, "report_seeds", tuple(int(s) for s in self.report_seeds))
        if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
            raise ValueError(f"seeds must be distinct non-negative integers, got {self.seeds}")
        if not set(self.selection_seeds + self.report_seeds) <= set(seeds):
            raise ValueError("selection and report seeds must be among the seeds")
        if set(self.selection_seeds) & set(self.report_seeds):
            raise ValueError("selection and report seeds overlap")
        if self.n0 < 2:
            raise ValueError("n0 must be at least 2")
        if self.max_budget is not None and self.max_budget <= self.n0:
            raise ValueError(f"max_budget must exceed n0 = {self.n0}")
        if not 0 < self.val_fraction < 1:
            raise ValueError("val_fraction must be in (0, 1)")

    def as_dict(self) -> dict:
        return {"name": self.name, "seeds": list(self.seeds), "max_budget": self.max_budget,
                "selection_seeds": list(self.selection_seeds),
                "report_seeds": list(self.report_seeds), "n0": self.n0,
                "val_fraction": self.val_fraction}


PROTOCOLS = {
    "quick": Protocol(),
    "paper": Protocol("paper", tuple(range(1, 21)), None, tuple(range(1, 11)),
                      tuple(range(11, 21))),
}


def protocol(name: str | Protocol = "quick", *, seeds=None, max_budget="default") -> Protocol:
    """A preset (``"quick"``: seeds 1-5 up to 256 labels; ``"paper"``: seeds 1-20 over the
    whole pool, strategy selection on seeds 1-10 and report on 11-20), optionally with other
    seeds or another ``max_budget``, which makes it ``"custom"``."""
    base = name if isinstance(name, Protocol) else PROTOCOLS.get(name)
    if base is None:
        raise ValueError(f"unknown protocol {name!r}; choose from {', '.join(PROTOCOLS)}")
    changes = {}
    if seeds is not None and tuple(seeds) != base.seeds:
        seeds = tuple(int(s) for s in seeds)
        changes["seeds"] = seeds
        changes["selection_seeds"] = tuple(s for s in base.selection_seeds if s in seeds)
        changes["report_seeds"] = tuple(s for s in base.report_seeds if s in seeds)
    if max_budget != "default" and max_budget != base.max_budget:
        changes["max_budget"] = max_budget
    if not changes:
        return base
    return replace(base, name="custom", **changes)


def parse_seeds(text: str) -> tuple[int, ...]:
    """``"1-5"``, ``"1,3,7"`` or ``"1-3,8"`` as a tuple of seeds."""
    seeds: list[int] = []
    for part in str(text).split(","):
        lo, sep, hi = part.strip().partition("-")
        if not lo.isdigit() or (sep and not hi.isdigit()):
            raise ValueError(f"cannot read seeds {text!r}; use e.g. 1-5 or 1,3,7")
        seeds.extend(range(int(lo), int(hi) + 1) if sep else [int(lo)])
    return tuple(seeds)


def pfam_universe(ds) -> set[str]:
    """The BGCs with Pfam-name inputs (2,626): every built-in encoder embeds them, so the
    built-in AL tasks are restricted to them."""
    return set(ds.read_json("inputs/pfam_domains_named.json"))


def default_stream(task_name: str) -> int:
    """Permutation stream of a task without ``al_stream``: stable across runs and platforms."""
    return 10_000 + zlib.crc32(task_name.encode()) % 1_000_000


def budget_grid(pool_size: int, n0: int = N0, max_budget: int | None = None) -> list[int]:
    """``n_k = round(n0 * 2^(k/2))`` below the pool size, then the pool size; with
    ``max_budget``, only the budgets up to it."""
    if pool_size <= n0:
        raise ValueError(f"the pool ({pool_size} rows) must exceed n0 = {n0}")
    grid: list[int] = []
    k = 0
    while True:
        n = int(math.floor(n0 * 2.0 ** (k / 2.0) + 0.5))
        if n >= pool_size:
            break
        if not grid or n > grid[-1]:
            grid.append(n)
        k += 1
    grid.append(int(pool_size))
    if max_budget is not None:
        grid = [n for n in grid if n <= max_budget]
    return grid


def seed_permutation(stream: int, seed: int, pool_size: int) -> np.ndarray:
    """Pool-local permutation of a seed; its first ``n0`` entries are the initial labels."""
    rng = np.random.default_rng([int(seed), int(stream)])
    return rng.permutation(int(pool_size)).astype(np.int64)


def carve(n_train: int, strata=None, groups=None, val_fraction: float = VAL_FRACTION
          ) -> tuple[np.ndarray, np.ndarray]:
    """Fixed (seed 0) split of the training rows into (pool, validation), train-local and
    sorted. Groups: ``StratifiedGroupKFold`` with ``round(1 / val_fraction)`` folds, the first
    fold for validation (stratified on ``strata`` when given)."""
    if groups is not None:
        from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

        n_splits = int(round(1 / val_fraction))
        if strata is not None:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=VAL_SEED)
            pool, val = next(iter(cv.split(np.zeros(n_train), strata, groups)))
        else:
            cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=VAL_SEED)
            pool, val = next(iter(cv.split(np.zeros(n_train), groups=groups)))
        return np.sort(pool).astype(np.int64), np.sort(val).astype(np.int64)
    from sklearn.model_selection import train_test_split

    n_val = int(round(val_fraction * n_train))
    pool, val = train_test_split(np.arange(n_train), test_size=n_val, random_state=VAL_SEED,
                                 shuffle=True, stratify=strata)
    return np.sort(pool).astype(np.int64), np.sort(val).astype(np.int64)


@dataclass
class Partition:
    """Pool, validation and test rows (indices into :class:`Rows`), and the budget grid."""

    pool: np.ndarray
    val: np.ndarray
    test: np.ndarray
    grid: list[int]
    stream: int
    n0: int
    notes: dict = field(default_factory=dict)

    def permutation(self, seed: int) -> np.ndarray:
        return seed_permutation(self.stream, seed, len(self.pool))

    def digest(self) -> str:
        h = hashlib.sha256()
        for a in (self.pool, self.val, self.test, np.asarray(self.grid)):
            h.update(np.asarray(a, dtype=np.int64).tobytes())
            h.update(b"|")
        h.update(f"{self.stream}/{self.n0}".encode())
        return h.hexdigest()


def partition(rows: Rows, stream: int, proto: Protocol = PROTOCOLS["quick"]) -> Partition:
    """The fixed pool / validation / test partition and budget grid of ``rows``."""
    train = np.flatnonzero(~rows.test)
    test = np.flatnonzero(rows.test)
    strata = None if rows.strata is None else np.asarray(rows.strata).astype(str)[train]
    groups = None if rows.groups is None else np.asarray(rows.groups).astype(str)[train]
    pool_local, val_local = carve(len(train), strata, groups, proto.val_fraction)
    pool, val = train[pool_local], train[val_local]
    grid = budget_grid(len(pool), proto.n0, proto.max_budget)
    if len(grid) < 2:
        raise ValueError(f"the budget grid {grid} needs at least two budgets")
    notes = {"n_rows": int(len(rows.ids)), "n_pool": int(len(pool)), "n_val": int(len(val)),
             "n_test": int(len(test))}
    if groups is not None:
        g = np.asarray(rows.groups).astype(str)
        notes.update({f"n_groups_{name}": int(len(set(g[idx].tolist())))
                      for name, idx in (("pool", pool), ("val", val), ("test", test))})
    if strata is not None:
        notes["n_strata"] = int(len(Counter(strata.tolist())))
    return Partition(pool, val, test, grid, int(stream), proto.n0, notes)
