"""One learning curve: fit, score, acquire, over a task's budget grid.

Round 0 labels the first ``n0`` rows of the seed's pool permutation. Each round fits the
readout on the labelled pool rows (always in ascending row order), scores the validation
and test rows with that fit, and hands the same fit to the strategy, which labels enough
rows to reach the next budget.
"""

from __future__ import annotations

import time

import numpy as np

from . import strategies as S
from .design import Partition


def run_trajectory(readout, part: Partition, strategy: S.Strategy, seed: int) -> dict:
    """``{"seed", "strategy", "grid", "initial", "rounds": [{"n", "val", "test",
    "acquired", ...}], "chance"}``; ``acquired`` holds pool positions in pick order."""
    perm = part.permutation(seed)
    grid = list(part.grid)
    labelled = np.sort(perm[:grid[0]])
    initial = labelled.copy()
    cache: dict = {}
    rounds, fitted = [], None
    for r, n in enumerate(grid):
        t0 = time.time()
        acquired, info = np.zeros(0, dtype=np.int64), {}
        if r > 0:
            ctx = S.Context(readout, fitted, part.pool, labelled, perm, int(seed), r, cache)
            acquired, info = S.acquire(strategy, ctx, n - len(labelled))
            labelled = np.sort(np.concatenate([labelled, acquired]))
        rows = part.pool[labelled]
        fitted = readout.fit(rows, int(seed))
        val = readout.score(readout.predict(fitted, part.val), part.val)
        test = readout.score(readout.predict(fitted, part.test), part.test)
        for name, value in (("val", val), ("test", test)):
            if not np.isfinite(value):
                raise ValueError(f"{strategy.name} seed {seed} n={n}: {name} score is {value}")
        fit_info = readout.info(fitted)
        rounds.append({"n": int(n), "val": float(val), "test": float(test),
                       "acquired": acquired.astype(int).tolist(),
                       **({"acq": info} if info else {}),
                       **({"fit": fit_info} if fit_info else {}),
                       "seconds": round(time.time() - t0, 3)})
    record = {"seed": int(seed), "strategy": strategy.name, "grid": grid,
              "initial": initial.astype(int).tolist(), "rounds": rounds,
              "chance": {"val": float(readout.chance(part.val)),
                         "test": float(readout.chance(part.test))}}
    check_trajectory(record, perm, len(part.pool))
    return record


def check_trajectory(record: dict, perm: np.ndarray, pool_size: int) -> None:
    """Budgets follow the grid, labelled sets are nested and distinct, random acquisition is
    the permutation's continuation, and a curve that reaches the pool labels all of it."""
    grid = record["grid"]
    n0 = grid[0]
    seen = list(record["initial"])
    if [r["n"] for r in record["rounds"]] != grid or seen != sorted(int(i) for i in perm[:n0]):
        raise AssertionError("rounds do not follow the budget grid and seed permutation")
    for r, n in zip(record["rounds"][1:], grid[1:]):
        seen += r["acquired"]
        if len(seen) != n or len(set(seen)) != n:
            raise AssertionError(f"labelled set at n={n} is not {n} distinct rows")
    if record["strategy"] == "random" and seen[n0:] != [int(i) for i in perm[n0:len(seen)]]:
        raise AssertionError("random acquisition left the seed's permutation")
    if grid[-1] == pool_size and sorted(seen) != list(range(pool_size)):
        raise AssertionError("the last round does not label the whole pool")
