"""Learning-curve statistics: AULC, seed bootstraps, exact sign-flip tests, multiplicity
corrections and label savings. Seeds are the resampling unit; the test rows stay fixed."""

from __future__ import annotations

import numpy as np

from .design import AULC_LOW_MAX

BOOT_SEED = 20261010
N_BOOT_SEEDS = 10_000
N_BOOT_BAND = 2_000
SAVINGS_REF_POINTS = (32, 64, 128)


def aulc(ns, ys, n_max: int | None = None) -> np.ndarray:
    """Trapezoid area under ``ys`` over log2 n, divided by the log2 range: a mean score.

    ``ys`` is (..., len(ns)); budgets above ``n_max`` are left out.
    """
    ns = np.asarray(ns, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    sel = np.ones(len(ns), dtype=bool) if n_max is None else ns <= n_max
    if sel.sum() < 2:
        raise ValueError("an AULC needs at least two budgets")
    x = np.log2(ns[sel])
    y = ys[..., sel]
    area = (0.5 * (y[..., 1:] + y[..., :-1]) * np.diff(x)).sum(axis=-1)
    return area / (x[-1] - x[0])


def aulc_low(ns, ys) -> np.ndarray:
    """AULC over the budgets up to 256 labels."""
    return aulc(ns, ys, n_max=AULC_LOW_MAX)


def sign_flip_p(d) -> float:
    """Exact two-sided sign-flip p-value of mean(d) = 0 over all 2^n sign patterns."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    if n == 0 or n > 24:
        raise ValueError(f"the exact sign-flip test enumerates 2^n patterns; n = {n}")
    obs = abs(d.sum())
    tol = 1e-12 * max(1.0, float(np.abs(d).sum()))
    total, count = 1 << n, 0
    bits = np.arange(n, dtype=np.int64)
    for start in range(0, total, 1 << 16):
        codes = np.arange(start, min(total, start + (1 << 16)), dtype=np.int64)
        signs = 1.0 - 2.0 * ((codes[:, None] >> bits[None, :]) & 1)
        count += int(np.count_nonzero(np.abs(signs @ d) >= obs - tol))
    return count / total


def holm(p) -> np.ndarray:
    """Holm step-down adjusted p-values, in input order."""
    p = np.asarray(p, dtype=np.float64)
    m = len(p)
    adj, running = np.empty(m), 0.0
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def bh(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, in input order."""
    p = np.asarray(p, dtype=np.float64)
    m = len(p)
    order = np.argsort(p, kind="stable")
    ranked = p[order] * m / np.arange(1, m + 1)
    adj = np.empty(m)
    adj[order] = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    return adj


def seed_bootstrap(d, n_boot: int = N_BOOT_SEEDS, seed: int = BOOT_SEED) -> dict:
    """Mean of per-seed values, 95 % percentile interval over resampled seeds, and the
    two-sided bootstrap p-value ``min(1, 2 (k + 1) / (B + 1))``, ``k`` the smaller count of
    resampled means at or below / at or above 0."""
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    k = min(int(np.sum(means <= 0)), int(np.sum(means >= 0)))
    return {"mean": float(d.mean()), "ci": [float(np.percentile(means, 2.5)),
                                            float(np.percentile(means, 97.5))],
            "p": min(1.0, 2.0 * (k + 1) / (n_boot + 1)), "n_positive": int((d > 0).sum()),
            "n": int(len(d))}


def mean_band(Y, n_boot: int = N_BOOT_BAND, seed: int = BOOT_SEED):
    """Seed-mean curve and its 95 % band over resampled seeds; ``Y`` is (seeds, budgets)."""
    Y = np.asarray(Y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = Y[rng.integers(0, Y.shape[0], size=(n_boot, Y.shape[0]))].mean(axis=1)
    return Y.mean(axis=0), np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def first_crossing(ns, ys, level: float) -> tuple[float | None, str | None]:
    """First budget at which the curve reaches ``level``, interpolated on log2 n.

    ``(n, None)`` for a crossing, ``(ns[0], "left")`` if the first budget already reaches
    it, ``(None, "right")`` if no budget does.
    """
    ns = np.asarray(ns, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if ys[0] >= level:
        return float(ns[0]), "left"
    for i in range(1, len(ns)):
        if ys[i] >= level:
            t = (level - ys[i - 1]) / (ys[i] - ys[i - 1])
            x = np.log2(ns[i - 1]) + t * (np.log2(ns[i]) - np.log2(ns[i - 1]))
            return float(2.0 ** x), None
    return None, "right"


def savings(ns, ref_y, tgt_y, ref_points=SAVINGS_REF_POINTS) -> dict:
    """Label savings of a target curve over a reference curve (both seed means).

    For each reference point ``n_j``, the level ``L_j`` is the reference score there and the
    ratio is ``n_ref(L_j) / n_tgt(L_j)`` (first crossings). The saving is the geometric mean
    of the ratios. Flags: ``"lower"`` (the target already reaches a level at its first
    budget: a lower bound), ``"upper"`` (the target never reaches it, ratio ``n_ref / n_max``;
    or the reference reaches it first: an upper bound), ``"undetermined"`` (both reach it at
    the first budget, ratio 1); one flag when the censored components agree, ``"mixed"``
    otherwise.
    """
    ns = [int(n) for n in ns]
    missing = [n for n in ref_points if n not in ns]
    if missing:
        raise ValueError(f"label savings need the budgets {list(ref_points)}; missing {missing}")
    comps = []
    for n_pt in ref_points:
        L = float(ref_y[ns.index(n_pt)])
        n_r, c_r = first_crossing(ns, ref_y, L)
        n_t, c_t = first_crossing(ns, tgt_y, L)
        if c_t == "right":
            comps.append({"level": L, "n_ref": n_r, "n_tgt": None, "ratio": n_r / ns[-1],
                          "flag": "upper"})
        elif c_t == "left" and c_r == "left":
            comps.append({"level": L, "n_ref": n_r, "n_tgt": n_t, "ratio": 1.0,
                          "flag": "undetermined"})
        else:
            flag = "lower" if c_t == "left" else "upper" if c_r == "left" else None
            comps.append({"level": L, "n_ref": n_r, "n_tgt": n_t, "ratio": n_r / n_t,
                          "flag": flag})
    flags = {c["flag"] for c in comps if c["flag"] is not None}
    flag = None if not flags else (flags.pop() if len(flags) == 1 else "mixed")
    value = float(np.exp(np.mean(np.log([c["ratio"] for c in comps]))))
    return {"savings": value, "flag": flag, "components": comps}
