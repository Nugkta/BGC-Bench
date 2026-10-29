"""Metrics and their 95 % bootstrap intervals.

Bootstraps draw ``n_boot`` resamples using ``numpy.random.default_rng(seed)`` and take the
2.5 / 97.5 percentiles. Classification, regression and accuracy resample test rows;
retrieval resamples queries within class-membership strata. Classification and regression
use ``nanpercentile`` to omit undefined resampled statistics.

Retrieval metrics and micro-AUPRC are exact expectations over a uniformly random order
within each group of tied scores.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def _ci(samples: np.ndarray, nan: bool) -> list[float]:
    if not nan:
        return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return [float(np.nanpercentile(samples, 2.5)), float(np.nanpercentile(samples, 97.5))]


def _bootstrap(stat, n: int, n_boot: int, seed: int, nan: bool) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = np.full(n_boot, np.nan)
    for b in range(n_boot):
        samples[b] = stat(rng.integers(0, n, size=n))
    return _ci(samples, nan)


# Classification
def macro_auroc_multilabel(Y: np.ndarray, S: np.ndarray) -> float:
    """Mean AUROC over the label columns of a (n, C) 0/1 matrix that have both values."""
    per_class = []
    for i in range(Y.shape[1]):
        col = Y[:, i]
        if col.sum() == 0 or col.sum() == len(col):
            continue
        per_class.append(float(roc_auc_score(col, S[:, i])))
    return float(np.mean(per_class)) if per_class else float("nan")


def macro_auroc_multiclass(y: np.ndarray, S: np.ndarray) -> float:
    """Mean one-vs-rest AUROC over the classes ``i`` (columns of ``S``) for which ``y == i``
    has both positives and negatives."""
    per_class = []
    for i in range(S.shape[1]):
        binary = (y == i).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            continue
        per_class.append(float(roc_auc_score(binary, S[:, i])))
    return float(np.mean(per_class)) if per_class else float("nan")


def auroc_with_ci(y: np.ndarray, S: np.ndarray, *, multilabel: bool, n_boot: int,
                  seed: int) -> dict:
    """Macro-AUROC (per label column if ``multilabel``, else one-vs-rest per class) with a
    95 % interval from resampled test rows."""
    fn = macro_auroc_multilabel if multilabel else macro_auroc_multiclass
    out = {"point": fn(y, S)}
    if n_boot > 0:
        out["ci"] = _bootstrap(lambda idx: fn(y[idx], S[idx]), len(y), n_boot, seed, nan=True)
    return out


def expected_average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision of one ranking, as an exact expectation over orderings within ties.

    A positive in a tie group of size ``g`` holding ``p`` positives, with ``a`` items and
    ``c`` positives strictly above it, has expected precision
    ``(1/g) sum_{r=1..g} (c + 1 + (r - 1)(p - 1)/(g - 1)) / (a + r)``; AP is the mean over
    positives. Ties are exact equality of the float64 scores.
    """
    y = np.asarray(y).ravel()
    s = np.asarray(s, dtype=np.float64).ravel()
    if y.shape != s.shape or not np.isin(y, (0, 1)).all():
        raise ValueError("expected 0/1 labels and scores of the same size")
    if not np.isfinite(s).all():
        raise ValueError("non-finite score")
    m = int(y.sum())
    if m == 0:
        raise ValueError("average precision needs at least one positive")
    order = np.argsort(-s, kind="stable")
    ss, ys = s[order], y[order].astype(np.int64)
    n = len(ss)
    starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    g = np.diff(np.r_[starts, n]).astype(np.float64)
    p = np.add.reduceat(ys, starts).astype(np.float64)
    c = np.r_[0.0, np.cumsum(p)[:-1]]
    a = starts.astype(np.float64)
    harm = np.add.reduceat(1.0 / np.arange(1, n + 1, dtype=np.float64), starts)
    has = p > 0
    g, p, c, a, harm = g[has], p[has], c[has], a[has], harm[has]
    beta = np.where(g > 1, (p - 1.0) / np.maximum(g - 1.0, 1.0), 0.0)
    e_prec = (g * beta + (c + 1.0 - beta * (a + 1.0)) * harm) / g
    return float(np.sum(p * e_prec) / m)


def multilabel_auroc(Y: np.ndarray, S: np.ndarray) -> dict:
    """Weighted macro-AUROC, macro-AUROC, micro-AUROC and micro-AUPRC of a (n, C) 0/1 matrix.

    The macro metrics average the per-class AUROC over the classes with both a positive and
    a negative row; the weighted one weights each class by its positives (scikit-learn's
    ``average="weighted"`` over those classes). The micro metrics pool every (row, class)
    pair; micro-AUPRC is :func:`expected_average_precision`.
    """
    auc, pos = [], []
    for i in range(Y.shape[1]):
        n_pos = Y[:, i].sum()
        if 0 < n_pos < len(Y):
            auc.append(float(roc_auc_score(Y[:, i], S[:, i])))
            pos.append(float(n_pos))
    y, s = Y.ravel(), S.ravel()
    nan = float("nan")
    return {
        "weighted_macro_auroc": float(np.dot(pos, auc) / np.sum(pos)) if auc else nan,
        "macro_auroc": float(np.mean(auc)) if auc else nan,
        "micro_auroc": float(roc_auc_score(y, s)) if 0 < y.sum() < y.size else nan,
        "micro_auprc": expected_average_precision(y, s) if y.sum() > 0 else nan,
    }


def multilabel_with_ci(Y: np.ndarray, S: np.ndarray, *, n_boot: int, seed: int) -> dict:
    """:func:`multilabel_auroc` with 95 % intervals.

    ``point`` / ``ci`` hold the weighted macro-AUROC; the other three metrics come with
    ``<name>_ci``, from the same resamples.
    """
    m = multilabel_auroc(Y, S)
    out = {"point": m.pop("weighted_macro_auroc"), **m}
    if n_boot > 0:
        rng = np.random.default_rng(seed)  # same draws as _bootstrap
        samples = {k: np.full(n_boot, np.nan) for k in ("weighted_macro_auroc", *m)}
        for b in range(n_boot):
            idx = rng.integers(0, len(Y), size=len(Y))
            for k, v in multilabel_auroc(Y[idx], S[idx]).items():
                samples[k][b] = v
        out["ci"] = _ci(samples.pop("weighted_macro_auroc"), nan=True)
        out.update({f"{k}_ci": _ci(v, nan=True) for k, v in samples.items()})
    return out


def top1_accuracy(y: np.ndarray, S: np.ndarray) -> float:
    """Fraction of rows whose highest score is the true class (ties go to the lowest index)."""
    return float(np.mean(np.argmax(S, axis=1) == np.asarray(y)))


def accuracy_with_ci(y: np.ndarray, S: np.ndarray, *, n_boot: int, seed: int) -> dict:
    """Top-1 accuracy with a 95 % interval from resampled test rows."""
    out = {"point": top1_accuracy(y, S)}
    if n_boot > 0:
        out["ci"] = _bootstrap(lambda idx: top1_accuracy(y[idx], S[idx]), len(y), n_boot,
                               seed, nan=False)
    return out


# Regression
def macro_spearman(Y: np.ndarray, P: np.ndarray) -> float:
    """Mean Spearman rho over non-constant target and prediction columns."""
    rhos = []
    for i in range(Y.shape[1]):
        yt, yp = Y[:, i], P[:, i]
        if np.std(yt) == 0 or np.std(yp) == 0:
            continue
        rho = float(spearmanr(yt, yp).statistic)
        if not np.isnan(rho):
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else float("nan")


def spearman_with_ci(Y: np.ndarray, P: np.ndarray, *, n_boot: int, seed: int) -> dict:
    """:func:`macro_spearman` with a 95 % interval from resampled test rows."""
    out = {"point": macro_spearman(Y, P)}
    if n_boot > 0:
        out["ci"] = _bootstrap(lambda idx: macro_spearman(Y[idx], P[idx]), len(Y), n_boot,
                               seed, nan=True)
    return out


# Retrieval (tie-aware)
def _comb_ratio(t: np.ndarray, g: np.ndarray, j: np.ndarray) -> np.ndarray:
    """C(t, j) / C(g, j) elementwise for 0 <= t <= g, as an exact product of ratios."""
    t = np.asarray(t, dtype=np.int64)
    g = np.asarray(g, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    out = np.ones(t.shape, dtype=np.float64)
    out[j > t] = 0.0
    need = (j >= 1) & (j <= t)
    if np.any(need):
        tt, gg, jj = t[need], g[need], j[need]
        i = np.arange(int(jj.max()), dtype=np.int64)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = (tt[:, None] - i[None, :]) / (gg[:, None] - i[None, :])
        f = np.where(i[None, :] < jj[:, None], f, 1.0)
        out[need] = np.prod(f, axis=1)
    return out


def expected_hit_at_k(scores: np.ndarray, positives: np.ndarray, k: int) -> np.ndarray:
    """Per query, the expected any-hit@k over uniform orderings of tied candidates.

    With ``a`` candidates scoring strictly above the best positive and a tie group of size
    ``g`` holding ``p`` positives at that score (``t = g - p`` negatives), the hit
    probability is 0 if ``k <= a`` and otherwise ``1 - C(t, j) / C(g, j)`` with
    ``j = min(k - a, g)``. Ties are exact equality of the float64 scores.
    """
    S = np.asarray(scores, dtype=np.float64)
    P = np.asarray(positives, dtype=bool)
    if S.ndim != 2 or P.shape != S.shape:
        raise ValueError(f"scores {S.shape} and positives {P.shape} must be equal 2-D shapes")
    if not P.any(axis=1).all():
        raise ValueError("every query needs at least one positive")
    if not np.isfinite(S).all():
        raise ValueError("non-finite score")
    n_q, n_c = S.shape
    order = np.argsort(-S, axis=1, kind="stable")
    Ss = np.take_along_axis(S, order, axis=1)
    Ps = np.take_along_axis(P, order, axis=1)

    pos = np.broadcast_to(np.arange(n_c, dtype=np.int64), (n_q, n_c))
    first = np.ones((n_q, n_c), dtype=bool)
    first[:, 1:] = Ss[:, 1:] != Ss[:, :-1]
    last = np.ones((n_q, n_c), dtype=bool)
    last[:, :-1] = Ss[:, :-1] != Ss[:, 1:]
    start = np.maximum.accumulate(np.where(first, pos, 0), axis=1)
    rev = np.maximum.accumulate(np.where(last[:, ::-1], pos, 0), axis=1)
    end = ((n_c - 1) - rev)[:, ::-1]
    g_all = end - start + 1
    cum_p = np.cumsum(Ps, axis=1, dtype=np.int64)
    before = np.where(start > 0, np.take_along_axis(cum_p, np.maximum(start - 1, 0), axis=1), 0)
    p_all = np.take_along_axis(cum_p, end, axis=1) - before

    rows = np.arange(n_q)
    f = np.argmax(Ps, axis=1)  # first positive in descending-score order
    a, g, p = start[rows, f], g_all[rows, f], p_all[rows, f]
    j = np.minimum(np.maximum(k - a, 0), g)
    return np.where(k <= a, 0.0, 1.0 - _comb_ratio(g - p, g, j))


def class_macro(values: np.ndarray, member: np.ndarray) -> float:
    """Mean over classes (columns of ``member``) of the per-class mean of ``values``."""
    values = np.asarray(values, dtype=np.float64)
    member = np.asarray(member, dtype=bool)
    counts = member.sum(axis=0)
    sums = values @ member.astype(np.float64)
    per_class = [float(sums[k] / counts[k]) for k in range(member.shape[1]) if counts[k] > 0]
    return float(np.mean(per_class))


def bootstrap_weights(member: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_q) resampling counts drawn within class-membership strata."""
    member = np.asarray(member, dtype=bool)
    n_q = member.shape[0]
    _, sid = np.unique(member, axis=0, return_inverse=True)
    sid = np.asarray(sid).reshape(-1)
    rng = np.random.default_rng(seed)
    W = np.zeros((n_boot, n_q), dtype=np.float64)
    for s in range(int(sid.max()) + 1):
        idx = np.flatnonzero(sid == s)
        n_s = len(idx)
        draws = rng.integers(0, n_s, size=(n_boot, n_s))
        flat = (draws + np.arange(n_boot, dtype=np.int64)[:, None] * n_s).ravel()
        W[:, idx] = np.bincount(flat, minlength=n_boot * n_s).reshape(n_boot, n_s)
    return W


def class_macro_ci(values: np.ndarray, member: np.ndarray, *, n_boot: int,
                   seed: int) -> list[float]:
    """95 % interval of :func:`class_macro`, resampling queries within their class."""
    if n_boot == 0:
        return [float("nan"), float("nan")]
    member = np.asarray(member, dtype=bool)
    W = bootstrap_weights(member, n_boot, seed)
    Mf = member.astype(np.float64)
    present = member.any(axis=0)
    den = (W @ Mf)[:, present]
    v = np.asarray(values, dtype=np.float64)
    num = ((W * v[None, :]) @ Mf)[:, present]
    macro = (num / den).mean(axis=1)
    return _ci(macro, nan=False)
