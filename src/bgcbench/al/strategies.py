"""Acquisition strategies and their registry.

A strategy is a function ``fn(ctx, k)`` that returns ``k`` distinct unlabelled pool
positions (indices into the pool, in pick order), optionally with a dict of details:
``(picks, info)``. :class:`Context` gives it the round's state. Register new strategies
with :func:`register_strategy`; one that needs BADGE-style gradients is skipped ("na") on
tasks whose readout has none.

Built-in strategies:

``random``     the continuation of the seed's pool permutation, so every encoder labels the
               same rows
``typiclust``  TypiClust (Hacohen et al., 2022): cluster the pool (standardised vectors,
               50 principal components) into ``min(labelled + k, pool / 10)`` clusters; visit
               clusters with the fewest labels first, taking the most typical unlabelled
               member (inverse mean squared distance to its 20 nearest neighbours)
``badge``      BADGE (Ash et al., 2020): k-means++ seeding on the readout's gradient
               embeddings, starting from the largest gradient
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .design import BADGE_STREAM

TYPICLUST_PCA_DIM = 50
TYPICLUST_KNN = 20
TYPICLUST_MIN_CLUSTER = 3          # smaller clusters are visited only once the rest run out
TYPICLUST_POOL_PER_CLUSTER = 10    # at most one cluster per 10 pool rows
TYPICLUST_KMEANS_MAX = 50          # KMeans up to 50 clusters, MiniBatchKMeans above
BADGE_D2_RTOL = 1e-10              # D^2 below this share of ||g_i||^2 + ||g_c||^2 is rounding


@dataclass
class Context:
    """One acquisition round.

    ``readout`` and ``fitted``: the task's readout and the model fitted on the labelled
    rows. ``pool``: the pool's row indices (into the task's AL rows). ``labelled``: sorted
    pool positions labelled so far. ``perm``: the seed's pool permutation. ``seed``: the
    learning curve's seed. ``round``: the index of the budget being acquired for (1 for the
    first acquisition). ``cache``: a dict kept for the whole learning curve.
    """

    readout: object
    fitted: object
    pool: np.ndarray
    labelled: np.ndarray
    perm: np.ndarray
    seed: int
    round: int
    cache: dict = field(default_factory=dict)

    @property
    def pool_features(self) -> np.ndarray:
        """The readout's vectors of the pool rows."""
        return self.readout.features[self.pool]

    @property
    def unlabelled(self) -> np.ndarray:
        """Sorted unlabelled pool positions."""
        return np.setdiff1d(np.arange(len(self.pool)), self.labelled)

    def rng(self, stream: int) -> np.random.Generator:
        """A generator for this seed and ``stream``."""
        return np.random.default_rng([int(self.seed), int(stream)])


@dataclass(frozen=True)
class Strategy:
    name: str
    fn: Callable
    needs_gradients: bool = False
    description: str = ""


REGISTRY: dict[str, Strategy] = {}
BUILTIN = ("random", "typiclust", "badge")


def register_strategy(name: str, fn: Callable | None = None, *, needs_gradients: bool = False,
                      description: str = "", replace: bool = False):
    """Register ``fn(ctx, k)`` under ``name``; works as a decorator without ``fn``.

    ``needs_gradients``: the strategy calls ``ctx.readout.gradients`` and is skipped on
    tasks whose readout has none. The built-in strategies cannot be replaced.
    """
    def add(f: Callable) -> Callable:
        if not name or not isinstance(name, str):
            raise ValueError("a strategy needs a name")
        current = REGISTRY.get(name)
        if current is not None and current.fn is not f:
            if name in BUILTIN:
                raise ValueError(f"{name!r} is a built-in strategy and cannot be replaced")
            if not replace and _qualname(current.fn) != _qualname(f):
                raise ValueError(f"strategy {name!r} is already registered by "
                                 f"{_qualname(current.fn)}; pass replace=True to override")
        doc = description or (f.__doc__ or "").strip().split("\n")[0]
        REGISTRY[name] = Strategy(name, f, needs_gradients, doc)
        return f

    return add if fn is None else add(fn)


def _qualname(fn: Callable) -> str:
    return f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', repr(fn))}"


def get_strategy(name: str) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; registered: {', '.join(REGISTRY)}")
    return REGISTRY[name]


def list_strategies() -> list[str]:
    """Names of the registered strategies, built-in ones first."""
    return list(REGISTRY)


def select(strategies=None) -> list[Strategy]:
    """Resolve names in registry order (``None``: every registered strategy)."""
    if strategies is None:
        return list(REGISTRY.values())
    if isinstance(strategies, str):
        strategies = [strategies]
    unknown = sorted(set(strategies) - set(REGISTRY))
    if unknown:
        raise ValueError(f"unknown strategy(s) {unknown}; choose from {list(REGISTRY)}")
    return [s for name, s in REGISTRY.items() if name in set(strategies)]


def acquire(strategy: Strategy, ctx: Context, k: int) -> tuple[np.ndarray, dict]:
    """Run a strategy and check that it returned ``k`` distinct unlabelled positions."""
    out = strategy.fn(ctx, k)
    picks, info = out if isinstance(out, tuple) else (out, {})
    picks = np.asarray(picks, dtype=np.int64).reshape(-1)
    if len(picks) != k:
        raise ValueError(f"strategy {strategy.name!r} returned {len(picks)} picks, expected {k}")
    if len(set(picks.tolist())) != k:
        raise ValueError(f"strategy {strategy.name!r} returned duplicate picks")
    if picks.min() < 0 or picks.max() >= len(ctx.pool):
        raise ValueError(f"strategy {strategy.name!r} returned positions outside the pool")
    if np.isin(picks, ctx.labelled).any():
        raise ValueError(f"strategy {strategy.name!r} picked labelled rows")
    return picks, dict(info or {})


# Random
def random(ctx: Context, k: int) -> np.ndarray:
    """The next k positions of the seed's pool permutation."""
    n_have = len(ctx.labelled)
    if n_have + k > len(ctx.perm):
        raise ValueError(f"cannot take {k} more from a pool of {len(ctx.perm)}")
    return ctx.perm[n_have:n_have + k].copy()


# TypiClust
def typiclust_features(X_pool: np.ndarray, seed: int = 0) -> np.ndarray:
    """Standardised pool vectors reduced to 50 principal components (fitted label-free)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(np.asarray(X_pool, dtype=np.float64))
    n_comp = min(TYPICLUST_PCA_DIM, Xs.shape[0], Xs.shape[1])
    return PCA(n_components=n_comp, svd_solver="full", random_state=seed).fit_transform(Xs)


def typiclust_n_clusters(n_labelled: int, k: int, pool_size: int) -> int:
    return int(min(n_labelled + k, pool_size // TYPICLUST_POOL_PER_CLUSTER))


def _cluster(Z: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.exceptions import ConvergenceWarning

    if n_clusters <= TYPICLUST_KMEANS_MAX:
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    else:
        km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=5000, n_init=3, random_state=seed)
    with warnings.catch_warnings():
        # Duplicate vectors (e.g. A-domains sharing a parent-BGC vector) can leave fewer
        # distinct clusters.
        warnings.simplefilter("ignore", ConvergenceWarning)
        return km.fit_predict(Z).astype(np.int64)


def typicality(Z: np.ndarray, k_nn: int) -> np.ndarray:
    """1 / (mean squared distance to the ``k_nn`` nearest other points + 1e-5)."""
    Z = np.asarray(Z, dtype=np.float64)
    sq = np.sum(Z * Z, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T), 0.0)
    np.fill_diagonal(d2, np.inf)
    return 1.0 / (np.sort(d2, axis=1)[:, :k_nn].mean(axis=1) + 1e-5)


def _cluster_order(ids: np.ndarray, sizes: np.ndarray, lab_counts: np.ndarray) -> np.ndarray:
    """Cluster ids by labelled count (ascending), then size (descending), then id."""
    return ids[np.lexsort((ids, -sizes, lab_counts))]


def typiclust_select(Z: np.ndarray, labelled: np.ndarray, k: int, seed: int
                     ) -> tuple[np.ndarray, dict]:
    """TypiClust picks over the pool's reduced vectors ``Z``.

    Cluster the whole pool; order the clusters of at least 3 points by labelled count, size
    and id; cycle through them, each time taking the unlabelled member of highest
    typicality (``min(20, m // 2)`` neighbours among the cluster's ``m`` unlabelled members;
    ties: lowest position), skipping exhausted clusters. Once every cluster is exhausted, the
    small clusters are visited in the same way.
    """
    P = Z.shape[0]
    labelled = np.asarray(labelled, dtype=np.int64)
    if k <= 0:
        return np.zeros(0, dtype=np.int64), {}
    if len(labelled) + k > P:
        raise ValueError(f"typiclust: {k} picks exceed the {P - len(labelled)} unlabelled rows")
    n_clusters = typiclust_n_clusters(len(labelled), k, P)
    labels = _cluster(Z, n_clusters, seed)
    taken = np.zeros(P, dtype=bool)
    taken[labelled] = True
    ids, sizes = np.unique(labels, return_counts=True)
    lab_counts = np.bincount(labels[labelled], minlength=n_clusters)[ids]
    keep = sizes >= TYPICLUST_MIN_CLUSTER
    kept = _cluster_order(ids[keep], sizes[keep], lab_counts[keep])
    small = _cluster_order(ids[~keep], sizes[~keep], lab_counts[~keep])
    picks: list[int] = []
    n_fallback = 0
    for phase, clusters in (("kept", kept), ("small", small)):
        if len(clusters) == 0:
            continue
        alive = np.array([np.any((labels == c) & ~taken) for c in clusters])
        i = 0
        while len(picks) < k and alive.any():
            j = i % len(clusters)
            i += 1
            if not alive[j]:
                continue
            members = np.flatnonzero((labels == clusters[j]) & ~taken)
            if members.size == 1:
                pick = int(members[0])
                alive[j] = False
            else:
                typ = typicality(Z[members], min(TYPICLUST_KNN, members.size // 2))
                pick = int(members[np.lexsort((members, -typ))[0]])
            picks.append(pick)
            taken[pick] = True
            n_fallback += phase == "small"
        if len(picks) >= k:
            break
    if len(picks) != k:
        raise RuntimeError(f"typiclust: {len(picks)} of {k} picks after both cluster passes")
    return np.asarray(picks, dtype=np.int64), {"n_clusters": int(n_clusters),
                                               "n_kept_clusters": int(len(kept)),
                                               "n_fallback_picks": int(n_fallback)}


def typiclust(ctx: Context, k: int):
    """TypiClust (Hacohen et al., 2022) on the encoder's vectors."""
    if "typiclust_Z" not in ctx.cache:
        ctx.cache["typiclust_Z"] = typiclust_features(ctx.pool_features, seed=0)
    return typiclust_select(ctx.cache["typiclust_Z"], ctx.labelled, k, ctx.seed)


# BADGE
def kmeanspp_seeding(emb, k: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """``k`` distinct rows of a gradient embedding by k-means++ seeding.

    The first centre is the row of largest norm (lowest index on ties); each next one is
    drawn with probability proportional to the squared distance to the nearest centre. If
    every unchosen row is at distance 0, the draw is uniform over them.
    """
    from .readouts import GradEmbedding

    if isinstance(emb, np.ndarray):
        emb = GradEmbedding(E=emb)
    n = emb.n
    if k <= 0:
        return np.zeros(0, dtype=np.int64), {"n_uniform_draws": 0}
    if k > n:
        raise ValueError(f"k = {k} exceeds the {n} candidate rows")
    sq = emb.sqnorm()

    def d2_to(c: int) -> np.ndarray:
        d2 = sq + sq[c] - 2.0 * emb.dots(c)
        d2[d2 <= BADGE_D2_RTOL * (sq + sq[c])] = 0.0
        return np.maximum(d2, 0.0)

    first = int(np.argmax(sq))
    chosen = [first]
    is_chosen = np.zeros(n, dtype=bool)
    is_chosen[first] = True
    d2 = d2_to(first)
    n_uniform = 0
    while len(chosen) < k:
        w = np.where(is_chosen, 0.0, d2)
        total = w.sum()
        if total > 0 and np.isfinite(total):
            nxt = int(rng.choice(n, p=w / total))
        else:
            nxt = int(rng.choice(np.flatnonzero(~is_chosen)))
            n_uniform += 1
        chosen.append(nxt)
        is_chosen[nxt] = True
        d2 = np.minimum(d2, d2_to(nxt))
    return np.asarray(chosen, dtype=np.int64), {"n_uniform_draws": int(n_uniform)}


def badge(ctx: Context, k: int):
    """BADGE (Ash et al., 2020) on the readout's gradient embeddings."""
    unl = ctx.unlabelled
    emb = ctx.readout.gradients(ctx.fitted, ctx.pool[unl])
    picks, info = kmeanspp_seeding(emb, k, ctx.rng(BADGE_STREAM + ctx.round))
    return unl[picks], {**info, "grad_dim": emb.dim}


register_strategy("random", random)
register_strategy("typiclust", typiclust)
register_strategy("badge", badge, needs_gradients=True)
