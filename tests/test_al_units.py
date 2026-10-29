"""Active learning without the dataset: budgets, protocols, partitions, strategies, readouts
and statistics."""

import numpy as np
import pytest

from bgcbench import al
from bgcbench.al import design as D
from bgcbench.al import readouts as R
from bgcbench.al import stats
from bgcbench.al import strategies as S


@pytest.fixture
def strategy_registry():
    saved = dict(S.REGISTRY)
    yield
    S.REGISTRY.clear()
    S.REGISTRY.update(saved)


def test_budget_grid_grows_by_sqrt2_and_ends_at_the_pool():
    assert D.budget_grid(1553) == [16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512, 724, 1024,
                                   1448, 1553]
    assert D.budget_grid(1553, max_budget=256)[-1] == 256
    assert D.budget_grid(40) == [16, 23, 32, 40]
    with pytest.raises(ValueError):
        D.budget_grid(16)


def test_protocol_presets_and_overrides():
    quick, paper = al.protocol("quick"), al.protocol("paper")
    assert quick.seeds == (1, 2, 3, 4, 5) and quick.max_budget == 256
    assert paper.seeds == tuple(range(1, 21)) and paper.max_budget is None
    assert paper.selection_seeds == tuple(range(1, 11))
    assert al.protocol("paper", seeds=range(1, 21)) is paper
    custom = al.protocol("paper", seeds=[3, 12])
    assert custom.name == "custom" and custom.selection_seeds == (3,)
    assert custom.report_seeds == (12,)
    assert al.protocol("quick", max_budget=None).name == "custom"
    assert al.parse_seeds("1-3,8") == (1, 2, 3, 8)
    with pytest.raises(ValueError):
        al.parse_seeds("1-")
    with pytest.raises(ValueError):
        al.protocol("slow")


def toy_rows(n=200, groups=False):
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, n).astype(str)
    return al.Rows([f"B{i}" for i in range(n)], np.arange(n) % 5 == 0, strata=labels,
                   groups=np.arange(n) // 4 if groups else None)


def test_partition_is_fixed_and_groups_stay_together():
    rows = toy_rows()
    a, b = al.partition(rows, 7), al.partition(rows, 7)
    assert a.digest() == b.digest()
    assert len(a.val) == round(0.2 * 160) and len(a.pool) + len(a.val) == 160
    assert not set(a.pool) & set(a.val) and not set(a.pool) & set(a.test)
    assert np.array_equal(a.permutation(3), b.permutation(3))
    assert not np.array_equal(a.permutation(3), al.partition(rows, 8).permutation(3))
    g = al.partition(toy_rows(groups=True), 7)
    groups = np.arange(200) // 4
    assert not set(groups[g.pool]) & set(groups[g.val])


def test_rows_check_their_inputs():
    with pytest.raises(ValueError, match="unique"):
        al.Rows(["a", "a", "b"], [True, False, False])
    with pytest.raises(ValueError, match="both test and training"):
        al.Rows(["a", "b"], [True, True])


def test_default_stream_is_stable():
    assert D.default_stream("my_task") == D.default_stream("my_task")
    assert D.default_stream("my_task") != D.default_stream("other")
    assert D.default_stream("my_task") >= 10_000


def test_typiclust_picks_typical_points_of_unlabelled_clusters():
    rng = np.random.default_rng(0)
    centres = np.array([[0, 0], [10, 0], [0, 10]], dtype=float)
    Z = np.concatenate([c + rng.normal(0, 0.5, (40, 2)) for c in centres])
    labelled = np.array([0])
    picks, info = S.typiclust_select(Z, labelled, 2, seed=1)
    assert info["n_clusters"] == 3
    assert {int(p) // 40 for p in picks} == {1, 2}      # the two clusters without labels
    again, _ = S.typiclust_select(Z, labelled, 2, seed=1)
    assert np.array_equal(picks, again)


def test_kmeanspp_starts_from_the_largest_gradient_and_is_reproducible():
    rng = np.random.default_rng(0)
    E, X = rng.standard_normal((30, 3)), rng.standard_normal((30, 4))
    emb = R.GradEmbedding(E=E, X=X)
    dense = emb.dense()
    assert np.allclose(emb.sqnorm(), (dense ** 2).sum(axis=1))
    assert np.allclose(emb.dots(5), dense @ dense[5])
    picks, _ = S.kmeanspp_seeding(emb, 6, np.random.default_rng([1, 1001]))
    assert picks[0] == int(np.argmax((dense ** 2).sum(axis=1)))
    assert len(set(picks.tolist())) == 6
    again, _ = S.kmeanspp_seeding(emb, 6, np.random.default_rng([1, 1001]))
    assert np.array_equal(picks, again)
    uniform, info = S.kmeanspp_seeding(np.zeros((5, 2)), 3, np.random.default_rng(0))
    assert info["n_uniform_draws"] == 2 and len(set(uniform.tolist())) == 3


def test_strategy_registry(strategy_registry):
    assert S.list_strategies()[:3] == ["random", "typiclust", "badge"]
    assert S.get_strategy("badge").needs_gradients

    @al.register_strategy("first")
    def first(ctx, k):
        """The lowest unlabelled pool positions."""
        return ctx.unlabelled[:k]

    assert S.get_strategy("first").description == "The lowest unlabelled pool positions."
    assert [s.name for s in S.select(["first", "random"])] == ["random", "first"]
    with pytest.raises(ValueError, match="built-in"):
        al.register_strategy("random", first)
    with pytest.raises(ValueError, match="unknown strategy"):
        S.select(["nope"])


def test_acquire_rejects_bad_picks():
    ctx = S.Context(None, None, np.arange(10), np.array([0, 1]), np.arange(10), 1, 1)
    for bad in ([2, 2], [0, 3], [3, 11], [3]):
        with pytest.raises(ValueError):
            S.acquire(S.Strategy("bad", lambda c, k, b=bad: np.array(b)), ctx, 2)
    picks, info = S.acquire(S.Strategy("ok", lambda c, k: (np.array([5, 3]), {"x": 1})), ctx, 2)
    assert picks.tolist() == [5, 3] and info == {"x": 1}


def test_readouts_survive_constant_labels():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 5)).astype(np.float32)
    Y = np.zeros((40, 3), dtype=np.float32)
    Y[::2, 0] = 1
    ml = R.MultiLabel(X, Y)
    fit = ml.fit(np.arange(20), 1)
    P = ml.predict(fit, np.arange(20, 40))
    assert np.all(P[:, 1:] == 0) and P.shape == (20, 3)
    y = np.array(["a"] * 30 + ["b"] * 10)
    clf = R.Classifier(X, y, classes=["a", "b", "c"])
    fit = clf.fit(np.arange(10), 1)                 # only class "a"
    assert np.all(clf.predict(fit, np.arange(30, 40))[:, 0] == 1)
    assert clf.gradients(fit, np.arange(5)).dim == 1
    reg = R.Regressor(X, np.c_[np.arange(40.0), np.ones(40)])
    with pytest.raises(ValueError, match="constant"):
        reg.score(reg.predict(reg.fit(np.arange(20), 1), np.arange(20, 40)), np.arange(20, 40))
    assert not reg.has_gradients and ml.has_gradients


def test_mean_spearman_scores_a_constant_prediction_as_zero():
    Y = np.c_[np.arange(10.0), np.arange(10.0)[::-1]]
    P = np.c_[np.arange(10.0), np.ones(10)]
    assert R.mean_spearman(Y, P) == pytest.approx(0.5)


def test_aulc_is_a_mean_score_on_a_log2_axis():
    ns = [16, 32, 64, 128, 256, 512]
    assert stats.aulc_low(ns, np.full(6, 0.7)) == pytest.approx(0.7)
    ys = np.log2(ns)
    assert stats.aulc_low(ns, ys) == pytest.approx((4 + 8) / 2)
    assert stats.aulc(ns, ys).shape == ()


def test_tests_and_corrections():
    assert stats.sign_flip_p(np.ones(5)) == pytest.approx(2 / 32)
    assert stats.sign_flip_p([1.0, -1.0]) == 1.0
    assert stats.holm([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])
    assert stats.bh([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.04, 0.04])
    b = stats.seed_bootstrap(np.ones(10))
    assert b["mean"] == 1 and b["p"] == pytest.approx(2 / 10_001) and b["ci"] == [1.0, 1.0]
    mean, lo, hi = stats.mean_band(np.array([[0.0, 1.0], [1.0, 3.0]]))
    assert mean.tolist() == [0.5, 2.0] and np.all(lo <= mean) and np.all(mean <= hi)


def test_label_savings():
    ns = [16, 32, 64, 128, 256]
    ref = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    # Levels 0.2, 0.3, 0.4: reached at n = 2^(4 + 1/3), 32 and 64, so the ratios are
    # 2^(2/3), 2 and 2.
    tgt = np.array([0.15, 0.3, 0.4, 0.5, 0.6])
    s = stats.savings(ns, ref, tgt)
    assert s["savings"] == pytest.approx(2 ** (8 / 9)) and s["flag"] is None
    assert stats.savings(ns, ref, ref - 1)["flag"] == "upper"
    assert stats.savings(ns, ref, ref + 1)["flag"] == "lower"
    with pytest.raises(ValueError, match="budgets"):
        stats.savings([16, 45, 64], ref[:3], tgt[:3])
