"""Pieces of the retrieval protocol that must hold bit for bit."""

import numpy as np
import pandas as pd
import torch

from bgcbench.tasks import retrieval


def test_sampler_cycles_through_each_bgcs_products():
    items = [np.array([0]), np.array([1, 2, 3]), np.array([4, 5])]
    s = retrieval.RoundRobinSampler(items, seed=0)
    draws = np.stack([s.draw_epoch() for _ in range(6)])
    assert (draws[:, 0] == 0).all()
    for start in (0, 3):  # every product once per cycle
        assert sorted(draws[start:start + 3, 1]) == [1, 2, 3]
    for start in (0, 2, 4):
        assert sorted(draws[start:start + 2, 2]) == [4, 5]


def test_sampler_is_seeded_and_ignores_single_product_bgcs():
    a = retrieval.RoundRobinSampler([np.array([1, 2, 3]), np.array([4, 5])], seed=3)
    b = retrieval.RoundRobinSampler([np.array([9]), np.array([1, 2, 3]), np.array([7]),
                                     np.array([4, 5])], seed=3)
    for _ in range(5):
        da, db = a.draw_epoch(), b.draw_epoch()
        assert list(da) == [db[1], db[3]]


def test_masked_links_are_removed_from_both_softmaxes():
    torch.manual_seed(0)
    zb = torch.nn.functional.normalize(torch.randn(6, 4, dtype=torch.float64), dim=-1)
    zc = torch.nn.functional.normalize(torch.randn(6, 4, dtype=torch.float64), dim=-1)
    mask = torch.zeros(6, 6, dtype=torch.bool)
    mask[0, 3] = mask[3, 0] = mask[2, 5] = True
    scale = torch.tensor(2.0, dtype=torch.float64)
    got = retrieval.masked_info_nce(zb, zc, scale, mask)

    logits = scale * zb @ zc.T

    def ce(L, M):  # mean over rows of -log softmax at the diagonal, masked columns dropped
        terms = []
        for i in range(len(L)):
            keep = [j for j in range(len(L)) if j == i or not M[i, j]]
            terms.append(torch.logsumexp(L[i, keep], 0) - L[i, i])
        return torch.stack(terms).mean()

    want = (ce(logits, mask) + ce(logits.T, mask.T)) / 2
    torch.testing.assert_close(got, want, rtol=0, atol=1e-12)


def test_masked_info_nce_diagonal_is_never_masked():
    zb = torch.eye(3)
    m = torch.eye(3, dtype=torch.bool)
    one = torch.tensor(1.0)
    no_mask = retrieval.masked_info_nce(zb, zb, one, torch.zeros(3, 3, dtype=torch.bool))
    assert torch.equal(retrieval.masked_info_nce(zb, zb, one, m), no_mask)


def test_identical_vectors_get_identical_scores():
    rng = np.random.default_rng(0)
    q = rng.standard_normal((5, 8)).astype(np.float32)
    q[3] = q[1]
    c = rng.standard_normal((7, 8)).astype(np.float32)
    c[6] = c[2]
    S = retrieval._scores_unique(q, c)
    assert S.dtype == np.float64
    assert (S[3] == S[1]).all() and (S[:, 6] == S[:, 2]).all()


def test_projection_keeps_duplicate_rows_identical():
    head = retrieval.ProjectionHead(8, 4)
    X = np.random.default_rng(1).standard_normal((6, 8)).astype(np.float32)
    X[5] = X[0]
    Z = retrieval._project_unique(head.eval(), X)
    assert (Z[5] == Z[0]).all()


def toy_universe(n_per_class: int = 15, classes=("A", "B", "C")) -> retrieval.Universe:
    """One product per BGC, random vectors: enough rows for a fold per held-out class."""
    rng = np.random.default_rng(0)
    labels = np.repeat(np.array(classes), n_per_class)
    n = len(labels)
    split_df = pd.DataFrame({"bgc_id": [f"B{i}" for i in range(n)], "primary_class": labels})
    return retrieval.Universe(split_df, rng.standard_normal((n, 6)).astype(np.float32), labels,
                              [np.array([i]) for i in range(n)],
                              rng.standard_normal((n, 5)).astype(np.float32))


def test_class_holdout_trains_one_head_per_class_and_pools_the_interval():
    u = toy_universe()
    a = retrieval.evaluate_seed(u, "t3_class_holdout", 0, n_boot=50)
    assert a == retrieval.evaluate_seed(u, "t3_class_holdout", 0, n_boot=50)
    assert a["n_train"] == [30, 30, 30] and a["n_test"] == [15, 15, 15]
    assert 0 <= a["ci"][0] <= a["ci"][1] <= 1 and 0 <= a["point"] <= 1
