"""Tie-aware metrics are exact expectations over tie orderings: check them by enumeration."""

import itertools

import numpy as np
import pytest

from bgcbench import metrics


def enumerated_hit_at_k(scores, positives, k):
    """Mean any-hit@k over every ordering that sorts the candidates by descending score."""
    n = len(scores)
    hits = []
    for perm in itertools.permutations(range(n)):
        s = scores[list(perm)]
        if np.all(s[:-1] >= s[1:]):
            hits.append(float(positives[list(perm)][:k].any()))
    return np.mean(hits)


@pytest.mark.parametrize("seed", range(40))
def test_expected_hit_at_k_matches_enumeration(seed):
    rng = np.random.default_rng(seed)
    n_c = int(rng.integers(2, 8))
    S = rng.integers(0, 3, size=(3, n_c)).astype(np.float64)  # many ties
    P = rng.random((3, n_c)) < 0.3
    P[np.arange(3), rng.integers(0, n_c, size=3)] = True
    for k in range(1, n_c + 1):
        got = metrics.expected_hit_at_k(S, P, k)
        want = [enumerated_hit_at_k(S[q], P[q], k) for q in range(3)]
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_expected_hit_at_k_without_ties_is_plain_hit():
    S = np.array([[0.9, 0.1, 0.5, 0.3]])
    P = np.array([[False, False, False, True]])
    assert metrics.expected_hit_at_k(S, P, 2)[0] == 0.0
    assert metrics.expected_hit_at_k(S, P, 3)[0] == 1.0


def test_expected_hit_at_k_rejects_queries_without_positives():
    with pytest.raises(ValueError):
        metrics.expected_hit_at_k(np.zeros((1, 3)), np.zeros((1, 3), dtype=bool), 1)


def test_class_macro_averages_classes_not_queries():
    values = np.array([1.0, 1.0, 1.0, 0.0])
    member = np.array([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=bool)
    assert metrics.class_macro(values, member) == 0.5


def test_class_macro_ci_is_reproducible_and_brackets_the_point():
    rng = np.random.default_rng(0)
    values = rng.random(60)
    member = np.zeros((60, 3), dtype=bool)
    member[np.arange(60), rng.integers(0, 3, 60)] = True
    a = metrics.class_macro_ci(values, member, n_boot=200, seed=7)
    b = metrics.class_macro_ci(values, member, n_boot=200, seed=7)
    assert a == b
    assert a[0] <= metrics.class_macro(values, member) <= a[1]


def test_bootstrap_weights_resample_within_strata():
    member = np.array([[1, 0]] * 5 + [[0, 1]] * 3, dtype=bool)
    W = metrics.bootstrap_weights(member, 50, seed=1)
    assert (W[:, :5].sum(axis=1) == 5).all() and (W[:, 5:].sum(axis=1) == 3).all()


def test_classification_bootstrap_is_seeded():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, 80)
    S = rng.random((80, 3))
    a = metrics.auroc_with_ci(y, S, multilabel=False, n_boot=100, seed=42)
    b = metrics.auroc_with_ci(y, S, multilabel=False, n_boot=100, seed=42)
    assert a == b and a["ci"][0] <= a["point"] <= a["ci"][1]


def test_top1_accuracy():
    S = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    assert metrics.top1_accuracy(np.array([1, 0, 0]), S) == pytest.approx(2 / 3)


def enumerated_average_precision(y, s):
    """Mean AP over every ordering that sorts the items by descending score."""
    aps = []
    for perm in itertools.permutations(range(len(s))):
        if np.all(s[list(perm)][:-1] >= s[list(perm)][1:]):
            hits = y[list(perm)]
            prec = np.cumsum(hits) / np.arange(1, len(hits) + 1)
            aps.append(float((prec * hits).sum() / hits.sum()))
    return np.mean(aps)


@pytest.mark.parametrize("seed", range(40))
def test_expected_average_precision_matches_enumeration(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 8))
    s = rng.integers(0, 3, size=n).astype(np.float64)  # many ties
    y = (rng.random(n) < 0.4).astype(int)
    y[rng.integers(0, n)] = 1
    assert metrics.expected_average_precision(y, s) == pytest.approx(
        enumerated_average_precision(y, s), abs=1e-12)


def test_multilabel_auroc_matches_sklearn_on_the_defined_classes():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(5)
    Y = (rng.random((60, 6)) < 0.3).astype(np.float32)
    Y[:, 4] = 0  # no positive: left out of the macro metrics
    S = rng.random((60, 6))
    m = metrics.multilabel_auroc(Y, S)
    keep = [0, 1, 2, 3, 5]
    assert m["weighted_macro_auroc"] == pytest.approx(
        roc_auc_score(Y[:, keep], S[:, keep], average="weighted"), abs=1e-12)
    assert m["macro_auroc"] == pytest.approx(
        roc_auc_score(Y[:, keep], S[:, keep], average="macro"), abs=1e-12)
    assert m["micro_auroc"] == pytest.approx(roc_auc_score(Y.ravel(), S.ravel()), abs=1e-12)


def test_multilabel_bootstrap_is_seeded_and_brackets_the_point():
    rng = np.random.default_rng(4)
    Y = (rng.random((80, 5)) < 0.3).astype(np.float32)
    S = rng.random((80, 5)) + Y
    a = metrics.multilabel_with_ci(Y, S, n_boot=100, seed=42)
    b = metrics.multilabel_with_ci(Y, S, n_boot=100, seed=42)
    assert a == b
    for point, ci in [("point", "ci"), ("macro_auroc", "macro_auroc_ci"),
                      ("micro_auroc", "micro_auroc_ci"), ("micro_auprc", "micro_auprc_ci")]:
        assert a[ci][0] <= a[point] <= a[ci][1]
