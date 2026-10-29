"""Probe behavior when user embeddings cover a subset of a split."""

import numpy as np
import pandas as pd
import pytest

import bgcbench
from bgcbench import data, metrics, probes
from bgcbench.tasks import chem, fine


@pytest.mark.parametrize("train_labels", [["A"] * 4, ["A", "", "A", ""]])
def test_fine_with_one_trainable_class(monkeypatch, train_labels):
    ids = np.array(["a", "b", "c", "d", "e", "f"])
    rows = pd.DataFrame({"bgc_id": ids, "split": ["train"] * 4 + ["test"] * 2,
                         "labels": train_labels + ["A", "B"]})
    monkeypatch.setattr(data.MAIN, "split", lambda _: rows)
    monkeypatch.setattr(fine, "classes", lambda ds: ["A", "B"])
    captured, score = [], metrics.multilabel_with_ci

    def record(Y, S, **kwargs):
        captured.append(S.copy())
        return score(Y, S, **kwargs)

    monkeypatch.setattr(metrics, "multilabel_with_ci", record)
    X = np.arange(12, dtype=np.float32).reshape(6, 2)
    out = fine.evaluate(ids, X, "partial", n_boot=0)
    assert np.isfinite(out["point"])
    assert out["n_classes"] == 2 and out["n_train"] == 4 and out["n_test"] == 2
    np.testing.assert_array_equal(captured[0][:, 1], 0)
    if all(train_labels):
        np.testing.assert_array_equal(captured[0][:, 0], 1)
    else:
        labels = np.array([s == "A" for s in train_labels])
        reference = probes.logreg().fit(X[:4], labels).predict_proba(X[4:])[:, 1]
        np.testing.assert_allclose(captured[0][:, 0], reference)


@pytest.mark.parametrize("task", ["coarse", "fine", "chem"])
@pytest.mark.parametrize("kept", ["a", "b"])
def test_empty_train_or_test_is_unavailable(monkeypatch, task, kept):
    rows = pd.DataFrame({"bgc_id": ["a", "b"], "split": ["train", "test"]})
    descriptors = pd.DataFrame({"bgc_id": ["a", "b"],
                                **{t: [0.0, 1.0] for t in chem.TARGETS}})
    monkeypatch.setattr(data.MAIN, "split", lambda _: rows)
    monkeypatch.setattr(data.MAIN, "read_csv", lambda _: descriptors)
    result = bgcbench.evaluate({kept: np.ones(2)}, tasks=task, n_boot=0, log=lambda *_: None)
    assert len(result.cells) == 3
    for cell in result.cells.values():
        assert cell["status"] == "na"
        assert "embedded training and test rows" in cell["note"]
