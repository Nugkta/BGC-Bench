"""Fine classes: the NPClassifier classes of a cluster's products, multi-label, 48 classes.

A BGC's labels are the NPClassifier classes of all its products, restricted to the 48
classes held by at least 10 BGCs (``tasks/t2ml_labels.csv``). Every split trains the same
48 classes. Probe: one standardised logistic regression per class with a training positive;
other classes score 0. Rows follow split order and include BGCs with embeddings. The random
column uses fold 4 of the stratified 5-fold file as the test set.

Metric: macro-AUROC weighted by each class's test positives, over the classes with a test
positive and a test negative. The cell also holds micro-AUROC and micro-AUPRC over all
(BGC, class) pairs and the unweighted macro-AUROC, with intervals from the same resamples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import data, metrics, probes
from .base import Task

SPLITS = {"R": "t2ml_stratified_5fold", "S": "t2ml_scaffold_holdout",
          "G": "t2ml_gcf_c07_holdout"}
TEST_FOLD = 4
SEP = "; "


def classes(ds: data.Dataset = data.MAIN) -> list[str]:
    """The label space: every class of an enrolled BGC, sorted (48)."""
    table = ds.read_csv("tasks/t2ml_labels.csv")
    return sorted({c for s in table.loc[table["enrolled"], "labels"] for c in s.split(SEP)})


def label_matrix(labels: pd.Series, names: list[str]) -> np.ndarray:
    """(n, C) 0/1 float32 matrix; an empty label cell gives an all-zero row."""
    index = {c: i for i, c in enumerate(names)}
    Y = np.zeros((len(labels), len(names)), dtype=np.float32)
    for r, s in enumerate(labels):
        if isinstance(s, str) and s:
            Y[r, [index[c] for c in s.split(SEP)]] = 1
    return Y


def evaluate(ids: np.ndarray, X: np.ndarray, split: str, *, n_boot: int = 500,
             seed: int = 42, ds: data.Dataset = data.MAIN) -> dict:
    """One fine column: BGC vectors ``(ids, X)`` scored on ``split`` of ``ds``."""
    Xa, df, coverage = data.align(ids, X, ds.split(split))
    if "fold" in df.columns:
        train, test = (df["fold"] != TEST_FOLD).to_numpy(), (df["fold"] == TEST_FOLD).to_numpy()
    else:
        train, test = (df["split"] == "train").to_numpy(), (df["split"] == "test").to_numpy()
    data.require_train_test(train, test)
    names = classes(ds)
    Y = label_matrix(df["labels"], names)
    fit = np.flatnonzero(Y[train].sum(axis=0) > 0)

    scores = np.zeros((int(test.sum()), len(names)))
    if len(fit) == 1 and Y[train, fit[0]].all():
        scores[:, fit] = 1.0
    elif len(fit):
        clf = probes.ovr_logreg().fit(Xa[train], Y[train][:, fit])
        proba = clf.predict_proba(Xa[test])
        # sklearn returns [negative, positive] probabilities for a single label.
        scores[:, fit] = proba[:, 1:2] if len(fit) == 1 else proba
    out = metrics.multilabel_with_ci(Y[test], scores, n_boot=n_boot, seed=seed)
    return {**out, "n_train": int(train.sum()), "n_test": int(test.sum()),
            "n_classes": len(names), **coverage}


class Fine(Task):
    name, metric, splits, threads = "fine", "weighted macro-AUROC", SPLITS, 1

    def evaluate(self, reps, split, *, n_boot, device):
        b = reps["bgc"]
        return evaluate(b.ids, b.X, split, n_boot=n_boot, ds=self.data)
