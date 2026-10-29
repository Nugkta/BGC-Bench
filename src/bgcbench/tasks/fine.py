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


def rarest_label_strata(Y: np.ndarray, names: list[str]) -> np.ndarray:
    """Each row's rarest class (fewest positives in ``Y``, ties by class name); strata with
    a single row are pooled into one."""
    counts = (Y > 0).sum(axis=0)
    order = sorted(range(len(names)), key=lambda j: (int(counts[j]), names[j]))
    rank = np.empty(len(names), dtype=np.int64)
    rank[order] = np.arange(len(names))
    strata = np.array([names[js[np.argmin(rank[js])]] if len(js) else "" for js in
                       (np.flatnonzero(row > 0) for row in Y)], dtype=object)
    values, sizes = np.unique(strata.astype(str), return_counts=True)
    small = set(values[sizes < 2].tolist())
    if small and sum(int(n) for v, n in zip(values, sizes) if v in small) < 2:
        raise ValueError("a single row would form the pooled stratum")
    return np.array(["__pooled__" if s in small else s for s in strata], dtype=object)


def al_rows(ds: data.Dataset = data.MAIN):
    """Active learning on the random split, fold 4 as the test set; the validation rows are
    stratified on each training BGC's rarest class."""
    from ..al import Rows

    df = ds.split(SPLITS["R"])
    test = (df["fold"] == TEST_FOLD).to_numpy()
    Y = label_matrix(df["labels"], classes(ds))
    strata = np.full(len(df), "", dtype=object)
    strata[~test] = rarest_label_strata(Y[~test], classes(ds))
    return Rows(df["bgc_id"], test, strata=strata, table=df)


def al_readout(reps: dict, rows, ds: data.Dataset = data.MAIN):
    from ..al import MultiLabel

    Y = label_matrix(rows.table["labels"], classes(ds))
    return MultiLabel(rows.features(reps["bgc"]), Y, metric="weighted_macro_auroc")


class Fine(Task):
    name, metric, splits, threads = "fine", "weighted macro-AUROC", SPLITS, 1
    al_stream = 103

    def evaluate(self, reps, split, *, n_boot, device):
        b = reps["bgc"]
        return evaluate(b.ids, b.X, split, n_boot=n_boot, ds=self.data)

    def al_rows(self):
        return al_rows(self.data)

    def al_readout(self, reps, rows, *, device):
        return al_readout(reps, rows, self.data)
