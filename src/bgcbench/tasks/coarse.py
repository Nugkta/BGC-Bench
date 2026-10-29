"""Coarse classes: six MIBiG 4.0 biosynthetic classes, multi-label, macro-AUROC.

Probe: one standardised logistic regression per class. Rows are the split's BGCs that have
an embedding, in split order; the test rows are scored once and bootstrapped.
"""

from __future__ import annotations

import numpy as np

from .. import data, metrics, probes
from .base import Task

CLASSES = ["NRP", "Other", "Polyketide", "RiPP", "Saccharide", "Terpene"]
SPLITS = {"R": "t1ml_random", "S": "t1ml_scaffold_holdout", "G": "t1ml_gcf_holdout_c07"}


def evaluate(ids: np.ndarray, X: np.ndarray, split: str, *, n_boot: int = 500,
             seed: int = 42, ds: data.Dataset = data.MAIN) -> dict:
    """One coarse column: BGC vectors ``(ids, X)`` scored on ``split`` of ``ds``."""
    Xa, df, coverage = data.align(ids, X, ds.split(split))
    train = (df["split"] == "train").to_numpy()
    test = (df["split"] == "test").to_numpy()
    data.require_train_test(train, test)
    Y = df[CLASSES].to_numpy().astype(np.float32)

    clf = probes.ovr_logreg().fit(Xa[train], Y[train])
    scores = clf.predict_proba(Xa[test])
    out = metrics.auroc_with_ci(Y[test], scores, multilabel=True, n_boot=n_boot, seed=seed)
    return {**out, "n_train": int(train.sum()), "n_test": int(test.sum()), **coverage}


def al_rows(ds: data.Dataset = data.MAIN):
    """Active learning on the random split: its BGCs with Pfam inputs; the validation rows
    are stratified on each BGC's first class."""
    from ..al import Rows, design

    df = ds.split(SPLITS["R"])
    df = df[df["bgc_id"].isin(design.pfam_universe(ds))].reset_index(drop=True)
    strata = np.asarray(CLASSES)[df[CLASSES].to_numpy().argmax(axis=1)]
    return Rows(df["bgc_id"], (df["split"] == "test").to_numpy(), strata=strata, table=df)


def al_readout(reps: dict, rows):
    from ..al import MultiLabel

    Y = rows.table[CLASSES].to_numpy().astype(np.float32)
    return MultiLabel(rows.features(reps["bgc"]), Y, metric="macro_auroc")


class Coarse(Task):
    name, metric, splits, threads = "coarse", "macro-AUROC", SPLITS, 8
    al_stream = 101

    def evaluate(self, reps, split, *, n_boot, device):
        b = reps["bgc"]
        return evaluate(b.ids, b.X, split, n_boot=n_boot, ds=self.data)

    def al_rows(self):
        return al_rows(self.data)

    def al_readout(self, reps, rows, *, device):
        return al_readout(reps, rows)
