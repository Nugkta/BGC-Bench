"""A-domain substrate: the substrate (an amino or aryl acid) an NRPS adenylation domain
activates, top-1 accuracy.

Scored per A-domain, not per BGC: 28 substrates seen at least ten times ("level B"). Each
encoder supplies one vector per A-domain (``domain_id``), from the domain's own sequence
or, for Pfam-token models, the token the domain occupies; see ``bgcbench.encoders``.
Both columns require vectors for every domain in their split.

5-fold: folds are grouped by BGC, each domain is predicted once out of fold. GCF: one
holdout of whole BiG-SCAPE families (c = 0.7). The cell's ``majority`` is the share of the
most frequent substrate among the scored domains (from their own labels), a reference for
the accuracy.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import LabelEncoder

from .. import data, metrics, probes
from .base import Task

SPLITS = {"F": "t5_bgc_grouped_5fold", "G": "t5_gcf_holdout_c07"}
EXPECTED_SCORED = {"t5_bgc_grouped_5fold": 2027, "t5_gcf_holdout_c07": 292}


def _predict_full(clf, X: np.ndarray, n_classes: int) -> np.ndarray:
    """predict_proba scattered back to all classes (a rare substrate can miss a fold)."""
    proba = clf.predict_proba(X)
    if proba.shape[1] == n_classes:
        return proba
    full = np.zeros((proba.shape[0], n_classes), dtype=proba.dtype)
    full[:, np.asarray(clf[-1].classes_, dtype=int)] = proba
    return full


def evaluate(domain_ids: np.ndarray, X: np.ndarray, split: str, *, n_boot: int = 500,
             seed: int = 42, ds: data.Dataset = data.MAIN) -> dict:
    """One substrate column: A-domain vectors ``(domain_ids, X)`` scored on ``split`` of
    ``ds``."""
    rows = ds.split(split)
    rows = rows[rows["level_b"]]
    rows = rows[rows["substrate"].notna()].reset_index(drop=True)
    Xa, df, coverage = data.align(domain_ids, X, rows, key="domain_id", require_all=True)

    le = LabelEncoder().fit(df["substrate"])
    y = le.transform(df["substrate"])
    n_classes = len(le.classes_)
    scores = np.zeros((len(y), n_classes))
    if "fold" in df.columns:
        tests = [(df["fold"] == f).to_numpy() for f in sorted(df["fold"].unique())]
        parts = [(~test, test) for test in tests]
    else:
        parts = [((df["split"] == "train").to_numpy(), (df["split"] == "test").to_numpy())]
    for train, test in parts:
        clf = probes.logreg().fit(Xa[train], y[train])
        scores[test] = _predict_full(clf, Xa[test], n_classes)
    scored = np.logical_or.reduce([test for _, test in parts])
    if int(scored.sum()) != EXPECTED_SCORED[split]:
        raise AssertionError(f"{split}: {int(scored.sum())} scored domains, expected "
                             f"{EXPECTED_SCORED[split]}")

    out = metrics.accuracy_with_ci(y[scored], scores[scored], n_boot=n_boot, seed=seed)
    majority = float(df.loc[scored, "substrate"].value_counts().iloc[0] / scored.sum())
    return {**out, "n_scored": int(scored.sum()), "n_classes": n_classes,
            "majority": majority, **coverage}


AL_TEST_FOLD = 4


def al_rows(ds: data.Dataset = data.MAIN):
    """Active learning on the 5-fold split's scored domains, fold 4 as the test set; the
    validation rows are stratified on the substrate and keep each BGC's domains together."""
    from ..al import Rows

    df = ds.split(SPLITS["F"])
    df = df[df["level_b"] & df["substrate"].notna()].reset_index(drop=True)
    return Rows(df["domain_id"], (df["fold"] == AL_TEST_FOLD).to_numpy(), kind="adomain",
                strata=df["substrate"].astype(str).to_numpy(),
                groups=df["bgc_id"].astype(str).to_numpy(), table=df)


def al_readout(reps: dict, rows):
    from ..al import Classifier

    return Classifier(rows.features(reps["adomain"]), rows.table["substrate"].astype(str),
                      metric="top1")


class Substrate(Task):
    name, metric, splits, threads = "substrate", "top-1 accuracy", SPLITS, 4
    requires = ("adomain",)
    threads_override = {"evo2_7b": 8, "lcplm": 8}  # the published Evo 2 and LC-PLM cells used 8
    al_stream = 105

    def evaluate(self, reps, split, *, n_boot, device):
        a = reps["adomain"]
        return evaluate(a.ids, a.X, split, n_boot=n_boot, ds=self.data)

    def al_rows(self):
        return al_rows(self.data)

    def al_readout(self, reps, rows, *, device):
        return al_readout(reps, rows)
