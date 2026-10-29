"""Chemical properties: 13 RDKit descriptors of the first listed product, mean Spearman rho.

Probe: standardised multi-output ridge regression (alpha = 1). Rows are the split's BGCs
that have both descriptors and an embedding, in split order.
"""

from __future__ import annotations

import numpy as np

from .. import data, metrics, probes
from .base import Task

SPLITS = {"R": "t1_random", "S": "t1_scaffold_holdout", "G": "t1_gcf_holdout_c07"}
TARGETS = [
    # drug-likeness and ADMET (GuacaMol physicochemical set, QED)
    "MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA", "NumRotatableBonds", "qed",
    # natural-product structure
    "BertzCT", "NumAliphaticRings", "NumAromaticRings", "FractionCSP3", "NumStereocenters",
    "HalogenAtomCount",
]


def _targets(df, mask: np.ndarray) -> np.ndarray:
    """Descriptor matrix of the masked rows: float64, column-major (the layout of the
    published Ridge solves)."""
    Y = df.loc[mask, TARGETS].to_numpy()
    if Y.dtype != np.float64:
        raise AssertionError(f"target matrix is {Y.dtype}, expected float64")
    # Already column-major under pandas 3.0.2; made explicit so other versions match too.
    return np.asfortranarray(Y)


def evaluate(ids: np.ndarray, X: np.ndarray, split: str, *, n_boot: int = 500,
             seed: int = 42, ds: data.Dataset = data.MAIN) -> dict:
    """One chem column: BGC vectors ``(ids, X)`` scored on ``split`` of ``ds``."""
    rows = ds.split(split).merge(ds.read_csv("tasks/chem_descriptors.csv"), on="bgc_id",
                                 how="inner")
    Xa, df, coverage = data.align(ids, X, rows)
    train = (df["split"] == "train").to_numpy()
    test = (df["split"] == "test").to_numpy()
    data.require_train_test(train, test)
    X_tr, X_te = Xa[train], Xa[test]
    Y_tr, Y_te = _targets(df, train), _targets(df, test)
    # Descriptors that RDKit could not compute are NaN; such rows are left out. (The shipped
    # table has none; indexing only when needed keeps the column-major layout above.)
    ok_tr, ok_te = ~np.isnan(Y_tr).any(axis=1), ~np.isnan(Y_te).any(axis=1)
    if not ok_tr.all() or not ok_te.all():
        X_tr, Y_tr, X_te, Y_te = X_tr[ok_tr], Y_tr[ok_tr], X_te[ok_te], Y_te[ok_te]
    if not len(Y_tr) or not len(Y_te):
        raise data.CoverageError("need training and test rows with complete descriptors")

    pred = probes.ridge().fit(X_tr, Y_tr).predict(X_te)
    out = metrics.spearman_with_ci(Y_te, pred, n_boot=n_boot, seed=seed)
    return {**out, "n_train": int(len(Y_tr)), "n_test": int(len(Y_te)), **coverage}


def al_rows(ds: data.Dataset = data.MAIN):
    """Active learning on the random split: its BGCs with Pfam inputs and all 13
    descriptors."""
    from ..al import Rows, design

    df = ds.split(SPLITS["R"]).merge(ds.read_csv("tasks/chem_descriptors.csv"), on="bgc_id",
                                     how="inner")
    df = df[df["bgc_id"].isin(design.pfam_universe(ds))]
    df = df[~df[TARGETS].isna().any(axis=1)].reset_index(drop=True)
    return Rows(df["bgc_id"], (df["split"] == "test").to_numpy(), table=df)


def al_readout(reps: dict, rows):
    """Ridge on the 13 descriptors; a descriptor predicted as a constant scores rho = 0."""
    from ..al import Regressor

    return Regressor(rows.features(reps["bgc"]), rows.table[TARGETS].to_numpy(),
                     metric="mean_spearman")


class Chem(Task):
    name, metric, splits, threads = "chem", "mean Spearman rho", SPLITS, 8
    al_stream = 102

    def evaluate(self, reps, split, *, n_boot, device):
        b = reps["bgc"]
        return evaluate(b.ids, b.X, split, n_boot=n_boot, ds=self.data)

    def al_rows(self):
        return al_rows(self.data)

    def al_readout(self, reps, rows, *, device):
        return al_readout(reps, rows)
