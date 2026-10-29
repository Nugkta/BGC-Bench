"""Toy AL tasks and a toy strategy that need no dataset, imported by tests/test_al_run.py
(also from worker processes, via ``modules``)."""

import numpy as np
import pandas as pd

import bgcbench
from bgcbench import al

N = 240


def table() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, N)
    return pd.DataFrame({"bgc_id": [f"B{i:04d}" for i in range(N)],
                         "label": [f"c{v}" for v in y], "value": y + rng.normal(0, 0.5, N),
                         "test": np.arange(N) % 4 == 0})


def vectors(d: int = 6, shift: float = 1.0) -> dict:
    t = table()
    X = np.random.default_rng(1).standard_normal((N, d)).astype(np.float32)
    X[:, 0] += shift * t["label"].str[1].astype(int).to_numpy()
    return dict(zip(t["bgc_id"], X))


@bgcbench.register_task
class ToyClass(bgcbench.Task):
    name, metric = "toy_class", "top-1 accuracy"
    splits = {"R": "none"}

    def evaluate(self, reps, split, *, n_boot, device):
        return {"point": 0.0}

    def al_rows(self):
        t = table()
        return al.Rows(t["bgc_id"], t["test"].to_numpy(), strata=t["label"], table=t)

    def al_readout(self, reps, rows, *, device):
        return al.Classifier(rows.features(reps["bgc"]), rows.table["label"])


@bgcbench.register_task
class ToyValue(bgcbench.Task):
    name, metric = "toy_value", "Spearman rho"
    splits = {"R": "none"}

    def evaluate(self, reps, split, *, n_boot, device):
        return {"point": 0.0}

    def al_rows(self):
        t = table()
        return al.Rows(t["bgc_id"], t["test"].to_numpy(), table=t)

    def al_readout(self, reps, rows, *, device):
        return al.Regressor(rows.features(reps["bgc"]), rows.table["value"])


@al.register_strategy("lowest")
def lowest(ctx, k):
    """The unlabelled pool rows with the smallest first coordinate."""
    unl = ctx.unlabelled
    return unl[np.argsort(ctx.pool_features[unl, 0], kind="stable")[:k]]
