"""The three linear probes: standardise the frozen vectors, then fit one sklearn model."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


# lbfgs (logistic regression) and the Cholesky solve (ridge) are deterministic: no seeds.
def _logreg() -> LogisticRegression:
    return LogisticRegression(max_iter=5000, C=1.0)


def logreg() -> Pipeline:
    """Multinomial logistic regression (A-domain substrate)."""
    return make_pipeline(StandardScaler(), _logreg())


def ovr_logreg() -> Pipeline:
    """One binary logistic regression per label (coarse and fine classes, multi-label)."""
    return make_pipeline(StandardScaler(), OneVsRestClassifier(_logreg()))


def ridge() -> Pipeline:
    """Multi-output ridge regression, alpha = 1 (chemical properties)."""
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
