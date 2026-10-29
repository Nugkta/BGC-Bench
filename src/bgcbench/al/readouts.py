"""Readouts: the model a task refits on the labelled rows at every acquisition round.

A readout holds the encoder's vectors for the task's AL rows (``features``, label-free, used
by TypiClust) and the labels. Each round the loop calls ``fit`` on the labelled rows, then
``predict`` and ``score`` on the validation and test rows; ``gradients`` gives BADGE its
gradient embedding (``None``: BADGE does not apply). Rows are integer indices into
:class:`bgcbench.al.Rows`, always in ascending order.

The ready-made readouts are the linear probes of the main table, made safe for small label
sets: a label that is constant among the labelled rows is predicted as that constant, and a
class absent from them scores 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from .. import metrics, probes


@dataclass
class GradEmbedding:
    """Rows ``g_i`` of a BADGE gradient embedding, explicit (``X is None``: ``g_i = E[i]``) or
    as a Kronecker product ``g_i = E[i] (x) X[i]`` that is never formed:
    ``g_i . g_j = (E[i] . E[j]) (X[i] . X[j])``."""

    E: np.ndarray
    X: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.E = np.ascontiguousarray(self.E, dtype=np.float64)
        if self.X is not None:
            self.X = np.ascontiguousarray(self.X, dtype=np.float64)
            if self.X.shape[0] != self.E.shape[0]:
                raise ValueError("GradEmbedding: E and X have different row counts")

    @property
    def n(self) -> int:
        return int(self.E.shape[0])

    @property
    def dim(self) -> int:
        return int(self.E.shape[1] * (1 if self.X is None else self.X.shape[1]))

    def sqnorm(self) -> np.ndarray:
        e = np.einsum("ij,ij->i", self.E, self.E)
        return e if self.X is None else e * np.einsum("ij,ij->i", self.X, self.X)

    def dots(self, i: int) -> np.ndarray:
        """``g_j . g_i`` for every row ``j``."""
        e = self.E @ self.E[i]
        return e if self.X is None else e * (self.X @ self.X[i])

    def dense(self) -> np.ndarray:
        if self.X is None:
            return self.E.copy()
        return (self.E[:, :, None] * self.X[:, None, :]).reshape(self.n, -1)


def classification_gradients(proba: np.ndarray, X_scaled: np.ndarray,
                             multilabel: bool) -> GradEmbedding:
    """BADGE's last-layer gradient ``(p - e_yhat) (x) x`` with the predicted label ``yhat``
    (argmax, or ``p > 0.5`` per label), factorised."""
    proba = np.asarray(proba, dtype=np.float64)
    if multilabel:
        err = proba - (proba > 0.5).astype(np.float64)
    else:
        err = proba.copy()
        err[np.arange(len(proba)), np.argmax(proba, axis=1)] -= 1.0
    return GradEmbedding(E=err, X=np.asarray(X_scaled, dtype=np.float64))


class Readout:
    """Base class: set ``metric``, keep the vectors in ``features`` and implement ``fit``,
    ``predict`` and ``score``; ``gradients`` and ``chance`` are optional."""

    metric: str = ""

    def __init__(self, features: np.ndarray) -> None:
        self.features = np.asarray(features)

    def fit(self, rows: np.ndarray, seed: int):
        """Fit on the labelled rows; ``seed`` is the learning curve's seed."""
        raise NotImplementedError

    def predict(self, fitted, rows: np.ndarray):
        raise NotImplementedError

    def score(self, pred, rows: np.ndarray) -> float:
        raise NotImplementedError

    def gradients(self, fitted, rows: np.ndarray) -> GradEmbedding | None:
        """BADGE gradient embedding of ``rows`` (``None``: not available)."""
        return None

    def chance(self, rows: np.ndarray) -> float:
        """The score of an uninformative model on ``rows`` (NaN if unknown)."""
        return float("nan")

    def info(self, fitted) -> dict:
        """Per-round details worth recording."""
        return {}

    @property
    def has_gradients(self) -> bool:
        return type(self).gradients is not Readout.gradients


def _metric(name_or_fn, names: dict):
    if callable(name_or_fn):
        return getattr(name_or_fn, "__name__", "custom"), name_or_fn
    if name_or_fn not in names:
        raise ValueError(f"unknown metric {name_or_fn!r}; choose from {', '.join(names)} or "
                         f"pass a function (y_true, pred) -> float")
    return name_or_fn, names[name_or_fn]


# Multi-label classification
class _MultiLabelFit:
    def __init__(self, n_labels: int, X: np.ndarray, Y: np.ndarray) -> None:
        from sklearn.preprocessing import StandardScaler

        self.n_labels = n_labels
        self.const, self.live = {}, []
        for j in range(n_labels):
            u = np.unique(Y[:, j])
            if len(u) == 1:
                self.const[j] = float(u[0])
            else:
                self.live.append(j)
        self.clf = None
        if len(self.live) >= 2:
            cols = Y if len(self.live) == n_labels else Y[:, self.live]
            self.clf = probes.ovr_logreg().fit(X, cols)
        elif len(self.live) == 1:
            # One varying label: a plain binary logistic regression with the same settings.
            self.clf = probes.logreg().fit(X, Y[:, self.live[0]].astype(np.int64))
        self.scaler = self.clf[0] if self.clf is not None else StandardScaler().fit(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros((X.shape[0], self.n_labels), dtype=np.float64)
        for j, v in self.const.items():
            out[:, j] = v
        if self.clf is not None:
            p = self.clf.predict_proba(X)
            if len(self.live) == 1:
                out[:, self.live[0]] = p[:, 1]
            else:
                out[:, self.live] = p
        return out


def _weighted_macro_auroc(Y, S) -> float:
    return metrics.multilabel_auroc(Y, S)["weighted_macro_auroc"]


class MultiLabel(Readout):
    """One standardised logistic regression per label (as the coarse and fine tasks).

    ``Y``: (n, C) 0/1 labels. ``metric``: ``"macro_auroc"``, ``"weighted_macro_auroc"``
    (each class weighted by its positives) or a function ``(Y_true, scores) -> float``.
    """

    METRICS = {"macro_auroc": metrics.macro_auroc_multilabel,
               "weighted_macro_auroc": _weighted_macro_auroc}

    def __init__(self, features, Y, metric="macro_auroc") -> None:
        super().__init__(features)
        self.Y = np.asarray(Y, dtype=np.float32)
        if self.Y.ndim != 2 or len(self.Y) != len(self.features):
            raise ValueError(f"Y must be (n rows, n labels), got {self.Y.shape}")
        self.metric, self._score = _metric(metric, self.METRICS)

    def fit(self, rows, seed):
        return _MultiLabelFit(self.Y.shape[1], self.features[rows], self.Y[rows])

    def predict(self, fitted, rows):
        return fitted.predict_proba(self.features[rows])

    def score(self, pred, rows):
        return float(self._score(self.Y[rows], pred))

    def gradients(self, fitted, rows):
        X = self.features[rows]
        return classification_gradients(fitted.predict_proba(X), fitted.scaler.transform(X),
                                        True)

    def chance(self, rows):
        return 0.5


# Single-label classification
class _ClassifierFit:
    def __init__(self, n_classes: int, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.preprocessing import StandardScaler

        self.n_classes = n_classes
        self.present = np.unique(y).astype(np.int64)
        self.clf = None
        if len(self.present) == 1:
            self.scaler = StandardScaler().fit(X)
        else:
            # Relabelling the present classes 0..m-1 in order leaves the fit unchanged.
            self.clf = probes.logreg().fit(X, np.searchsorted(self.present, y))
            self.scaler = self.clf[0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros((X.shape[0], self.n_classes), dtype=np.float64)
        if self.clf is None:
            out[:, self.present[0]] = 1.0
        else:
            out[:, self.present] = self.clf.predict_proba(X)
        return out


class Classifier(Readout):
    """A standardised multinomial logistic regression over a fixed class list (as the
    substrate task).

    ``y``: one label per row. ``classes``: the label space (default: the sorted labels of
    every row). ``metric``: ``"top1"``, ``"macro_auroc"`` (one-vs-rest) or a function
    ``(y_true as class indices, scores) -> float``.
    """

    METRICS = {"top1": metrics.top1_accuracy, "macro_auroc": metrics.macro_auroc_multiclass}

    def __init__(self, features, y, classes=None, metric="top1") -> None:
        super().__init__(features)
        labels = np.asarray(y)
        self.classes = np.asarray(sorted(set(labels.tolist())) if classes is None else classes)
        index = {c: i for i, c in enumerate(self.classes.tolist())}
        missing = sorted({c for c in labels.tolist() if c not in index})
        if missing:
            raise ValueError(f"labels outside the class list, e.g. {missing[:3]}")
        self.y = np.array([index[c] for c in labels.tolist()], dtype=np.int64)
        if len(self.y) != len(self.features):
            raise ValueError(f"{len(self.y)} labels for {len(self.features)} rows")
        self.metric, self._score = _metric(metric, self.METRICS)

    def fit(self, rows, seed):
        return _ClassifierFit(len(self.classes), self.features[rows], self.y[rows])

    def predict(self, fitted, rows):
        return fitted.predict_proba(self.features[rows])

    def score(self, pred, rows):
        return float(self._score(self.y[rows], pred))

    def gradients(self, fitted, rows):
        X = self.features[rows]
        if fitted.clf is None:
            # A constant predictor has no gradient: BADGE falls back to uniform draws.
            return GradEmbedding(E=np.zeros((len(rows), 1)))
        return classification_gradients(fitted.predict_proba(X), fitted.scaler.transform(X),
                                        False)

    def chance(self, rows):
        return 1.0 / len(self.classes) if self.metric == "top1" else 0.5


# Regression
def mean_spearman(Y: np.ndarray, P: np.ndarray) -> float:
    """Mean Spearman rho over the target columns; a constant prediction column scores 0."""
    rhos = []
    for j in range(Y.shape[1]):
        if np.std(Y[:, j]) == 0:
            raise ValueError(f"target column {j} is constant on the scored rows")
        rhos.append(0.0 if np.std(P[:, j]) == 0 else float(spearmanr(Y[:, j], P[:, j]).statistic))
    return float(np.mean(rhos))


class _RegressorFit:
    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.reg = probes.ridge().fit(X, Y)
        # A target constant in the labelled rows has a constant prediction in exact
        # arithmetic; pinning it keeps rounding noise from ranking it.
        self.const = {j: float(Y[0, j]) for j in range(Y.shape[1]) if np.all(Y[:, j] == Y[0, j])}

    def predict(self, X: np.ndarray) -> np.ndarray:
        P = np.array(self.reg.predict(X), dtype=np.float64).reshape(len(X), -1)
        for j, v in self.const.items():
            P[:, j] = v
        return P


class Regressor(Readout):
    """Standardised multi-output ridge regression, alpha = 1 (as the chem task).

    ``Y``: (n, T) float targets. ``metric``: ``"mean_spearman"`` or a function
    ``(Y_true, pred) -> float``. No gradient embedding, so BADGE does not apply.
    """

    METRICS = {"mean_spearman": mean_spearman}

    def __init__(self, features, Y, metric="mean_spearman") -> None:
        super().__init__(features)
        self.Y = np.asarray(Y, dtype=np.float64)
        if self.Y.ndim == 1:
            self.Y = self.Y[:, None]
        if len(self.Y) != len(self.features) or not np.isfinite(self.Y).all():
            raise ValueError("Y must hold finite targets, one row per AL row")
        self.metric, self._score = _metric(metric, self.METRICS)

    def fit(self, rows, seed):
        # Column-major targets, as the chem task fits them: the float32 solve depends on the
        # layout.
        return _RegressorFit(self.features[rows], np.asfortranarray(self.Y[rows]))

    def predict(self, fitted, rows):
        return fitted.predict(self.features[rows])

    def score(self, pred, rows):
        return float(self._score(self.Y[rows], pred))

    def chance(self, rows):
        return 0.0
