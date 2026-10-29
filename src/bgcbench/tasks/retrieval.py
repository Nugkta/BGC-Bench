"""BGC -> molecule retrieval: class-macro R@10 with a trained linear contrastive head.

Universe: the 1,939 BGCs with a product SMILES that every encoder of the paper embeds
(``tasks/t3_universe.csv``), 3,426 BGC-molecule pairs, 3,110 distinct molecules (canonical
SMILES). A test BGC queries the gallery of every distinct molecule produced by a test BGC;
a hit is any of its own products in the top 10. Hits are exact expectations over tied
scores, averaged within each primary class (the six MIBiG classes plus Hybrid, 7), then over
classes.

Training per split, seed and fold:

* two linear heads (BGC -> 256, molecule -> 256, L2-normalised) and a learnable
  temperature, initialised right after ``torch.manual_seed(fold_seed)``;
* AdamW, lr 1e-4, weight decay 1e-4, batch 256, at most 50 epochs; batches of fewer than
  ``MIN_BATCH`` (8) rows are skipped; early stopping on the training loss (patience 5, min
  delta 1e-4), then the best epoch's weights are restored;
* every training BGC is one row per epoch, paired round-robin with one of its products
  (private RNG ``default_rng([seed, 1])``); the batch order comes from
  ``default_rng(seed)``; in-batch pairs that are true BGC-molecule links are masked out of
  both InfoNCE softmaxes.

Seeds 0, 1, 2; the class holdout trains one head per held-out class with fold seed
``seed * 1000 + fold``. A cell is the mean over the three seeds, and its interval spans
the lowest and highest per-seed 95 % interval (class-stratified query bootstrap, 500
resamples; seed = fold seed, or 42 for the class holdout, whose folds are pooled).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import data, metrics, set_deterministic
from .base import Task

SPLITS = {"R": "t3_random", "S": "t3_scaffold_holdout", "G": "t3_gcf_holdout_c07",
          "C": "t3_class_holdout"}
SEEDS = (0, 1, 2)
K = 10
MIN_BATCH = 8
EXPECTED = {"n_bgcs": 1939, "n_pairs": 3426, "n_compounds": 3110}
HYPER = dict(out_dim=256, n_epochs=50, batch_size=256, lr=1e-4, weight_decay=1e-4,
             patience=5, min_delta=1e-4)
LOGIT_SCALE_INIT = math.log(1.0 / 0.07)
LOGIT_SCALE_MAX = math.log(100.0)
CLASS_HOLDOUT_CI_SEED = 42


# Model
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


class ContrastiveHead(nn.Module):
    def __init__(self, bgc_dim: int, chem_dim: int, out_dim: int) -> None:
        super().__init__()
        self.bgc_head = ProjectionHead(bgc_dim, out_dim)
        self.chem_head = ProjectionHead(chem_dim, out_dim)
        self.log_logit_scale = nn.Parameter(torch.tensor(LOGIT_SCALE_INIT))

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.log_logit_scale.clamp(max=LOGIT_SCALE_MAX).exp()


def masked_info_nce(z_bgc, z_chem, logit_scale, mask) -> torch.Tensor:
    """Symmetric InfoNCE; ``mask[i, j]`` (i != j) marks a true link removed from both softmaxes.

    Masked logits are set to -inf; the diagonal remains the sampled positive.
    """
    logits = logit_scale * z_bgc @ z_chem.T
    mask = mask.to(device=logits.device, dtype=torch.bool, copy=True)
    mask.fill_diagonal_(False)
    logits = logits.masked_fill(mask, float("-inf"))
    targets = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2


class RoundRobinSampler:
    """One product per BGC per epoch, cycling through each BGC's shuffled product list.

    Only BGCs with several products draw from the RNG (in BGC order, at start and at every
    reshuffle), so single-product BGCs never shift anyone else's draws.
    """

    def __init__(self, bgc_items: list[np.ndarray], seed: int) -> None:
        self.items = [np.asarray(a, dtype=np.int64) for a in bgc_items]
        self.multi = [i for i, a in enumerate(self.items) if a.size > 1]
        self.rng = np.random.default_rng([seed, 1])
        self.order = list(self.items)
        for i in self.multi:
            self.order[i] = self.rng.permutation(self.items[i])
        self.ptr = np.zeros(len(self.items), dtype=np.int64)
        self.fixed = np.array([a[0] for a in self.items], dtype=np.int64)

    def draw_epoch(self) -> np.ndarray:
        out = self.fixed.copy()
        for i in self.multi:
            p = int(self.ptr[i])
            if p == self.order[i].size:
                self.order[i] = self.rng.permutation(self.items[i])
                p = 0
            out[i] = self.order[i][p]
            self.ptr[i] = p + 1
        return out


def train_head(bgc_emb: np.ndarray, item_emb: np.ndarray, bgc_items: list[np.ndarray],
               relation: np.ndarray, seed: int, device: str) -> ContrastiveHead:
    """Fit the contrastive head on the training BGCs; returns it with the best epoch restored."""
    h = HYPER
    n_train = len(bgc_emb)
    sampler = RoundRobinSampler(bgc_items, seed)
    dev = torch.device(device)
    torch.manual_seed(seed)
    model = ContrastiveHead(bgc_emb.shape[1], item_emb.shape[1], h["out_dim"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=h["lr"], weight_decay=h["weight_decay"])
    bgc_t = torch.tensor(np.ascontiguousarray(bgc_emb), dtype=torch.float32, device=dev)
    item_t = torch.tensor(np.ascontiguousarray(item_emb), dtype=torch.float32, device=dev)
    rel_t = torch.tensor(relation, dtype=torch.bool, device=dev)

    rng = np.random.default_rng(seed)
    best_loss, wait, best_state = float("inf"), 0, None
    for _ in range(h["n_epochs"]):
        model.train()
        perm = rng.permutation(n_train)
        drawn = sampler.draw_epoch()
        slots = torch.from_numpy(np.stack((perm, drawn[perm]))).to(dev)
        losses = []
        for start in range(0, n_train, h["batch_size"]):
            stop = min(start + h["batch_size"], n_train)
            if stop - start < MIN_BATCH:
                continue
            idx_t, items_t = slots[0, start:stop], slots[1, start:stop]
            mask = rel_t[idx_t.unsqueeze(1), items_t.unsqueeze(0)]
            loss = masked_info_nce(model.bgc_head(bgc_t[idx_t]), model.chem_head(item_t[items_t]),
                                   model.logit_scale, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.detach())
        epoch_mean = float(np.mean(torch.stack(losses).double().cpu().numpy()))
        if best_loss - epoch_mean > h["min_delta"]:
            best_loss, wait = epoch_mean, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= h["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()


def _project_unique(head: nn.Module, X: np.ndarray) -> np.ndarray:
    """Project each distinct input row once, so identical inputs keep identical outputs."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    uniq, inv = np.unique(X, axis=0, return_inverse=True)
    dev = next(head.parameters()).device
    with torch.no_grad():
        z = head(torch.tensor(uniq, dtype=torch.float32, device=dev)).cpu().numpy()
    return z[np.asarray(inv).reshape(-1)]


def _scores_unique(zq: np.ndarray, zc: np.ndarray) -> np.ndarray:
    """float64 cosine scores, computed once per distinct pair of vectors (ties stay exact)."""
    uq, iq = np.unique(np.asarray(zq, dtype=np.float32), axis=0, return_inverse=True)
    uc, ic = np.unique(np.asarray(zc, dtype=np.float32), axis=0, return_inverse=True)
    S = uq.astype(np.float64) @ uc.astype(np.float64).T
    return S[np.ix_(np.asarray(iq).reshape(-1), np.asarray(ic).reshape(-1))]


# Data
@dataclass
class Universe:
    split_df: pd.DataFrame           # split rows of the universe BGCs, split order
    bgc_emb: np.ndarray
    classes: np.ndarray
    bgc_compounds: list[np.ndarray]  # compound indices per BGC, MIBiG order
    compound_emb: np.ndarray


def load_compounds(ds: data.Dataset = data.MAIN) -> tuple[np.ndarray, np.ndarray]:
    """(compound_ids, float32 vectors) of the shipped MoLFormer file."""
    with np.load(ds.path("tasks/compounds.npz"), allow_pickle=False) as z:
        ids, emb = z["compound_ids"].astype(str), np.asarray(z["embeddings"], dtype=np.float32)
    if emb.ndim != 2 or len(emb) != len(ids) or not np.isfinite(emb).all():
        raise ValueError("compound embeddings must be a finite (n_compounds, dim) matrix")
    return ids, emb


def chem_encoders() -> dict:
    """``{name: metadata}`` of the molecule vector sets in ``tasks/chem_encoders/encoders.json``,
    MoLFormer included (``molformer.npz`` repeats ``tasks/compounds.npz``)."""
    return data.read_json("tasks/chem_encoders/encoders.json")


def chem_encoder_path(name: str) -> Path:
    """Path of a shipped molecule vector set, e.g. ``chem_encoder_path("ecfp4")``."""
    known = chem_encoders()
    if name not in known:
        raise ValueError(f"unknown chemical encoder {name!r}; choose from {', '.join(known)}")
    return data.path(f"tasks/chem_encoders/{name}.npz")


def load_universe(ids: np.ndarray, X: np.ndarray, split: str,
                  chem: tuple[np.ndarray, np.ndarray] | None = None,
                  ds: data.Dataset = data.MAIN) -> Universe:
    """The retrieval universe of ``split`` with BGC vectors ``(ids, X)`` and molecule vectors
    ``chem`` (default: the shipped MoLFormer vectors)."""
    universe = set(ds.read_csv("tasks/t3_universe.csv")["bgc_id"])
    rows = ds.split(split)
    rows = rows[rows["bgc_id"].isin(universe)].reset_index(drop=True)
    bgc_emb, rows, _ = data.align(ids, X, rows, require_all=True)

    pairs = ds.read_csv("tasks/t3_bgc_compound_pairs.csv")
    sub = pairs[pairs["bgc_id"].isin(universe)].sort_values(["bgc_id", "mibig_order"])
    cmp_ids = np.array(sorted(sub["compound_id"].unique()), dtype=str)
    got = {"n_bgcs": len(rows), "n_pairs": len(sub), "n_compounds": len(cmp_ids)}
    if got != EXPECTED:
        raise AssertionError(f"retrieval universe {got} != {EXPECTED}")
    col = {c: i for i, c in enumerate(cmp_ids)}
    lists = sub.groupby("bgc_id", sort=False)["compound_id"].apply(list).to_dict()
    bgc_compounds = [np.array([col[c] for c in lists[b]], dtype=np.int64) for b in rows["bgc_id"]]

    chem_ids, chem_emb = chem if chem is not None else load_compounds(ds)
    index = {c: i for i, c in enumerate(chem_ids)}
    missing = [c for c in cmp_ids if c not in index]
    if missing:
        raise data.CoverageError(f"{len(missing)} of {len(cmp_ids)} molecules have no vector, "
                                 f"e.g. {missing[:3]}; retrieval needs every one")
    return Universe(rows, bgc_emb, rows["primary_class"].to_numpy().astype(str), bgc_compounds,
                    chem_emb[[index[c] for c in cmp_ids]])


# Evaluation
def _folds(u: Universe, split: str, seed: int):
    if split == "t3_class_holdout":
        cls = np.asarray(u.split_df["primary_class"].values, dtype=object)
        for i, held in enumerate(sorted(u.split_df["primary_class"].unique())):
            te = cls == held
            yield ~te, te, seed * 1000 + i
    else:
        col = np.asarray(u.split_df["split"].values, dtype=object)
        yield col == "train", col == "test", seed


def _fold_hits(u: Universe, tr: np.ndarray, te: np.ndarray, fold_seed: int, device: str):
    """Expected hit@10 per test BGC, and the test classes, for one trained fold."""
    tr_idx, te_idx = np.flatnonzero(tr), np.flatnonzero(te)
    if len(tr_idx) < 8 or len(te_idx) < 2:
        raise ValueError(f"fold too small: {len(tr_idx)} train / {len(te_idx)} test BGCs")
    train_cmp = np.unique(np.concatenate([u.bgc_compounds[i] for i in tr_idx]))
    col = {c: j for j, c in enumerate(train_cmp)}
    items = [np.array([col[c] for c in u.bgc_compounds[i]], dtype=np.int64) for i in tr_idx]
    relation = np.zeros((len(tr_idx), len(train_cmp)), dtype=bool)
    for r, it in enumerate(items):
        relation[r, it] = True
    model = train_head(u.bgc_emb[tr_idx], u.compound_emb[train_cmp], items, relation,
                       fold_seed, device)

    test_cmp = np.unique(np.concatenate([u.bgc_compounds[i] for i in te_idx]))
    tcol = {c: j for j, c in enumerate(test_cmp)}
    positives = np.zeros((len(te_idx), len(test_cmp)), dtype=bool)
    for r, i in enumerate(te_idx):
        positives[r, [tcol[c] for c in u.bgc_compounds[i]]] = True
    z_b = _project_unique(model.bgc_head, u.bgc_emb[te_idx])
    z_c = _project_unique(model.chem_head, u.compound_emb[test_cmp])
    hits = metrics.expected_hit_at_k(_scores_unique(z_b, z_c), positives, K)
    return hits, u.classes[te_idx], len(tr_idx), len(te_idx), len(test_cmp)


def _one_hot(labels: np.ndarray) -> np.ndarray:
    classes = np.asarray(sorted(set(labels.tolist())), dtype=str)
    return labels[:, None] == classes[None, :]


def evaluate_seed(u: Universe, split: str, seed: int, *, n_boot: int = 500,
                  device: str = "cpu") -> dict:
    """Class-macro R@10 and its 95 % interval for one seed."""
    set_deterministic(seed)
    folds = [_fold_hits(u, tr, te, fs, device) + (fs,) for tr, te, fs in _folds(u, split, seed)]
    if len(folds) == 1:
        hits, cls, n_tr, n_te, n_items, fs = folds[0]
        member = _one_hot(cls)
        point = metrics.class_macro(hits, member)
        ci = metrics.class_macro_ci(hits, member, n_boot=n_boot, seed=fs)
        return {"point": point, "ci": ci, "n_train": n_tr, "n_test": n_te, "n_items": n_items}
    # Class holdout: every fold is one class; pool the folds for the interval.
    point = float(np.mean([metrics.class_macro(f[0], _one_hot(f[1])) for f in folds]))
    hits = np.concatenate([f[0] for f in folds])
    member = np.zeros((len(hits), len(folds)), dtype=bool)
    start = 0
    for k, f in enumerate(folds):
        member[start:start + len(f[0]), k] = True
        start += len(f[0])
    ci = metrics.class_macro_ci(hits, member, n_boot=n_boot, seed=CLASS_HOLDOUT_CI_SEED)
    return {"point": point, "ci": ci, "n_train": [f[2] for f in folds],
            "n_test": [f[3] for f in folds], "n_items": [f[4] for f in folds]}


def evaluate(ids: np.ndarray, X: np.ndarray, split: str, *, n_boot: int = 500,
             chem: tuple[np.ndarray, np.ndarray] | None = None, device: str = "cpu",
             ds: data.Dataset = data.MAIN) -> dict:
    """One retrieval column: BGC vectors ``(ids, X)`` scored on ``split`` of ``ds`` over the
    seeds ``SEEDS``; ``chem`` replaces the MoLFormer molecule vectors, ``device`` trains the
    head."""
    u = load_universe(ids, X, split, chem, ds)
    per_seed = [evaluate_seed(u, split, s, n_boot=n_boot, device=device) for s in SEEDS]
    return {
        "point": float(np.mean([r["point"] for r in per_seed])),
        "ci": [min(r["ci"][0] for r in per_seed), max(r["ci"][1] for r in per_seed)],
        "per_seed": {int(s): r for s, r in zip(SEEDS, per_seed)},
        "n_train": per_seed[0]["n_train"], "n_test": per_seed[0]["n_test"],
        "n_split": int(len(u.split_df)), "n_embedded": int(len(u.split_df)),
    }


class Retrieval(Task):
    name, metric, splits, threads = "retrieval", "class-macro R@10", SPLITS, 1
    requires = ("bgc", "molecule")

    def evaluate(self, reps, split, *, n_boot, device):
        b, m = reps["bgc"], reps["molecule"]
        return evaluate(b.ids, b.X, split, n_boot=n_boot, chem=(m.ids, m.X),
                        device=device, ds=self.data)
