"""Representation kinds: the sets of vectors that tasks score and encoders provide.

A kind is one vector per item, keyed by item ids:

``bgc``       one vector per BGC (``bgc_ids``); encoders provide it with ``embed_bgcs``
``adomain``   one vector per NRPS A-domain (``domain_ids``); ``embed_adomains``, which
              receives the A-domain table
``molecule``  one vector per compound (``compound_ids``); not from the encoder: the shipped
              MoLFormer vectors unless the user gives others

An encoder provides a kind through the kind's ``method``, called with the kind's ``table``
(when it has one) and the BGC inputs; a method that is missing or returns ``None`` provides
nothing. A kind with a ``default`` is filled in when a task needs it and no vectors were
given. Tasks name the kinds they need in ``Task.requires``. New kinds are registered with
:func:`register_representation`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class Embeddings:
    """Row-aligned ids and a float32 matrix."""

    ids: np.ndarray
    X: np.ndarray

    def __post_init__(self) -> None:
        self.ids = np.asarray(self.ids).astype(str)
        self.X = np.asarray(self.X)
        if self.X.ndim != 2 or len(self.X) != len(self.ids):
            raise ValueError(f"expected (n, d) vectors for {len(self.ids)} ids, got {self.X.shape}")
        if len(self.ids) == 0:
            raise ValueError("no embeddings")
        if self.X.dtype != np.float32:
            self.X = self.X.astype(np.float32)
        if not np.isfinite(self.X).all():
            raise ValueError("embeddings contain NaN or inf")
        if len(set(self.ids.tolist())) != len(self.ids):
            raise ValueError("embedding ids are not unique")

    @classmethod
    def coerce(cls, obj, id_key: str = "bgc_ids") -> "Embeddings":
        """From an Embeddings, a ``{id: vector}`` dict, an ``(ids, X)`` pair or an .npz path.

        Vectors may be NumPy arrays or torch tensors (on any device).
        """
        if isinstance(obj, Embeddings):
            return obj
        if isinstance(obj, dict):
            return cls(np.array(list(obj), dtype=str),
                       np.stack([_to_numpy(v) for v in obj.values()]))
        if isinstance(obj, (tuple, list)) and len(obj) == 2:
            return cls(_to_numpy(obj[0]), _to_numpy(obj[1]))
        if not isinstance(obj, (str, os.PathLike)) or Path(obj).suffix != ".npz":
            raise ValueError(f"{obj!r}: expected an .npz path with arrays '{id_key}' and "
                             f"'embeddings', a {{id: vector}} dict, an (ids, X) pair or "
                             f"Embeddings")
        with np.load(Path(obj), allow_pickle=False) as z:
            keys = [k for k in (id_key, "ids", "bgc_ids", "domain_ids", "compound_ids")
                    if k in z]
            if not keys or "embeddings" not in z:
                raise ValueError(f"{obj}: expected arrays '{id_key}' and 'embeddings'")
            try:
                ids = z[keys[0]]
            except ValueError as exc:
                raise ValueError(f"{obj}: the ids are stored as Python objects; save them as "
                                 f"strings, e.g. np.savez(path, {id_key}=ids.astype(str), "
                                 f"...)") from exc
            return cls(ids, z["embeddings"])

    def save(self, path: Path, id_key: str = "bgc_ids") -> None:
        """Write ``{id_key: ids, "embeddings": X}`` to ``path`` (replaced atomically)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **{id_key: self.ids, "embeddings": self.X})
        tmp.replace(path)


def _to_numpy(x):
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu()
        if x.is_floating_point():  # bfloat16 has no NumPy dtype
            x = x.float()
        x = x.numpy()
    return np.asarray(x)


@dataclass(frozen=True)
class Representation:
    """A registered representation kind (see the module docstring)."""

    kind: str
    id_key: str                       # id array name in .npz files
    method: str | None = None         # Encoder method that provides it
    table: Callable | None = None     # () -> the items to embed, passed to ``method``
    default: Callable | None = None   # () -> vectors used when none are given
    label: str | None = None          # in "no <label> vectors"

    @property
    def name(self) -> str:
        return self.label or self.kind


def _adomain_table():
    from .encoders import base

    return base.adomain_table()


def _molecules():
    from .tasks import retrieval

    return retrieval.load_compounds()


REGISTRY: dict[str, Representation] = {
    "bgc": Representation("bgc", "bgc_ids", "embed_bgcs", label="BGC"),
    "adomain": Representation("adomain", "domain_ids", "embed_adomains", table=_adomain_table,
                              label="A-domain"),
    "molecule": Representation("molecule", "compound_ids", default=_molecules),
}
BUILTIN = tuple(REGISTRY)


def register_representation(kind: str, id_key: str = "ids", *, method: str | None = None,
                            table: Callable | None = None, default: Callable | None = None,
                            label: str | None = None,
                            replace: bool = False) -> Representation:
    """Register a representation kind and return it.

    For example ``register_representation("gene", "gene_ids", method="embed_genes",
    table=my_gene_table)``. Registering a kind again with the same ``id_key`` and ``method``
    replaces it; other changes need ``replace=True``. Built-in kinds cannot be replaced.
    """
    rep = Representation(kind, id_key, method, table, default, label)
    current = REGISTRY.get(kind)
    if current is not None:
        if kind in BUILTIN:
            raise ValueError(f"{kind!r} is a built-in representation and cannot be replaced")
        if not replace and (current.id_key, current.method) != (id_key, method):
            raise ValueError(f"representation {kind!r} is already registered with id_key "
                             f"{current.id_key!r}, method {current.method!r}; pass "
                             f"replace=True to override")
    REGISTRY[kind] = rep
    return rep


def get(kind: str) -> Representation:
    """The registered kind, or an unregistered one with generic ``ids``."""
    return REGISTRY.get(kind) or Representation(kind, "ids")


def coerce(kind: str, obj) -> Embeddings:
    """:meth:`Embeddings.coerce` with the kind's id array name."""
    return Embeddings.coerce(obj, get(kind).id_key)
