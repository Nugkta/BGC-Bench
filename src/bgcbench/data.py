"""Datasets: download, loading and alignment.

The module-level functions read the main BGC-Bench dataset: the pinned revision on the
Hugging Face Hub, or the local copy that ``BGCBENCH_DATA`` names. A task that ships its own
data names it in ``Task.dataset`` and reads it through ``Task.data``, a :class:`Dataset`
with the same methods.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import DATASET_REPO, DATASET_REVISION, home

# Column dtypes of the id, label and split columns the tasks read. Writing them out keeps the
# parsed frames independent of pandas' type inference.
_DTYPES = {
    "bgc_id": str, "split": str, "fold": "int64", "primary_class": str, "labels": str,
    "classes_all": str, "enrolled": bool, "n_products": "int64", "n_products_smiles": "int64",
    "n_products_classified": "int64", "domain_id": str, "gcf_id": str, "substrate": str,
    "substrate_group": str, "level_b": bool, "evidence_tier": str,
    "NRP": "int64", "Other": "int64", "Polyketide": "int64", "RiPP": "int64",
    "Saccharide": "int64", "Terpene": "int64",
    "compound_id": str, "canonical_smiles": str, "mibig_order": "int64",
    "multiplicity": "int64",
    "gene_id": str, "cds_aa_hash": str, "domain_ordinal": "int64", "token_index": "int64",
    "token_index_raw": "int64", "env_start": "int64", "env_stop": "int64", "aa_seq": str,
    "stachelhaus_code": str, "n_substrates": "int64",
}


class CoverageError(ValueError):
    """The embeddings do not cover the rows a task needs."""


class Dataset:
    """A dataset directory: ``"org/repo@revision"`` on the Hugging Face Hub (downloaded to
    ``$BGCBENCH_HOME/data/org/repo/revision`` on first use) or a local directory.

    Splits live in ``splits/<name>.csv``. Files are read relative to the root; JSON files
    are parsed once per instance.
    """

    def __init__(self, source) -> None:
        self.source = str(source)
        self._root: Path | None = None
        self._json: dict = {}

    def __repr__(self) -> str:
        return f"Dataset({self.source!r})"

    def root(self) -> Path:
        """Directory holding the dataset (downloaded on first use)."""
        if self._root is None:
            self._root = self._locate()
        return self._root

    def _locate(self) -> Path:
        local = Path(self.source).expanduser()
        if local.is_dir():
            return local.resolve()
        repo, at, revision = self.source.partition("@")
        if not at or repo.count("/") != 1 or not revision:
            raise FileNotFoundError(f"dataset {self.source!r} is neither a directory nor "
                                    f"'org/repo@revision'")
        return _download(repo, revision, home() / "data" / repo / revision)

    def cache_clear(self) -> None:
        """Forget the located root and the parsed JSON files."""
        self._root = None
        self._json.clear()

    def path(self, rel: str) -> Path:
        """Absolute path of an existing dataset file."""
        p = self.root() / rel
        if not p.is_file():
            raise FileNotFoundError(f"{p} is missing")
        return p

    def read_csv(self, rel: str) -> pd.DataFrame:
        """A CSV file with the known columns parsed to fixed dtypes."""
        p = self.path(rel)
        header = pd.read_csv(p, nrows=0).columns
        return pd.read_csv(p, dtype={c: _DTYPES[c] for c in header if c in _DTYPES})

    def read_json(self, rel: str):
        """A JSON file, parsed once."""
        if rel not in self._json:
            with open(self.path(rel)) as fh:
                self._json[rel] = json.load(fh)
        return self._json[rel]

    def split(self, name: str) -> pd.DataFrame:
        """One split file, e.g. ``split("t1ml_random")``."""
        return self.read_csv(f"splits/{name}.csv")


class _Main(Dataset):
    """The pinned BGC-Bench release, or the local copy named by ``BGCBENCH_DATA``."""

    def _locate(self) -> Path:
        local = os.environ.get("BGCBENCH_DATA")
        if not local:
            return _download(DATASET_REPO, DATASET_REVISION, home() / "data" / DATASET_REVISION)
        path = Path(local).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"dataset directory does not exist: {path}")
        return path


def _download(repo: str, revision: str, local_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo, repo_type="dataset", revision=revision,
                                  local_dir=local_dir))


MAIN = _Main(f"{DATASET_REPO}@{DATASET_REVISION}")
_DATASETS: dict[str, Dataset] = {}


def get_dataset(source) -> Dataset:
    """The shared :class:`Dataset` of a source (so each is located and parsed once)."""
    key = str(source)
    if key not in _DATASETS:
        _DATASETS[key] = Dataset(key)
    return _DATASETS[key]


def root() -> Path:
    """Directory holding the main dataset (downloaded on first use)."""
    return MAIN.root()


def path(rel: str) -> Path:
    """Absolute path of an existing file of the main dataset."""
    return MAIN.path(rel)


def read_csv(rel: str) -> pd.DataFrame:
    """A CSV file of the main dataset (see :meth:`Dataset.read_csv`)."""
    return MAIN.read_csv(rel)


def read_json(rel: str):
    """A JSON file of the main dataset, parsed once."""
    return MAIN.read_json(rel)


def split(name: str) -> pd.DataFrame:
    """One split file of the main dataset, e.g. ``split("t1ml_random")``."""
    return MAIN.split(name)


def align(ids: np.ndarray, X: np.ndarray, df: pd.DataFrame, key: str = "bgc_id",
          *, require_all: bool = False) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Rows of ``X`` matching ``df[key]``, in the order of ``df``.

    Rows of ``df`` without an embedding are dropped (``require_all`` turns that into an
    error). Returns ``(X_aligned, df_aligned, coverage)``.
    """
    ids = np.asarray(ids).astype(str)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("embedding ids are not unique")
    index = {b: i for i, b in enumerate(ids)}
    if df.empty:
        raise CoverageError(f"the split has no {key} rows")
    if not len(ids):
        raise CoverageError("no embeddings given")
    present = df[key].isin(index).to_numpy()
    coverage = {"n_split": int(len(df)), "n_embedded": int(present.sum())}
    if not present.any():
        raise CoverageError(f"none of the {len(ids)} embedding ids is a {key} of this split "
                            f"(expected e.g. {df[key].iloc[0]!r}, got e.g. {ids[0]!r})")
    if require_all and not present.all():
        missing = df.loc[~present, key].tolist()
        raise CoverageError(f"{len(missing)} of {len(df)} {key}s have no embedding, "
                            f"e.g. {missing[:3]}; this task needs every one")
    kept = df[present].reset_index(drop=True)
    rows = [index[b] for b in kept[key]]
    return X[rows], kept, coverage


def require_train_test(train: np.ndarray, test: np.ndarray) -> None:
    """Require at least one embedded row on each side of a probe split."""
    if not train.any() or not test.any():
        raise CoverageError(f"need embedded training and test rows, got "
                            f"{int(train.sum())} train / {int(test.sum())} test")
