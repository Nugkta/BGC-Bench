"""Encoder interface, built-in encoder registry and on-disk embedding cache.

An encoder turns the dataset's model inputs into one vector per BGC (``embed_bgcs``) and,
optionally, one vector per NRPS A-domain (``embed_adomains``, for the substrate task).
The inputs it reads are named by ``input``:

``pfam_acc``    Pfam accessions per BGC (``{bgc_id: ["PF00501", ...]}``, 2,631 BGCs)
``pfam_named``  Pfam domain names per BGC (``{bgc_id: ["AMP-binding", ...]}``, 2,626 BGCs)
``protein``     CDS translations per BGC (``{bgc_id: ["MKT...", ...]}``)
``nucleotide``  cluster DNA per BGC (``{bgc_id: "ATG..."}``)

``embed_adomains`` receives the A-domain table together with the BGC inputs. The table has
one row per domain and no labels: ``domain_id``, ``bgc_id``, ``aa_seq`` (the Pfam
envelope), ``nt_seq`` (its codons), ``token_index`` / ``token_index_raw`` (position of the
domain's AMP-binding token in the named / accession Pfam list). Return ``None`` if the
encoder has no A-domain representation; the substrate columns are then reported as "--".
Further representation kinds (:mod:`bgcbench.representations`) are provided the same way,
by a method of the name the kind registers.

Everything an encoder provides is computed once and cached as ``<kind>.npz`` in
``$BGCBENCH_HOME/embeddings/<name>/<key>/``, the key hashing the encoder class, ``name``,
``version``, ``input`` and the contents of the dataset files encoders read
(``ENCODER_FILES``).
"""

from __future__ import annotations

import hashlib
import json
import platform
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .. import home
from .. import representations as kinds
from ..representations import Embeddings

INPUT_FILES = {
    "pfam_acc": "inputs/pfam_domains.json",
    "pfam_named": "inputs/pfam_domains_named.json",
    "protein": "inputs/protein_sequences.json",
    "nucleotide": "inputs/nucleotide_seqs.json",
}
# Everything an encoder may read: the model inputs and the A-domain table.
ENCODER_FILES = ("inputs", "tasks/adomain_instances.csv", "tasks/adomain_nucleotide.json")
# The A-domain columns encoders see (the table's labels and split groups are left out).
ADOMAIN_COLUMNS = ("domain_id", "bgc_id", "aa_seq", "nt_seq", "token_index", "token_index_raw")


class NoGPUError(RuntimeError):
    """An encoder that embeds on a GPU was started without one."""


class Encoder:
    """Base class for BGC encoders. Subclass it and implement ``embed_bgcs``.

    ``embed_bgcs`` receives all BGCs of the chosen input and returns ``{bgc_id: vector}``
    or ``(ids, X)``. Encoders handle their own batching.
    """

    name: str = "custom"
    input: str = "protein"
    #: Change it to invalidate the embedding cache (new weights, pooling or code); or pass
    #: ``refresh=True``. The cache is keyed by class, ``name``, ``version``, ``input`` and the
    #: contents of the dataset files encoders read (``ENCODER_FILES``).
    version: str = "1"
    #: Python modules the encoder needs besides the main environment (checked before
    #: downloading weights).
    needs: tuple[str, ...] = ()
    #: Embeds on a GPU (its reference vectors were made on one). With no device given and no
    #: GPU visible it refuses to start instead of falling back to the CPU.
    gpu: bool = False
    #: Runs on a CUDA GPU only (CUDA kernels), even when a device is given.
    cuda_only: bool = False

    def __init__(self, device: str | None = None) -> None:
        """``device=None`` picks ``cuda`` when a GPU is visible and the CPU otherwise; an
        encoder with ``gpu`` set raises instead of using the CPU unless ``device="cpu"``."""
        if device is None:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            elif self.cuda_only:
                raise NoGPUError(f"{self.name} runs on a CUDA GPU only and none is visible. "
                                 f"Scoring cached vectors needs no GPU.")
            elif self.gpu:
                raise NoGPUError(
                    f"{self.name} embeds on a GPU and none is visible: run it on a GPU node, "
                    f"or pass device='cpu' (CLI: --device cpu) to embed on the CPU, which is "
                    f"much slower and gives slightly different vectors. Scoring cached "
                    f"vectors needs no GPU.")
            else:
                device = "cpu"
        if self.cuda_only and not str(device).startswith("cuda"):
            raise NoGPUError(f"{self.name} runs on a CUDA GPU only, got device {device!r}")
        self.device = device

    def embed_bgcs(self, inputs: dict):
        raise NotImplementedError

    def embed_adomains(self, adomains, inputs: dict):
        return None

    # Caching
    def cache_dir(self) -> Path:
        """This encoder's embedding cache folder (see :func:`cache_dir`)."""
        return cache_dir(self)

    def representations(self, *, refresh: bool = False) -> dict[str, Embeddings]:
        """``{kind: Embeddings}`` of every kind this encoder provides, computed once and then
        read from the cache (``<kind>.npz`` next to ``manifest.json``)."""
        cdir = self.cache_dir()
        hit = None if refresh else read_representations(cdir, gpu=self.gpu)
        if hit is not None:
            return hit
        inputs = load_inputs(self.input)
        reps = {}
        for rep in kinds.REGISTRY.values():
            fn = getattr(self, rep.method, None) if rep.method else None
            if fn is None:
                continue
            raw = fn(inputs) if rep.table is None else fn(rep.table(), inputs)
            if raw is not None:
                reps[rep.kind] = Embeddings.coerce(raw, rep.id_key)
        if "bgc" not in reps:
            raise ValueError(f"{type(self).__name__}.embed_bgcs returned None")
        for kind, emb in reps.items():
            emb.save(cdir / f"{kind}.npz", kinds.get(kind).id_key)
        (cdir / "manifest.json").write_text(json.dumps({
            "encoder": self.name, "version": self.version, "input": self.input,
            "device": device_name(self.device), "device_type": str(self.device).split(":")[0],
            "dim": int(reps["bgc"].X.shape[1]),
            "kinds": {kind: int(len(emb.ids)) for kind, emb in reps.items()},
        }, indent=2))
        return reps


def cache_dir(owner) -> Path:
    """Cache folder of an encoder class or instance (class, name, version, input, inputs digest)."""
    from .. import data

    cls = owner if isinstance(owner, type) else type(owner)
    ident = (f"{cls.__module__}.{cls.__qualname__}|{owner.name}|{owner.version}|"
             f"{owner.input}|{inputs_digest(data.root())}")
    return home() / "embeddings" / owner.name / hashlib.sha256(ident.encode()).hexdigest()[:12]


@lru_cache(maxsize=None)
def inputs_digest(root: Path) -> str:
    """Cache-key digest (sha256) over the paths and bytes of the dataset files encoders read.

    A dataset release that leaves these files unchanged keeps every embedding cache valid.
    """
    files = []
    for rel in ENCODER_FILES:
        p = root / rel
        files += sorted(p.rglob("*")) if p.is_dir() else [p]
    h = hashlib.sha256()
    for p in files:
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts):
            with p.open("rb") as fh:
                h.update(f"{p.relative_to(root).as_posix()}|".encode())
                h.update(hashlib.file_digest(fh, "sha256").digest())
    return h.hexdigest()


def read_representations(cdir: Path, *, gpu: bool = False) -> dict[str, Embeddings] | None:
    """``{kind: Embeddings}`` stored in a cache folder, read without loading any model.

    Kinds that are not registered in this process are skipped. With ``gpu`` (an encoder that
    embeds on a GPU), a cache made on the CPU is used with a warning.
    """
    man_p = cdir / "manifest.json"
    if not man_p.exists():
        return None
    man = json.loads(man_p.read_text())
    if gpu and man.get("device_type") == "cpu":
        warnings.warn(f"{man['encoder']}: the cached vectors were embedded on the CPU "
                      f"({man['device']}); pass refresh=True (CLI: --refresh) to embed on a GPU",
                      stacklevel=2)
    return {kind: Embeddings.coerce(cdir / f"{kind}.npz", kinds.get(kind).id_key)
            for kind in man["kinds"] if kind in kinds.REGISTRY}


def load_inputs(kind: str) -> dict:
    """The BGC inputs one encoder reads; proteins become ``{bgc_id: [aa_seq, ...]}``."""
    from .. import data

    raw = data.read_json(INPUT_FILES[kind])
    if kind == "protein":
        return {b: [c["aa_seq"] for c in cds] for b, cds in raw.items()}
    return raw


def adomain_table():
    """A-domain instances with their protein envelope and nucleotide sequence
    (``ADOMAIN_COLUMNS``)."""
    from .. import data

    df = data.read_csv("tasks/adomain_instances.csv")
    nt = data.read_json("tasks/adomain_nucleotide.json")
    df["nt_seq"] = df["domain_id"].map(nt)
    return df[list(ADOMAIN_COLUMNS)]


def adomain_sequences(adomains) -> tuple[list[str], list[str]]:
    """``(domain_ids, envelopes)`` of the A-domains that have a protein envelope, in table
    order."""
    ok = adomains["aa_seq"].notna() & (adomains["aa_seq"].str.len() > 0)
    rows = adomains[ok]
    return rows["domain_id"].tolist(), rows["aa_seq"].tolist()


def adomain_token_states(adomains, states: dict, inputs: dict) -> dict:
    """State at each A-domain's AMP-binding token of the named Pfam list (``token_index``).

    ``states`` maps a BGC to ``(mean vector, per-position states)``; the token at each
    position is checked against the encoder's input.
    """
    out = {}
    for d, b, t in zip(adomains["domain_id"], adomains["bgc_id"], adomains["token_index"]):
        if b not in states or t < 0:
            continue
        per_pos = states[b][1]
        if len(per_pos) != len(inputs[b]) or inputs[b][t] != "AMP-binding":
            raise RuntimeError(f"{d}: token {t} of {b} is not 'AMP-binding' in this encoder's "
                               f"input")
        out[d] = per_pos[t]
    return out


def device_name(device: str) -> str:
    """The GPU model for a CUDA device, else the CPU model."""
    if str(device).startswith("cuda"):
        import torch

        return torch.cuda.get_device_name(torch.device(device))
    return cpu_model()


def cpu_model() -> str:
    """The CPU model name (from ``/proc/cpuinfo`` on Linux)."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


# Registry
@dataclass(frozen=True)
class Spec:
    module: str
    cls: str
    display: str  # name in result tables
    input: str
    env: str      # "main" or "gpu" (envs/gpu)
    licence: str  # of the weights


REGISTRY: dict[str, Spec] = {
    "bag_of_pfam": Spec("baselines", "BagOfPfam", "Bag-of-Pfam", "pfam_acc", "main", "-"),
    "pfam_rand": Spec("baselines", "RandomPfam", "Random-Pfam", "pfam_acc", "main", "-"),
    "bigcarp": Spec("bigcarp", "BiGCARP", "BiGCARP", "pfam_named", "main", "MIT"),
    "kawano": Spec("kawano", "Kawano", "Kawano", "pfam_named", "main", "CC BY-NC 4.0"),
    "deepbgc": Spec("deepbgc", "DeepBGC", "DeepBGC", "pfam_acc", "main", "MIT"),
    "aa_kmer3": Spec("baselines", "AminoAcid3mer", "Amino-acid 3-mer", "protein", "main", "-"),
    "bgc_mac": Spec("esm", "BGCMAC", "BGC-MAC (ESM-2 650M)", "protein", "main", "MIT"),
    "bgcat": Spec("esm", "BGCat", "BGCat (ESM C 600M)", "protein", "main",
                  "Cambrian non-commercial"),
    "bgc_prophet": Spec("esm", "BGCProphet", "BGC-Prophet (ESM-2 8M)", "protein", "main", "MIT"),
    "lcplm": Spec("lcplm", "LCPLM", "LC-PLM", "protein", "gpu", "CC BY-NC 4.0"),
    "kmer6": Spec("baselines", "Nucleotide6mer", "Nucleotide 6-mer", "nucleotide", "main", "-"),
    "evo2_7b": Spec("evo2", "Evo2", "Evo 2 7B", "nucleotide", "gpu", "Apache-2.0"),
}


def list_encoders() -> list[str]:
    """Names of the built-in encoders, in table order."""
    return list(REGISTRY)


def display_name(name: str) -> str:
    """Table name of a built-in encoder; other names are shown as given."""
    return REGISTRY[name].display if name in REGISTRY else name


def encoder_class(name: str) -> type[Encoder]:
    """The class of a built-in encoder (imported on first use)."""
    import importlib

    if name not in REGISTRY:
        raise KeyError(f"unknown encoder {name!r}; built-in: {', '.join(REGISTRY)}")
    spec = REGISTRY[name]
    return getattr(importlib.import_module(f"bgcbench.encoders.{spec.module}"), spec.cls)


def get_encoder(name: str, **kwargs) -> Encoder:
    """A built-in encoder; ``kwargs`` (e.g. ``device``) go to its constructor."""
    return encoder_class(name)(**kwargs)
