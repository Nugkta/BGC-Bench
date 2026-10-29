"""Training-free baselines: Pfam composition and k-mer frequencies.

Bag-of-Pfam       L2-normalised counts over the 3,932 Pfam accessions of the dataset (fixed
                  vocabulary file). Substrate task: the parent BGC's vector, identical for
                  every A-domain of a BGC (the composition floor).
Random-Pfam       mean of one fixed N(0, 1) 256-d vector per accession (seed 42, shipped),
                  then L2-normalised. No A-domain vector: every A-domain is the same token,
                  PF00501.
Amino-acid 3-mer  frequencies of the 8,000 amino-acid 3-mers counted inside each CDS
                  (windows with non-standard residues dropped); A-domains: the Pfam envelope.
Nucleotide 6-mer  frequencies of the 2,080 canonical (reverse-complement-collapsed) DNA 6-mers
                  of the cluster; A-domains: the envelope's codons.

All four run on the CPU whatever device is given.
"""

from __future__ import annotations

import numpy as np

from .. import data
from .base import Encoder

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def _lut(alphabet: str) -> np.ndarray:
    lut = np.full(256, 255, dtype=np.uint8)
    for code, ch in enumerate(alphabet):
        lut[ord(ch)] = code
    return lut


def canonical_kmer_map(k: int) -> tuple[np.ndarray, int]:
    """k-mer index -> column with each k-mer merged with its reverse complement."""
    n = 4 ** k
    rc = np.zeros(n, dtype=np.int64)
    for idx in range(n):
        rev, rest = 0, idx
        for _ in range(k):
            rev = (rev << 2) | (3 - (rest & 3))
            rest >>= 2
        rc[idx] = rev
    cols, mapping = np.unique(np.minimum(np.arange(n), rc), return_inverse=True)
    return mapping, len(cols)


def _window_index(code: np.ndarray, k: int, powers: np.ndarray) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(code, k)
    valid = (windows != 255).all(axis=1)
    return (windows.astype(np.int64) * powers).sum(axis=1)[valid]


def _freq(idx: np.ndarray, dim: int) -> np.ndarray:
    counts = np.bincount(idx, minlength=dim).astype(np.float32)
    total = counts.sum()
    return counts / total if total else counts


class _Baseline(Encoder):
    def __init__(self, device: str | None = None) -> None:
        super().__init__("cpu")


class BagOfPfam(_Baseline):
    name, input = "bag_of_pfam", "pfam_acc"

    def embed_bgcs(self, inputs):
        vocab = data.read_json("inputs/bag_of_pfam_vocab.json")
        col = {p: i for i, p in enumerate(vocab)}
        out = {}
        for bgc_id, pfams in sorted(inputs.items()):
            vec = np.zeros(len(vocab), dtype=np.float32)
            for p in pfams:
                if p in col:
                    vec[col[p]] += 1
            norm = np.linalg.norm(vec)
            if norm > 0:
                out[bgc_id] = vec / norm
        self._parent = out
        return out

    def embed_adomains(self, adomains, inputs):
        parent = self._parent
        rows = [(d, parent[b]) for d, b in zip(adomains["domain_id"], adomains["bgc_id"])
                if b in parent]
        return dict(rows)


class RandomPfam(_Baseline):
    name, input = "pfam_rand", "pfam_acc"

    def embed_bgcs(self, inputs):
        with np.load(data.path("inputs/pfam_rand_vectors.npz"), allow_pickle=False) as z:
            vocab, vectors, unk = z["vocab"].astype(str), z["vectors"], z["unk"]
        table = dict(zip(vocab, vectors))
        out = {}
        for bgc_id, pfams in sorted(inputs.items()):
            if not pfams:
                continue
            v = np.stack([table.get(p, unk) for p in pfams]).mean(axis=0)
            norm = np.linalg.norm(v)
            if norm > 0:
                out[bgc_id] = (v / norm).astype(np.float32)
        return out


class AminoAcid3mer(_Baseline):
    name, input = "aa_kmer3", "protein"
    k = 3

    def _setup(self):
        return _lut(AA_ALPHABET), len(AA_ALPHABET) ** self.k, \
            (len(AA_ALPHABET) ** np.arange(self.k - 1, -1, -1)).astype(np.int64)

    def embed_bgcs(self, inputs):
        lut, dim, powers = self._setup()
        out = {}
        for bgc_id in sorted(inputs):
            # windows never span two CDS: count per CDS, then pool
            per_cds = [_window_index(lut[np.frombuffer(s.encode(), dtype=np.uint8)], self.k, powers)
                       for s in inputs[bgc_id] if len(s) >= self.k]
            if per_cds:
                out[bgc_id] = _freq(np.concatenate(per_cds), dim)
        return out

    def embed_adomains(self, adomains, inputs):
        lut, dim, powers = self._setup()
        out = {}
        for d, seq in zip(adomains["domain_id"], adomains["aa_seq"]):
            if not isinstance(seq, str) or not seq:
                continue
            code = lut[np.frombuffer(seq.upper().encode(), dtype=np.uint8)]
            out[d] = (_freq(_window_index(code, self.k, powers), dim) if len(code) >= self.k
                      else np.zeros(dim, dtype=np.float32))
        return out


class Nucleotide6mer(_Baseline):
    name, input = "kmer6", "nucleotide"
    k = 6

    def _setup(self):
        mapping, dim = canonical_kmer_map(self.k)
        return _lut("ACGT"), mapping, dim, (4 ** np.arange(self.k - 1, -1, -1)).astype(np.int64)

    def embed_bgcs(self, inputs):
        lut, mapping, dim, powers = self._setup()
        out = {}
        for bgc_id in sorted(inputs):
            code = lut[np.frombuffer(inputs[bgc_id].encode(), dtype=np.uint8)]
            if len(code) >= self.k:
                out[bgc_id] = _freq(mapping[_window_index(code, self.k, powers)], dim)
        return out

    def embed_adomains(self, adomains, inputs):
        lut, mapping, dim, powers = self._setup()
        out = {}
        for d, aa, nt in zip(adomains["domain_id"], adomains["aa_seq"], adomains["nt_seq"]):
            if not isinstance(aa, str) or not aa or not isinstance(nt, str):
                continue
            code = lut[np.frombuffer(nt.upper().encode(), dtype=np.uint8)]
            out[d] = (_freq(mapping[_window_index(code, self.k, powers)], dim)
                      if len(code) >= self.k else np.zeros(dim, dtype=np.float32))
        return out
