"""Evo 2 7B (Brixi et al. 2025): a StripedHyena 2 DNA language model over the cluster sequence.

BGC vector: the activations of layer ``blocks.28.mlp.l3`` averaged over the cluster's
nucleotide positions (4096-d). Clusters longer than 48 kb are cut into 48 kb windows whose
means are combined length-weighted. A-domain vector: the same pass on the codons of the
domain's Pfam envelope (468-1,263 bp, always one window).

Each window is right-padded with ``N`` to a 7-smooth length for cuFFT, then sliced to its
original length before pooling. The causal model uses bfloat16 projections and fixed 48 kb
windows, requiring at least 48 GB of GPU memory. Select the GPU with ``CUDA_VISIBLE_DEVICES``;
the model uses logical ``cuda:0``. Dependencies: ``envs/gpu`` (evo2, vtx, flash-attn).
"""

from __future__ import annotations

import bisect
import os

import numpy as np
import torch

from .base import Encoder
from .hub import model_file

LAYER = "blocks.28.mlp.l3"
WINDOW = 48_000


def _smooth_sizes(limit: int) -> list[int]:
    """Numbers up to ``limit`` with no prime factor above 7, ascending."""
    out, p2 = [], 1
    while p2 <= limit:
        p3 = p2
        while p3 <= limit:
            p5 = p3
            while p5 <= limit:
                p7 = p5
                while p7 <= limit:
                    out.append(p7)
                    p7 *= 7
                p5 *= 5
            p3 *= 3
        p2 *= 2
    return sorted(out)


_SMOOTH = _smooth_sizes(1 << 17)


def pad_length(n: int) -> int:
    """Smallest 7-smooth length >= n."""
    return _SMOOTH[bisect.bisect_left(_SMOOTH, n)]


class Evo2(Encoder):
    name, input, gpu, cuda_only = "evo2_7b", "nucleotide", True, True
    needs = ("evo2", "vortex", "flash_attn")

    def __init__(self, device: str | None = None) -> None:
        # Lets the 48 kb windows fit a 48 GB GPU, as for the published vectors; it takes effect
        # only before the process's first CUDA allocation and does not change the vectors.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        super().__init__(device)
        if torch.device(self.device).index not in (None, 0):
            raise ValueError("Evo 2 uses logical cuda:0; select the GPU with CUDA_VISIBLE_DEVICES")
        self.device = "cuda:0"
        import evo2.models

        evo2.models.HAS_TE = False  # bf16 projections, as for the published vectors
        self.model = evo2.models.Evo2("evo2_7b", local_path=model_file("evo2_7b", "evo2_7b.pt"))

    @torch.no_grad()
    def embed_sequence(self, seq: str) -> np.ndarray:
        pooled, weights = [], []
        for start in range(0, len(seq), WINDOW):
            window = seq[start:start + WINDOW]
            n = len(window)
            padded = window + "N" * (pad_length(n) - n)
            ids = torch.tensor(self.model.tokenizer.tokenize(padded),
                               dtype=torch.int).unsqueeze(0).to(self.device)
            _, emb = self.model(ids, return_embeddings=True, layer_names=[LAYER])
            pooled.append(emb[LAYER][:, :n].float().mean(dim=1).squeeze(0).cpu())
            weights.append(n)
        w = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
        return ((torch.stack(pooled) * w).sum(0) / w.sum()).numpy()

    def embed_bgcs(self, inputs):
        return {b: self.embed_sequence(inputs[b]) for b in sorted(inputs) if inputs[b]}

    def embed_adomains(self, adomains, inputs):
        rows = adomains[adomains["nt_seq"].notna()].sort_values("domain_id")
        ids = rows["domain_id"].tolist()
        return ids, np.stack([self.embed_sequence(s) for s in rows["nt_seq"]])
