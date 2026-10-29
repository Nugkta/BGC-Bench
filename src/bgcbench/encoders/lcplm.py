"""LC-PLM (Wang et al. 2025): a bidirectional Mamba protein language model, applied per CDS.

BGC vector: each CDS is embedded on its own (batch size 1, no truncation), its last-layer
states are averaged over the residue positions, and the CDS vectors are averaged (1536-d).
A-domain vector: the same pass on the domain's Pfam envelope.

Forward/reverse projections are tied after loading. Sequences run individually without
padding, with ``fused_add_norm=True`` to include the final normalization.
Dependencies: ``envs/gpu`` (mamba-ssm and causal-conv1d CUDA kernels).
"""

from __future__ import annotations

import numpy as np
import torch

from .base import Encoder, adomain_sequences
from .hub import model_dir

#: Longest CDS in the dataset is 18,447 residues; every CDS is one window on a 48 GB GPU.
MAX_CTX = 20_000


class LCPLM(Encoder):
    name, input, gpu, cuda_only = "lcplm", "protein", True, True
    needs = ("mamba_ssm", "causal_conv1d")

    def __init__(self, device: str | None = None) -> None:
        super().__init__(device)
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        # LC-PLM borrows the ESM-2 tokenizer (one token per residue plus <cls>/<eos>)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir("esm2_8m", allow_patterns=["*.json", "*.txt"]))
        model = AutoModelForMaskedLM.from_pretrained(model_dir("lcplm"), trust_remote_code=True)
        _retie(model)
        n_params = sum({id(p): p.numel() for p in model.parameters()}.values())
        if n_params > 1.0e9:
            raise RuntimeError(f"{n_params / 1e6:.1f} M parameters after tying the reverse "
                               f"projections; expected 752.8 M")
        if model.config.fused_add_norm is not True:
            raise RuntimeError("fused_add_norm=True is required for final-layer normalization")
        self.model = model.eval().to(self.device)

    @torch.inference_mode()
    def embed_sequence(self, seq: str) -> np.ndarray:
        """Residue mean of one sequence, in windows of MAX_CTX combined length-weighted."""
        pooled, weights = [], []
        for start in range(0, len(seq), MAX_CTX):
            window = seq[start:start + MAX_CTX]
            ids = self.tokenizer(window, return_tensors="pt")["input_ids"].to(self.device)
            if ids.shape[1] != len(window) + 2:
                raise RuntimeError(f"{ids.shape[1]} tokens for {len(window)} residues")
            h = self.model(input_ids=ids, output_hidden_states=True,
                           return_dict=True).hidden_states[-1]
            pooled.append(h[0, 1:-1, :].float().mean(dim=0).cpu())
            weights.append(len(window))
        w = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
        return ((torch.stack(pooled) * w).sum(0) / w.sum()).numpy()

    def embed_bgcs(self, inputs):
        out = {}
        for bgc_id in sorted(inputs):
            seqs = [s for s in inputs[bgc_id] if s]
            if seqs:
                vecs = [torch.from_numpy(self.embed_sequence(s)) for s in seqs]
                out[bgc_id] = torch.stack(vecs).mean(dim=0).numpy()
        return out

    def embed_adomains(self, adomains, inputs):
        ids, seqs = adomain_sequences(adomains)
        return ids, np.stack([self.embed_sequence(s) for s in seqs])


def _retie(model) -> None:
    """Share each layer's forward in/out projections with its reverse Mamba."""
    for layer in model.bimamba.backbone.layers:
        mixer = layer.mixer
        if getattr(mixer, "mamba_rev", None) is None:
            continue
        for proj in ("in_proj", "out_proj"):
            fwd, rev = getattr(mixer.mamba_fwd, proj), getattr(mixer.mamba_rev, proj)
            rev.weight = fwd.weight
            if fwd.bias is not None:
                rev.bias = fwd.bias
