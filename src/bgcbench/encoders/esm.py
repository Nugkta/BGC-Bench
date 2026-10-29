"""Protein language models used by three BGC encoders, applied to every CDS of a cluster.

BGC-MAC      ESM-2 650M (facebook/esm2_t33_650M_UR50D), 1280-d
BGC-Prophet  ESM-2 8M (facebook/esm2_t6_8M_UR50D), 320-d
BGCat        ESM C 600M (biohub/esmc-600m-2024-12), 1152-d

BGC vector: each CDS is cut to its first 1,022 residues, its last-layer states are averaged
over the residue positions, and the CDS vectors are averaged (all CDS, in order). CDS are
fed in batches of 4 (ESM-2) or 8 (ESM C) within a BGC.
A-domain vector: the same backbone on the domain's Pfam envelope (156-421 residues), batches
of 8 in table order; ESM-2 averages the residue states, ESM C takes position 0 (the BOS
token).

ESM-2 uses ``transformers.EsmModel``. ESM C uses ``esm`` with bfloat16 on GPU and float32
on CPU.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import Encoder, adomain_sequences
from .hub import model_dir, model_file

MAX_RES = 1022


class _CDSMean(Encoder):
    """Shared CDS loop: per-CDS residue means, averaged per BGC."""

    input, gpu = "protein", True
    bgc_batch = 4
    adomain_batch = 8

    def embed_bgcs(self, inputs):
        out = {}
        for bgc_id in sorted(inputs):
            seqs = [s for s in inputs[bgc_id] if s]
            if not seqs:
                continue
            vecs = []
            for i in range(0, len(seqs), self.bgc_batch):
                vecs.extend(self.residue_means(seqs[i:i + self.bgc_batch]))
            out[bgc_id] = np.mean(vecs, axis=0)
        return out

    def embed_adomains(self, adomains, inputs):
        ids, seqs = adomain_sequences(adomains)
        vecs = []
        for i in range(0, len(seqs), self.adomain_batch):
            vecs.extend(self.adomain_vectors(seqs[i:i + self.adomain_batch]))
        return ids, np.stack(vecs).astype(np.float32)

    def adomain_vectors(self, seqs):
        return self.residue_means(seqs)


class _ESM2(_CDSMean):
    key = ""

    def __init__(self, device: str | None = None) -> None:
        super().__init__(device)
        from transformers import EsmModel, EsmTokenizer

        path = model_dir(self.key, allow_patterns=["*.json", "*.txt", "model.safetensors"])
        tok = EsmTokenizer.from_pretrained(path)
        self.vocab = tok.get_vocab()
        self.cls, self.eos, self.pad, self.unk = (tok.cls_token_id, tok.eos_token_id,
                                                  tok.pad_token_id, tok.unk_token_id)
        self.model = EsmModel.from_pretrained(path, add_pooling_layer=False,
                                              attn_implementation="eager")
        self.model.to(self.device).eval()

    def _tokens(self, seqs: list[str]) -> torch.Tensor:
        """fair-esm batch converter: <cls> residues <eos>, right-padded with <pad>."""
        width = max(len(s) for s in seqs) + 2
        out = torch.full((len(seqs), width), self.pad, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s) + 2] = torch.tensor(
                [self.cls] + [self.vocab.get(c, self.unk) for c in s] + [self.eos])
        return out

    @torch.no_grad()
    def residue_means(self, seqs: list[str]) -> list[np.ndarray]:
        seqs = [s[:MAX_RES] for s in seqs]
        tokens = self._tokens(seqs).to(self.device)
        reps = self.model(input_ids=tokens, attention_mask=(tokens != self.pad).long()
                          ).last_hidden_state
        return [reps[i, 1:len(s) + 1, :].mean(dim=0).float().cpu().numpy()
                for i, s in enumerate(seqs)]


class BGCMAC(_ESM2):
    name, key = "bgc_mac", "esm2_650m"


class BGCProphet(_ESM2):
    name, key = "bgc_prophet", "esm2_8m"


class BGCat(_CDSMean):
    name = "bgcat"
    bgc_batch = 8

    def __init__(self, device: str | None = None) -> None:
        super().__init__(device)
        from esm.models.esmc import ESMC
        from esm.tokenization import get_esmc_model_tokenizers

        dev = torch.device(self.device)
        with dev:
            model = ESMC(d_model=1152, n_heads=18, n_layers=36,
                         tokenizer=get_esmc_model_tokenizers(), use_flash_attn=False).eval()
        ckpt = model_file("esmc_600m", "data/weights/esmc_600m_2024_12_v0.pth")
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        self.model = model.to(torch.bfloat16) if dev.type != "cpu" else model

    @torch.no_grad()
    def _embed(self, seqs: list[str]) -> torch.Tensor:
        return self.model(self.model._tokenize(seqs)).embeddings

    def residue_means(self, seqs):
        seqs = [s[:MAX_RES] for s in seqs]
        emb = self._embed(seqs)
        return [emb[i, 1:len(s) + 1, :].float().mean(dim=0).cpu().numpy()
                for i, s in enumerate(seqs)]

    def adomain_vectors(self, seqs):
        seqs = [s[:MAX_RES] for s in seqs]
        emb = self._embed(seqs)
        return [emb[i, 0, :].float().cpu().numpy() for i in range(len(seqs))]
