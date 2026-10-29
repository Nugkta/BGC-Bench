"""BiGCARP (Rios-Martinez et al. 2023): ByteNet masked language model over Pfam domain names.

Input: the ordered Pfam domain *names* of each BGC (the vocabulary is name-keyed; feeding
accessions maps every token to UNK). BGC vector: mean over the BGC's positions of the
final-layer states (256-d). A-domain vector: the state at the domain's AMP-binding token.

Batches contain 64 BGCs; the '-' padding token is excluded from every convolution.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .base import Encoder, adomain_token_states
from .bytenet import ByteNetLM, pad
from .hub import model_file

BATCH = 64


class BiGCARP(Encoder):
    name, input, gpu = "bigcarp", "pfam_named", True
    version = "1"

    def __init__(self, device: str | None = None) -> None:
        super().__init__(device)
        from safetensors.torch import load_file

        cfg = json.loads(Path(model_file("bigcarp", "config.json")).read_text())
        vocab = json.loads(Path(model_file("bigcarp", "vocab.json")).read_text())
        self.domains = vocab["domains"]
        self.pad_idx = vocab["specials"]["-"]
        self.mask_idx = vocab["specials"]["#"]
        self.model = ByteNetLM(
            n_tokens=vocab["size"], d_embedding=cfg["d_embedding"], d_model=cfg["d_model"],
            n_layers=cfg["n_layers"], kernel_size=cfg["kernel_size"], r=cfg["r"],
            padding_idx=self.mask_idx,  # '#' is the checkpoint's padding_idx; batches pad with '-'
            final_ln=True, slim=True, activation="gelu",
        )
        self.model.load_state_dict(load_file(model_file("bigcarp", "model.safetensors")))
        self.model.to(self.device).eval()

    def tokenize(self, names: list[str]) -> torch.Tensor:
        unk = self.domains["UNK"]
        return torch.tensor([self.domains.get(d, unk) for d in names], dtype=torch.long)

    @torch.no_grad()
    def states(self, inputs: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """``{bgc_id: (mean vector, per-position states)}`` for every non-empty BGC."""
        ids = [b for b in sorted(inputs) if inputs[b]]
        tokens = [self.tokenize(inputs[b]) for b in ids]
        out = {}
        for start in range(0, len(ids), BATCH):
            batch = tokens[start:start + BATCH]
            src = pad(batch, self.pad_idx).to(self.device)
            mask = (src != self.pad_idx).float().unsqueeze(-1)
            hidden = self.model.hidden(src, input_mask=mask)
            for i, t in enumerate(batch):
                h = hidden[i, :len(t), :]
                out[ids[start + i]] = (h.mean(dim=0).cpu().numpy(), h.cpu().numpy())
        return out

    def embed_bgcs(self, inputs):
        self._states = self.states(inputs)
        return {b: s[0] for b, s in self._states.items()}

    def embed_adomains(self, adomains, inputs):
        return adomain_token_states(adomains, self._states, inputs)
