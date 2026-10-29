"""Kawano et al. 2026, Model I: a RoBERTa masked language model over Pfam domain names.

Input: the ordered Pfam domain names of each BGC, one word-level token per domain (no
special tokens are added). BGC vector: mean of the last hidden layer over the attended
positions of a 512-token, max-length-padded input (1024-d). A-domain vector: the last-layer
state at the domain's AMP-binding token, from a separate unpadded forward pass.

Position ids follow the checkpoint's ``pad_token_id = 1`` ([CLS], never emitted), so the
[PAD] positions (id 3) also get position ids; the attention mask and the pooling exclude them.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import Encoder, adomain_token_states
from .hub import model_dir

MAX_LENGTH = 512


class Kawano(Encoder):
    name, input, gpu = "kawano", "pfam_named", True

    def __init__(self, device: str | None = None) -> None:
        super().__init__(device)
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast, RobertaForMaskedLM

        path = model_dir("kawano")
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer.from_file(f"{path}/tokenizer.json"))
        self.tokenizer.add_special_tokens(
            {"mask_token": "[MASK]", "pad_token": "[PAD]", "sep_token": "[SEP]"})
        self.model = RobertaForMaskedLM.from_pretrained(path).to(self.device).eval()

    @torch.no_grad()
    def _forward(self, names: list[str], padded: bool):
        enc = self.tokenizer(names, is_split_into_words=True, max_length=MAX_LENGTH,
                             padding="max_length" if padded else False, truncation=True,
                             return_tensors="pt")
        ids = enc["input_ids"].to(self.device)
        att = enc["attention_mask"].to(self.device)
        hidden = self.model(input_ids=ids, attention_mask=att,
                            output_hidden_states=True).hidden_states[-1]
        return enc, att, hidden

    def embed_bgcs(self, inputs):
        out = {}
        for bgc_id in sorted(inputs):
            if not inputs[bgc_id]:
                continue
            _, att, hidden = self._forward(inputs[bgc_id], padded=True)
            mask = att.unsqueeze(-1).float()
            out[bgc_id] = ((hidden * mask).sum(1) / mask.sum(1))[0].cpu().numpy()
        return out

    def per_position(self, names: list[str]) -> np.ndarray:
        """One state per domain (sub-tokens averaged; one token per domain in practice)."""
        enc, _, hidden = self._forward(names, padded=False)
        words = enc.word_ids(batch_index=0)
        rows = []
        for d in range(min(len(names), MAX_LENGTH)):
            pos = [j for j, w in enumerate(words) if w == d]
            if pos:
                rows.append(hidden[0, pos, :].mean(dim=0).cpu().numpy())
        return np.stack(rows)

    def embed_adomains(self, adomains, inputs):
        parents = {b for b, t in zip(adomains["bgc_id"], adomains["token_index"])
                   if t >= 0 and inputs.get(b)}
        states = {b: (None, self.per_position(inputs[b])) for b in sorted(parents)}
        return adomain_token_states(adomains, states, inputs)
