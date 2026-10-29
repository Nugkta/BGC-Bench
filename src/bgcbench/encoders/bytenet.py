"""ByteNet language model used by BiGCARP (non-causal, dilated residual 1-D CNN).

Vendored from microsoft/protein-sequence-models, commit af695772c4a1c056d930c95ec7e6428aa042f5cd
(sequence_models/convolutional.py: MaskedConv1d, ByteNetBlock, ByteNet, ByteNetLM;
sequence_models/layers.py: PositionFeedForward; sequence_models/collaters.py: _pad, here
``pad``), reduced to the code path BiGCARP uses (no causal convolutions, no low-rank or
frozen embeddings, no dropout at inference). ``ByteNetLM.hidden`` (the final-layer states
without the decoder) is added. Module names are unchanged so the published state dict loads
as is.

Licence of the original code (retained as its condition 1 requires):

Copyright (c) 2022
Microsoft [All rights reserved].

1. Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
THIS SOFTWARE IS PROVIDED BY [Name of Organization] “AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL [Name of Organisation] BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MaskedConv1d(nn.Conv1d):
    """Conv1d over (N, L, C) inputs with 'same' padding; zeroes masked positions first."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, dilation=dilation,
                         padding=padding)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class PositionFeedForward(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.conv = nn.Conv1d(d_in, d_out, 1)

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class ByteNetBlock(nn.Module):
    def __init__(self, d_in, d_h, d_out, kernel_size, dilation=1, activation="relu"):
        super().__init__()
        self.conv = MaskedConv1d(d_h, d_h, kernel_size=kernel_size, dilation=dilation)
        act = nn.GELU if activation == "gelu" else nn.ReLU
        self.sequence1 = nn.Sequential(nn.LayerNorm(d_in), act(), PositionFeedForward(d_in, d_h),
                                       nn.LayerNorm(d_h), act())
        self.sequence2 = nn.Sequential(nn.LayerNorm(d_h), act(), PositionFeedForward(d_h, d_out))

    def forward(self, x, input_mask=None):
        return x + self.sequence2(self.conv(self.sequence1(x), input_mask=input_mask))


class ByteNet(nn.Module):
    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                 padding_idx=None, slim=True, activation="relu"):
        super().__init__()
        self.embedder = nn.Embedding(n_tokens, d_embedding, padding_idx=padding_idx)
        self.up_embedder = PositionFeedForward(d_embedding, d_model)
        log2 = int(np.log2(r)) + 1
        dilations = [2 ** (n % log2) for n in range(n_layers)]
        d_h = d_model // 2 if slim else d_model
        self.layers = nn.ModuleList([
            ByteNetBlock(d_model, d_h, d_model, kernel_size, dilation=d, activation=activation)
            for d in dilations
        ])

    def forward(self, x, input_mask=None):
        e = self.up_embedder(self.embedder(x))
        for layer in self.layers:
            e = layer(e, input_mask=input_mask)
        return e


class ByteNetLM(nn.Module):
    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                 padding_idx=None, final_ln=False, slim=True, activation="relu"):
        super().__init__()
        self.embedder = ByteNet(n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                                padding_idx=padding_idx, slim=slim, activation=activation)
        self.decoder = PositionFeedForward(d_model, n_tokens)
        self.last_norm = nn.LayerNorm(d_model) if final_ln else nn.Identity()

    def hidden(self, x, input_mask=None):
        """Final-layer states after the last LayerNorm (what BiGCARP embeddings use)."""
        return self.last_norm(self.embedder(x, input_mask=input_mask))

    def forward(self, x, input_mask=None):
        return self.decoder(self.hidden(x, input_mask=input_mask))


def pad(tokenized: list[torch.Tensor], value: int) -> torch.Tensor:
    """Right-pad token tensors to one (batch, max_len) tensor."""
    out = torch.zeros((len(tokenized), max(len(t) for t in tokenized)),
                      dtype=tokenized[0].dtype) + value
    for row, t in enumerate(tokenized):
        out[row, :len(t)] = t
    return out
