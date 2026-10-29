"""DeepBGC (Hannigan et al. 2019): Pfam2vec + bidirectional LSTM, ported from Keras to torch.

DeepBGC's v0.1.0 detector release uses 128 LSTM units per direction with tanh and hard-sigmoid gates.
Inputs combine a 100-d Pfam2vec vector (zero for unknown accessions) with two protein-border
flags. This benchmark approximates those flags by marking the first and last token of the
cluster. The BGC vector is the mean of the concatenated forward/backward states (256-d).
Substrate columns are unavailable for this encoder. It embeds on the CPU whatever device is
given (a sequential recurrence, for which the CPU is the fast choice).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .base import Encoder
from .hub import model_file


def _hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x * 0.2 + 0.5, 0.0, 1.0)


def _lstm(x: torch.Tensor, kernel, recurrent, bias, reverse: bool) -> torch.Tensor:
    """Keras LSTM (gate order i, f, c, o) over one sequence (L, d); returns (L, units)."""
    units = recurrent.shape[0]
    xw = x @ kernel + bias
    h = torch.zeros(units, dtype=x.dtype)
    c = torch.zeros(units, dtype=x.dtype)
    out = torch.empty(len(x), units, dtype=x.dtype)
    steps = range(len(x) - 1, -1, -1) if reverse else range(len(x))
    for t in steps:
        z = xw[t] + h @ recurrent
        i = _hard_sigmoid(z[:units])
        f = _hard_sigmoid(z[units:2 * units])
        c = f * c + i * torch.tanh(z[2 * units:3 * units])
        o = _hard_sigmoid(z[3 * units:])
        h = o * torch.tanh(c)
        out[t] = h
    return out


class DeepBGC(Encoder):
    name, input = "deepbgc", "pfam_acc"

    def __init__(self, device: str | None = None) -> None:
        super().__init__("cpu")
        from safetensors.numpy import load_file

        w = load_file(model_file("deepbgc", "model.safetensors"))
        self.w = {k: torch.from_numpy(v) for k, v in w.items()}
        vocab = json.loads(Path(model_file("deepbgc", "pfam2vec_vocab.json")).read_text())
        self.row = {p: i for i, p in enumerate(vocab)}

    def features(self, pfams: list[str]) -> np.ndarray:
        dim = self.w["pfam2vec"].shape[1]
        X = np.zeros((len(pfams), dim + 2), dtype=np.float32)
        table = self.w["pfam2vec"].numpy()
        for t, p in enumerate(pfams):
            if p in self.row:
                X[t, :dim] = table[self.row[p]]
        X[0, dim] = 1.0
        X[-1, dim + 1] = 1.0
        return X

    @torch.no_grad()
    def hidden(self, pfams: list[str]) -> np.ndarray:
        x = torch.from_numpy(self.features(pfams))
        fwd = _lstm(x, self.w["forward.kernel"], self.w["forward.recurrent_kernel"],
                    self.w["forward.bias"], reverse=False)
        bwd = _lstm(x, self.w["backward.kernel"], self.w["backward.recurrent_kernel"],
                    self.w["backward.bias"], reverse=True)
        return torch.cat([fwd, bwd], dim=1).numpy()

    def embed_bgcs(self, inputs):
        from .. import limit_threads

        with limit_threads(1):  # small sequential matmuls: one thread, a fixed summation order
            return {b: self.hidden(p).mean(axis=0) for b, p in sorted(inputs.items()) if p}
