"""BiGCARP masks its batch padding: a BGC's vectors do not depend on the other BGCs in its batch."""

import numpy as np
import torch

from bgcbench.encoders.bigcarp import BiGCARP
from bgcbench.encoders.bytenet import ByteNetLM


def tiny_bigcarp() -> BiGCARP:
    torch.manual_seed(0)
    enc = BiGCARP.__new__(BiGCARP)
    enc.device = "cpu"
    enc.domains = {f"D{i}": i for i in range(3, 40)} | {"UNK": 1}
    enc.pad_idx, enc.mask_idx = 0, 2
    enc.model = ByteNetLM(n_tokens=40, d_embedding=8, d_model=16, n_layers=6, kernel_size=3, r=8,
                          padding_idx=enc.mask_idx, final_ln=True, slim=True,
                          activation="gelu").eval()
    return enc


def test_vectors_do_not_depend_on_batch_padding():
    enc = tiny_bigcarp()
    rng = np.random.default_rng(0)
    short = [f"D{i}" for i in rng.integers(3, 40, 5)]
    long = [f"D{i}" for i in rng.integers(3, 40, 60)]
    alone = enc.states({"a": short})["a"]
    batched = enc.states({"a": short, "b": long})["a"]
    for x, y in zip(alone, batched):
        np.testing.assert_allclose(x, y, atol=1e-5)
