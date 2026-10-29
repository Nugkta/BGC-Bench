"""A user-defined encoder runs end to end and its embeddings are cached."""

import numpy as np
import pytest

import bgcbench
from bgcbench import Encoder


class ToyEncoder(Encoder):
    name, input = "toy_length", "protein"
    calls = 0

    def embed_bgcs(self, inputs):
        type(self).calls += 1
        rng = np.random.default_rng(0)
        proj = rng.standard_normal((3, 16))
        return {b: np.log1p([len(s), sum(map(len, s)), max(map(len, s))]) @ proj
                for b, s in inputs.items() if s}


def test_custom_encoder_end_to_end(dataset, home):
    res = bgcbench.run(ToyEncoder(device="cpu"), tasks=["coarse", "fine", "chem"], n_boot=20,
                       log=lambda *_: None)
    assert set(res.cells) == {f"{t}_{s}" for t in ("coarse", "fine", "chem") for s in "RSG"}
    for cell in res.cells.values():
        assert 0 <= cell["ci"][0] <= cell["ci"][1] and np.isfinite(cell["point"])
    for letter in "RSG":
        fine = res.cells[f"fine_{letter}"]
        assert fine["n_classes"] == 48
        for key in ("macro_auroc", "micro_auroc", "micro_auprc"):
            assert 0 <= fine[f"{key}_ci"][0] <= fine[f"{key}_ci"][1] <= 1
    calls = ToyEncoder.calls
    reps = ToyEncoder(device="cpu").representations()
    assert ToyEncoder.calls == calls and set(reps) == {"bgc"}  # served from the cache


def test_substrate_without_adomain_vectors_is_na(dataset, home):
    res = bgcbench.run(ToyEncoder(device="cpu"), tasks=["substrate"], n_boot=0,
                       log=lambda *_: None)
    assert all(c["status"] == "na" for c in res.cells.values())


def test_evaluate_reads_npz(dataset, tmp_path):
    rng = np.random.default_rng(1)
    ids = np.array(sorted(bgcbench.encoders.base.load_inputs("pfam_acc")))
    np.savez(tmp_path / "e.npz", bgc_ids=ids, embeddings=rng.standard_normal((len(ids), 8)))
    res = bgcbench.evaluate(tmp_path / "e.npz", name="rand", tasks=["coarse"], n_boot=0,
                            log=lambda *_: None)
    assert 0.3 < res.cells["coarse_R"]["point"] < 0.7


def test_gpu_encoders_without_a_gpu_need_device_cpu(monkeypatch):
    import torch

    from bgcbench.encoders.base import REGISTRY, NoGPUError, encoder_class

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class OnGPU(Encoder):
        name, gpu = "on_gpu", True

    class OnCUDA(Encoder):
        name, gpu, cuda_only = "on_cuda", True, True

    with pytest.raises(NoGPUError, match="--device cpu"):
        OnGPU()
    assert OnGPU(device="cpu").device == "cpu"
    assert ToyEncoder().device == "cpu"
    for device in (None, "cpu"):
        with pytest.raises(NoGPUError, match="CUDA GPU only"):
            OnCUDA(device=device)
    gpu = {n for n in REGISTRY if encoder_class(n).gpu}
    assert gpu == {"bigcarp", "kawano", "bgc_mac", "bgcat", "bgc_prophet", "lcplm", "evo2_7b"}
    assert {n for n in REGISTRY if encoder_class(n).cuda_only} == {"lcplm", "evo2_7b"}
