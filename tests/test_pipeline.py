"""Argument validation, coverage reporting and the Bag-of-Pfam values of record."""

import numpy as np
import pytest

import bgcbench


def test_evaluate_validates_its_arguments():
    X = {"BGC0000001": np.zeros(3)}
    with pytest.raises(ValueError):
        bgcbench.evaluate(X, tasks=["coarse", "nonsense"])
    with pytest.raises(ValueError):
        bgcbench.evaluate(X, tasks="coarse", n_boot=-1)
    for threads in (0, -1):
        with pytest.raises(ValueError, match="threads"):
            bgcbench.evaluate(X, tasks="coarse", threads=threads)


def test_wrong_ids_and_missing_cohort_rows_are_reported_not_raised(dataset):
    rng = np.random.default_rng(0)
    wrong = {f"BGC{i:07d}.1": rng.standard_normal(4) for i in range(1, 50)}
    res = bgcbench.evaluate(wrong, tasks="coarse", n_boot=0, log=lambda *_: None)
    assert all(c["status"] == "na" and "none of the" in c["note"] for c in res.cells.values())
    from bgcbench.encoders.base import load_inputs
    ids = sorted(load_inputs("pfam_acc"))[::2]  # half the BGCs: retrieval needs all 1,939
    res = bgcbench.evaluate({b: rng.standard_normal(4) for b in ids}, tasks="retrieval",
                            n_boot=0, log=lambda *_: None)
    assert all(c["status"] == "na" for c in res.cells.values())


BAG_OF_PFAM = {"fine_R": 0.88043, "fine_S": 0.82519, "fine_G": 0.75317,  # values of record
               "chem_R": 0.35413, "chem_S": 0.28091, "chem_G": 0.28322,
               "substrate_F": 0.19635, "substrate_G": 0.15068}


def test_scores_match_the_stored_values(dataset, home):
    res = bgcbench.run("bag_of_pfam", tasks=["fine", "chem", "substrate"], n_boot=0,
                       log=lambda *_: None)
    ref = BAG_OF_PFAM
    assert set(res.cells) == set(ref)
    for col, cell in res.cells.items():
        assert abs(cell["point"] - ref[col]) < 1e-3, col


def test_n_boot_and_threads_are_checked_before_embedding(home, no_dataset):
    for kwargs in ({"n_boot": -1}, {"threads": 0}):
        with pytest.raises(ValueError):
            bgcbench.run("lcplm", **kwargs)  # would need the GPU environment to embed


def test_missing_gpu_modules_are_named(home, no_dataset, monkeypatch):
    cls = bgcbench.encoders.base.encoder_class("lcplm")
    monkeypatch.setattr(cls, "needs", ("numpy", "no_such_module_xyz"))
    with pytest.raises(ImportError, match="needs no_such_module_xyz,"):
        bgcbench.run("lcplm")
