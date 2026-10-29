"""Embedding cache keys and the coercion of user vectors."""

import numpy as np
import pytest
import torch

import bgcbench
from bgcbench import Encoder
from bgcbench.encoders.base import Embeddings, cache_dir


class A(Encoder):
    name = "same"


class B(Encoder):
    name = "same"


def test_cache_key_separates_classes_names_and_versions(home, no_dataset, monkeypatch):
    monkeypatch.setattr(bgcbench.encoders.base, "inputs_digest", lambda root: "fixed")
    a, b = A(device="cpu"), B(device="cpu")
    assert cache_dir(a) != cache_dir(b)
    a2 = A(device="cpu")
    a2.name = "other"
    a3 = A(device="cpu")
    a3.version = "2"
    assert len({cache_dir(a), cache_dir(a2), cache_dir(a3)}) == 3
    assert cache_dir(A) == cache_dir(a)  # class and default instance agree


def test_cache_key_depends_only_on_encoder_input_files(tmp_path):
    from bgcbench.encoders.base import inputs_digest

    def release(name, nucleotides, card):
        root = tmp_path / name
        (root / "inputs").mkdir(parents=True)
        (root / "tasks").mkdir()
        (root / "inputs/nucleotide_seqs.json").write_text(nucleotides)
        (root / "tasks/adomain_instances.csv").write_text("domain_id\n")
        (root / "tasks/adomain_nucleotide.json").write_text("{}")
        (root / "README.md").write_text(card)
        (root / "tasks/chem_encoders").mkdir()
        (root / "tasks/chem_encoders" / f"{card}.npz").write_text(card)
        return root

    a = release("a", '{"BGC1": "ATG"}', "v1")
    b = release("b", '{"BGC1": "ATG"}', "v2")   # new card and chemical files only
    c = release("c", '{"BGC1": "ATGC"}', "v1")  # changed model input
    assert inputs_digest(a) == inputs_digest(b)
    assert inputs_digest(a) != inputs_digest(c)
    (b / "tasks/adomain_nucleotide.json").write_text('{"d": "ATG"}')
    inputs_digest.cache_clear()
    assert inputs_digest(a) != inputs_digest(b)


def test_coerce_accepts_pairs_lists_tensors_and_explains_object_ids(tmp_path):
    ids = np.array(["x", "y"])
    X = np.eye(2, dtype=np.float32)
    for obj in [(ids, X), [ids, X], (ids, torch.eye(2, requires_grad=True)),
                (ids, torch.eye(2, dtype=torch.bfloat16)),
                {"x": torch.ones(2), "y": torch.zeros(2)}]:
        e = Embeddings.coerce(obj)
        assert e.X.shape == (2, 2) and e.X.dtype == np.float32
    np.savez(tmp_path / "obj.npz", bgc_ids=np.array(["x", "y"], dtype=object), embeddings=X)
    with pytest.raises(ValueError, match="as strings"):
        Embeddings.coerce(tmp_path / "obj.npz")
    with pytest.raises(ValueError, match="no embeddings"):
        Embeddings.coerce((np.array([], dtype=str), np.zeros((0, 3))))
    for bad in (tmp_path / "e.npy", tmp_path / "e.pkl", np.zeros((2, 2))):
        with pytest.raises(ValueError, match="expected an .npz path"):
            Embeddings.coerce(bad)
