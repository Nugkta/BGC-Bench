"""The authors' release tool (tools/build_release.py): cards, file inventories, the shipped
extraction report and uploads, on small synthetic folders."""

import json

import numpy as np
import pytest

from tools.build_release import (CHEM_ENCODERS, MIRRORS, build_chem_encoders, public_report,
                                 upload, write_manifest, write_model_card)


def test_lcplm_card_extracts_in_gpu_environment(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    write_model_card("lcplm", tmp_path)
    card = (tmp_path / "README.md").read_text()
    assert "cd envs/gpu && uv sync --frozen" in card
    assert "bgcbench embed --encoder lcplm" in card
    assert "cd ../.. && uv run bgcbench run --encoder lcplm" in card


def test_manifest_lists_current_files_and_sizes(tmp_path):
    payload = tmp_path / "inputs.json"
    payload.write_text("{}")
    write_manifest(tmp_path)
    assert json.loads((tmp_path / "manifest.json").read_text()) == {
        "files": {"inputs.json": {"size": 2}}}
    payload.write_text("[1]")
    (tmp_path / "extra.txt").write_text("extra")
    write_manifest(tmp_path)
    assert json.loads((tmp_path / "manifest.json").read_text()) == {
        "files": {"inputs.json": {"size": 3}, "extra.txt": {"size": 5}}}


@pytest.mark.parametrize("key", MIRRORS)
def test_model_card_lists_files_and_sizes(tmp_path, key):
    (tmp_path / "config.json").write_text("{}")
    write_model_card(key, tmp_path)
    card = (tmp_path / "README.md").read_text()
    assert "| file | bytes |" in card and "| `config.json` | 2 |" in card
    assert "not an official release of the original authors" in card
    assert f"Licence: {MIRRORS[key]['licence_name']}" in card
    assert ("non-commercial" in card) == (key in ("kawano", "lcplm"))
    assert "git clone https://github.com/Nugkta/BGC-Bench" in card
    assert sorted(p.name for p in tmp_path.iterdir()) == ["README.md", "config.json"]


def test_extraction_report_keeps_only_the_listed_keys():
    report = {"pass": True, "n_bgcs": 3, "total_time_s": 1.0, "regression_check": {},
              "provenance": {"model_id": "m", "revision": "r", "job_id": "1",
                             "branch": "b", "argv": ["x.py"], "checksums": {},
                             "encoder": {"pooling": "mean", "model_load_time_s": 2.0}}}
    assert public_report(report) == {
        "pass": True, "n_bgcs": 3,
        "provenance": {"model_id": "m", "revision": "r", "encoder": {"pooling": "mean"}}}


def test_upload_tags_only_the_configured_dataset_revision(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import huggingface_hub

    import bgcbench

    calls = []

    class FakeApi:
        def repo_exists(self, *args, **kwargs):
            return False

        def create_repo(self, *args, **kwargs):
            calls.append("create")
            assert kwargs["private"] is False

        def upload_folder(self, **kwargs):
            assert kwargs["folder_path"] == str(tmp_path)
            calls.append("upload")
            return SimpleNamespace(oid="test-commit")

        def create_tag(self, *args, **kwargs):
            assert kwargs["revision"] == "test-commit"
            calls.append("tag")

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    (tmp_path / "inputs.json").write_text("{}")
    upload(tmp_path, "test/dataset", "dataset", bgcbench.DATASET_REVISION)
    assert calls == ["create", "upload", "tag"]
    calls.clear()
    with pytest.raises(SystemExit, match="DATASET_REVISION"):
        upload(tmp_path, "test/dataset", "dataset", "different-version")
    assert calls == []


def test_chem_encoders_follow_compounds_order(tmp_path):
    research, out = tmp_path / "research", tmp_path / "out"
    ids = np.array(["cmp_a", "cmp_b", "cmp_c"])
    rng = np.random.default_rng(0)
    (out / "tasks").mkdir(parents=True)
    ref = rng.standard_normal((3, 4)).astype(np.float32)
    np.savez(out / "tasks/compounds.npz", compound_ids=ids, embeddings=ref)
    (research / "data/embeddings/molformer_compounds").mkdir(parents=True)
    np.savez(research / "data/embeddings/molformer_compounds/compounds.npz",
             compound_ids=ids, embeddings=ref)
    for name in CHEM_ENCODERS[1:]:
        d = research / "data/embeddings/chem_compounds" / name
        d.mkdir(parents=True)
        np.savez(d / "compounds.npz", compound_ids=ids,
                 embeddings=rng.standard_normal((3, 2)).astype(np.float32))
        base = name.removesuffix("_pca128")
        (d / "extraction_report.json").write_text(json.dumps({"registry_entry": {
            "name": name, "display": name, "family": "f", "base": base, "dim": 2,
            "variant": "pca128" if name != base else "raw", "definition": "d"}}))
    build_chem_encoders(research, out)
    index = json.loads((out / "tasks/chem_encoders/encoders.json").read_text())
    assert list(index) == list(CHEM_ENCODERS) and len(index) == 16
    assert index["unimol_pca128"]["weights"]["repo_id"] == "dptech/Uni-Mol-Models"
    assert "weights" not in index["ecfp4"]
    assert index["ecfp4"]["definition"].startswith("Morgan radius 2")
    assert index["maccs"]["definition"] == "d"  # taken from the research registry
    with np.load(out / "tasks/chem_encoders/molformer.npz") as z:
        assert (z["compound_ids"] == ids).all() and np.array_equal(z["embeddings"], ref)

    np.savez(research / "data/embeddings/chem_compounds/ecfp4/compounds.npz",
             compound_ids=ids[::-1], embeddings=np.zeros((3, 2), np.float32))
    with pytest.raises(AssertionError, match="ecfp4"):
        build_chem_encoders(research, out)
