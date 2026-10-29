"""Dataset location, file reading and alignment."""

import numpy as np
import pandas as pd
import pytest

from bgcbench import data


@pytest.fixture
def toy_dataset(tmp_path, monkeypatch):
    (tmp_path / "splits").mkdir()
    f = tmp_path / "splits" / "toy.csv"
    f.write_text("bgc_id,split\nB1,train\nB2,test\n")
    monkeypatch.setenv("BGCBENCH_DATA", str(tmp_path))
    data.MAIN.cache_clear()
    yield tmp_path
    data.MAIN.cache_clear()


def test_local_dataset_reads_split_files(toy_dataset):
    assert data.split("toy")["bgc_id"].tolist() == ["B1", "B2"]


def test_split_files_are_reread_on_each_call(toy_dataset):
    data.split("toy")
    (toy_dataset / "splits" / "toy.csv").write_text("bgc_id,split\nB1,test\nB2,test\n")
    assert data.split("toy")["split"].tolist() == ["test", "test"]


def test_missing_file_reports_its_path(toy_dataset):
    with pytest.raises(FileNotFoundError, match="missing.csv"):
        data.split("missing")


def test_missing_local_directory_is_reported(toy_dataset, monkeypatch):
    monkeypatch.setenv("BGCBENCH_DATA", str(toy_dataset / "missing"))
    with pytest.raises(FileNotFoundError, match="dataset directory"):
        data.root()


def test_main_dataset_downloads_the_configured_revision(toy_dataset, monkeypatch):
    import huggingface_hub

    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(toy_dataset)

    monkeypatch.delenv("BGCBENCH_DATA")
    monkeypatch.setenv("BGCBENCH_HOME", str(toy_dataset / "cache"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert data.split("toy")["bgc_id"].tolist() == ["B1", "B2"]
    assert calls == [{"repo_id": data.DATASET_REPO, "repo_type": "dataset",
                      "revision": data.DATASET_REVISION,
                      "local_dir": toy_dataset / "cache" / "data" / data.DATASET_REVISION}]


def test_align_keeps_split_order_and_reports_coverage():
    df = pd.DataFrame({"bgc_id": ["C", "A", "B", "D"]})
    X = np.arange(6, dtype=np.float32).reshape(3, 2)
    Xa, kept, cov = data.align(np.array(["A", "B", "C"]), X, df)
    assert kept["bgc_id"].tolist() == ["C", "A", "B"]
    np.testing.assert_array_equal(Xa, X[[2, 0, 1]])
    assert cov == {"n_split": 4, "n_embedded": 3}


def test_align_require_all_and_duplicates():
    df = pd.DataFrame({"bgc_id": ["A", "B"]})
    with pytest.raises(data.CoverageError, match="have no embedding"):
        data.align(np.array(["A"]), np.zeros((1, 2)), df, require_all=True)
    with pytest.raises(ValueError, match="not unique"):
        data.align(np.array(["A", "A"]), np.zeros((2, 2)), df)


def test_align_reports_an_empty_split_as_a_coverage_error():
    with pytest.raises(data.CoverageError, match="no bgc_id rows"):
        data.align(np.array(["A"]), np.zeros((1, 2)), pd.DataFrame({"bgc_id": []}))
