import os

import pytest


@pytest.fixture(scope="session")
def dataset():
    """The BGC-Bench dataset (downloaded once).

    Skips when the dataset cannot be located or downloaded, unless BGCBENCH_DATA is set.
    """
    from bgcbench import data

    try:
        return data.root()
    except Exception as exc:
        if os.environ.get("BGCBENCH_DATA"):
            raise
        pytest.skip(f"dataset not available: {exc}")


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An empty ``BGCBENCH_HOME`` (embedding caches)."""
    monkeypatch.setenv("BGCBENCH_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def no_dataset(tmp_path, monkeypatch):
    """An empty main dataset: cache keys hash the encoder inputs, of which there are none."""
    from bgcbench import data

    (tmp_path / "data").mkdir()
    monkeypatch.setenv("BGCBENCH_DATA", str(tmp_path / "data"))
    data.MAIN.cache_clear()
    yield
    data.MAIN.cache_clear()
