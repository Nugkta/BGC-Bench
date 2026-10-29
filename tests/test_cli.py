"""The command line, end to end on small inputs (the API tests do not go through argparse)."""

import json

import numpy as np
import pytest

from bgcbench import cli
from bgcbench.encoders.base import cache_dir, load_inputs, read_representations


def test_cli_run_and_evaluate(dataset, tmp_path, monkeypatch):
    monkeypatch.setenv("BGCBENCH_HOME", str(tmp_path / "home"))
    out = tmp_path / "run.json"
    assert cli.main(["run", "--encoder", "pfam_rand", "--tasks", "fine", "--n-boot", "0",
                     "--device", "cpu", "--retrieval-device", "cpu", "--out", str(out)]) == 0
    cells = json.loads(out.read_text())["cells"]
    assert set(cells) == {"fine_R", "fine_S", "fine_G"}
    assert [cells[c]["n_test"] for c in ("fine_R", "fine_S", "fine_G")] == [241, 267, 212]

    ids = np.array(sorted(load_inputs("pfam_acc")))
    emb = tmp_path / "mine.npz"
    np.savez(emb, bgc_ids=ids, embeddings=np.random.default_rng(0).standard_normal((len(ids), 8)))
    out2 = tmp_path / "mine.json"
    assert cli.main(["evaluate", "--embeddings", str(emb), "--tasks", "coarse", "--n-boot", "0",
                     "--out", str(out2)]) == 0
    assert json.loads(out2.read_text())["encoder"] == "mine"


def test_cli_list(capsys):
    assert cli.main(["list"]) == 0
    rows = {line.split()[0]: line for line in capsys.readouterr().out.splitlines()[1:]}
    assert " cuda " in rows["evo2_7b"] and " gpu " in rows["bigcarp"]
    assert " cpu " in rows["deepbgc"]


def test_cli_rejects_an_unknown_encoder(capsys):
    with pytest.raises(SystemExit):
        cli.main(["run", "--encoder", "nope"])
    assert "unknown encoder 'nope'" in capsys.readouterr().err


def test_cli_embed_caches_without_scoring(dataset, tmp_path, monkeypatch, capsys):
    from bgcbench.encoders.baselines import RandomPfam

    monkeypatch.setenv("BGCBENCH_HOME", str(tmp_path))

    def unexpected(*args, **kwargs):
        raise AssertionError("embed must not evaluate tasks")

    monkeypatch.setattr("bgcbench.evaluate", unexpected)
    monkeypatch.setattr("bgcbench.pipeline.evaluate", unexpected)
    assert cli.main(["embed", "--encoder", "pfam_rand", "--device", "cpu"]) == 0
    reps = read_representations(cache_dir(RandomPfam))
    assert len(reps["bgc"].ids) == 2626 and "adomain" not in reps
    assert "cached 2626 BGC and 0 A-domain vectors" in capsys.readouterr().out
