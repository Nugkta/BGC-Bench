"""Active-learning runs on toy tasks (no dataset): results, caching, worker processes,
custom strategies, summaries and the command line."""

import json
from pathlib import Path

import numpy as np
import pytest

import bgcbench
from bgcbench import al, cli
from bgcbench.al import strategies as S
from bgcbench.tasks import base

TOY = Path(__file__).with_name("al_toy_tasks.py")


@pytest.fixture
def toy():
    saved = dict(base.REGISTRY), dict(S.REGISTRY)
    cli.import_module(str(TOY))
    import sys

    yield sys.modules["bgcbench_tasks_al_toy_tasks"]
    for reg, entries in zip((base.REGISTRY, S.REGISTRY), saved):
        reg.clear()
        reg.update(entries)


def quiet(*_):
    pass


def run(toy, **kw):
    opts = dict(name="toy", tasks=["toy_class", "toy_value"], seeds=[1, 2], max_budget=64,
                log=quiet)
    return al.evaluate(toy.vectors(), **{**opts, **kw})


def test_custom_tasks_and_strategy_end_to_end(toy):
    res = run(toy)
    assert res.protocol["name"] == "custom"
    assert list(res.curves["toy_class"]) == ["random", "typiclust", "badge", "lowest"]
    assert res.curves["toy_value"]["badge"]["status"] == "na"
    info = res.tasks["toy_class"]
    assert info["official"] is False and info["grid"] == [16, 23, 32, 45, 64]
    assert info["n_pool"] + info["n_val"] == 180 and info["chance"]["test"] == pytest.approx(1 / 3)
    first = {s: c["seeds"][1]["test"][0] for s, c in res.curves["toy_class"].items()}
    assert len(set(first.values())) == 1          # every strategy starts from the same labels
    a = res.aulc_low("toy_class", "random")
    assert a.shape == (2,) and np.all((a > 0.3) & (a <= 1))
    table = res.table()
    assert "toy_class*" in table and "not in the official suite" in table


def test_tasks_without_al_and_unknown_strategies_are_refused(toy):
    class Plain(bgcbench.Task):
        name, splits = "plain", {"R": "none"}

        def evaluate(self, reps, split, *, n_boot, device):
            return {"point": 0.0}

    with pytest.raises(ValueError, match="no active-learning protocol"):
        al.evaluate(toy.vectors(), tasks=[Plain()], log=quiet)
    with pytest.raises(ValueError, match="unknown strategy"):
        al.evaluate(toy.vectors(), tasks=["toy_class"], strategies=["nope"], log=quiet)


def test_missing_vectors_make_the_task_na(toy):
    vec = toy.vectors()
    vec.pop("B0001")
    res = al.evaluate(vec, name="part", tasks=["toy_class"], seeds=[1], max_budget=32,
                      log=quiet)
    assert res.tasks["toy_class"]["status"] == "na"
    assert "have no embedding" in res.tasks["toy_class"]["note"]


def test_cache_resumes_and_workers_give_the_same_curves(toy, tmp_path):
    lines = []
    res = run(toy, cache=tmp_path / "cells", strategies=["random", "lowest"], log=lines.append)
    assert "8 learning curves to compute" in lines[0]
    files = sorted((tmp_path / "cells" / "toy_class").glob("*.json"))
    assert [f.name for f in files] == ["lowest__seed1.json", "lowest__seed2.json",
                                       "random__seed1.json", "random__seed2.json"]
    lines.clear()
    again = run(toy, cache=tmp_path / "cells", strategies=["random", "lowest"], log=lines.append)
    assert "0 learning curves to compute, 8 from the cache" in lines[0]
    assert again.curves["toy_class"]["lowest"]["seeds"][2]["test"] == \
        res.curves["toy_class"]["lowest"]["seeds"][2]["test"]
    par = run(toy, strategies=["random", "lowest", "badge"], workers=2, modules=[str(TOY)])
    for task in ("toy_class", "toy_value"):
        for s in ("random", "lowest"):
            assert par.curves[task][s]["seeds"] == {
                k: {**v, "seconds": par.curves[task][s]["seeds"][k]["seconds"]}
                for k, v in res.curves[task][s]["seeds"].items()}


def test_result_round_trips_through_json(toy, tmp_path):
    res = run(toy, strategies=["random"])
    res.to_json(tmp_path / "toy.json")
    back = al.load(tmp_path / "toy.json")
    assert back.aulc_low("toy_class", "random").tolist() == \
        res.aulc_low("toy_class", "random").tolist()


def test_summary_compares_encoders(toy, tmp_path):
    paths = []
    for name, shift in (("strong", 2.0), ("weak", 0.3)):
        res = al.evaluate(toy.vectors(shift=shift), name=name, tasks=["toy_class"],
                          protocol="paper", seeds=[1, 2, 11, 12], max_budget=None, log=quiet)
        res.to_json(tmp_path / f"{name}.json")
        paths.append(tmp_path / f"{name}.json")
    summary = al.summarize(paths, reference="weak")
    assert set(summary.aulc["encoder"]) == {"strong", "weak"}
    assert len(summary.gains) == 2 * 3 and summary.gains["p_bh"].between(0, 1).all()
    assert set(summary.task_gains["strategy"]) == {"typiclust", "badge", "lowest"}
    strong = summary.savings.set_index("encoder").loc["strong", "savings"]
    assert strong > 1
    assert set(summary.selection["encoder"]) == {"strong", "weak"}
    assert set(summary.selected_curves["line"]) == {"selected", "random"}
    written = summary.write(tmp_path / "summary")
    assert (tmp_path / "summary" / "summary.md").read_text().startswith("# Active-learning")
    assert len(written) == 8
    other = al.evaluate(toy.vectors(), name="quick", tasks=["toy_class"], seeds=[1],
                        strategies=["random"], log=quiet)
    with pytest.raises(ValueError, match="one protocol"):
        al.summarize([paths[0], other])


def test_cli_al(toy, tmp_path, capsys):
    npz = tmp_path / "toyvec.npz"
    vec = toy.vectors()
    np.savez(npz, bgc_ids=np.array(list(vec)), embeddings=np.stack(list(vec.values())))
    out = tmp_path / "al"
    args = ["al", "evaluate", "--embeddings", str(npz), "--module", str(TOY), "--tasks",
            "toy_class", "--strategies", "random", "lowest", "--seeds", "1-2", "--max-budget",
            "45", "--out", str(out)]
    assert cli.main(args) == 0
    result = json.loads((out / "toyvec.json").read_text())
    assert result["curves"]["toy_class"]["lowest"]["seeds"]["2"]["n"] == [16, 23, 32, 45]
    assert (out / "cells" / "toyvec" / "toy_class" / "random__seed1.json").exists()
    assert cli.main(args + ["--name", "again"]) == 0
    capsys.readouterr()
    assert cli.main(["al", "summarize", str(out), "--reference", "toyvec"]) == 0
    assert "## toy_class" in capsys.readouterr().out
    assert (out / "summary" / "gains.csv").exists()
    assert cli.main(["al", "list", "--module", str(TOY)]) == 0
    listing = capsys.readouterr().out
    assert "toy_class" in listing and "lowest" in listing
    with pytest.raises(SystemExit):
        cli.main(["al", "evaluate", "--embeddings", str(npz), "--seeds", "x"])
