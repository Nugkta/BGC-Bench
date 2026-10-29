"""Extra tasks and representation kinds: registration, selection, caching and the CLI."""

import json

import numpy as np
import pytest

import bgcbench
from bgcbench import Task, cli, data, metrics, probes, representations
from bgcbench.encoders import base as encoders
from bgcbench.tasks import SUITE_V1, base


class ToyNRP(Task):
    """NRP vs. the rest on the coarse random split: a task added from outside the package."""

    name, metric = "toy", "NRP AUROC"
    splits = {"R": "t1ml_random"}

    def evaluate(self, reps, split, *, n_boot, device):
        b = reps["bgc"]
        Xa, df, coverage = data.align(b.ids, b.X, data.split(split))
        train = (df["split"] == "train").to_numpy()
        test = (df["split"] == "test").to_numpy()
        data.require_train_test(train, test)
        y = df["NRP"].to_numpy()
        proba = probes.logreg().fit(Xa[train], y[train]).predict_proba(Xa[test])
        out = metrics.auroc_with_ci(y[test], proba, multilabel=False, n_boot=n_boot, seed=42)
        return {**out, "n_test": int(test.sum()), **coverage}


class Mean(Task):
    """Needs no dataset: the mean of the vectors."""

    name, metric = "mean", "mean"
    splits = {"A": "none"}

    def evaluate(self, reps, split, *, n_boot, device):
        return {"point": float(reps["bgc"].X.mean())}


@pytest.fixture
def registry():
    saved = dict(base.REGISTRY), dict(representations.REGISTRY)
    yield
    for reg, entries in zip((base.REGISTRY, representations.REGISTRY), saved):
        reg.clear()
        reg.update(entries)


def quiet(*_):
    pass


def vectors(n=3, d=4):
    return {f"B{i}": np.full(d, i, dtype=np.float32) for i in range(n)}


def test_extra_task_runs_next_to_official_ones_and_is_marked(dataset, home, registry):
    bgcbench.register_task(ToyNRP)
    res = bgcbench.run("bag_of_pfam", tasks=["toy", "coarse"], n_boot=20, log=quiet)
    assert list(res.cells) == ["coarse_R", "coarse_S", "coarse_G", "toy_R"]
    toy = res.cells["toy_R"]
    assert 0.5 < toy["point"] <= 1 and toy["ci"][0] <= toy["point"] <= toy["ci"][1]
    assert res.tasks == {"coarse": {"version": "1", "metric": "macro-AUROC", "official": True},
                         "toy": {"version": "1", "metric": "NRP AUROC", "official": False}}
    table = res.table()
    assert "toy_R*" in table and "coarse_R*" not in table
    assert table.splitlines()[-1] == "* not in the official suite: toy (v1)"
    res.to_json(home / "r.json")
    assert json.loads((home / "r.json").read_text())["tasks"]["toy"]["official"] is False


def test_default_is_the_official_suite(registry):
    bgcbench.register_task(Mean)
    assert "mean" in bgcbench.list_tasks()
    assert [t.name for t in base.select(None)] == list(SUITE_V1)
    assert [t.name for t in base.select(["mean", "fine"])] == ["fine", "mean"]


def test_names_are_unique_and_official_tasks_stay(registry):
    bgcbench.register_task(Mean)
    bgcbench.register_task(Mean)  # the same class again, e.g. a re-imported module
    other = type("OtherMean", (Mean,), {})
    with pytest.raises(ValueError, match="already registered by"):
        bgcbench.register_task(other)
    bgcbench.register_task(other, replace=True)
    assert type(base.get_task("mean")) is other
    impostor = type("Coarse", (Mean,), {"name": "coarse"})
    for cls in (impostor, type(base.get_task("coarse"))):
        with pytest.raises(ValueError, match="official task"):
            bgcbench.register_task(cls, replace=True)
    with pytest.raises(ValueError, match="no splits"):
        bgcbench.register_task(type("Empty", (Task,), {"name": "empty"}))


def test_task_instances_register_themselves(registry):
    for _ in range(2):
        res = bgcbench.evaluate(vectors(), tasks=[Mean()], log=quiet)
    assert res.cells["mean_A"]["point"] == 1.0
    assert np.isnan(res.cells["mean_A"]["ci"]).all()
    assert "mean" in bgcbench.list_tasks()


def test_unknown_task_fails_before_embedding(registry):
    class Never(bgcbench.Encoder):
        name = "never"

        def embed_bgcs(self, inputs):
            raise AssertionError("must not embed")

    with pytest.raises(ValueError, match="unknown task"):
        bgcbench.run(Never(device="cpu"), tasks=["coarse", "nonsense"])


def test_missing_representation_is_na(registry):
    class NeedsGenes(Mean):
        name, requires = "genes", ("bgc", "gene")

    res = bgcbench.evaluate(vectors(), tasks=[NeedsGenes()], log=quiet)
    assert res.cells == {"genes_A": {"status": "na", "note": "no gene vectors"}}


@pytest.mark.parametrize("cell, error", [
    (None, "returned NoneType"), ({"ci": [0, 1]}, "'point'"), ({"point": "0.5"}, "'point'"),
    ({"point": float("inf")}, "'point'"), ({"point": 0.5, "ci": [0.1]}, "'ci'"),
    ({"point": 0.5, "ci": [None, None]}, "'ci'"),
])
def test_malformed_task_output_names_the_task(registry, cell, error):
    class Bad(Mean):
        name = "bad"

        def evaluate(self, reps, split, *, n_boot, device):
            return cell

    with pytest.raises((TypeError, ValueError), match=rf"task 'bad' \(bad_A\).*{error}"):
        bgcbench.evaluate(vectors(), tasks=[Bad()], log=quiet)


def test_cli_task_module(tmp_path, registry, capsys):
    mod = tmp_path / "my_tasks.py"
    mod.write_text("import bgcbench\n\n"
                   "@bgcbench.register_task\n"
                   "class Max(bgcbench.Task):\n"
                   "    name, metric, splits = 'max', 'max', {'A': 'none'}\n\n"
                   "    def evaluate(self, reps, split, *, n_boot, device):\n"
                   "        return {'point': float(reps['bgc'].X.max())}\n")
    emb = tmp_path / "mine.npz"
    np.savez(emb, bgc_ids=np.array(["B0", "B1"]), embeddings=np.array([[0.0, 1.0], [2.0, 3.0]]))
    out = tmp_path / "mine.json"
    assert cli.main(["evaluate", "--embeddings", str(emb), "--task-module", str(mod),
                     "--tasks", "max", "--n-boot", "0", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["cells"]["max_A"]["point"] == 3.0
    assert "max_A*" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["evaluate", "--embeddings", str(emb), "--tasks", "nonsense", "--n-boot", "0",
                  "--out", str(out)])
    assert "unknown task" in capsys.readouterr().err


# Representation kinds
class Counts(Task):
    """Item counts of the adomain, molecule and token vectors it is given."""

    name, metric, splits = "counts", "count", {"A": "none"}
    requires = ("token",)

    def evaluate(self, reps, split, *, n_boot, device):
        return {"point": float(sum(len(e.ids) for e in reps.values())),
                "kinds": sorted(reps)}


class TokenEncoder(bgcbench.Encoder):
    name, input = "token_encoder", "pfam_acc"
    calls = 0

    def embed_bgcs(self, inputs):
        type(self).calls += 1
        return {b: np.ones(2) for b in inputs}

    def embed_tokens(self, tokens, inputs):
        return {t: np.full(3, i) for i, t in enumerate(tokens)}


@pytest.fixture
def tokens(registry, home, no_dataset, monkeypatch):
    bgcbench.register_representation("token", "token_ids", method="embed_tokens",
                                     table=lambda: ["t1", "t2", "t3"])
    monkeypatch.setattr(encoders, "load_inputs", lambda kind: {"B0": ["PF1"], "B1": ["PF2"]})
    monkeypatch.setattr(encoders, "adomain_table", lambda: None)
    TokenEncoder.calls = 0


def test_new_representation_end_to_end_and_cached(tokens):
    for _ in range(2):
        res = bgcbench.run(TokenEncoder(device="cpu"), tasks=[Counts()], n_boot=0, log=quiet)
        assert res.cells["counts_A"]["point"] == 3 and res.cells["counts_A"]["kinds"] == ["token"]
    assert TokenEncoder.calls == 1  # the second run read the cache
    cdir = TokenEncoder(device="cpu").cache_dir()
    manifest = json.loads((cdir / "manifest.json").read_text())
    assert manifest["kinds"] == {"bgc": 2, "token": 3} and manifest["device_type"] == "cpu"
    with np.load(cdir / "token.npz") as z:
        assert z["token_ids"].tolist() == ["t1", "t2", "t3"]
    bgc, ad = bgcbench.embed(TokenEncoder(device="cpu"))
    assert len(bgc.ids) == 2 and ad is None


def test_representation_names_are_unique(registry):
    bgcbench.register_representation("token", "token_ids", method="embed_tokens")
    bgcbench.register_representation("token", "token_ids", method="embed_tokens")
    with pytest.raises(ValueError, match="already registered"):
        bgcbench.register_representation("token", "ids", method="embed_tokens")
    with pytest.raises(ValueError, match="built-in"):
        bgcbench.register_representation("adomain", "domain_ids", replace=True)


def test_cli_rep_matches_the_adomain_and_chem_embedding_flags(tmp_path, registry, capsys):
    bgcbench.register_representation("token", "token_ids")

    class AllKinds(Counts):
        name, requires = "all_kinds", ("token", "adomain", "molecule")

    bgcbench.register_task(AllKinds)
    files = {}
    for kind, key, n in [("bgc", "bgc_ids", 2), ("token", "token_ids", 3),
                         ("adomain", "domain_ids", 4), ("molecule", "compound_ids", 5)]:
        files[kind] = tmp_path / f"{kind}.npz"
        np.savez(files[kind], **{key: np.array([f"x{i}" for i in range(n)]),
                                 "embeddings": np.ones((n, 2))})
    out = tmp_path / "out.json"
    common = ["evaluate", "--embeddings", str(files["bgc"]), "--rep", f"token={files['token']}",
              "--tasks", "all_kinds", "--n-boot", "0", "--out", str(out)]
    for flags in (["--adomain-embeddings", str(files["adomain"]),
                   "--chem-embeddings", str(files["molecule"])],
                  ["--rep", f"adomain={files['adomain']}",
                   "--rep", f"molecule={files['molecule']}"]):
        assert cli.main(common + flags) == 0
        cell = json.loads(out.read_text())["cells"]["all_kinds_A"]
        assert cell["point"] == 12 and cell["kinds"] == ["adomain", "molecule", "token"]
    with pytest.raises(SystemExit):
        cli.main(common + ["--adomain-embeddings", str(files["adomain"]),
                           "--rep", f"adomain={files['adomain']}"])
    assert "not both" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(common + ["--rep", "token"])
    assert "KIND=PATH" in capsys.readouterr().err


def test_cli_chem_encoder_names_the_molecule_vectors(tmp_path, registry, monkeypatch, capsys):
    shipped = tmp_path / "data" / "tasks" / "chem_encoders"
    shipped.mkdir(parents=True)
    (shipped / "encoders.json").write_text(json.dumps({"ecfp4": {"dim": 2}}))
    np.savez(shipped / "ecfp4.npz", compound_ids=np.array(["c0", "c1", "c2"]),
             embeddings=np.ones((3, 2)))
    monkeypatch.setenv("BGCBENCH_DATA", str(tmp_path / "data"))
    data.MAIN.cache_clear()

    class Molecules(Counts):
        name, requires = "molecules", ("molecule",)

    bgcbench.register_task(Molecules)
    emb, out = tmp_path / "mine.npz", tmp_path / "out.json"
    np.savez(emb, bgc_ids=np.array(["B0"]), embeddings=np.ones((1, 2)))
    common = ["evaluate", "--embeddings", str(emb), "--tasks", "molecules", "--n-boot", "0",
              "--out", str(out)]
    try:
        assert cli.main(common + ["--chem-encoder", "ecfp4"]) == 0
        assert "(molecule: ecfp4)" in capsys.readouterr().out
        with pytest.raises(SystemExit):
            cli.main(common + ["--chem-encoder", "nope"])
        assert "unknown chemical encoder 'nope'; choose from ecfp4" in capsys.readouterr().err
    finally:
        data.MAIN.cache_clear()
    res = json.loads(out.read_text())
    assert res["cells"]["molecules_A"]["point"] == 3
    assert res["tasks"]["molecules"]["replaced"] == {"molecule": "ecfp4"}
    with pytest.raises(SystemExit):
        cli.main(common + ["--chem-encoder", "ecfp4", "--chem-embeddings", str(emb)])


# Task-owned data
def write_split(root, rows):
    (root / "splits").mkdir(parents=True)
    (root / "splits" / "own.csv").write_text(
        "bgc_id,split\n" + "".join(f"{b},{s}\n" for b, s in rows))


def test_task_reads_its_own_dataset(tmp_path, registry, monkeypatch):
    own, main = tmp_path / "own", tmp_path / "main"
    write_split(own, [("B0", "train"), ("B1", "train"), ("B2", "test")])
    write_split(main, [("B0", "test")])
    monkeypatch.setenv("BGCBENCH_DATA", str(main))  # the main dataset only
    data.MAIN.cache_clear()

    class OwnSplit(Task):
        name, metric, splits, dataset = "own", "test rows", {"A": "own"}, str(own)

        def evaluate(self, reps, split, *, n_boot, device):
            b = reps["bgc"]
            _, df, coverage = data.align(b.ids, b.X, self.data.split(split))
            return {"point": float((df["split"] == "test").sum()), **coverage}

    try:
        res = bgcbench.evaluate(vectors(4), tasks=[OwnSplit()], log=quiet)
        assert data.split("own")["bgc_id"].tolist() == ["B0"]
    finally:
        data.MAIN.cache_clear()
    cell = res.cells["own_A"]
    assert cell["point"] == 1 and cell["n_split"] == 3
    assert res.tasks["own"]["dataset"] == str(own)


def test_hub_datasets_are_pinned_and_downloaded_under_home(home, monkeypatch):
    import huggingface_hub

    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        kwargs["local_dir"].mkdir(parents=True)
        return str(kwargs["local_dir"])

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    ds = data.Dataset("someone/extra-data@v2")
    assert ds.root() == home / "data" / "someone" / "extra-data" / "v2"
    assert calls[0]["repo_id"] == "someone/extra-data" and calls[0]["revision"] == "v2"
    for source in ("someone/extra-data", "no-such-directory"):
        with pytest.raises(FileNotFoundError, match="org/repo@revision"):
            data.Dataset(source).root()
