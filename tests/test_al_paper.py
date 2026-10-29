"""The built-in AL tasks against the paper's active-learning run: the fixed partitions and seed
orderings, and the first rounds of Bag-of-Pfam's seed-1 curves."""

import hashlib

import numpy as np
import pytest

from bgcbench import al
from bgcbench.tasks import get_task

# sha256 over the row ids, pool / validation / test rows, budget grid and the 20 seed
# permutations of the paper protocol, as in the paper's run.
DESIGN = {
    "coarse": "c13232d3eaec91dcb5a7e9e0c7cf897961f8138a25a19fac2da3d41ccc546057",
    "fine": "daeb640c46788e07e0d9f0fcf97d3293520ef070cd47f36967825270e9f6b842",
    "chem": "3b2f87e6e7f6d9cf4c0964a7b7097d689b9cd7de747c2a8eeb029d880ad40510",
    "retrieval": "4955785b19275145f7f5521162cf061447738f1c3530aaada5147398324f083a",
    "substrate": "84d2faaf80fa0b127d1e49a1c8b8be5862a13416542b2a3207b4ef856c878cc8",
}
SIZES = {"coarse": (1553, 388, 487), "fine": (784, 196, 241), "chem": (1248, 312, 379),
         "retrieval": (1242, 310, 387), "substrate": (1296, 325, 406)}

# (val, test) at n = 16, 23, 32 of Bag-of-Pfam, seed 1, in the paper's run.
PAPER = {
    ("coarse", "random"): [(0.8627785445234285, 0.8870451967241969),
                           (0.8588352095694999, 0.8847561965821354),
                           (0.8134752499584209, 0.8598103498812376)],
    ("coarse", "typiclust"): [(0.8627785445234285, 0.8870451967241969),
                              (0.8762381794486931, 0.8950258282111583),
                              (0.8669723320985138, 0.8861224482126469)],
    ("coarse", "badge"): [(0.8627785445234285, 0.8870451967241969),
                          (0.8568887255013197, 0.8834792480657732),
                          (0.8403948579237736, 0.8655088171038111)],
    ("chem", "random"): [(0.2868312163207656, 0.3028257129480317),
                         (0.23401509728791536, 0.2843064141389633),
                         (0.3741763940434427, 0.3425071876845716)],
    ("chem", "typiclust"): [(0.2868312163207656, 0.3028257129480317),
                            (0.2472926064970474, 0.27467784629990366),
                            (0.26206192316088217, 0.2731997097502194)],
    ("fine", "random"): [(0.578182165440988, 0.5720152027084547),
                         (0.6449188749023042, 0.6363541489792405),
                         (0.6908339865616019, 0.6768274104662394)],
    ("fine", "typiclust"): [(0.578182165440988, 0.5720152027084547),
                            (0.7079277642495895, 0.688336013773665),
                            (0.7586593208244136, 0.7323454265760659)],
    ("fine", "badge"): [(0.578182165440988, 0.5720152027084547),
                        (0.6358392025387427, 0.6252818485863302),
                        (0.6739420547985226, 0.6547944585591237)],
    ("substrate", "random"): [(0.08307692307692308, 0.07389162561576355),
                              (0.09230769230769231, 0.06896551724137931),
                              (0.12, 0.12807881773399016)],
    ("substrate", "typiclust"): [(0.08307692307692308, 0.07389162561576355),
                                 (0.08307692307692308, 0.09359605911330049),
                                 (0.08923076923076922, 0.09113300492610837)],
    ("substrate", "badge"): [(0.08307692307692308, 0.07389162561576355),
                             (0.07692307692307693, 0.07881773399014778),
                             (0.06461538461538462, 0.07389162561576355)],
    ("retrieval", "random"): [(0.11299484812094178, 0.07219470608156531),
                              (0.13081840387553204, 0.06678733233340733),
                              (0.1693951483262052, 0.08816132677139651)],
    ("retrieval", "typiclust"): [(0.11299484812094178, 0.07219470608156531),
                                 (0.10201310955299632, 0.11599352344273872),
                                 (0.10181154980434447, 0.11224498937582662)],
    ("retrieval", "badge"): [(0.11299484812094178, 0.07219470608156531),
                             (0.15266371789229763, 0.10125797886700313),
                             (0.19748503017987679, 0.12450362510546656)],
}


@pytest.mark.parametrize("name", list(DESIGN))
def test_partitions_and_seed_orderings_match_the_paper(dataset, name):
    task = get_task(name)
    rows = task.al_rows()
    part = al.partition(rows, task.al_stream, al.protocol("paper"))
    assert (len(part.pool), len(part.val), len(part.test)) == SIZES[name]
    h = hashlib.sha256("\x1f".join(rows.ids.tolist()).encode())
    for a in (part.pool, part.val, part.test, np.asarray(part.grid)):
        h.update(np.asarray(a, dtype=np.int64).tobytes())
    for seed in range(1, 21):
        h.update(part.permutation(seed).tobytes())
    assert h.hexdigest() == DESIGN[name]


def test_built_in_universe_is_every_bgc_the_encoders_embed(dataset):
    from bgcbench import data
    from bgcbench.al import design

    assert len(design.pfam_universe(data.MAIN)) == 2626


def test_bag_of_pfam_curves_match_the_paper(dataset, home):
    res = al.run("bag_of_pfam", seeds=[1], max_budget=32, log=lambda *a: None)
    assert res.curves["chem"]["badge"]["status"] == "na"
    for (task, strategy), expected in PAPER.items():
        cell = res.curves[task][strategy]["seeds"][1]
        tol = 1e-6 if task == "retrieval" else 1e-9     # torch training: last-digit drift
        assert cell["n"] == [16, 23, 32]
        got = list(zip(cell["val"], cell["test"]))
        assert np.allclose(got, expected, rtol=0, atol=tol), (task, strategy, got)


def test_cli_al_run_embeds_and_runs(dataset, home, tmp_path):
    from bgcbench import cli

    out = tmp_path / "al"
    assert cli.main(["al", "run", "--encoder", "pfam_rand", "--device", "cpu", "--tasks", "chem",
                     "coarse", "--strategies", "random", "--seeds", "1", "--max-budget", "23",
                     "--out", str(out)]) == 0
    res = al.load(out / "pfam_rand.json")
    assert res.protocol["name"] == "custom" and res.env["embedding_device"]
    assert res.curves["chem"]["random"]["seeds"]["1"]["n"] == [16, 23]
