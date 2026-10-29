"""The active-learning reproduction script, with a stand-in for uv (no data, weights or GPU)."""

import csv
import json
import os
import subprocess
from pathlib import Path
from textwrap import dedent

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/reproduce_al.sh"
ENCODERS = ("bag_of_pfam", "pfam_rand", "bigcarp", "kawano", "deepbgc", "bgc_mac", "bgcat",
            "bgc_prophet", "lcplm", "evo2_7b", "kmer6", "aa_kmer3")


@pytest.fixture
def runner(tmp_path, monkeypatch):
    bin_dir = tmp_path / "mock bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(dedent("""\
        #!/usr/bin/env python3
        import csv
        import json
        import os
        import sys
        from pathlib import Path

        args = sys.argv[1:]
        project = Path(args[args.index("--project") + 1])
        with open(os.environ["COMMAND_LOG"], "a") as log:
            log.write(json.dumps({"args": args, "project": str(project)}) + "\\n")
        if args[0] == "sync":
            sys.exit(0)
        args = args[args.index("--no-sync") + 1:]
        if args[0] == "python":
            os.execv(sys.executable, [sys.executable, *args[1:]])
        if args[1] == "embed":
            encoder = args[args.index("--encoder") + 1]
            if project.name != "gpu" and encoder in os.environ.get("NOT_CACHED", "").split():
                print(f"{encoder} needs the GPU environment", file=sys.stderr)
                sys.exit(1)
            sys.exit(0)
        if args[1:3] == ["al", "summarize"]:
            out = Path(args[args.index("--out") + 1])
            out.mkdir(parents=True)
            val = {"kmer6": 0.9, "bgcat": 0.8, "bag_of_pfam": 0.5}
            with (out / "selection.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["task", "encoder", "strategy", "val_aulc_low_selection",
                            "val_aulc_low_selection_random"])
                for task in ("coarse", "substrate"):
                    for enc in json.loads(os.environ["ENCODERS"]):
                        w.writerow([task, enc, "typiclust", val.get(enc, 0.1), val.get(enc, 0.1)])
            with (out / "selected_curves.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["task", "encoder", "strategy", "line", "n", "mean", "band_lo",
                            "band_hi"])
                for task in ("coarse", "substrate"):
                    for enc in json.loads(os.environ["ENCODERS"]):
                        for line, s in (("selected", "typiclust"), ("random", "random")):
                            for n in (16, 23):
                                w.writerow([task, enc, s, line, n, 0.5, 0.4, 0.6])
    """))
    fake_uv.chmod(0o755)
    log = tmp_path / "commands.jsonl"
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("COMMAND_LOG", str(log))
    monkeypatch.setenv("ENCODERS", json.dumps(ENCODERS))
    monkeypatch.setenv("BGCBENCH_WORKERS", "3")

    def run(*args):
        return subprocess.run(["bash", str(SCRIPT), *args], cwd=tmp_path, text=True,
                              capture_output=True, timeout=60)

    return run, log


def test_reproduce_al_runs_the_paper_protocol_and_writes_the_figure_lines(runner, tmp_path,
                                                                           monkeypatch):
    run, log = runner
    monkeypatch.setenv("NOT_CACHED", "evo2_7b")
    completed = run("al results")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    calls = [json.loads(line)["args"] for line in log.read_text().splitlines()]
    al_run = next(c for c in calls if "al" in c and "run" in c)
    assert al_run[al_run.index("--workers") + 1] == "3"
    assert al_run[al_run.index("--protocol") + 1] == "paper"
    assert tuple(al_run[al_run.index("--encoder") + 1:]) == ENCODERS
    gpu = [json.loads(line) for line in log.read_text().splitlines()
           if json.loads(line)["project"].endswith("envs/gpu")]
    assert [c["args"][0] for c in gpu] == ["sync", "run"]
    assert "evo2_7b" in gpu[1]["args"]

    with (tmp_path / "al results/summary/figure_lines.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    lines = {(r["task"], r["role"], r["encoder"], r["line"]) for r in rows}
    assert ("coarse", "baseline", "kmer6", "random") in lines
    assert ("coarse", "protein_lm", "bgcat", "selected") in lines
    assert ("substrate", "evo2", "evo2_7b", "random") in lines
    assert not any(role == "baseline" and line == "selected" for _, role, _, line in lines)
    assert "Figure data saved to" in completed.stdout


def test_help_and_bad_arguments(runner):
    run, log = runner
    assert run("--help").returncode == 0
    assert run("a", "b").returncode == 2
    assert not log.exists()
