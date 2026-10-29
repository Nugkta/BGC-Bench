"""Exercise the full shell workflow without downloading weights or using a GPU."""

import csv
import json
import os
import subprocess
from pathlib import Path
from textwrap import dedent

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/reproduce_main_table.sh"
ENCODERS = (
    "bag_of_pfam", "pfam_rand", "bigcarp", "kawano", "deepbgc", "aa_kmer3",
    "bgc_mac", "bgcat", "bgc_prophet", "lcplm", "kmer6", "evo2_7b",
)


@pytest.fixture
def runner(tmp_path, monkeypatch):
    bin_dir = tmp_path / "mock bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(dedent("""\
        #!/usr/bin/env python3
        import json
        import os
        import sys
        from pathlib import Path

        args = sys.argv[1:]
        project = Path(args[args.index("--project") + 1])
        with open(os.environ["COMMAND_LOG"], "a") as log:
            log.write(json.dumps({
                "args": args, "project": str(project),
                "cache": os.environ.get("BGCBENCH_HOME"),
                "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }) + "\\n")
        assert "--frozen" in args
        if args[0] == "sync":
            step = "setup_gpu" if project.name == "gpu" else "setup_main"
        else:
            args = args[args.index("--no-sync") + 1:]
            if args[:2] == ["python", "-c"]:
                step = "check_gpu"
            elif args[0] == "python":
                os.execv(sys.executable, [sys.executable, *args[1:]])
            else:
                encoder = args[args.index("--encoder") + 1]
                step = args[1] + "_" + encoder

        if os.environ.get("FAIL_STEP") == step:
            print("Deliberate failure: " + step, file=sys.stderr)
            sys.exit(17)
        if step.startswith("run_"):
            cells = {}
            for task, letters in (("coarse", "RSG"), ("fine", "RSG"),
                                  ("chem", "RSG"), ("retrieval", "RSGC"),
                                  ("substrate", "FG")):
                for letter in letters:
                    cells[task + "_" + letter] = (
                        {"status": "na", "note": "no adomain vectors"}
                        if encoder == "deepbgc" and task == "substrate"
                        else {"point": 0.123456789, "ci": [0.1, 0.2]}
                    )
            out = Path(args[args.index("--out") + 1])
            out.write_text(json.dumps({"encoder": encoder, "cells": cells}))
    """))
    fake_uv.chmod(0o755)
    log = tmp_path / "commands.jsonl"
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("COMMAND_LOG", str(log))
    monkeypatch.setenv("BGCBENCH_HOME", str(tmp_path / "shared cache"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")

    def run(*args):
        return subprocess.run(
            ["bash", str(SCRIPT), *args], cwd=tmp_path, text=True,
            capture_output=True, timeout=30,
        )

    return run, log


def test_reproduce_all_encoders_and_summarise(runner, tmp_path):
    run, log = runner
    completed = run("paper results")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    out = tmp_path / "paper results"
    assert {p.stem for p in out.glob("*.json")} == set(ENCODERS)
    with (out / "main_table.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["encoder"] for row in rows] == list(ENCODERS)
    assert len(rows[0]) == 16
    assert rows[0]["coarse_R"] == "0.123456789"
    deepbgc = next(row for row in rows if row["encoder"] == "deepbgc")
    assert deepbgc["substrate_F"] == deepbgc["substrate_G"] == ""
    assert "| deepbgc |" in (out / "main_table.md").read_text()
    assert "-- | -- |" in (out / "main_table.md").read_text()
    assert json.loads((out / "bigcarp.json").read_text())["cells"]["coarse_R"]["ci"] == [0.1, 0.2]

    calls = [json.loads(line) for line in log.read_text().splitlines()]
    scoring = [c for c in calls if "bgcbench" in c["args"]
               and c["args"][c["args"].index("bgcbench") + 1] == "run"]
    embedding = [c for c in calls if "embed" in c["args"]]
    assert len(scoring) == 12
    assert all(c["project"] == str(ROOT) for c in scoring)
    assert len(embedding) == 2
    assert all(c["project"] == str(ROOT / "envs/gpu") for c in embedding)
    assert all(c["cache"] == str(tmp_path / "shared cache") for c in calls)
    assert all(c["cuda"] == "2" for c in calls)
    assert "Results saved to" in completed.stdout
    assert (out / "logs/run_evo2_7b.log").is_file()


@pytest.mark.parametrize("step", ["setup_gpu", "check_gpu", "run_bgcat", "embed_lcplm"])
def test_failure_stops_pipeline_without_publishing_a_table(runner, tmp_path, monkeypatch, step):
    run, log = runner
    monkeypatch.setenv("FAIL_STEP", step)
    completed = run("failed results")
    assert completed.returncode == 17
    out = tmp_path / "failed results"
    assert not (out / "main_table.csv").exists()
    assert not (out / "main_table.md").exists()
    assert "Deliberate failure: " + step in (out / f"logs/{step}.log").read_text()
    assert "Results saved to" not in completed.stdout
    assert "evo2_7b" not in log.read_text()


def test_help_does_not_start_setup(runner, tmp_path):
    run, log = runner
    completed = run("--help")
    assert completed.returncode == 0
    assert "OUTPUT_DIR" in completed.stdout
    assert not log.exists()
