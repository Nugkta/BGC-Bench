"""Cross-encoder summary of active-learning results.

From the :class:`ALResult` of several encoders (same protocol), :func:`summarize` builds:

``aulc``            AULC_low per encoder, task and strategy (validation and test seed means,
                    test 95 % seed-bootstrap interval)
``gains``           seed-paired test AULC_low difference of each strategy from random
                    acquisition per encoder: 95 % interval and p-value from 10,000 seed
                    resamples, Benjamini-Hochberg across every row
``task_gains``      the difference averaged over a task's encoders within each seed: exact
                    sign-flip test over seeds, Holm across rows
``savings``         labels saved under random acquisition relative to a reference encoder
                    (geometric mean over the reference's scores at 32, 64 and 128 labels)
``selection``       with selection and report seeds (the paper protocol): each encoder's
                    strategy with the best validation AULC_low on the selection seeds, and its
                    test AULC_low on the report seeds
``selected_curves`` the selected strategy's and random acquisition's report-seed mean test
                    curves with 95 % seed-bootstrap bands
``curves``          every round of every curve
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import stats
from .runner import ALResult, load

ALPHA = 0.05


@dataclass
class Summary:
    protocol: dict
    aulc: pd.DataFrame
    gains: pd.DataFrame
    task_gains: pd.DataFrame
    savings: pd.DataFrame
    selection: pd.DataFrame
    selected_curves: pd.DataFrame
    curves: pd.DataFrame
    notes: list[str] = field(default_factory=list)

    TABLES = ("aulc", "gains", "task_gains", "savings", "selection", "selected_curves", "curves")

    def write(self, out) -> list[Path]:
        """Write every table as ``<name>.csv`` and the Markdown overview ``summary.md``."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for name in self.TABLES:
            path = out / f"{name}.csv"
            getattr(self, name).to_csv(path, index=False)
            paths.append(path)
        (out / "summary.md").write_text(self.markdown() + "\n")
        return paths + [out / "summary.md"]

    def markdown(self) -> str:
        p = self.protocol
        budget = "the full pool" if p["max_budget"] is None else f"{p['max_budget']} labels"
        lines = ["# Active-learning summary", "",
                 f"Protocol `{p['name']}`: seeds {_seed_text(p['seeds'])}, budgets up to "
                 f"{budget}. "
                 "Scores are test AULC_low (16-256 labels, log2 axis) unless noted.", ""]
        for task in self.aulc["task"].unique():
            sub = self.aulc[self.aulc["task"] == task]
            piv = sub.pivot(index="encoder", columns="strategy", values="test_aulc_low")
            piv = piv[[s for s in sub["strategy"].unique()]]
            lines += [f"## {task}", "", _md_table(piv.reset_index(), "{:.4f}"), ""]
            g = self.gains[self.gains["task"] == task]
            if len(g):
                tab = g.assign(delta=[f"{d:+.4f}{' *' if s else ''}" for d, s in
                                      zip(g["delta"], g["significant"])])
                piv = tab.pivot(index="encoder", columns="strategy", values="delta")
                lines += ["Difference from random acquisition (* BH-adjusted p < 0.05):", "",
                          _md_table(piv.reset_index(), "{}"), ""]
        if len(self.task_gains):
            lines += ["## Encoder-mean gains", "",
                      _md_table(self.task_gains[["task", "strategy", "delta", "p", "p_holm",
                                                 "n_encoders"]], "{:.4g}"), ""]
        if len(self.savings):
            piv = self.savings.pivot(index="encoder", columns="task", values="savings_text")
            lines += [f"## Label savings vs {self.savings['reference'].iloc[0]} (random)", "",
                      _md_table(piv.reset_index(), "{}"), ""]
        if len(self.selection):
            lines += ["## Validation-selected strategies (report seeds)", "",
                      _md_table(self.selection[["task", "encoder", "strategy",
                                                "val_aulc_low_selection",
                                                "test_aulc_low_report",
                                                "test_aulc_low_report_random"]], "{:.4f}"), ""]
        lines += [f"- {n}" for n in self.notes]
        return "\n".join(lines).rstrip()


def _seed_text(seeds) -> str:
    seeds = list(seeds)
    if seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"{seeds[0]}-{seeds[-1]}"
    return ", ".join(map(str, seeds))


def _md_table(df: pd.DataFrame, fmt: str) -> str:
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "--" if not np.isfinite(v) else fmt.format(v)
        return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def _ok(res: ALResult, task: str, strategy: str) -> bool:
    return res.curves.get(task, {}).get(strategy, {}).get("status") == "ok"


def summarize(results, reference: str | None = "bag_of_pfam") -> Summary:
    """Summarise :class:`ALResult` objects or result JSON paths of one protocol."""
    results = [load(r) if not isinstance(r, ALResult) else r for r in results]
    if not results:
        raise ValueError("no results to summarise")
    names = [r.name for r in results]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate encoder names: {names}")
    proto = results[0].protocol
    for r in results[1:]:
        if r.protocol != proto:
            raise ValueError(f"{r.name} used protocol {r.protocol}, {results[0].name} "
                             f"{proto}; summarise results of one protocol")
    notes: list[str] = []
    tasks = list(dict.fromkeys(t for r in results for t in r.curves))
    aulc_rows, gain_rows, curve_rows = [], [], []
    for r in results:
        curve_rows += r.long_rows()
        for task in tasks:
            for strategy, cell in r.curves.get(task, {}).items():
                if cell.get("status") != "ok":
                    continue
                val, test = r.aulc_low(task, strategy, "val"), r.aulc_low(task, strategy)
                b = stats.seed_bootstrap(test)
                aulc_rows.append({"encoder": r.name, "task": task, "strategy": strategy,
                                  "n_seeds": len(test), "val_aulc_low": float(val.mean()),
                                  "test_aulc_low": float(test.mean()),
                                  "test_ci_lo": b["ci"][0], "test_ci_hi": b["ci"][1]})
                if strategy != "random" and _ok(r, task, "random"):
                    d = test - r.aulc_low(task, "random")
                    b = stats.seed_bootstrap(d)
                    gain_rows.append({"encoder": r.name, "task": task, "strategy": strategy,
                                      "delta": b["mean"], "ci_lo": b["ci"][0],
                                      "ci_hi": b["ci"][1], "p": b["p"],
                                      "n_positive": b["n_positive"], "n_seeds": b["n"]})
    gains = pd.DataFrame(gain_rows, columns=["encoder", "task", "strategy", "delta", "ci_lo",
                                             "ci_hi", "p", "n_positive", "n_seeds"])
    if len(gains):
        gains["p_bh"] = stats.bh(gains["p"].to_numpy())
        gains["significant"] = gains["p_bh"] < ALPHA
    else:
        gains["p_bh"], gains["significant"] = [], []
    return Summary(proto, pd.DataFrame(aulc_rows), gains, _task_gains(results, tasks, notes),
                   _savings(results, tasks, reference, notes),
                   *_selection(results, tasks, proto, notes),
                   pd.DataFrame(curve_rows), notes)


def _task_gains(results, tasks, notes) -> pd.DataFrame:
    rows = []
    for task in tasks:
        strategies = list(dict.fromkeys(s for r in results for s in r.curves.get(task, {})))
        for strategy in strategies:
            if strategy == "random":
                continue
            encs = [r for r in results if _ok(r, task, strategy) and _ok(r, task, "random")]
            if not encs:
                continue
            D = np.stack([r.aulc_low(task, strategy) - r.aulc_low(task, "random") for r in encs])
            per_seed = D.mean(axis=0)
            if len(per_seed) <= 24:
                p, test = stats.sign_flip_p(per_seed), "sign-flip"
            else:
                p, test = stats.seed_bootstrap(per_seed)["p"], "bootstrap"
            rows.append({"task": task, "strategy": strategy, "delta": float(per_seed.mean()),
                         "p": p, "test": test, "n_encoders": len(encs),
                         "n_seeds": len(per_seed)})
    out = pd.DataFrame(rows, columns=["task", "strategy", "delta", "p", "test", "n_encoders",
                                      "n_seeds"])
    out["p_holm"] = stats.holm(out["p"].to_numpy()) if len(out) else []
    out["significant"] = out["p_holm"] < ALPHA
    if len(out) and len(results) == 1:
        notes.append("encoder-mean gains come from a single encoder")
    return out


def _savings(results, tasks, reference, notes) -> pd.DataFrame:
    cols = ["encoder", "task", "reference", "savings", "flag", "savings_text"]
    if reference is None:
        return pd.DataFrame(columns=cols)
    ref = {r.name: r for r in results}.get(reference)
    if ref is None:
        notes.append(f"no label savings: the reference encoder {reference!r} is not among the "
                     f"results")
        return pd.DataFrame(columns=cols)
    rows = []
    for task in tasks:
        if not _ok(ref, task, "random"):
            continue
        _, ns, R = ref.scores(task, "random")
        if not set(stats.SAVINGS_REF_POINTS) <= set(ns):
            notes.append(f"no label savings on {task}: the grid lacks "
                         f"{list(stats.SAVINGS_REF_POINTS)}")
            continue
        for r in results:
            if not _ok(r, task, "random"):
                continue
            _, ns_t, T = r.scores(task, "random")
            if ns_t != ns:
                raise ValueError(f"{r.name} and {reference} have different grids on {task}")
            s = stats.savings(ns, R.mean(axis=0), T.mean(axis=0))
            mark = {"lower": ">= ", "upper": "<= "}.get(s["flag"], "")
            rows.append({"encoder": r.name, "task": task, "reference": reference,
                         "savings": s["savings"], "flag": s["flag"],
                         "savings_text": f"{mark}{s['savings']:.2f}"
                         + (f" ({s['flag']})" if s["flag"] in ("mixed", "undetermined") else "")})
    return pd.DataFrame(rows, columns=cols)


def _selection(results, tasks, proto, notes) -> tuple[pd.DataFrame, pd.DataFrame]:
    sel_seeds, rep_seeds = proto.get("selection_seeds") or [], proto.get("report_seeds") or []
    cols = ["task", "encoder", "strategy", "val_aulc_low_selection", "test_aulc_low_report",
            "test_aulc_low_report_random"]
    ccols = ["task", "encoder", "strategy", "line", "n", "mean", "band_lo", "band_hi"]
    if not sel_seeds or not rep_seeds:
        notes.append("no strategy selection: the protocol has no selection and report seeds "
                     "(use the paper protocol)")
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=ccols)
    rows, crows = [], []
    for task in tasks:
        for r in results:
            ok = [s for s, c in r.curves.get(task, {}).items() if c.get("status") == "ok"]
            if not ok:
                continue
            vals = [float(r.aulc_low(task, s, "val", sel_seeds).mean()) for s in ok]
            best = ok[int(np.argmax(vals))]
            has_random = "random" in ok
            rows.append({"task": task, "encoder": r.name, "strategy": best,
                         "val_aulc_low_selection": max(vals),
                         "test_aulc_low_report": float(r.aulc_low(task, best, "test",
                                                                  rep_seeds).mean()),
                         "test_aulc_low_report_random": float(r.aulc_low(
                             task, "random", "test", rep_seeds).mean()) if has_random else np.nan,
                         **{f"val_aulc_low_selection_{s}": v for s, v in zip(ok, vals)}})
            for line, strategy in (("selected", best), ("random", "random")):
                if strategy not in ok:
                    continue
                _, ns, Y = r.scores(task, strategy, "test", rep_seeds)
                mean, lo, hi = stats.mean_band(Y)
                crows += [{"task": task, "encoder": r.name, "strategy": strategy, "line": line,
                           "n": n, "mean": m, "band_lo": a, "band_hi": b}
                          for n, m, a, b in zip(ns, mean, lo, hi)]
    return pd.DataFrame(rows), pd.DataFrame(crows, columns=ccols)
