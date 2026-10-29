#!/usr/bin/env bash
# Run the paper's active-learning protocol for all 12 encoders: 5 tasks, 3 strategies, 20 seeds.
set -euo pipefail

usage() {
    printf 'Usage: bash scripts/reproduce_al.sh [OUTPUT_DIR]\n'
    printf 'Default output: results/al_paper/ in the repository. Requires uv.\n'
    printf 'Uses the embeddings cached by reproduce_main_table.sh; missing ones need a CUDA GPU.\n'
    printf 'BGCBENCH_WORKERS sets the number of parallel processes (default: all CPUs).\n'
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if (( $# > 1 )) || [[ ${1:-} == -* ]]; then
    usage >&2
    exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
    printf 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/\n' >&2
    exit 1
fi

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
output_dir=${1:-"$repo_dir/results/al_paper"}
mkdir -p -- "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd)
mkdir -p -- "$output_dir/logs"
cd -- "$repo_dir"
export PYTHONUNBUFFERED=1
workers=${BGCBENCH_WORKERS:-$(nproc)}

main=(uv run --project "$repo_dir" --frozen --no-sync)
gpu=(uv run --project "$repo_dir/envs/gpu" --frozen --no-sync)

step() {
    local name=$1
    shift
    printf '\n== %s ==\n' "$name"
    "$@" 2>&1 | tee "$output_dir/logs/$name.log"
}

step setup_main uv sync --project "$repo_dir" --frozen

# Embeddings: read from the cache, or computed here (learned encoders need a GPU).
for encoder in bag_of_pfam pfam_rand aa_kmer3 kmer6 deepbgc bigcarp kawano bgc_prophet bgcat bgc_mac; do
    step "embed_$encoder" "${main[@]}" bgcbench embed --encoder "$encoder"
done
# LC-PLM and Evo 2 embed in envs/gpu; once cached, the main environment scores them.
for encoder in lcplm evo2_7b; do
    if ! step "embed_$encoder" "${main[@]}" bgcbench embed --encoder "$encoder"; then
        step setup_gpu uv sync --project "$repo_dir/envs/gpu" --frozen
        step "embed_${encoder}_gpu" "${gpu[@]}" bgcbench embed --encoder "$encoder"
    fi
done

# 3,240 learning curves on one CPU thread each; finished curves are cached in cells/, so
# rerunning the same command resumes.
step al_run "${main[@]}" bgcbench al run --protocol paper --workers "$workers" \
    --out "$output_dir" --encoder bag_of_pfam pfam_rand bigcarp kawano deepbgc bgc_mac bgcat \
    bgc_prophet lcplm evo2_7b kmer6 aa_kmer3
step al_summarize "${main[@]}" bgcbench al summarize "$output_dir" --out "$output_dir/summary"

# The lines of the paper's active-learning figure: per task, the compositional baseline with
# the best validation AULC_low under random acquisition, BiGCARP, the best protein language
# model and Evo 2, each with its validation-selected strategy and with random acquisition.
"${main[@]}" python - "$output_dir/summary" <<'PY'
import sys
from pathlib import Path

import pandas as pd

out = Path(sys.argv[1])
sel = pd.read_csv(out / "selection.csv")
curves = pd.read_csv(out / "selected_curves.csv")
compositional = ["bag_of_pfam", "pfam_rand", "aa_kmer3", "kmer6"]
protein = ["bgc_mac", "bgcat", "bgc_prophet", "lcplm"]
rows = []
for task, s in sel.groupby("task", sort=False):
    s = s.set_index("encoder")
    base = s.loc[[e for e in compositional if e in s.index], "val_aulc_low_selection_random"]
    prot = s.loc[[e for e in protein if e in s.index], "val_aulc_low_selection"]
    lines = [("baseline", base.idxmax(), ["random"]),
             ("bigcarp", "bigcarp", ["selected", "random"]),
             ("protein_lm", prot.idxmax(), ["selected", "random"]),
             ("evo2", "evo2_7b", ["selected", "random"])]
    for role, encoder, kinds in lines:
        for line in kinds:
            c = curves[(curves["task"] == task) & (curves["encoder"] == encoder)
                       & (curves["line"] == line)]
            rows.append(c.assign(role=role))
fig = pd.concat(rows)[["task", "role", "encoder", "strategy", "line", "n", "mean", "band_lo",
                       "band_hi"]]
fig.to_csv(out / "figure_lines.csv", index=False)
print(f"\nFigure data saved to {out / 'figure_lines.csv'}")
print(f"Results saved to {out}")
PY
