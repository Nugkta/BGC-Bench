#!/usr/bin/env bash
# Run all 12 encoders with the paper's default evaluation protocol.
set -euo pipefail

usage() {
    printf 'Usage: bash scripts/reproduce_main_table.sh [OUTPUT_DIR]\n'
    printf 'Default output: results/paper/ in the repository. Requires uv and a CUDA GPU.\n'
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
output_dir=${1:-"$repo_dir/results/paper"}
mkdir -p -- "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd)
mkdir -p -- "$output_dir/logs"
cd -- "$repo_dir"
export PYTHONUNBUFFERED=1

main=(uv run --project "$repo_dir" --frozen --no-sync)
gpu=(uv run --project "$repo_dir/envs/gpu" --frozen --no-sync)

step() {
    local name=$1
    shift
    printf '\n== %s ==\n' "$name"
    "$@" 2>&1 | tee "$output_dir/logs/$name.log"
}

step setup_main uv sync --project "$repo_dir" --frozen
step setup_gpu uv sync --project "$repo_dir/envs/gpu" --frozen
step check_gpu "${gpu[@]}" python -c \
    'import sys, torch; sys.exit(0 if torch.cuda.is_available() else "A CUDA GPU is required to run all encoders.")'

# Keep embedding and scoring sequential so only one encoder occupies the GPU at a time.
for encoder in bag_of_pfam pfam_rand aa_kmer3 kmer6 deepbgc bigcarp kawano bgc_prophet bgcat bgc_mac; do
    step "run_$encoder" "${main[@]}" bgcbench run --encoder "$encoder" \
        --out "$output_dir/$encoder.json"
done

# These two encoders need CUDA kernels from envs/gpu; all scoring uses the main environment.
for encoder in lcplm evo2_7b; do
    step "embed_$encoder" "${gpu[@]}" bgcbench embed --encoder "$encoder"
    step "run_$encoder" "${main[@]}" bgcbench run --encoder "$encoder" \
        --out "$output_dir/$encoder.json"
done

# Summarise scores in paper order. Full intervals and metadata remain in each JSON.
"${main[@]}" python - "$output_dir" <<'PY'
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
encoders = (
    "bag_of_pfam", "pfam_rand", "bigcarp", "kawano", "deepbgc", "aa_kmer3",
    "bgc_mac", "bgcat", "bgc_prophet", "lcplm", "kmer6", "evo2_7b",
)
columns = [
    "coarse_R", "coarse_S", "coarse_G", "fine_R", "fine_S", "fine_G",
    "chem_R", "chem_S", "chem_G", "retrieval_R", "retrieval_S", "retrieval_G",
    "retrieval_C", "substrate_F", "substrate_G",
]
rows = []
for encoder in encoders:
    result = json.loads((out / f"{encoder}.json").read_text())
    cells = result["cells"]
    rows.append([encoder] + [
        None if cells[c].get("status") == "na" else cells[c]["point"]
        for c in columns
    ])

header = ["encoder", *columns]
with (out / "main_table.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(header)
    writer.writerows(rows)

lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
for name, *scores in rows:
    values = ["--" if s is None else f"{s:.4f}" for s in scores]
    lines.append("| " + " | ".join([name, *values]) + " |")
table = "\n".join(lines) + "\n"
(out / "main_table.md").write_text(table)
print("\n" + table)
print(f"Results saved to {out}")
PY
