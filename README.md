# BGC-Bench

Evaluate biosynthetic gene cluster (BGC) embeddings on MIBiG 4.0. BGC-Bench compares
frozen representations across five tasks using random and out-of-distribution splits,
producing the 15 columns of the paper's main table.

This repository accompanies
*Data-Scarce, Multi-Granular, Multimodal: Benchmark Design for Biosynthetic Gene Clusters*
(NeurIPS 2026, Evaluations and Datasets Track).

Use one of the 12 built-in encoders or evaluate your own vectors. The
[dataset](https://huggingface.co/datasets/Nugkta/bgc-bench) includes model inputs, labels
and splits; no BiG-SCAPE or HMMER preprocessing is needed. Built-in embeddings are
computed locally from pinned model weights.

## Quick start

```bash
git clone https://github.com/Nugkta/BGC-Bench
cd BGC-Bench
uv sync --frozen

uv run bgcbench list                    # list the built-in encoders
uv run bgcbench run --encoder deepbgc   # run a CPU example
```

[uv](https://docs.astral.sh/uv/) installs Python 3.12 if needed. On first use, `run`
downloads the dataset (about 220 MB) and encoder weights, computes embeddings, and scores
all five tasks. It prints scores with 95% bootstrap intervals and saves
`results/<encoder>.json` with scores, intervals, sample counts and environment metadata.

Embeddings are cached, so later runs only repeat scoring. DeepBGC has no A-domain
embeddings, so its two substrate columns appear as `--`. Use `--tasks coarse fine` to
evaluate a subset of tasks, or run the command once per encoder to compare models.

The [encoder table](#built-in-encoders) lists compute requirements. Most learned encoders
use a GPU; LC-PLM and Evo 2 need the [separate GPU environment](#lc-plm-and-evo-2).

The default dataset and embedding cache is `~/.cache/bgcbench`; set `BGCBENCH_HOME` to
change it. See [configuration](docs/advanced.md#configuration) for local data and weights.

## Reproduce the main table

Run all 12 encoders in about 10 hours on a single 80 GB A100 GPU:

```bash
bash scripts/reproduce_main_table.sh
```

The [script](scripts/reproduce_main_table.sh) installs both environments and reuses cached
embeddings. Requires Linux x86_64, `uv`, at least 48 GB of GPU memory and a CUDA 13 driver.

Results go to `results/paper/`: one JSON per encoder with scores and 95% intervals,
`main_table.csv` and `main_table.md` with all 15 score columns, and per-step logs in
`logs/`. Pass an output directory as the script's argument to save them elsewhere.
The script stops on failure; rerun the same command after resolving the error.

Dataset revisions, model weights and evaluation seeds are fixed. Results record version
and environment metadata; small numerical differences can occur across hardware.
When evaluating exported LC-PLM or Evo 2 vectors, use `--name lcplm` or `--name evo2_7b`
to preserve the encoder's evaluation settings.

## Tasks

| Columns | Task | Metric | Probe |
|---|---|---|---|
| Coarse R / S / G | Six MIBiG classes, multi-label | Macro-AUROC | One-vs-rest logistic regression |
| Fine R / S / G | 48 NPClassifier product classes, multi-label | Weighted macro-AUROC | One-vs-rest logistic regression |
| Chem R / S / G | 13 RDKit descriptors of the first product | Mean Spearman rho | Ridge regression |
| Retrieval R / S / G / C | Rank a BGC's products in the test-molecule gallery | Class-macro R@10 | Linear contrastive head with frozen MoLFormer vectors |
| Substrate F / G | A-domain substrate, 28 classes | Top-1 accuracy | Logistic regression |

Split letters: **R** = random, **S** = Murcko-scaffold holdout, **G** = gene-cluster-family
holdout, **C** = class holdout, **F** = five-fold evaluation grouped by BGC.

Weighted macro-AUROC weights each class by its test positives. Retrieval counts a hit when
any product ranks in the top 10, handles ties, and averages three seeds. Evaluation uses
fixed seeds and 500 bootstrap resamples. See the
[dataset card](https://huggingface.co/datasets/Nugkta/bgc-bench) for label and split details.

## Built-in encoders

| Encoder | Model | Input | Weights licence | Embedding time |
|---|---|---|---|---|
| `bag_of_pfam` | Bag-of-Pfam | Pfam accessions | — | Seconds, CPU |
| `pfam_rand` | Random-Pfam | Pfam accessions | — | Seconds, CPU |
| `bigcarp` | BiGCARP | Pfam names | MIT | 10 s (A100) |
| `kawano` | Kawano et al. Model I | Pfam names | CC BY-NC 4.0 | 40 s (A100) |
| `deepbgc` | DeepBGC | Pfam accessions | MIT | 35 s, CPU |
| `aa_kmer3` | Amino-acid 3-mers | Proteins | — | Seconds, CPU |
| `bgc_mac` | ESM-2 650M (BGC-MAC backbone) | Proteins | MIT | 49 min (A100) |
| `bgcat` | ESM C 600M (BGCat backbone) | Proteins | Cambrian non-commercial | 8 min (A100) |
| `bgc_prophet` | ESM-2 8M (BGC-Prophet backbone) | Proteins | MIT | 3 min (L40S) |
| `lcplm` | LC-PLM | Proteins | CC BY-NC 4.0 | 72 min (L40S) |
| `kmer6` | Nucleotide 6-mers | DNA | — | Seconds, CPU |
| `evo2_7b` | Evo 2 7B, layer 28 | DNA | Apache-2.0 | 5.2 h (L40S) |

Embedding times are approximate for the full dataset. On an AMD EPYC 7413,
scoring takes 3–9 min for CPU baselines, about 12 min for Kawano,
19 min for LC-PLM and 90 min for Evo 2.

GPU encoders require a visible GPU by default. All except LC-PLM and Evo 2 also support
`--device cpu`, with slower embedding.

Weights are downloaded automatically. Sources and revisions are pinned in
[`src/bgcbench/__init__.py`](src/bgcbench/__init__.py); model cards document any conversions.
Follow each model's licence and cite its original paper. `bgcat` is Built with ESM; see
[third-party notices](THIRD_PARTY_NOTICES.md) for ESM licensing details.

### LC-PLM and Evo 2

LC-PLM and Evo 2 use a separate Linux x86_64 environment and require an NVIDIA driver
supporting CUDA 13:

```bash
cd envs/gpu
uv sync --frozen
CUDA_VISIBLE_DEVICES=0 .venv/bin/bgcbench embed --encoder evo2_7b
cd ../..
uv run bgcbench run --encoder evo2_7b
```

Use `--encoder lcplm` in both commands for LC-PLM. This computes and caches embeddings in
the GPU environment, then scores them in the main environment. Evo 2 requires at least
48 GB of GPU memory.

## Evaluate your own embeddings

Save BGC vectors in an `.npz` file with two arrays:

| Array | Contents |
|---|---|
| `bgc_ids` | MIBiG accessions, such as `BGC0000001` |
| `embeddings` | An `n × d` matrix, one row per accession |

```bash
uv run bgcbench evaluate --embeddings my.npz
```

For substrate evaluation, also provide A-domain vectors in an `.npz` with `domain_ids`
(from `tasks/adomain_instances.csv`) and `embeddings`:

```bash
uv run bgcbench evaluate --embeddings my.npz --adomain-embeddings ad.npz
```

Without A-domain vectors, substrate columns appear as `--`. Classification and regression
report coverage and omit BGCs with missing vectors. Retrieval and substrate evaluation
require complete coverage of their respective cohorts.

To integrate a model directly, see [adding an encoder](docs/advanced.md#adding-an-encoder).
The guide also covers [alternative molecule embeddings](docs/advanced.md#alternative-molecule-embeddings),
[custom tasks](docs/advanced.md#adding-a-task) and
[new representation kinds](docs/advanced.md#adding-a-representation).

## Active learning

The active-learning setting refits each task's probe as the label budget grows. Each run
starts from 16 labels and grows by a factor of about √2. Random acquisition, TypiClust and
BADGE choose the rows to label next. BADGE needs gradients, so it is skipped for chemical
properties.

```bash
uv run bgcbench al run --encoder bag_of_pfam             # quick protocol, CPU
uv run bgcbench al run --encoder bigcarp --tasks coarse fine --strategies random typiclust
uv run bgcbench al evaluate --embeddings my.npz --adomain-embeddings ad.npz
```

The default `quick` protocol runs seeds 1–5 up to 256 labels. `--protocol paper` runs the
paper's setting: seeds 1–20 over the whole pool, with strategies selected on seeds 1–10 by
validation score. `--seeds` and `--max-budget` override either protocol. Each learning curve
runs on one CPU thread; `--workers` sets how many run in parallel.

Results go to `results/al/<name>.json`. The table reports test AULC_low, the area under the
learning curve from 16 to 256 labels on a log2 axis, with 95% seed-bootstrap intervals. Each
finished curve is cached in `results/al/cells/`, so an interrupted run resumes.

`uv run bgcbench al summarize results/al` compares encoders. It reports gains over random
acquisition with corrected tests, label savings relative to Bag-of-Pfam, and the strategy
selected for each encoder.

To reproduce the paper's 3,240 learning curves, about 30 CPU hours, run:

```bash
bash scripts/reproduce_al.sh
```

The script reuses embeddings cached by `reproduce_main_table.sh`. Small differences in the
vectors can change which examples a strategy selects, so curves can differ from the paper's
when embeddings vary across hardware. With the released embeddings, six encoders reproduce the
paper's curves exactly, and the gains of the other six stay within 0.5 points.

See [active learning](docs/advanced.md#active-learning-for-your-own-tasks-and-strategies) to
add tasks and strategies.

## Tests

```bash
uv run pytest
```

## Licence

Code: [MIT](LICENSE), with [third-party notices](THIRD_PARTY_NOTICES.md) for vendored code.
Dataset: CC BY 4.0, built from MIBiG 4.0. Model weights follow their upstream licences
(see the encoder table).

## Citation

```bibtex
@inproceedings{xiao2026bgcbench,
  title     = {Data-Scarce, Multi-Granular, Multimodal: Benchmark Design for Biosynthetic Gene Clusters},
  author    = {Xiao, Hanlin and Takano, Eriko and {\'A}lvarez, Mauricio A. and Breitling, Rainer},
  booktitle = {Advances in Neural Information Processing Systems (Evaluations and Datasets Track)},
  year      = {2026}
}
```

Please also cite MIBiG 4.0 (Zdouc et al., *Nucleic Acids Res.* 53:D678, 2025) and the
papers of the encoders you use.
