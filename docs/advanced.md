# Advanced usage

This guide covers configuration and extensions to [BGC-Bench](../README.md).

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `BGCBENCH_HOME` | Dataset and embedding cache | `~/.cache/bgcbench` |
| `HF_HOME` | Hugging Face cache, including model weights | Hugging Face default |
| `BGCBENCH_DATA` | Local copy of the main dataset | Download the pinned dataset |
| `BGCBENCH_MODELS` | Local weights under `<dir>/<key>/` | Download missing weights |

Model keys are listed in `MODELS` in
[`src/bgcbench/__init__.py`](../src/bgcbench/__init__.py). Local model files or folders
are used when present; missing ones are downloaded.

## Adding an encoder

Subclass `Encoder` and return one vector per BGC. Model inputs are already included in
the dataset. This example assumes you supply a `my_model` function:

```python
import bgcbench

class MyEncoder(bgcbench.Encoder):
    name, input = "my_encoder", "protein"   # or "pfam_acc", "pfam_named", "nucleotide"

    def embed_bgcs(self, inputs):           # {bgc_id: [CDS protein sequences]}
        return {b: my_model(seqs) for b, seqs in inputs.items()}

    def embed_adomains(self, adomains, inputs):   # optional, for the substrate task
        return {d: my_model([s]) for d, s in zip(adomains["domain_id"], adomains["aa_seq"])}

result = bgcbench.run(MyEncoder())
print(result.table())
```

Embeddings are cached. After changing your model, bump its `version` attribute or pass
`refresh=True` to recompute vectors.

## Alternative molecule embeddings

Retrieval uses the dataset's frozen MoLFormer vectors by default. To replace them, pass
`--chem-embeddings chem.npz` to `evaluate`. The file needs `compound_ids` and `embeddings`,
with one vector per molecule in `tasks/compounds.npz`.

Both `run` and `evaluate` accept `--chem-encoder NAME` to select a shipped vector set:
`molformer`, `chemberta2_mtr`, `unimol`, `ecfp4`, `atompair`, `maccs`, `rdkit_desc` or
`random`. Each also has a `<name>_pca128` variant. The dataset's
`tasks/chem_encoders/encoders.json` describes these sets.

```bash
uv run bgcbench run --encoder deepbgc --chem-encoder ecfp4
```

Using alternative molecule vectors changes the main-table protocol. Result tables and
JSON files identify the replacement vectors.

## Adding a task

Subclass `Task`, register it, and select it by name. Custom tasks run only when explicitly
selected; `tasks=None` runs the official suite. Tables mark custom tasks with a star, and
the result JSON records `"official": false` and the task version.

```python
import bgcbench
from bgcbench import data, metrics, probes

@bgcbench.register_task
class NRPvsRest(bgcbench.Task):
    name, version, metric = "nrp_binary", "1", "AUROC"
    splits = {"R": "t1ml_random", "G": "t1ml_gcf_holdout_c07"}
    requires = ("bgc",)
    threads = 8

    def evaluate(self, reps, split, *, n_boot, device):
        b = reps["bgc"]
        X, df, coverage = data.align(b.ids, b.X, self.data.split(split))
        train = (df["split"] == "train").to_numpy()
        test = (df["split"] == "test").to_numpy()
        data.require_train_test(train, test)
        y = df["NRP"].to_numpy()
        proba = probes.logreg().fit(X[train], y[train]).predict_proba(X[test])
        out = metrics.auroc_with_ci(y[test], proba, multilabel=False, n_boot=n_boot, seed=42)
        return {**out, "n_train": int(train.sum()), "n_test": int(test.sum()), **coverage}

result = bgcbench.run("bigcarp", tasks=["coarse", "nrp_binary"])
```

For the command line, save the task class (without the final `bgcbench.run` line) in
`my_tasks.py` and import it with `--task-module`:

```bash
uv run bgcbench run --encoder bigcarp --task-module my_tasks.py --tasks coarse nrp_binary
```

`--task-module` accepts a file or module name and can be repeated.

Keep the evaluation protocol fixed across encoders:

- Ship split files in `splits/<name>.csv`. Keep held-out groups separate from training.
- Use fixed seeds for bootstrap sampling, folds and training. The pipeline sets seed 42
  before each column; `threads` controls the task's CPU thread count.
- Validate the number of scored samples and return counts alongside the metric.
- Return `{"point": score, "ci": [lo, hi], ...}`. If omitted, intervals are stored as
  `null` values in JSON. Raise `data.CoverageError` for incomplete required coverage;
  the column will appear as `--`.
- Bump the task's `version` when its data, protocol or metric changes.

Set `Task.dataset` to a local directory or `"org/repo@revision"` to use another dataset.
Access it through `self.data.path`, `read_csv`, `read_json` and `split`. `BGCBENCH_DATA`
overrides the main dataset only.

## Adding a representation

A representation kind maps item IDs to vectors. The built-in kinds are `bgc` (`bgc_ids`),
`adomain` (`domain_ids`) and `molecule` (`compound_ids`). Register a new kind with its ID
array name and encoder method. An optional `table` supplies the items to embed:

```python
bgcbench.register_representation("gene", "gene_ids", method="embed_genes",
                                  table=my_gene_table)  # () -> [(gene_id, protein), ...]

class MyEncoder(bgcbench.Encoder):
    ...
    def embed_genes(self, genes, inputs):
        return {g: my_model(seq) for g, seq in genes}
```

`run` caches each representation as `<kind>.npz`. Bump the encoder's `version` after
adding a method. Tasks declare their required kinds in `requires`; missing vectors
produce `--` columns.

Pass custom vectors through `evaluate(..., reps={"gene": "genes.npz"})` or
`--rep gene=genes.npz`. For built-in kinds, `--rep adomain=ad.npz` is equivalent to
`--adomain-embeddings ad.npz`.
