"""Authors' tool: assemble the Hugging Face dataset and model mirrors from the research repo.

The ``dataset`` step documents how the released files were derived from the authors'
research repository (not public) and cannot be rerun without it; the other steps work on
any release folder.

Release steps:

  dataset     copy splits / task tables / model inputs / GCF / provenance files from the
              research repository; derive the descriptor table, the retrieval universe, the
              molecule vector files and the fixed baseline files
  card        write README.md (dataset card) and croissant.json
  manifest    write manifest.json (file inventory and sizes)
  mirror      write a model mirror's card
  upload      create the Hugging Face repo if needed (public unless --private), upload a
              folder, tag the commit

Example:
  python tools/build_release.py dataset --research /path/to/research-repo --out /tmp/release
"""

from __future__ import annotations

import argparse
import csv
import datetime
import fnmatch
import glob
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = [
    "t1ml_random", "t1ml_scaffold_holdout", "t1ml_gcf_holdout_c07",
    "t2ml_stratified_5fold", "t2ml_scaffold_holdout", "t2ml_gcf_c07_holdout",
    "t1_random", "t1_scaffold_holdout", "t1_gcf_holdout_c07",
    "t3_random", "t3_scaffold_holdout", "t3_gcf_holdout_c07", "t3_class_holdout",
    "t5_bgc_grouped_5fold", "t5_gcf_holdout_c07",
]
CHEM_TARGETS = [
    "MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA", "NumRotatableBonds", "qed",
    "BertzCT", "NumAliphaticRings", "NumAromaticRings", "FractionCSP3", "NumStereocenters",
    "HalogenAtomCount",
]
# The 18 encoders whose shared coverage defined the published retrieval universe: the
# paper's twelve encoders and its ablation variants (`lcplm_percds` is the `lcplm` encoder).
T3_ELIGIBLE = (
    "aa_kmer2", "aa_kmer3", "aa_kmer3_trunc1022", "bag_of_pfam", "bgc_mac", "bgc_prophet",
    "bgcat", "bigcarp", "bigcarp_cls", "deepbgc", "evo2_7b", "kawano", "kmer4", "kmer5",
    "kmer6", "lcplm_percds", "lcplm_percds_trunc1022", "pfam_rand",
)
COPY = {
    "tasks/t2ml_labels.csv": "data/processed/t2ml_labels.csv",
    "tasks/t3_bgc_compound_pairs.csv": "data/processed/t3_bgc_compound_pairs.csv",
    "tasks/adomain_instances.csv": "data/processed/adomain_instances.csv",
    "tasks/adomain_nucleotide.json": "data/processed/adomain_nucleotide.json",
    "inputs/pfam_domains.json": "data/processed/model_inputs/pfam_domains.json",
    "inputs/pfam_domains_named.json": "data/processed/model_inputs/pfam_domains_named.json",
    "inputs/pfam_acc_to_name.json": "data/processed/model_inputs/pfam_acc_to_name.json",
    "inputs/protein_sequences.json": "data/processed/model_inputs/protein_sequences.json",
    "inputs/nucleotide_seqs.json": "data/processed/model_inputs/nucleotide_seqs.json",
    "gcf/gcf_cutoff_summary.csv": "data/processed/gcf_cutoff_summary.csv",
    "provenance/mibig_metadata.csv": "data/processed/mibig_metadata.csv",
    "provenance/npclassifier_labels.csv": "data/processed/npclassifier_labels.csv",
    "provenance/murcko_scaffolds.csv": "data/processed/murcko_scaffolds.csv",
}
for _c in ("0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"):
    COPY[f"gcf/gcf_assignments_c{_c}.csv"] = f"data/processed/gcf_assignments_c{_c}.csv"
# Molecule vectors of the paper's chemical-encoder comparison: seven encoders and a random
# control, each native and as PCA-128 (fitted on all 3,118 molecules).
CHEM_RAW = ("molformer", "chemberta2_mtr", "unimol", "ecfp4", "atompair", "maccs",
            "rdkit_desc", "random")
CHEM_ENCODERS = CHEM_RAW + tuple(f"{n}_pca128" for n in CHEM_RAW)
CHEM_WEIGHTS = {
    "molformer": {"repo_id": "ibm/MoLFormer-XL-both-10pct",
                  "revision": "7b12d946c181a37f6012b9dc3b002275de070314"},
    "chemberta2_mtr": {"repo_id": "DeepChem/ChemBERTa-77M-MTR",
                       "revision": "66b895cab8adebea0cb59a8effa66b2020f204ca"},
    "unimol": {"repo_id": "dptech/Uni-Mol-Models",
               "revision": "9f19c45c718192888a1c8a1c905f69f0755ea502",
               "file": "mol_pre_no_h_220816.pt"},
}
# Definitions in encoders.json written here rather than taken from the research registry.
CHEM_DEFINITIONS = {
    "ecfp4": "Morgan radius 2, 2048 bits, binary "
             "(rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048))",
    "atompair": "rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048) defaults, bit vector "
                "with count simulation",
    "unimol": "Uni-Mol v1 no-H checkpoint, 512-d CLS, mol.dict.txt in file order; seeded "
              "conformers (explicit-H ETKDGv3 seed 42, a maxIterations-5000 retry, a "
              "random-coordinate retry, else 2D; MMFF where parametrised; H stripped after), "
              "no atom cropping",
}
# What the shipped MoLFormer extraction report keeps: the vectors' definition and the checks
# they pass (run-specific fields such as job ids, paths and timings are left out).
REPORT_KEYS = ("pass", "n_bgcs", "n_pairs", "n_compounds", "n_rdkit_failures", "dropped",
               "n_has_dot", "n_pairs_has_dot", "token_stats", "count_check", "reference_facts",
               "table_stats", "definitions")
REPORT_PROVENANCE = ("model_id", "revision", "torch_version", "transformers_version",
                     "tokenizers_version", "rdkit_version", "numpy_version", "pandas_version",
                     "python_version", "device_name", "utc_time")
REPORT_ENCODER = ("model_id", "revision", "device", "float32_matmul_precision",
                  "tokenizer_class", "model_class", "model_max_position_embeddings",
                  "loading_info", "batch_size", "padding", "truncation", "pooling",
                  "encode_order", "n_compounds_with_unk_tokens", "reencode_check")


# Dataset
def build_dataset(research: Path, out: Path) -> None:
    for name in SPLITS:
        _copy(research / "data/splits" / f"{name}.csv", out / "splits" / f"{name}.csv")
    for dst, src in COPY.items():
        _copy(research / src, out / dst)

    # Chemical-property targets: the 13 descriptors of the paper's panel. Columns are cut at
    # the text level: pandas' default float parser is not correctly rounded, so the floats
    # the paper used are those parsed from these exact characters.
    src = research / "data/processed/chem_descriptors.csv"
    with open(src, newline="") as fh:
        rows = list(csv.reader(fh))
    keep = [rows[0].index(c) for c in ["bgc_id", *CHEM_TARGETS]]
    with open(out / "tasks/chem_descriptors.csv", "w", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerows([[r[i] for i in keep] for r in rows])
    a = pd.read_csv(src)[["bgc_id", *CHEM_TARGETS]]
    b = pd.read_csv(out / "tasks/chem_descriptors.csv")
    assert a.equals(b), "descriptor subset changed values"

    # Retrieval universe: split rows shared by MoLFormer and the 18 eligible encoders.
    universe = None
    for split in ("t3_random", "t3_scaffold_holdout", "t3_gcf_holdout_c07", "t3_class_holdout"):
        ids = set(pd.read_csv(research / "data/splits" / f"{split}.csv")["bgc_id"])
        for enc in ("molformer", *T3_ELIGIBLE):
            with np.load(research / "data/embeddings" / enc / "embeddings.npz",
                         allow_pickle=True) as z:
                ids &= set(z["bgc_ids"].astype(str))
        assert universe is None or ids == universe, "retrieval universe differs between splits"
        universe = ids
    assert len(universe) == 1939, len(universe)
    pd.DataFrame({"bgc_id": sorted(universe)}).to_csv(out / "tasks/t3_universe.csv", index=False)

    # MoLFormer molecule vectors (ibm/MoLFormer-XL-both-10pct, pooler output).
    with np.load(research / "data/embeddings/molformer_compounds/compounds.npz") as z:
        np.savez(out / "tasks/compounds.npz", compound_ids=z["compound_ids"],
                 canonical_smiles=z["canonical_smiles"], embeddings=z["embeddings"])
    report = json.loads(
        (research / "data/embeddings/molformer_compounds/extraction_report.json").read_text())
    (out / "tasks/compounds_extraction_report.json").write_text(
        json.dumps(public_report(report), indent=2) + "\n")
    build_chem_encoders(research, out)

    # Fixed baseline files: Bag-of-Pfam vocabulary and the Random-Pfam vectors (seed 42).
    domains = json.loads((research / "data/processed/model_inputs/pfam_domains.json").read_text())
    vocab = sorted({p for lst in domains.values() for p in lst})
    (out / "inputs/bag_of_pfam_vocab.json").write_text(json.dumps(vocab))
    rng = np.random.default_rng(42)
    vecs = np.stack([rng.standard_normal(256).astype(np.float32) for _ in vocab])
    unk = rng.standard_normal(256).astype(np.float32)
    np.savez(out / "inputs/pfam_rand_vectors.npz", vocab=np.array(vocab), vectors=vecs, unk=unk)
    print(f"dataset staged in {out}: {sum(1 for p in out.rglob('*') if p.is_file())} files")


def public_report(report: dict) -> dict:
    """The keys of the MoLFormer extraction report named in ``REPORT_*``, in report order."""
    out = {k: report[k] for k in REPORT_KEYS if k in report}
    prov = report.get("provenance", {})
    out["provenance"] = {k: prov[k] for k in REPORT_PROVENANCE if k in prov}
    if "encoder" in prov:
        out["provenance"]["encoder"] = {k: prov["encoder"][k] for k in REPORT_ENCODER
                                        if k in prov["encoder"]}
    return out


def build_chem_encoders(research: Path, out: Path) -> None:
    """tasks/chem_encoders/<name>.npz (compound_ids, embeddings) and encoders.json.

    Every file holds the molecules of tasks/compounds.npz in its order; the MoLFormer file
    repeats its vectors so that all encoders load the same way.
    """
    with np.load(out / "tasks/compounds.npz") as z:
        ref_ids, ref_emb = z["compound_ids"], z["embeddings"]
    (out / "tasks/chem_encoders").mkdir(parents=True, exist_ok=True)
    index = {}
    for name in CHEM_ENCODERS:
        src = research / "data/embeddings/chem_compounds" / name
        npz = (research / "data/embeddings/molformer_compounds/compounds.npz"
               if name == "molformer" else src / "compounds.npz")
        with np.load(npz) as z:
            ids, emb = z["compound_ids"], z["embeddings"]
        assert np.array_equal(ids, ref_ids), f"{name}: molecules differ from compounds.npz"
        assert emb.dtype == np.float32 and np.isfinite(emb).all(), name
        if name == "molformer":
            assert np.array_equal(emb, ref_emb), "molformer differs from compounds.npz"
            entry = {"display": "MoLFormer", "family": "SMILES transformer", "variant": "raw",
                     "base": "molformer", "dim": int(emb.shape[1]),
                     "definition": "MoLFormer-XL pooler output (tasks/compounds.npz)"}
        else:
            reg = json.loads((src / "extraction_report.json").read_text())["registry_entry"]
            assert reg["name"] == name and reg["dim"] == emb.shape[1], name
            entry = {k: reg[k] for k in ("display", "family", "variant", "base", "dim",
                                         "definition")}
            entry["definition"] = CHEM_DEFINITIONS.get(name, entry["definition"])
        if entry["base"] in CHEM_WEIGHTS:
            entry["weights"] = CHEM_WEIGHTS[entry["base"]]
        index[name] = entry
        np.savez_compressed(out / "tasks/chem_encoders" / f"{name}.npz", compound_ids=ids,
                            embeddings=emb)
    (out / "tasks/chem_encoders/encoders.json").write_text(json.dumps(index, indent=1) + "\n")


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# Model mirrors
MIRRORS = {
    "bigcarp": dict(
        title="BiGCARP", licence="mit", licence_name="MIT",
        paper="Rios-Martinez, Bhattacharya, Amini, Crawford & Yang (2023). Deep self-supervised "
              "learning for biosynthetic gene cluster detection and product classification. "
              "PLOS Computational Biology 19(5): e1011162. https://doi.org/10.1371/journal.pcbi.1011162",
        code="https://github.com/microsoft/bigcarp",
        source="Zenodo record 6857704 (v1.0, MIT), https://doi.org/10.5281/zenodo.6857704: "
               "`bigcarp_esm1bfinetune.pt` and `final_domain_vocab.json`",
        conversion="`model.safetensors` holds the checkpoint's `model_state_dict` with unchanged "
                   "tensor values; `vocab.json` is `final_domain_vocab.json` byte for byte; "
                   "`config.json` records the ByteNet hyper-parameters.",
        model="ByteNet masked language model (32 dilated CNN layers, d_model 256) over Pfam "
              "domain names. BGC-Bench uses the final-layer states: their mean over the BGC's "
              "domains (256-d), and the state at an A-domain's AMP-binding token.",
        bibtex="""@article{bigcarp,
  title   = {Deep Self-Supervised Learning for Biosynthetic Gene Cluster Detection and Product Classification},
  author  = {Rios-Martinez, Carolina and Bhattacharya, Nicholas and Amini, Ava P. and Crawford, Lorin and Yang, Kevin K.},
  journal = {PLOS Computational Biology}, volume = {19}, number = {5}, pages = {e1011162}, year = {2023},
  doi     = {10.1371/journal.pcbi.1011162}
}"""),
    "deepbgc": dict(
        title="DeepBGC", licence="mit", licence_name="MIT",
        paper="Hannigan et al. (2019). A deep learning genome-mining strategy for biosynthetic "
              "gene cluster prediction. Nucleic Acids Research 47(18): e110. "
              "https://doi.org/10.1093/nar/gkz654",
        code="https://github.com/Merck/deepbgc",
        source="GitHub release v0.1.0 (MIT), "
               "https://github.com/Merck/deepbgc/releases/download/v0.1.0/deepbgc.pkl",
        conversion="The Keras detector was loaded with DeepBGC 0.1.12 / TensorFlow 2.15.1 and its "
                   "arrays written to `model.safetensors` (float32): the forward and backward LSTM "
                   "kernels, recurrent kernels and biases (Keras gate order i, f, c, o), the "
                   "output head, and the Pfam2vec table (rows in the order of "
                   "`pfam2vec_vocab.json`). A PyTorch re-implementation (hard sigmoid "
                   "recurrent activation) reproduces the Keras hidden states to max |diff| 4.5e-7.",
        model="Bidirectional LSTM (2 x 128 units) over Pfam2vec domain vectors plus two "
              "gene-border flags (102-d input). BGC-Bench uses the mean of the concatenated "
              "hidden states over the BGC's Pfam accessions (256-d), approximating the "
              "gene-border flags with the first and last token of each cluster.",
        bibtex="""@article{deepbgc,
  title   = {A Deep Learning Genome-Mining Strategy for Biosynthetic Gene Cluster Prediction},
  author  = {Hannigan, Geoffrey D. and Prihoda, David and Palicka, Andrej and others},
  journal = {Nucleic Acids Research}, volume = {47}, number = {18}, pages = {e110}, year = {2019},
  doi     = {10.1093/nar/gkz654}
}"""),
    "kawano": dict(
        title="Kawano et al. Model I", licence="cc-by-nc-4.0", licence_name="CC BY-NC 4.0",
        paper="Kawano, Shiraishi, Kuzuyama & Umemura (2026). A novel transformer-based platform "
              "for the prediction and design of biosynthetic gene clusters for (un)natural "
              "products. PLOS Computational Biology 22(2): e1013181. "
              "https://doi.org/10.1371/journal.pcbi.1013181",
        code="https://github.com/umemura-m/bgc-transformer",
        source="Zenodo record 17577731 (CC BY-NC 4.0), https://doi.org/10.5281/zenodo.17577731: "
               "`Kawano_etal_modelI.v2.tar.gz`",
        conversion="Files copied byte for byte from the archive: "
                   "`models/model_I_roberta_bgc/out.bgc/{config.json, model.safetensors}`, and "
                   "`models/vocab_pfam_product_hmmer33.json` saved as `tokenizer.json`. The "
                   "config keeps `pad_token_id = 1` although the tokenizer pads with id 3; "
                   "RoBERTa derives position ids from it, so it is left as released. "
                   "`LICENSE` is the upstream file: a summary of CC BY-NC 4.0 that links the "
                   "legal code (https://creativecommons.org/licenses/by-nc/4.0/legalcode).",
        model="RoBERTa masked language model (8 layers, 1024-d) with one word-level token per "
              "Pfam domain name. BGC-Bench uses the last hidden layer: its masked mean over a "
              "512-token, max-length-padded input, and the state at an A-domain's AMP-binding "
              "token.",
        bibtex="""@article{kawano2026roberta,
  title   = {A Novel Transformer-Based Platform for the Prediction and Design of Biosynthetic Gene Clusters for (Un)Natural Products},
  author  = {Kawano, Tomoki and Shiraishi, Taro and Kuzuyama, Tomohisa and Umemura, Maiko},
  journal = {PLOS Computational Biology}, volume = {22}, number = {2}, pages = {e1013181}, year = {2026},
  doi     = {10.1371/journal.pcbi.1013181}
}"""),
    "lcplm": dict(
        title="LC-PLM", licence="cc-by-nc-4.0", licence_name="CC BY-NC 4.0",
        paper="Wang, Wang, Sadeh, Zancato, Achille, Karypis & Rangwala (2025). LC-PLM: "
              "long-context protein language modeling using bidirectional Mamba with shared "
              "projection layers. Transactions on Machine Learning Research. "
              "https://openreview.net/forum?id=dWvztQzfy4 (arXiv:2411.08909)",
        code="https://github.com/amazon-science/LC-PLM",
        source="GitHub repository amazon-science/LC-PLM, commit "
               "80388cad25a8a4f0c1d1f0922fb8a6712e182cb4 (CC BY-NC 4.0; weights in Git LFS)",
        conversion="Files copied byte for byte from that commit (`config.json`, "
                   "`configuration_lcplm.py`, `modeling_lcplm.py`, `layer_norm.py`, "
                   "`model.safetensors`, `LICENSE`). `layer_norm.py` is under Apache-2.0 "
                   "(Tri Dao, modified by Amazon; its header is kept); the weights and the "
                   "other files are under CC BY-NC 4.0.",
        model="Bidirectional Mamba (BiMamba-S, 48 layers, 1536-d) protein language model; "
              "tokenizer: `facebook/esm2_t6_8M_UR50D`. Load with `trust_remote_code=True`. "
              "BGC-Bench ties the reverse-direction projections after loading, embeds one "
              "sequence at a time without padding, and uses `fused_add_norm=True`. "
              "BGC-Bench averages the last layer over residues per CDS, then over CDS (1536-d).",
        bibtex="""@article{wang2025lcplm,
  title   = {{LC-PLM}: Long-Context Protein Language Modeling Using Bidirectional {Mamba} with Shared Projection Layers},
  author  = {Wang, Yingheng and Wang, Zichen and Sadeh, Gil and Zancato, Luca and Achille, Alessandro and Karypis, George and Rangwala, Huzefa},
  journal = {Transactions on Machine Learning Research}, year = {2025},
  url     = {https://openreview.net/forum?id=dWvztQzfy4}
}"""),
}
GITHUB = "https://github.com/Nugkta/BGC-Bench"


def write_model_card(key: str, folder: Path) -> None:
    """README.md of a mirror folder: upstream source, format, model, file sizes, use."""
    m = MIRRORS[key]
    files = [r for r in release_files(folder) if r != "README.md"]
    rows = "\n".join(f"| `{r}` | {(folder / r).stat().st_size:,} |" for r in files)
    clone = f"git clone {GITHUB} && cd BGC-Bench\nuv sync\n"
    if key == "lcplm":
        usage = (clone + "cd envs/gpu && uv sync --frozen\n"
                 "CUDA_VISIBLE_DEVICES=0 .venv/bin/bgcbench embed --encoder lcplm\n"
                 "cd ../.. && uv run bgcbench run --encoder lcplm")
    else:
        usage = clone + f"uv run bgcbench run --encoder {key}"
    use = "; non-commercial use only" if "nc" in m["licence"].split("-") else ""
    card = f"""---
license: {m['licence']}
tags:
- biology
- biosynthetic-gene-clusters
- bgc-bench
---

# {m['title']} (BGC-Bench mirror)

This repository redistributes the {m['title']} weights released by the original authors,
so that [BGC-Bench]({GITHUB}) can download them at a fixed tag.
It is not an official release of the original authors, and the weights are not retrained.
Licence: {m['licence_name']}, as upstream{use}. Please cite the original paper (below).
Intended use: frozen feature extraction in BGC-Bench; see the original paper for the model's
training data and limitations.

## Upstream

- Paper: {m['paper']}
- Code: {m['code']}
- Weights: {m['source']}

## Format

{m['conversion']}

## Model

{m['model']}

## Files

| file | bytes |
|---|---:|
{rows}

## Use

```bash
{usage}
```

## Citation

```bibtex
{m['bibtex']}
```
"""
    (folder / "README.md").write_text(card)
    print(f"{folder}/README.md: {len(files)} files")


def release_files(folder: Path) -> list[str]:
    """Files of a release folder, relative and sorted; dot-paths (.cache/, .DS_Store) skipped."""
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()
                  and not any(part.startswith(".") for part in p.relative_to(folder).parts))


def upload(folder: Path, repo_id: str, repo_type: str, tag: str, *,
           private: bool = False) -> None:
    """Create the repo if needed (public unless ``private``), upload the folder and tag the
    commit. An existing repo keeps its visibility.

    Refuses when the tag exists already. Files the repo holds but the folder does not are
    deleted in the same upload, so the tagged commit holds exactly the folder (earlier tags
    keep them). A dataset tag must match the revision configured in the package.
    """
    from huggingface_hub import HfApi

    if repo_type == "dataset":
        import bgcbench

        if tag != bgcbench.DATASET_REVISION:
            raise SystemExit(f"dataset tag {tag} is not the configured "
                             f"{bgcbench.DATASET_REVISION}; update DATASET_REVISION "
                             f"in src/bgcbench/__init__.py first")
    api = HfApi()
    extra = set()
    if api.repo_exists(repo_id, repo_type=repo_type):
        if tag in {t.name for t in api.list_repo_refs(repo_id, repo_type=repo_type).tags}:
            raise SystemExit(f"{repo_id} already has tag {tag}")
        local = set(release_files(folder))
        extra = set(api.list_repo_files(repo_id, repo_type=repo_type)) - local - {".gitattributes"}
        print(f"removing from {repo_id}: {sorted(extra) or 'nothing'}")
    api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
    info = api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type=repo_type,
                             ignore_patterns=[".*", "*/.*"], commit_message=f"BGC-Bench {tag}",
                             delete_patterns=[glob.escape(f) for f in sorted(extra)] or None)
    api.create_tag(repo_id, tag=tag, repo_type=repo_type, revision=info.oid, exist_ok=False)
    print(f"{repo_type} {repo_id}@{tag} -> {info.oid} (pin tag {tag} in src/bgcbench/__init__.py)")


# Card and manifest
HF_DATASET = "Nugkta/bgc-bench"
CROISSANT_CONTEXT = {
    "@language": "en", "@vocab": "https://schema.org/", "citeAs": "cr:citeAs",
    "column": "cr:column", "conformsTo": "dct:conformsTo", "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/", "prov": "http://www.w3.org/ns/prov#",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"}, "dct": "http://purl.org/dc/terms/",
    "examples": {"@id": "cr:examples", "@type": "@json"}, "extract": "cr:extract",
    "field": "cr:field", "fileProperty": "cr:fileProperty", "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet", "format": "cr:format", "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset", "jsonPath": "cr:jsonPath", "key": "cr:key",
    "md5": "cr:md5", "parentField": "cr:parentField", "path": "cr:path",
    "recordSet": "cr:recordSet", "references": "cr:references", "regex": "cr:regex",
    "repeated": "cr:repeated", "replace": "cr:replace", "sc": "https://schema.org/",
    "separator": "cr:separator", "source": "cr:source", "subField": "cr:subField",
    "transform": "cr:transform", "samplingRate": "cr:samplingRate",
    "equivalentProperty": "cr:equivalentProperty",
}
FORMATS = {".csv": "text/csv", ".json": "application/json", ".npz": "application/x-npz",
           ".md": "text/markdown"}
# Croissant description of each file (fnmatch patterns, first match wins), as in the card's
# Files table.
FILE_DESCRIPTIONS = (
    ("splits/t1ml_*.csv", "Coarse-class split: bgc_id, split and the six MIBiG class labels."),
    ("splits/t2ml_stratified_5fold.csv", "Fine-class five-fold split stratified on the 48 "
     "labels (fold 4 is the test fold): bgc_id, fold, primary_class, labels."),
    ("splits/t2ml_*.csv", "Fine-class holdout split: bgc_id, split, primary_class, labels."),
    ("splits/t1_*.csv", "Chemical-property split: bgc_id, split, primary_class."),
    ("splits/t3_class_holdout.csv", "Retrieval class holdout: bgc_id and the primary_class "
     "held out in its fold (seven folds)."),
    ("splits/t3_*.csv", "Retrieval split: bgc_id, split, primary_class."),
    ("splits/t5_*.csv", "A-domain substrate split: domain_id, bgc_id, gcf_id, substrate, "
     "substrate_group, level_b (scored), evidence_tier, and fold or split."),
    ("tasks/t2ml_labels.csv", "Fine-class labels: the NPClassifier classes of all products, "
     "those among the 48 scored classes, and whether the BGC is scored."),
    ("tasks/chem_descriptors.csv", "13 RDKit 2026.3.1 descriptors of the first product SMILES."),
    ("tasks/t3_bgc_compound_pairs.csv", "BGC-compound pairs: compound_id, canonical SMILES, "
     "rank among the BGC's products and listing multiplicity."),
    ("tasks/t3_universe.csv", "The 1,939 BGCs scored in retrieval."),
    ("tasks/compounds.npz", "compound_ids, canonical_smiles and 768-d MoLFormer vectors of the "
     "3,118 distinct products (NumPy .npz)."),
    ("tasks/compounds_extraction_report.json", "How the MoLFormer vectors were computed and the "
     "count checks they pass."),
    ("tasks/chem_encoders/encoders.json", "Definition, width and weights of each molecule "
     "vector set in tasks/chem_encoders/."),
    ("tasks/chem_encoders/*.npz", "compound_ids and embeddings of the 3,118 molecules from one "
     "chemical encoder, in the order of tasks/compounds.npz (see encoders.json)."),
    ("tasks/adomain_instances.csv", "NRPS A-domains: parent BGC and gene, PF00501 envelope and "
     "protein sequence, Stachelhaus code, substrate, token positions in the Pfam lists."),
    ("tasks/adomain_nucleotide.json", "Codons of each A-domain envelope, keyed by domain_id."),
    ("inputs/pfam_domains.json", "Ordered Pfam accessions per BGC."),
    ("inputs/pfam_domains_named.json", "Ordered Pfam names per BGC."),
    ("inputs/pfam_acc_to_name.json", "Pfam accession to name map (24,076 families)."),
    ("inputs/protein_sequences.json", "CDS translations with coordinates per BGC."),
    ("inputs/nucleotide_seqs.json", "Cluster DNA per BGC (GenBank records)."),
    ("inputs/bag_of_pfam_vocab.json", "Pfam vocabulary of the Bag-of-Pfam baseline."),
    ("inputs/pfam_rand_vectors.npz", "Seeded random 256-d Pfam vectors of the Random-Pfam "
     "baseline (a generated control)."),
    ("gcf/gcf_cutoff_summary.csv", "BiG-SCAPE 2.0.3 family statistics per cutoff."),
    ("gcf/gcf_assignments_*.csv", "BiG-SCAPE 2.0.3 gene cluster family of each BGC at one "
     "cutoff (the benchmark uses 0.7)."),
    ("provenance/mibig_metadata.csv", "MIBiG 4.0 classes, products and organism per BGC "
     "(provenance; not read by the scorer)."),
    ("provenance/npclassifier_labels.csv", "NPClassifier pathway, superclass and class per "
     "product (provenance; not read by the scorer)."),
    ("provenance/murcko_scaffolds.csv", "Bemis-Murcko scaffold per product (provenance; not "
     "read by the scorer)."),
)


def _describe(rel: str) -> str:
    for pattern, text in FILE_DESCRIPTIONS:
        if fnmatch.fnmatch(rel, pattern):
            return text
    raise SystemExit(f"no Croissant description for {rel}; add one to FILE_DESCRIPTIONS")


def _datatype(series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "sc:Boolean"
    if pd.api.types.is_integer_dtype(series):
        return "sc:Integer"
    if pd.api.types.is_float_dtype(series):
        return "sc:Float"
    return "sc:Text"


def _file_checksum(path: Path) -> str:
    """sha256 of a file: Croissant expects a checksum for every FileObject."""
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


CITATION = """@inproceedings{xiao2026bgcbench,
  title     = {Data-Scarce, Multi-Granular, Multimodal: Benchmark Design for Biosynthetic Gene Clusters},
  author    = {Xiao, Hanlin and Takano, Eriko and {\\'A}lvarez, Mauricio A. and Breitling, Rainer},
  booktitle = {Advances in Neural Information Processing Systems (Evaluations and Datasets Track)},
  year      = {2026}
}"""
# Croissant RAI and provenance fields, from the dataset card's build, use and maintenance
# sections.
RAI = {
    "rai:dataCollection": "Derived from the MIBiG 4.0 release (annotations and GenBank "
                          "records); Pfam domains and gene cluster families computed with "
                          "BiG-SCAPE 2.0.3; product classes from NPClassifier; molecular "
                          "descriptors and scaffolds from RDKit 2026.3.1. The only generated "
                          "content is seeded random control vectors "
                          "(inputs/pfam_rand_vectors.npz, tasks/chem_encoders/random*.npz).",
    "rai:dataCollectionMissingData": "BGCs without a product SMILES are absent from the "
                                     "chemical tasks, 5 BGCs have no Pfam domain, the 739 "
                                     "domains (0.7 %) of families missing from the Pfam name "
                                     "map are left out of the named lists, and products "
                                     "NPClassifier leaves unclassified carry no fine label.",
    "rai:dataLimitations": "MIBiG is small and curated. Labels inherit their tools' "
                           "assumptions (NPClassifier, RDKit, BiG-SCAPE, a count-matching rule "
                           "for A-domains).",
    "rai:dataBiases": "Well-studied taxa and classes are over-represented: 755 of the 2,437 "
                      "active MIBiG entries come from Streptomyces.",
    "rai:personalSensitiveInformation": "None: the data describe microbial gene clusters and "
                                        "their products.",
    "rai:dataUseCases": "Comparing frozen BGC representations on five tasks under random, "
                        "scaffold, gene-cluster-family and class holdouts. The scores rank "
                        "encoders on MIBiG-derived labels; they are not established as "
                        "estimates of absolute performance on other clusters (for example from "
                        "metagenomes), for product-structure prediction or design, or for "
                        "decisions outside model comparison.",
    "rai:dataSocialImpact": "Better BGC representations serve natural-product discovery "
                            "(antibiotics, anticancer agents, agricultural bioactives). Risks "
                            "are over-reading small-sample scores and carrying MIBiG's taxonomic "
                            "skew into model selection; the release ships bootstrap intervals "
                            "and out-of-distribution columns to counter both. It adds no "
                            "capability for designing harmful compounds beyond what MIBiG and the "
                            "evaluated models already provide.",
    "rai:hasSyntheticData": False,
    "rai:dataReleaseMaintenancePlan": "Maintained by the authors. Each release is an immutable "
                                      "tag of the Hugging Face repository; corrections or "
                                      "extensions get a new tag, and the BGC-Bench code pins the "
                                      "tag it was tested with. Problems are reported as issues "
                                      "at https://github.com/Nugkta/BGC-Bench.",
    "prov:wasDerivedFrom": [
        {"@id": "https://doi.org/10.5281/zenodo.14835872",
         "name": "MIBiG 4.0 GenBank records (mibig_gbk_4.0.tar.gz)"},
        {"@id": "https://dl.secondarymetabolites.org/mibig/mibig_json_4.0.tar.gz",
         "name": "MIBiG 4.0 annotations (mibig_json_4.0.tar.gz)"},
        {"@id": "https://www.ebi.ac.uk/interpro/entry/pfam/", "name": "Pfam-A"},
    ] + [{"@id": f"https://huggingface.co/{w['repo_id']}/tree/{w['revision']}",
          "name": f"{w['repo_id']} (molecule vectors)"} for w in CHEM_WEIGHTS.values()],
    "prov:wasGeneratedBy": [
        {"@type": "prov:Activity", "prov:label": label, "sc:description": text}
        for label, text in (
            ("Pfam annotation", "BiG-SCAPE 2.0.3 (pyhmmer, Pfam trusted cutoffs) on Pfam-A.hmm "
             "downloaded 2026-04-06; domains ordered by CDS start, then envelope start."),
            ("Gene cluster families", "BiG-SCAPE 2.0.3 cluster on the GenBank records, "
             "cutoffs 0.3-0.9."),
            ("Product classes", "NPClassifier (GNPS2 API) queried 2026-04-06 and 2026-10-08."),
            ("Molecules", "RDKit 2026.3.1 canonical SMILES, descriptors and Bemis-Murcko "
             "scaffolds."),
            ("A-domains", "MIBiG module annotations paired with PF00501 hits by order within "
             "each gene."),
            ("Splits", "Random (seed 42), scaffold, gene-cluster-family (cutoff 0.7) and class "
             "holdouts."),
            ("Molecule vectors", "MoLFormer, ChemBERTa-2 and Uni-Mol vectors at pinned model "
             "revisions; RDKit fingerprints and descriptors; PCA to 128 dimensions."),
        )
    ],
}


def write_card(out: Path, version: str) -> None:
    """README.md (dataset card, from tools/dataset_card.md) and croissant.json."""
    card = ((Path(__file__).parent / "dataset_card.md").read_text()
            .replace("{github}", GITHUB).replace("{version}", version))
    (out / "README.md").write_text(card)
    files = [rel for rel in release_files(out)
             if rel not in ("croissant.json", "manifest.json", "README.md")]
    dist, records = [], []
    today = datetime.date.today().isoformat()
    for rel in files:
        p = out / rel
        dist.append({"@type": "cr:FileObject", "@id": rel, "name": rel,
                     "contentUrl": f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{version}/{rel}",
                     "description": _describe(rel),
                     "encodingFormat": FORMATS.get(p.suffix, "application/octet-stream"),
                     "contentSize": f"{p.stat().st_size} B", "sha256": _file_checksum(p)})
        if p.suffix == ".csv":
            df = pd.read_csv(p, low_memory=False)
            rid = rel.removesuffix(".csv").replace("/", "_")
            records.append({"@type": "cr:RecordSet", "@id": rid, "name": rid, "field": [
                {"@type": "cr:Field", "@id": f"{rid}/{c}", "name": c, "dataType": _datatype(df[c]),
                 "source": {"fileObject": {"@id": rel}, "extract": {"column": c}}}
                for c in df.columns]})
    meta = {
        "@context": CROISSANT_CONTEXT, "@type": "sc:Dataset", "name": "BGC-Bench",
        "conformsTo": ["http://mlcommons.org/croissant/1.1",
                       "http://mlcommons.org/croissant/RAI/1.0"],
        "description": "Five frozen-embedding tasks (coarse and fine class, chemical "
                       "properties, BGC-to-compound retrieval, A-domain substrate) on MIBiG 4.0 "
                       "biosynthetic gene clusters, with random, scaffold, gene-cluster-family "
                       "and class holdout splits, and model inputs.",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "keywords": ["biosynthetic gene clusters", "natural products", "MIBiG", "benchmark",
                     "representation learning", "protein language models"],
        "url": f"https://huggingface.co/datasets/{HF_DATASET}", "version": version.lstrip("v"),
        "datePublished": today, "dateModified": today,
        "isLiveDataset": False,
        "creator": [{"@type": "sc:Person", "name": n} for n in
                    ("Hanlin Xiao", "Eriko Takano", "Mauricio A. Álvarez", "Rainer Breitling")],
        "citeAs": CITATION,
        "isBasedOn": {"@type": "sc:Dataset", "name": "MIBiG 4.0",
                      "url": "https://mibig.secondarymetabolites.org/",
                      "license": "https://creativecommons.org/licenses/by/4.0/"},
        **RAI,
        "distribution": dist, "recordSet": records,
    }
    (out / "croissant.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
    print(f"README.md, croissant.json: {len(dist)} files, {len(records)} record sets")


def write_manifest(out: Path) -> None:
    """manifest.json: every file of the release folder with its size in bytes."""
    files = {rel: {"size": (out / rel).stat().st_size}
             for rel in release_files(out) if rel != "manifest.json"}
    (out / "manifest.json").write_text(json.dumps({"files": files}, indent=1, sort_keys=True) + "\n")
    print(f"manifest.json: {len(files)} files")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dataset")
    d.add_argument("--research", type=Path, required=True)
    d.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--out", type=Path, required=True)
    k = sub.add_parser("card")
    k.add_argument("--out", type=Path, required=True)
    k.add_argument("--version", default=None, help="dataset tag (default: DATASET_REVISION)")
    c = sub.add_parser("mirror")
    c.add_argument("key", choices=sorted(MIRRORS))
    c.add_argument("--dir", type=Path, required=True)
    u = sub.add_parser("upload")
    u.add_argument("--dir", type=Path, required=True)
    u.add_argument("--repo", required=True)
    u.add_argument("--type", choices=["model", "dataset"], required=True)
    u.add_argument("--tag", required=True)
    u.add_argument("--private", action="store_true", help="create a new repo as private")
    args = ap.parse_args()
    if args.cmd == "dataset":
        build_dataset(args.research.resolve(), args.out)
    elif args.cmd == "manifest":
        write_manifest(args.out)
    elif args.cmd == "card":
        import bgcbench

        write_card(args.out, args.version or bgcbench.DATASET_REVISION)
    elif args.cmd == "mirror":
        write_model_card(args.key, args.dir)
    elif args.cmd == "upload":
        upload(args.dir, args.repo, args.type, args.tag, private=args.private)


if __name__ == "__main__":
    main()
