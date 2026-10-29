---
license: cc-by-4.0
pretty_name: BGC-Bench
size_categories:
- 1K<n<10K
tags:
- biology
- biosynthetic-gene-clusters
- natural-products
- genomics
- benchmark
- mibig
viewer: false
---

# BGC-Bench (MIBiG 4.0)

BGC-Bench evaluates frozen biosynthetic gene cluster (BGC) representations on five tasks
under random and out-of-distribution splits. Linear probes score four of the tasks and a
linear contrastive head the fifth (retrieval), giving fifteen evaluation columns. This card
describes release {version}.

Existing BGC benchmarks mostly classify clusters into a few coarse classes on small curated
datasets, which separates models poorly and says little about downstream use. BGC-Bench adds
finer and chemical targets, BGC-to-molecule retrieval and A-domain substrates, scored under
random, scaffold, gene-cluster-family and class holdouts with bootstrap intervals, so that
encoders can be compared with statistical confidence under distribution shift.

The dataset (about 220 MB) contains MIBiG 4.0 inputs, labels and splits: ordered Pfam
domains, CDS proteins, cluster DNA and A-domain sequences. The code at {github} downloads a
pinned revision.

```bash
git clone {github} && cd BGC-Bench && uv sync
uv run bgcbench run --encoder deepbgc          # a built-in encoder (runs on the CPU)
uv run bgcbench evaluate --embeddings my.npz   # your own vectors (bgc_ids + embeddings)
```

## The five tasks (fifteen columns)

| task | question | metric | splits (columns) |
|---|---|---|---|
| coarse class | which of six MIBiG classes (multi-label) | macro-AUROC, one-vs-rest logistic regression | `t1ml_random`, `t1ml_scaffold_holdout`, `t1ml_gcf_holdout_c07` (R, S, G) |
| fine class | which of 48 NPClassifier classes the cluster's products belong to (multi-label) | weighted macro-AUROC, one-vs-rest logistic regression | `t2ml_stratified_5fold` (fold 4 = test), `t2ml_scaffold_holdout`, `t2ml_gcf_c07_holdout` |
| chemical properties | 13 RDKit descriptors of the first product | mean Spearman rho, ridge regression | `t1_random`, `t1_scaffold_holdout`, `t1_gcf_holdout_c07` |
| retrieval | rank the cluster's products in the test-molecule gallery | class-macro R@10 (a query is a hit when any of its products ranks in the top 10; ties scored by expectation), linear contrastive head on frozen MoLFormer molecule vectors, mean of 3 seeds | `t3_random`, `t3_scaffold_holdout`, `t3_gcf_holdout_c07`, `t3_class_holdout` (7 folds) |
| A-domain substrate | substrate (an amino or aryl acid) activated by an NRPS adenylation domain (28 classes) | top-1 accuracy, logistic regression | `t5_bgc_grouped_5fold`, `t5_gcf_holdout_c07` |

**Split axes.** *Random*: 80/20 over BGCs (seed 42); fine classes test fold 4 of a five-fold split
stratified on the 48 labels, and A-domain substrates use all five folds, grouped by BGC.
*Scaffold holdout*: 20 % of Bemis-Murcko
scaffold groups of the first product go to test (ring-free products form one group).
*GCF holdout*: 20 % of BiG-SCAPE 2.0.3 gene cluster families at cutoff 0.7 go to test.
*Class holdout* (retrieval): each MIBiG class, or the multi-class "Hybrid" group, is held out
in turn. Each holdout controls its own axis only: a scaffold-holdout test BGC can share a
family with training, and vice versa.

## Files

| path | content |
|---|---|
| `splits/` | the 15 split files (`bgc_id` or `domain_id`; `split`, `fold`, or the held-out `primary_class` for the retrieval class holdout; labels) |
| `tasks/t2ml_labels.csv` | fine-class labels of the 1,710 BGCs with a Pfam domain and a classified product: the NPClassifier classes of all products (`classes_all`), those among the 48 (`labels`), and whether the BGC is scored (`enrolled`, 1,221) |
| `tasks/chem_descriptors.csv` | 13 RDKit 2026.3.1 descriptors of the first product SMILES, 1,944 BGCs |
| `tasks/t3_bgc_compound_pairs.csv` | (BGC, compound) pairs: `compound_id` = `cmp_` + sha1(canonical SMILES)[:12], `mibig_order` = rank among the BGC's distinct products by first listing, `multiplicity` = times listed |
| `tasks/t3_universe.csv` | the 1,939 BGCs scored in retrieval (3,426 pairs, 3,110 molecules) |
| `tasks/compounds.npz` | `compound_ids`, `canonical_smiles` and 768-d MoLFormer vectors (`ibm/MoLFormer-XL-both-10pct` @ `7b12d946`, pooler output) of the 3,118 distinct products |
| `tasks/compounds_extraction_report.json` | how the MoLFormer vectors were made (model, revision, library versions, pooling) and the count checks they pass |
| `tasks/chem_encoders/<name>.npz` | `compound_ids` and `embeddings` of the same 3,118 molecules, in the order of `tasks/compounds.npz`, from the paper's chemical-encoder comparison: `molformer` (768-d), `chemberta2_mtr` (ChemBERTa-2 77M-MTR, 384-d), `unimol` (Uni-Mol v1, 512-d), `ecfp4` (2,048 bits), `atompair` (2,048 bits), `maccs` (167 keys), `rdkit_desc` (209 RDKit descriptors) and a `random` control (768-d), each also as `<name>_pca128` (PCA to 128 dimensions, fitted on all 3,118 molecules) |
| `tasks/chem_encoders/encoders.json` | per file: display name, family, native or PCA, width, how the vectors were computed and, for the neural encoders, the Hub repository and revision of the weights |
| `tasks/adomain_instances.csv` | 2,504 NRPS A-domains: parent BGC and gene, PF00501 envelope (`env_start`, `env_stop`, `aa_seq`), Stachelhaus code, substrate, position of the domain's token in the Pfam lists (`token_index` in the named list, `token_index_raw` in the accession list) |
| `tasks/adomain_nucleotide.json` | the codons of each A-domain envelope (back-mapped through the parent CDS, translation-checked) |
| `inputs/pfam_domains.json` | ordered Pfam accessions per BGC (2,631 BGCs; 5 have none) |
| `inputs/pfam_domains_named.json` | the same lists as Pfam names (2,626 BGCs), for name-keyed models (BiGCARP, Kawano) |
| `inputs/pfam_acc_to_name.json` | Pfam accession to name map (24,076 families) |
| `inputs/protein_sequences.json` | CDS translations with coordinates per BGC (2,631 BGCs) |
| `inputs/nucleotide_seqs.json` | cluster DNA per BGC (2,636 GenBank records) |
| `inputs/bag_of_pfam_vocab.json`, `inputs/pfam_rand_vectors.npz` | fixed files of the two Pfam baselines (3,932 accessions; N(0, 1) 256-d vectors, seed 42) |
| `gcf/` | BiG-SCAPE 2.0.3 family assignments at cutoffs 0.3-0.9 (the benchmark uses 0.7) and a per-cutoff summary |
| `provenance/` | MIBiG 4.0 metadata (classes, products, organism), NPClassifier labels and Murcko scaffolds per product; the sources of the labels and splits, not read by the scorer |
| [`manifest.json`](manifest.json) | every file with its size in bytes |
| [`croissant.json`](croissant.json) | Croissant 1.1 metadata with Responsible AI fields: one file object per file (description, size, sha256) and one record set per CSV |

## How it was built

- **MIBiG 4.0** (Zdouc et al., *Nucleic Acids Res.* 53:D678, 2025). Annotations from the
  release `mibig_json_4.0.tar.gz` (dl.secondarymetabolites.org): 3,013 entries, of which the
  2,437 active ones carry the labels. Sequences from the 4.0 GenBank release
  (`mibig_gbk_4.0.tar.gz`, Zenodo record
  14835872): 2,636 records, the 2,437 active entries plus 199 pending ones.
- **Cohorts.** 2,636 DNA records; BiG-SCAPE's 500 kb length cap leaves 2,631
  for the Pfam and protein inputs; 5 of those have no Pfam domain (2,626 named lists);
  2,433 active BGCs have a family; 1,944 active BGCs have a product SMILES; 1,939 of them
  are embedded by every encoder of the paper and form the retrieval cohort.
- **Classes.** MIBiG 4.0's six classes (PKS, NRPS, ribosomal, terpene, saccharide, other;
  stored as Polyketide, NRP, RiPP, Terpene, Saccharide, Other). 456 BGCs have more than one;
  they are positive for each in the coarse task and form "Hybrid" where one label is needed.
  The chemical-property scaffold split leaves out the 61 Saccharide-only BGCs.
- **Fine classes.** NPClassifier (Kim et al. 2021; GNPS2 API, queried 2026-04-06 and
  2026-10-08) classes of every product ("unclassified" is not a class). A BGC's labels are
  the union over its products, kept to the 48 classes held by at least 10 BGCs; the 1,221
  BGCs with at least one of them and at least one Pfam domain are scored. Every split trains
  all 48 classes; a class without a test positive on a split (4 on the scaffold holdout, 8 on
  the GCF holdout) is not scored there. The score is the per-class AUROC averaged with weights
  equal to each class's test positives (weighted macro-AUROC); micro-AUROC, micro-AUPRC and
  unweighted macro-AUROC are reported with it.
- **Molecules.** SMILES from the 4.0 JSON in listing order, canonicalised with RDKit
  2026.3.1 defaults (stereo tags kept); one row per distinct (BGC, molecule).
- **Pfam domains.** BiG-SCAPE 2.0.3 (pyhmmer, Pfam trusted cutoffs, domain overlap 0.1) on
  `Pfam-A.hmm` as downloaded on 2026-04-06 (27,481 profiles; entries dated to July 2025);
  domains ordered by CDS start, then envelope start. The accession-to-name map holds the
  ACC and NAME fields of an earlier Pfam-A release (24,076 families, version suffixes
  removed); the 739 domains (0.7 %) of families it lacks are left out of the named lists.
- **Families.** BiG-SCAPE 2.0.3 `cluster` on the GenBank records: region level, glocal
  alignment, mix weights, singletons included, affinity propagation; cutoffs 0.3-0.9.
- **A-domains.** MIBiG 4.0 module annotations name each A-domain's gene and substrate but
  give no coordinates; the i-th annotated A-domain of a gene is paired with the i-th
  PF00501 hit from the N terminus, and genes whose counts disagree are dropped (2,504
  domains, 529 BGCs). Scored: single-substrate domains whose substrate occurs at least 10
  times (`level_b`, 28 classes, 2,027 domains); folds are stratified on the NRPSPredictor2
  substrate group (`substrate_group`) and grouped by BGC. `evidence_tier` collapses MIBiG's
  evidence codes (`experimental` > `structure_inferred` > `homology` > `sequence_predicted`;
  `none` when MIBiG gives no code).

## Intended use and limitations

The benchmark compares frozen BGC feature extractors on the five tasks above. Its scores rank
encoders on MIBiG-derived labels; they are not established as estimates of absolute
performance on other clusters (for example from metagenomes or unannotated genomes), for
product-structure prediction or design, or for decisions outside model comparison.

- MIBiG is small and curated: it over-represents well-studied taxa (755 of the 2,437 active
  entries come from *Streptomyces*) and well-studied classes.
- Labels inherit their tools' assumptions: NPClassifier for fine classes, RDKit for
  descriptors and scaffolds, BiG-SCAPE for families, a count-matching rule for A-domains
  (some genes with integer-named modules may be mis-paired).
- Report overlap between an evaluated model's training labels and the benchmark labels,
  including supervised training on MIBiG products.
- Scores rest on 1,221 to 2,437 labelled rows per task; compare encoders with the bootstrap
  intervals and the holdout columns, not the random split alone.
- The data describe microbial gene clusters and their products and contain no personal
  data. Better BGC representations serve natural-product discovery; the benchmark adds no
  capability for designing harmful compounds beyond what MIBiG and the evaluated models
  already provide.

## Licence and attribution

The authors' files (splits, task tables, derived inputs) are released under CC BY 4.0.
Third-party content keeps its source's terms:

- MIBiG 4.0 annotations and sequences: CC BY 4.0; please cite MIBiG together with BGC-Bench.
- Pfam accessions, names and domain annotations (`inputs/pfam_*`, the Pfam-derived columns
  of `tasks/adomain_instances.csv`): Pfam (Mistry et al. 2021) data are freely available
  under CC0; the annotations were computed with BiG-SCAPE 2.0.3 (Navarro-Muñoz et al. 2020),
  which also produced the families in `gcf/`.
- NPClassifier classes (`provenance/npclassifier_labels.csv`, the fine-class labels): output
  of the NPClassifier web service (Kim et al. 2021), which declares no licence; cite
  NPClassifier when using them.
- Molecule vectors: MoLFormer (`ibm/MoLFormer-XL-both-10pct`, Apache-2.0), Uni-Mol
  (`dptech/Uni-Mol-Models`, MIT) and ChemBERTa-2 (`DeepChem/ChemBERTa-77M-MTR`, no licence
  declared upstream; the two `chemberta2_mtr` files are included to reproduce the paper's
  chemical-encoder comparison, and CC BY 4.0 covers them only as far as the authors hold
  rights in them); fingerprints and descriptors from RDKit 2026.3.1 (BSD-3-Clause).

## Citation

```bibtex
@inproceedings{xiao2026bgcbench,
  title     = {Data-Scarce, Multi-Granular, Multimodal: Benchmark Design for Biosynthetic Gene Clusters},
  author    = {Xiao, Hanlin and Takano, Eriko and {\'A}lvarez, Mauricio A. and Breitling, Rainer},
  booktitle = {Advances in Neural Information Processing Systems (Evaluations and Datasets Track)},
  year      = {2026}
}
@article{mibig4,
  title   = {{MIBiG} 4.0: advancing biosynthetic gene cluster curation through global collaboration},
  author  = {Zdouc, Mitja M. and others},
  journal = {Nucleic Acids Research}, volume = {53}, pages = {D678--D690}, year = {2025},
  doi     = {10.1093/nar/gkae1115}
}
@article{mistry2021pfam,
  title   = {{Pfam}: The Protein Families Database in 2021},
  author  = {Mistry, Jaina and Chuguransky, Sara and Williams, Lowri and others},
  journal = {Nucleic Acids Research}, volume = {49}, number = {D1}, pages = {D412--D419}, year = {2021},
  doi     = {10.1093/nar/gkaa913}
}
@article{navarro2020bigscape,
  title   = {A Computational Framework to Explore Large-Scale Biosynthetic Diversity},
  author  = {Navarro-Mu{\~n}oz, Jorge C. and Selem-Mojica, Nelly and Mullowney, Michael W. and others},
  journal = {Nature Chemical Biology}, volume = {16}, number = {1}, pages = {60--68}, year = {2020},
  doi     = {10.1038/s41589-019-0400-9}
}
@article{npclassifier,
  title   = {{NPClassifier}: A Deep Neural Network-Based Structural Classification Tool for Natural Products},
  author  = {Kim, Hyun Woo and Wang, Mingxun and Leber, Christopher A. and others},
  journal = {Journal of Natural Products}, volume = {84}, number = {11}, pages = {2795--2807}, year = {2021},
  doi     = {10.1021/acs.jnatprod.1c00399}
}
```

## Versions and maintenance

This is release {version}. Each release is a tag of this repository; tags are not moved or
deleted, and a corrected or extended dataset gets a new tag. The code pins the tag it was
tested with (`DATASET_REVISION` in `src/bgcbench/__init__.py`), so published results stay
reproducible. Maintained by the authors at the University of Manchester; report problems as
issues at {github}.
