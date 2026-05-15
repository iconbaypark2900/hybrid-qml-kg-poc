# Method Reproduction Checklist

Each item must pass before the OSF preprint is submitted (M5: 2026-08-31).

---

## Core QML-KG Pipeline

- [x] Hetionet v1.0 edges loaded from `data/hetionet-v1.0-edges.sif`
- [x] CtD (Compound-treats-Disease) edge extraction isolated from training data
- [x] RotatE embeddings trained at 128d (primary), 256d, 512d
- [x] Compound-disease pair features: Hadamard, L1, L2, average operators
- [x] Pauli feature map (depth 2) applied before QSVC
- [x] ZZ feature map (supplementary comparison)
- [x] QSVC trained with quantum kernel
- [x] VQC trained with parameterized ansatz
- [x] Classical baselines: LR, RF, GBM, XGB
- [x] Stacking ensemble (quantum + classical meta-learner)
- [x] Hard negative sampling during training
- [x] 5-fold nested cross-validation
- [x] PR-AUC primary metric (confirmed 0.7987 for headline config)
- [x] ROC-AUC secondary metric
- [x] Bootstrap confidence intervals
- [x] Preregistered seeds (`utils/preregistered_constants.py`)

## Single-Cell Evidence Layer

- [ ] h5ad / 10x Genomics format ingestion
- [ ] Cell quality filtering (min/max genes, mito %)
- [ ] Doublet detection (Scrublet)
- [ ] Log-normalization + highly variable gene selection
- [ ] PCA dimensionality reduction
- [ ] kNN graph + Leiden clustering
- [ ] UMAP visualization
- [ ] Harmony batch correction (multi-sample datasets)
- [ ] Wilcoxon differential expression (disease vs. control per cell type)
- [ ] Cell-type-specific disease signature export (JSON schema)
- [ ] Pathway enrichment (GSEA / ORA)

## Perturbation Reversal Layer

- [ ] LINCS L1000 signature loading
- [ ] Gene symbol normalization to HGNC
- [ ] Compound ID normalization to DrugBank / Hetionet namespace
- [ ] Per-compound up/down gene signature construction
- [ ] Gene-level rank correlation reversal score
- [ ] Cell-type-stratified reversal score
- [ ] Pathway-level reversal score
- [ ] Reversal feature vector exported for stacking

## Evidence Fusion

- [ ] KG + QML + omics features combined in single vector
- [ ] Configurable per-source weights
- [ ] Confidence tier assignment (Tier 1–4)
- [ ] Human-readable explanation string generated per candidate
- [ ] `--mode kg-only` reproduces original PR-AUC 0.7987
- [ ] `--mode kg+omics` delta PR-AUC logged

## Validation

- [ ] Top-N candidates checked against known indications (DrugCentral / ChEMBL)
- [ ] ClinicalTrials.gov phase label attached to each candidate
- [ ] PubMed co-occurrence count
- [ ] Candidates labeled: known / supported / novel / exploratory

## Operational

- [ ] `./scripts/dgx/run_smoke_test.sh` exits 0 from clean env
- [ ] Full pipeline reproducible from single command
- [ ] Artifacts committed with SHA256 manifest
