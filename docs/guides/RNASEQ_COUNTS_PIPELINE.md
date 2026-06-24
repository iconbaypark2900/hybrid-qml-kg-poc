# RNA-seq Counts Pipeline

This project supports a counts-first RNA-seq workflow for local/DGX analysis.
It does not perform wet-lab sequencing, FASTQ trimming, alignment, or transcript
quantification in this milestone.

## Accepted Inputs

- `.h5ad` files through the existing `single_cell_layer.ingest_h5ad` loader.
- 10x matrix directories through the existing `single_cell_layer.ingest_10x`
  loader, with matching metadata.
- CSV/TSV gene-by-sample count matrices:

```text
gene,case_a,case_b,control_a,control_b
GENE1,100,120,10,12
```

For count matrices, provide metadata with sample IDs matching the count columns:

```text
sample_id,condition,batch,cell_type,tissue
case_a,disease,batch1,T_cell,blood
control_a,control,batch2,T_cell,blood
```

## Run

```bash
python scripts/run_rnaseq_counts_pipeline.py \
  --input data/rnaseq/counts.csv \
  --format count-matrix \
  --metadata data/rnaseq/metadata.csv \
  --out-dir artifacts/single_cell \
  --signatures-dir artifacts/signatures \
  --disease-id Disease::DOID:example
```

Outputs include QC summaries, normalized counts, differential-expression rows,
`disease_signature.json`, optional `cell_type_signatures.json`, and a manifest.

For an external validation cohort, normalize without deriving a signature from
validation labels:

```bash
python scripts/run_rnaseq_counts_pipeline.py \
  --input data/rnaseq/external_counts.csv \
  --format count-matrix \
  --metadata data/rnaseq/external_metadata.csv \
  --out-dir artifacts/single_cell/external \
  --skip-de
```

This emits QC, normalized counts, and a manifest only.

## Wet-Lab Handoff

Ask the sequencing provider or collaborator for either:

- gene-level count matrix plus metadata, or
- `.h5ad` / 10x matrix output.

Metadata must include disease/control labels. Batch, tissue, and cell-type labels
are strongly recommended because downstream KG+omics evidence uses them for
interpretability.

## Future FASTQ Extension

Raw FASTQ support should be added later as optional wrappers around open-source
tools such as FastQC/MultiQC, fastp, Salmon/kallisto, STARsolo, kb-python, or
alevin-fry. Those tools should remain optional and local-first.
