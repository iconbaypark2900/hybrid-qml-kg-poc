# ZNE implementation vs preregistration — audit outline

Working document for **Task 2.3** in the publication roadmap: reconcile code with
[`preregistration/osf_preregistration_v1.md`](../../preregistration/osf_preregistration_v1.md)
**§5.7 Pauli Path ZNE error mitigation** (and related hardware context in **§5.6**).

Update each row after code review: **Match**, **Partial**, or **Gap**, with a one-line note.

## §5.7 — Required behavior (from preregistration)

For each hardware kernel matrix entry, the study commits to:

| # | Requirement | Implementation file(s) to inspect | Status | Notes |
|---|-------------|-------------------------------------|--------|-------|
| 1 | Noise scaling **1×** (raw) | `quantum_layer/quantum_error_mitigation.py`, `advanced_error_mitigation.py`, call sites in pipeline/executor | | |
| 2 | Noise scaling **3×** via fold-based circuit folding | same | | |
| 3 | Noise scaling **5×** via fold-based circuit folding | same | | |
| 4 | Extrapolate to zero-noise via **linear regression** | same | | |
| 5 | Extrapolate via **Richardson extrapolation** | same | | |
| 6 | **Report** raw unmitigated + both extrapolations | logging / results schema / paper-facing exports | | |

## §5.6 — Hardware routing (context)

| # | Requirement | Implementation | Status | Notes |
|---|-------------|----------------|--------|-------|
| 1 | Primary **IBM Torino**, backup **Brisbane** | `quantum_layer/quantum_executor.py`, configs under `config/quantum_config*.yaml` | | |
| 2 | Scope: 3 problem sizes × 3 snapshots × 100 samples | job scripts / orchestration | | |

## §5.5 — Variational classifier (out of scope for ZNE, but same files often touched)

| # | Prereg statement | Code | Status | Notes |
|---|------------------|------|--------|-------|
| 1 | VQC supplementary; best ~0.5474 RealAmplitudes | `quantum_layer/qml_model.py` | | |

## Outcomes

- **If all §5.7 rows Match:** hardware experiments are primarily **configuration + execution** (Task 2.4).
- **Any Gap:** document required code or protocol changes before claiming preregistered ZNE in H2/H3.

*Last scaffolded: documentation pass — replace statuses with findings as the audit proceeds.*
