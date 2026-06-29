# CREEDS Coverage — Breast CtD Repurposing

**Policy (paper Methods):** Primary evidence uses **human** CREEDS profiles only (`--creeds-organism human`). When human profiles are absent, reversal scores are **0.0** and candidates rank on KG+QML only. An exploratory policy (`--creeds-organism any`) includes rat/mouse profiles and must be labeled **non-human** in UI and text.

## Breast cancer (`Disease::DOID:1612`) — 11 CtD candidates from multiseed KG export

| Match (human) | Match (any) | Compound | Notes |
|---------------|-------------|----------|-------|
| yes | yes | Vemurafenib | Human CREEDS profile |
| yes | yes | Cisplatin | Human CREEDS profile |
| yes | yes | Paclitaxel | Human CREEDS profile |
| yes | yes | Doxorubicin | Human CREEDS profile |
| no | yes | Prednisolone | Rat profile only |
| no | yes | Irinotecan | Rat profile only |
| no | no | Toremifene | No CREEDS overlap |
| no | no | Vinblastine | No CREEDS overlap |
| no | no | Fingolimod | No CREEDS overlap |
| no | no | Capecitabine | No CREEDS overlap |
| no | no | Dacarbazine | No CREEDS overlap |

**Counts:** 4/11 human, 6/11 with `organism=any`, 5/11 permanently unmatched in current CREEDS v1.0 index.

## Full 200-pair export (human, cosine)

From `repurposing_full_200_cosine`: **46/200** CREEDS matched (23%). Unmatched pairs retain KG-dominated fusion scores.

## Data gap mitigation (future)

1. Supplement CREEDS with LINCS/CMap tidy profiles for unmatched DrugBank IDs.
2. Keep `organism=any` as a separate workbench disease (`brca_external_validation_organism_any`) — already wired.
3. Do **not** treat unmatched reversal=0 as negative evidence; it means **missing perturbation data**.

## Artifacts

- Human bundle: `artifacts/repurposing/brca_external_validation/`
- Any-organism bundle: `artifacts/repurposing/brca_external_validation_organism_any/`
- 200-pair run: `results/rnaseq_repurposing_run/repurposing_full_200_cosine/`
