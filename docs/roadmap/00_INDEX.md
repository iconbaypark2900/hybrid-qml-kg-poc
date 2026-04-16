# Roadmap — Master Index

**Last updated:** 2026-04-16  
**Scope:** Everything not yet done, ordered by scientific impact and delivery risk.

This folder documents gaps between the current state of the project and a
publication-ready, production-grade hybrid QML drug-repurposing platform.
Each file is self-contained and actionable.

---

## Files in this folder

| File | Covers | Est. effort |
|------|--------|-------------|
| [01_paper_v2_experiments.md](01_paper_v2_experiments.md) | Missing runs explicitly promised for paper v2 | 2–4 days |
| [02_scientific_gaps.md](02_scientific_gaps.md) | Methodological gaps that weaken the scientific claim | 1–3 days |
| [03_frontend_completion.md](03_frontend_completion.md) | Next.js UI parity with Streamlit + new features | 3–5 days |
| [04_quantum_scaling.md](04_quantum_scaling.md) | Quantum path improvements: Nyström, VQC, KTA, kernel learning | 3–7 days |
| [05_platform_extensions.md](05_platform_extensions.md) | Longer-term extensions: DRKG, multi-relational, ClinicalTrials API | 2–4 weeks |

---

## Priority order

### Tier 1 — Blocks arXiv v2 submission
1. MoA feature benchmark (`--use_moa_features` on primary CtD config)
2. CpD relation run (`--relation CpD`, no code changes needed)
3. Multi-seed evaluation (5 seeds, report mean ± std)
4. Fix LaTeX citation keys (`[?]` in every in-text citation)
5. Render and commit the 3 required figures (Appendix A spec)
6. Add degree-heuristic and random baselines to Table 3

### Tier 2 — Strengthens the quantum claim
7. Ablation matrix: 4 conditions (A classical-full, B classical-16D, C quantum-only, D ensemble)
8. Kernel-target alignment analysis as a feature-map selection diagnostic
9. Noisy simulator benchmark (`config/quantum_config_noisy.yaml`)
10. IBM Quantum Heron full benchmark with noise characterization

### Tier 3 — Frontend parity
11. Pipeline job triggering from Next.js UI
12. Per-prediction MoA explanation panel
13. Benchmark registry UI (run comparison table)
14. Experiment history chart
15. Remaining placeholder pages: `/quantum`, `/visualization` review

### Tier 4 — Platform extensions
16. ClinicalTrials.gov live query integration
17. CpD, DrD, DaG, CbG multi-relational extension
18. Nyström approximation at scale (CbG 11K edges)
19. VQC architecture search (100+ iterations, reps=6–8)
20. DRKG extension (4.4M edges, 97K entities)
21. Variational quantum kernel learning

---

## What is NOT in scope here

These are covered by existing docs and should not be duplicated:

- Hard negative mining wire-up → `docs/CHANGES_NEEDED.md` §1
- Benchmark registry creation → `docs/CHANGES_NEEDED.md` §3
- `train_on_heron.py` spec → `docs/CHANGES_NEEDED.md` §2
- `docs/BENCHMARK_SPEC.md` creation → `docs/CHANGES_NEEDED.md` §4
- Embedding coverage gap fix → `docs/CHANGES_NEEDED.md` §5
- Frontend rollout plan → `docs/planning/FRONTEND_ROLLOUT_PLAN.md`
