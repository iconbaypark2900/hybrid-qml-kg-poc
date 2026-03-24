# Multi-model fusion pipeline integration

Design notes, checklist, and acceptance criteria for wiring `MultiModelFusion` into [scripts/run_optimized_pipeline.py](../../scripts/run_optimized_pipeline.py).

*To be filled when integration work begins.*

## Checklist

- [ ] Identify insertion point in pipeline (after ensemble / classical / QSVC fit)
- [ ] Define inputs: RF, ET, QSVC predictions (or equivalent models)
- [ ] Choose default fusion method (e.g. bayesian_averaging per NEXT_TASKS_IMPLEMENTATION)
- [ ] Add `--run_multimodel_fusion` flag and `--fusion_method` option
- [ ] Wire fusion output into ranking and JSON results
- [ ] Update OPTIMIZATION_QUICKSTART with fusion usage
