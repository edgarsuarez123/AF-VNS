# AF VNS — Round 10: Threshold Tuning + Better Data + Better Features + Partial Unfreeze

## Docs Update — 2026-10-05
- [x] Rewrote README.md as unified 3-pipeline reference with Mermaid architecture diagrams
- [x] Added `scripts/demo_af_pipeline.py` — synthetic ECG+HRV inference demo, no dataset required
- Stroke demo: `scripts/demo_stroke_pipeline.py` lives on `feature/stroke-avns`
- Tinnitus demo: `scripts/demo_tinnitus_pipeline.py` lives on `feature/tinnitus-avns`

## Context

Round 9 achieved C2017 OOD AUROC=0.7432 (gap=0.0068 to NFR-2.1 target of 0.75).
Sensitivity is clinically strong (91.6%) but specificity is poor (32.9%) — the model
is confidently predicting AF on many NSR records. Four improvements ordered by risk/reward:

1. **Option 4** — Threshold analysis (instant, no retraining)
2. **Options 1+3** — Add MIMIC controls to Phase 2 + add pNN50/CoV features (single cache rebuild)
3. **Option 2** — Partial CNN unfreezing in Phase 2 (retrain Phase 2 only)

AUROC trajectory: 0.6022 → 0.6333 → 0.6747 → 0.6777 → **0.7432 (C2017 OOD)**

---

## Steps

| Step | Description | Status | Completed At |
|------|-------------|--------|--------------|
| 43 | Add `analyze_thresholds()` to evaluate.py + `_run_c2017_inference()` helper | Done | 2026-03-17 |
| 44 | Make threshold configurable across all eval functions (config.yaml) | Done | 2026-03-17 |
| 45 | Tests for threshold analysis + run analysis | Done | 2026-03-17 — 7/7 tests pass |
| 46 | Add pNN50 and CoV to hrv_time.py | Pending | |
| 47 | Update FEATURE_ORDER in pipeline.py (7→9) | Pending | |
| 48 | Update config.yaml hrv_n_features 7→9 + verify refs | Pending | |
| 49 | Tests for new HRV features | Pending | |
| 50 | Update test fixtures from 7→9 features | Pending | |
| 51 | Add MIMIC controls to Phase 2 splitter | Pending | |
| 52 | Add `collect_mimic_labels()` to dataset_parsers.py | Pending | |
| 53 | Update precompute_cache.py for split-based Phase 2 filtering + tests | Pending | |
| 54 | Cache rebuild + retrain Phase 1 & 2 + evaluate (background) | Pending | |
| 55 | Restructure CNN into named blocks (early_layers, last_conv_block) | Pending | |
| 56 | Modify Phase 2 freezing for partial unfreeze + discriminative LR | Pending | |
| 57 | Tests for partial unfreezing | Pending | |
| 58 | Retrain Phase 2 + evaluate | Pending | |

---

## Resume From Here

**Current state (2026-03-17) — Round 10 STARTING:**
- Round 9 complete: AUROC=0.7432, Sens=0.9159, Spec=0.3286
- Phase 2 checkpoint: models/checkpoints/phase2_model.pth
- Start with Step 43 (threshold analysis)
