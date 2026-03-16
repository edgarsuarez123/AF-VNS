# AF VNS — Round 9: Two-Phase Transfer Learning + OOD Evaluation

## Context

Round 8 AUROC = 0.6777 (target >= 0.75, gap = 0.0223).

**Major discovery (2026-03-16):** PhysioNet Challenge 2017 makes up 93% of training data
(5,788 of 6,213 subjects) but its 30-61s records can't produce meaningful 300s HRV features.
The GRU/Transformer are learning from zero-padded garbage. This is likely the primary
reason AUROC is stuck below 0.70.

**Architecture change:** Switching from single-phase joint training to two-phase transfer
learning, as specified in the project document (patent WO 2024/081854 A1):
- Phase 1: Pre-train ALL layers on MIMIC-3 (learn noisy ICU ECG patterns)
- Phase 2: Freeze backbone (CNN/GRU/Transformer), fine-tune head on AFDB/NSRDB/LTAFDB

**OOD evaluation:** Remove C2017 from training and use as sole OOD holdout (5,788 records,
ambulatory AliveCor ECG — completely different device from training data). No MIMIC-3
holdout needed anymore — all MIMIC-3 goes into Phase 1 training.

---

## Steps

| Step | Description | Status | Completed At |
|------|-------------|--------|--------------|
| 31 | Download 300 more MIMIC-3 records (n-af=350, n-control=350, seed=44) | In Progress | running as background process |
| 32 | Remove Challenge 2017 from training pipeline (config flag) | Done | 2026-03-16 |
| 33 | Create phase-specific splits (Phase 1: MIMIC-3 85/15, Phase 2: AFDB/NSRDB/LTAFDB 70/15/15) | Done | 2026-03-16 |
| 34 | Expand model head (Linear 256→64→1 for fine-tuning capacity) | Done | 2026-03-16 |
| 35 | Add phase filtering to precompute_cache.py (--phase 1 or 2) | Done | 2026-03-16 |
| 36 | Implement two-phase training in train.py (--phase, freezing, phase-aware paths) | Done | 2026-03-16 |
| 37 | Add C2017 OOD evaluation to evaluate.py (--challenge2017) | Done | 2026-03-16 |
| 38 | Build Phase 1 cache (MIMIC-3 only) + train Phase 1 | Not Started | waiting for Step 31 |
| 39 | Build Phase 2 cache (AFDB/NSRDB/LTAFDB) + train Phase 2 | Not Started | |
| 40 | Evaluate: Phase 2 test AUROC + C2017 OOD AUROC | Not Started | |
| 41 | Write tests for transfer learning | Done | 2026-03-16 |
| 42 | Update progress.txt with Round 9 results | Not Started | |

---

## Step 31: Download 300 More MIMIC-3

```bash
.venv/Scripts/python -m src.data.download_mimic3_waveforms \
  --n-af 350 --n-control 350 --seed 44 \
  --username edgarsuarez124 --password <redacted>
```

Existing 200+200 are skipped (counted toward target). Net new: ~150 AF + ~150 control.
Result: ~700 total MIMIC-3 records. ALL go into Phase 1 training (no holdout).

## Step 32: Remove C2017 from Training

- `config.yaml`: add `data.include_challenge2017: false`
- `dataset_parsers.py`: check flag in `parse_all()`, `collect_all_subject_ids()`, `iter_all_records()`
- Keep parser functions intact for OOD eval

## Step 33: Phase-Specific Splits

- `splitter.py`: new `create_phase_splits()` — separates by subject_id pattern
- Phase 1: ALL MIMIC-3 (^p\d{6}_), 85/15 train/val
- Phase 2: AFDB + NSRDB + LTAFDB (non-MIMIC, non-C2017), 70/15/15
- No MIMIC-3 holdout — C2017 is sole OOD

## Step 34: Expand Head

- Current: `Dropout(0.2) → Linear(256, 1)` = 257 params
- New: `Dropout(0.2) → Linear(256, 64) → ReLU → Dropout(0.1) → Linear(64, 1)` = ~16.5K params
- `config.yaml`: add `model.head_hidden_dim: 64`

## Step 35: Phase Filtering in Cache

- `precompute_cache.py`: add `--phase` arg
- Phase 1: only MIMIC-3 subjects → `cache_phase1/`
- Phase 2: only AFDB/NSRDB/LTAFDB → `cache_phase2/`

## Step 36: Two-Phase Training

- Phase 1: all layers trainable, lr=1e-3, patience=15, max_epochs=100
- Phase 2: freeze CNN/GRU/Transformer, train head only, lr=5e-4, patience=10, max_epochs=50
- Load Phase 1 checkpoint at start of Phase 2

## Step 37: C2017 OOD Eval

- `evaluate.py`: new `evaluate_challenge2017()` function
- CNN works on 30-61s records; HRV masking handles short context

## Expected Dataset

| Source | Records | Phase | Role |
|--------|---------|-------|------|
| MIMIC-3 | ~700 (ALL) | Phase 1 pre-train | Learn noisy ICU ECG |
| AFDB | 23 | Phase 2 fine-tune | AF morphology |
| NSRDB | 18 | Phase 2 fine-tune | NSR baseline |
| LTAFDB | 84 | Phase 2 fine-tune | Long-term AF patterns |
| Challenge 2017 | 5,788 | OOD eval (sole holdout) | True generalization metric |

---

## Resume From Here

**Current state (2026-03-16):**
- Round 8 complete. AUROC trajectory: 0.6022 → 0.6333 → 0.6747 → 0.6777
- Step 31 (MIMIC-3 download) in progress as background process
- Steps 32-37, 41 DONE (code changes + tests). 88/88 tests passing
- Commits: 10bebbd (Steps 32-36), 5ac6d8d (Steps 37, 41)
- Key insight: C2017 (93% of training) had garbage HRV — removing it + two-phase transfer learning is the path to 0.75+
