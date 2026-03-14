# AF VNS — Phase 1 Pipeline Fix Plan

## Context

Two rounds of fixes to get the training pipeline producing valid data:

**Round 1 (Steps 0-5, done 2026-03-13):** Fixed 7 bugs causing loss=0, AUROC=NaN — scaler per-column NaN handling, subwindow config, MIN_SAMPLES, HRV_STEPS constants. 25/25 tests passing.

**Round 2 (Steps 6-13, current):** Deep audit found 5 more issues that would invalidate the cache. Cache rebuild killed at 67% to avoid wasting time. All fixes applied before single rebuild.

---

## Step 0: Update PLAN.md

Replace the entire content of `PLAN.md` at project root with this plan (all steps, code changes, testing plan, and verification). This becomes the single source of truth for the fix effort.

**After**: Update `progress.txt` section 2.9 noting PLAN.md has been updated with the full fix plan.

---

## Step 1: Fix scaler transform (CRITICAL)

### 1a. `src/features/scaler.py` lines 94-103 — per-column transform

**Bug**: `nan_mask = np.any(np.isnan(X_flat), axis=1)` on line 99 leaves entire rows as NaN if ANY feature is NaN. After `nan_to_num(nan=0.0)`, valid features in sparse rows become 0.

**Fix**: Replace the row-level mask with a per-column loop (mirrors the per-column logic already in `fit_scaler()`).

**After**: Update `progress.txt` noting scaler transform() bug fixed (per-column NaN handling).

---

## Step 2: Fix scaler pre-filters

### 2a. `src/training/precompute_cache.py` lines 149-160 — remove row-level pre-filter

**Bug**: `mask = ~np.any(np.isnan(X), axis=1)` on line 151 requires ALL 7 features non-NaN per row. ~0 rows pass with sparse data.

**Fix**: Remove the mask. Pass all rows to `fit_scaler()` which already handles per-column NaN internally.

### 2b. `src/training/train.py` lines 98-105 — same pre-filter bug in on-the-fly path

**Fix**: Same pattern — remove the mask.

**After**: Update `progress.txt` noting both scaler pre-filter bugs fixed in precompute_cache.py and train.py.

---

## Step 3: Fix HRV feature density

### 3a. `src/features/pipeline.py` lines 46-59 — subwindow_sec default overrides config

**Bug**: Default `subwindow_sec: float = 30.0` is truthy, so `subwindow_sec or config.get(...)` on line 59 always short-circuits to 30.0. Config is never read.

**Fix**: Change default to `None`, use explicit `if is None`.

### 3b. `config.yaml` lines 26, 59 — subwindow and seq_len

- Line 26: `subwindow_sec: 30` -> `subwindow_sec: 60`
- Line 59: `hrv_seq_len: 10` -> `hrv_seq_len: 5`

### 3c. `src/features/hrv_freq.py` line 12 — lower MIN_SAMPLES

- `MIN_SAMPLES = 64` -> `MIN_SAMPLES = 32`

### 3d. Hardcoded HRV_STEPS constants

- `src/training/precompute_cache.py` line 34: derive from config instead of hardcoded 10.
- `src/training/train.py` line 68: `HRV_STEPS = 10` -> `HRV_STEPS = 5`, update comment.

**After**: Update `progress.txt` noting all HRV density fixes applied.

---

## Step 4: Testing plan (run BEFORE precompute cache)

Add 8 new tests to `tests/test_preprocessing.py`, then run the full test suite.

| # | Test Function | What It Verifies |
|---|--------------|------------------|
| 1 | `test_scaler_transform_sparse_nan` | Bug 1a: valid columns scaled even when other columns are NaN |
| 2 | `test_scaler_transform_fully_nan_rows` | Bug 1a edge: fully-NaN rows stay NaN, other rows scaled |
| 3 | `test_pipeline_shape_60s_subwindow` | Bugs 3a/3b: shape is (5, 7) with explicit 60s |
| 4 | `test_pipeline_reads_config_subwindow` | Bug 3a: config subwindow_sec read when not passed |
| 5 | `test_hrv_freq_with_32_to_63_samples` | Bug 3c: freq HRV non-NaN with 48 intervals |
| 6 | `test_hrv_freq_below_min_samples` | Bug 3c: freq HRV still NaN below 32 |
| 7 | `test_end_to_end_pipeline_scaler_nonzero` | Full chain: ECG -> pipeline -> scaler -> non-zero |
| 8 | `test_fit_scaler_with_all_nan_columns` | Bugs 2a/2b: fit_scaler handles all-NaN columns |

**Gate**: ALL tests (existing + new) must pass before proceeding to Step 5.

---

## Step 5: Rebuild precompute cache

After all tests pass:

1. Delete corrupt cache and stale scaler
2. Rebuild with 4 workers
3. Verify cache integrity (shape, nonzero%, class distribution)

**Expected**:
- `train_hrv_scaled.npy` shape: (157337, 5, 7) — note 5 time steps, not 10
- Nonzero percentage significantly higher than before (was ~0%)
- Labels have both AF (1) and NSR (0) classes

---

## Files to modify (7 files + PLAN.md)

| File | Changes |
|------|---------|
| `PLAN.md` | Replace with this plan (Step 0) |
| `src/features/scaler.py` | Replace `transform()` row-level NaN mask with per-column loop |
| `src/training/precompute_cache.py` | Remove pre-filter mask, derive HRV_STEPS from config |
| `src/training/train.py` | Remove pre-filter mask, update HRV_STEPS, update comment |
| `src/features/pipeline.py` | Change `subwindow_sec` default to `None`, fix config read logic |
| `config.yaml` | `subwindow_sec: 60`, `hrv_seq_len: 5` |
| `src/features/hrv_freq.py` | `MIN_SAMPLES = 32` |
| `tests/test_preprocessing.py` | Add 8 new test functions |

## Status

| Step | Status      | Completed At |
|------|-------------|--------------|
| 0    | Done        | 2026-03-13   |
| 1    | Done        | 2026-03-13   |
| 2    | Done        | 2026-03-13   |
| 3    | Done        | 2026-03-13   |
| 4    | Done        | 2026-03-13   |
| 5    | Killed      | 2026-03-13   |
| 6    | Done        | 2026-03-13   |
| 7    | Done        | 2026-03-13   |
| 8    | Done        | 2026-03-13   |
| 9    | Done        | 2026-03-13   |
| 10   | Done        | 2026-03-13   |
| 11   | Done        | 2026-03-13   |
| 12   | In progress |              |
| 13   | Pending     |              |
| 14   | Done        | 2026-03-14   |

---

## Round 3: Training Loop Quality Fixes (Post-Audit, 2026-03-14)

Senior ML engineer audit found 7 issues in the training loop and model code.
**None affect the precompute cache.** All changes are in training code, model defaults, and docstrings.

### Step 14: Training loop + code quality fixes

| # | Fix | File | Why |
|---|-----|------|-----|
| 1 | LR scheduler (ReduceLROnPlateau, patience=5, factor=0.5) | `train.py` | Without decay, training plateaus ~epoch 20 and wastes remaining epochs |
| 2 | Early stopping (patience=15) | `train.py` | Stops training if no AUROC improvement for 15 epochs — saves hours |
| 3 | Gradient clipping (max_norm=1.0) | `train.py` | Prevents weight explosion from noisy batches |
| 4 | Memory-mapped cache loading (mmap_mode="r") | `train.py` | Loads data on demand instead of 1.5+ GB into RAM at once |
| 5 | Fix `t.size` → `t.numel()` in on-the-fly path | `train.py` | `t.size` is a method on torch tensors, not an int — would crash |
| 6 | TransformerConfig default seq_len 10→5 | `transformer.py` | Matches actual config (was overridden at runtime but default was wrong) |
| 7 | Fix stale docstrings (seq_len=10→seq_len) | `ensemble.py`, `rnn.py`, `transformer.py` | Documentation accuracy |

All 33 tests passing after fixes.

**Step 5 note:** Cache rebuild killed at 67% — deep audit found 5 more issues (Steps 6-11) that would invalidate the cache. Fix all before single rebuild.

---

## Round 2: Pre-Cache Data Pipeline Fixes

### Issues Found (2026-03-13 audit)

| # | Issue | Severity | Effect |
|---|-------|----------|--------|
| 1 | AF=250Hz, NSR=128Hz — CNN learns sampling rate as label proxy | CRITICAL | Model cheats via padding pattern; HRV precision differs by class |
| 2 | `np.any()` artifact rejection kills valid AF windows | CRITICAL | ~89% of HRV rows are NaN — rejects the positive class |
| 3 | SampEn/DFA always NaN (need ~200/100 RR, only ~70 per 60s sub) | HIGH | 0% valid nonlinear features |
| 4 | NaN→0 masks missing data (no logging) | HIGH | Model can't distinguish zero from missing |
| 5 | Non-overlapping windows waste 50% of small dataset | MAJOR | Only 41 records, every window matters |

---

### Step 6: config.yaml — add new parameters

Add 3 new keys:
```yaml
data:
  target_fs: 250              # uniform sampling rate for all records
  stride_sec: 5               # 50% overlap for training windows only

artifact:
  rr_fraction_threshold: 0.30 # reject only if >30% of beats are outliers
```

---

### Step 7: Resample to uniform 250 Hz — `src/data/dataset_parsers.py`

**Why:** AF (afdb) = 250 Hz, NSR (nsrdb) = 128 Hz. CNN sees 2500 vs 1280 samples for 10s — trivial shortcut to classify by padding. Also biases HRV peak detection precision.

**Changes:**
- Add `from scipy.signal import resample`
- New helper `_resample_to_target_fs(signal, fs_orig, fs_target)` using FFT-based resampling
- In `parse_all()`: read `target_fs` from config, resample all records that don't match

---

### Step 8: Artifact correction + fraction-based rejection — `src/features/artifact_scrubber.py`

**Why:** `np.any(dev_pct > 60)` rejects window if ANY single beat deviates >60%. AF has irregular RR by definition. Aim 2 says "artifact detection and **correction**" not rejection.

**Changes:**
- New function `correct_rr_intervals(rr, rr_deviation_percent, config_path)`:
  - Identifies outlier beats (>rr_deviation_percent from median)
  - Replaces via `np.interp` from neighboring valid beats
  - Returns `(corrected_rr, fraction_corrected)`
- Modify `should_reject_window()`:
  - Add `rr_fraction_threshold` param (from config if None)
  - Replace `np.any()` with: call `correct_rr_intervals`, reject only if `fraction > rr_fraction_threshold`

---

### Step 9: Pipeline — full-window nonlinear + corrected RR — `src/features/pipeline.py`

**Why:** SampEn needs ~200 RR, DFA needs ~100. 60s subwindow gives ~70 (always NaN). Full 5-min gives ~350 — enough for both.

**Changes to `waveform_to_hrv_sequence()`:**
- Import `correct_rr_intervals` from artifact_scrubber
- **Before loop:** Get full 5-min RR, correct outliers, compute `d_nl_full = compute_hrv_nonlinear(full_rr_corrected)` once
- **Per subwindow:** Get subwindow RR → `should_reject_window()` (fraction-based) → `correct_rr_intervals()` → compute time/freq HRV on corrected RR → use `d_nl_full` for nonlinear columns

---

### Step 10: Overlapping stride + NaN logging

**`src/data/dataloaders.py`:** Read `stride_sec` from config, pass to train PhysioDataset only (val/test stay non-overlapping for unbiased eval).

**`src/training/precompute_cache.py` line 161:** Add NaN % logging before `nan_to_num`. Target <10% after fixes (was ~89%).

**`src/training/build_model.py` line 50:** Default seq_len fallback 10→5.

---

### Step 11: Tests

**Update `tests/test_models.py`:** seq_len 10→5 in 4 places (lines 30, 37, 39, 47).

**Update `tests/test_preprocessing.py`:** Add `rr_fraction_threshold=0.30` to existing artifact tests (lines 94, 102).

**New tests in `tests/test_preprocessing.py`:**

| Test | Verifies |
|------|----------|
| `test_correct_rr_interpolates_outliers` | correct_rr_intervals replaces outlier beats, returns fraction |
| `test_correct_rr_all_outliers` | All beats outlier → median fill, fraction=1.0 |
| `test_artifact_fraction_accepts_few_outliers` | 2/50 outliers (4%) → NOT rejected |
| `test_artifact_fraction_rejects_many_outliers` | 20/50 outliers (40%) → rejected |
| `test_pipeline_nonlinear_valid` | 300s synthetic ECG → SampEn/DFA not all-NaN |
| `test_pipeline_corrected_rr_af_pattern` | AF-like irregular RR → valid time-domain features |

**New tests in `tests/test_dataloader.py`:**

| Test | Verifies |
|------|----------|
| `test_resampling_uniform_fs` | parse_all with target_fs=250 → all records fs=250.0 |
| `test_stride_increases_samples` | stride_sec=5 > stride_sec=10 sample count |

---

### Step 12: Smoke test + cache rebuild

1. `pytest tests/ -v` — all pass (25 existing + 8 new)
2. Smoke test on 1 real AF + 1 real NSR record: verify fs=250, HRV NaN count low
3. Rebuild cache: `.venv/Scripts/python -m src.training.precompute_cache --config config.yaml --workers 4`
4. Verify NaN % in logs (target <10%)

---

### Step 13: Train

After cache verified:
- `.venv/Scripts/python -m src.training.train --use-cache`
- Target: AUROC ≥ 0.75 (NFR-2.1)

---

## Resume From Here

**Next action:** Start Step 6 (config.yaml updates), then proceed sequentially through Step 13.
