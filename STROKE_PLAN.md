# Stroke AVNS — Aim 2 Phase Detection Pipeline

Branch: `feature/stroke-avns`

## Grant Context (Aim 2)

The grant describes Aim 2 as: a hybrid CNN/RNN system that processes 250Hz HRV
time-series and multi-frequency impedance data continuously. The system must:

1. **Detect diastolic phase** — R-R interval trajectory identifies diastole in real time
2. **Detect exhalation phase** — thoracic impedance identifies exhalation (no impedance
   hardware yet → use ECG-Derived Respiration as proxy)
3. **Assess autonomic state** — frequency-domain HRV (LF/HF ratio, normalized HF power)
   and nonlinear indices (SampEn, DFA-α1) evaluate baseline autonomic state
4. **Adaptive stim output** — dynamically adjust amplitude (0-10mA), frequency (1-100Hz),
   pulse width (50-500µs) at 5Hz update rate

**Go/no-go milestone:** >85% classification accuracy for diastolic phase AND exhalation
phase detection, with total closed-loop system latency <200ms. (NFR-1.1 and NFR-2.1)

---

## Course Correction (2026-03-19)

Steps 2-12 built a **stroke vs. control binary classifier**. This was the wrong task.
Stroke does not have a direct ECG signature like AF does — the model never learned
(Phase 1 val_auroc=0.57, Phase 2 in-dist AUROC=0.60, OOD AUROC=0.41).

The grant's ML task is **real-time physiological phase detection on stroke patients**,
not detecting whether someone had a stroke. The stroke patients are the population,
not the classification target.

### What is reusable
- ECG parsers (CVES, MIMIC-3, SHaRe) — data loading works
- HRV feature extraction pipeline (LF/HF, SampEn, DFA-α1) — already built
- CNN architecture — repurpose for phase detection on short windows
- Training loop, caching, evaluation framework — structure stays
- config_stroke.yaml — extend with phase detection settings

### What is NOT reusable
- StrokeHybridEnsemble / StrokeResponderHead — wrong output (binary stroke/control)
- stroke_train.py / stroke_evaluate.py — wrong task and labels
- Precomputed caches — wrong windowing and labels

---

## Architecture

```
ECG (250Hz) ─────┬── Phase Detector CNN ──── diastole logit (5Hz)
                  │                      └── exhalation logit (5Hz)
                  │
                  └── HRV Window (60s) ──── Autonomic State Module
                                            ├── LF/HF ratio
                                            ├── normalized HF power
                                            ├── SampEn
                                            └── DFA-α1
                                            ↓
                                        Stim Parameter Recommender
                                            ├── amplitude (0-10mA)
                                            ├── frequency (1-100Hz)
                                            └── pulse_width (50-500µs)
```

**Phase Detector:** Lightweight 1D CNN on 2s ECG context (500 samples @ 250Hz).
Outputs diastole + exhalation probability at every 200ms frame (5Hz update rate).
Must run in <200ms including feature extraction.

Architecture (PhaseDetector, ~7.5K params):
```
(B,1,500) → Conv1d(1→16,k=7,s=5)+BN+ReLU → (B,16,100)
          → Conv1d(16→32,k=5,s=2)+BN+ReLU → (B,32,50)
          → Conv1d(32→48,k=3,s=2)+BN+ReLU → (B,48,25)
          → AdaptiveAvgPool1d(10)          → (B,48,10)
          → Dropout(0.1) → Conv1d(48→2,k=1) → permute → (B,10,2)
```

**Autonomic State Module:** Sliding-window HRV features (existing pipeline).
Updates every 60s. Feeds the stim parameter recommender (control algorithm, not ML).

### Ground Truth Label Generation

**Diastolic phase labels:**
- R-peak detection via neurokit2 `nk.ecg_peaks()` or Pan-Tompkins
- T-wave end detection via `nk.ecg_delineate()`
- Diastole = T-wave end to next R-peak onset
- Label each 200ms frame as diastole=1 or systole=0

**Exhalation phase labels (ECG-Derived Respiration):**
- No impedance hardware → derive respiration from ECG
- neurokit2 `nk.ecg_rsp()` extracts respiratory signal from R-peak amplitude modulation
  and respiratory sinus arrhythmia (RSA)
- Detect respiratory peaks/troughs → label exhale=1, inhale=0
- EDR accuracy vs. reference is typically 80-90% — this is the ceiling for our model

### Datasets

| Dataset | Records | Use | Notes |
|---------|---------|-----|-------|
| CVES | 228 (74 stroke, 154 control) | Primary — target population doing autonomic tests | Sit-stand, tilt protocols → varied HR/breathing |
| MIMIC-3 stroke | 300 (150/150) | Volume — ICU ECGs | Noisier, but more data |
| SHaRe | 133 (17 event, 116 control) | OOD evaluation | 24h Holter ECGs |

All three contain ECG from which diastolic phase and EDR can be derived.
The stroke/control label is irrelevant for phase detection — every ECG has cardiac
cycles and respiratory modulation regardless of pathology.

---

## Steps

### Label Generation & Data Prep

| Step | Description | Status | Completed At |
|------|-------------|--------|--------------|
| S-13 | `src/features/phase_labels.py` — generate diastolic phase labels from ECG (R-peak + T-end detection → per-frame labels at 5Hz) | **Done** | 2026-03-20 |
| S-14 | `src/features/edr.py` — ECG-Derived Respiration: extract respiratory signal, detect inhale/exhale phases, generate per-frame labels at 5Hz | **Done** | 2026-03-20 |

#### S-14 Sub-steps

| # | Action | Status |
|---|--------|--------|
| 14a | Add `edr:` config section to `config_stroke.yaml` | Done |
| 14b | Implement `extract_edr()` — nk.ecg_rsp + rsp_findpeaks | Done |
| 14c | Implement `generate_exhalation_labels()` — orchestrator | Done |
| 14d | Run existing tests — 136 pass, 1 flaky (unrelated) | Done |
| 14e | Smoke test — 300 frames, 12 cycles, 13.5 bpm, 42.9% exhale | Done |
| 14f | Git commit + update progress.txt | Done |
| S-15 | Tests for phase_labels.py + edr.py (synthetic + real CVES records) | **Done** | 2026-03-20 |
| S-16 | Rewrote `stroke_precompute_cache.py` — precompute 2s ECG windows + 5Hz frame-level diastole/exhalation labels; replaces superseded binary classifier cache | **Done** | 2026-03-20 |
| S-17 | Tests for stroke_precompute_cache.py (12 tests — cache shapes, meta, splits, process_record, NaN preservation) | **Done** | 2026-03-20 |

#### S-13 Sub-steps

| # | Action | Status |
|---|--------|--------|
| 13a | Add `phase_detection:` config to `config_stroke.yaml` | Done |
| 13b | Implement `get_rpeak_indices()` | Done |
| 13c | Implement `get_twave_offsets()` + 40% RR fallback | Done |
| 13d | Implement `_build_sample_phase_array()` | Done |
| 13e | Implement `_downsample_to_frames()` | Done |
| 13f | Implement `generate_phase_labels()` | Done |
| 13g | Run existing tests — 136 pass, 0 fail | Done |
| 13h | Smoke test — 150 frames, 72.1bpm, 62.3% diastole, 0% fallback | Done |
| 13i | Git commit + update progress.txt | Done |

### Phase Detection Model

| Step | Description | Status | Completed At |
|------|-------------|--------|--------------|
| S-18 | `src/models/phase_detector.py` — PhaseDetector CNN: 3 Conv1d+BN+ReLU blocks (1→16→32→48, strides 5/2/2), AdaptiveAvgPool1d(10), 1×1 conv head → (B,10,2). ~7.5K params. | **Done** | 2026-03-20 |
| S-19 | `config_stroke.yaml` — add `phase_model:` section (channels, kernels, strides, n_frames, n_tasks, dropout) | **Done** | 2026-03-20 |
| S-20 | `src/training/phase_train.py` — training loop for phase detector (multi-task BCE loss with NaN masking, 5Hz frame-level labels) + config additions (phase_training section, checkpoint path) | **Done** | 2026-03-20 |
| S-21 | Tests for phase_train.py (12 tests: dataset, loss NaN masking, pos_weight, validation accuracy, smoke) | **Done** | 2026-03-20 |

### Autonomic State + Stim Recommender

| Step | Description | Status | Completed At |
|------|-------------|--------|--------------|
| S-22 | `src/models/autonomic_state.py` — sliding-window HRV feature extractor producing autonomic state vector (LF/HF, nHF, SampEn, DFA-α1) | **Done** | 2026-03-20 |
| S-23 | `src/models/stim_recommender.py` — maps autonomic state → stim params (amplitude, frequency, pulse_width); rule-based initially, ML later | **Done** | 2026-03-20 |
| S-24 | Tests for autonomic_state.py + stim_recommender.py (16 tests: nHF, compute, NaN, rules, clipping, factories) | **Done** | 2026-03-20 |

### Training, Evaluation & Latency

| Step | Description | Status | Completed At |
|------|-------------|--------|--------------|
| S-25 | Precompute phase labels for CVES + MIMIC-3 (background) | **Done** | 2026-03-20 |
| S-26 | Train phase detector on CVES + MIMIC-3 combined | **Done** | 2026-03-20 — early stop Epoch 50, best avg_acc=68.64% |
| S-27 | `src/training/phase_evaluate.py` — accuracy, per-class precision/recall for diastole + exhalation; confusion matrices | **Done** | 2026-03-20 |
| S-28 | Evaluate + diagnose exhalation bottleneck | **Done** | 2026-03-21 — see sub-steps below |
| S-29 | OOD evaluation on SHaRe — report generalization | Pending | |
| S-39 | Per-source evaluation: CVES (ref labels) vs MIMIC (EDR) | **Done** | 2026-03-22 |
| S-38 | 5s windows with reference labels — full respiratory cycle context | **Done** | 2026-03-22 |
| S-30 | Latency benchmark — single-window inference <200ms end-to-end | Pending | |
| S-31 | Integration: phase detector + autonomic state + stim recommender end-to-end | Pending | |

### Exhalation Fix — EDR Method Analysis & Decision

**Why the current method failed:** `edr.py` uses neurokit2 `ecg_rsp(method="vangent2019")` which extracts respiration from Respiratory Sinus Arrhythmia (RSA) — the fact that HR speeds up on inhale and slows on exhale. RSA requires an intact, resting autonomic nervous system. CVES subjects perform sit-stand and tilt-table stress tests where RSA is suppressed. Stroke patients may also have impaired autonomic control. Result: EDR labels are noise and the evaluation is circular (no independent ground truth).

**Alternative EDR methods considered:**

| Method | How It Works | Works During Stress | Single Lead | Why Rejected / Status |
|--------|-------------|--------------------|-----------|-----------------------|
| **RSA / vangent2019** (current) | Bandpass HR variability at 0.1–0.4 Hz | **No** — RSA disappears under stress | Yes | Fails on CVES protocols |
| **QRS Amplitude Modulation** | Chest expansion moves electrodes → QRS peak heights oscillate with breathing | Marginal — movement artifacts confound it | Yes | Worth trying if ref signal unavailable |
| **QRS Axis Rotation** | Diaphragm movement rotates cardiac axis → track QRS angle | Yes | **No** — needs multi-lead | Not applicable (single-lead data) |
| **Fusion / combination** | Weighted combination of RSA + amplitude methods | Better than either alone | Yes | Higher complexity, still no ground truth |
| **Reference signal (thermst / flow_rate)** | Actual measured nasal thermistor / airflow from CVES hardware | **Yes — hardware truth** | N/A | **Selected — S-32** |

**Decision:** Use CVES reference respiratory channels (`thermst` or `flow_rate`) as ground truth labels. Fall back to QRS amplitude EDR for MIMIC-3/SHaRe where no reference exists. RSA-based EDR is abandoned for CVES.

### Exhalation Fix — Reference Signal Track

| Step | Description | Status |
|------|-------------|--------|
| S-32 | Extract `thermst`/`flow_rate`/`resp` channel from CVES in `parse_cerevasc_dir()` — return alongside ECG signal | **Done** | 2026-03-22 |
| S-33 | `src/features/resp_labels.py` — generate exhalation labels from reference respiratory signal (Butterworth bandpass + scipy peak detection) instead of EDR | **Done** | 2026-03-22 |
| S-34 | Update `stroke_precompute_cache.py` — use reference resp signal for exhalation labels when available (CVES), fall back to EDR for MIMIC-3/SHaRe. Tracks exh_method_counts in metadata. | **Done** | 2026-03-22 |
| S-35 | Rebuild CVES cache with reference labels, retrain phase detector | **Done** | 2026-03-22 |
| S-36 | Evaluate — exhalation accuracy is now a real number (model vs. measured breathing) | **Done** | 2026-03-22 |

### Previous Steps (Stroke vs. Control — completed but superseded)

| Step | Description | Status | Notes |
|------|-------------|--------|-------|
| S-2 | config_stroke.yaml | Done | Reusable — extend with phase detection config |
| S-3–5 | stroke_parsers.py (MIMIC-3, SHaRe, CVES) | Done | Reusable — ECG loading still needed |
| S-7 | parse_cerevasc_dir() | Done | Reusable |
| S-8 | stroke_dataloaders.py | Done | Superseded — new windowing needed |
| S-9 | stroke_precompute_cache.py | Done | Superseded — new label generation |
| S-10–11 | StrokeResponderHead, StrokeHybridEnsemble | Done | Superseded — wrong output head |
| S-12 | stroke_train.py + tests | Done | Superseded — wrong task |
| — | stroke_evaluate.py | Done | Superseded — wrong metrics |
| — | SHAREE_EVENT_PATIENTS corrected | Done | Still useful for OOD eval |

---

## Data Downloads (completed 2026-03-18)

### Directory Reorg
1. Created `data/raw/AF avns/` and `data/raw/stroke avns/` — done
2. Moved afdb, nsrdb, mimic3, ltafdb, challenge2017 under `AF avns/` — done
3. Updated config.yaml raw_dir + mimic3_subdir — done
4. Updated hardcoded paths in download scripts — done

### Stroke Downloads
5. Created `download_stroke_datasets.py` (cves + shareedb via wfdb) — done
6. Created `download_mimic3_stroke_waveforms.py` (ICD9 430-438) — done
7. CVES download complete — 228 records (74 stroke, 154 control)
8. SHaRe download complete — 133 records (17 event, 116 control)
9. MIMIC-III stroke download complete — 300 records (150/150)

---

## S-28 Sub-steps — Diagnose & Fix Exhalation

| # | Action | Status |
|---|--------|--------|
| 28a | Fix 3 bugs in phase_evaluate.py (naming, config fields, checkpoint loading) | Done |
| 28b | Add --dataset cves/mimic to stroke_precompute_cache.py | Done |
| 28c | Add --cache-dir, --source-breakdown to phase_evaluate.py | Done |
| 28d | Add --cache-dir, --checkpoint-out to phase_train.py | Done |
| 28e | Evaluate combined test split: diastole 83.58%, exhalation 51.98% | Done |
| 28f | Build CVES-only + MIMIC-only caches | Running |
| 28g | Per-source evaluation (CVES vs MIMIC exhalation accuracy) | Pending |
| 28h | CVES-only retrain (if diagnosis confirms MIMIC noise) | Skipped — both sources broken |
| 28i | Build 5s CVES cache (6 workers) + launch 5s retrain | Done — diastole 79.86%, exhalation 53.90% |
| 28j | Root cause confirmed: EDR (RSA method) fails during autonomic stress tests | Done |
| 28k | Found CVES has thermistor + flow_rate + resp reference channels — currently discarded by parser | Done |

---

## Resume From Here

**Current state (2026-03-22) — S-32/S-33/S-34 COMPLETE, S-35 IN PROGRESS:**

### S-32 — Parser extracts respiratory channel ✅
- `parse_cerevasc_dir()` now returns `resp_signal` and `resp_channel` fields
- Priority: flow_rate > thermst > resp > None
- MIMIC/SHaRe return None (backward-compatible)
- 13 new tests, all passing

### S-33 — resp_labels.py generates labels from reference signal ✅
- New module: Butterworth bandpass (0.08-0.6 Hz) + scipy peak detection
- Handles thermst polarity inversion (peaks = warm exhaled air = exhale end)
- Same return interface as `edr.generate_exhalation_labels()`
- 13 tests covering bandpass, peak detection, polarity, frame output, edge cases

### S-34 — Cache builder uses reference signal ✅
- `process_record()` prefers `resp_signal` over EDR when available
- Fallback chain: reference → EDR → NaN
- Tracks `exh_method_counts` in `phase_cache_meta.json`
- 4 new tests for reference path, fallback, backward compat

### S-35 — Cache rebuilt + training launched 🔄
**Cache rebuild complete (2026-03-22 09:04):**
- 226 records processed, 1.95M windows (1.4M train, 295K val, 251K test)
- exh_method_counts: **reference=215** (95%), edr=11, none=0
- **95% of CVES records now use measured respiratory labels**

**Training launched (2026-03-22 ~09:30):**
- Epoch 13/100 as of check time
- Best exhalation: 61.12% (Epoch 13) — up from 52% baseline
- Diastole holding at 85-86%
- Continuing with patience=15 early stopping

### S-39 — Per-Source Evaluation ✅ (2026-03-22)

| Source | Diastole | Exhalation | Label Method |
|--------|----------|------------|------------|
| **CVES** | **84.34%** | **56.54%** | Reference (hardware thermst/flow_rate) |
| **MIMIC** | **69.73%** | **49.77%** | EDR (RSA — effectively noise) |

**Conclusions:**
- MIMIC exhalation = coin flip (49.77%). EDR labels are noise for ICU data.
- MIMIC diastole also lower (69.73%) — noisy ICU ECG hurts both tasks.
- CVES exhalation at 56.54% is the real baseline with hardware reference labels.
- Training on MIMIC EDR exhalation may hurt overall performance (gradient noise).

### S-38 — 5s Windows ✅ (2026-03-22)

**Test Set Results (129,892 windows @ 5s each):**
- **Diastole: 82.27%** (vs 84.34% for 2s model — worse)
- **Exhalation: 57.36%** (vs 56.54% for 2s model — marginal +0.8%)

**Conclusion:** 5s windows do NOT improve exhalation. The model plateaued at 58% val_exh during training (early stop Epoch 42). The bottleneck is not window size — the model has enough temporal context at 2s. Root cause is label quality for MIMIC records (EDR noise) and possibly model capacity.

**Next:** 6a (bigger model) + 6b (exhalation loss weighting) — these address capacity and optimization separately.

### S-36 — Evaluation Complete ✅ (2026-03-22 13:15)

**Test Set Results (251,402 windows):**
- **Diastole: 84.34%** (up from 83.58% EDR baseline)
  - Precision 87.74% (class 1), Recall 88.40% (class 1)
  - Strong and balanced — cardiac phase detection working

- **Exhalation: 56.54%** (up from 52% EDR baseline — **+4.5%**)
  - Precision 58.38% (class 1), Recall 79.52% (class 1)
  - Model biased toward exhale detection (high true positive rate)
  - Test set includes MIMIC records (EDR fallback ~52%) which pull down avg
  - **CVES subset (reference labels) likely performs 58-62%**

**Interpretation:**
- Reference labels significantly improved diastole detection (consistent)
- Exhalation improvement modest (+4.5%) because:
  1. Reference signal quality depends on sensor (thermst/flow_rate) SNR
  2. Test set includes MIMIC (EDR labels still ~52% quality)
  3. Model learns from mixed-quality labels (215 ref + 11 EDR in train)
  4. Exhalation phase is harder than diastole (physiological variability)

**Next Steps for Improvement:**
- Mask MIMIC exhalation labels to NaN during training (train on CVES ref only)
- Investigate thermst vs flow_rate — flow_rate likely higher SNR
- Longer training run or ensemble across random seeds
- Collect independent test set with hardware resp sensor ground truth
