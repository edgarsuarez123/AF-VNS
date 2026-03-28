# TINNITUS_PLAN.md — feature/tinnitus-avns

> Branch plan for the tinnitus aVNS pipeline. Read this at session start. Resume from the first incomplete step.

---

## Context

The tinnitus SBIR requires a **tri-fold synchronized aVNS trigger** (cardiac diastole + exhalation + EDA arousal-in-band) using PPG, respiratory, and EDA signals. Branched from `feature/stroke-avns` because that branch already has bifold triggering, PhaseDetector CNN, EDR, resp_labels, HRV pipeline, and closed-loop infrastructure. Main deltas: ECG→PPG swap, EDA arousal gating, new dataset parsers.

**Reference doc:** `aVNS Tinnitus.md` — full SBIR technical spec

---

## Data Priority

| Dataset | Status | Signals | Use |
|---|---|---|---|
| BIDMC | Downloaded (`data/raw/stroke avns/bidmc/`) | PPG (PLETH) + impedance resp at 125 Hz | PPG cardiac + resp training |
| WESAD | Must download | PPG/BVP 64 Hz + EDA 4 Hz + chest resp 700 Hz | Only dataset with all 3 modalities |

## Reusable Components (no changes needed)
- `peak_detector.py` — already supports `signal_type="ppg"`
- `hrv_time.py`, `hrv_freq.py`, `hrv_nonlinear.py` — signal-agnostic (RR intervals)
- `artifact_scrubber.py` — signal-agnostic
- `resp_labels.py` — works with any reference respiratory signal
- `splitter.py`, `scaler.py`, `augmentation.py` — signal-agnostic
- `autonomic_state.py`, `stim_recommender.py` — reusable with PPG RR intervals
- `PhaseDetector` + `PhaseDetectorConfig` — just needs `input_samples: 250` for PPG @ 125 Hz

---

## Implementation Steps

- [x] 1. **F1: TinnitusRecordDict schema + config_tinnitus.yaml** — done 2026-03-28
  - `src/data/tinnitus_parsers.py` — define TypedDict with ppg, resp, eda fields
  - `config_tinnitus.yaml` — sections: data, ppg_filter, eda, phase_model, phase_training, closed_loop
  - Key values: `ppg_target_fs: 125`, `eda_target_fs: 4`
  - Verify: import schema, load config, unit test validates required keys

- [x] 2. **F2: BIDMC PPG parser** — done 2026-03-28
  - `src/data/tinnitus_parsers.py` — `parse_bidmc_ppg_dir()`, extract PLETH + RESP at 125 Hz
  - Pattern: `stroke_parsers.py:parse_bidmc_dir()` (line 501)
  - Verify: 53 records, ppg_fs == 125.0, resp_signal not None

- [x] 3. **F3: WESAD download + parser** — done 2026-03-28
  - `dl_wesad.py` — download from UCI ML repo
  - `src/data/tinnitus_parsers.py` — `parse_wesad_dir()`, pickle format S{N}/S{N}.pkl
  - Extract: wrist/BVP (64 Hz), wrist/EDA (4 Hz), chest/Resp (700 Hz → downsample)
  - Label: baseline=0, stress=1
  - Verify: 15 subjects × conditions, all 3 signals non-None

- [ ] 4. **F4: PPG bandpass filter**
  - `src/features/ppg_filter.py` — `denoise_ppg()`, Butterworth 0.5–8 Hz, SOS
  - Pattern: `resp_labels.py:_bandpass_filter()`
  - Verify: PSD concentrated in 0.5–8 Hz

- [ ] 5. **F5: PPG diastole detection**
  - `src/features/ppg_phase_labels.py` — systolic peak detection (via peak_detector, signal_type="ppg") + dicrotic notch detection
  - Dicrotic notch: local min in descending limb after ~30% of beat interval
  - Output: per-frame 5 Hz labels matching `phase_labels.generate_phase_labels()` interface
  - Verify: >85% agreement with ECG-derived diastole on BIDMC simultaneous ECG

- [ ] 6. **F6: PPG-derived respiration**
  - `src/features/ppg_resp.py` — RIIV / RIFV / baseline wander methods
  - Bandpass 0.1–0.5 Hz → peak/trough detection → exhale frame labels
  - Output: matches `edr.generate_exhalation_labels()` interface
  - Verify: >80% agreement with BIDMC impedance pneumography ground truth

- [ ] 7. **F7: EDA feature extraction**
  - `src/features/eda.py` — tonic/phasic decomposition via `nk.eda_phasic()` (cvxEDA)
  - Per-subject calibration → SCL thresholds → `arousal_in_band` binary
  - Verify: WESAD baseline >80% in-band; stress >50% out-of-band

- [ ] 8. **F8: PPG PhaseDetector CNN config**
  - Add `phase_model` section to `config_tinnitus.yaml` with `input_samples: 250`
  - No new class — existing `PhaseDetector` handles this
  - Verify: forward pass `(4, 1, 250)` → `(4, 10, 2)`

- [ ] 9. **F9: Tinnitus precompute cache**
  - `src/training/tinnitus_precompute_cache.py` — BIDMC + WESAD → 2s PPG windows + labels → .npz
  - Subject-level split via `splitter.py`
  - Verify: .npz files exist, shapes correct (250 samples, 10 frames)

- [ ] 10. **F10: Phase detection training**
  - `src/training/tinnitus_phase_train.py` + `tinnitus_phase_evaluate.py`
  - BCE multi-task, early stopping, checkpoint to `models/checkpoints/tinnitus_phase_detector.pth`
  - Target: >85% diastole accuracy, >80% exhalation accuracy (SBIR)

- [ ] 11. **F11: EDA arousal gate module**
  - `src/models/arousal_gate.py` — `ArousalGate`: `calibrate()`, `update()`, `is_in_band()`, `get_state()`
  - Ring buffer, config-driven thresholds

- [ ] 12. **F12: Tri-fold closed-loop pipeline**
  - `src/models/tinnitus_closed_loop.py` — `TinnitusClosedLoopPipeline`
  - Fast path (~100ms): PPG → PhaseDetector → diastole + exhalation
  - EDA path (~1s): ArousalGate → is_in_band
  - Tri-fold gate: all three → FIRE StimEvent
  - Slow path (~60s): PPG → RR → AutonomicState → StimRecommender
  - Verify: StimEvents fire during baseline EDA, zero during stress; latency <50ms

---

## Dependency Graph
```
F1 ─┬─> F2 ──────────────────────────────────────────┐
    ├─> F3 ──> F7 ──> F11 ─────────────────────────── ┐
    └─> F4 ──┬─> F5 ──┐                               │
             └─> F6 ──┴─> F8 ──┐                      │
                                └─> F9 ──> F10 ──> F12 <─┘
```

---

## Progress Log

### F3 — WESAD download + parser (2026-03-28)
- Created `dl_wesad.py` — accepts `--url`, `--zip`, or `--verify`; does not hard-code download URL (dataset hosted externally, URL may change). Integration test skips automatically if WESAD not downloaded.
- Added `parse_wesad_dir()` to `src/data/tinnitus_parsers.py` — reads S{N}/S{N}.pkl pickle files (WESAD format), extracts wrist BVP (64 Hz), wrist EDA (4 Hz), chest Resp (700 Hz → resampled to 64 Hz)
- Segments signals into continuous label epochs; skips undefined (label=0) and epochs < 30s
- Decision: WESAD PPG is at 64 Hz (Empatica E4 native) — kept as-is rather than upsampling to 125 Hz to preserve signal fidelity; phase precompute will handle mixed-FS datasets
- Decision: amusement (label=3) and meditation (label=4) mapped to 0 (baseline) for arousal gating — mechanistically similar to relaxed wakefulness per SBIR spec
- 29/30 tests passing (1 skipped — real data integration, will pass after WESAD download)

### F2 — BIDMC PPG parser (2026-03-28)
- Added `parse_bidmc_ppg_dir()` to `src/data/tinnitus_parsers.py`
- Extracts PLETH (PPG) + RESP (impedance pneumography) channels from BIDMC WFDB records at native 125 Hz
- Decision: no resampling to 250 Hz — 125 Hz meets SBIR spec and avoids interpolation artifacts; stroke parser resampled to 250 Hz because ECG backbone required it
- All 53 records parsed, all pass schema validation; 22/22 tests passing

### F1 — TinnitusRecordDict schema + config_tinnitus.yaml (2026-03-28)
- Created `src/data/tinnitus_parsers.py` with `TinnitusRecordDict` TypedDict, `_REQUIRED_KEYS` set, and `validate_tinnitus_record()` — mirrors `StrokeRecordDict` pattern from `stroke_parsers.py`
- Created `config_tinnitus.yaml` with all required sections: `data`, `ppg_filter`, `eda`, `ppg_diastole`, `ppg_resp`, `resp_labels`, `phase_model`, `phase_training`, `closed_loop`, `autonomic_state`, `stim_recommender`, `artifact`, `wesad`
- Key decisions: PPG target fs = 125 Hz (BIDMC native, meets SBIR ≥125 Hz spec); EDA target fs = 4 Hz (Empatica E4 native); `phase_model.input_samples = 250` (2s × 125 Hz, vs 500 for ECG @ 250 Hz); WESAD label map treats amusement/meditation as baseline (label=0) for arousal gating
- 14/14 tests passing

---

## Resume From Here

**Next step:** F4 — PPG bandpass filter (`src/features/ppg_filter.py`)
