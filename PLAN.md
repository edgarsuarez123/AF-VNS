# AF VNS — Round 6: MIMIC Mixed Training + Attention Masking + Data Augmentation

## Context

Round 5 MIMIC-III AUROC = 0.6022 despite expanding from 41 to 5,915 training records.
Root cause: **domain shift** — all training data is research/portable-grade; MIMIC is ICU bedside.
Adding more PhysioNet data doesn't bridge the gap. The model needs ICU signals during training.

Additionally: 36% of training samples have 4/5 zero-padded HRV timesteps (GRU/Transformer
learn padding shortcuts), and no augmentation exists for real-world noise conditions.

---

## Status

| Step | Description | Status | Completed At |
|------|-------------|--------|--------------|
| 23a | Snapshot 60 MIMIC holdout IDs to JSON | Done | 2026-03-15 |
| 23b | Download script: skip-if-exists + env var creds | Done | 2026-03-15 |
| 23c | Download 400 MIMIC records (manual) | Skipped | — needs creds; current 60 used as holdout only |
| 23d | Add mimic3_holdout config key | Done | 2026-03-15 |
| 23e | Include MIMIC train records in data pipeline | Done | 2026-03-15 |
| 23f | Default evaluate to holdout-only | Done | 2026-03-15 |
| 23g | Tests for MIMIC holdout logic | Done | 2026-03-15 |
| 24a | Save HRV lengths in cache | Done | 2026-03-15 |
| 24b | PrecomputedDataset returns lengths | Done | 2026-03-15 |
| 24c | GRU: pack_padded_sequence | Done | 2026-03-15 |
| 24d | Transformer: src_key_padding_mask | Done | 2026-03-15 |
| 24e | Ensemble passes lengths through | Done | 2026-03-15 |
| 24f | Training loop passes lengths | Done | 2026-03-15 |
| 24g | Evaluate passes lengths | Done | 2026-03-15 |
| 24h | Tests for attention masking | Done | 2026-03-15 |
| 25a | Create augmentation module | Done | 2026-03-15 |
| 25b | Add augmentation config section | Done | 2026-03-15 |
| 25c | Integrate augmentation into PrecomputedDataset | Done | 2026-03-15 |
| 25d | Tests for augmentation | Done | 2026-03-15 |
| 26a | Delete stale cache + rebuild | Done | 2026-03-15 |
| 26b | Train (masking+aug only, no MIMIC train data) | Done | 2026-03-15 |
| 26c | Evaluate on MIMIC holdout | Done | 2026-03-15 |
| 26d | Download 400 MIMIC records | Done | 2026-03-15 |
| 26e | Expand holdout 60→160, rebuild cache | Done | 2026-03-15 |
| 26f | Retrain + re-evaluate | Done | 2026-03-15 |

---

## Round 7 Results (MIMIC mixed training, 160-record holdout)

  Train: val_auroc=0.9582, early stop epoch 30
  MIMIC holdout (160 records, 80 AF + 80 NSR):
    AUROC=0.6747, F1=0.6296, Sens=0.6375, Spec=0.6125
  vs Round 5: AUROC +0.072, Specificity +0.146 (major win — model no longer AF-biased)
  NFR-3.1 target (AUROC ≥ 0.75): not yet met — gap = 0.0253

## Round 8 Results (label smoothing + per-source MIMIC stride)

  Train: val_auroc=0.9602, early stop epoch 42
  MIMIC holdout (160 records, 80 AF + 80 NSR):
    AUROC=0.6777, F1=0.6554, Sens=0.7250, Spec=0.5125
  vs Round 7: AUROC +0.003, F1 +0.026, Sens +0.088, Spec -0.100
  Analysis: label smoothing shifted model toward predicting AF — sensitivity up but
    specificity down. AUROC nearly flat. NFR-3.1 gap = 0.0223.

## Resume From Here

**Current state (2026-03-16):**
- Round 8 complete. 80/80 tests passing.
- 298 MIMIC training records. 160-record holdout locked.
- AUROC trajectory: 0.6022 → 0.6333 → 0.6747 → 0.6777
- Next: Step 28 (download 300 more MIMIC records, needs PhysioNet creds) — biggest remaining lever.

---

# Round 8: Label Smoothing + More MIMIC Data + MIMIC Stride Reduction

## Context

AUROC gap to NFR-3.1 target (0.75): **0.0253**. Three levers identified:
1. Model outputs prob=0.000 or prob=1.000 on nearly everything — overconfident, hurts calibration and AUROC on OOD data. Label smoothing fixes this.
2. Only 298 MIMIC training records — more ICU data is the biggest long-term lever.
3. MIMIC records are 500-900s but stride=300s gives only 1-2 windows each. Halving stride doubles MIMIC training windows for free.
4. HRV NaN rate ~42% — largely structural (Challenge 2017 short records). Attention masking already handles zeros; further reduction is lower priority.

---

## Steps

| Step | Description | Status | Completed At |
|------|-------------|--------|--------------|
| 27 | Label smoothing (ε=0.1) in training loss | Done | 2026-03-16 |
| 28 | Download 300 more MIMIC records (seed=44) | Not Started | |
| 29 | Per-source stride: MIMIC stride=150s, others=300s | Done | 2026-03-16 |
| 30 | Rebuild cache + retrain + eval | Done | 2026-03-16 |

---

## Step 27: Label Smoothing

**File:** `src/training/train.py`

Replace hard 0/1 labels with smoothed targets before loss:
```python
# label_smoothing: replace 0→epsilon, 1→1-epsilon
epsilon = config.get("training", {}).get("label_smoothing", 0.0)
if epsilon > 0:
    labels_t = labels_t * (1 - epsilon) + epsilon * 0.5
```

Add to `config.yaml`:
```yaml
training:
  label_smoothing: 0.1
```

Apply in both cache training loop and non-cache loop. No cache rebuild needed — labels are smoothed at training time, not stored.

**Expected impact:** +0.02-0.04 AUROC on OOD data. Forces model away from prob=0.000/1.000 extremes.

---

## Step 28: Download 300 More MIMIC Records

Run with seed=44, skip-if-exists handles duplicates:
```bash
export PHYSIONET_USER=edgarsuarez123
export PHYSIONET_PASS=<password>
python -m src.data.download_mimic3_waveforms \
  --n-af 300 --n-control 300 --seed 44 --out-dir data/raw/mimic3
```

After download: update holdout (add ~75 AF + 75 NSR to holdout from new batch → ~310 total holdout), rest goes to training (~450 MIMIC training records total).

---

## Step 29: Per-Source Stride Reduction for MIMIC

**File:** `src/training/precompute_cache.py`

Add logic: if record subject_id matches MIMIC pattern (starts with `p0`), use `mimic_stride_sec` from config instead of global `stride_sec`.

**File:** `config.yaml`:
```yaml
data:
  stride_sec: 300        # default for afdb/nsrdb/ltafdb/c17
  mimic_stride_sec: 150  # halved stride for MIMIC → 2-4x more windows per record
```

**Expected impact:** ~600 → ~1200+ MIMIC training windows from existing records.

---

## Step 30: Rebuild Cache + Retrain + Eval

```bash
rm models/artifacts/cache/*.npy models/artifacts/scaler.pkl models/artifacts/split.json
python -m src.training.precompute_cache --config config.yaml --workers 6
python -m src.training.train --use-cache
python -m src.training.evaluate --mimic3
```

Target: AUROC ≥ 0.75 on holdout.

---

## Resume From Here

**Current state (2026-03-16):**
- Steps 27 (label smoothing) and 29 (per-source MIMIC stride) implemented. 80/80 tests passing.
- 298 MIMIC training records included. 160-record holdout locked.
- AUROC trajectory: 0.6022 → 0.6333 → 0.6747
- Next: Step 28 (download 300 more MIMIC records) then Step 30 (cache rebuild + retrain + eval).
