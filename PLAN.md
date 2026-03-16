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

## Resume From Here

**Current state (2026-03-15):**
- Round 7 complete. 75/75 tests passing.
- 298 MIMIC training records included. 160-record holdout locked.
- AUROC trajectory: 0.6022 → 0.6333 → 0.6747

**Ideas for Round 8 (gap to 0.75):**
1. Download more MIMIC records (target 400 AF + 400 NSR total training)
2. Reduce stride_sec for MIMIC records to 150s (2x windows per record)
3. Label smoothing to reduce overconfident binary predictions
4. Longer training with lower LR floor
