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
| 23a | Snapshot 60 MIMIC holdout IDs to JSON | Not Started | |
| 23b | Download script: skip-if-exists + env var creds | Not Started | |
| 23c | Download 400 MIMIC records (manual) | Blocked (needs creds) | |
| 23d | Add mimic3_holdout config key | Not Started | |
| 23e | Include MIMIC train records in data pipeline | Not Started | |
| 23f | Default evaluate to holdout-only | Not Started | |
| 23g | Tests for MIMIC holdout logic | Not Started | |
| 24a | Save HRV lengths in cache | Not Started | |
| 24b | PrecomputedDataset returns lengths | Not Started | |
| 24c | GRU: pack_padded_sequence | Not Started | |
| 24d | Transformer: src_key_padding_mask | Not Started | |
| 24e | Ensemble passes lengths through | Not Started | |
| 24f | Training loop passes lengths | Not Started | |
| 24g | Evaluate passes lengths | Not Started | |
| 24h | Tests for attention masking | Not Started | |
| 25a | Create augmentation module | Not Started | |
| 25b | Add augmentation config section | Not Started | |
| 25c | Integrate augmentation into PrecomputedDataset | Not Started | |
| 25d | Tests for augmentation | Not Started | |
| 26a | Delete stale cache + rebuild | Not Started | |
| 26b | Train | Not Started | |
| 26c | Evaluate on MIMIC holdout | Not Started | |

---

## Resume From Here

**Current state (2026-03-15):**
- 60 MIMIC-III records exist in data/raw/mimic3/
- 61/61 tests passing from previous rounds
- Starting Round 6 implementation
