# RS-Personalized AI-Driven Adaptive Stimulation (Aim 2)

Phase 1 – Algorithmic & control software for auricular VNS, interfacing with Spark Biomedical Sparrow Link.

## Setup

1. **Create and activate virtual environment** (from project root):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install PyTorch** (choose one):

   - **With CUDA** (NVIDIA GPU; use Python 3.10–3.12 for prebuilt CUDA wheels):
     ```powershell
     py -3.11 -m venv .venv
     .\.venv\Scripts\Activate.ps1
     pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
     pip install -r requirements.txt
     ```
     If you get "No matching distribution", your Python may be too new (e.g. 3.14); use `py -3.11` or install Python 3.12 and use that for the venv. Verify with `python -c "import torch; print(torch.cuda.is_available())"`.
   - **CPU only** (default from PyPI):
     ```powershell
     pip install torch torchaudio
     ```

3. **Install remaining dependencies**:

   ```powershell
   pip install -r requirements.txt
   ```

4. **Verify**: `python -c "import torch, mne, neurokit2, pylsl, yaml, sklearn, pywt; print('OK')"`

## Data (Step 2)

- **Download:** Run `python -m src.data.download_physionet` (or `python src/data/download_physionet.py`) to fetch MIT-BIH AF and Normal Sinus Rhythm into `data/raw/afdb` and `data/raw/nsrdb`. Requires network and `wfdb`.
- **MIMIC-III:** Place exported files under `data/raw/mimic3/`. Expected layout: CSV(s) with columns `subject_id`, `time`, `ecg` (or `ppg`), and optional `label`; or EDF files (one per subject). If the directory is missing or empty, the pipeline skips MIMIC-III and continues with PhysioNet only.
- **Split and loaders:** After first parse, create the 70/15/15 split by subject ID (e.g. call `create_split(parsed_records, config['data']['split_path'])`), then use `get_dataloaders(create_split_if_missing=True)` to build train/val/test DataLoaders with 10 s and 5 min windows.

## Training (Step 3)

1. **Download data** (one time): `python -m src.data.download_physionet`
2. **Optional – fast training (one-time precompute):** Run `python -m src.training.precompute_cache --workers 0` once to build the cache (HRV + denoised 10s) under `models/artifacts/cache/`. Use `--workers 0` to use all CPU cores (much faster than the default single-threaded ~1.5 samples/sec). Then run `python -m src.training.train --use-cache` for fast epochs (target ≤1 h per epoch, GPU-bound only). Without `--use-cache`, training uses the on-the-fly path (slower, recomputes HRV and denoising each epoch).
3. **Train:** `python -m src.training.train` (optionally `--max-epochs 1` for a quick smoke run). A progress bar shows scaler fit and each epoch’s batches; logs and checkpoint as above.
4. **Evaluate:** `python -m src.training.evaluate --split test` (optionally `--no-plots`). Prints AUROC, F1, sensitivity, specificity; saves ROC and confusion matrix to `models/artifacts/` if plots enabled.
5. **Visualize training curves:** `python -m src.training.visualize` (reads `training_log.csv`, saves `models/artifacts/training_curves.png`).
6. **Optional – split folders:** `python -m src.data.materialize_split` creates `data/splits/train/`, `data/splits/val/`, `data/splits/test/` with symlinks for visibility (requires `models/artifacts/split.json` from a prior training run).

## Project layout

- `src/data` – ingestion, parsers, splitter, DataLoaders; LSL streamer (deferred to Live Integration)  
- `src/features` – preprocessing, HRV, wavelet, scaler  
- `src/models` – CNN, RNN, Transformer, ensemble  
- `src/hardware` – Sparrow API, watchdog, state machine  
- `src/training` – train and evaluate scripts  
- `config.yaml` – hyperparameters, paths, LSL rate, watchdog limits  

## Docs

- [Phase 1 Implementation Plan](Phase1-Implementation-Plan-Aim2.md)  
- [SRS & Testing Plan](SRS-Testing-Plan-Aim2.md)  

Update `config.yaml` (watchdog, artifact thresholds) per Sparrow/FDA limits and validation.
