# RS-Personalized AI-Driven Adaptive Stimulation (Aim 2)

Phase 1 – Algorithmic & control software for auricular VNS, interfacing with Spark Biomedical Sparrow Link.

## Setup

1. **Create and activate virtual environment** (from project root):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install PyTorch** (choose one):

   - **With CUDA** (match your driver: cu118, cu121, or cu124):
     ```powershell
     pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```
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
