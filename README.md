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

## Project layout

- `src/data` – ingestion, parsers, DataLoaders, LSL streamer  
- `src/features` – preprocessing, HRV, wavelet, scaler  
- `src/models` – CNN, RNN, Transformer, ensemble  
- `src/hardware` – Sparrow API, watchdog, state machine  
- `src/training` – train and evaluate scripts  
- `config.yaml` – hyperparameters, paths, LSL rate, watchdog limits  

## Docs

- [Phase 1 Implementation Plan](Phase1-Implementation-Plan-Aim2.md)  
- [SRS & Testing Plan](SRS-Testing-Plan-Aim2.md)  

Update `config.yaml` (watchdog, artifact thresholds) per Sparrow/FDA limits and validation.
