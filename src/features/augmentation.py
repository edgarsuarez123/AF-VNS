"""
On-the-fly waveform augmentation for training noise robustness.

Simulates real-world conditions for earpiece deployment:
  1. Gaussian noise — ambient electrical noise
  2. Amplitude scaling — electrode impedance variation
  3. Baseline wander — movement/breathing artifacts
  4. Signal dropout — loose electrode contact
  5. Powerline interference — 50/60 Hz electrical environment

Pure numpy, sub-millisecond per call. Returns augmented copy (same shape/dtype).
"""

import numpy as np


def augment_waveform(
    waveform: np.ndarray,
    fs: float,
    config: dict = None,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Apply random augmentations to a 1D waveform.

    Parameters
    ----------
    waveform : 1D float array
    fs : sampling frequency in Hz
    config : augmentation config dict (from config.yaml augmentation section)
    rng : numpy random Generator (for reproducibility)

    Returns
    -------
    augmented : same shape/dtype as input
    """
    if rng is None:
        rng = np.random.default_rng()
    if config is None:
        config = {}

    out = waveform.astype(np.float32, copy=True)
    n = len(out)
    if n == 0:
        return out

    # 1. Gaussian noise (always applied)
    snr_range = config.get("gaussian_snr_range", [20, 40])
    snr_db = rng.uniform(snr_range[0], snr_range[1])
    sig_power = np.mean(out ** 2)
    if sig_power > 0:
        noise_power = sig_power / (10 ** (snr_db / 10))
        out += rng.normal(0, np.sqrt(noise_power), n).astype(np.float32)

    # 2. Amplitude scaling (always applied)
    scale_range = config.get("amplitude_scale_range", [0.8, 1.2])
    scale = rng.uniform(scale_range[0], scale_range[1])
    out *= scale

    # 3. Baseline wander (always applied)
    freq_range = config.get("baseline_freq_range", [0.1, 0.5])
    amp_range = config.get("baseline_amp_range", [0.0, 0.1])
    wander_freq = rng.uniform(freq_range[0], freq_range[1])
    wander_amp = rng.uniform(amp_range[0], amp_range[1])
    t = np.arange(n, dtype=np.float32) / fs
    phase = rng.uniform(0, 2 * np.pi)
    out += (wander_amp * np.sin(2 * np.pi * wander_freq * t + phase)).astype(np.float32)

    # 4. Signal dropout (probabilistic)
    dropout_prob = config.get("dropout_prob", 0.2)
    if rng.random() < dropout_prob:
        dur_range = config.get("dropout_duration_range", [0.05, 0.3])
        dropout_dur = rng.uniform(dur_range[0], dur_range[1])
        dropout_samples = int(dropout_dur * fs)
        if dropout_samples > 0 and dropout_samples < n:
            start = rng.integers(0, max(1, n - dropout_samples))
            out[start: start + dropout_samples] = 0.0

    # 5. Powerline interference (probabilistic)
    powerline_prob = config.get("powerline_prob", 0.3)
    if rng.random() < powerline_prob:
        amp_range_pl = config.get("powerline_amp_range", [0.0, 0.05])
        pl_amp = rng.uniform(amp_range_pl[0], amp_range_pl[1])
        pl_freq = rng.choice([50.0, 60.0])
        pl_phase = rng.uniform(0, 2 * np.pi)
        out += (pl_amp * np.sin(2 * np.pi * pl_freq * t + pl_phase)).astype(np.float32)

    return out
