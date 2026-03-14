"""
Worker for parallel precompute: process one chunk of samples.
Separate module so ProcessPoolExecutor workers on Windows import this (not __main__).
"""

import os

import numpy as np

from src.features.pipeline import waveform_10s_denoised, waveform_to_hrv_sequence


def process_chunk(args):
    """
    Process one chunk of samples. args is (chunk_data, config_path).
    chunk_data is a list of (short_np, long_np, fs, label).
    Returns a list of (short_d, hrv, label) in the same order.
    """
    chunk_data, config_path = args
    os.environ["OMP_NUM_THREADS"] = "1"
    results = []
    for (short_np, long_np, fs, label) in chunk_data:
        hrv = waveform_to_hrv_sequence(long_np, fs, config_path=config_path)
        short_d = waveform_10s_denoised(short_np, fs, config_path=config_path)
        results.append((np.asarray(short_d, dtype=np.float32), hrv, float(label)))
    return results
