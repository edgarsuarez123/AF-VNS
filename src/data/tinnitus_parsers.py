"""
Tinnitus AVNS — dataset parsers.

TinnitusRecordDict schema:
    subject_id    str
    ppg_signal    np.ndarray          1D float64 (PPG / BVP waveform)
    ppg_fs        float               PPG sampling rate (Hz)
    resp_signal   np.ndarray | None   1D float64 respiratory reference
    resp_fs       float | None        respiratory sampling rate (Hz)
    resp_channel  str | None          "impedance" / "chest_belt" / "ppg_derived"
    eda_signal    np.ndarray | None   1D float64 EDA waveform (µS)
    eda_fs        float | None        EDA sampling rate (Hz)
    label         int                 0=baseline/control, 1=stress/event
    session_id    str | None
    condition     str | None          "trifold" / "cardiac_only" / "open_loop"

Nothing in the AF or stroke pipelines is imported or modified here.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

TinnitusRecordDict = Dict[str, Any]

# Required keys for schema validation
_REQUIRED_KEYS = {
    "subject_id",
    "ppg_signal",
    "ppg_fs",
    "resp_signal",
    "resp_fs",
    "resp_channel",
    "eda_signal",
    "eda_fs",
    "label",
    "session_id",
    "condition",
}


def validate_tinnitus_record(record: TinnitusRecordDict) -> bool:
    """Return True if record contains all required keys with correct types."""
    missing = _REQUIRED_KEYS - set(record.keys())
    if missing:
        logger.warning("TinnitusRecordDict missing keys: %s", missing)
        return False
    if not isinstance(record["ppg_signal"], np.ndarray):
        logger.warning("ppg_signal must be np.ndarray")
        return False
    if record["ppg_signal"].ndim != 1:
        logger.warning("ppg_signal must be 1D")
        return False
    if record["ppg_fs"] <= 0:
        logger.warning("ppg_fs must be > 0")
        return False
    return True


def _load_tinnitus_config(config_path: str) -> dict:
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
