"""Tests for tinnitus pipeline schema and config."""

import numpy as np
import pytest
import yaml

from src.data.tinnitus_parsers import (
    TinnitusRecordDict,
    _REQUIRED_KEYS,
    validate_tinnitus_record,
)


def _make_valid_record(**overrides) -> TinnitusRecordDict:
    record: TinnitusRecordDict = {
        "subject_id": "test_001",
        "ppg_signal": np.zeros(1250, dtype=np.float64),
        "ppg_fs": 125.0,
        "resp_signal": None,
        "resp_fs": None,
        "resp_channel": None,
        "eda_signal": None,
        "eda_fs": None,
        "label": 0,
        "session_id": None,
        "condition": None,
    }
    record.update(overrides)
    return record


class TestTinnitusRecordDict:
    def test_valid_minimal_record(self):
        record = _make_valid_record()
        assert validate_tinnitus_record(record)

    def test_valid_full_record(self):
        record = _make_valid_record(
            resp_signal=np.zeros(1250, dtype=np.float64),
            resp_fs=125.0,
            resp_channel="impedance",
            eda_signal=np.zeros(40, dtype=np.float64),
            eda_fs=4.0,
            label=1,
            session_id="ses01",
            condition="trifold",
        )
        assert validate_tinnitus_record(record)

    def test_missing_key_fails(self):
        record = _make_valid_record()
        del record["ppg_signal"]
        assert not validate_tinnitus_record(record)

    def test_all_required_keys_present(self):
        record = _make_valid_record()
        assert _REQUIRED_KEYS.issubset(set(record.keys()))

    def test_ppg_signal_must_be_ndarray(self):
        record = _make_valid_record(ppg_signal=[0.0, 1.0, 2.0])
        assert not validate_tinnitus_record(record)

    def test_ppg_signal_must_be_1d(self):
        record = _make_valid_record(ppg_signal=np.zeros((10, 2), dtype=np.float64))
        assert not validate_tinnitus_record(record)

    def test_ppg_fs_must_be_positive(self):
        record = _make_valid_record(ppg_fs=0.0)
        assert not validate_tinnitus_record(record)


class TestTinnitusConfig:
    CONFIG_PATH = "config_tinnitus.yaml"

    def test_config_loads(self):
        with open(self.CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        assert cfg is not None

    def test_required_sections_present(self):
        with open(self.CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        for section in ["data", "ppg_filter", "eda", "phase_model", "phase_training", "closed_loop"]:
            assert section in cfg, f"Missing section: {section}"

    def test_ppg_target_fs(self):
        with open(self.CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        assert cfg["data"]["ppg_target_fs"] == 125

    def test_eda_target_fs(self):
        with open(self.CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        assert cfg["data"]["eda_target_fs"] == 4

    def test_phase_model_input_samples(self):
        with open(self.CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        # 2s × 125Hz = 250 samples (not 500 like ECG @ 250Hz)
        assert cfg["phase_model"]["input_samples"] == 250

    def test_phase_model_n_tasks(self):
        with open(self.CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        assert cfg["phase_model"]["n_tasks"] == 2  # diastole + exhalation

    def test_wesad_label_map_keys(self):
        with open(self.CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        label_map = cfg["wesad"]["label_map"]
        # Labels 0-4 must all be present
        for k in [0, 1, 2, 3, 4]:
            assert k in label_map, f"Missing WESAD label {k}"
