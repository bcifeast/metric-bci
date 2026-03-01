"""
Global configuration for the metric_bci package.

This module centralizes dataset root paths and standard EEG frequency band
definitions used across the pipeline and experiment runners.
"""

import os

# Package root (parent of this package directory)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Root directory under which all EEG dataset subdirectories reside
DATA_DIR = os.path.join(BASE_DIR, "Datasets")

# Canonical EEG frequency bands (Hz): [low, high]
FREQUENCY_BANDS = {
    "delta": [1, 4],
    "theta": [4, 8],
    "alpha": [8, 12],
    "beta": [12, 30],
    "gamma": [30, 50],
    "broadBand": [8, 30],
}

