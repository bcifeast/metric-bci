"""
Dataset loading utilities for motor-imagery EEG experiments.

This module provides a single entry point for loading the motor-imagery datasets.
Three datasets (dataset1, dataset2, dataset3) are documented in the README;
dataset4 is supported in code for internal use. Dataset-specific paths,
preprocessing, and train/test splitting are encapsulated here.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from mne import Epochs, events_from_annotations, pick_types, set_log_level
from mne.datasets import eegbci
from mne.filter import filter_data
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf

from .config import DATA_DIR

# Suppress verbose MNE logging in this module
set_log_level("ERROR")


ArrayLike = np.ndarray


def load_dataset(
    dataset_name: str,
    subject: int,
    freq_range: Tuple[float, float],
    data_dir: str = DATA_DIR,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """
    Load one subject's data from one of the supported motor-imagery datasets.

    Parameters
    ----------
    dataset_name : str
        One of ``"dataset1"``, ``"dataset2"``, ``"dataset3"``; ``"dataset4"`` is also supported in code.
    subject : int
        Subject index as in the original experiment protocols.
    freq_range : tuple of (float, float)
        Band-pass filter range in Hz, (low, high).
    data_dir : str, optional
        Root path for dataset directories; defaults to ``config.DATA_DIR``.

    Returns
    -------
    X_train, y_train, X_test, y_test : ndarray
        Band-pass filtered train and test arrays, suitable for CSP and downstream pipelines.
    """

    if dataset_name == "dataset1":
        # PhysioNet EEG Motor Movement/Imagery (MNE fetches on demand)
        runs = [5, 6, 9, 10, 13, 14]  # motor imagery: hands vs feet
        raw_files = [
            read_raw_edf(f, preload=True, verbose=False)
            for f in eegbci.load_data(subject, runs)
        ]
        raw = concatenate_raws(raw_files)

        picks = pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
        )

        # Band-pass filter on continuous data prior to epoching
        raw.filter(freq_range[0], freq_range[1], method="iir", picks=picks, verbose=False)
        events, _ = events_from_annotations(
            raw, event_id=dict(T1=2, T2=3), verbose=False
        )

        event_id = dict(hands=2, feet=3)
        tmin, tmax = 1.0, 2.0
        epochs = Epochs(
            raw,
            events,
            event_id,
            tmin,
            tmax,
            proj=True,
            picks=picks,
            baseline=None,
            preload=True,
            verbose=False,
        )

        X = 1e6 * epochs.get_data()
        y = epochs.events[:, -1] - 2

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )
        return X_train, y_train, X_test, y_test

    # Sampling rate for datasets 2–4 (dataset2 overrides to 100 Hz)
    sfreq = 250.0

    if dataset_name == "dataset2":
        sfreq = 100.0
        base_path = os.path.join(data_dir, "1_Dataset2inNotebook", "100Hz", f"s{subject}")
        train_mat = f"{base_path}_train_data_100.mat"
        train_lab = f"{base_path}_train_label_100.mat"
        test_mat = f"{base_path}_test_data_100.mat"
        test_lab = f"{base_path}_test_label_100.mat"

        if not os.path.exists(train_mat):
            raise FileNotFoundError(f"Dataset2 files not found at base path: {base_path}")

        X_train = scipy.io.loadmat(train_mat)["dataset"]
        y_train = scipy.io.loadmat(train_lab)["labels"].flatten()
        X_test = scipy.io.loadmat(test_mat)["dataset"]
        y_test = scipy.io.loadmat(test_lab)["labels"].flatten()

    elif dataset_name == "dataset3":
        base_path = os.path.join(data_dir, "2a_Dataset3inNotebook", f"s{subject}")
        train_mat = f"{base_path}_train_data_dataset2a.mat"
        if not os.path.exists(train_mat):
            raise FileNotFoundError(f"Dataset3 files not found at base path: {base_path}")
        X_train = scipy.io.loadmat(train_mat)["dataset"]
        y_train = scipy.io.loadmat(f"{base_path}_train_label_dataset2a.mat")[
            "labels"
        ].flatten()
        X_test = scipy.io.loadmat(f"{base_path}_test_data_dataset2a.mat")["dataset"]
        y_test = scipy.io.loadmat(f"{base_path}_test_label_dataset2a.mat")[
            "labels"
        ].flatten()

    elif dataset_name == "dataset4":
        base_path = os.path.join(data_dir, "4_Dataset5inNotebook", f"s{subject}")
        data = scipy.io.loadmat(f"{base_path}_data.mat")["dataset"]
        labels = scipy.io.loadmat(f"{base_path}_label.mat")["labels"].flatten()
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.33, random_state=42
        )

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name!r}")

    # Band-pass filter to the requested frequency range (datasets 2–4)
    X_train_filt = filter_data(
        X_train,
        sfreq=sfreq,
        l_freq=freq_range[0],
        h_freq=freq_range[1],
        method="iir",
    )
    X_test_filt = filter_data(
        X_test,
        sfreq=sfreq,
        l_freq=freq_range[0],
        h_freq=freq_range[1],
        method="iir",
    )

    return X_train_filt, y_train, X_test_filt, y_test

