"""
Experiment runner: band × subject × CSP components × k_lmnn grid.

Executes the same experimental grid as the original notebooks via the
metric_bci pipeline and writes one CSV per frequency band for downstream analysis.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import DATA_DIR, FREQUENCY_BANDS
from .datasets import load_dataset
from .pipeline import CSPLMNNPipeline


# Subject IDs excluded from dataset1 in the original protocol
DATASET1_EXCLUDE_SUBJECTS = (88, 92, 100)

# CSV column order matching the original notebook output (metrics grouped by type)
CSV_COLUMN_ORDER = [
    "subject",
    "band",
    "csp_nc",
    "k_lmnn",
    "separability",
    "acc_train_ada",
    "acc_test_ada",
    "acc_train_svm",
    "acc_test_svm",
    "acc_train_lda",
    "acc_test_lda",
    "f1_train_ada",
    "f1_test_ada",
    "f1_train_svm",
    "f1_test_svm",
    "f1_train_lda",
    "f1_test_lda",
    "mcc_train_ada",
    "mcc_test_ada",
    "mcc_train_svm",
    "mcc_test_svm",
    "mcc_train_lda",
    "mcc_test_lda",
    "best_params_ada",
    "best_params_svm",
    "best_params_lda",
]


def get_subject_list(
    dataset_name: str,
    max_subjects: Optional[int] = None,
) -> List[int]:
    """
    Return the subject ID list for the given dataset as in the original protocols.

    Parameters
    ----------
    dataset_name : str
        One of ``"dataset1"``, ``"dataset2"``, ``"dataset3"``, ``"dataset4"``.
    max_subjects : int, optional
        If provided, the returned list is truncated to the first max_subjects (e.g. for quick runs).

    Returns
    -------
    list of int
        Subject identifiers.
    """
    if dataset_name == "dataset1":
        out = [s for s in range(1, 106) if s not in DATASET1_EXCLUDE_SUBJECTS]
    elif dataset_name == "dataset2":
        out = list(range(1, 5))
    elif dataset_name == "dataset3":
        out = list(range(1, 10))
    elif dataset_name == "dataset4":
        out = np.concatenate([
            np.arange(7, 20),
            np.arange(21, 35),
            np.arange(36, 39),
            np.arange(41, 53),
            np.arange(54, 62),
            [63],
            np.arange(65, 68),
        ]).astype(int).tolist()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}")

    if max_subjects is not None:
        out = out[:max_subjects]
    return out


def _evaluate_to_row(
    subject: int,
    band_name: str,
    nc: int,
    k_lmnn: int,
    res: Dict[str, Any],
) -> Dict[str, Any]:
    """Map the pipeline evaluate() output to a single flat row matching the original notebook schema."""
    row: Dict[str, Any] = {
        "subject": subject,
        "band": band_name,
        "csp_nc": nc,
        "k_lmnn": k_lmnn,
        "separability": res["separability"],
    }
    for clf in ("ada", "svm", "lda"):
        m = res["classifiers"][clf]
        row[f"acc_train_{clf}"] = m["train_acc"]
        row[f"acc_test_{clf}"] = m["test_acc"]
        row[f"f1_train_{clf}"] = m["train_f1"]
        row[f"f1_test_{clf}"] = m["test_f1"]
        row[f"mcc_train_{clf}"] = m["train_mcc"]
        row[f"mcc_test_{clf}"] = m["test_mcc"]
        row[f"best_params_{clf}"] = m["best_params"]
    return row


def run_full_experiment(
    dataset_name: str,
    subjects: Optional[Sequence[int]] = None,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    nc_list: Sequence[int] = (4, 8, 12, 16),
    k_lmnn_range: Tuple[int, int] = (1, 17),
    output_dir: str = ".",
    data_dir: str = DATA_DIR,
    n_jobs: int = -1,
    max_subjects: Optional[int] = None,
) -> List[str]:
    """
    Run the full experimental grid (band × subject × CSP components × k_lmnn) and write one CSV per band.

    Parameters
    ----------
    dataset_name : str
        One of ``"dataset1"``, ``"dataset2"``, ``"dataset3"``, ``"dataset4"``.
    subjects : sequence of int, optional
        Subject IDs to include. If None, taken from get_subject_list(dataset_name, max_subjects).
    bands : dict, optional
        Mapping band_name -> (low_hz, high_hz). If None, uses config.FREQUENCY_BANDS.
    nc_list : sequence of int
        Numbers of CSP components to sweep (e.g. (4, 8, 12, 16)).
    k_lmnn_range : tuple of (int, int)
        (start, stop) for k_lmnn (e.g. (1, 17) for k in 1..16).
    output_dir : str
        Directory in which to write the CSV files.
    data_dir : str
        Root path for datasets, passed to load_dataset.
    n_jobs : int
        Number of parallel jobs for GridSearchCV (-1 for all available cores).
    max_subjects : int, optional
        If set and subjects is None, only the first max_subjects from get_subject_list are used.

    Returns
    -------
    list of str
        Paths to the written CSV files.
    """
    if bands is None:
        bands = dict(FREQUENCY_BANDS)
    if subjects is None:
        subjects = get_subject_list(dataset_name, max_subjects=max_subjects)
    else:
        subjects = list(subjects)

    os.makedirs(output_dir, exist_ok=True)
    saved: List[str] = []

    for band_name, freq_range in bands.items():
        results: List[Dict[str, Any]] = []
        first_result_printed = False
        print(f"[{dataset_name}] band={band_name} freq_range={freq_range}")

        for subject in subjects:
            try:
                X_train, y_train, X_test, y_test = load_dataset(
                    dataset_name,
                    subject,
                    freq_range,
                    data_dir=data_dir,
                )
            except Exception as e:
                print(f"  subject {subject} load failed: {e}")
                continue

            print(f"  subject {subject} (train {X_train.shape[0]}, test {X_test.shape[0]})")

            for nc in nc_list:
                for k_lmnn in range(k_lmnn_range[0], k_lmnn_range[1]):
                    model = CSPLMNNPipeline(
                        m_filters=nc,
                        use_lmnn=True,
                        k_neighbors=k_lmnn,
                        n_jobs=n_jobs,
                    )
                    res = model.evaluate(X_train, y_train, X_test, y_test)
                    row = _evaluate_to_row(subject, band_name, nc, k_lmnn, res)
                    results.append(row)

                    # Print first result as example (acc + cpu times)
                    if not first_result_printed:
                        ada = res["classifiers"]["ada"]
                        print(
                            f"  Example: acc_train={ada['train_acc']:.1f}%, acc_test={ada['test_acc']:.1f}%, "
                            f"train_cpu_time={ada['train_cpu_time']:.1f}s, test_cpu_time={ada['test_cpu_time']:.3f}s"
                        )
                        first_result_printed = True

        if results:
            df = pd.DataFrame(results, columns=CSV_COLUMN_ORDER)
            filename = os.path.join(
                output_dir,
                f"csp_lmnn_results_{band_name}_{dataset_name}.csv",
            )
            df.to_csv(filename, index=False)
            saved.append(filename)
            print(f"  Saved {filename} ({len(results)} rows)")

    return saved
