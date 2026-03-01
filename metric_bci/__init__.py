"""
metric_bci: CSP and metric learning (LMNN) pipelines for motor-imagery EEG.

This package provides config, dataset loaders, a CSP+LMNN classification pipeline,
and experiment runners for motor-imagery EEG. Three datasets are documented in
the README (dataset1, dataset2, dataset3); the API is designed for reuse and
extension to further BCI experiments.
"""

from . import config
from .datasets import load_dataset
from .pipeline import CSPLMNNPipeline
from .experiments import run_full_experiment, get_subject_list

__all__ = [
    "config",
    "load_dataset",
    "CSPLMNNPipeline",
    "run_full_experiment",
    "get_subject_list",
]

