"""
Full experiment grid (README Section 5): dataset1, alpha band, first 2 subjects.
Run from repository root: python run_full_experiment.py
"""
from metric_bci.experiments import run_full_experiment, get_subject_list

# Example: dataset1, alpha band only, first 2 subjects (for a quick test)
subjects = get_subject_list("dataset1", max_subjects=2)
saved_files = run_full_experiment(
    dataset_name="dataset1",
    subjects=subjects,
    bands={"alpha": (8, 12)},
    nc_list=(4, 8),
    k_lmnn_range=(1, 5),
    output_dir=".",
)
print("Saved files:", saved_files)
