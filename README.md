# metric-bci
Application of metric learning on CSP feature space. Common Spatial Patterns (CSP) is a widely used technique for EEG feature extraction, yet it often struggles with noise sensitivity and limited locality preservation. This work introduces a lightweight pipeline that enhances CSP representations using Large Margin Nearest Neighbors (LMNN), a metric-learning approach that preserves local neighborhood structure in a low-dimensional space. CSP features are first extracted and then mapped via LMNN to improve discriminability and robustness.

Experiments on four MI-EEG datasets show consistent test-accuracy improvements for most participants, with the largest gains observed on Dataset 1 (≈10% increase in mean α-band accuracy across AdaBoost, SVM, and LDA). The improvements are particularly pronounced for low-accuracy subjects (baseline < 60%), as confirmed by paired Wilcoxon signed-rank tests (p < 10⁻⁶).

Overall, the CSP+LMNN pipeline provides a computationally efficient and interpretable solution, making it a practical candidate for real-world BCI applications.
## 📚 Citation

If you use this work, please cite:

```bibtex
@article{balli2026,
  title={Improving Motor Imagery based BCI through Metric Learning},
  author={Balli, T., Yetkin, E. Fatih},
  journal={Biomedical Signal Processing and Control},
  year={to appeared}
}
```

![CSP+LMNN pipeline](figs/graphical_abstract.png)

The algorithmic framework of the code can be seen as follows: the main theme is the application of metric learning approach (LMNN) on CSP feature space in EEG. 

<p align="center">
  <img src="figs/fig1.png" width="400">
</p>

## 🔧 How to use?

This repository provides a modular Python package for applying metric learning (LMNN) on CSP-based EEG features for motor imagery (MI) BCI. The code is organized as a single package, `metric_bci`, with a clear separation of configuration, data loading, pipeline, and experiment orchestration. It supports four motor-imagery datasets through a unified API and reproduces the experimental grid (band × subject × CSP components × k_lmnn) from the original notebooks.

### 1. Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/bcifeast/metric-bci.git
cd metric-bci
```

We recommend using **conda** to manage the environment and avoid package conflicts:

```bash
conda create -n metric_learning_env python=3.11 -y
conda activate metric_learning_env
```

Install dependencies with **pip** inside the environment for exact version matching as tested by the authors:

```bash
pip install -r requirements.txt
```

**Alternatively, using Python’s venv:** If you prefer a virtualenv instead of conda:

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Linux / macOS
# or:  .venv\Scripts\activate   on Windows
pip install -r requirements.txt
```

**Optional:** To use Jupyter Notebooks with this environment, register it as a kernel:

```bash
python -m ipykernel install --user --name metric_learning_env --display-name "Python (Metric Learning)"
```

Run all commands (including Python and Jupyter) from the **repository root** (the directory that contains the `metric_bci` folder) so that `import metric_bci` works correctly.

### 2. Required libraries

The project depends on specific library versions for reproducibility and to avoid known conflicts (e.g. with numpy≥2.0 and scikit-learn≥1.6). The following are listed in `requirements.txt`:

| Package       | Version  |
|---------------|----------|
| numpy         | 1.26.4   |
| scipy         | 1.11.4   |
| scikit-learn  | 1.5.2    |
| mne           | 1.7.1    |
| pandas        | 2.2.2    |
| matplotlib    | 3.8.4    |
| seaborn       | 0.13.2   |
| openTSNE      | 1.0.2    |
| pyriemann     | 0.6      |
| metric-learn  | 0.7.0    |
| joblib        | 1.4.2    |
| tqdm          | 4.66.5   |
| ipykernel     | 6.29.5   |

Do **not** install **moabb**, **numpy≥2**, or **scikit-learn≥1.6** in this environment; they can cause dependency or compatibility issues.

### 3. Datasets and citation

The pipeline uses four motor-imagery EEG datasets. **Dataset 1** is obtained automatically by the code; **datasets 2–4** must be obtained by the user from the original sources. Processed or raw dataset files are not distributed in this repository and should be excluded from version control (e.g. via `.gitignore` for `.mat`, `.edf`, `.csv`). When using any of these data in publications, the corresponding citations below are required.

| Code name   | Description / source | How to obtain | Citation / license |
|------------|----------------------|---------------|--------------------|
| **dataset1** | PhysioNet EEG Motor Movement/Imagery | Fetched automatically by MNE when `load_dataset("dataset1", ...)` is called. | Cite: Schalk et al., BCI2000, IEEE TBME 51(6):1034–1043, 2004; and PhysioNet (Goldberger et al., Circulation 101(23), 2000). Data: [Open Data Commons Attribution License v1.0](https://physionet.org/content/eegmmidb/1.0.0/). |
| **dataset2** | BCI Competition IV, Data set 1 (Berlin BCI) | Download from the official source. Listed under “External links” at [BNCI Horizon 2020](http://bnci-horizon-2020.eu/database/data-sets); see also [BCI Competition IV](https://www.bbci.de/competition/iv/). Place files under the path given by `metric_bci.config.DATA_DIR` as expected by `load_dataset` (see `datasets.py`). | License: CC BY-ND 4.0. You must cite the data providers and the competition; see the dataset description on the competition page. Redistribution of processed versions may not be permitted; use original downloads only. |
| **dataset3** | BCI Competition IV, Data set 2a (Graz, 4-class MI) | Same as above; Data set 2a is listed at the top of the [BNCI data sets page](http://bnci-horizon-2020.eu/database/data-sets). | License: CC BY-ND 4.0. Citation and attribution as required by the competition and the dataset providers. |
| **dataset4** | Referred to in code as `4_Dataset5inNotebook` | Source and exact dataset identity to be confirmed. Once confirmed, download and folder layout will be documented here. | To be updated. |

If you use **dataset2** or **dataset3**, do not redistribute the data or processed derivatives in this repo; link only to the official sources and comply with CC BY-ND 4.0 and the citation requirements stated there.

### 4. Quick start example

Minimal example: load one subject from **dataset1** (PhysioNet EEG Motor Movement/Imagery, downloaded automatically via MNE), run the CSP+LMNN pipeline, and report test accuracy.

```python
from metric_bci.pipeline import CSPLMNNPipeline
from metric_bci.datasets import load_dataset
from metric_bci.config import FREQUENCY_BANDS

# Load one subject, alpha band (8–12 Hz). Dataset1 is fetched via MNE if needed.
dataset_name = "dataset1"
subject = 1
freq_range = FREQUENCY_BANDS["alpha"]  # (8, 12)

X_train, y_train, X_test, y_test = load_dataset(
    dataset_name=dataset_name,
    subject=subject,
    freq_range=freq_range,
)

# Build pipeline: m_filters = CSP components, k_neighbors = LMNN k
model = CSPLMNNPipeline(
    m_filters=4,
    use_lmnn=True,
    k_neighbors=3,
    classifier="svm",
)

model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")
```

For **datasets 2–4**, place the data under the path given by `metric_bci.config.DATA_DIR` (default: `Datasets/` in the repo root) with the folder layout expected by `load_dataset`; see the docstring of `load_dataset` in `metric_bci.datasets` for details.

### 5. Running the full experiment grid

To reproduce the paper’s experimental grid (all bands × subjects × CSP component counts × k_lmnn) and save one CSV per band:

```python
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
# Output: CSV files, e.g. csp_lmnn_results_alpha_dataset1.csv
```

Omit `subjects` and `bands` to run over all subjects and all bands defined in `config.FREQUENCY_BANDS`. Use `max_subjects` in `get_subject_list` for shorter runs.

### 6. Using different classifiers

The pipeline supports the built-in classifier names `"svm"`, `"lda"`, and `"ada"`, or any scikit-learn estimator that implements `fit` and `predict`:

```python
from sklearn.ensemble import AdaBoostClassifier
from metric_bci.pipeline import CSPLMNNPipeline

model = CSPLMNNPipeline(
    m_filters=6,
    k_neighbors=5,
    classifier=AdaBoostClassifier(n_estimators=200),
)
model.fit(X_train, y_train)
```

### 7. Parallel processing

Cross-validated hyperparameter search inside the pipeline can use multiple cores via the `n_jobs` parameter:

```python
model = CSPLMNNPipeline(
    m_filters=6,
    k_neighbors=5,
    n_jobs=-1,  # use all available cores
)
model.fit(X_train, y_train)
```

### 8. Project structure

The repository is organized as follows:

```
metric_bci/
├── __init__.py
├── config.py       # DATA_DIR, FREQUENCY_BANDS
├── datasets.py     # load_dataset(dataset_name, subject, freq_range, data_dir)
├── pipeline.py     # CSPLMNNPipeline (CSP + optional LMNN + classifier)
└── experiments.py  # run_full_experiment, get_subject_list
```

- **config.py**: Root path for datasets and standard EEG frequency band definitions.
- **datasets.py**: Single entry point for loading the four motor-imagery datasets; returns `(X_train, y_train, X_test, y_test)` with band-pass filtering applied.
- **pipeline.py**: `CSPLMNNPipeline` implements CSP, optional LMNN, and a configurable classifier with a scikit-learn-style `fit` / `predict` / `score` API; the `evaluate` method runs the full grid over AdaBoost, SVM, and LDA with cross-validated hyperparameter search and returns metrics plus separability.
- **experiments.py**: `run_full_experiment` runs the band × subject × nc × k_lmnn grid and writes one CSV per band in the format used in the original notebooks; `get_subject_list` returns the subject IDs per dataset.

CSP is provided by **MNE** (`mne.decoding.CSP`), and LMNN by **metric-learn**; both are used inside `CSPLMNNPipeline`. The package does not expose separate `metric_bci.features` or `metric_bci.metric_learning` modules.

### 9. Reproducibility and extensibility

The package is designed so that the same experimental protocol (datasets, bands, subjects, CSP/LMNN parameters, classifiers) can be rerun via `run_full_experiment` and the provided notebooks. Configuration is centralized in `config.py`; new bands or data paths can be added there. The pipeline accepts any classifier compatible with scikit-learn’s estimator interface, and the four-dataset loader can be extended in `datasets.py` for additional datasets following the same `(X_train, y_train, X_test, y_test)` contract.

---

## About

Application of metric learning on CSP feature space for motor-imagery BCI.
