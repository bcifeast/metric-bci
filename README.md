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

This repository provides a modular and extensible Python framework for applying metric learning methods (LMNN) on CSP-based EEG feature representations for motor imagery (MI) based BCI systems.

The framework follows a modular, pipeline-based object-oriented design philosophy inspired by modern machine learning frameworks, enabling flexible integration into custom EEG processing pipelines and facilitating rapid experimentation and reproducible research.

### 1. Installation

First, clone the repository:
```bash
git clone https://github.com/bcifeast/metric-bci.git
cd metric-bci
```

Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Required Libraries

The framework relies on widely used scientific Python libraries. All dependencies are listed in `requirements.txt`:

* numpy, scipy
* scikit-learn
* metric-learn (LMNN backend)
* mne
* matplotlib, seaborn
* pandas
* jupyter

### 3. Quick Start Example

Below is a minimal example demonstrating the CSP + LMNN pipeline for MI-EEG classification.

```python
from metric_bci.pipeline import CSPLMNNPipeline
from metric_bci.datasets import load_bci_iv_2a

# Load dataset
X_train, y_train, X_test, y_test = load_bci_iv_2a()

# Build pipeline (m: CSP filters, k: LMNN neighbors)
model = CSPLMNNPipeline(
    m_filters=4,
    k_neighbors=3,
    classifier="svm"
)

# Train model
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")
```

### 4. Using Different Classifiers

The pipeline supports multiple classifiers including SVM, LDA, and AdaBoost. Any machine learning classifier following the fit/predict paradigm can be directly integrated.

```python
from sklearn.ensemble import AdaBoostClassifier
from metric_bci.pipeline import CSPLMNNPipeline

model = CSPLMNNPipeline(
    m_filters=6,
    k_neighbors=5,
    classifier=AdaBoostClassifier(n_estimators=200)
)
model.fit(X_train, y_train)
```

### 5. Low-Level Module Usage (Advanced)

For advanced users, CSP and LMNN modules can be used independently to experiment with custom pipelines or alternative classifiers:

```python
from metric_bci.features import CSP
from metric_bci.metric_learning import LMNN

# CSP feature extraction
csp = CSP(m_filters=6)
X_csp = csp.fit_transform(X_train, y_train)

# Metric learning
lmnn = LMNN(k=3, out_dim=2)
X_ml = lmnn.fit_transform(X_csp, y_train)
```

### 6. Project Structure

To maintain reproducibility and extensibility, the repository is structured as follows:

```text
metric_bci/
│
├── datasets/          # EEG dataset loaders
├── preprocessing/     # Filtering, segmentation, normalization
├── features/          # CSP, FBCSP, etc.
├── metric_learning/   # LMNN and other metric learning algorithms
├── pipelines/         # High-level ML pipelines
├── notebooks/         # Reproducible experiments for the paper
└── utils/             # Helper functions
```

### 7. Reproducibility & Extensibility

This library is designed to be fully modular and extensible. New feature extraction methods, metric learning algorithms, or classifiers can be added by implementing independent modules that follow a unified `fit`/`transform`/`predict` API design.
