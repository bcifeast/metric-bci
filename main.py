"""
Quick start script (README Section 4): dataset1, one subject, alpha band.
Run from repository root: python main.py
"""
import time

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

t0 = time.perf_counter()
model.fit(X_train, y_train)
train_cpu_time = time.perf_counter() - t0

t0 = time.perf_counter()
test_acc = model.score(X_test, y_test)
test_cpu_time = time.perf_counter() - t0

# Quick start uses fit/score (no CV). For CV-based train_acc, see run_full_experiment.py.
print(f"Test accuracy: {test_acc * 100:.2f}%")
print(f"Train CPU time: {train_cpu_time:.2f} s  |  Test CPU time: {test_cpu_time:.2f} s")
