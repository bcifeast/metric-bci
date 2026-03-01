"""
Benchmark: current pipeline (notebook methodology, 3D Pipeline+GridSearchCV)
vs legacy pipeline (CSP+LMNN once, then GridSearchCV on 2D) — same data, one evaluate().

Requires: metric_bci/pipeline_legacy.py (copy of the old pipeline from the other dir).
  cp /path/to/old/metric_bci/pipeline.py metric_bci/pipeline_legacy.py

Run from repo root: python benchmark_pipeline_speed.py
"""
import time
import warnings

# Reduce log noise during benchmark
warnings.filterwarnings("ignore", category=FutureWarning)

from metric_bci.datasets import load_dataset
from metric_bci.config import FREQUENCY_BANDS

dataset_name = "dataset1"
subject = 1
freq_range = FREQUENCY_BANDS["alpha"]
nc, k_lmnn = 4, 3
n_jobs = 1  # same for both for fair comparison

print("Loading data (once)...")
X_train, y_train, X_test, y_test = load_dataset(
    dataset_name=dataset_name, subject=subject, freq_range=freq_range
)
# Ensure float64 so legacy path doesn't hit MNE error if it doesn't convert
X_train = X_train.astype("float64", order="C")
X_test = X_test.astype("float64", order="C")
print(f"  X_train {X_train.shape}, X_test {X_test.shape}\n")

# --- Current pipeline (notebook methodology) ---
from metric_bci.pipeline import CSPLMNNPipeline as PipelineCurrent

model_current = PipelineCurrent(
    m_filters=nc, use_lmnn=True, k_neighbors=k_lmnn, n_jobs=n_jobs, random_state=42
)
t0 = time.perf_counter()
_ = model_current.evaluate(X_train, y_train, X_test, y_test)
t_current = time.perf_counter() - t0
print(f"Current (Pipeline+3D GridSearchCV): {t_current:.1f} s")

# --- Legacy pipeline (if present) ---
try:
    from metric_bci.pipeline_legacy import CSPLMNNPipeline as PipelineLegacy

    model_legacy = PipelineLegacy(
        m_filters=nc, use_lmnn=True, k_neighbors=k_lmnn, n_jobs=n_jobs, random_state=42
    )
    t0 = time.perf_counter()
    _ = model_legacy.evaluate(X_train, y_train, X_test, y_test)
    t_legacy = time.perf_counter() - t0
    print(f"Legacy (CSP+LMNN once, 2D GridSearchCV): {t_legacy:.1f} s")
    print(f"Ratio (current/legacy): {t_current / t_legacy:.2f}x")
except ImportError as e:
    print("Legacy: skipped (no metric_bci/pipeline_legacy.py or import failed)")
    print(f"  {e}")

print("\nDone.")
