"""
CSP with optional LMNN and downstream classifiers.

This module implements the CSP (+ optional LMNN) and classifier grid used in
the original notebooks as a single, reusable estimator with a scikit-learn-style API.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils import validation as _sklearn_validation

from mne.decoding import CSP
from metric_learn import LMNN

# Suppress FutureWarnings from metric_learn (LMNN). AdaBoost uses algorithm="SAMME".
warnings.filterwarnings("ignore", category=FutureWarning, module="metric_learn")


@contextmanager
def _allow_nd_for_check_array():
    """
    Context manager that temporarily relaxes sklearn's array validation so that
    MNE CSP can receive 3D arrays of shape (n_epochs, n_channels, n_times).
    """
    orig_check_array = _sklearn_validation.check_array

    def _patched_check_array(array, *args, allow_nd=False, **kwargs):
        if getattr(array, "ndim", 0) >= 3:
            allow_nd = True
        return orig_check_array(array, *args, allow_nd=allow_nd, **kwargs)

    _sklearn_validation.check_array = _patched_check_array
    try:
        yield
    finally:
        _sklearn_validation.check_array = orig_check_array


@contextmanager
def _suppress_metric_future_warnings():
    """
    Suppress FutureWarnings from metric_learn LMNN while fitting.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"metric_learn")
        yield


ClassifierName = Literal["ada", "svm", "lda"]


class _EnsureFloat64(BaseEstimator):
    """
    Transformer that forces 3D (epochs) or 2D arrays to float64, C-contiguous.
    MNE CSP requires float64; when GridSearchCV uses n_jobs>1, joblib can pass
    views/float32 to the pipeline. This step guarantees CSP always receives float64.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "_EnsureFloat64":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64, order="C")

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return np.asarray(X, dtype=np.float64, order="C")


@dataclass
class ClassifierResult:
    train_acc: float
    test_acc: float
    train_f1: float
    test_f1: float
    train_mcc: float
    test_mcc: float
    best_params: Dict[str, Any]
    train_cpu_time: float
    test_cpu_time: float


def _separability_score(X: np.ndarray, y: np.ndarray) -> float:
    """
    Separability index as the ratio of between-class to within-class scatter trace
    (LDA-style criterion).
    """
    classes = np.unique(y)
    mean_overall = np.mean(X, axis=0)
    n_features = X.shape[1]
    n_classes = len(classes)

    Sw = np.zeros((n_features, n_features))
    Sb = np.zeros((n_features, n_features))

    for c in classes:
        Xc = X[y == c]
        mean_c = np.mean(Xc, axis=0)
        Sw += (Xc - mean_c).T @ (Xc - mean_c)
        diff = (mean_c - mean_overall).reshape(n_features, 1)
        Sb += n_classes * (diff @ diff.T)

    # Avoid division by zero when within-class scatter is negligible
    if np.isclose(np.trace(Sw), 0.0):
        return 0.0
    return float(np.trace(Sb) / np.trace(Sw))


class CSPLMNNPipeline(BaseEstimator, ClassifierMixin):
    """
    Common Spatial Pattern (CSP) with optional Large Margin Nearest Neighbor (LMNN)
    projection, followed by a configurable classifier.

    Supports the standard fit/predict/score interface. For full experimental
    grids (AdaBoost, SVM, LDA with cross-validated hyperparameter search and
    multiple metrics), use the ``evaluate`` method.
    """

    def __init__(
        self,
        m_filters: int = 4,
        use_lmnn: bool = True,
        k_neighbors: int = 3,
        classifier: ClassifierName | BaseEstimator = "svm",
        n_jobs: int = 1,
        random_state: Optional[int] = 42,
    ) -> None:
        self.m_filters = m_filters
        self.use_lmnn = use_lmnn
        self.k_neighbors = k_neighbors
        self.classifier = classifier
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Fitted components (used by fit/predict/score)
        self.csp_: Optional[CSP] = None
        self.lmnn_: Optional[LMNN] = None
        self.clf_: Optional[BaseEstimator] = None

    # -------------------------------------------------------------------------
    # scikit-learn estimator API
    # -------------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSPLMNNPipeline":
        """
        Fit CSP, optional LMNN, and the chosen classifier on the given epochs.

        Notes
        -----
        sklearn's Pipeline is not used because its validation assumes 2D input,
        whereas CSP requires 3D arrays (n_epochs, n_channels, n_times). Validation
        is temporarily relaxed via a context manager during CSP fit/transform.
        """
        # 1) Fit CSP on raw epochs (3D -> 2D)
        self.csp_ = CSP(
            n_components=self.m_filters,
            reg="empirical",
            log=True,
            cov_est="epoch",
        )
        with _allow_nd_for_check_array():
            X_csp = self.csp_.fit_transform(X, y)

        # 2) Optional LMNN on CSP features
        if self.use_lmnn:
            self.lmnn_ = LMNN(
                n_neighbors=self.k_neighbors,
                learn_rate=1e-6,
                n_components=2,
                verbose=False,
                random_state=self.random_state,
            )
            with _suppress_metric_future_warnings():
                X_feat = self.lmnn_.fit_transform(X_csp, y)
        else:
            self.lmnn_ = None
            X_feat = X_csp

        # 3) Fit the downstream classifier on the final feature space
        self.clf_ = self._resolve_classifier(self.classifier, self.random_state)
        with _suppress_metric_future_warnings():
            self.clf_.fit(X_feat, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.csp_ is None or self.clf_ is None:
            raise RuntimeError("The pipeline must be fitted before calling predict.")
        with _allow_nd_for_check_array():
            X_csp = self.csp_.transform(X)
        if self.use_lmnn and self.lmnn_ is not None:
            X_feat = self.lmnn_.transform(X_csp)
        else:
            X_feat = X_csp
        return self.clf_.predict(X_feat)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return float(accuracy_score(y, y_pred))

    # -------------------------------------------------------------------------
    # Full evaluation (multiple classifiers, grid search, all metrics)
    # -------------------------------------------------------------------------
    def evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run CSP (+ optional LMNN) and evaluate with AdaBoost, SVM, and LDA,
        each with internal cross-validated hyperparameter search.

        Returns a dictionary containing the test-set separability score and
        per-classifier metrics (accuracy, F1, MCC, best hyperparameters).
        """
        # MNE CSP requires float64; GridSearchCV may pass views/slices that are not.
        X_train = np.asarray(X_train, dtype=np.float64, order="C")
        X_test = np.asarray(X_test, dtype=np.float64, order="C")

        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state)

        results: Dict[str, ClassifierResult] = {}

        # Separability: fit CSP (+ optional LMNN) once on full train, transform test
        # (same as in the original notebooks; not used for classifier CV).
        csp_sep = CSP(
            n_components=self.m_filters,
            reg="empirical",
            log=True,
            cov_est="epoch",
        )
        with _allow_nd_for_check_array():
            X_train_csp = csp_sep.fit_transform(X_train, y_train)
            X_test_csp = csp_sep.transform(X_test)

        if self.use_lmnn:
            lmnn_sep = LMNN(
                n_neighbors=self.k_neighbors,
                learn_rate=1e-6,
                n_components=2,
                verbose=False,
                random_state=self.random_state,
            )
            with _suppress_metric_future_warnings():
                lmnn_sep.fit(X_train_csp, y_train)
                X_test_feat = lmnn_sep.transform(X_test_csp)
        else:
            X_test_feat = X_test_csp

        separability = _separability_score(X_test_feat, y_test)

        # Classifiers: Pipeline(CSP, [LMNN], clf) with 3D X_train so that each CV fold
        # fits CSP (and LMNN) only on that fold's train part (notebook methodology).
        # First step forces float64 so MNE CSP always receives correct dtype (needed when n_jobs>1).
        def _make_pipeline(clf: BaseEstimator) -> Pipeline:
            steps: list = [
                ("to_float64", _EnsureFloat64()),
                (
                    "csp",
                    CSP(
                        n_components=self.m_filters,
                        reg="empirical",
                        log=True,
                        cov_est="epoch",
                    ),
                ),
            ]
            if self.use_lmnn:
                steps.append(
                    (
                        "lmnn",
                        LMNN(
                            n_neighbors=self.k_neighbors,
                            learn_rate=1e-6,
                            n_components=2,
                            verbose=False,
                            random_state=self.random_state,
                        ),
                    ),
                )
            steps.append(("clf", clf))
            return Pipeline(steps)

        def run_grid(
            base_clf: BaseEstimator,
            param_grid: Dict[str, Any],
        ) -> ClassifierResult:
            pipe = _make_pipeline(base_clf)
            grid = GridSearchCV(
                pipe,
                param_grid=param_grid,
                cv=cv,
                scoring="balanced_accuracy",
                n_jobs=self.n_jobs,
            )
            with _allow_nd_for_check_array(), _suppress_metric_future_warnings():
                t0 = time.perf_counter()
                grid.fit(X_train, y_train)
                train_cpu_time = time.perf_counter() - t0
                y_train_pred = grid.best_estimator_.predict(X_train)
                t0 = time.perf_counter()
                y_test_pred = grid.best_estimator_.predict(X_test)
                test_cpu_time = time.perf_counter() - t0

            # train_acc: CV mean score (notebook uses best_score_ * 100)
            return ClassifierResult(
                train_acc=grid.best_score_ * 100.0,
                test_acc=accuracy_score(y_test, y_test_pred) * 100.0,
                train_f1=f1_score(y_train, y_train_pred, average="binary"),
                test_f1=f1_score(y_test, y_test_pred, average="binary"),
                train_mcc=matthews_corrcoef(y_train, y_train_pred),
                test_mcc=matthews_corrcoef(y_test, y_test_pred),
                best_params=grid.best_params_,
                train_cpu_time=train_cpu_time,
                test_cpu_time=test_cpu_time,
            )

        # AdaBoost: learning_rate per notebook (CSP-only [0.1, 0.5], CSP+LMNN [0.05, 0.1, 0.5])
        ada_res = run_grid(
            AdaBoostClassifier(algorithm="SAMME", random_state=self.random_state),
            {
                "clf__n_estimators": [50, 100, 200],
                "clf__learning_rate": [0.05, 0.1, 0.5]
                if self.use_lmnn
                else [0.1, 0.5],
            },
        )
        results["ada"] = ada_res

        # SVM with cross-validated hyperparameter search
        svm_res = run_grid(
            SVC(),
            {
                "clf__kernel": ["linear", "rbf", "sigmoid"],
                "clf__C": [0.1, 1, 10],
                "clf__gamma": ["scale"],
            },
        )
        results["svm"] = svm_res

        # LDA with cross-validated hyperparameter search
        lda_res = run_grid(
            LinearDiscriminantAnalysis(),
            {
                "clf__solver": ["lsqr"],
                "clf__shrinkage": ["auto"],
            },
        )
        results["lda"] = lda_res

        # Return a plain dict (separability + per-classifier metrics as dicts)
        results_dict: Dict[str, Any] = {
            "separability": separability,
            "classifiers": {
                name: vars(res) for name, res in results.items()
            },
        }
        return results_dict

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _resolve_classifier(
        clf: ClassifierName | BaseEstimator,
        random_state: Optional[int] = None,
    ) -> BaseEstimator:
        """Resolve a classifier name or estimator instance to a concrete estimator."""
        if isinstance(clf, str):
            name = clf.lower()
            if name == "ada":
                return AdaBoostClassifier(algorithm="SAMME", random_state=random_state)
            if name == "svm":
                return SVC(random_state=random_state)
            if name == "lda":
                return LinearDiscriminantAnalysis()
            raise ValueError(f"Unknown classifier alias: {clf!r}")
        return clf

