# metric-bci
Application of metric learning on CSP feature space. Common Spatial Patterns (CSP) is a widely used technique for EEG feature extraction, yet it often struggles with noise sensitivity and limited locality preservation. This work introduces a lightweight pipeline that enhances CSP representations using Large Margin Nearest Neighbors (LMNN), a metric-learning approach that preserves local neighborhood structure in a low-dimensional space. CSP features are first extracted and then mapped via LMNN to improve discriminability and robustness.

Experiments on four MI-EEG datasets show consistent test-accuracy improvements for most participants, with the largest gains observed on Dataset 1 (≈10% increase in mean α-band accuracy across AdaBoost, SVM, and LDA). The improvements are particularly pronounced for low-accuracy subjects (baseline < 60%), as confirmed by paired Wilcoxon signed-rank tests (p < 10⁻⁶).

Overall, the CSP+LMNN pipeline provides a computationally efficient and interpretable solution, making it a practical candidate for real-world BCI applications.

![CSP+LMNN pipeline](figs/graphicalAbstract.png)
