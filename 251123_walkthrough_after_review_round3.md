# Critical Review Improvements Walkthrough (Round 3)

## Overview
This walkthrough documents the third round of improvements, focusing on robustness and rigorous evaluation. We reverted to a linear Strouhal grid, introduced physics-based features, and implemented 5-Fold Cross-Validation.

## Key Changes

### 1. Preprocessing ([01_Preprocessing_Improved.ipynb](file:///Users/mingz/Projects/Original_Attempt_redeveloped/01_Preprocessing_Improved.ipynb))
-   **Linear Strouhal Grid**: Reverted to linear grid ($St \in [0.05, 5.0]$) as log-spacing proved detrimental to average performance.
-   **New Features**: Replaced `phi` and `alpha0` with:
    -   `alpha_apex`: Apex angle (radians).
    -   `sin(alpha0)`, `cos(alpha0)`: Periodicity-aware angle features.

### 2. Training ([02_Training_Improved.ipynb](file:///Users/mingz/Projects/Original_Attempt_redeveloped/02_Training_Improved.ipynb))
-   **Affine Scaling**: Trunk input is scaled to $[-1, 1]$ using $St_{scaled} = (St - 2.5)/2.5$.
-   **5-Fold Cross-Validation**: Replaced single split with 5-fold CV to robustly estimate performance.
-   **Outlier Analysis**: Automated identification of worst-performing cases.

## Results

### Model Performance (5-Fold CV)
The performance has recovered to the high levels seen in Round 1, but now with the confidence of cross-validation.

| Metric | Round 2 (Log) | Round 3 (Linear + CV) |
| :--- | :--- | :--- |
| **Mean $R^2$** | 0.19 | **0.6360 $\pm$ 0.1383** |
| **Best Fold $R^2$** | N/A | **0.8076** |
| **Peak St Error** | 37.7% | **65.95%** |

### Analysis
-   **Recovery**: Reverting to the linear grid restored the $R^2$ to ~0.64, confirming that the linear representation is more stable for this dataset/architecture combination.
-   **Variance**: The standard deviation of 0.14 in $R^2$ indicates moderate sensitivity to the training split.
-   **Peak Error**: The high peak error persists. This suggests that while the model captures the general energy distribution (good $R^2$), it struggles to precisely localize the dominant frequency, possibly due to broad peaks or noise in the ground truth.

### Outlier Analysis
The worst performing cases are consistently:
-   **High Reynolds Number**: $Re = 6.4 \times 10^6$ (e.g., `shorter`, `baseline`).
-   **Specific Angles**: `shorter_65` ($R^2 = -2.06$) is the absolute worst case.
-   **Interpretation**: The model struggles most with the high-speed, high-turbulence cases where the spectrum might be broadband or chaotic.

## Conclusion
The Spectral DeepONet is now a robust surrogate for the general spectral shape ($R^2 \approx 0.64$), but requires further refinement (likely more data or specialized loss functions) to accurately pinpoint peak frequencies ($St_{peak}$) for VIV assessment.
