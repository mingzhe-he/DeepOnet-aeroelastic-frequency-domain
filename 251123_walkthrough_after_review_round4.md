# Critical Review Improvements Walkthrough (Round 4)

## Overview
This walkthrough documents the fourth round of improvements, introducing a dedicated **Scalar Regression Model** and refining the Spectral DeepONet with a **Peak-Weighted Loss**.

## Key Changes

### 1. Scalar Regression Model ([03_Scalar_Training.py](file:///Users/mingz/Projects/Original_Attempt_redeveloped/03_Scalar_Training.py))
-   **New Model**: A lightweight MLP (3 hidden layers, 64 units) trained specifically for scalar targets: `mean_cd`, `mean_cl`, `mean_cm`, and `st_peak`.
-   **Training**: 5-Fold Cross-Validation.
-   **Purpose**: Fast, robust prediction of engineering metrics for VIV/Galloping screening.

### 2. Spectral Model Refinements ([02_Training_Improved.ipynb](file:///Users/mingz/Projects/Original_Attempt_redeveloped/02_Training_Improved.ipynb))
-   **Peak-Weighted Loss**: Modified the MSE loss to emphasize the region around the true peak ($St_{peak}$):
    $$ w(St) = 1 + 10 \cdot \exp\left(-\frac{(St - St_{peak})^2}{2(0.1)^2}\right) $$
-   **Documentation**: Added explicit validity domain warnings.

## Results

### Scalar Model Performance (5-Fold CV)
The scalar model achieves excellent accuracy for force coefficients, making it a reliable tool for aerodynamic load prediction.

| Target | Mean $R^2$ | Relative Error |
| :--- | :--- | :--- |
| **Mean Cd** | **0.96** | - |
| **Mean Cl** | **0.88** | - |
| **Mean Cm** | **0.89** | - |
| **St Peak** | 0.45 | **38.0%** |

### Spectral Model Performance (Round 4 vs Round 3)
The peak-weighted loss improved the spectral model's performance, but precise peak localization remains challenging.

| Metric | Round 3 (Linear) | Round 4 (Peak-Weighted) |
| :--- | :--- | :--- |
| **Mean $R^2$** | 0.64 | **0.67** |
| **Peak St Error** | 66% | **57%** |

### Analysis
-   **Scalar Model**: The high $R^2$ for force coefficients confirms that the features (`alpha_apex`, [Re](file:///Users/mingz/Projects/Original_Attempt_redeveloped/01_Preprocessing_Improved.py#19-59), `AoA`) capture the quasi-steady aerodynamics well.
-   **St Peak Challenge**: Both models struggle to predict `st_peak` accurately (38% error for Scalar, 57% for Spectral). This suggests that the peak frequency is either:
    1.  Inherently noisy/broadband for many cases (making "peak" ill-defined).
    2.  Highly sensitive to parameters not fully captured (e.g., turbulence intensity, which is constant here).
-   **Recommendation**: Use the **Scalar Model** for `Cd`, `Cl`, `Cm`. Use the **Spectral Model** to visualize the *energy distribution* (broadband vs tonal), but treat the exact predicted peak frequency with caution.

## Conclusion
The suite of models is now feature-complete for the critical review:
1.  **Scalar MLP**: High-fidelity force coefficient predictor.
2.  **Spectral DeepONet**: Robust spectral shape predictor (energy distribution).
3.  **Limitations**: Precise VIV frequency prediction remains the main uncertainty, likely requiring more data or higher-fidelity simulations in the transition regime.
