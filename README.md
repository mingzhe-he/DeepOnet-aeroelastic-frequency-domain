# DeepONet for Aeroelastic Frequency Domain Analysis

This repository contains the implementation of a **Spectral DeepONet** and a **Scalar MLP** for surrogate modeling of aerodynamic forces on isosceles triangular cross-sections. The project aims to predict the Power Spectral Density (PSD) of lift coefficients and key scalar metrics (mean forces, Strouhal peak) for VIV and galloping risk assessment.

## Project Overview

The goal is to map geometric and flow parameters to the spectral response of the structure.
*   **Inputs:** Apex Angle ($\alpha_{apex}$), Reynolds Number ($Re$), Angle of Attack ($\alpha_0$).
*   **Outputs:**
    *   **Spectral Model:** $\log_{10}(PSD)$ of the Lift Coefficient ($C_L$) over a Strouhal range $St \in [0.05, 5.0]$.
    *   **Scalar Model:** Mean $C_D, C_L, C_M$ and Peak Strouhal Number ($St_{peak}$).

## Development History & Key Decisions

This codebase evolved through four rounds of critical review and refinement:

### Round 1: Initial Implementation
*   **Baseline:** Established the DeepONet architecture (Branch/Trunk nets).
*   **Issue:** Poor performance due to lack of input normalization and raw FFT magnitudes.

### Round 2: Log-Spaced Grid Experiment
*   **Attempt:** Switched to a log-spaced Strouhal grid to emphasize low frequencies.
*   **Finding:** Performance degraded ($R^2$ dropped to 0.19). The model struggled with the non-uniform density of points.
*   **Decision:** Revert to linear grid for stability.

### Round 3: Robustness & Physics Features
*   **Features:** Introduced physics-based inputs:
    *   $\alpha_{apex}$ (Apex Angle) instead of generic shape factors.
    *   $\sin(\alpha_0), \cos(\alpha_0)$ to respect angular periodicity.
    *   Affine scaling for Strouhal input ($St \in [-1, 1]$).
*   **Validation:** Implemented **5-Fold Cross-Validation**.
*   **Result:** Performance recovered ($R^2 \approx 0.64$), but Peak Strouhal Error remained high (~66%).

### Round 4: Scalar Model & Peak-Weighted Loss
*   **Scalar Model:** Created a dedicated MLP (`03_Scalar_Training.py`) for force coefficients.
    *   **Result:** Excellent accuracy for $C_D$ ($R^2=0.96$) and $C_L/C_M$ ($R^2 \approx 0.88$).
*   **Spectral Refinement:** Added a **Peak-Weighted MSE Loss** to the DeepONet.
    *   **Result:** Spectral $R^2$ improved to 0.67, Peak Error reduced to 57%.
*   **Conclusion:** The Scalar Model is recommended for engineering load estimates, while the Spectral Model provides the energy distribution shape.

## Repository Structure

*   `01_Preprocessing_Improved.py` / `.ipynb`: Data cleaning, filtering, and PSD computation.
*   `02_Training_Improved.py` / `.ipynb`: Spectral DeepONet training with Peak-Weighted Loss and 5-Fold CV.
*   `03_Scalar_Training.py`: Scalar MLP training for mean coefficients and peak frequency.
*   `processed_data/`: Directory for generated NPZ/H5 datasets (ignored in git).
*   `checkpoints_*/`: Directory for saved models (ignored in git).

## User Guide

### 1. Prerequisites
Install dependencies using `uv` or `pip`:
```bash
uv pip install numpy pandas scipy matplotlib torch h5py scikit-learn tqdm
```

### 2. Preprocessing
Run the preprocessing script to generate the dataset from raw OpenFOAM data:
```bash
python 01_Preprocessing_Improved.py
```
*   **Input:** Raw data in `../data` (configurable in script).
*   **Output:** `processed_data/improved_ml_dataset.npz` and `improved_psd_dataset.h5`.

### 3. Training the Spectral Model
Train the DeepONet to predict Log-PSD:
```bash
python 02_Training_Improved.py
```
*   **Features:** 5-Fold CV, Peak-Weighted Loss.
*   **Outputs:** Loss curves, PSD comparison plots, and model checkpoints.

### 4. Training the Scalar Model
Train the MLP for mean coefficients:
```bash
python 03_Scalar_Training.py
```
*   **Outputs:** $R^2$ and Relative Error metrics for $C_D, C_L, C_M, St_{peak}$.

## Validity Domain
> [!IMPORTANT]
> This model is trained for **Isosceles Triangles** within the following bounds:
> *   $H/D \in \{1/3, 1/2, 2/3\}$
> *   $Re_D \in [1.5 \times 10^6, 6.45 \times 10^6]$
> *   $AoA \in [55^\circ, 125^\circ]$
>
> Predictions outside this range are extrapolations and should be treated with caution.

## Key Findings
1.  **Linear vs Log Grid:** A linear Strouhal grid proved more stable for this DeepONet architecture than a log-spaced grid.
2.  **Scalar vs Spectral:** Direct scalar regression is far more accurate for mean coefficients ($R^2 > 0.9$) than deriving them from the spectral prediction.
3.  **Peak Prediction:** Precise prediction of the shedding frequency ($St_{peak}$) is challenging (errors ~40-60%) likely due to broad/noisy peaks in the turbulent wake data.

## Authors
Mingzhe He
