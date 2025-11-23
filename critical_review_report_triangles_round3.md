# Third-Round Critical Review – Spectral DeepONet and Scalar Features (Isosceles Triangles)

_Reviewer_: Senior professor in fluid dynamics and machine learning  
_Code reviewed_: `01_Preprocessing_Improved.py` / notebook, `02_Training_Improved.py` / notebook, `processed_data/*`, `251123_walkthrough_after_review_round3.md`, earlier reports and walkthroughs.

This review assumes, as before, that all DES simulations and training data are for **isosceles triangular cross-sections** with base `D=3 m` and varying height `H`, at three inflow speeds and a range of AoAs. The goal is to assess the **round-3 changes** and to identify any remaining issues and improvement opportunities for a production-quality surrogate.

The round-3 updates focus on:

- Reverting to a **linear Strouhal grid**,
- Introducing **physics-based angle features** (`alpha_apex`, `sin(alpha0)`, `cos(alpha0)`),
- Applying an **affine scaling** to Strouhal for the trunk,
- Implementing **5-fold cross-validation** and basic outlier analysis.

---

## 1. Preprocessing: Features and Spectral Representation

### 1.1 Geometry and metadata

`AeroelasticDataProcessor` in `01_Preprocessing_Improved.py`:

- Treats shapes:
  - `baseline`, `baseline_lowU`, `baseline_mediumU`: `D=3.0`, `H=1.5`,
  - `taller`: `D=3.0`, `H=2.0`,
  - `shorter`: `D=3.0`, `H=1.0`,
  which correspond to isosceles triangles with different heights (apex angles).
- For each case, it:
  - Reads `U_ref` and `lRef` from the `forceCoeffs.dat` header,
  - Overrides `U_ref` for:
    - `baseline_lowU` → 5.0 m/s,
    - `baseline_mediumU` → 10.0 m/s,
  - Leaves header `U_ref` for `baseline`, `taller`, `shorter` (21.5 m/s),
  - Reads time series (`time, cm, cd, cl`) via `ForceCoeffsReader`,
  - Skips the first 20% of the record (settling time),
  - Applies low-pass filtering and downsampling to 100 Hz.

This is consistent with the previous round and is physically sensible for triangles.

### 1.2 Filtering and downsampling

`ForceProcessor.filter_and_downsample`:

- Estimates the original sampling frequency `fs` from Δt,
- If `fs > fs_target (100 Hz)`:
  - Applies a 4th-order low-pass Butterworth filter at `f_cutoff = 40 Hz` via `filtfilt`,
  - Interpolates onto a uniform time grid with `fs_target` using linear interpolation,
- Otherwise returns the original DataFrame.

For your Strouhal range (`St ∈ [0.05,5]`) and speeds (`U_ref` up to 21.5 m/s), the corresponding physical frequency band is well within 40 Hz, so this preprocessing step removes high-frequency numerical noise while preserving the physics relevant for VIV. The use of `filtfilt` avoids phase distortion.

### 1.3 Spectral representation: log-PSD on linear Strouhal grid

Round 3 keeps the **log-PSD** representation but reverts to a **linear Strouhal grid**:

- `ForceProcessor.compute_psd_strouhal`:
  - Computes Welch PSD of Cl,
  - Converts frequency `f` to `St = f·D/U_ref`,
  - Computes `log10(psd + 1e-10)`,
  - Interpolates onto `st_grid = np.linspace(0.05, 5.0, 128)`.

- `scan_and_process`:
  - Uses this linear `st_grid` for all cases,
  - Finds `st_peak` by argmax of `log_psd_st`,
  - Computes `f_peak = st_peak·U_ref/D`.

From the Round 3 walkthrough and your metrics, reverting to the linear grid has indeed **restored log-PSD R² ≈ 0.64** (with CV), suggesting that the previous log-spacing of St had degraded the match between model and data for this particular dataset and network.

Assessment:

- Log-PSD + linear Strouhal grid is a solid choice here, especially given the relatively narrow St range and the limited number of frequency points (128).
- Interpolation to the grid via `np.interp` is simple and efficient; with your filtering/downsampling it should not be unduly sensitive to noise.

Minor note:

- You still rely on default `nperseg=1024` for Welch. For shorter filtered signals, you may want to set `nperseg` adaptively (e.g. `min(1024, len(x)//2)`) to avoid warnings or poor estimates for very short time series. This is a robustness detail, not a major flaw.

### 1.4 New feature set for scalar data

`scan_and_process` now computes:

- `alpha_apex = 2*atan(D/(2H))` – the true apex angle in radians (triangular-geometry parameter),
- `alpha0 = radians(angle − 90)` – centered AoA,
- `sin_alpha0`, `cos_alpha0` – periodicity-aware angle features,
- `Re = U_ref·D/ν` – Reynolds number based on base width,
- plus `mean_cd`, `mean_cl`, `mean_cm`, `st_peak`, etc.

`create_improved_ml_dataset` then:

- Defines features:
  - `X = [alpha_apex, Re, sin_alpha0, cos_alpha0]`,
- Defines targets:
  - `y = [mean_cd, mean_cl, mean_cm, st_peak]`,
- Computes `X_mean`, `X_std` and saves `X`, `y`, `X_mean`, `X_std`, `case_ids`, and feature/target names in `improved_ml_dataset.npz`,
- Saves time series + `st_grid` + `log_psd` in `improved_psd_dataset.h5`.

Assessment:

- Using `alpha_apex` instead of `phi` is a nice refinement: apex angle is a directly interpretable shape parameter for triangles.
- Encoding AoA as `(sin_alpha0, cos_alpha0)` is the right move for periodicity, especially if you later extend the AoA range.
- Maintaining `Re_D` (based on D) is physically standard, and your overrides for U_ref now provide actual variation in Re across the dataset.

This feature set is better tuned to triangular aerodynamics than the earlier `[phi, Re, alpha0]`.

---

## 2. Spectral DeepONet – Current State and Critique

### 2.1 Inputs, outputs, and scaling

In `02_Training_Improved.py`:

- Branch inputs:
  - 4D vector `[alpha_apex, Re, sin_alpha0, cos_alpha0]` per case, normalized by `(X_mean, X_std)` from preprocessing.
- Trunk inputs:
  - Strouhal grid `st_grid ∈ [0.05,5]` per case, transformed via an **affine scaling**:
    - `St_scaled = (St − 2.5) / 2.5` → approximately [-0.98, 1.0].
- Targets:
  - `log_psd(St)` on the 128-point grid.

This addresses the core normalization issues from earlier rounds. Both branch and trunk now receive O(1) inputs, which is appropriate for tanh-based networks. The move to 4D branch inputs should not pose a problem given the moderate network size.

### 2.2 Architecture

`DeepONet`:

- Branch:
  - Input: 4,
  - Hidden: 2×64 tanh,
  - Output: 64 latent.
- Trunk:
  - Input: 1 (scaled St),
  - Hidden: 2×64 tanh,
  - Output: 64 latent.
- Output: dot product + bias.

This is a compact and expressive architecture for the spectral mapping. Given the improved feature set and scaling, there is no immediate reason to further complicate the architecture.

### 2.3 Training: k-fold cross-validation and early stopping

You replaced the earlier single train/val split with **5-fold CV**:

- `AeroDataLoader` loads all cases into `spectral_data`.
- `KFold(n_splits=5, shuffle=True, random_state=42)` splits case IDs, with:
  - For each fold:
    - Train dataset: 4/5 of the cases,
    - Validation dataset: 1/5 of the cases.
- For each fold, you:
  - Instantiate a fresh `DeepONet`,
  - Train with:
    - Adam, lr=1e−3,
    - Batch size 64,
    - Max 500 epochs,
    - Early stopping tracked via **validation loss**:
      - Best val loss is saved as `deeponet_fold{fold}.pth`,
      - `patience` is set to 50, although the early-stopping condition is commented out in the current script.
- After training a fold, you:
  - Reload the best checkpoint for that fold,
  - Evaluate R², MSE, peak Strouhal error, and RMS error on that fold’s validation cases.

This is a substantial step towards **robust performance estimation**:

- Cross-validation gives an empirical distribution of R² and errors across folds, reducing sensitivity to a single particular split.
- Saving the best model per fold based on validation loss is in line with standard early-stopping practice.

However:

- The early-stopping break (`if patience_counter >= patience: break`) is currently commented out, so each fold runs all epochs regardless of plateauing validation loss. This is mostly an efficiency issue, not a correctness problem.

### 2.4 Evaluation metrics and outlier analysis

The `calculate_metrics` function computes, for each case in a given dataset:

- `R²` and `MSE` on log-PSD across the full Strouhal grid,
- **Peak Strouhal error**:
  - `st_peak_true` = argmax of true log-PSD,
  - `st_peak_pred` = argmax of predicted log-PSD,
  - Error = `|st_peak_pred − st_peak_true| / st_peak_true`,
- **RMS error** in log-PSD:
  - `rms_true = sqrt(mean(log_psd²))`,
  - `rms_pred = sqrt(mean(pred²))`,
  - Error = `|rms_pred − rms_true| / rms_true`.

After CV, you aggregate:

- Mean R² and its standard deviation across folds,
- Mean peak Strouhal error across folds,
- Identify worst 5 cases by **R²** over all validation predictions across folds, and report their:
  - case_id,
  - R²,
  - `alpha_apex`, Re, and α0 (derived from `sin_alpha0`).

From the Round 3 walkthrough:

- Mean R² ≈ 0.636 ± 0.138 across folds,
- Best fold R² ≈ 0.81,
- Mean peak Strouhal error ≈ 66%,
- Worst cases cluster at high Re (6.45e6) and specific AoAs (e.g. `shorter_65`).

Interpretation:

- The DeepONet is now **reliably learning the overall spectral shape** (log-PSD) across the dataset, with only moderate inter-fold variability.
- However, the **peak localisation** is still weak, with ~66% average relative error in St_peak. This indicates that:
  - Either the peaks are broad/flat/noisy (so argmax is unstable),
  - Or the model’s spectral resolution in the peak region is insufficient,
  - Or the loss function (uniform MSE over all St) is not pushing the network to fine-tune the exact peak location.

This is consistent with your summary: the surrogate is a good model for the **energy distribution**, but not yet precise enough for **high-fidelity VIV peak prediction**.

### 2.5 Concerns and improvement options

1. **Peak-sensitive loss or auxiliary peak head**
   - Given that VIV risk hinges on accurate St_peak, you might:
     - Add a small **auxiliary loss** term that penalizes errors in St_peak derived from the predicted spectrum:
       - e.g., `L_peak = λ_peak·(|St_peak_pred − St_peak_true| / St_peak_true)`,
     - Or add a **multi-task head** that directly predicts `st_peak` from branch inputs (and a global summary of the trunk), trained jointly with the spectral loss.
   - This would nudge the model towards better peak localisation without sacrificing overall spectral shape.

2. **Peak-focused weighting in the MSE**
   - Currently, each St point contributes equally. You might:
     - Weight the loss more around the true peak:
       - Define a Gaussian weight function centered at `St_peak_true` with a modest width;
       - Use this to multiply the squared error.
   - This could significantly reduce St_peak error while keeping global R² high.

3. **Handling broad/ambiguous peaks**
   - For cases with very broad or multi-modal spectra, St_peak may be physically ambiguous. In those cases, even a perfect model cannot achieve small relative errors in peak location using a pure argmax criterion.
   - A more robust measure for VIV might be:
     - The **centroid** of the dominant energy band,
     - Or the **frequency of the first moment of energy** over a selected St range.
   - You could change both the evaluation metric and the target scalar from “argmax of log-PSD” to such a more stable functional.

4. **Use of `N_FREQ_SAMPLES`, `PEAK_WINDOW`, `PEAK_RATIO`**
   - These hyperparameters are currently unused in the Round 3 script. They should either be:
     - Removed to avoid confusion, or
     - Reintroduced intentionally if you decide to sample frequencies instead of using the full grid.

---

## 3. Scalar Regression on improved_ml_dataset.npz

Round 3 establishes a cleaner scalar feature set, but there is no scalar training script yet. For production usage (VIV and galloping), a **separate scalar model** for `[mean_cd, mean_cl, st_peak]` remains highly recommended.

### 3.1 Data confirmation

`improved_ml_dataset.npz` contains:

- `X` (N_cases, 4): `[alpha_apex, Re, sin_alpha0, cos_alpha0]`,
- `y` (N_cases, 4): `[mean_cd, mean_cl, mean_cm, st_peak]`,
- `X_mean`, `X_std`,
- `case_ids`, `feature_names`, `target_names`.

This matches the design described above and is a good basis for scalar regression.

### 3.2 Recommended scalar model (summary)

As in the previous round’s report, a robust scalar surrogate could be:

- Model: 3-layer MLP (PyTorch or similar):
  - Input: 4 normalized features,
  - Hidden: 2–3 layers, 32–64 tanh/ReLU units,
  - Output: 4 normalized targets.
- Training:
  - Train/val/test split or k-fold CV,
  - Normalize outputs (mean/std),
  - Loss: MSE on normalized outputs,
  - Metrics: R² and relative errors per component, plus special attention to `st_peak`.

Given the updated features and improved spectral model, the scalar MLP should achieve very high R² on mean coefficients and moderate error on St_peak, sufficient for design-level use within the trained domain.

### 3.3 Interaction with the spectral model

You have two routes for VIV:

1. **Scalar route**: `st_peak_pred` from scalar MLP → `f_shed_pred = St_peak_pred·U_ref/D`.
2. **Spectral route**: `log_psd_pred(St)` from DeepONet → derive St_peak or an energy-centroid-based frequency.

Using both and comparing them could also serve as a **consistency check**:

- Large disagreements between the two predictors can be treated as a flag for potential extrapolation or data issues.

---

## 4. Cross-Validation and Coverage

Round 3’s adoption of **5-fold CV** with shuffling and a fixed random seed (42) is a significant step towards rigorous evaluation.

### 4.1 Coverage of `(alpha_apex, Re, AoA)`

Given the limited number of distinct shapes (3 apex angles) and Re levels (3 speeds), k-fold CV with random shuffling will typically:

- Mix all shapes and Re values into each fold,
- Provide a good estimate of interpolation performance around the existing grid.

However, for **production interpretation**, it is helpful to additionally perform **structured CV**:

- **Leave-one-Re-out**:
  - Train on two Re levels, test on the third,
  - Quantifies generalisation across speed.
- **Leave-one-shape-out**:
  - Train on two apex angles (H/D), test on the third,
  - Quantifies generalisation across triangular geometry.

The current 5-fold CV mostly measures interpolation performance; structured CV would tell you how far you can trust the model in terms of **extrapolating to new triangles or speeds**.

### 4.2 Worst-case analysis

Your automated outlier analysis reports the worst R² cases and their `(alpha_apex, Re, alpha0)`:

- Worst cases tend to be at high Re (6.45e6) and particular AoAs (e.g. `shorter_65`),
- This suggests either:
  - More complex/broadband spectra at those operating points,
  - Or that the triangular wake becomes highly 3D/chaotic there.

From a practical standpoint:

- These cases should be flagged as **low-confidence** for the spectral surrogate,
- You might consider running **additional CFD simulations** around those conditions to better constrain the surrogate,
- Or treat them as near the limits of validity for the current model.

---

## 5. Remaining Opportunities and Recommendations

To further lift the models towards production-grade performance for VIV and galloping analysis of isosceles triangles:

1. **Implement the scalar MLP now**  
   - You already have the dataset and normalization; adding a compact scalar regressor for `[mean_cd, mean_cl, st_peak]` is low effort and high reward.
   - This will give you immediate, fast predictions for the quantities directly needed for Den Hartog and VIV screening.

2. **Augment the spectral loss with peak-focused terms**  
   - Add either:
     - A peak-location penalty term, or
     - Frequency-dependent weights emphasizing the peak band.
   - Re-evaluate cross-validated St_peak errors after this change.

3. **Consider alternative peak definitions**  
   - For broad or noisy peaks, replace the argmax-based `St_peak` with:
     - The energy-weighted centroid over a band, or
     - The frequency of the maximum of a smoothed log-PSD (e.g. after a small moving-average).
   - Update both preprocessing (scalar target) and evaluation to this more robust definition.

4. **Tidy unused hyperparameters and legacy code**  
   - Remove or repurpose `N_FREQ_SAMPLES`, `PEAK_WINDOW`, and `PEAK_RATIO` in `02_Training_Improved.py` to avoid confusion.
   - Update or deprecate `test_viz.py`, which still uses the old magnitude-based FFT approach and an outdated DeepONet architecture.

5. **Document validity domain and uncertainty limits**  
   - Clearly state in documentation and model wrappers:
     - `H/D ∈ {1/3, 1/2, 2/3}`, `Re_D ∈ [1.5e6, 6.45e6]`, `AoA ∈ [55°,125°]`,
     - All shapes are isosceles triangles; predictions outside this family are extrapolations.
   - Consider simple ensemble-based uncertainty (training several models with different initializations and reporting spread) to flag high-uncertainty predictions.

---

## 6. Summary

Round 3 has solidified the spectral DeepONet into a **well-normalized, cross-validated** surrogate that:

- Achieves **mean R² ≈ 0.64 ± 0.14** on log-PSD across folds,
- Robustly captures the **overall spectral energy distribution** for isosceles triangular cross-sections over the current design grid.

The main remaining shortcoming is the relatively high **St_peak error (~66%)**, which limits the surrogate’s precision for VIV peak-frequency prediction. This is likely due to a combination of broad/noisy peaks and a loss function that does not specifically target peak accuracy.

Coupled with a yet-to-be-implemented **scalar MLP** for `[mean_cd, mean_cl, st_peak]`, and with modest enhancements to the spectral loss to emphasize peaks, the pipeline can be pushed close to production-level performance for VIV and galloping assessment of triangles in the trained domain. Further gains will then come primarily from **additional CFD data** (especially in the poorly captured high-Re, specific-AoA regimes) and careful refinement of peak definitions and evaluation metrics aligned with engineering decision needs. 

