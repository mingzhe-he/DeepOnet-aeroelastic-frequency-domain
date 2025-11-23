# Second-Round Critical Review – Isosceles Triangular Cross-Sections

_Reviewer_: Senior professor in fluid dynamics and machine learning  
_Code reviewed_: `01_Preprocessing_Improved.py` / notebook, `02_Training_Improved.py` / notebook, `processed_data/*`, `251123_walkthrough_after_critical_review.md`, and earlier versions.

This review explicitly assumes that **all DES simulations and all ML training data correspond to isosceles triangular cross-sections**, not rectangular prisms. I will:

1. Re-state what the updated pipeline now does (with this geometry in mind),
2. Critically assess the methodology and implementation, focusing on data preprocessing, feature engineering, normalization, model architecture, training, and evaluation,
3. Propose concrete, production-oriented improvements for both:
   - the **spectral DeepONet** (log-PSD on Strouhal grid),
   - a new **scalar regressor** for `[mean_cd, mean_cl, st_peak]`,
4. Sketch a robust scalar model architecture and training setup suitable for production use.

---

## 0. Problem Statement and Physical Context

### 0.1 Target application

You are building a surrogate model for **unsteady aerodynamics of isosceles triangles** in cross-flow, to support:

- **Vortex-Induced Vibration (VIV)** risk assessment:
  - Need the **Strouhal number** (dominant shedding frequency) as a function of geometry (`H`, `D`), flow speed (`U_ref`), and angle of attack `AoA`.
- **Galloping** risk assessment:
  - Need **time-averaged Cd and Cl** and their variation with AoA, to evaluate **Den Hartog** criteria (sign of effective aerodynamic damping).

The input space is:

- Geometry: isosceles triangle with base `D` and height `H`. In your current dataset:
  - `D = 3 m` fixed,  
  - `H = 1.0, 1.5, 2.0 m` (three aspect ratios).
- Flow speed: `U_ref ∈ {5, 10, 21.5} m/s` (low, medium, high).
- Angle of attack: AoA ≈ 55°–125° (in 5° increments).

The outputs of interest are:

- **Spectral:** log-PSD of lift coefficient `Cl` over a Strouhal grid `St ∈ [0.05, 5.0]`,
- **Scalar:** `[mean_cd, mean_cl, mean_cm, st_peak]`.

### 0.2 Overall assessment

After the first review, you have:

- Corrected the **U_ref** metadata so low/medium/high-speed cases are distinct in `Re_D`,
- Introduced consistent **filtering/downsampling** and a **Strouhal-grid log-PSD** representation,
- Normalized branch inputs and simplified the DeepONet architecture,
- Achieved strong validation performance: `Average R²(Log-PSD) ≈ 0.65`, `Best R² ≈ 0.97`.

The pipeline is now **physically sound and reasonably mature**. The remaining work is about:

- Tightening a few implementation details,
- Adding a **scalar regression model** for `[mean_cd, mean_cl, st_peak]`,
- Improving **robustness and interpretability** so the surrogate can be safely used in production for VIV and galloping assessment of isosceles triangles near your design domain.

---

## 1. Data Preprocessing (Isosceles Triangles)

### 1.1 Geometry and shapes

In `01_Preprocessing_Improved.py`, `AeroelasticDataProcessor` assumes:

- `shape_params`:
  - `baseline`, `baseline_lowU`, `baseline_mediumU`: `D = 3.0`, `H = 1.5`,
  - `taller`: `D = 3.0`, `H = 2.0`,
  - `shorter`: `D = 3.0`, `H = 1.0`.

From your clarification, these labels correspond to **isosceles triangles** with different heights:

- Base `D` fixed, height `H` varying ⇒ different **apex angles** and slenderness.

The shape parameter you compute later,

- `φ = arctan(2H/D)`,

is in this context essentially a re-parameterization of the **half-apex angle**:

- For an isosceles triangle with base `D` and height `H`, half-apex angle `α_half` satisfies `tan(α_half) = (D/2)/H = D/(2H)`,
- So `φ = arctan(2H/D) = π/2 − α_half`.

Thus, `φ` is a **one-to-one encoding of apex angle** and is a perfectly reasonable geometric feature for triangles.

### 1.2 Time-series processing

For each case:

- `ForceCoeffsReader` reads `time, cm, cd, cl` from `forceCoeffs.dat`.
- `AeroelasticDataProcessor`:
  - Extracts `U_ref` and `lRef` from headers,
  - **Overrides** `U_ref` for:
    - `baseline_lowU` → 5.0 m/s,
    - `baseline_mediumU` → 10.0 m/s,
  - Leaves `U_ref` from header for `baseline`, `taller`, `shorter` (21.5 m/s).
- `ForceProcessor`:
  - Drops the first 20% of the record (`t_start = t_end * 0.2`) to exclude transients,
  - Applies `filter_and_downsample`:
    - Estimates original fs from `Δt`,
    - If `fs > fs_target (100 Hz)`:
      - Applies a 4th-order low-pass Butterworth at `f_cutoff = 40 Hz` using `filtfilt` for zero-phase,
      - Resamples time to a uniform grid with `fs_target`,
    - Otherwise leaves the series unchanged.

Assessment:

- For your Strouhal band (`St ∈ [0.05,5]`), and typical `U_ref`, `D`, the corresponding frequency range is up to O(30–40 Hz). Choosing `f_cutoff = 40 Hz` and `fs_target = 100 Hz` is appropriate and keeps the spectrum well-resolved without being dominated by high-frequency noise.
- Using `filtfilt` avoids phase distortion, which is important if you later use the time series for reconstruction.
- The 20% settling-time cut is still a heuristic. For production robustness, you may eventually want a **diagnostic based on Cl RMS stabilization** (e.g. detect when RMS deviation falls below a threshold), but this is not critical right now.

### 1.3 Spectral representation on Strouhal grid

`ForceProcessor.compute_psd_strouhal` now:

- Computes Welch PSD of Cl with estimated `fs` and `nperseg=1024`,
- Converts frequency `f` to Strouhal `St = f·D/U_ref`,
- Computes `log_psd = log10(psd + 1e-10)`,
- Interpolates onto a **fixed Strouhal grid**:
  - `st_grid = linspace(0.05, 5.0, 128)`,
  - Returns `log_psd_interp` on that grid.

This is precisely the right direction:

- The Strouhal grid accounts for both geometry (`D`) and flow speed (`U_ref`) in a dimensionless way,
- Working in **log-PSD** stabilizes training and reflects human perception (orders of magnitude),
- A fixed grid ensures the trunk input domain is **case-independent**.

Minor considerations:

- At very low U_ref (5 m/s), the maximum Strouhal from Welch may not reach 5.0, but given your `fs_target = 100 Hz`, this is still likely adequate. It is worth periodically verifying that the Welch `St` grid always covers [0.05, 5] so interpolation does not rely on extrapolation at the upper end.

### 1.4 Summary data for scalar regression

`create_improved_ml_dataset`:

- Computes:
  - `phi = arctan(2H/D)` (shape/apex-geometry parameter),
  - `Re = U_ref D / ν` (Reynolds number based on base width),
  - `alpha0 = radians(angle − 90°)` (AoA centered on 90°).
- Builds:
  - `X = [phi, Re, alpha0]`,
  - `y = [mean_cd, mean_cl, mean_cm, st_peak]`.
- Computes:
  - `X_mean`, `X_std` over all cases (per-dimension),
  - Saves `X`, `y`, `X_mean`, `X_std`, `case_ids`, and feature/target names to `improved_ml_dataset.npz`.
- Saves filtered time series and `st_grid`, `log_psd` per case to `improved_psd_dataset.h5`.

This gives you **everything needed** to build both scalar and spectral models in a clean, consistent way.

---

## 2. Feature Selection and Engineering

### 2.1 Current features for triangles

The current branch features are:

- `φ = arctan(2H/D)` – encodes triangle apex geometry,
- `Re_D = U_ref D / ν` – classical Reynolds number based on cross-flow width,
- `α₀ = radians(angle − 90°)` – AoA centered around bluff orientation.

For isosceles triangles with fixed base and varying height, this triplet is a **reasonable minimal description**:

- `φ` distinguishes the three **apex angles** (shorter/baseline/taller),
- `Re_D` distinguishes the **three flow speeds**,
- `α₀` captures the strong dependence of separation and reattachment patterns on AoA.

### 2.2 Identifiability and coverage (triangular case)

With the corrected U_ref overrides, the mapping:

- `(φ, Re_D, α₀) → {mean coefficients, spectrum}`

is now **single-valued** for your dataset: there are **no longer** distinct CFD runs with identical `(φ, Re_D, α₀)` but different outputs. This fixes the major identifiability problem from the first review.

However:

- Coverage in `(φ, Re_D, α₀)` is still **discrete and sparse**:
  - Only 3 φ values (three apex heights),
  - 3 Re levels,
  - Finite AoA range (55°–125° with 5° steps).
- As long as you use the surrogate **within this domain** (or only mildly extrapolate in AoA), the model can function as a production tool. But caution is needed when:
  - Extrapolating to significantly different apex angles (H/D outside [1/3, 2/3]),
  - Extrapolating in Re beyond {1.5e6, 3e6, 6.45e6},
  - Extrapolating AoA far outside the training range (e.g. 30°, 150°).

Recommendation:

- For production use, explicitly **document the valid domain**:
  - `H/D ∈ {1/3, 1/2, 2/3}`, `Re_D ∈ [1.5e6, 6.45e6]`, `AoA ∈ [55°,125°]`.
- For any new application (e.g. different D, different apex angles), either:
  - Enrich the training dataset with additional CFD runs, or
  - Treat predictions as **rough guidance**, not as quantitative design data.

### 2.3 Potential feature refinements specific to triangles

Going beyond the minimal set, you could consider:

1. **Apex-angle explicit encoding**  
   Instead of `φ`, you may use the **half-apex angle** directly:
   - `α_half = atan(D/(2H))`,
   - Or the full apex angle `α_apex = 2·atan(D/(2H))`.
   A nice property is that aerodynamic regimes (e.g. slender vs blunt triangles) are often discussed in terms of apex angle; encoding that directly can help interpretability and may generalize better if you extend to more shapes.

2. **Angle encoding**  
   For large AoA ranges, representing `α₀` as `(sin α₀, cos α₀)` can help with periodicity and avoid artificial discontinuities near ±π. In your current range (~±0.61 rad), this is not critical, but if you expand to full 360° this would be recommended.

3. **Amplitude-related features**  
   For the spectral model, including a simple amplitude descriptor (e.g. RMS of Cl, or peak `log_psd` at `St_peak`) as an additional feature could help the network disambiguate subtle pattern changes, especially if you later incorporate triangles with different sharpness or leading-edge rounding.

At your current dataset size, I would keep the branch feature vector **compact** (3–5 inputs) to reduce overfitting risk.

---

## 3. Frequency-Domain Representation and Sampling

The updated pipeline implements the major improvements from the first review:

- All training now uses:
  - A **fixed Strouhal grid** (`st_grid ∈ [0.05,5]`, 128 points),
  - **Log-PSD of Cl** as the target.
- There is no longer any per-epoch random sampling of frequencies; you train on the full grid per case.

This has multiple advantages:

- The trunk network’s input domain is simple and compact,
- Loss and evaluation are computed on the **same grid and quantity**,
- You avoid aliasing of high-frequency noise and achieve strong R² on log-PSD.

Remaining refinements:

1. **Normalize or transform Strouhal before feeding the trunk**
   - Currently you feed `St` as is (`0.05–5`) to tanh-based trunk layers. This is acceptable (the range is modest), but you can improve conditioning slightly by:
     - Simple affine scaling, e.g. `St̃ = (St − 0.5) / 0.5` so most points lie in roughly [−1,9], or
     - Using `log10(St)` as input (remaps the grid more evenly if you later use a log spacing).
   - Not essential, but worth experimenting with for marginal gains.

2. **Log-spacing of the Strouhal grid (optional)**
   - For VIV analysis, the peak region and the lower frequencies are more important than the extreme high-frequency tail.
   - A **log-spaced** Strouhal grid (e.g. 0.05 to 5 with log10 spacing) could:
     - Offer more resolution around low St,
     - Reduce the influence of the high-frequency tail on the loss.
   - This would require updating both preprocessing and training to use the new grid consistently.

3. **POD/PCA reduction of log-PSD (medium-term enhancement)**

Even though R² is now good, the spectra almost certainly lie in a low-dimensional manifold. You could:

- Stack all log-PSD vectors into a matrix `S ∈ ℝ^{N_cases × N_freq}`,
- Perform PCA/POD and retain `K` modes capturing e.g. 99% variance,
- Train the DeepONet (or a simple MLP) to predict the **mode coefficients** instead of the full 128-point spectrum.

Benefits:

- Reduced output dimension and smoother fitting,
- Easier error analysis in terms of dominant spectral modes,
- Better interpretability (e.g. “mode 1 = base shedding, mode 2 = harmonics/triangular separation bubble”).

Given your present performance, this is not required, but it is an excellent path if you plan to expand the dataset.

---

## 4. Normalization and Scaling

### 4.1 Branch input normalization

You now:

- Save `X_mean`, `X_std` for `[φ, Re_D, α₀]` in `improved_ml_dataset.npz`,
- In `02_Training_Improved.py`, load these stats and normalize:
  - `design_params_norm = (design_params − X_mean) / X_std`.

This is correct and fully resolves the earlier tanh saturation issue. The scales are:

- `φ` O(1),
- `Re_D` O(10⁶),
- `α₀` O(10⁰),

and standardization brings them all to O(1).

For production robustness:

- Ensure you **never recompute** `X_mean`, `X_std` at inference time; always use the saved values from the dataset used to train the model.
- If you later expand the dataset significantly (e.g. other triangles or speeds), you may want to **freeze** these stats to a reference set to avoid shifting the scaling over time; but this is more of an MLOps consideration than a modelling issue.

### 4.2 Strouhal and log-PSD scaling

- Trunk input is currently raw `St` in [0.05,5], which is acceptable.
- The target is `log10(PSD)`, which is the right choice for:
  - Handling large dynamic range,
  - Giving roughly equal weight to peaks and tails in a quadratic loss.

Output normalization beyond log-transform (e.g. subtract mean log-PSD) is not necessary, especially since the DeepONet already trains well.

---

## 5. Spectral DeepONet: Architecture, Training, Evaluation

### 5.1 Architecture

`DeepONet` in `02_Training_Improved.py`:

- Branch:
  - Input: 3 (normalized [φ, Re, α₀]),
  - 2 hidden layers of 64 tanh neurons,
  - Output: 64-dimensional latent representation.
- Trunk:
  - Input: 1 (St),
  - 2 hidden layers of 64 tanh neurons,
  - Output: 64-dimensional latent representation.
- Output: dot product of branch and trunk outputs + scalar bias.

This is a **reasonable, compact DeepONet** for your problem:

- Low-dimensional inputs (3D → 64),
- 128 frequency points per case,
- Dataset on the order of ~100 cases.

Given your current R²(Log-PSD) ~0.65 on validation and up to ~0.97 for the best case, the architecture appears appropriate and not obviously under- or over-parameterized.

Potential enhancements:

1. **Smoothness regularization in St**
   - Physical spectra are smooth in log space (especially after Welch). You could add a **penalty on second-order differences** in the predicted log-PSD across St:
     - `L_smooth = λ_smooth · mean( (ŷ_{i+1} − 2ŷ_i + ŷ_{i−1})² )`,
   - This can reduce overfitting to noise in the tails and slightly improve generalization.

2. **Multi-task head to predict scalar spectral features**
   - You could allow the DeepONet branch/trunk combination to also output:
     - predicted `st_peak`, or
     - predicted `log_Cl_rms`,
   - alongside the log-PSD, sharing most of the network and adding a small extra head.
   - This is more complex; given that you will design a dedicated scalar model anyway (Section 6), I would not prioritize this unless you want tight coupling between scalar and spectral predictions.

### 5.2 Training setup

- Batch size: 64 (cases per batch, each with full St grid),
- Learning rate: 1e−3,
- Epochs: 1000,
- Optimizer: Adam,
- Loss: MSE on log-PSD (uniform weights),
- Split: 80/20 train/validation by case,
- Diagnostics:
  - Training vs validation loss curves,
  - R²(Log-PSD) and MSE(Log-PSD) on validation,
  - Best/median/worst validation cases plotted.

This is a solid training regime. Recommendations for production reliability:

1. **Early stopping / model selection**
   - Instead of always taking epoch 1000, you should:
     - Track validation loss (or mean R²),
     - Save a checkpoint whenever it improves,
     - Use the **best validation epoch** as your production model.

2. **Random seed and cross-validation**
   - Fix a random seed (NumPy and PyTorch) so train/val splits and initializations are reproducible.
   - Optionally, perform **k-fold cross-validation** over cases to quantify variability in performance and detect any sensitivity to the specific train/val split.

3. **Outlier analysis**
   - For the worst validation cases (e.g. R² < 0), inspect:
     - Whether they correspond to boundary AoA, Re, or apex angle,
     - Whether the CFD for those triangles exhibits unusual behavior.
   - If these are “physics outliers,” consider adding more CFD runs near that regime; if they are numerical artifacts, you may want to refine the CFD there.

### 5.3 Evaluation metrics

You currently compute:

- MSE and R² on log-PSD over the full St grid on the validation set.

Additional, more task-oriented metrics for VIV/galloping assessment:

1. **Peak Strouhal error**
   - From true and predicted log-PSD, extract:
     - `St_peak_true`, `St_peak_pred`,
   - Compute:
     - `ΔSt = St_peak_pred − St_peak_true`,
     - relative error `|ΔSt| / St_peak_true`,
   - For VIV, this is more directly relevant than global spectral R².

2. **Integrated energy / Cl RMS**
   - Compute `Cl_rms` from predicted PSD (via Parseval’s relation) and compare to true `Cl_rms`.
   - This links directly to **force amplitude** relevant for VIV amplitude estimation.

3. **Phase-independent time-domain metrics (optional)**
   - If you reconstruct Cl(t) using predicted magnitude and some phase model:
     - Compute correlation coefficient and RMS error between reconstructed and true Cl(t) envelopes or Hilbert transforms.
   - This is more involved and may not be necessary for initial production deployment.

---

## 6. Scalar Regression Model for [mean_cd, mean_cl, st_peak]

For production use, especially for galloping and VIV screening, you need a **robust, interpretable scalar surrogate** for:

- `mean_cd`, `mean_cl`, `st_peak` (and optionally `mean_cm`).

You already have a well-prepared scalar dataset in `improved_ml_dataset.npz`. Below I sketch a model and training setup that is both simple and strong.

### 6.1 Data and preprocessing

From `improved_ml_dataset.npz`, you have:

- `X` of shape `(N_cases, 3)` with columns `[phi, Re_D, alpha0]`,
- `y` of shape `(N_cases, 4)` with columns `[mean_cd, mean_cl, mean_cm, st_peak]`,
- `X_mean`, `X_std` for normalization,
- `case_ids`.

For scalar regression, I recommend:

1. **Use normalized X**:

```python
X_norm = (X - X_mean) / X_std
```

2. **Standardize outputs per-component**:

Compute:

```python
y_mean = y.mean(axis=0)          # shape (4,)
y_std  = y.std(axis=0)
y_std[y_std == 0] = 1.0
y_norm = (y - y_mean) / y_std
```

Store `y_mean`, `y_std` for later de-normalization.

3. Optionally:

- Work with `log10(st_peak)` as the 4th target, especially if you plan to extend to a wider St range. For your current St range, direct regression is fine.

### 6.2 Model architecture (PyTorch sketch)

A compact, multi-output MLP works very well here. For example:

```python
import torch
import torch.nn as nn

class ScalarAeroMLP(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=64, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        # x: [batch, 3]
        return self.net(x)   # [batch, 4] -> normalized [mean_cd, mean_cl, mean_cm, st_peak]
```

Notes:

- Three hidden layers of 64 tanh units is overkill for 3 inputs, but safe; you can reduce to 2 layers if you prefer.
- One joint network for all four outputs allows the model to exploit correlations between Cd, Cl, Cm, and St_peak.

### 6.3 Training setup

Training loop (conceptually similar to the DeepONet training):

- Split cases into train/validation (e.g. 80/20, stratified by shape and U_ref if possible),
- Hyperparameters:
  - Batch size: 16–32,
  - Learning rate: 1e−3,
  - Epochs: up to ~1000 with early stopping,
  - Optimizer: Adam (optionally with weight decay 1e−4 as mild regularization).
- Loss:

```python
criterion = nn.MSELoss()

for epoch in range(EPOCHS):
    model.train()
    ...
    preds = model(x_batch)            # normalized outputs
    loss = criterion(preds, y_batch)  # y_batch is normalized y
    ...
```

At validation time:

- Compute MSE and R² for each target separately:
  - For `Cd`, `Cl`, `Cm`: use MSE on **de-normalized** values, plus R²,
  - For `St_peak`: likewise, with emphasis on relative error `|ΔSt| / St_true`.

For production readiness:

- Use **early stopping** based on validation loss,
- Save the best model weights and the normalization parameters (`X_mean`, `X_std`, `y_mean`, `y_std`) together.

### 6.4 Practical use for VIV and galloping

For a new isosceles triangle in the same family (valid domain) with given `(H, D, U_ref, AoA)`:

1. Compute:
   - `phi = arctan(2H/D)`,
   - `Re_D = U_ref D / ν`,
   - `alpha0 = radians(AoA − 90°)`.
2. Normalize:
   - `x = [phi, Re_D, alpha0]`,
   - `x_norm = (x − X_mean) / X_std`.
3. Predict normalized outputs:
   - `y_pred_norm = model(x_norm)`,
   - `y_pred = y_pred_norm * y_std + y_mean`.
4. Extract:
   - `mean_cd_pred, mean_cl_pred, mean_cm_pred, st_peak_pred`.
5. VIV:
   - Shedding frequency: `f_shed_pred = st_peak_pred * U_ref / D`.
6. Galloping:
   - Repeat prediction at AoA±ΔAoA (e.g. ±3°) to estimate `dCl/dα`:
     - `alpha0_plus`, `alpha0_minus`,
     - `Cl_plus`, `Cl_minus`,
     - `dCl_dalpha ≈ (Cl_plus − Cl_minus) / (2Δα)`.
   - Compute Den Hartog-like parameter, e.g. `H = dCl/dα + Cd` (with the sign convention matching your coordinate system).
   - Identify AoA ranges where `H < 0` as galloping-critical.

This scalar model can be used independently from the spectral DeepONet for fast screening.

---

## 7. Additional Recommendations for Production Use

To raise both surrogates (spectral DeepONet and scalar MLP) to “production-grade”:

1. **Explicit test set**
   - Reserve a subset of cases (e.g. specific AoAs at each U_ref and H/D) as a **hold-out test set** used only for final evaluation.
   - Report metrics (R², relative errors) on this set for both scalar and spectral models.

2. **Uncertainty awareness**
   - At minimum:
     - Flag input configurations that lie outside the convex hull of training `(φ, Re_D, α₀)`,
     - Warn users when the surrogate is extrapolating.
   - If desired, implement a simple **ensemble** (e.g. 5 independently trained MLPs and DeepONets) and use the spread of predictions as an uncertainty indicator.

3. **Physical sanity checks**
   - Enforce or at least check:
     - `St_peak_pred > 0`,
     - Reasonable ranges for Cd, Cl (e.g. Cd between -2 and 6, Cl within physically plausible limits for triangles),
     - Smooth variation with AoA (no spurious oscillations).
   - These can be enforced via light regularization (e.g. penalizing large second derivatives with respect to AoA) or post-processing (clipping outliers).

4. **Documentation of validity domain**
   - Clearly state in code and documentation:
     - The training ranges in `(H/D, Re_D, AoA)`,
     - The assumption of isosceles triangles with the specific base and height variations.

5. **Align helper scripts**
   - Update or retire `test_viz.py`, which still uses the old FFT-magnitude-based approach and the old DeepONet architecture. For consistent diagnostics:
     - Either adapt it to the current Strouhal log-PSD model, or
     - Consolidate all verification in `02_Training_Improved.py`.

---

## 8. Summary

With the corrections and improvements you have already implemented, plus the scalar regression model sketched above, your pipeline is now well-positioned to act as a **production-quality surrogate** for:

- Predicting **Strouhal number** and **mean Cd/Cl/Cm** for isosceles triangular cross-sections within the trained domain,
- Supporting **VIV** screening via `St_peak_pred` and **galloping** assessment via `mean Cd/Cl` and `dCl/dα`.

The remaining steps to reach a truly robust tool are mostly about:

- Tightening training (early stopping, test sets, seeds),
- Adding uncertainty awareness and physical sanity checks,
- Possibly adopting dimensionality reduction (PCA/POD) for the spectral model if you expand the dataset.

Within the current domain of DES training data, there is no obvious fundamental limitation preventing you from reaching near “CFD-grade” accuracy in the mean coefficients and Strouhal predictions; the current R²(Log-PSD) and the simple scalar regression architecture outlined here should be able to deliver the level of performance required for design and risk assessment, provided you remain within the trained geometry and flow ranges. 

