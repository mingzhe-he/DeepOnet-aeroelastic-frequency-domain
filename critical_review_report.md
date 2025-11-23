# Critical Review of the Improved Spectral DeepONet Pipeline

_Reviewer_: Senior professor in fluid dynamics and machine learning  
_Code reviewed_: `01_Preprocessing_Improved.ipynb`, `02_Training_Improved.ipynb` / `02_Training_Improved.py`, `test_viz.py`, `processed_data/*`, and the earlier notebooks `250622_MH_Project_Preprocessing.ipynb`, `250623_C_train_fixed.ipynb`.

---

## 0. Problem Statement and Overall Assessment

### 0.1 Goal of the Pipeline

From the current code and prior notebooks, your objective is:

- Given a small number of **aeroelastic CFD cases** (rectangular cylinders with different heights and angles of attack),
- Each case described by **low‑dimensional design parameters**
  - geometric/flow features: `φ = arctan(2H/D)`, `Re = U_ref D / ν`, `α₀ = (angle − 90°)` in radians,
- Learn a surrogate that maps

> **(φ, Re, α₀) ↦ frequency–domain representation of lift coefficient Cl(f)**

so that you can:

- Reconstruct the **full Cl(t) time history** for new configurations, and
- Optionally also predict scalar quantities (`mean_cd`, `mean_cl`, `mean_cm`, `St_peak`).

Your current approach is a **DeepONet-like neural operator in frequency space**:

- **Branch net**: takes `(φ, Re, α₀)` (3D) as input.
- **Trunk net**: takes frequency `f` as input.
- Output: predicted **amplitude spectrum** `|Cl̂(f)|` (you treat it as “PSD” in the plots, but it is actually the FFT magnitude).

The updated pipeline implements sensible physics‑based features and more targeted frequency sampling near the spectral peak, but the model still exhibits:

- Very poor **R² scores** (often strongly negative),
- Predictions that are almost **case‑independent** (collapse towards a “typical” spectrum),
- Spectral reconstructions that match only in a very loose sense.

My overall assessment:

> The weak performance is not due to a single bug, but to a combination of **data representation, normalization, identifiability, and problem formulation issues**. As currently set up, the operator you ask the network to learn is **ill‑posed** given the dataset size, and the lack of normalization almost guarantees that both branch and trunk networks saturate and behave nearly constant.

In the sections below I review each component and propose **scientifically grounded, high‑leverage improvements**.

---

## 1. Data Preprocessing

### 1.1 Force data extraction and basic processing

**What the code does**

- `ForceCoeffsReader` reads `forceCoeffs.dat` (OpenFOAM output) and extracts:
  - `time`, `cm`, `cd`, `cl` (ignoring `Cl(f)`, `Cl(r)` if present).
- `AeroelasticDataProcessor.scan_and_process`:
  - Traverses directories of the form  
    `BASE_PATH/{baseline, baseline_lowU, baseline_mediumU, taller, shorter}/{angle}/postProcessing/cylinder/0/forceCoeffs.dat`.
  - Associates each **shape** with fixed `(D, H)`:
    - `baseline`, `baseline_lowU`, `baseline_mediumU`: `D = 3.0`, `H = 1.5`,
    - `taller`: `D = 3.0`, `H = 2.0`,
    - `shorter`: `D = 3.0`, `H = 1.0`.
  - Extracts `U_ref` and `lRef` from the header.
  - Skips the first **20%** of the time history when computing statistics and PSD.
- `ForceProcessor.compute_statistics` computes mean and std of `cd`, `cl`, `cm` on the **steady part**.
- `ForceProcessor.compute_psd` uses **Welch’s method** (`scipy.signal.welch`) to estimate PSD of `cl` and returns `freqs`, `psd`, and sampling frequency `fs`.
- For each case, the processor saves:
  - A summary row in `summary_df` with metadata and statistics,
  - A full time series entry in `time_series_data` with `time`, `cl`, `cd`, `cm`, and metadata.

**Assessment**

- The **20% settling‑time exclusion** is a reasonable heuristic and is consistently applied in the updated scripts (for both PSD computation and training).
- Using Welch’s method for PSD (in the preprocessing notebook) is a **good choice**: it reduces variance and produces smoother spectra than a raw FFT.
- The metadata extraction logic is robust and correctly recovers `U_ref` and `lRef`.

**Issues / opportunities**

1. **Time series stored without settling‑time removal**  
   - In `AeroelasticDataProcessor.scan_and_process`, `time_series_data[case_id]` stores the *full* `df['time']` and `df['cl']`, while the PSD and statistics use the truncated series (`t ≥ 0.2 t_end`).  
   - Later, in `AeroDataLoader.load_data` (training script), you re‑apply the 20% cutoff. This inconsistency is not catastrophic, but it is unnecessary; it slightly complicates reproducibility and can lead to mismatches if you change one side and forget to change the other.

2. **Sampling rate and numerical noise**  
   - From the `preprocessing_summary.csv`, the Strouhal peak is `St_peak ≈ 0.545`, with `f_peak ≈ 3.90625 Hz` for `U_ref = 21.5 m/s` and `D = 3.0 m`, consistent with classical cylinder shedding.  
   - That implies a **shedding period** of ~0.26 s.
   - Your simulation output uses a **very small time step** (inferred from the Nyquist in the FFT, you are effectively sampling at O(10³–10⁴ Hz)), meaning you have **hundreds of samples per shedding period**.
   - For the physics you care about (vortex shedding and perhaps the first few harmonics), this is **vastly oversampled**. Anything above, say, 20–30 × `f_peak` is effectively **turbulent/noise content** from the perspective of an operator mapping geometry/angle to Cl dynamics.

**Answer to your specific question (4000 Hz sampling, downsampling/filtering):**

- Yes, **downsampling after appropriate low‑pass filtering** would almost certainly help:
  - Choose a cutoff frequency `f_c ≈ 5–10 * f_peak` (e.g. 20–40 Hz here).
  - Apply a low‑pass filter (or equivalently, compute a PSD using Welch with segment lengths tuned to that band), then **decimate** to an effective sampling rate just above `2 f_c`.
  - This will:
    - Reduce **high‑frequency numerical noise** in the spectrum,
    - Reduce the **dimensionality** of the frequency domain representation,
    - Improve **conditioning** for learning and for numerical integration of PSDs.
- As long as you low‑pass before decimation, aliasing is not a concern, because your signal of interest is band‑limited by physics (vortex shedding).

**Recommendation 1 — time series preprocessing**

- Standardize the pipeline so that **all downstream uses** of the time series (both Strouhal calculation and DeepONet training) use a **common, filtered representation**:
  1. Apply a **settling‑time cutoff** based on either:
     - a fixed percentage of simulation time, or
     - a threshold on Cl RMS stabilisation (physically better, but more work).
  2. Apply a **detrending + windowing** (e.g. subtract mean, Hann window).
  3. Apply **Welch PSD** or a short‑time Fourier transform with overlapping windows.
  4. Either:
     - Work directly in a **fixed frequency/Strouhal grid** with PSD estimates, or
     - Downsample the time series (after low‑pass) and then FFT.

---

## 2. Feature Selection and Engineering

### 2.1 Current features (`create_improved_ml_dataset`)

For each case, you compute:

- `φ = arctan(2H/D)` – a **shape parameter** encoding aspect ratio (equivalently, H/D),
- `Re = U_ref D / ν` – Reynolds number based on the cross‑stream width D,
- `α₀ = radians(angle − 90°)` – angle of attack relative to 90°.

You then set:

- `X = [φ, Re, α₀]` as the input feature vector,
- `y = [mean_cd, mean_cl, mean_cm, st_peak]` as targets for the scalar regression problem,
- Save full time series in `improved_psd_dataset.h5` with `φ`, `Re`, `α₀`, `st_peak` in the group attributes.

These are physically meaningful features:

- `φ` (or H/D) captures **blockage / slenderness**,
- `Re` captures viscous vs inertial balance in the classical sense (cylinder literature is built around Re based on D),
- `α₀` (centered) captures orientation relative to the main flow.

### 2.2 Clarifying the current dataset vs intended configuration

From the physics and your description, the intent is:

- `baseline`: baseline shape at **U_ref ≈ 21.5 m/s**,
- `baseline_lowU`: same geometry at **lower U_ref** (e.g. 5 m/s),
- `baseline_mediumU`: same geometry at **intermediate U_ref** (e.g. 10 m/s),
- `taller` / `shorter`: different H (thus different H/D) at the same set of angles and inflow speeds.

However, in the **processed dataset currently used for training** (`processed_data/preprocessing_summary.csv`):

- All three baseline families (`baseline`, `baseline_lowU`, `baseline_mediumU`) have
  - `D = 3.0`, `H = 1.5 ⇒ φ ≈ 0.7854`,
  - `U_ref = 21.5 m/s ⇒ Re ≈ 6.45 × 10^6`,
  - `α₀` depends only on `angle`.
- There is no evidence of 5 m/s or 10 m/s runs in the summary; either:
  - The `forceCoeffs.dat` headers for the low/medium‑U simulations still contain `magUInf = 21.5`, or
  - The improved preprocessing is currently pointed only at the 21.5 m/s dataset.

This means that **in the data you are actually feeding to the model right now**, Re is effectively constant across all baseline families. From a modelling perspective, that is a data‑selection / preprocessing issue, not a problem with the definition of Re itself.

### 2.3 Hidden identifiability issue: repeated inputs with different outputs (in the current dataset)

### 2.2 Hidden identifiability issue: repeated inputs with different outputs

From `processed_data/preprocessing_summary.csv`:

- For `baseline`, `baseline_lowU`, `baseline_mediumU`:
  - `D = 3.0`, `H = 1.5 ⇒ φ ≈ 0.7854`,
  - `U_ref = 21.5 m/s ⇒ Re ≈ 6.45 × 10^6` for every case,
  - `α₀` depends only on `angle`.
- Therefore, for example:
  - `baseline_90`, `baseline_lowU_90`, `baseline_mediumU_90` all have **identical** `(φ, Re, α₀)` in `X`.
  - But their **mean Cl, Cd, Cm, and spectra clearly differ** in the CSV and in your visualizations.

This means that, from the model’s perspective, **multiple distinct outputs correspond to the same input**:

> The mapping (φ, Re, α₀) → PSD is **non‑single‑valued** in your current dataset.

No deterministic function approximator (DeepONet or otherwise) can resolve this ambiguity. The best it can do is approximate something like a **conditional mean spectrum** over all cases with the same `(φ, Re, α₀)`, which guarantees:

- Systematic errors between different “families” (baseline vs baseline_lowU vs baseline_mediumU),
- Poor R² scores even if the network trains perfectly, because the **irreducible variance** is large.

This is a core reason why your model tends to produce **almost “average” spectra** and why case‑wise R² can become strongly negative.

**Recommendation 2 — make the mapping identifiable**

- First, **fix the preprocessing/data selection** so that the intended low‑ and medium‑speed runs are actually included with correct `U_ref` in the summary (or regenerate the forceCoeffs headers). Once that is done, `Re_D = U_ref D / ν` will naturally vary across `{lowU, mediumU, baseline}`.
- Then:
  1. Maintain a **physically meaningful Reynolds number based on D**, `Re_D = U_ref D / ν`, rather than redefining Re on H. Differences in H should be represented via separate geometric features (H/D or φ), not by changing the Re definition.
  2. If, even after fixing U_ref, there remain systematic differences between families not explained by `(H/D, Re_D, α₀)` (e.g. different inflow turbulence or structural parameters), introduce an additional feature:
     - A categorical variable encoding `{baseline, baseline_lowU, baseline_mediumU, taller, shorter}` (one‑hot or embedded), or
     - Better, the underlying physical quantity (e.g. turbulence intensity) as a continuous feature.
  3. Alternatively, for cases where identical `(φ, Re_D, α₀)` truly correspond to multiple CFD runs intended as **replicates**, average their spectra and scalar quantities before training, so that the model learns the **mean behaviour**.

With corrected U_ref and a clear separation between “speed effects” (via Re_D) and “shape effects” (via H/D or φ), the mapping `(φ, Re_D, α₀) → PSD` becomes much more identifiable.

### 2.4 Recommended core feature set (before additional engineering)

Given the above, a clean, physically grounded base feature set for the branch input is:

- One geometric shape parameter:
  - Either `H_over_D = H/D`, or
  - `φ = arctan(2H/D)` (bijectively related to H/D).
- A flow parameter:
  - `Re_D = U_ref D / ν` (classical cylinder Re). Since ν and D are fixed, this is equivalent to encoding U_ref, but Re_D keeps the formula aligned with standard practice and generalizes naturally if D changes.
- Angle of attack:
  - `α₀ = radians(angle − 90°)`, or, if desired, `(sin α₀, cos α₀)` for a periodic encoding.

In practice, for the current dataset I would take

- `X_base = [ H_over_D (or φ), Re_D, α₀ ]`

and not add raw U_ref as a separate input (it is redundant with Re_D when D and ν are fixed).

Once the low/medium/high‑speed runs are correctly incorporated with their proper `U_ref`, this base feature set is already sufficient for the core physics.

### 2.5 Additional feature engineering

Some further physically motivated refinements:

1. **Strouhal‑normalized frequency**  
   - For spectral modelling, the natural variable is **Strouhal** `St = f D / U_ref`, not `f` in Hz.
   - Since `D` and `U_ref` are fixed in your current dataset, this is a scaling change, but it will matter once you generalize beyond these conditions.
   - Additionally, you already know `St_peak` for each case; you can normalize frequency as `f̃ = f / f_peak` and train the trunk net on `log f̃`. This helps collapse spectra across cases.

2. **Angle encoding**  
   - `α₀` lies in roughly `[-0.61, 0.61]` radians. For generality, representing angle as `(sin α₀, cos α₀)` often helps capture periodicity, but in your limited range this is not critical.

3. **Amplitude normalization**  
   - The amplitude of `Cl` spectra varies across cases, but much of that variation is captured by **low‑order statistics**: `std_cl`, `Cl_rms`, peak magnitude at `f_peak`.
   - A robust strategy is to:
     - Normalize the spectrum by one scalar (e.g. `Cl_rms` or peak magnitude),
     - Train the network to learn the **shape** of the spectrum,
     - Predict the scalar separately (simple regression),
     - Reconstruct the absolute spectrum as `shape(f) × amplitude_scalar`.

This reduces the effective complexity of the spectral mapping and helps the network focus on physically meaningful differences.

---

## 3. Frequency‑Domain Representation and Sampling

### 3.1 What the training script does (`AeroDataLoader`, `ImprovedAeroDataset`)

- `AeroDataLoader.load_data`:
  - Loads `X`, `y`, and `case_ids` from `improved_ml_dataset.npz`.
  - For each `case_id` in `improved_psd_dataset.h5`:
    - Loads `cl` and `time`,
    - Excludes the first 20% of the time history,
    - Subtracts the mean from `cl`,
    - Computes FFT: `fft_vals = fft(cl)`, `freqs = fftfreq(n, dt)`,
    - Keeps positive frequencies: `n_pos = n//2`, `freqs = freqs[:n_pos]`,
    - Defines magnitude as `mag = 2|fft_vals[:n_pos]|/n`,
    - Identifies peak frequency `f_peak` via `np.argmax(magnitude)`.
  - Stores per‑case data:
    - `design_params = X[i] = [φ, Re, α₀]`,
    - `freqs`, `magnitude`, `f_peak`.

- `ImprovedAeroDataset`:
  - For each case, at each epoch:
    - Samples `n_samples` frequencies (`N_FREQ_SAMPLES = 1000` in config) via **importance sampling**:
      - ~60% (`peak_ratio = 0.6`) in a window `[f_peak − w, f_peak + w]` (with `w = 0.2 Hz` by default),
      - ~40% uniformly across all frequencies.
    - Computes per‑sample **weights**:
      - `w = 1 + 10*(mag_selected/max_mag)^2` (emphasizes high‑magnitude regions).
    - Returns:
      - `branch_in`: shape `[n_samples, 3]` (repeated `design_params`),
      - `trunk_in`: `[n_samples, 1]` (selected frequencies),
      - `target`: `[n_samples, 1]` (selected magnitudes),
      - `weights`: `[n_samples, 1]`.

### 3.2 Conceptual issues

1. **Mismatch between training focus and evaluation domain**  
   - Training emphasizes:
     - Frequencies near `f_peak` (importance sampling + high weights),
     - High‑magnitude regions elsewhere.
   - However, `calculate_metrics` and `plot_psd_comparison` compute R² and MSE on the **entire frequency range**, with **uniform weighting**.
   - Thus, your loss function and your evaluation metric **optimize different objectives**:
     - The model is optimized to get the **peak and its neighbourhood** right (in a weighted sense),
     - But you judge it by performance on **all frequencies, equally**, including a long tail dominated by numerical noise and low energy, which the model has seen only sparsely and with low weight.
   - This discrepancy can easily produce **very negative R²** even if the model is doing a decent job near the peak.

2. **Raw FFT magnitude vs PSD**  
   - In preprocessing, you use **Welch PSD** for `st_peak`.  
   - In training, you use raw FFT magnitude `2|FFT|/N` without windowing or averaging.
   - This raw amplitude is **noisy**, sensitive to the finite sample length, and carries more variance than necessary.
   - Modelling **log‑PSD** or **log amplitude** from a smoothed estimate is empirically more stable and matches the way we perceive spectra.

3. **Unnormalized frequency input**  
   - The trunk net receives `freqs` in Hz directly, potentially up to the Nyquist (`O(10³–10⁴ Hz)`).
   - With several tanh layers, any frequency much above O(1–10 Hz) will push activations into **saturation**, making the trunk essentially insensitive to differences in those frequencies.
   - This leads to:
     - Very poor resolution of high‑frequency behaviour,
     - Gradients that vanish for the parts of the frequency range where the training signal is already weak.

4. **Very high spectral dimensionality relative to dataset size**

   - Each case has O(10³–10⁴) meaningful frequency samples; you downsample to 1000 per case per epoch, but overall the model is implicitly trying to learn a mapping from a 3D input to a **high‑dimensional function**.
   - With on the order of only ~100 cases, this is a **severely underdetermined functional regression problem**.
   - That is acceptable if the spectra live in a very low‑dimensional manifold (which is partly true here), but you are not exploiting that structure explicitly (see Section 6).

**Recommendation 3 — align training and evaluation and improve spectral representation**

1. **Train and evaluate on the same frequency domain and weighting**
   - Decide explicitly what you care about:
     - The entire spectrum, or
     - A band around `f_peak`, or
     - A set of low‑order spectral features (see Section 6).
   - Then:
     - If the goal is “peak and near‑peak behaviour”, compute R² and MSE **only on that band**, or compute a **weighted R²** using the same weights as the loss.
     - If you want a global R², then:
       - Either sample **frequencies uniformly** (possibly in log scale) and use uniform weights, or
       - Use a fixed spectral grid (after smoothing) and train directly on all grid points.

2. **Use a Strouhal‑ or normalized frequency**
   - Replace raw `f` by:
     - `St = f D / U_ref`, or
     - `f̃ = f / f_peak` (per case), and train the trunk on `log f̃`.
   - This collapses frequency ranges across cases and avoids the “tanh saturation” at very high frequencies.

3. **Model log‑amplitude or log‑PSD**
   - Let the network output `ŷ = log10(magnitude + ε)` and train with MSE on `ŷ`.
   - At inference, reconstruct `magnitude = 10^ŷ − ε`.
   - This:
     - Ensures positivity of the reconstructed spectrum,
     - Downweights relative importance of large peaks in the loss,
     - Better reflects human and physical perception of spectral differences (decades rather than absolute linear size).

---

## 4. Normalization and Scaling

### 4.1 Current state: essentially no normalization

In `02_Training_Improved.py`:

- Branch inputs: `x_branch = [φ, Re, α₀]` (or equivalently `[H/D, Re_D, α₀]`) are used **as is**.
  - Typical scales:
    - `φ ∈ [~0.6, ~0.8]`,
    - `Re ≈ 6.45×10^6` (almost constant across the dataset),
    - `α₀ ∈ [−0.6, 0.6]`.
- Trunk input: `x_trunk = f` (Hz) from nearly 0 up to the Nyquist (O(10³–10⁴) Hz).
- Target: `magnitude` (linear amplitude) with a wide dynamic range.
- No `StandardScaler` or `RobustScaler` is used (unlike in your earlier notebook); there is **no centering or scaling** for any of these quantities.

### 4.2 Consequences for optimization

With tanh activations, the network expects **O(1)** inputs. Instead:

- For the trunk:
  - A single initial linear layer with weights of order `O(1)` applied to `f ≈ 10³` yields preactivations `O(10³)`, immediately saturating `tanh` at ±1.
  - As a result, the trunk output becomes nearly **piecewise constant in frequency**, with essentially no gradient information for most of the spectrum.

- For the branch:
  - The `Re` component dominates the initial linear transform by several orders of magnitude and is nearly **constant across samples**.
  - Even though `Re` contains no useful variation here, it drives the branch network into saturation and suppresses the influence of φ and α₀ unless the weights are extremely small.

This leads to:

- **Poor expressive power**: both branch and trunk effectively see almost no variation in some directions.
- **Training instability**: gradients vanish in saturated regions, and the optimizer cannot meaningfully adjust the weights to capture subtle spectral differences.

This alone can explain a large part of the very poor R² scores.

**Recommendation 4 — implement rigorous normalization**

1. **Branch input normalization**
   - Because you have only a few discrete values for H/D and U_ref at present, a **physically chosen scaling** is preferable to fully data‑driven mean/std:
     - Geometry: use `H_over_D` (or φ) directly, or map it linearly to roughly [−1, 1] if desired.
     - Reynolds number: define `Re_D = U_ref D / ν` and work with `Rẽ = Re_D / 1e7` (or another fixed reference Re) so that `Rẽ` is O(1).
     - Angle: normalize `α₀` by π/2 so it lies in roughly [−1, 1], i.e. `α̃ = α₀ / (π/2)`.
   - This yields branch inputs of order one without hinging the scaling on a tiny sample mean/std, and will remain sensible if you later add more shapes or speeds.

2. **Frequency normalization**
   - Define `f̃ = (f − f_min) / (f_max − f_min)` or `St̃ = log10(St)`, then feed `f̃` to the trunk.
   - Or normalize by `f_peak` per case: `f̃ = log(f / f_peak)`, and restrict to a band like `f/f_peak ∈ [0.2, 5]`.

3. **Output normalization**
   - Put the model in the business of predicting **log‑amplitude** or amplitude divided by a scale:
     - `y_train = log10(magnitude / magnitude_ref + ε)`,
     - Teach the network to predict `y_train`,
     - Reconstruct magnitude via `magnitude = magnitude_ref·10^y_train − ε`.

Empirically, for spectral regression problems in fluid dynamics and acoustics, this sort of normalization often makes the difference between an untrainable model and a successful surrogate.

---

## 5. Model Architecture and Training Setup

### 5.1 DeepONet architecture

- Branch net:
  - Input dim: 3 (`φ, Re, α₀`),
  - Hidden: 5 fully connected layers, 64 units, tanh activations,
  - Output: 64‑dim latent vector.
- Trunk net:
  - Input dim: 1 (`f`),
  - Same 5×64 tanh architecture, output 64‑dim latent vector.
- Output:
  - Dot product of branch and trunk outputs + bias:  
    `ŷ(f, design) = Σ_k b_k(design)·t_k(f) + bias`.

This is a standard DeepONet structure and is expressive enough for the intended mapping, provided the inputs and outputs are properly normalized and the dataset is not degenerate.

**Concerns**

1. **Overparameterization given dataset size**
   - With ~100 cases and modest functional variation, a 5‑layer × 64‑unit network on both branch and trunk is likely **overkill**.
   - Overparameterization is not necessarily bad if you regularize well and the data are noise‑free, but here the spectra are noisy and the mapping is ill‑posed (Section 2.2).
   - A simpler architecture (e.g. 2–3 layers, 32–64 units) would be easier to train and interpret.

2. **No regularization besides implicit SGD and the weighted loss**
   - There is no explicit L2 regularization or spectral regularization.
   - However, at this stage you are not overfitting; you are struggling to achieve even reasonable training performance, so regularization is not the main issue.

### 5.2 Training setup

- Hyperparameters:
  - Batch size: 64 (cases per batch),
  - `N_FREQ_SAMPLES = 1000` frequencies per case per epoch,
  - Learning rate: `1e−3`,
  - Epochs: 500,
  - Optimizer: Adam, default parameters.
- Loss:
  - Weighted MSE: `mean(weights * (pred − target)^2)`.
- Data handling:
  - `DataLoader` shuffles cases; each case yields `n_samples` samples via importance sampling at each epoch.
- Evaluation:
  - After training, `calculate_metrics` uses the **trained model on the full frequency grid per case** and computes MSE and R² (no separate validation/test split).

**Concerns**

1. **No train/validation split**
   - All metrics are computed on the same data used for training, so poor R² reflects **fundamental model mis‑specification**, not overfitting.
   - For debugging and hyperparameter tuning, you should still reserve some cases as a **validation set** to detect improvements reliably.

2. **Hard‑to‑diagnose training dynamics**
   - You only track the weighted training loss and report R² at the end.
   - Given the normalization issues, it would be useful to monitor:
     - Unweighted MSE on the full grid,
     - R² on log‑amplitude in the peak band,
     - Possibly, error on derived quantities (e.g. reconstructed Cl_rms).

3. **Importance sampling hyperparameters**
   - The constants `PEAK_WINDOW` and `PEAK_RATIO` defined at the top of the script are not actually fed into `ImprovedAeroDataset` (you use default `peak_window_width=0.2`, `peak_ratio=0.6`), so the configuration is slightly misleading.
   - More importantly, the choice of a **fixed 0.2 Hz peak window** is not dimensionless. For other flow speeds or shapes, you would want the window defined in terms of **Strouhal or f/f_peak**, not absolute Hz.

**Recommendation 5 — simplify and instrument training**

- Start with a **simpler architecture** (e.g. 3 layers × 64 units) and proper normalization. Once that performs reasonably, consider extensions.
- Introduce a **validation set** (e.g. 20% of cases, stratified by shape family) and monitor:
  - Training vs validation loss,
  - R² in the peak band and for derived quantities.
- Replace the fixed `peak_window_width` by a **relative window**: e.g., `f ∈ [0.5 f_peak, 1.5 f_peak]`.

---

## 6. Evaluation Strategy and Metrics

### 6.1 Current evaluation

- `calculate_metrics`:
  - For each case:
    - Uses the trained model to predict `magnitude` on the full `freqs` grid,
    - Computes MSE and R² on the **linear magnitude**.
- `plot_psd_comparison`:
  - Plots true vs predicted magnitude (log–log) and reports R².
- `reconstruct_time_history`:
  - Uses predicted magnitude and **true phase** to reconstruct Cl(t) via inverse FFT,
  - Plots true vs reconstructed time series.
- `plot_error_analysis`:
  - Plots R² vs `Re` (normalized) using the second component of `design_params`.

**Issues**

1. **R² on raw magnitude is a poor measure of spectral similarity**
   - Spectra vary over several **orders of magnitude**.
   - Small absolute errors near the peak but moderate relative errors elsewhere can lead to very negative R² values.
   - From a physical standpoint, we care more about:
     - Correct **location and height of dominant peaks**,
     - Correct **integrated energy** (Cl_rms),
     - Overall **shape** in a log–log sense.

2. **Reynolds number as evaluation axis is uninformative**
   - As observed, `Re` is essentially constant across your current dataset, so plotting R² vs `Re` is not meaningful.
   - If you later extend the dataset to multiple U_ref or D, this plot will become more informative; for now, it mainly confirms that the model’s performance does not depend on Re (because it cannot).

3. **No task‑aligned metrics**
   - If the ultimate goal is to reconstruct Cl(t) with reasonable accuracy, you should evaluate:
     - Error in **Cl_rms** and **mean Cl**,
     - Error in **peak frequency** and **peak amplitude**,
     - Time‑domain measures: e.g. correlation coefficient of reconstructed vs true Cl(t), phase of dominant oscillation, etc.

**Recommendation 6 — define physically meaningful metrics**

- In addition to (or instead of) MSE/R² on linear magnitude:
  1. **Peak metrics**
     - Extract from the spectrum:
       - `f_peak_pred`, `mag_peak_pred`, possibly secondary peaks,
     - Evaluate errors in `St_peak`, `mag_peak`, and peak width.
  2. **Energy metrics**
     - Compute `Cl_rms` from the predicted spectrum (via Parseval’s theorem) and compare to true `Cl_rms`.
  3. **Shape metrics**
     - Use MSE or R² on **log magnitude** in a band `[0.3 f_peak, 3 f_peak]`.
  4. **Time‑domain metrics**
     - On reconstructed Cl(t), compute correlation, RMS error, and possibly phase lag of the dominant mode.

These metrics will give a much clearer picture of whether the surrogate is capturing the physics you care about, even if global R² on the full spectrum remains modest.

---

## 7. Concrete, Scientifically Grounded Improvements

Here I summarize the main changes I would recommend, ordered roughly from **highest impact / easiest** to **more structural redesigns**.

### 7.1 Fix normalization and frequency scaling

1. **Normalize branch inputs**
   - Compute mean and standard deviation of `[φ, Re, α₀]` across all cases and transform to zero mean, unit variance (or rescale Re by a fixed factor like 1e7).
   - Store normalization parameters with the dataset and use them consistently in training and inference.

2. **Normalize frequency**
   - Prefer using **Strouhal** `St = f D / U_ref` or `f/f_peak` as the trunk input.
   - Normalize to O(1) and optionally work in log space: `St̃ = log10(St)` or `log(f/f_peak)`.

3. **Predict log amplitude**
   - Train the network on `log10(magnitude + ε)` or `log10(PSD + ε)`.
   - This both stabilizes training and aligns better with typical spectral analysis practices.

These steps alone should dramatically improve trainability and bring R² into a reasonable range, even before more advanced changes.

### 7.2 Resolve identifiability and replicate handling

- Either:
  - Add a **categorical feature** (e.g. one‑hot) for `shape_family ∈ {baseline, baseline_lowU, baseline_mediumU, taller, shorter}`, so that the network can represent differences between these families, or
  - Treat runs with identical `(φ, Re, α₀)` as **replicates** and average their spectra (and scalar targets) before training.

This is crucial: as long as different spectra share identical input features, the best achievable R² at the case level is limited by this inherent ambiguity.

### 7.3 Use smoothed PSDs on a fixed Strouhal grid

Replace the raw FFT used in the training script by:

1. From the filtered time series (post‑settling, low‑passed), compute a **Welch PSD** of Cl:
   - Choose overlapping windows (e.g. 50% overlap, Hann window),
   - Use enough segments to reduce variance.
2. Convert frequency to **Strouhal** and interpolate PSD onto a fixed grid:
   - E.g., `St ∈ [0.05, 5]` with 128–256 log‑spaced points.
3. Store for each case:
   - `St_grid` (common for all cases),
   - `PSD_cl(St_grid)` or `log10(PSD_cl(St_grid))`.

Advantages:

- The input to the trunk net becomes **case‑independent** (same grid for all),
- The problem reduces to learning a mapping from design parameters to a **vector of length ~128** rather than to a function defined on a dense FFT grid,
- Spectra are much smoother and less contaminated by high‑frequency noise.

You can then either:

- Keep the DeepONet formulation with trunk defined on the common Strouhal grid, or
- Move to a simpler MLP that directly predicts the full PSD vector or its low‑dimensional representation (next subsection).

### 7.4 Dimensionality reduction of spectra (POD/PCA)

Given the limited number of cases, it is highly advantageous to perform a **POD/PCA of the spectra**:

1. Stack all log‑PSD vectors into a matrix `S ∈ ℝ^{N_cases × N_freq}`.
2. Compute the principal components (PCA) or POD modes:
   - `S ≈ Σ_{k=1}^K a_k φ_k`, with `K` chosen so that ≥ 99% of variance is captured.
   - For your dataset, I would be surprised if `K > 5–10`.
3. Train a simple regression model (MLP, Gaussian process, or even linear regression) that maps `(φ, Re, α₀, family)` to the **mode coefficients** `a_k`.
4. Reconstruct the spectrum as:
   - `logPSD_pred(St) = Σ_k a_k(design) φ_k(St)`,
   - `PSD_pred = 10^{logPSD_pred}`.

This approach is:

- **Standard** in reduced‑order modelling (Karhunen–Loève/POD + regression),
- Particularly effective when you have **few cases** but each case has a rich spatial/temporal structure,
- Much easier to debug and tune than a large DeepONet.

### 7.5 Align training and evaluation objectives

- Decide explicitly:
  - Do you care most about the **dominant unsteady load** (peak region)?
  - Or about **overall spectral content**, including high‑frequency noise?
- Then:
  - If the focus is the peak and its few harmonics, restrict both training and evaluation to a band around `St_peak` and use physically meaningful metrics (peak location, amplitude, Cl_rms).
  - If you truly want the full spectrum, adopt a **fixed Strouhal grid** and a **log‑spectral error** metric, possibly weighted by PSD to emphasize energetic regions.

### 7.6 Time‑domain modelling alternatives (for future work)

Longer‑term, with more data, you could consider:

- **Sequence‑to‑sequence models** (e.g. 1D CNNs or transformers) that map design parameters to Cl(t) directly, with loss functions in both time and frequency domains.
- **Physics‑informed neural networks (PINNs)** that solve a reduced aeroelastic model and treat the CFD data as calibration targets (e.g. fitting parameters in a low‑order vortex shedding model).
- **Neural operators** (Fourier neural operators) applied in time, learning the map from inflow conditions to Cl(t).

Given the current dataset size, however, the **POD + regression** path described above is likely to be more robust and interpretable.

---

## 8. Summary

To directly address your example questions:

- **Training on selected frequencies near the peak vs predicting full spectrum**  
  - Your improved training does heavily emphasize frequencies near the peak, but you evaluate using uniform R² over the entire spectrum. This mismatch, combined with noisy high‑frequency content, almost guarantees poor global R². Aligning the training loss and the evaluation metric (or restricting both to a physically relevant band) is essential.

- **High sampling rate and noisy PSD**  
  - The original 4000 Hz (or similar) sampling is far higher than needed for a vortex shedding frequency near 4 Hz. Downsampling after low‑pass filtering or using smoothed PSD estimates (Welch) on a fixed Strouhal grid is not only safe but **strongly recommended** to reduce noise and improve learnability.

In addition, the most critical structural issues are:

1. **Lack of normalization** of inputs, frequency, and outputs, leading to saturated networks and poor trainability.
2. **Non‑identifiable mapping** due to multiple cases with identical `(φ, Re, α₀)` but different spectra.
3. **Very high effective output dimensionality** relative to the number of cases, without explicit exploitation of low‑dimensional spectral structure.
4. **Misalignment between training objective and evaluation metric.**

Addressing these points—particularly normalization, identifiability, and dimensionality reduction—will give you a much more robust and scientifically sound basis for building a spectral surrogate of Cl(t). Once those foundations are in place, refinements in network architecture and hyperparameters will start to pay off; without them, even very sophisticated models will continue to struggle. 
