import json
import os

notebook_path = "/Users/mingz/Projects/Original_Attempt_redeveloped/02_Training_Improved.ipynb"

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# 1. Modify AeroDataLoader
found_loader = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "class AeroDataLoader" in source:
            # Find the insertion point
            target_str = "                cl = grp['cl'][:]\n                time = grp['time'][:]\n                \n                # FFT Computation"
            replacement_str = "                cl = grp['cl'][:]\n                time = grp['time'][:]\n                \n                # Exclude settling time (first 20%)\n                t_start = time[-1] * 0.2\n                mask = time >= t_start\n                time = time[mask]\n                cl = cl[mask]\n                \n                # FFT Computation"
            
            if target_str in source:
                new_source = source.replace(target_str, replacement_str)
                # Split back into lines, keeping newlines
                cell['source'] = [line + '\n' for line in new_source.split('\n')[:-1]] + [new_source.split('\n')[-1]]
                # Fix potential double newlines at end of lines if split adds them, but split('\n') consumes them.
                # Actually, let's just use splitlines(True)
                cell['source'] = new_source.splitlines(keepends=True)
                found_loader = True
                print("Modified AeroDataLoader.")
            else:
                print("Target string not found in AeroDataLoader cell.")
            break

if not found_loader:
    print("AeroDataLoader cell not found.")

# 2. Add Visualization Functions to the last cell
# The last cell is currently:
# if __name__ == "__main__":
#     model, dataset = train_model()

new_viz_code = """if __name__ == "__main__":
    model, dataset = train_model()
    
    # 5. Verification / Visualization
    
    def calculate_metrics(model, dataset, device):
        \"\"\"Calculate R2 and MSE for all cases\"\"\"
        model.eval()
        r2_scores = []
        mse_scores = []
        
        print("Calculating metrics...")
        with torch.no_grad():
            for i in range(len(dataset)):
                # Get full frequency range for this case
                case = dataset.data[i]
                design_params = case['design_params']
                freqs = case['freqs']
                magnitude = case['magnitude']
                
                # Create inputs
                n_points = len(freqs)
                branch_in = np.tile(design_params, (n_points, 1))
                trunk_in = freqs.reshape(-1, 1)
                
                branch_tensor = torch.FloatTensor(branch_in).to(device)
                trunk_tensor = torch.FloatTensor(trunk_in).to(device)
                
                # Predict
                pred = model(branch_tensor, trunk_tensor).cpu().numpy().flatten()
                
                # Metrics
                mse = np.mean((pred - magnitude)**2)
                
                # R2 score
                ss_res = np.sum((magnitude - pred)**2)
                ss_tot = np.sum((magnitude - np.mean(magnitude))**2)
                r2 = 1 - ss_res / (ss_tot + 1e-8)
                
                r2_scores.append(r2)
                mse_scores.append(mse)
                
        return np.array(r2_scores), np.array(mse_scores)

    def plot_psd_comparison(model, dataset, case_idx, device, save_path=None):
        \"\"\"Plot True vs Predicted PSD\"\"\"
        model.eval()
        case = dataset.data[case_idx]
        
        design_params = case['design_params']
        freqs = case['freqs']
        magnitude = case['magnitude']
        
        # Predict
        n_points = len(freqs)
        branch_in = np.tile(design_params, (n_points, 1))
        trunk_in = freqs.reshape(-1, 1)
        
        branch_tensor = torch.FloatTensor(branch_in).to(device)
        trunk_tensor = torch.FloatTensor(trunk_in).to(device)
        
        with torch.no_grad():
            pred = model(branch_tensor, trunk_tensor).cpu().numpy().flatten()
            
        # Calculate R2 for this plot
        ss_res = np.sum((magnitude - pred)**2)
        ss_tot = np.sum((magnitude - np.mean(magnitude))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
            
        plt.figure(figsize=(10, 6))
        plt.loglog(freqs, magnitude, label='True PSD', alpha=0.7)
        plt.loglog(freqs, pred, label='Predicted PSD', linestyle='--', alpha=0.7)
        plt.title(f"PSD Comparison (Case {case_idx})\\n$R^2 = {r2:.4f}$")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        if save_path:
            plt.savefig(save_path)
            print(f"PSD plot saved to {save_path}")
        plt.show()
        
        return pred

    def reconstruct_time_history(model, dataset, case_idx, device, npz_path, h5_path, save_path=None):
        \"\"\"Reconstruct time history using predicted magnitude and true phase\"\"\"
        # 1. Get predicted magnitude
        pred_magnitude = plot_psd_comparison(model, dataset, case_idx, device, save_path=None) # Don't save yet
        
        # 2. Get True Phase from original data
        # Load case_ids from npz to find the correct ID
        data = np.load(npz_path, allow_pickle=True)
        case_ids = data['case_ids']
        case_id = case_ids[case_idx]
        
        with h5py.File(h5_path, 'r') as f:
            grp = f[case_id]
            cl = grp['cl'][:]
            time = grp['time'][:]
            
            # Apply same settling time filter
            t_start = time[-1] * 0.2
            mask = time >= t_start
            time = time[mask]
            cl = cl[mask]
            cl = cl - np.mean(cl) # Remove mean
            
            # FFT to get phase
            fft_vals = fft(cl)
            phase = np.angle(fft_vals)
            
        # 3. Reconstruct
        n = len(cl)
        n_pos = n // 2
        
        # Inverse scaling: magnitude * N / 2
        pred_fft_mag = pred_magnitude * n / 2.0
        
        # Map predicted magnitude to full length
        full_pred_mag = np.zeros(n)
        
        # Handle length mismatch if any (dataset might have truncated freqs differently)
        # In AeroDataLoader: freqs = freqs[:n_pos]
        # So pred_magnitude length is n_pos.
        
        # Fill positive
        # Ensure we don't overflow if n_pos differs slightly
        k_max = min(len(pred_fft_mag), n_pos)
        full_pred_mag[:k_max] = pred_fft_mag[:k_max]
        
        # Mirror for negative freqs
        if n % 2 == 0:
            # Even N
            for k in range(1, k_max):
                full_pred_mag[n-k] = pred_fft_mag[k]
            # Nyquist at n_pos
            if k_max == n_pos:
                 full_pred_mag[n_pos] = np.abs(fft_vals[n_pos]) # Use true magnitude for Nyquist
        else:
            # Odd N
            for k in range(1, k_max):
                full_pred_mag[n-k] = pred_fft_mag[k]
                
        # Reconstruct complex array
        recon_fft = full_pred_mag * np.exp(1j * phase)
        
        # IFFT
        from scipy.fft import ifft
        recon_cl = ifft(recon_fft).real
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(time, cl, label='True History', alpha=0.7)
        plt.plot(time, recon_cl, label='Reconstructed (Pred Mag + True Phase)', linestyle='--', alpha=0.7)
        plt.title(f"Time History Reconstruction (Case {case_idx})")
        plt.xlabel("Time (s)")
        plt.ylabel("Cl")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Time history plot saved to {save_path}")
        plt.show()

    def plot_error_analysis(r2_scores, dataset, save_path=None):
        \"\"\"Plot R2 vs Reynolds Number\"\"\"
        # Extract Re from dataset
        Re_values = []
        for case in dataset.data:
            # design_params: [phi, Re, alpha0]
            Re_values.append(case['design_params'][1])
            
        plt.figure(figsize=(8, 6))
        plt.scatter(Re_values, r2_scores, alpha=0.6)
        plt.xlabel("Reynolds Number (Normalized)")
        plt.ylabel("$R^2$ Score")
        plt.title("Model Performance vs Reynolds Number")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Error analysis plot saved to {save_path}")
        plt.show()

    # --- Run Verification ---
    
    # 1. Calculate Metrics
    r2_scores, mse_scores = calculate_metrics(model, dataset, DEVICE)
    print(f"Average R2: {np.mean(r2_scores):.4f}")
    print(f"Average MSE: {np.mean(mse_scores):.6f}")
    
    # 2. Plot Error Analysis
    plot_error_analysis(r2_scores, dataset, save_path=os.path.join(CHECKPOINT_DIR, "r2_vs_re.png"))
    
    # 3. Visualize Selected Cases (Best, Median, Worst)
    sorted_indices = np.argsort(r2_scores)
    best_idx = sorted_indices[-1]
    median_idx = sorted_indices[len(sorted_indices)//2]
    worst_idx = sorted_indices[0]
    
    cases_to_plot = [
        (best_idx, "best"),
        (median_idx, "median"),
        (worst_idx, "worst")
    ]
    
    h5_path = os.path.join(DATA_DIR, "improved_psd_dataset.h5")
    npz_path = os.path.join(DATA_DIR, "improved_ml_dataset.npz")
    
    for idx, label in cases_to_plot:
        print(f"\\nVisualizing {label} case (Index {idx}, R2={r2_scores[idx]:.4f})...")
        
        # PSD
        plot_psd_comparison(
            model, dataset, idx, DEVICE, 
            save_path=os.path.join(CHECKPOINT_DIR, f"psd_{label}_case_{idx}.png")
        )
        
        # Time History
        reconstruct_time_history(
            model, dataset, idx, DEVICE, npz_path, h5_path,
            save_path=os.path.join(CHECKPOINT_DIR, f"time_hist_{label}_case_{idx}.png")
        )
"""

# Replace the last cell
last_cell = nb['cells'][-1]
if last_cell['cell_type'] == 'code':
    last_cell['source'] = new_viz_code.splitlines(keepends=True)
    print("Modified last cell.")
else:
    print("Last cell is not code, appending new cell.")
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": new_viz_code.splitlines(keepends=True)
    }
    nb['cells'].append(new_cell)

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=2)
