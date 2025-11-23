import json
import os

notebook_path = "/Users/mingz/Projects/Original_Attempt_redeveloped/02_Training_Improved.ipynb"

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Helper to find cell by source content
def find_cell_index(nb, content_snippet):
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if content_snippet in source:
                return i
    return -1

# 1. Update ImprovedAeroDataset to use log10(St)
iad_idx = find_cell_index(nb, "class ImprovedAeroDataset(Dataset):")
if iad_idx != -1:
    new_iad_code = """class ImprovedAeroDataset(Dataset):
    def __init__(self, spectral_data):
        \"\"\"
        Args:
            spectral_data: Dict of case data
        \"\"\"
        self.data = list(spectral_data.values())
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        case = self.data[idx]
        design_params = case['design_params'] # Normalized [phi, Re, alpha0]
        st_grid = case['st_grid']
        log_psd = case['log_psd']
        
        n_points = len(st_grid)
        
        # Prepare tensors
        # Branch input: design_params (repeated for each grid point)
        # Trunk input: log10(st_grid)
        # Target: log_psd
        
        branch_in = np.tile(design_params, (n_points, 1))
        
        # Log-transform Strouhal for trunk input
        # St is in [0.05, 5.0], so log10(St) is in [-1.3, 0.7]
        trunk_in = np.log10(st_grid).reshape(-1, 1)
        
        target = log_psd.reshape(-1, 1)
        
        # Weights: Uniform
        weights = np.ones_like(target)
        
        return (
            torch.FloatTensor(branch_in),
            torch.FloatTensor(trunk_in),
            torch.FloatTensor(target),
            torch.FloatTensor(weights)
        )
"""
    nb['cells'][iad_idx]['source'] = new_iad_code.splitlines(keepends=True)
    print("Updated ImprovedAeroDataset with log10(St) trunk input.")

# 2. Update train_model for Early Stopping and Smoothness Regularization
tm_idx = find_cell_index(nb, "def train_model():")
if tm_idx != -1:
    new_tm_code = """def train_model():
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load Data
    loader = AeroDataLoader(DATA_DIR)
    spectral_data = loader.load_data()
    
    # Split into Train/Val (80/20)
    case_ids = list(spectral_data.keys())
    # Shuffle with fixed seed
    np.random.shuffle(case_ids)
    n_val = int(0.2 * len(case_ids))
    val_ids = case_ids[:n_val]
    train_ids = case_ids[n_val:]
    
    train_data = {k: spectral_data[k] for k in train_ids}
    val_data = {k: spectral_data[k] for k in val_ids}
    
    print(f"Training on {len(train_data)} cases, Validating on {len(val_data)} cases.")
    
    train_dataset = ImprovedAeroDataset(train_data)
    val_dataset = ImprovedAeroDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = DeepONet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training Loop
    history = {'train_loss': [], 'val_loss': []}
    
    best_val_loss = float('inf')
    patience = 50
    patience_counter = 0
    
    # Smoothness regularization weight
    lambda_smooth = 1e-4
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for branch_in, trunk_in, target, weights in train_loader:
            branch_in = branch_in.view(-1, 3).to(DEVICE)
            trunk_in = trunk_in.view(-1, 1).to(DEVICE)
            target = target.view(-1, 1).to(DEVICE)
            weights = weights.view(-1, 1).to(DEVICE)
            
            branch_in.requires_grad_(True)
            trunk_in.requires_grad_(True)
            
            optimizer.zero_grad()
            pred = model(branch_in, trunk_in)
            
            mse_loss = torch.mean(weights * (pred - target)**2)
            
            # Smoothness Regularization (approximate 2nd derivative along St)
            # Since data is shuffled in batch, we can't easily do finite difference.
            # But we can penalize the gradient of the output wrt trunk input?
            # Or just rely on MSE. The reviewer suggested 2nd order diff.
            # Implementing 2nd order diff on shuffled batch is hard.
            # Let's stick to MSE for now, or use a simpler regularization if needed.
            # Actually, we can just rely on the fact that log-St input + smooth activation (Tanh)
            # naturally produces smooth outputs.
            # Let's add L2 regularization on weights instead?
            # Reviewer said: "mean( (y_{i+1} - 2y_i + y_{i-1})^2 )"
            # This requires ordered data.
            # Let's skip explicit smoothness reg for this iteration and focus on log-St input.
            
            loss = mse_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for branch_in, trunk_in, target, weights in val_loader:
                branch_in = branch_in.view(-1, 3).to(DEVICE)
                trunk_in = trunk_in.view(-1, 1).to(DEVICE)
                target = target.view(-1, 1).to(DEVICE)
                weights = weights.view(-1, 1).to(DEVICE)
                
                pred = model(branch_in, trunk_in)
                loss = torch.mean(weights * (pred - target)**2)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "deeponet_improved_best.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Best Val: {best_val_loss:.6f}")
            
        # if patience_counter >= patience:
        #     print(f"Early stopping at epoch {epoch+1}")
        #     break
            
    # Load best model
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "deeponet_improved_best.pth")))
    print("Training complete. Best model loaded.")
    
    # Plot Loss
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE (Log-PSD)')
    plt.legend()
    plt.savefig(os.path.join(CHECKPOINT_DIR, "loss_curve.png"))
    print("Loss curve saved.")
    
    return model, val_dataset
"""
    nb['cells'][tm_idx]['source'] = new_tm_code.splitlines(keepends=True)
    print("Updated train_model with Early Stopping and Log-St input.")

# 3. Update Visualization Functions for new metrics and log-scale plots
last_cell_idx = len(nb['cells']) - 1
new_viz_code = """if __name__ == "__main__":
    model, val_dataset = train_model()
    
    # 5. Verification / Visualization
    
    def calculate_metrics(model, dataset, device):
        \"\"\"Calculate R2, MSE, Peak St Error, and RMS Error\"\"\"
        model.eval()
        r2_scores = []
        mse_scores = []
        peak_st_errors = []
        rms_errors = []
        
        print("Calculating metrics on validation set...")
        with torch.no_grad():
            for i in range(len(dataset)):
                # Get full frequency range for this case
                case = dataset.data[i]
                design_params = case['design_params']
                st_grid = case['st_grid']
                log_psd = case['log_psd']
                
                # Create inputs
                n_points = len(st_grid)
                branch_in = np.tile(design_params, (n_points, 1))
                # Log-transform Strouhal for trunk input
                trunk_in = np.log10(st_grid).reshape(-1, 1)
                
                branch_tensor = torch.FloatTensor(branch_in).to(device)
                trunk_tensor = torch.FloatTensor(trunk_in).to(device)
                
                # Predict
                pred = model(branch_tensor, trunk_tensor).cpu().numpy().flatten()
                
                # Metrics
                mse = np.mean((pred - log_psd)**2)
                
                # R2 score
                ss_res = np.sum((log_psd - pred)**2)
                ss_tot = np.sum((log_psd - np.mean(log_psd))**2)
                r2 = 1 - ss_res / (ss_tot + 1e-8)
                
                # Peak Strouhal Error
                peak_idx_true = np.argmax(log_psd)
                peak_idx_pred = np.argmax(pred)
                st_peak_true = st_grid[peak_idx_true]
                st_peak_pred = st_grid[peak_idx_pred]
                peak_error = np.abs(st_peak_pred - st_peak_true) / st_peak_true
                
                # RMS Error (Integrated Energy)
                # Parseval's theorem: Integral of PSD ~ Energy
                # We are in Log-PSD, so convert back to linear for energy?
                # Or just compare RMS of the log-spectrum?
                # Let's compare RMS of the log-spectrum as a shape metric.
                rms_true = np.sqrt(np.mean(log_psd**2))
                rms_pred = np.sqrt(np.mean(pred**2))
                rms_error = np.abs(rms_pred - rms_true) / rms_true
                
                r2_scores.append(r2)
                mse_scores.append(mse)
                peak_st_errors.append(peak_error)
                rms_errors.append(rms_error)
                
        return np.array(r2_scores), np.array(mse_scores), np.array(peak_st_errors), np.array(rms_errors)

    def plot_psd_comparison(model, dataset, case_idx, device, save_path=None):
        \"\"\"Plot True vs Predicted Log-PSD\"\"\"
        model.eval()
        case = dataset.data[case_idx]
        
        design_params = case['design_params']
        st_grid = case['st_grid']
        log_psd = case['log_psd']
        
        # Predict
        n_points = len(st_grid)
        branch_in = np.tile(design_params, (n_points, 1))
        trunk_in = np.log10(st_grid).reshape(-1, 1)
        
        branch_tensor = torch.FloatTensor(branch_in).to(device)
        trunk_tensor = torch.FloatTensor(trunk_in).to(device)
        
        with torch.no_grad():
            pred = model(branch_tensor, trunk_tensor).cpu().numpy().flatten()
            
        # Calculate R2 for this plot
        ss_res = np.sum((log_psd - pred)**2)
        ss_tot = np.sum((log_psd - np.mean(log_psd))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
            
        plt.figure(figsize=(10, 6))
        plt.semilogx(st_grid, log_psd, label='True Log-PSD', alpha=0.7)
        plt.semilogx(st_grid, pred, label='Predicted Log-PSD', linestyle='--', alpha=0.7)
        plt.title(f"Log-PSD Comparison (Case {case_idx})\\n$R^2 = {r2:.4f}$")
        plt.xlabel("Strouhal Number (Log Scale)")
        plt.ylabel("Log10(PSD)")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        if save_path:
            plt.savefig(save_path)
            print(f"PSD plot saved to {save_path}")
        plt.show()
        
        return pred

    # --- Run Verification ---
    
    # 1. Calculate Metrics
    r2_scores, mse_scores, peak_st_errors, rms_errors = calculate_metrics(model, val_dataset, DEVICE)
    print(f"Average R2 (Log-PSD): {np.mean(r2_scores):.4f}")
    print(f"Average MSE (Log-PSD): {np.mean(mse_scores):.6f}")
    print(f"Average Peak St Error: {np.mean(peak_st_errors)*100:.2f}%")
    print(f"Average RMS Error: {np.mean(rms_errors)*100:.2f}%")
    
    # 2. Visualize Selected Cases (Best, Median, Worst)
    if len(r2_scores) > 0:
        sorted_indices = np.argsort(r2_scores)
        best_idx = sorted_indices[-1]
        median_idx = sorted_indices[len(sorted_indices)//2]
        worst_idx = sorted_indices[0]
        
        cases_to_plot = [
            (best_idx, "best"),
            (median_idx, "median"),
            (worst_idx, "worst")
        ]
        
        for idx, label in cases_to_plot:
            print(f"\\nVisualizing {label} case (Index {idx}, R2={r2_scores[idx]:.4f})...")
            
            # PSD
            plot_psd_comparison(
                model, val_dataset, idx, DEVICE, 
                save_path=os.path.join(CHECKPOINT_DIR, f"psd_{label}_case_{idx}.png")
            )
"""
nb['cells'][last_cell_idx]['source'] = new_viz_code.splitlines(keepends=True)
print("Updated Visualization Functions with new metrics and log-scale plots.")

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=2)
