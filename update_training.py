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

# 1. Update AeroDataLoader
adl_idx = find_cell_index(nb, "class AeroDataLoader:")
if adl_idx != -1:
    new_adl_code = """class AeroDataLoader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.npz_path = self.data_dir / "improved_ml_dataset.npz"
        self.h5_path = self.data_dir / "improved_psd_dataset.h5"
        
    def load_data(self):
        # Load summary features and normalization stats
        data = np.load(self.npz_path, allow_pickle=True)
        self.X = data['X']  # [phi, Re, alpha0]
        self.y = data['y']
        self.case_ids = data['case_ids']
        self.feature_names = data['feature_names']
        
        # Load normalization stats
        self.X_mean = data['X_mean']
        self.X_std = data['X_std']
        
        # Load spectral data (Log-PSD on Strouhal grid)
        self.spectral_data = {}
        
        with h5py.File(self.h5_path, 'r') as f:
            for i, case_id in enumerate(self.case_ids):
                if case_id not in f:
                    continue
                    
                grp = f[case_id]
                
                # Load Strouhal grid and Log-PSD
                st_grid = grp['st_grid'][:]
                log_psd = grp['log_psd'][:]
                
                # Get design params for this case
                design_params = self.X[i]
                
                # Normalize design params
                design_params_norm = (design_params - self.X_mean) / self.X_std
                
                self.spectral_data[case_id] = {
                    'design_params': design_params_norm, # Normalized
                    'design_params_raw': design_params,  # Original for reference
                    'st_grid': st_grid,
                    'log_psd': log_psd
                }
                
        print(f"Loaded {len(self.spectral_data)} cases.")
        return self.spectral_data
"""
    nb['cells'][adl_idx]['source'] = new_adl_code.splitlines(keepends=True)
    print("Updated AeroDataLoader.")

# 2. Update ImprovedAeroDataset
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
        
        # We use the full Strouhal grid for training (128 points is small enough)
        # No need for random sampling per epoch if the grid is fixed and small.
        
        n_points = len(st_grid)
        
        # Prepare tensors
        # Branch input: design_params (repeated for each grid point)
        # Trunk input: st_grid (normalized/log transformed if needed? St is already O(1))
        # Target: log_psd
        
        branch_in = np.tile(design_params, (n_points, 1))
        trunk_in = st_grid.reshape(-1, 1)
        target = log_psd.reshape(-1, 1)
        
        # Weights: Uniform for now, or could weight by PSD magnitude
        # Since we are predicting Log-PSD, uniform weights on the log scale 
        # implicitly handle the dynamic range.
        weights = np.ones_like(target)
        
        return (
            torch.FloatTensor(branch_in),
            torch.FloatTensor(trunk_in),
            torch.FloatTensor(target),
            torch.FloatTensor(weights)
        )
"""
    nb['cells'][iad_idx]['source'] = new_iad_code.splitlines(keepends=True)
    print("Updated ImprovedAeroDataset.")

# 3. Update DeepONet
don_idx = find_cell_index(nb, "class DeepONet(nn.Module):")
if don_idx != -1:
    new_don_code = """class DeepONet(nn.Module):
    def __init__(self, branch_dim=3, trunk_dim=1, hidden_dim=64, latent_dim=64):
        super().__init__()
        
        # Simplified Branch Net (3 layers)
        self.branch = nn.Sequential(
            nn.Linear(branch_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Simplified Trunk Net (3 layers)
        self.trunk = nn.Sequential(
            nn.Linear(trunk_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x_branch, x_trunk):
        # x_branch: [batch, branch_dim]
        # x_trunk: [batch, trunk_dim]
        
        b_out = self.branch(x_branch)
        t_out = self.trunk(x_trunk)
        
        # Dot product
        # sum(b * t, dim=1)
        out = torch.sum(b_out * t_out, dim=1, keepdim=True) + self.bias
        return out
"""
    nb['cells'][don_idx]['source'] = new_don_code.splitlines(keepends=True)
    print("Updated DeepONet.")

# 4. Update train_model to include validation split
tm_idx = find_cell_index(nb, "def train_model():")
if tm_idx != -1:
    new_tm_code = """def train_model():
    # Load Data
    loader = AeroDataLoader(DATA_DIR)
    spectral_data = loader.load_data()
    
    # Split into Train/Val (80/20)
    # We split by case keys to ensure no leakage
    case_ids = list(spectral_data.keys())
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
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for branch_in, trunk_in, target, weights in train_loader:
            branch_in = branch_in.view(-1, 3).to(DEVICE)
            trunk_in = trunk_in.view(-1, 1).to(DEVICE)
            target = target.view(-1, 1).to(DEVICE)
            weights = weights.view(-1, 1).to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(branch_in, trunk_in)
            
            loss = torch.mean(weights * (pred - target)**2)
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
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
    # Save Model
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "deeponet_improved.pth"))
    print("Training complete. Model saved.")
    
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
    print("Updated train_model.")

# 5. Update Visualization Functions (Last Cell)
# We need to rewrite the visualization functions to handle Log-PSD and Strouhal
last_cell = nb['cells'][-1]
new_viz_code = """if __name__ == "__main__":
    model, val_dataset = train_model()
    
    # 5. Verification / Visualization
    
    def calculate_metrics(model, dataset, device):
        \"\"\"Calculate R2 and MSE for all cases (on Log-PSD)\"\"\"
        model.eval()
        r2_scores = []
        mse_scores = []
        
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
                trunk_in = st_grid.reshape(-1, 1)
                
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
                
                r2_scores.append(r2)
                mse_scores.append(mse)
                
        return np.array(r2_scores), np.array(mse_scores)

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
        trunk_in = st_grid.reshape(-1, 1)
        
        branch_tensor = torch.FloatTensor(branch_in).to(device)
        trunk_tensor = torch.FloatTensor(trunk_in).to(device)
        
        with torch.no_grad():
            pred = model(branch_tensor, trunk_tensor).cpu().numpy().flatten()
            
        # Calculate R2 for this plot
        ss_res = np.sum((log_psd - pred)**2)
        ss_tot = np.sum((log_psd - np.mean(log_psd))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
            
        plt.figure(figsize=(10, 6))
        plt.plot(st_grid, log_psd, label='True Log-PSD', alpha=0.7)
        plt.plot(st_grid, pred, label='Predicted Log-PSD', linestyle='--', alpha=0.7)
        plt.title(f"Log-PSD Comparison (Case {case_idx})\\n$R^2 = {r2:.4f}$")
        plt.xlabel("Strouhal Number")
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
    r2_scores, mse_scores = calculate_metrics(model, val_dataset, DEVICE)
    print(f"Average R2 (Log-PSD): {np.mean(r2_scores):.4f}")
    print(f"Average MSE (Log-PSD): {np.mean(mse_scores):.6f}")
    
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
nb['cells'][-1]['source'] = new_viz_code.splitlines(keepends=True)
print("Updated Visualization Functions.")

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=2)
