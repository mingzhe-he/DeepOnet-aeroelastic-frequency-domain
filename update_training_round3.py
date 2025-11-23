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

# 1. Update AeroDataLoader to load new features
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
        self.X = data['X']  # [alpha_apex, Re, sin_alpha0, cos_alpha0]
        self.y = data['y']
        self.case_ids = data['case_ids']
        self.feature_names = data['feature_names']
        
        # Load normalization stats
        self.X_mean = data['X_mean']
        self.X_std = data['X_std']
        
        # Load spectral data (Log-PSD on Linear Strouhal grid)
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

# 2. Update ImprovedAeroDataset to use Affine Scaling
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
        design_params = case['design_params'] # Normalized [alpha_apex, Re, sin_alpha0, cos_alpha0]
        st_grid = case['st_grid']
        log_psd = case['log_psd']
        
        n_points = len(st_grid)
        
        # Prepare tensors
        # Branch input: design_params (repeated for each grid point)
        # Trunk input: Affine scaled Strouhal
        # Target: log_psd
        
        branch_in = np.tile(design_params, (n_points, 1))
        
        # Affine Scaling for Trunk Input
        # St in [0.05, 5.0] -> Map to approx [-1, 1]
        # Center = 2.525, Scale = 2.475
        trunk_in = ((st_grid - 2.5) / 2.5).reshape(-1, 1)
        
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
    print("Updated ImprovedAeroDataset with Affine Scaling.")

# 3. Update DeepONet (Branch Dim = 4)
don_idx = find_cell_index(nb, "class DeepONet(nn.Module):")
if don_idx != -1:
    new_don_code = """class DeepONet(nn.Module):
    def __init__(self, branch_dim=4, trunk_dim=1, hidden_dim=64, latent_dim=64):
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
    print("Updated DeepONet with branch_dim=4.")

# 4. Update train_model for 5-Fold CV
tm_idx = find_cell_index(nb, "def train_model():")
if tm_idx != -1:
    new_tm_code = """from sklearn.model_selection import KFold

def train_model_cv(n_splits=5):
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load Data
    loader = AeroDataLoader(DATA_DIR)
    spectral_data = loader.load_data()
    case_ids = np.array(list(spectral_data.keys()))
    
    # K-Fold CV
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(case_ids)):
        print(f"\\n--- Fold {fold+1}/{n_splits} ---")
        
        train_ids = case_ids[train_idx]
        val_ids = case_ids[val_idx]
        
        train_data = {k: spectral_data[k] for k in train_ids}
        val_data = {k: spectral_data[k] for k in val_ids}
        
        train_dataset = ImprovedAeroDataset(train_data)
        val_dataset = ImprovedAeroDataset(val_data)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Model
        model = DeepONet(branch_dim=4).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Training Loop
        best_val_loss = float('inf')
        patience = 50
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            
            for branch_in, trunk_in, target, weights in train_loader:
                branch_in = branch_in.view(-1, 4).to(DEVICE)
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
            
            # Validation
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for branch_in, trunk_in, target, weights in val_loader:
                    branch_in = branch_in.view(-1, 4).to(DEVICE)
                    trunk_in = trunk_in.view(-1, 1).to(DEVICE)
                    target = target.view(-1, 1).to(DEVICE)
                    weights = weights.view(-1, 1).to(DEVICE)
                    
                    pred = model(branch_in, trunk_in)
                    loss = torch.mean(weights * (pred - target)**2)
                    total_val_loss += loss.item()
                    
            avg_val_loss = total_val_loss / len(val_loader)
            
            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"deeponet_fold{fold}.pth"))
                patience_counter = 0
            else:
                patience_counter += 1
                
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.6f}")
                
            # if patience_counter >= patience:
            #     break
        
        # Load best model for this fold
        model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, f"deeponet_fold{fold}.pth")))
        
        # Evaluate Fold
        r2, mse, peak_err, rms_err = calculate_metrics(model, val_dataset, DEVICE)
        fold_results.append({
            'r2': np.mean(r2),
            'mse': np.mean(mse),
            'peak_err': np.mean(peak_err),
            'rms_err': np.mean(rms_err),
            'r2_all': r2,
            'case_ids': val_ids
        })
        print(f"Fold {fold+1} Results: R2={np.mean(r2):.4f}, PeakErr={np.mean(peak_err)*100:.2f}%")
        
    return fold_results, spectral_data
"""
    nb['cells'][tm_idx]['source'] = new_tm_code.splitlines(keepends=True)
    print("Updated train_model_cv.")

# 5. Update Visualization/Main Block
last_cell_idx = len(nb['cells']) - 1
new_viz_code = """if __name__ == "__main__":
    
    # Helper metrics function (needs to be defined before train_model_cv calls it)
    def calculate_metrics(model, dataset, device):
        \"\"\"Calculate R2, MSE, Peak St Error, and RMS Error\"\"\"
        model.eval()
        r2_scores = []
        mse_scores = []
        peak_st_errors = []
        rms_errors = []
        
        with torch.no_grad():
            for i in range(len(dataset)):
                case = dataset.data[i]
                design_params = case['design_params']
                st_grid = case['st_grid']
                log_psd = case['log_psd']
                
                n_points = len(st_grid)
                branch_in = np.tile(design_params, (n_points, 1))
                trunk_in = ((st_grid - 2.5) / 2.5).reshape(-1, 1) # Affine Scaling
                
                branch_tensor = torch.FloatTensor(branch_in).to(device)
                trunk_tensor = torch.FloatTensor(trunk_in).to(device)
                
                pred = model(branch_tensor, trunk_tensor).cpu().numpy().flatten()
                
                mse = np.mean((pred - log_psd)**2)
                ss_res = np.sum((log_psd - pred)**2)
                ss_tot = np.sum((log_psd - np.mean(log_psd))**2)
                r2 = 1 - ss_res / (ss_tot + 1e-8)
                
                peak_idx_true = np.argmax(log_psd)
                peak_idx_pred = np.argmax(pred)
                st_peak_true = st_grid[peak_idx_true]
                st_peak_pred = st_grid[peak_idx_pred]
                peak_error = np.abs(st_peak_pred - st_peak_true) / st_peak_true
                
                rms_true = np.sqrt(np.mean(log_psd**2))
                rms_pred = np.sqrt(np.mean(pred**2))
                rms_error = np.abs(rms_pred - rms_true) / rms_true
                
                r2_scores.append(r2)
                mse_scores.append(mse)
                peak_st_errors.append(peak_error)
                rms_errors.append(rms_error)
                
        return np.array(r2_scores), np.array(mse_scores), np.array(peak_st_errors), np.array(rms_errors)

    # Run CV
    results, spectral_data = train_model_cv(n_splits=5)
    
    # Aggregate Results
    avg_r2 = np.mean([r['r2'] for r in results])
    std_r2 = np.std([r['r2'] for r in results])
    avg_peak = np.mean([r['peak_err'] for r in results])
    
    print(f"\\n=== Cross-Validation Results ===")
    print(f"Mean R2: {avg_r2:.4f} +/- {std_r2:.4f}")
    print(f"Mean Peak St Error: {avg_peak*100:.2f}%")
    
    # Outlier Analysis
    print("\\n=== Outlier Analysis ===")
    all_r2 = []
    all_ids = []
    for r in results:
        all_r2.extend(r['r2_all'])
        all_ids.extend(r['case_ids'])
        
    all_r2 = np.array(all_r2)
    all_ids = np.array(all_ids)
    
    # Find worst 5 cases
    sorted_idx = np.argsort(all_r2)
    worst_indices = sorted_idx[:5]
    
    print("Top 5 Worst Cases:")
    for idx in worst_indices:
        case_id = all_ids[idx]
        r2_val = all_r2[idx]
        case_meta = spectral_data[case_id]['design_params_raw'] # [alpha_apex, Re, sin, cos]
        print(f"Case: {case_id}, R2: {r2_val:.4f}")
        print(f"  Params: Apex={np.degrees(case_meta[0]):.1f}deg, Re={case_meta[1]:.1e}, Alpha0={np.degrees(np.arcsin(case_meta[2])):.1f}deg")
"""
nb['cells'][last_cell_idx]['source'] = new_viz_code.splitlines(keepends=True)
print("Updated Main Block with CV and Outlier Analysis.")

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=2)
