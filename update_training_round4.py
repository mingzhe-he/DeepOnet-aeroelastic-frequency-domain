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

# 1. Update train_model_cv to use Peak-Weighted Loss
tm_idx = find_cell_index(nb, "def train_model_cv(n_splits=5):")
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
    
    # Peak-Weighted Loss Function
    def peak_weighted_mse(pred, target, st_grid_batch):
        # pred, target: [batch_size, 1]
        # st_grid_batch: [batch_size, 1] (Affine scaled St)
        
        # We need the TRUE St_peak for each sample in the batch to compute weights.
        # However, the DataLoader flattens everything.
        # We need to pass St_peak as an auxiliary input or pre-compute weights in the Dataset.
        # Let's update the Dataset to return weights!
        # See ImprovedAeroDataset update below.
        
        # If weights are passed from DataLoader, we just use them.
        pass

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
                
                # Weighted MSE
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
    print("Updated train_model_cv with Weighted MSE.")

# 2. Update ImprovedAeroDataset to compute Peak Weights
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
        
        # Find True Peak Strouhal
        peak_idx = np.argmax(log_psd)
        st_peak = st_grid[peak_idx]
        
        n_points = len(st_grid)
        
        # Prepare tensors
        branch_in = np.tile(design_params, (n_points, 1))
        
        # Affine Scaling for Trunk Input
        trunk_in = ((st_grid - 2.5) / 2.5).reshape(-1, 1)
        
        target = log_psd.reshape(-1, 1)
        
        # Compute Peak-Focused Weights
        # w = 1 + alpha * exp( - (St - St_peak)^2 / (2 * sigma^2) )
        alpha = 10.0
        sigma = 0.1
        
        weights = 1.0 + alpha * np.exp( - (st_grid - st_peak)**2 / (2 * sigma**2) )
        weights = weights.reshape(-1, 1)
        
        return (
            torch.FloatTensor(branch_in),
            torch.FloatTensor(trunk_in),
            torch.FloatTensor(target),
            torch.FloatTensor(weights)
        )
"""
    nb['cells'][iad_idx]['source'] = new_iad_code.splitlines(keepends=True)
    print("Updated ImprovedAeroDataset with Peak Weights.")

# 3. Add Documentation Cell (Validity Domain)
# We'll insert a markdown cell at the top or update the first one
first_cell = nb['cells'][0]
if first_cell['cell_type'] == 'markdown':
    first_cell['source'].append("\n\n> [!IMPORTANT]\n> **Validity Domain**: This model is trained for Isosceles Triangles with:\n> - $H/D \\in \{1/3, 1/2, 2/3\}$\n> - $Re_D \\in [1.5 \\times 10^6, 6.45 \\times 10^6]$\n> - $AoA \\in [55^\\circ, 125^\\circ]$\n> Predictions outside this range are extrapolations and should be treated with caution.")
    print("Added Validity Domain documentation.")

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=2)
