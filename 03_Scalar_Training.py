import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from pathlib import Path
import os

# Configuration
DATA_DIR = "processed_data"
CHECKPOINT_DIR = "checkpoints_scalar"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"Using device: {DEVICE}")

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 1000

class ScalarDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ScalarMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def load_data():
    npz_path = os.path.join(DATA_DIR, "improved_ml_dataset.npz")
    data = np.load(npz_path, allow_pickle=True)
    
    X = data['X'] # [alpha_apex, Re, sin_alpha0, cos_alpha0]
    y = data['y'] # [mean_cd, mean_cl, mean_cm, st_peak]
    
    # Normalize X (using saved stats or recomputing? Let's use saved for consistency)
    X_mean = data['X_mean']
    X_std = data['X_std']
    X_norm = (X - X_mean) / X_std
    
    # Normalize y
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)
    y_std[y_std == 0] = 1.0
    y_norm = (y - y_mean) / y_std
    
    return X_norm, y_norm, y_mean, y_std, data['case_ids']

def train_scalar_model():
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    X, y, y_mean, y_std, case_ids = load_data()
    
    # K-Fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold+1}/5 ---")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_dataset = ScalarDataset(X_train, y_train)
        val_dataset = ScalarDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        model = ScalarMLP().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = 50
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            avg_train_loss = total_loss / len(train_loader)
            
            # Validation
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    pred = model(bx)
                    loss = criterion(pred, by)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"scalar_fold{fold}.pth"))
                patience_counter = 0
            else:
                patience_counter += 1
                
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.6f}")
                
            if patience_counter >= patience:
                break
        
        # Evaluate
        model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, f"scalar_fold{fold}.pth")))
        model.eval()
        
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(DEVICE)
            pred_norm = model(X_val_tensor).cpu().numpy()
            
        # De-normalize
        pred = pred_norm * y_std + y_mean
        y_true = y_val * y_std + y_mean
        
        # Metrics per component
        # [mean_cd, mean_cl, mean_cm, st_peak]
        target_names = ['Cd', 'Cl', 'Cm', 'St_peak']
        fold_metrics = {}
        
        print("Fold Metrics:")
        for i, name in enumerate(target_names):
            p = pred[:, i]
            t = y_true[:, i]
            
            mse = np.mean((p - t)**2)
            r2 = 1 - np.sum((t - p)**2) / np.sum((t - np.mean(t))**2)
            
            if name == 'St_peak':
                rel_error = np.mean(np.abs(p - t) / np.abs(t))
                print(f"  {name}: R2={r2:.4f}, RelErr={rel_error*100:.2f}%")
                fold_metrics[f'{name}_RelErr'] = rel_error
            else:
                print(f"  {name}: R2={r2:.4f}")
                
            fold_metrics[f'{name}_R2'] = r2
            
        fold_results.append(fold_metrics)

    # Aggregate
    print("\n=== Scalar Model CV Results ===")
    metrics_keys = fold_results[0].keys()
    for k in metrics_keys:
        vals = [r[k] for r in fold_results]
        print(f"{k}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

if __name__ == "__main__":
    train_scalar_model()
