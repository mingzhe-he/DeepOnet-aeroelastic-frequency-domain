import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from scipy.fft import fft, ifft, fftfreq
from torch.utils.data import Dataset, DataLoader

# Configuration
DATA_DIR = "processed_data"
CHECKPOINT_DIR = "checkpoints_improved"
DEVICE = torch.device("cpu")

# Define Model (Must match training script)
class DeepONet(nn.Module):
    def __init__(self, branch_dim=3, trunk_dim=1, hidden_dim=64, latent_dim=64):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(branch_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.trunk = nn.Sequential(
            nn.Linear(trunk_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x_branch, x_trunk):
        b_out = self.branch(x_branch)
        t_out = self.trunk(x_trunk)
        out = torch.sum(b_out * t_out, dim=1, keepdim=True) + self.bias
        return out

# Define Dataset (Simplified for loading)
class AeroDataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.npz_path = os.path.join(data_dir, "improved_ml_dataset.npz")
        self.h5_path = os.path.join(data_dir, "improved_psd_dataset.h5")
        
    def load_data(self):
        data = np.load(self.npz_path, allow_pickle=True)
        self.X = data['X']
        self.case_ids = data['case_ids']
        
        self.spectral_data = {}
        with h5py.File(self.h5_path, 'r') as f:
            for i, case_id in enumerate(self.case_ids):
                if case_id not in f: continue
                grp = f[case_id]
                cl = grp['cl'][:]
                time = grp['time'][:]
                
                # Exclude settling time
                t_start = time[-1] * 0.2
                mask = time >= t_start
                time = time[mask]
                cl = cl[mask]
                
                cl = cl - np.mean(cl)
                dt = np.mean(np.diff(time))
                n = len(cl)
                freqs = fftfreq(n, dt)
                fft_vals = fft(cl)
                n_pos = n // 2
                freqs = freqs[:n_pos]
                magnitude = 2.0 * np.abs(fft_vals[:n_pos]) / n
                
                self.spectral_data[case_id] = {
                    'design_params': self.X[i],
                    'freqs': freqs,
                    'magnitude': magnitude
                }
        return self.spectral_data

class ImprovedAeroDataset(Dataset):
    def __init__(self, spectral_data):
        self.data = list(spectral_data.values())
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# Visualization Functions
def plot_psd_comparison(model, dataset, case_idx, device, save_path=None):
    model.eval()
    case = dataset.data[case_idx]
    design_params = case['design_params']
    freqs = case['freqs']
    magnitude = case['magnitude']
    
    n_points = len(freqs)
    branch_in = np.tile(design_params, (n_points, 1))
    trunk_in = freqs.reshape(-1, 1)
    
    branch_tensor = torch.FloatTensor(branch_in).to(device)
    trunk_tensor = torch.FloatTensor(trunk_in).to(device)
    
    with torch.no_grad():
        pred = model(branch_tensor, trunk_tensor).cpu().numpy().flatten()
        
    ss_res = np.sum((magnitude - pred)**2)
    ss_tot = np.sum((magnitude - np.mean(magnitude))**2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    
    print(f"Case {case_idx}: R2 = {r2}")
    
    if save_path:
        plt.figure()
        plt.loglog(freqs, magnitude, label='True')
        plt.loglog(freqs, pred, label='Pred')
        plt.savefig(save_path)
        plt.close()
    return pred

def reconstruct_time_history(model, dataset, case_idx, device, npz_path, h5_path, save_path=None):
    pred_magnitude = plot_psd_comparison(model, dataset, case_idx, device)
    
    data = np.load(npz_path, allow_pickle=True)
    case_ids = data['case_ids']
    case_id = case_ids[case_idx]
    
    with h5py.File(h5_path, 'r') as f:
        grp = f[case_id]
        cl = grp['cl'][:]
        time = grp['time'][:]
        t_start = time[-1] * 0.2
        mask = time >= t_start
        time = time[mask]
        cl = cl[mask]
        cl = cl - np.mean(cl)
        fft_vals = fft(cl)
        phase = np.angle(fft_vals)
        
    n = len(cl)
    n_pos = n // 2
    pred_fft_mag = pred_magnitude * n / 2.0
    full_pred_mag = np.zeros(n)
    k_max = min(len(pred_fft_mag), n_pos)
    full_pred_mag[:k_max] = pred_fft_mag[:k_max]
    
    if n % 2 == 0:
        for k in range(1, k_max):
            full_pred_mag[n-k] = pred_fft_mag[k]
        if k_max == n_pos:
             full_pred_mag[n_pos] = np.abs(fft_vals[n_pos])
    else:
        for k in range(1, k_max):
            full_pred_mag[n-k] = pred_fft_mag[k]
            
    recon_fft = full_pred_mag * np.exp(1j * phase)
    recon_cl = ifft(recon_fft).real
    
    if save_path:
        plt.figure()
        plt.plot(time, cl, label='True')
        plt.plot(time, recon_cl, label='Recon')
        plt.savefig(save_path)
        plt.close()
    print(f"Reconstruction done for case {case_idx}")

# Main Execution
if __name__ == "__main__":
    # Load Model
    model = DeepONet().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "deeponet_improved.pth"), map_location=DEVICE))
    
    # Load Data
    loader = AeroDataLoader(DATA_DIR)
    spectral_data = loader.load_data()
    dataset = ImprovedAeroDataset(spectral_data)
    
    # Test on index 0
    h5_path = os.path.join(DATA_DIR, "improved_psd_dataset.h5")
    npz_path = os.path.join(DATA_DIR, "improved_ml_dataset.npz")
    
    reconstruct_time_history(model, dataset, 0, DEVICE, npz_path, h5_path, save_path="test_recon.png")
