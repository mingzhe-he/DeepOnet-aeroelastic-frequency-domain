import json
import os

notebook_path = "/Users/mingz/Projects/Original_Attempt_redeveloped/01_Preprocessing_Improved.ipynb"

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

# 1. Update ForceProcessor
fp_idx = find_cell_index(nb, "class ForceProcessor:")
if fp_idx != -1:
    new_fp_code = """class ForceProcessor:
    \"\"\"Processes force time series data\"\"\"
    
    def __init__(self, df, t_start=None):
        self.df = df
        if t_start is not None:
            self.df = self.df[self.df['time'] >= t_start]
            
    def compute_statistics(self):
        \"\"\"Compute mean and std of coefficients\"\"\"
        stats = {}
        for col in ['cd', 'cl', 'cm']:
            stats[f'mean_{col}'] = self.df[col].mean()
            stats[f'std_{col}'] = self.df[col].std()
        return stats
        
    def compute_psd(self, col='cl', fs=None):
        \"\"\"Compute Power Spectral Density using Welch's method\"\"\"
        if fs is None:
            # Estimate sampling frequency
            dt = np.mean(np.diff(self.df['time']))
            fs = 1.0 / dt
            
        x = self.df[col].values
        # Remove mean
        x = x - np.mean(x)
        
        freqs, psd = welch(x, fs, nperseg=1024)
        return freqs, psd, fs

    def compute_psd_strouhal(self, u_ref, D, st_grid, col='cl', fs=None):
        \"\"\"Compute Log-PSD on a fixed Strouhal grid\"\"\"
        if fs is None:
            dt = np.mean(np.diff(self.df['time']))
            fs = 1.0 / dt
            
        x = self.df[col].values
        x = x - np.mean(x)
        
        # Welch's method
        freqs, psd = welch(x, fs, nperseg=1024)
        
        # Convert to Strouhal: St = f * D / U_ref
        st = freqs * D / u_ref
        
        # Interpolate onto fixed Strouhal grid
        # Use log10(PSD) for better scaling
        # Add epsilon to avoid log(0)
        log_psd = np.log10(psd + 1e-10)
        
        # Interpolate
        log_psd_interp = np.interp(st_grid, st, log_psd)
        
        return log_psd_interp

    def filter_and_downsample(self, fs_target=100.0, f_cutoff=40.0):
        \"\"\"Low-pass filter and downsample the signal\"\"\"
        from scipy.signal import butter, filtfilt, resample
        
        # Estimate current fs
        dt = np.mean(np.diff(self.df['time']))
        fs = 1.0 / dt
        
        if fs <= fs_target:
            return self.df
            
        # Low-pass filter
        nyq = 0.5 * fs
        normal_cutoff = f_cutoff / nyq
        b, a = butter(4, normal_cutoff, btype='low', analog=False)
        
        for col in ['cl', 'cd', 'cm']:
            self.df[col] = filtfilt(b, a, self.df[col])
            
        # Downsample
        num_samples = int(len(self.df) * fs_target / fs)
        
        # Resample time and data
        new_time = np.linspace(self.df['time'].iloc[0], self.df['time'].iloc[-1], num_samples)
        
        new_data = {'time': new_time}
        for col in ['cl', 'cd', 'cm']:
            # Linear interpolation is safer for non-periodic transient data
            new_data[col] = np.interp(new_time, self.df['time'], self.df[col])
            
        return pd.DataFrame(new_data)
"""
    nb['cells'][fp_idx]['source'] = new_fp_code.splitlines(keepends=True)
    print("Updated ForceProcessor.")

# 2. Update AeroelasticDataProcessor.scan_and_process
adp_idx = find_cell_index(nb, "class AeroelasticDataProcessor:")
if adp_idx != -1:
    # We need to replace the whole class or just the method. Replacing whole class is safer to ensure context.
    # But the class is large. Let's try to construct the new class code.
    # The previous cell content:
    old_source = "".join(nb['cells'][adp_idx]['source'])
    
    # We need to modify scan_and_process inside it.
    # Let's just replace the whole cell with the new version.
    
    new_adp_code = """class AeroelasticDataProcessor:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        
        # Shape parameters (D=3.0m constant)
        # baseline, baseline_lowU, baseline_mediumU all have H=1.5
        self.shape_params = {
            'baseline': {'D': 3.0, 'H': 1.5},
            'baseline_lowU': {'D': 3.0, 'H': 1.5},
            'baseline_mediumU': {'D': 3.0, 'H': 1.5},
            'taller': {'D': 3.0, 'H': 2.0},
            'shorter': {'D': 3.0, 'H': 1.0}
        }
        
    def extract_metadata_from_header(self, file_path):
        \"\"\"Extract U_ref and lRef from forceCoeffs.dat header\"\"\"
        u_ref = None
        lRef = None
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('# magUInf'):
                        parts = line.split(':')
                        if len(parts) > 1:
                            u_ref = float(parts[1].strip())
                    elif line.startswith('# lRef'):
                        parts = line.split(':')
                        if len(parts) > 1:
                            lRef = float(parts[1].strip())
                    elif not line.startswith('#'):
                        break
            return u_ref, lRef
        except Exception as e:
            print(f"Error reading header of {file_path}: {e}")
            return None, None
        
    def scan_and_process(self):
        \"\"\"Scan directories and process all data\"\"\"
        summary_data = []
        time_series_data = {}
        
        if not self.base_path.exists():
            print(f"Base path {self.base_path} does not exist.")
            return pd.DataFrame(), {}
        
        shape_dirs = [d for d in self.base_path.iterdir() if d.is_dir() and d.name in self.shape_params]
        
        print(f"Found {len(shape_dirs)} matching shape directories in {self.base_path}")
        
        for shape_dir in tqdm(shape_dirs, desc=\"Processing shapes\"):
            shape_name = shape_dir.name
            D = self.shape_params[shape_name]['D']
            H = self.shape_params[shape_name]['H']
            
            angle_dirs = [d for d in shape_dir.iterdir() if d.is_dir()]
            
            for angle_dir in angle_dirs:
                try:
                    angle = float(angle_dir.name)
                except ValueError:
                    continue
                    
                force_file = angle_dir / 'postProcessing' / 'cylinder' / '0' / 'forceCoeffs.dat'
                
                if not force_file.exists():
                    continue
                    
                # Extract U_ref from header OR directory name
                u_ref, lRef = self.extract_metadata_from_header(force_file)
                
                # Override U_ref for low/medium speed cases if header is wrong
                if 'baseline_lowU' in shape_name:
                    u_ref = 5.0
                elif 'baseline_mediumU' in shape_name:
                    u_ref = 10.0
                elif u_ref is None:
                    print(f"Could not extract U_ref from {force_file}")
                    continue
                    
                reader = ForceCoeffsReader(force_file)
                df = reader.read()
                if df is None or len(df) < 100:
                    continue
                    
                # Process
                # Skip first 20% as settling time
                t_start = df['time'].iloc[-1] * 0.2
                processor = ForceProcessor(df, t_start=t_start)
                
                # Filter and Downsample
                df_filtered = processor.filter_and_downsample(fs_target=100.0, f_cutoff=40.0)
                # Update processor with filtered data
                processor = ForceProcessor(df_filtered) # t_start already applied
                
                stats = processor.compute_statistics()
                
                # Compute Strouhal Grid PSD
                # Define Strouhal grid: 0.05 to 5.0, 128 points
                st_grid = np.linspace(0.05, 5.0, 128)
                log_psd_st = processor.compute_psd_strouhal(u_ref, D, st_grid, col='cl')
                
                # Find peak Strouhal from the grid
                peak_idx = np.argmax(log_psd_st)
                st_peak = st_grid[peak_idx]
                f_peak = st_peak * u_ref / D
                
                case_id = f\"{shape_name}_{int(angle)}\"
                
                entry = {
                    'case_id': case_id,
                    'shape': shape_name,
                    'angle': angle,
                    'U_ref': u_ref,
                    'D': D,
                    'H': H,
                    'lRef': lRef,
                    'st_peak': st_peak,
                    'f_peak': f_peak,
                    **stats
                }
                summary_data.append(entry)
                
                time_series_data[case_id] = {
                    'time': df_filtered['time'].values,
                    'cl': df_filtered['cl'].values,
                    'cd': df_filtered['cd'].values,
                    'cm': df_filtered['cm'].values,
                    'st_grid': st_grid,
                    'log_psd': log_psd_st,
                    'metadata': {
                        'shape': shape_name,
                        'angle': angle,
                        'U_ref': u_ref,
                        'D': D,
                        'H': H
                    }
                }
                
        return pd.DataFrame(summary_data), time_series_data
"""
    nb['cells'][adp_idx]['source'] = new_adp_code.splitlines(keepends=True)
    print("Updated AeroelasticDataProcessor.")

# 3. Update create_improved_ml_dataset
ds_idx = find_cell_index(nb, "def create_improved_ml_dataset")
if ds_idx != -1:
    new_ds_code = """def create_improved_ml_dataset(summary_df, time_series_data):
    \"\"\"Create ML dataset with physics-based features and normalization\"\"\"
    
    # 1. Feature Engineering
    summary_df['phi'] = np.arctan(2 * summary_df['H'] / summary_df['D'])
    summary_df['Re'] = (summary_df['U_ref'] * summary_df['D']) / NU
    summary_df['alpha0'] = np.radians(summary_df['angle'] - 90.0)
    
    # 2. Prepare X and y
    X = summary_df[['phi', 'Re', 'alpha0']].values
    
    # Compute Normalization Statistics for X
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    # Avoid division by zero if std is 0
    X_std[X_std == 0] = 1.0
    
    y = summary_df[['mean_cd', 'mean_cl', 'mean_cm', 'st_peak']].values
    
    case_ids = summary_df['case_id'].values
    
    # 3. Save to NPZ (ML ready)
    npz_path = os.path.join(PROCESSED_DIR, "improved_ml_dataset.npz")
    np.savez(
        npz_path,
        X=X,
        y=y,
        X_mean=X_mean,
        X_std=X_std,
        case_ids=case_ids,
        feature_names=['phi', 'Re', 'alpha0'],
        target_names=['mean_cd', 'mean_cl', 'mean_cm', 'st_peak']
    )
    print(f"Saved ML dataset to {npz_path}")
    
    # 4. Save Time Series and Spectral Data to H5
    h5_path = os.path.join(PROCESSED_DIR, "improved_psd_dataset.h5")
    with h5py.File(h5_path, 'w') as f:
        for case_id in case_ids:
            grp = f.create_group(case_id)
            ts_data = time_series_data[case_id]
            
            grp.create_dataset('time', data=ts_data['time'])
            grp.create_dataset('cl', data=ts_data['cl'])
            grp.create_dataset('cd', data=ts_data['cd'])
            grp.create_dataset('cm', data=ts_data['cm'])
            grp.create_dataset('st_grid', data=ts_data['st_grid'])
            grp.create_dataset('log_psd', data=ts_data['log_psd'])
            
            for k, v in ts_data['metadata'].items():
                grp.attrs[k] = v
                
    print(f"Saved H5 dataset to {h5_path}")
    
    # Save summary CSV
    csv_path = os.path.join(PROCESSED_DIR, "preprocessing_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved summary to {csv_path}")
"""
    nb['cells'][ds_idx]['source'] = new_ds_code.splitlines(keepends=True)
    print("Updated create_improved_ml_dataset.")

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=2)
