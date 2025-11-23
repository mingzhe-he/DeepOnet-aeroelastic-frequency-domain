import json
import os
import numpy as np

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

# 1. Update AeroelasticDataProcessor.scan_and_process (Linear Grid)
adp_idx = find_cell_index(nb, "class AeroelasticDataProcessor:")
if adp_idx != -1:
    new_adp_code = """class AeroelasticDataProcessor:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        
        # Shape parameters (D=3.0m constant)
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
                
                # Override U_ref for low/medium speed cases
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
                t_start = df['time'].iloc[-1] * 0.2
                processor = ForceProcessor(df, t_start=t_start)
                
                # Filter and Downsample
                df_filtered = processor.filter_and_downsample(fs_target=100.0, f_cutoff=40.0)
                processor = ForceProcessor(df_filtered)
                
                stats = processor.compute_statistics()
                
                # Compute Strouhal Grid PSD (Linear Grid)
                # Reverting to linear grid as log-grid reduced performance
                st_grid = np.linspace(0.05, 5.0, 128)
                log_psd_st = processor.compute_psd_strouhal(u_ref, D, st_grid, col='cl')
                
                # Find peak
                peak_idx = np.argmax(log_psd_st)
                st_peak = st_grid[peak_idx]
                f_peak = st_peak * u_ref / D
                
                case_id = f\"{shape_name}_{int(angle)}\"
                
                # Compute New Features
                # alpha_apex (radians)
                alpha_apex = 2 * np.arctan(D / (2*H))
                
                # alpha0 (radians)
                alpha0 = np.radians(angle - 90)
                sin_alpha0 = np.sin(alpha0)
                cos_alpha0 = np.cos(alpha0)
                
                # Re
                Re = (u_ref * D) / NU
                
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
                    'alpha_apex': alpha_apex,
                    'alpha0': alpha0,
                    'sin_alpha0': sin_alpha0,
                    'cos_alpha0': cos_alpha0,
                    'Re': Re,
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

# 2. Update create_improved_ml_dataset to use new features
cimd_idx = find_cell_index(nb, "def create_improved_ml_dataset")
if cimd_idx != -1:
    new_cimd_code = """def create_improved_ml_dataset(summary_df, time_series_data):
    \"\"\"
    Create final NPZ and H5 datasets for ML
    \"\"\"
    print("Creating ML dataset...")
    
    # 1. Extract Features (X)
    # New Feature Set: [alpha_apex, Re, sin_alpha0, cos_alpha0]
    feature_cols = ['alpha_apex', 'Re', 'sin_alpha0', 'cos_alpha0']
    X = summary_df[feature_cols].values.astype(np.float32)
    
    # 2. Extract Targets (y) - Scalars
    target_cols = ['mean_cd', 'mean_cl', 'mean_cm', 'st_peak']
    y = summary_df[target_cols].values.astype(np.float32)
    
    # Compute Normalization Statistics for X
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    # Avoid division by zero
    X_std[X_std == 0] = 1.0
    
    case_ids = summary_df['case_id'].values
    
    # 3. Save to NPZ
    npz_path = os.path.join(PROCESSED_DIR, "improved_ml_dataset.npz")
    np.savez(
        npz_path,
        X=X,
        y=y,
        X_mean=X_mean,
        X_std=X_std,
        case_ids=case_ids,
        feature_names=feature_cols,
        target_names=target_cols
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
    
    return summary_df
"""
    nb['cells'][cimd_idx]['source'] = new_cimd_code.splitlines(keepends=True)
    print("Updated create_improved_ml_dataset.")

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=2)
