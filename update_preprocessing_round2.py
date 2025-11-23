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

# Update AeroelasticDataProcessor.scan_and_process to use log-spaced grid
adp_idx = find_cell_index(nb, "class AeroelasticDataProcessor:")
if adp_idx != -1:
    # We replace the whole class to ensure the change is applied correctly
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
                # Define Log-spaced Strouhal grid: 0.05 to 5.0, 128 points
                st_grid = np.logspace(np.log10(0.05), np.log10(5.0), 128)
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
    print("Updated AeroelasticDataProcessor with Log-spaced Strouhal grid.")

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=2)
