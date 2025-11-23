import json
import os

notebook_path = "/Users/mingz/Projects/Original_Attempt_redeveloped/01_Preprocessing_Improved.ipynb"

with open(notebook_path, 'r') as f:
    nb = json.load(f)

code = ""
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        code += source + "\n\n"

# Save as python script
script_path = "/Users/mingz/Projects/Original_Attempt_redeveloped/01_Preprocessing_Improved.py"
with open(script_path, 'w') as f:
    f.write(code)

print(f"Converted notebook to {script_path}")
