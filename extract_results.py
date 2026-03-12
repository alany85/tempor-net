import json
import glob
import re

for f in glob.glob("*.ipynb"):
    try:
        with open(f, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print(f"=== {f} ===")
            for i, cell in enumerate(data.get('cells', [])):
                if cell.get('cell_type') == 'code':
                    src = "".join(cell.get('source', []))
                    out = ""
                    for o in cell.get('outputs', []):
                        if o.get('output_type') == 'stream':
                            out += "".join(o.get('text', []))
                        elif o.get('output_type') == 'execute_result' or o.get('output_type') == 'display_data':
                            if 'text/plain' in o.get('data', {}):
                                out += "".join(o.get('data', {})['text/plain'])
                    
                    
                    if "Test Acc" in out or "test acc" in out.lower() or "Test accuracy:" in out:
                        for line in out.split('\n'):
                            if "Test" in line or "test" in line.lower():
                                print(f"Cell {i} Output: {line}")
                    
                    # Also look for train/val accuracies in source or variables printed
                    if "val_accs" in src or "train_accs" in src:
                        print(f"Cell {i} has val_accs/train_accs in source: {src[:50]}...")
                    if "[0." in out and (len(out) < 500 or "Epoch" in out):
                        lines = [l for l in out.split('\n') if "Epoch" in l or "[" in l]
                        if lines:
                            print(f"Cell {i} train log?: {lines[0][:100]}")
                            
    except Exception as e:
        print(e)
