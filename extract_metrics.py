import json
import glob

output = []
for f in glob.glob("*.ipynb"):
    try:
        with open(f, 'r', encoding='utf-8') as file:
            data = json.load(file)
            output.append(f"=== {f} ===")
            for cell in data.get('cells', []):
                if cell.get('cell_type') == 'code':
                    src = "".join(cell.get('source', []))
                    if 'history' in src or 'metrics' in src or 'val_acc' in src or 'train_acc' in src or 'test_acc' in src or 'Test' in src:
                        for o in cell.get('outputs', []):
                            if o.get('output_type') == 'stream':
                                output.append("".join(o.get('text', [])))
                            elif o.get('output_type') == 'execute_result' or o.get('output_type') == 'display_data':
                                if 'text/plain' in o.get('data', {}):
                                    output.append("".join(o.get('data', {})['text/plain']))
    except Exception as e:
        output.append(str(e))

with open("metrics_summary.txt", "w", encoding='utf-8') as fw:
    fw.write("\n".join(output))
