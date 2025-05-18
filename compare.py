import os
import glob
import re
import pandas as pd

SUMMARY_DIR = "data/summary/"   # adjust as needed

def parse_report(report_path, method):
    rows = []
    with open(report_path, "r") as f:
        content = f.read()

    # Split per dataset/model block
    blocks = re.split(r'Dataset:\s*', content)
    for block in blocks[1:]:
        header = block.split('\n', 1)[0]
        if "|" in header:
            dataset, model = [h.replace('Model:', '').strip() for h in header.split("|")]
        else:
            continue

        # Find the "macro avg" line
        macro_line = None
        for line in block.split('\n'):
            if line.strip().startswith("macro avg"):
                macro_line = line
                break
        if not macro_line:
            continue

        # Extract all floats from the macro avg line
        numbers = [float(n) for n in re.findall(r"[-+]?\d*\.\d+|\d+", macro_line)]
        # "macro avg" line format: precision, recall, f1-score, support
        macro_f1 = numbers[2] if len(numbers) >= 3 else None

        rows.append({
            "dataset": dataset,
            "model": model,
            "method": method,
            "macro_f1": macro_f1
        })
    return rows

def main():
    results = []
    txt_files = glob.glob(os.path.join(SUMMARY_DIR, "*.txt"))
    print(f"Found {len(txt_files)} report files in {SUMMARY_DIR}")

    for path in txt_files:
        method = os.path.splitext(os.path.basename(path))[0]
        print(f"Parsing {path} for method {method}")
        results += parse_report(path, method)

    df = pd.DataFrame(results)

    print("\nMacro-avg F1-score comparison (pivoted):\n")
    print(df.pivot_table(index=["dataset", "model"], columns="method", values="macro_f1"))
    # Optionally, save
    df.to_csv(os.path.join(SUMMARY_DIR, "macro_f1_summary.csv"), index=False)

if __name__ == "__main__":
    main()
