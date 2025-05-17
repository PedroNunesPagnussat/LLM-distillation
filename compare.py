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

        # Find the line that starts with "f1-score"
        f1_line = None
        for line in block.split('\n'):
            if line.strip().startswith("f1-score"):
                f1_line = line
                break
        if not f1_line:
            continue

        # Extract all floats from the line (for all columns)
        numbers = [float(n) for n in re.findall(r"[-+]?\d*\.\d+|\d+", f1_line)]
        # The macro avg f1-score is always the 4th index (0-based) if the columns are:
        # 0.0 | 1.0 | accuracy | macro avg | weighted avg
        # So in "f1-score" row: 0, 1, 2, 3, 4  (macro avg = 3, weighted avg = 4)
        macro_f1 = numbers[3] if len(numbers) >= 4 else None

        rows.append({
            "dataset": dataset.strip(),
            "model": model.strip(),
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
