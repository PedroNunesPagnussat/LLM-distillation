from transformers import pipeline
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm
import os
from pathlib import Path

# Base directory of the script
BASE_DIR = Path(__file__).resolve().parent

# Updated paths
TEST_PREFIX = BASE_DIR / 'data' / 'test'
RESULTS_PREFIX = BASE_DIR / 'data' / 'results' / 'zero_shot'
SUMMARY_PREFIX = BASE_DIR / 'data' / 'summary'
SUMMARY_FILE = SUMMARY_PREFIX / "zero_shot.txt"

# Ensure directories exist
RESULTS_PREFIX.mkdir(parents=True, exist_ok=True)
SUMMARY_PREFIX.mkdir(parents=True, exist_ok=True)

# Load test dataset
def load_test_dataset(name):
    path = TEST_PREFIX / f"{name}.csv"
    df = pd.read_csv(path)
    return df

# Define label mapping
LABELS = ["positivo", "negativo"]
LABEL_TO_INT = {"positivo": 1, "negativo": 0}

# Run zero-shot classification
def evaluate_zero_shot(model_path, model_name, dataset_name, summary_file):
    print(f"Running {model_name} on {dataset_name}", flush=True)
    classifier = pipeline("zero-shot-classification", model=model_path, device_map="auto")

    df = load_test_dataset(dataset_name)
    results = []

    for text in tqdm(df['review_text'].tolist(), desc=f"{model_name}-{dataset_name}"):
        output = classifier(text, candidate_labels=LABELS, truncation=True, max_length=512)
        predicted_label = output['labels'][0]
        results.append(LABEL_TO_INT[predicted_label])

    df['predicted'] = results
    report = classification_report(df['polarity'], df['predicted'], output_dict=True)

    # Save predictions
    result_filename = f"{dataset_name}_{model_name}.csv"
    df.to_csv(RESULTS_PREFIX / result_filename, index=False)

    # Write individual report immediately to summary file
    report_df = pd.DataFrame(report).transpose()
    summary_file.write(f"Dataset: {dataset_name} | Model: {model_name}\n")
    summary_file.write(report_df.to_string())
    summary_file.write("\n\n")
    summary_file.flush()
    os.fsync(summary_file.fileno())

    return report

if __name__ == "__main__":
    datasets = ['utlc_movies', 'utlc_apps', 'olist']
    models = {
        "neuralmind/bert-base-portuguese-cased": "BERTimbau",
        "xlm-roberta-base": "XLM-R"
    }

    with open(SUMMARY_FILE, "w") as summary_file:
        for dataset in datasets:
            for model_path, model_name in models.items():
                evaluate_zero_shot(model_path, model_name, dataset, summary_file)

    print("Evaluation complete. Reports saved.")
