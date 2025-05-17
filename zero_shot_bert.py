from transformers import pipeline
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm
import os
from pathlib import Path

# Paths


# Base directory of the script
BASE_DIR = Path(__file__).resolve().parent

# Updated paths
TEST_PREFIX = BASE_DIR / 'data' / 'test'
RESULTS_PREFIX = BASE_DIR / 'data' / 'results' / 'zero_shot'
SUMMARY_PREFIX = BASE_DIR / 'data' / 'summary'

# Ensure directories exist
RESULTS_PREFIX.mkdir(parents=True, exist_ok=True)
SUMMARY_PREFIX.mkdir(parents=True, exist_ok=True)
# Load test dataset
def load_test_dataset(name):
    path = os.path.join(TEST_PREFIX, f"{name}.csv")
    df = pd.read_csv(path)
    return df

# Define label mapping
LABELS = ["positivo", "negativo"]
LABEL_TO_INT = {"positivo": 1, "negativo": 0}

# Run zero-shot classification
def evaluate_zero_shot(model_name, dataset_name):
    print(f"Running {model_name} on {dataset_name}")
    classifier = pipeline("zero-shot-classification", model=model_name, device_map="auto")
    # classifier = pipeline("text-classification", model=model_name, device_map="auto")


    df = load_test_dataset(dataset_name)
    results = []

    for text in tqdm(df['review_text'].tolist(), desc=f"{model_name}-{dataset_name}"):
        output = classifier(text, candidate_labels=LABELS, truncation=True, max_length=512)
        predicted_label = output['labels'][0]
        results.append(LABEL_TO_INT[predicted_label])

    df['predicted'] = results
    report = classification_report(df['polarity'], df['predicted'], output_dict=True)
    
    # Save result file
    df.to_csv(os.path.join(RESULTS_PREFIX, f"{dataset_name}_{model_name.replace('/', '_')}.csv"), index=False)

    # Return classification report
    return report

if __name__ == "__main__":
    datasets = ['utlc_movies', 'utlc_apps', 'olist']
    models = {
        "neuralmind/bert-base-portuguese-cased": "BERTimbau",
        "xlm-roberta-base": "XLM-R"
    }

    all_reports = {}

    for dataset in datasets:
        all_reports[dataset] = {}
        for model_path, model_name in models.items():
            report = evaluate_zero_shot(model_path, dataset)
            all_reports[dataset][model_name] = report

    # Optionally save global report summary
    summary_path = os.path.join(SUMMARY_PREFIX, "zero_shot_summary.txt")
    with open(summary_path, 'w') as f:
        for dataset in all_reports:
            for model in all_reports[dataset]:
                f.write(f"Dataset: {dataset} | Model: {model}\n")
                f.write(pd.DataFrame(all_reports[dataset][model]).to_string())
                f.write("\n\n")

    print("Evaluation complete. Reports saved.")
