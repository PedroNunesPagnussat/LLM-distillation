import os
from pathlib import Path

import pandas as pd
import torch
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from loguru import logger

# === CONFIG ===
BASE_DIR = Path(__file__).resolve().parent
TRAIN_PREFIX = BASE_DIR / 'data' / 'train'
TEST_PREFIX = BASE_DIR / 'data' / 'test'
RESULTS_PREFIX = BASE_DIR / 'data' / 'results' / 'gt_bert_svm'
SUMMARY_FILE = BASE_DIR / 'data' / 'summary' / 'gt_bert_svm_summary.txt'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 512
SEED = 42

DATASETS = ['utlc_movies', 'utlc_apps', 'olist']
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

LABEL2ID = {"negativo": 0, "positivo": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

RESULTS_PREFIX.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)

# === EMBEDDING FUNCTION ===
@torch.no_grad()
def compute_embeddings(texts, tokenizer, model):
    model.eval()
    embeddings = []
    for text in tqdm(texts, desc="Embedding"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LENGTH)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        embeddings.append(cls_embedding)
    return embeddings

# === PIPELINE ===
def train_and_evaluate(dataset_name, summary_file):
    logger.info(f"Processing {dataset_name}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

    # Load data
    train_df = pd.read_csv(TRAIN_PREFIX / f"{dataset_name}.csv")
    test_df = pd.read_csv(TEST_PREFIX / f"{dataset_name}.csv")
    train_texts, train_labels = train_df["review_text"].tolist(), train_df["polarity"].astype(int).tolist()
    test_texts, test_labels = test_df["review_text"].tolist(), test_df["polarity"].astype(int).tolist()

    # Compute embeddings
    X_train = compute_embeddings(train_texts, tokenizer, model)
    X_test = compute_embeddings(test_texts, tokenizer, model)

    # Train SVM
    clf = SVC(kernel='linear', random_state=SEED)
    clf.fit(X_train, train_labels)

    # Predict and evaluate
    pred_labels = clf.predict(X_test)
    report = classification_report(test_labels, pred_labels, target_names=[ID2LABEL[0], ID2LABEL[1]], output_dict=True)

    # Save report
    report_df = pd.DataFrame(report).transpose()
    base = f"{dataset_name}_BERTimbau_SVM"
    report_path = RESULTS_PREFIX / f"{base}_report.csv"
    report_df.to_csv(report_path)
    logger.info(f"Saved classification report: {report_path}")

    # Save predictions
    test_df["predicted"] = pred_labels
    test_df["predicted_label"] = test_df["predicted"].map(ID2LABEL)
    test_df["true_label"] = test_df["polarity"].map(ID2LABEL)
    preds_path = RESULTS_PREFIX / f"{base}_predictions.csv"
    test_df.to_csv(preds_path, index=False)
    logger.info(f"Saved predictions: {preds_path}")

    # Summary output
    summary_file.write(f"Dataset: {dataset_name} | Model: BERTimbau + SVM\n")
    summary_file.write(report_df.to_string())
    summary_file.write("\n\n")
    summary_file.flush()
    os.fsync(summary_file.fileno())

# === MAIN LOOP ===
def main():
    with open(SUMMARY_FILE, "w") as summary_file:
        for dataset in DATASETS:
            train_and_evaluate(dataset, summary_file)
    print("All done!")

if __name__ == "__main__":
    main()
