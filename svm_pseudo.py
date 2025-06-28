import os
import random
from pathlib import Path
from time import time

import pandas as pd
import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from loguru import logger

# === CONFIG ===
BASE_DIR = Path(__file__).resolve().parent
PSEUDO_LABELS_PREFIX = BASE_DIR / 'data' / 'pseudo_labels'
TEST_PREFIX = BASE_DIR / 'data' / 'test'
RESULTS_PREFIX = BASE_DIR / 'data' / 'results' / 'pseudo_label_bert_svm'
SUMMARY_FILE = BASE_DIR / 'data' / 'summary' / 'pseudo_label_bert_svm_summary.txt'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 512
SEED = 42
BATCH_SIZE = 32

DATASETS = ['utlc_movies', 'utlc_apps', 'olist']
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

LABEL2ID = {"negativo": 0, "positivo": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

RESULTS_PREFIX.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)

# === REPRODUCIBILITY ===
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)

# === EMBEDDING FUNCTION ===
@torch.no_grad()
def compute_embeddings(texts, tokenizer, model, batch_size=BATCH_SIZE):
    model.eval()
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH).to(DEVICE)
        outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.extend(cls_embeddings)
    return embeddings

# === PIPELINE ===
def train_and_evaluate(dataset_name, summary_file):
    logger.info(f"Processing {dataset_name} (pseudo-labels)")
    start_time = time()

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    except Exception as e:
        logger.error(f"Error loading model {MODEL_NAME}: {e}")
        return

    try:
        train_df = pd.read_csv(PSEUDO_LABELS_PREFIX / f"{dataset_name}.csv")
        test_df = pd.read_csv(TEST_PREFIX / f"{dataset_name}.csv")

        train_texts = train_df["review_text"].tolist()
        train_labels = train_df["predicted"].astype(int).tolist()
        test_texts = test_df["review_text"].tolist()
        test_labels = test_df["polarity"].astype(int).tolist()
    except Exception as e:
        logger.error(f"Error loading data for {dataset_name}: {e}")
        return

    # Compute embeddings
    X_train = compute_embeddings(train_texts, tokenizer, model)
    X_test = compute_embeddings(test_texts, tokenizer, model)

    # Train SVM (no StandardScaler)
    clf = SVC(kernel='linear', random_state=SEED)
    clf.fit(X_train, train_labels)

    # Predict and evaluate
    pred_labels = clf.predict(X_test)
    report = classification_report(
        test_labels,
        pred_labels,
        target_names=[ID2LABEL[0], ID2LABEL[1]],
        output_dict=True
    )

    # Save report
    report_df = pd.DataFrame(report).transpose()
    base = f"{dataset_name}_BERTimbau_SVM_Pseudo"
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
    summary_file.write(f"Dataset: {dataset_name} | Model: BERTimbau + SVM (Pseudo-Labels, No Scaling)\n")
    summary_file.write(report_df.to_string())
    summary_file.write("\n\n")
    summary_file.flush()
    os.fsync(summary_file.fileno())

    logger.info(f"Finished {dataset_name} in {time() - start_time:.2f}s")

# === MAIN LOOP ===
def main():
    with open(SUMMARY_FILE, "w") as summary_file:
        for dataset in DATASETS:
            train_and_evaluate(dataset, summary_file)
    print("All done!")

if __name__ == "__main__":
    main()
