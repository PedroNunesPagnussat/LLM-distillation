import os
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
from datasets import Dataset
from sklearn.metrics import classification_report
from loguru import logger
import torch
import gc

from pathlib import Path

# Get base directory of the current script
BASE_DIR = Path(__file__).resolve().parent

# Define subdirectories relative to the script file
TRAIN_PREFIX = BASE_DIR / 'data' / 'train'
TEST_PREFIX = BASE_DIR / 'data' / 'test'
RESULTS_PREFIX = BASE_DIR / 'data' / 'results' / 'gt_bert'
SUMMARY_PREFIX = BASE_DIR / 'data' / 'summaries'
OUTPUT_PREFIX = BASE_DIR / 'data' / 'hf_dir'

# Ensure directories exist
RESULTS_PREFIX.mkdir(parents=True, exist_ok=True)
SUMMARY_PREFIX.mkdir(parents=True, exist_ok=True)
OUTPUT_PREFIX.mkdir(parents=True, exist_ok=True)

LABEL2ID = {"negativo": 0, "positivo": 1}
ID2LABEL = {0: "negativo", 1: "positivo"}

MAX_LENGTH = 500
BATCH_SIZE = 8
EPOCHS = 5
SEED = 42

set_seed(SEED)

MODELS = {
    "XLM-R": "xlm-roberta-base",
    "BERTimbau": "neuralmind/bert-base-portuguese-cased",
    
}
DATASETS = ['utlc_movies', 'utlc_apps', 'olist']

def load_and_prepare(dataset_name, tokenizer):
    def preprocess_function(examples):
        return tokenizer(
            examples["review_text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
    train_path = os.path.join(TRAIN_PREFIX, f"{dataset_name}.csv")
    test_path = os.path.join(TEST_PREFIX, f"{dataset_name}.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_df["label"] = train_df["polarity"].astype(int)
    test_df["label"] = test_df["polarity"].astype(int)
    train_ds = Dataset.from_pandas(train_df[["review_text", "label"]])
    test_ds = Dataset.from_pandas(test_df[["review_text", "label"]])
    train_ds = train_ds.map(preprocess_function, batched=True)
    test_ds = test_ds.map(preprocess_function, batched=True)
    return train_ds, test_ds, test_df

def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids
    report = classification_report(
        labels, preds,
        target_names=[ID2LABEL[0], ID2LABEL[1]],
        output_dict=True
    )
    return {"accuracy": report["accuracy"]}

def train_and_evaluate(model_name, model_checkpoint, dataset_name):
    logger.info(f"Training {model_name} on {dataset_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    
    train_ds, test_ds, test_df = load_and_prepare(dataset_name, tokenizer)
    training_args = TrainingArguments(
        output_dir=f"./{OUTPUT_PREFIX}_gt_bert_{model_name}_{dataset_name}",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        save_strategy="no",
        seed=SEED,
        load_best_model_at_end=False,
        # logging_steps=100,
        disable_tqdm=False,
        # fp16=True,
        # report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    logger.success(f"Finished training {model_name} on {dataset_name}")

    preds = trainer.predict(test_ds)
    pred_labels = preds.predictions.argmax(-1)
    true_labels = preds.label_ids

    # Save detailed classification report (CSV)
    report = classification_report(
        true_labels, pred_labels,
        target_names=[ID2LABEL[0], ID2LABEL[1]],
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    base = f"{dataset_name}_{model_name}"
    report_path = os.path.join(RESULTS_PREFIX, f"{base}_report.csv")
    report_df.to_csv(report_path)
    logger.info(f"Saved classification report: {report_path}")

    # Save predictions (CSV)
    test_df["predicted"] = pred_labels
    test_df["predicted_label"] = test_df["predicted"].map(ID2LABEL)
    test_df["true_label"] = test_df["label"].map(ID2LABEL)
    preds_path = os.path.join(RESULTS_PREFIX, f"{base}_predictions.csv")
    test_df.to_csv(preds_path, index=False)
    logger.info(f"Saved predictions: {preds_path}")

    return report

def main():
    summary = {}
    for dataset in DATASETS:
        summary[dataset] = {}
        for model_name, model_ckpt in MODELS.items():
            report = train_and_evaluate(model_name, model_ckpt, dataset)
            summary[dataset][model_name] = report
            gc.collect()
            torch.cuda.empty_cache()

    # Save overall summary
    summary_path = os.path.join(SUMMARY_PREFIX, "gt_bert_summary.txt")
    with open(summary_path, "w") as f:
        for dataset in summary:
            for model in summary[dataset]:
                f.write(f"Dataset: {dataset} | Model: {model}\n")
                f.write(pd.DataFrame(summary[dataset][model]).to_string())
                f.write("\n\n")
    print("All done!")

if __name__ == "__main__":
    main()
