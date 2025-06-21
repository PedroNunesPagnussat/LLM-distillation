import os
import pandas as pd
import requests

from sklearn.metrics import classification_report
from tqdm import tqdm
import time
import json
from pathlib import Path
from dotenv import load_dotenv
import csv

# load environment variables from .env file
load_dotenv()

# Base directory of the script
BASE_DIR = Path(__file__).resolve().parent

# Updated paths
TRAIN_PREFIX = BASE_DIR / 'data' / 'train'
TEST_PREFIX = BASE_DIR / 'data' / 'test'
RESULTS_PREFIX = BASE_DIR / 'data' / 'results' / 'ollama_zero_shot'
SUMMARY_PREFIX = BASE_DIR / 'data' / 'summary'
PSEUDO_LABELS_PREFIX = BASE_DIR / 'data' / 'pseudo_labels'
SUMMARY_FILE = SUMMARY_PREFIX / "ollama_zero_shot.txt"

# Ensure directories exist
RESULTS_PREFIX.mkdir(parents=True, exist_ok=True)
SUMMARY_PREFIX.mkdir(parents=True, exist_ok=True)
PSEUDO_LABELS_PREFIX.mkdir(parents=True, exist_ok=True)

# Define label mapping
LABELS = ["positivo", "negativo"]
LABEL_TO_INT = {"positivo": 1, "negativo": 0}
INT_TO_LABEL = {1: "positivo", 0: "negativo"}

# Model setup
OLLAMA_MODEL = "llama3.2:7b"  # Allow override if needed
OLLAMA_API_URL = "http://localhost:11434/api/chat"
exit()

SYSTEM_PROMPT = """
Você é um assistente especializado em análise de sentimento em português. Sua tarefa é analisar o sentimento de textos e responder somente em JSON, SEMPRE usando esta estrutura exata:

{
    "sentiment": "positivo" ou "negativo"
}

Nunca use outras palavras, categorias ou explicações. Apenas responda com "positivo" ou "negativo", sempre dentro do JSON especificado.
"""

def load_dataset(name, split="test"):
    prefix = TEST_PREFIX if split == "test" else TRAIN_PREFIX
    path = prefix / f"{name}.csv"
    df = pd.read_csv(path)
    return df

def create_sentiment_prompt(text):
    r =  f"""
    Análise de sentimento:

    Sua saída deverá ser em formato JSON com a seguinte estrutura:
    {{
        "sentiment": "positivo" ou "negativo"
    }}

    Classifique apenas como positivo ou negativo

    Texto: "{text}"
    """
    return r

def call_ollama_with_retry(prompt, max_retries=3, backoff_factor=2):
    for attempt in range(max_retries):
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT.strip()},
                    {"role": "user", "content": prompt.strip()},
                ],
                "options": {
                    "temperature": 0,
                    "format": "json"
                }
            }
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=90)
            response.raise_for_status()
            data = response.json()

            # Handle Ollama streaming responses (if present)
            if 'message' in data:
                content = data['message']['content']
            elif 'content' in data:
                content = data['content']
            else:
                content = str(data)

            # Parse JSON response
            try:
                json_response = json.loads(content)
                return json_response.get("sentiment", "").lower()
            except json.JSONDecodeError:
                if "positivo" in content.lower():
                    return "positivo"
                elif "negativo" in content.lower():
                    return "negativo"
                print(f"Unrecognized response format: {content}")
                return None

        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = backoff_factor ** attempt
                print(f"Ollama call failed: {e}. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"Ollama call failed after {max_retries} attempts: {e}")
                return None

def process_dataset(df, split="test"):
    results = []

    for text in tqdm(df['review_text'].tolist(), desc=f"Processing {split} data"):
        prompt = create_sentiment_prompt(text)
        response = call_ollama_with_retry(prompt)

        # Map the response to our label format
        if response == "positivo":
            predicted_label = 1
        elif response == "negativo":
            predicted_label = 0
        else:
            print(f"Unclear response: {response}. Defaulting to negative.")
            predicted_label = 0

        results.append(predicted_label)

    return results

def evaluate_llm_zero_shot(dataset_name, summary_file):
    print(f"Running LLM Zero-Shot on {dataset_name}", flush=True)

    # Process test data for evaluation
    test_df = load_dataset(dataset_name, "test")
    test_results = process_dataset(test_df, "test")
    test_df['predicted'] = test_results

    # Calculate metrics
    report = classification_report(test_df['polarity'], test_df['predicted'], output_dict=True)

    # Save test predictions
    result_filename = f"{dataset_name}_ollama_zero_shot.csv"
    test_df.to_csv(RESULTS_PREFIX / result_filename, index=False)

    # Write individual report to summary file
    report_df = pd.DataFrame(report).transpose()
    summary_file.write(f"Dataset: {dataset_name} | Model: LLM-Zero-Shot\n")
    summary_file.write(report_df.to_string())
    summary_file.write("\n\n")
    summary_file.flush()
    os.fsync(summary_file.fileno())

    # Process train data for pseudo-labels
    train_df = load_dataset(dataset_name, "train")
    train_results = process_dataset(train_df, "train")
    train_df['predicted'] = train_results
    train_df['predicted_label'] = [INT_TO_LABEL[label] for label in train_results]

    # Save pseudo-labels
    pseudo_labels_filename = f"{dataset_name}.csv"
    train_df.to_csv(PSEUDO_LABELS_PREFIX / pseudo_labels_filename, index=False)

    return report

if __name__ == "__main__":
    datasets = ['utlc_movies', 'utlc_apps', 'olist']
    with open(SUMMARY_FILE, "w") as summary_file:
        for dataset in datasets:
            evaluate_llm_zero_shot(dataset, summary_file)

    print("Evaluation complete. Reports saved.")
