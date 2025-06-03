import pandas as pd
from sklearn.model_selection import train_test_split
import os
from loguru import logger
from transformers import AutoTokenizer

SEED = 42  # Reproducibility

# Prefixes (internal only)
DATA_PREFIX = 'data/raw/'
TRAIN_PREFIX = 'data/train/'
TEST_PREFIX = 'data/test/'

COLUMNS_TO_KEEP = ['original_index', 'review_text', 'polarity']

TRAIN_ROWS = 10_000
TEST_ROWS = 50_000



# Load tokenizer (BERTimbau)
BERT_MODEL = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
MAX_TOKENS = 500

def load_and_clean_dataset(file_path):
    logger.info(f"Loading dataset: {file_path}")
    df = pd.read_csv(file_path)


    # Drop NaN polarity values

    logger.info(f"Before cleaning: {len(df)} samples")

    df = df.dropna(subset=['polarity'])    
    logger.info(f"After dropping NaN polarity: {len(df)} samples")


    df = df[df['rating'].isin([0, 1, 4, 5])]
    logger.info(f"After filtering ratings: {len(df)} samples")


    mask = (df['polarity'] == 1) & (df['rating'].isin([0, 1]))
    df = df[~mask]
    mask = (df['polarity'] == 0) & (df['rating'].isin([4, 5]))
    df = df[~mask]
    logger.info(f"After removing polarity-rating mismatches: {len(df)} samples")


    df = df.dropna(subset=['review_text'])
    df = df[df['review_text'].str.strip() != '']
    logger.info(f"After dropping NaN review_text: {len(df)} samples")

    df['review_text'] = df['review_text'].str.replace('\n', ' ', regex=False)
    df['review_text'] = df['review_text'].str.strip()
    df['review_text'] = df['review_text'].str.replace('"', '', regex=False)
    df['review_text'] = df['review_text'].str.replace("'", '', regex=False)

    logger.info(f"After initial cleaning: {len(df)} samples")


    logger.info("Cleaning dataset")
    df = df[COLUMNS_TO_KEEP]
    df.drop_duplicates(subset=['original_index'], inplace=True)
    df.dropna(subset=['review_text', 'polarity'], inplace=True)
    df = df[df['review_text'].str.strip() != '']

    # Token length filtering
    logger.info("Filtering samples by BERT token length <= 512")
    df['num_tokens'] = df['review_text'].apply(lambda x: len(tokenizer(x, truncation=False, add_special_tokens=True)['input_ids']))
    before = len(df)
    df = df[df['num_tokens'] <= MAX_TOKENS].copy()
    after = len(df)
    logger.info(f"Removed {before - after} samples over token limit")

    print(f"Final size for {file_path}: {len(df)} samples")


    return df.drop(columns=['num_tokens'])

def save_dataframe(df, path, name):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"[{name}] Saved to {path}")

def process_dataset(filename):


    file_path = os.path.join(DATA_PREFIX, filename)
    name = filename.replace(".csv", "")

    train_path = os.path.join(TRAIN_PREFIX, f"{name}.csv")
    test_path = os.path.join(TEST_PREFIX, f"{name}.csv")

    df = load_and_clean_dataset(file_path)
    
    
    test_rows = min(TEST_ROWS, len(df) - TRAIN_ROWS)
    train_df, test_df = train_test_split(df, train_size=TRAIN_ROWS, test_size=test_rows, shuffle=True, random_state=SEED)
    

    logger.success(f"[{name}] Train: {len(train_df)} | Test: {len(test_df)}")

    logger.info(f"[{name}] Train distribution:\n{train_df['polarity'].value_counts()}")
    logger.info(f"[{name}] Train distribution:\n{train_df['polarity'].value_counts(normalize=True) * 100}")

    logger.info(f"[{name}] Test distribution:\n{test_df['polarity'].value_counts()}")
    logger.info(f"[{name}] Test distribution:\n{test_df['polarity'].value_counts(normalize=True) * 100}")

    logger.info(f"[{name}] Test distribution:\n{test_df['polarity'].value_counts()}")

    return
    save_dataframe(train_df, train_path, f"{name} - Train")
    save_dataframe(test_df, test_path, f"{name} - Test")

    return train_df, test_df

# Example usage
process_dataset('utlc_apps.csv')
process_dataset('utlc_movies.csv')
process_dataset('olist.csv')
