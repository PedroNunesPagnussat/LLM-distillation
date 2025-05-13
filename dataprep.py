import pandas as pd
from sklearn.model_selection import train_test_split
import os
from loguru import logger

SEED = 42  # Reproducibility

# Prefixes (internal only)
DATA_PREFIX = 'data/raw/'
TRAIN_PREFIX = 'data/train/'
TEST_PREFIX = 'data/test/'
EXAMPLE_PREFIX = 'data/example/'

COLUMNS_TO_KEEP = ['original_index', 'review_text', 'polarity']

def load_and_clean_dataset(file_path):
    logger.info(f"Loading dataset: {file_path}")
    df = pd.read_csv(file_path)

    logger.info("Cleaning dataset")
    df = df[COLUMNS_TO_KEEP]
    df.drop_duplicates(subset=['original_index'], inplace=True)
    df.dropna(subset=['review_text', 'polarity'], inplace=True)
    df = df[df['review_text'].str.strip() != '']
    return df

def save_dataframe(df, path, name):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"[{name}] Saved to {path}")

def create_example_set(df, n_per_class=5):
    pos = df[df['polarity'] == 1].sample(n=n_per_class, random_state=SEED)
    neg = df[df['polarity'] == 0].sample(n=n_per_class, random_state=SEED)
    return pd.concat([pos, neg]).sample(frac=1, random_state=SEED).reset_index(drop=True)

def process_dataset(filename, rows=10_000):
    file_path = os.path.join(DATA_PREFIX, filename)
    name = filename.replace(".csv", "")

    train_path = os.path.join(TRAIN_PREFIX, f"{name}.csv")
    test_path = os.path.join(TEST_PREFIX, f"{name}.csv")
    example_path = os.path.join(EXAMPLE_PREFIX, f"{name}.csv")

    df = load_and_clean_dataset(file_path)

    required_size = 2 * rows + 10
    if len(df) < required_size:
        raise ValueError(f"[{name}] Not enough samples (required={required_size}, available={len(df)})")

    train_df, temp_df = train_test_split(df, train_size=rows, shuffle=True, random_state=SEED)
    test_df, remaining_df = train_test_split(temp_df, train_size=rows, shuffle=True, random_state=SEED)
    example_df = create_example_set(remaining_df)

    logger.success(f"[{name}] Train: {len(train_df)} | Test: {len(test_df)} | Example: {len(example_df)}")
    logger.info(f"[{name}] Train distribution:\n{train_df['polarity'].value_counts()}")
    logger.info(f"[{name}] Test distribution:\n{test_df['polarity'].value_counts()}")
    logger.info(f"[{name}] Example distribution:\n{example_df['polarity'].value_counts()}")

    save_dataframe(train_df, train_path, f"{name} - Train")
    save_dataframe(test_df, test_path, f"{name} - Test")
    save_dataframe(example_df, example_path, f"{name} - Example")

    return train_df, test_df, example_df

# Example usage
process_dataset('utlc_apps.csv')
process_dataset('utlc_movies.csv')
process_dataset('olist.csv')
