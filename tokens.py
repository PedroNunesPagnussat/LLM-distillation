import pandas as pd
import tiktoken
from pathlib import Path
import numpy as np

# Get base directory of the current script
BASE_DIR = Path(__file__).resolve().parent

# Define subdirectories relative to the script file
TRAIN_PREFIX = BASE_DIR / 'data' / 'train'
TEST_PREFIX = BASE_DIR / 'data' / 'test'
SUMMARY_PREFIX = BASE_DIR / 'data' / 'summary'
TOKEN_SUMMARY_FILE = SUMMARY_PREFIX / "token_counts.txt"

# Ensure summary directory exists
SUMMARY_PREFIX.mkdir(parents=True, exist_ok=True)

# Datasets to analyze
DATASETS = ['utlc_movies', 'utlc_apps', 'olist']

# GPT-4.1 Nano uses cl100k_base encoding
ENCODING_NAME = "cl100k_base"  # This is the encoding used by GPT-4.1 Nano

def count_tokens(dataset_name, split, encoding):
    """
    Count tokens in a dataset using the specified encoding
    
    Args:
        dataset_name: Name of the dataset (olist, utlc_apps, utlc_movies)
        split: Data split (train or test)
        encoding: The tiktoken encoding object
        
    Returns:
        Dictionary with token count statistics
    """
    # Determine file path based on split
    if split == 'train':
        file_path = TRAIN_PREFIX / f"{dataset_name}.csv"
    else:
        file_path = TEST_PREFIX / f"{dataset_name}.csv"
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Get the review texts
    texts = df['review_text'].tolist()
    
    # Tokenize all texts
    token_counts = []
    for text in texts:
        # Handle NaN values
        if pd.isna(text):
            token_counts.append(0)
            continue
            
        # Tokenize the text
        tokens = encoding.encode(text)
        token_counts.append(len(tokens))
    
    # Calculate statistics
    stats = {
        'dataset': dataset_name,
        'split': split,
        'total_tokens': sum(token_counts)
    }
    
    return stats

def main():
    """
    Main function to calculate token counts for all datasets using GPT-4.1 Nano tokenizer
    """
    all_stats = []
    
    # Load the GPT-4.1 Nano tokenizer (cl100k_base)
    print(f"Loading GPT-4.1 Nano tokenizer (encoding: {ENCODING_NAME})")
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    
    # Process each dataset and split
    for dataset in DATASETS:
        for split in ['train', 'test']:
            print(f"Processing {dataset} ({split})...")
            stats = count_tokens(dataset, split, encoding)
            all_stats.append(stats)
                
            # Print summary
            print(f"  Dataset: {stats['dataset']}, Split: {stats['split']}")
            print(f"  Total tokens: {stats['total_tokens']}")
            print()
    
    # Save results to file
    with open(TOKEN_SUMMARY_FILE, 'w') as f:
        f.write("GPT-4.1 Nano Token Count Summary\n")
        f.write("==============================\n\n")
        
        # Group by dataset and split
        for dataset in DATASETS:
            f.write(f"Dataset: {dataset}\n")
            f.write("-" * 50 + "\n")
            
            for split in ['train', 'test']:
                f.write(f"\n{split.upper()} SET:\n\n")
                
                # Find the stats for this combination
                for stats in all_stats:
                    if stats['dataset'] == dataset and stats['split'] == split:
                        f.write(f"  Total tokens: {stats['total_tokens']}\n")
                        f.write("\n")
            
            f.write("\n")
    
    print(f"Token count summary saved to {TOKEN_SUMMARY_FILE}")

if __name__ == "__main__":
    main()