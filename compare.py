import os
import matplotlib.pyplot as plt

# Parsed macro avg F1-scores for each dataset and model from the uploaded files
data = {
    "utlc_movies": {
        "XLM-R": 0.491275,
        "BERTimbau": 0.798253,
        "XLM-R (Pseudo-Label)": 0.560403,
        "BERTimbau (Pseudo-Label)": 0.613812,
        "XLM-R (Zero-Shot)": 0.134051,
        "BERTimbau (Zero-Shot)": 0.334423,
        "LLM-Zero-Shot": 0.608376,
    },
    "utlc_apps": {
        "XLM-R": 0.894020,
        "BERTimbau": 0.903460,
        "XLM-R (Pseudo-Label)": 0.841860,
        "BERTimbau (Pseudo-Label)": 0.844206,
        "XLM-R (Zero-Shot)": 0.469912,
        "BERTimbau (Zero-Shot)": 0.397784,
        "LLM-Zero-Shot": 0.847595,
    },
    "olist": {
        "XLM-R": 0.935229,
        "BERTimbau": 0.938642,
        "XLM-R (Pseudo-Label)": 0.761831,
        "BERTimbau (Pseudo-Label)": 0.911609,
        "XLM-R (Zero-Shot)": 0.457853,
        "BERTimbau (Zero-Shot)": 0.345398,
        "LLM-Zero-Shot": 0.914261,
    }
}


os.makedirs("./results", exist_ok=True)

for dataset, models in data.items():
    # Sort models by F1 score descending
    sorted_models = sorted(models.items(), key=lambda x: x[1], reverse=True)
    models_names = [m[0] for m in sorted_models]
    f1_scores = [m[1] * 100 for m in sorted_models]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models_names, f1_scores, color="teal")
    plt.title(f"Macro F1-Score Comparison for {dataset} (Descending Order)")
    plt.ylabel("Macro F1-Score")
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{score:.3f}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(f"./results/{dataset}.png")