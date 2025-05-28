import os
import matplotlib.pyplot as plt

# Parsed macro avg F1-scores for each dataset and model from the uploaded files
data = {
    "utlc_movies": {
        "LLM-Zero-Shot": 0.736134,
        "BERTimbau (Zero-Shot)": 0.427783,
        "XLM-R (Zero-Shot)": 0.175482,
        "BERTimbau (Pseudo-Label)": 0.720445,
        "XLM-R (Pseudo-Label)": 0.703931,
        "BERTimbau (Supervised)": 0.812299,
        "XLM-R (Supervised)": 0.468904
    },
    "utlc_apps": {
        "LLM-Zero-Shot": 0.862276,
        "BERTimbau (Zero-Shot)": 0.459281,
        "XLM-R (Zero-Shot)": 0.531341,
        "BERTimbau (Pseudo-Label)": 0.860732,
        "XLM-R (Pseudo-Label)": 0.861297,
        "BERTimbau (Supervised)": 0.891292,
        "XLM-R (Supervised)": 0.891871
    },
    "olist": {
        "LLM-Zero-Shot": 0.916081,
        "BERTimbau (Zero-Shot)": 0.607275,
        "XLM-R (Zero-Shot)": 0.326046,
        "BERTimbau (Pseudo-Label)": 0.914251,
        "XLM-R (Pseudo-Label)": 0.915931,
        "BERTimbau (Supervised)": 0.932754,
        "XLM-R (Supervised)": 0.930621
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