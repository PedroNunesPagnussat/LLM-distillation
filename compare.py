import os
import matplotlib.pyplot as plt
import numpy as np

# Constants
FIG_SIZE = (12, 6)
ANNOT_OFFSET = 1.5
os.makedirs("./results", exist_ok=True)

# Data (loaded from provided sources)
data = {
    "utlc_movies": {
        "BERTimbau (GT)": 0.798,
        "XLM-R (GT)": 0.491,
        "BERTimbau (Pseudo GPT-4.1 nano)": 0.614,
        "XLM-R (Pseudo GPT-4.1 nano)": 0.560,
        "BERTimbau (Pseudo llama3.2:3b)": 0.537,
        "XLM-R (Pseudo llama3.2:3b)": 0.547,
        "BERTimbau (Zero-Shot)": 0.334,
        "XLM-R (Zero-Shot)": 0.134,
        "LLM-Zero-Shot (GPT-4.1 nano)": 0.608,
        "LLM-Zero-Shot (llama3.2:3b)": 0.517,
    },
    "utlc_apps": {
        "BERTimbau (GT)": 0.903,
        "XLM-R (GT)": 0.894,
        "BERTimbau (Pseudo GPT-4.1 nano)": 0.844,
        "XLM-R (Pseudo GPT-4.1 nano)": 0.842,
        "BERTimbau (Pseudo llama3.2:3b)": 0.804,
        "XLM-R (Pseudo llama3.2:3b)": 0.814,
        "BERTimbau (Zero-Shot)": 0.398,
        "XLM-R (Zero-Shot)": 0.470,
        "LLM-Zero-Shot (GPT-4.1 nano)": 0.848,
        "LLM-Zero-Shot (llama3.2:3b)": 0.747,
    },
    "olist": {
        "BERTimbau (GT)": 0.939,
        "XLM-R (GT)": 0.935,
        "BERTimbau (Pseudo GPT-4.1 nano)": 0.912,
        "XLM-R (Pseudo GPT-4.1 nano)": 0.762,
        "BERTimbau (Pseudo llama3.2:3b)": 0.910,
        "XLM-R (Pseudo llama3.2:3b)": 0.892,
        "BERTimbau (Zero-Shot)": 0.345,
        "XLM-R (Zero-Shot)": 0.458,
        "LLM-Zero-Shot (GPT-4.1 nano)": 0.914,
        "LLM-Zero-Shot (llama3.2:3b)": 0.877,
    }
}

# Plotting function
def plot_bar(models, scores, title, filename, colors="steelblue"):
    plt.figure(figsize=FIG_SIZE)
    bars = plt.bar(models, scores, color=colors, width=0.65)
    plt.title(title)
    plt.ylabel("Macro F1-Score (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ANNOT_OFFSET,
            f"{score:.1f}",
            ha='center', va='bottom', fontsize=10
        )
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Generate sorted plots for each dataset
for dataset, models in data.items():
    sorted_models = sorted(models.items(), key=lambda x: x[1], reverse=True)
    models_names, f1_scores = zip(*sorted_models)
    plot_bar(
        models_names,
        [score * 100 for score in f1_scores],
        f"Macro F1-Score: All Models ({dataset})",
        f"./results/{dataset}_all_models_sorted.png"
    )

# Mean macro F1-score across datasets
all_models = set().union(*[m.keys() for m in data.values()])
mean_scores = {model: np.mean([data[d][model] for d in data if model in data[d]]) for model in all_models}
sorted_mean = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
models_names, mean_f1 = zip(*sorted_mean)

# Special color highlighting
bar_colors = [
    "gold" if "BERTimbau" in model else
    "dodgerblue" if "XLM-R" in model else
    "lightgray"
    for model in models_names
]

plot_bar(
    models_names,
    [score * 100 for score in mean_f1],
    "Mean Macro F1-Score Across Datasets (BERT Highlighted)",
    "./results/mean_macro_f1_highlighted.png",
    colors=bar_colors
)

# Additional plot highlighting LLM types
bar_colors_tagged = [
    "seagreen" if "GPT-4.1 nano" in model else
    "dodgerblue" if "llama3.2:3b" in model else
    "lightgray"
    for model in models_names
]

plot_bar(
    models_names,
    [score * 100 for score in mean_f1],
    "Mean Macro F1-Score Across Datasets (LLMs Highlighted by Model Type)",
    "./results/mean_macro_f1_llm_highlight_bytag.png",
    colors=bar_colors_tagged
)

# Separate BERT and LLM plots
bert_keys = [model for model in all_models if "BERTimbau" in model or "XLM-R" in model]
llm_keys = [model for model in all_models if "LLM" in model]

# BERT mean plot
bert_means = {k: mean_scores[k] * 100 for k in bert_keys}
sorted_bert = sorted(bert_means.items(), key=lambda x: x[1], reverse=True)
plot_bar(
    [k for k, _ in sorted_bert],
    [v for _, v in sorted_bert],
    "Mean Macro F1-Score: BERT Only",
    "./results/mean_macro_f1_bert_only.png"
)

# LLM mean plot
llm_means = {k: mean_scores[k] * 100 for k in llm_keys}
sorted_llm = sorted(llm_means.items(), key=lambda x: x[1], reverse=True)
plot_bar(
    [k for k, _ in sorted_llm],
    [v for _, v in sorted_llm],
    "Mean Macro F1-Score: LLM Only",
    "./results/mean_macro_f1_llm_only.png"
)
