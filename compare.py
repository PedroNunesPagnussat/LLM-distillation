import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Constants
FIG_SIZE = (12, 6)
ANNOT_OFFSET = 1.5
os.makedirs("./results", exist_ok=True)

# Updated data dictionary
data = {
    "utlc_movies": {
        "BERTimbau (GT)": 0.798253,
        "XLM-R (GT)": 0.491275,
        "SVM(GT)": 0.699057,
        "GPT 4.1 Nano (Zero-Shot)": 0.608376,
        "LLama3.2:3b (Zero-Shot)": 0.516872,
        "BERTimbau (GPT 4.1 Nano Labels)": 0.613812,
        "XLM-R (GPT 4.1 Nano Labels)": 0.560403,
        "SVM (GPT 4.1 Nano Labels)": 0.588023,
        "BERTimbau (LLama3.2:3b Labels)": 0.536916,
        "XLM-R (LLama3.2:3b Labels)": 0.547359,
        "BERTimbau (Zero-Shot)": 0.334423,
        "XLM-R (Zero-Shot)": 0.134051
    },
    "utlc_apps": {
        "BERTimbau (GT)": 0.90346,
        "XLM-R (GT)": 0.89402,
        "SVM (GT)": 0.867432,
        "GPT 4.1 Nano (Zero-Shot)": 0.847595,
        "LLama3.2:3b (Zero-Shot)": 0.74655,
        "BERTimbau (GPT 4.1 Nano Labels)": 0.844206,
        "XLM-R (GPT 4.1 Nano Labels)": 0.84186,
        "SVM (GPT 4.1 Nano Labels)": 0.830479,
        "BERTimbau (LLama3.2:3b Labels)": 0.756239,
        "XLM-R (LLama3.2:3b Labels)": 0.763973,
        "BERTimbau (Zero-Shot)": 0.397784,
        "XLM-R (Zero-Shot)": 0.469912
    },
    "olist": {
        "BERTimbau (GT)": 0.938642,
        "XLM-R (GT)": 0.935229,
        "SVM (GT)": 0.921072,
        "GPT 4.1 Nano (Zero-Shot)": 0.914261,
        "LLama3.2:3b (Zero-Shot)": 0.876507,
        "BERTimbau (GPT 4.1 Nano Labels)": 0.911609,
        "XLM-R (GPT 4.1 Nano Labels)": 0.761831,
        "SVM (GPT 4.1 Nano Labels)": 0.90372,
        "BERTimbau (LLama3.2:3b Labels)": 0.89155,
        "XLM-R (LLama3.2:3b Labels)": 0.888638,
        "BERTimbau (Zero-Shot)": 0.345398,
        "XLM-R (Zero-Shot)": 0.457853
    },
    "mean_macro_f1": {
        "BERTimbau (GT)": 0.880118,
        "XLM-R (GT)": 0.773508,
        "SVM (GT)": 0.829187,
        "GPT 4.1 Nano (Zero-Shot)": 0.790077,
        "LLama3.2:3b (Zero-Shot)": 0.71331,
        "BERTimbau (GPT 4.1 Nano Labels)": 0.789876,
        "XLM-R (GPT 4.1 Nano Labels)": 0.721365,
        "SVM (GPT 4.1 Nano Labels)": 0.774074,
        "BERTimbau (LLama3.2:3b Labels)": 0.728235,
        "XLM-R (LLama3.2:3b Labels)": 0.733323,
        "BERTimbau (Zero-Shot)": 0.359202,
        "XLM-R (Zero-Shot)": 0.353939
    }
}

# round every thing to 4 decimal places

# Save this to a JSON file for reference
with open("./results/data.json", "w") as f:
    json.dump(data, f, indent=4)

# Plotting function (unchanged)
def plot_bar(models, scores, title, filename, colors="steelblue"):
    # Atualiza os tamanhos das fontes
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12
    })

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
            bar.get_height() - 3,  # deslocamento para dentro da barra
            f"{score:.1f}",
            ha='center', va='top', fontsize=12, color='white'
        )

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# 1) Per-dataset sorted plots
for dataset, models in data.items():
    sorted_models = sorted(models.items(), key=lambda x: x[1], reverse=True)
    names, scores = zip(*sorted_models)
    plot_bar(
        names,
        [s * 100 for s in scores],
        f"Macro F1-Score: All Models ({dataset})",
        f"./results/{dataset}_all_models_sorted.png"
    )

# # BERT highlight
# colors_bert = [
#     "gold" if "BERTimbau" in m else "dodgerblue" if "XLM-R" in m else "lightgray"
#     for m in names
# ]
# plot_bar(
#     names,
#     [v * 100 for v in mean_vals],
#     "Mean Macro F1-Score Across Datasets (BERT Highlighted)",
#     "./results/mean_macro_f1_highlighted.png",
#     colors=colors_bert
# )

# # LLM type highlight
# colors_llm = [
#     "seagreen" if "GPT 4.1 Nano" in m else "dodgerblue" if "LLama3.2:3b" in m else "lightgray"
#     for m in names
# ]
# plot_bar(
#     names,
#     [v * 100 for v in mean_vals],
#     "Mean Macro F1-Score Across Datasets (LLMs Highlighted)",
#     "./results/mean_macro_f1_llm_highlight_bytag.png",
#     colors=colors_llm
# )

# # 4) Separate BERT-only and LLM-only plots
# bert_models = [m for m in all_models if any(x in m for x in ["BERTimbau", "XLM-R"])]
# llm_models  = [m for m in all_models if any(x in m for x in ["GPT 4.1 Nano", "LLama3.2:3b"])]

# bert_means = sorted(
#     [(m, mean_scores[m] * 100) for m in bert_models],
#     key=lambda x: x[1],
#     reverse=True
# )
# llm_means = sorted(
#     [(m, mean_scores[m] * 100) for m in llm_models],
#     key=lambda x: x[1],
#     reverse=True
# )

# plot_bar(
#     [m for m, _ in bert_means],
#     [v for _, v in bert_means],
#     "Mean Macro F1-Score: BERT Only",
#     "./results/mean_macro_f1_bert_only.png"
# )
# plot_bar(
#     [m for m, _ in llm_means],
#     [v for _, v in llm_means],
#     "Mean Macro F1-Score: LLM Only",
#     "./results/mean_macro_f1_llm_only.png"
# )
