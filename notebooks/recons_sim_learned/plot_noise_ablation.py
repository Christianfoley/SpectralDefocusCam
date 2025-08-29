# %%
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
from matplotlib.ticker import MultipleLocator


def get_scores_by_metric(scores):
    metrics = scores[list(scores.keys())[0]].keys()
    newtable = {m: None for m in metrics}

    for m in metrics:
        vals, noise = [], []
        for model, model_scores in scores.items():
            if "sharp" in model:
                continue
            # Adjusted regular expression pattern to capture noise values
            match = re.search(r"noise_([0-9]+(?:\.[0-9]+|(?:e[+-]\d+)?))", model)
            if match:
                noise_val = float(match.group(1))
                noise.append(noise_val)
                vals.append(model_scores[m])
        newtable[m] = {"scores": vals, "noise": noise}

    return newtable


def plot_scores(score_tables, husl_color_idcs, model_names):
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    colors = sns.husl_palette(n_colors=10, l=0.5)
    # line_styles = ["-", "--", "-.", ":"]
    # markers = ["o", "s", "x", "^"]
    #colors = ("red", "darkblue")
    line_styles = [
        "-",
    ] * 4  # "--", "-.", ":"]
    markers = [
        "o",
    ] * 4  # "s", "x", "^"]

    fig, axs = plt.subplots(1, 4, figsize=(28, 4), dpi=200)

    for i, score_table in enumerate(score_tables):
        color = colors[husl_color_idcs[i]]
        for j, (name, data) in enumerate(score_table.items()):
            axs[j].plot(
                data["noise"],
                data["scores"],
                marker=markers[j],
                markersize=14,
                color=colors[husl_color_idcs[i]],
                linestyle=line_styles[j],
                label=model_names[i] if j == 0 else None,
                linewidth=4,
            )
            if i == 0:
                axs[j].spines["top"].set_visible(False)
                axs[j].spines["right"].set_visible(False)
                axs[j].spines["left"].set_linewidth(2)
                axs[j].spines["bottom"].set_linewidth(2)
                axs[j].spines["left"].set_color("black")
                axs[j].spines["bottom"].set_color("black")
                axs[j].set_xscale("log")
                axs[j].tick_params(labelbottom=False, labelsize=20)
                axs[j].xaxis.set_major_locator(
                    MultipleLocator(10)
                )  # Set multiple locator for x-axis
                axs[j].grid(which="both", linestyle="-", linewidth=1, color="gray")
                axs[j].grid(
                    which="minor", linestyle="-", linewidth=0.5, color="gray"
                )  # Log scale grid for minor ticks
            if j == 0:
                 axs[j].legend(loc=(0.05, 0.58), fontsize=42, ncol=1, framealpha=1)
            # axs[j].set_title(name, fontsize=28)

    plt.tight_layout()
    plt.savefig("plot_noise_ablation.png")
    plt.show()


# %%

fista_scores = "fista_maskandshot_noise_scores.json"
lrnd_scores = "learned_maskandshot_noise_scores.json"

with open(lrnd_scores, "r") as f:
    lrnd_scores = get_scores_by_metric(json.load(f))

with open(fista_scores, "r") as f:
    fista_scores = get_scores_by_metric(json.load(f))

plot_scores([lrnd_scores, fista_scores], [0, 7], ["learned", "fista"])

# %%
