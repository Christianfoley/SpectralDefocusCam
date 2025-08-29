# %%
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re


def get_scores_by_metric(scores):
    metrics = scores[list(scores.keys())[0]].keys()
    newtable = {m: None for m in metrics}

    for m in metrics:
        vals, depth = [], []
        for model, model_scores in scores.items():
            if "sharp" in model:
                continue
            depth.append(int(re.search(r"\d+", model).group(0)))
            vals.append(model_scores[m])
        newtable[m] = {"scores": vals, "depth": depth}

    return newtable


def plot_scores(score_tables, husl_color_idcs, model_names):
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    colors = sns.husl_palette(10, l=0.5)
    line_styles = [
        "-",
    ] * 4  # "--", "-.", ":"]
    markers = [
        "o",
    ] * 4  # "s", "x", "^"]

    fig, axs = plt.subplots(1, 4, figsize=(40, 3.8), dpi=200)

    for i, score_table in enumerate(score_tables):
        color = colors[husl_color_idcs[i]]
        for j, (name, data) in enumerate(score_table.items()):
            axs[j].plot(
                data["depth"],
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
                axs[j].tick_params(labelbottom=False, labelsize=20)

            # if j == 0:
            axs[j].set_title(name, fontsize=28)
            #axs[j].legend(loc=(0.55, 0.58), fontsize=40, ncol=1, framealpha=1)
            

    plt.tight_layout()
    plt.savefig("plot_depth_ablation.png")
    plt.show()


# %%

fista_scores = "fista_scores.json"
lrnd_scores = "learned_scores_firstlastonly.json"

with open(lrnd_scores, "r") as f:
    lrnd_scores = get_scores_by_metric(json.load(f))

with open(fista_scores, "r") as f:
    fista_scores = get_scores_by_metric(json.load(f))

plot_scores([lrnd_scores, fista_scores], [0,7], ["learned", "fista"])

# %%
unet_scores = "classical_unet_scores.json"
condunet_scores = "learned_scores.json"
condunet_fstlst_scores = "learned_scores_firstlastonly.json"

with open(unet_scores, "r") as f:
    unet_scores = get_scores_by_metric(json.load(f))

with open(condunet_scores, "r") as f:
    condunet_scores = get_scores_by_metric(json.load(f))

with open(condunet_fstlst_scores, "r") as f:
    condunet_fstlst_scores = get_scores_by_metric(json.load(f))

plot_scores([unet_scores, condunet_scores, condunet_fstlst_scores], [0,7,2], ["No-PSFs", "All-PSFs", "Two-PSFs"])
