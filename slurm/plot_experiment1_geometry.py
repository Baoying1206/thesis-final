"""Publication figures for Experiment 1 (RQ1) geometry results
(FINAL_STUDY_PROTOCOL.md Sec 4.3). Reads the 3 models'
`slurm/experiment1_analysis/<model>_experiment1_geometry.json` files
(real data, produced by analyze_experiment1_geometry.py) -- this script
only plots already-computed numbers, it does not compute anything new
or re-derive statistics.

Produces, at the frozen primary position (t_generation_boundary) unless
--position is overridden:
  fig1_delta_co_mg_bootstrap.pdf/png  -- Delta_CO/Delta_MG per model,
      95% bootstrap CI error bars (the core within-vs-between-group
      cohesion evidence).
  fig2_cosine_heatmap.pdf/png         -- 6x6 pairwise cosine similarity
      matrix per model (CO block, then MG block), showing the taxonomy's
      within/between-group structure directly.
  fig3_layerwise_T_sweep.pdf/png      -- T statistic (canonical CO/MG
      partition) across every layer per model, primary layer marked --
      the full-layer sensitivity sweep, showing whether the primary-
      layer finding holds broadly across depth.

CPU-only, no model, no GPU. Run locally after pulling the real
analysis JSON files.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ANALYSIS_DIR = os.path.join(SCRIPT_DIR, "experiment1_analysis")
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_ANALYSIS_DIR, "figures")

MODELS = ["Qwen2.5-7B-Instruct", "Meta-Llama-3.1-8B-Instruct", "gemma-2-9b-it"]
MODEL_SHORT = {"Qwen2.5-7B-Instruct": "Qwen2.5-7B", "Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B", "gemma-2-9b-it": "Gemma-2-9B"}
PRIMARY_LAYERS = {"Qwen2.5-7B-Instruct": 16, "Meta-Llama-3.1-8B-Instruct": 19, "gemma-2-9b-it": 25}

CO_COLOR = "#4C72B0"
MG_COLOR = "#DD8452"

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def load_all(analysis_dir):
    data = {}
    for m in MODELS:
        path = os.path.join(analysis_dir, f"{m}_experiment1_geometry.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing {path} -- run analyze_experiment1_geometry.py for {m} first")
        with open(path, "r", encoding="utf-8") as f:
            data[m] = json.load(f)
    return data


def save(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)


def fig1_delta_bootstrap(data, position, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(MODELS))
    width = 0.35

    for offset, (key, color, label) in enumerate([
        ("Delta_CO", CO_COLOR, r"$\Delta_{CO}$"),
        ("Delta_MG", MG_COLOR, r"$\Delta_{MG}$"),
    ]):
        means, lo_err, hi_err = [], [], []
        for m in MODELS:
            boot = data[m]["by_position"][position]["bootstrap"][key]
            point = data[m]["by_position"][position][key]
            means.append(point)
            lo_err.append(point - boot["ci95"][0])
            hi_err.append(boot["ci95"][1] - point)
        xpos = x + (offset - 0.5) * width
        ax.bar(xpos, means, width, label=label, color=color,
               yerr=[lo_err, hi_err], capsize=4, edgecolor="black", linewidth=0.6)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODELS])
    ax.set_ylabel(r"$\Delta$ (within-group $-$ between-group cosine)")
    ax.set_title(f"CO/MG within-vs-between-group cohesion\n(position: {position}, 95% bootstrap CI, n=2000)")
    ax.legend(frameon=False)
    save(fig, out_dir, "fig1_delta_co_mg_bootstrap")


def fig2_cosine_heatmap(data, position, out_dir):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(4.2 * len(MODELS), 4))
    for ax, m in zip(axes, MODELS):
        matrix_dict = data[m]["by_position"][position]["cosine_matrix"]
        CO = data[m]["CO_mechs"]
        MG = data[m]["MG_mechs"]
        order = CO + MG
        grid = np.array([[matrix_dict[a][b] for b in order] for a in order])
        im = ax.imshow(grid, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(order)))
        ax.set_yticks(range(len(order)))
        short_labels = [o.replace("_", "\n") for o in order]
        ax.set_xticklabels(short_labels, rotation=90, fontsize=7)
        ax.set_yticklabels(short_labels, fontsize=7)
        ax.axhline(2.5, color="black", linewidth=1.2)
        ax.axvline(2.5, color="black", linewidth=1.2)
        for i in range(len(order)):
            for j in range(len(order)):
                value = grid[i, j]
                text_color = "white" if abs(value) > 0.6 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color=text_color)
        ax.set_title(MODEL_SHORT[m], fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.7, label="cosine similarity")
    fig.suptitle(f"Pairwise cosine similarity among calibrated mechanism directions\n(position: {position}; top-left 3x3 block = CO, bottom-right 3x3 block = MG)", y=1.08)
    fig.savefig(os.path.join(out_dir, "fig2_cosine_heatmap.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig2_cosine_heatmap.png"), bbox_inches="tight")
    plt.close(fig)


def fig3_layerwise_sweep(data, position, out_dir):
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for m, color in zip(MODELS, ["#4C72B0", "#DD8452", "#55A868"]):
        sweep = data[m]["by_position"][position].get("layerwise_sweep_point_estimate_only")
        if not sweep:
            continue
        layers = [row["layer"] for row in sweep]
        n_layers = len(layers)
        relative_depth = [l / (n_layers - 1) for l in layers]
        T_values = [row["T_taxonomy"] for row in sweep]
        ax.plot(relative_depth, T_values, label=MODEL_SHORT[m], color=color, linewidth=1.6)
        primary_layer = PRIMARY_LAYERS[m]
        primary_depth = primary_layer / (n_layers - 1)
        ax.axvline(primary_depth, color=color, linestyle="--", linewidth=1, alpha=0.6)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("relative depth (layer / n_layers)")
    ax.set_ylabel(r"$T$ = mean within-group cosine $-$ between-group cosine")
    ax.set_title(f"Canonical CO/MG partition's T statistic across layers\n(position: {position}; dashed lines = each model's frozen primary layer)", pad=14)
    ax.legend(frameon=False)
    save(fig, out_dir, "fig3_layerwise_T_sweep")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--position", default="t_generation_boundary",
                         choices=["t_generation_boundary", "t_final_user_end"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_all(args.analysis_dir)

    fig1_delta_bootstrap(data, args.position, args.output_dir)
    fig2_cosine_heatmap(data, args.position, args.output_dir)
    fig3_layerwise_sweep(data, args.position, args.output_dir)

    print(f"Saved 3 figures (.pdf + .png) to {args.output_dir}")


if __name__ == "__main__":
    main()
