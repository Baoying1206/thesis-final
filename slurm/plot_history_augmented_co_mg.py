"""Publication figures for the RQ2 Round 20 replacement design
(history-augmented canonical CO/MG, FINAL_STUDY_PROTOCOL.md Sec 13
Round 20). Reads the 3 models' real
`slurm/history_augmented_co_mg_analysis/<model>_history_augmented_{behavioral,representation}.json`
files (produced by analyze_history_augmented_co_mg.py) -- this script
only plots already-computed numbers, it does not compute anything new.

Produces:
  fig4_behavioral_corrected_effects.pdf/png -- corrected_CO/corrected_MG/
      Gamma per model, one panel per scaffold kind (neutral/progressive),
      95% bootstrap CI error bars, Holm-significant bars marked with *.
      This is the primary confirmatory result (Sec 13 Round 20 + follow-up).
  fig5_representation_cohesion.pdf/png -- within_CO/within_MG/between_CO_MG
      mean cosine per model, one panel per scaffold kind -- the
      cross-model-consistent finding (within_CO > within_MG in all 3
      models x 2 scaffold kinds), contrasted against the behavioral
      panel's model-specific heterogeneity.

CPU-only, no model, no GPU. Run locally after pulling the real
analysis JSON files (git pull, then this script).
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ANALYSIS_DIR = os.path.join(SCRIPT_DIR, "history_augmented_co_mg_analysis")
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_ANALYSIS_DIR, "figures")

MODELS = ["Qwen2.5-7B-Instruct", "Meta-Llama-3.1-8B-Instruct", "gemma-2-9b-it"]
MODEL_SHORT = {"Qwen2.5-7B-Instruct": "Qwen2.5-7B", "Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B", "gemma-2-9b-it": "Gemma-2-9B"}
SCAFFOLD_KINDS = ["neutral", "progressive"]
SCAFFOLD_LABEL = {"neutral": "neutral scaffold", "progressive": "progressive scaffold"}

CO_COLOR = "#4C72B0"
MG_COLOR = "#DD8452"
GAMMA_COLOR = "#55A868"
WITHIN_CO_COLOR = "#4C72B0"
WITHIN_MG_COLOR = "#DD8452"
BETWEEN_COLOR = "#8C8C8C"

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def load_all(analysis_dir, suffix):
    data = {}
    for m in MODELS:
        path = os.path.join(analysis_dir, f"{m}_history_augmented_{suffix}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing {path} -- run analyze_history_augmented_co_mg.py for {m} first")
        with open(path, "r", encoding="utf-8") as f:
            data[m] = json.load(f)
    return data


def save(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)


def fig4_behavioral_corrected_effects(behavioral, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    x = np.arange(len(MODELS))
    width = 0.25

    for ax, kind in zip(axes, SCAFFOLD_KINDS):
        for offset, (key, color, label) in enumerate([
            ("corrected_CO", CO_COLOR, r"corrected$_{CO}$"),
            ("corrected_MG", MG_COLOR, r"corrected$_{MG}$"),
            ("Gamma", GAMMA_COLOR, r"$\Gamma$"),
        ]):
            means, lo_err, hi_err, sig = [], [], [], []
            for m in MODELS:
                e = behavioral[m]["effects"]["by_kind"][kind][key]
                point = e["point"]
                means.append(point)
                lo_err.append(point - e["ci_2_5"])
                hi_err.append(e["ci_97_5"] - point)
                sig.append(e.get("p_holm_adjusted") is not None and e["p_holm_adjusted"] < 0.05)
            xpos = x + (offset - 1) * width
            bars = ax.bar(xpos, means, width, label=label, color=color,
                           yerr=[lo_err, hi_err], capsize=3, edgecolor="black", linewidth=0.6)
            for bar, m_val, hi, is_sig in zip(bars, means, hi_err, sig):
                if is_sig:
                    y = bar.get_height() + hi + 0.008 if m_val >= 0 else bar.get_height() - hi - 0.02
                    ax.text(bar.get_x() + bar.get_width() / 2, y, "*", ha="center", va="bottom",
                            fontsize=14, fontweight="bold")

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_SHORT[m] for m in MODELS])
        ax.set_title(SCAFFOLD_LABEL[kind])

    axes[0].set_ylabel("ASR delta (multi $-$ single), corrected against neutral baseline")
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle("History-augmented canonical CO/MG: primary confirmatory effects\n"
                  "(95% bootstrap CI, n=2000; * = Holm-significant within model, p<0.05)", y=1.06)
    save(fig, out_dir, "fig4_behavioral_corrected_effects")


def fig5_representation_cohesion(representation, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    x = np.arange(len(MODELS))
    width = 0.25

    for ax, kind in zip(axes, SCAFFOLD_KINDS):
        for offset, (key, color, label) in enumerate([
            ("within_CO_mean_cosine", WITHIN_CO_COLOR, "within-CO"),
            ("within_MG_mean_cosine", WITHIN_MG_COLOR, "within-MG"),
            ("between_CO_MG_mean_cosine", BETWEEN_COLOR, "between CO/MG"),
        ]):
            values = [representation[m]["by_scaffold_kind"][kind]["co_mg_cohesion"][key] for m in MODELS]
            xpos = x + (offset - 1) * width
            ax.bar(xpos, values, width, label=label, color=color, edgecolor="black", linewidth=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_SHORT[m] for m in MODELS])
        ax.set_title(SCAFFOLD_LABEL[kind])
        ax.set_ylim(0, 1)

    axes[0].set_ylabel(r"mean cosine similarity among $d_m^{history}$ vectors")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("History-augmented $d_m$ cohesion: within-CO vs. within-MG vs. between\n"
                  "(point estimate only; consistent across all 3 models x 2 scaffold kinds)", y=1.06)
    save(fig, out_dir, "fig5_representation_cohesion")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    behavioral = load_all(args.analysis_dir, "behavioral")
    representation = load_all(args.analysis_dir, "representation")

    fig4_behavioral_corrected_effects(behavioral, args.output_dir)
    fig5_representation_cohesion(representation, args.output_dir)

    print(f"Saved 2 figures (.pdf + .png) to {args.output_dir}")


if __name__ == "__main__":
    main()
