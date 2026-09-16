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
  fig6_raw_asr_per_condition.pdf/png -- raw strict_success_rate for all
      21 conditions, one panel per model (6 mechanisms and the neutral
      reference x 3 delivery forms), no
      bootstrap summarizing -- the un-aggregated numbers behind fig4,
      useful for an appendix or for readers who want the raw rates.
  fig7_z_behavior_forest.pdf/png -- point-biserial r (activation shift
      vs strict_success) with 95% bootstrap CI, one panel per model, all
      6 mechanisms x 2 scaffold kinds -- filled markers = Holm-significant
      within that (model, scaffold kind) cell. Shows the z-vs-behavior
      connection is narrow (mostly null/small, inconsistent signs) except
      one robust, replicated exception (Llama x payload_splitting).

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
MODEL_SHORT = {"Qwen2.5-7B-Instruct": "Qwen2.5-7B", "Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B", "gemma-2-9b-it": "Gemma-2-9B-it"}
SCAFFOLD_KINDS = ["neutral", "progressive"]
SCAFFOLD_LABEL = {"neutral": "neutral scaffold", "progressive": "progressive scaffold"}

CO_MECHANISMS = ["prefix_injection", "refusal_suppression", "persona_roleplay"]
MG_MECHANISMS = ["encoding_obfuscation", "payload_splitting", "distractors_negated"]
ALL_MECHANISM_GROUPS = CO_MECHANISMS + MG_MECHANISMS + ["neutral"]
FORMS = ["single", "multi_neutral", "multi_progressive"]
FORM_LABEL = {"single": "single", "multi_neutral": "multi (neutral)", "multi_progressive": "multi (progressive)"}
MECH_SHORT = {
    "prefix_injection": "prefix\ninjection", "refusal_suppression": "refusal\nsuppression", "persona_roleplay": "persona\nroleplay",
    "encoding_obfuscation": "encoding\nobfuscation", "payload_splitting": "payload\nsplitting", "distractors_negated": "distractors\nnegated",
    "neutral": "neutral",
}

CO_COLOR = "#4C72B0"
MG_COLOR = "#DD8452"
NEUTRAL_MECH_COLOR = "#8C8C8C"
GAMMA_COLOR = "#55A868"
WITHIN_CO_COLOR = "#4C72B0"
WITHIN_MG_COLOR = "#DD8452"
BETWEEN_COLOR = "#8C8C8C"
FORM_SHADES = ["#C9CDD3", "#7C93B5", "#2D4159"]  # single -> multi_neutral -> multi_progressive

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
                  "(95% CI, B=2000 bootstrap repetitions; * = Holm-significant within model, p<0.05)", y=1.06)
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

    axes[0].set_ylabel(r"mean cosine similarity among $\hat{d}^{\,\mathrm{history}}(m,k)$ vectors")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle(r"History-augmented $\hat{d}^{\,\mathrm{history}}(m,k)$ cohesion: within-CO vs. within-MG vs. between" "\n"
                  "(point estimate only; consistent across all 3 models x 2 scaffold kinds)", y=1.06)
    save(fig, out_dir, "fig5_representation_cohesion")


def fig6_raw_asr_per_condition(behavioral, out_dir):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(15, 4.6), sharey=True)
    x = np.arange(len(ALL_MECHANISM_GROUPS))
    width = 0.26

    for ax, m in zip(axes, MODELS):
        rates = behavioral[m]["per_condition_rates"]
        for offset, form in enumerate(FORMS):
            values = []
            for mech in ALL_MECHANISM_GROUPS:
                cond = f"{mech}_{form}"
                entry = rates.get(cond)
                values.append(entry["strict_success_rate"] if entry else 0.0)
            xpos = x + (offset - 1) * width
            ax.bar(xpos, values, width, label=FORM_LABEL[form], color=FORM_SHADES[offset],
                   edgecolor="black", linewidth=0.5)

        ax.axvline(2.5, color="black", linewidth=0.8, linestyle=":")
        ax.axvline(5.5, color="black", linewidth=0.8, linestyle=":")
        ax.set_xticks(x)
        labels = ax.set_xticklabels([MECH_SHORT[mech] for mech in ALL_MECHANISM_GROUPS], fontsize=7.5)
        for lbl, mech in zip(labels, ALL_MECHANISM_GROUPS):
            lbl.set_color(CO_COLOR if mech in CO_MECHANISMS else (MG_COLOR if mech in MG_MECHANISMS else NEUTRAL_MECH_COLOR))
        ax.set_title(MODEL_SHORT[m], fontsize=12)
        ax.set_ylim(0, 0.75)

    axes[0].set_ylabel("strict_success_rate (raw)")
    axes[-1].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Raw per-condition ASR, all 21 conditions\n"
                  "(6 mechanisms and the neutral reference x 3 delivery forms)\n"
                  "(no bootstrap summarizing; blue labels = CO, orange = MG, gray = neutral; dotted lines separate CO | MG | neutral)", y=1.08)
    save(fig, out_dir, "fig6_raw_asr_per_condition")


def fig7_z_behavior_forest(connection, out_dir):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(13, 6), sharex=True)
    row_labels = []
    for mech in ALL_MECHANISM_GROUPS[:-1]:  # REAL_MECHANISMS only (connection analysis excludes 'neutral')
        for kind in SCAFFOLD_KINDS:
            row_labels.append((mech, kind))
    y = np.arange(len(row_labels))[::-1]

    UNRELIABLE_COLOR = "#C7C9CD"
    for ax, m in zip(axes, MODELS):
        for yi, (mech, kind) in zip(y, row_labels):
            entry = connection[m]["by_scaffold_kind"][kind].get(mech)
            z = entry["z_vs_strict_success"] if entry else None
            r = z.get("point_biserial_r") if z else None
            color = CO_COLOR if mech in CO_MECHANISMS else MG_COLOR
            if r is None:
                ax.plot(0, yi, marker="x", color="#B0B3B8", markersize=6)
                continue
            # reliable=False (real-data incident, Sep 2026): extreme
            # class imbalance (min(success,fail)<5) degenerates the
            # bootstrap into an outlier-detection artifact regardless of
            # |r| -- plotted muted gray, never colored/filled, even if
            # the raw r looked large, so this never reads as a finding.
            reliable = z.get("reliable", True)
            lo, hi = z.get("r_ci_2_5"), z.get("r_ci_97_5")
            p_holm = z.get("r_p_holm_adjusted")
            is_sig = reliable and p_holm is not None and p_holm < 0.05
            plot_color = color if reliable else UNRELIABLE_COLOR
            if lo is not None and hi is not None:
                ax.plot([lo, hi], [yi, yi], color=plot_color, linewidth=1.3, alpha=0.8, zorder=1)
            ax.plot(r, yi, marker="o", markersize=6.5 if is_sig else 5,
                     markerfacecolor=plot_color if is_sig else "white", markeredgecolor=plot_color, markeredgewidth=1.4, zorder=2)

        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(MODEL_SHORT[m], fontsize=12)
        ax.set_xlim(-0.6, 0.6)
        ax.set_xlabel(r"point-biserial $r$ ($z$ vs. strict_success)")

    ytick_labels = [f"{MECH_SHORT[mech].replace(chr(10), ' ')} ({'N' if kind == 'neutral' else 'P'})" for mech, kind in row_labels]
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(ytick_labels, fontsize=8.5)
    for lbl, (mech, _) in zip(axes[0].get_yticklabels(), row_labels):
        lbl.set_color(CO_COLOR if mech in CO_MECHANISMS else MG_COLOR)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    fig.suptitle("Activation shift vs. behavioral success: point-biserial $r$, 95% bootstrap CI\n"
                  "(filled = Holm-significant; gray = unreliable, class imbalance <5 per group -- never a finding regardless of |r|; "
                  "N/P = neutral/progressive; x = undefined, zero-variance outcome)", y=1.05)
    save(fig, out_dir, "fig7_z_behavior_forest")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    behavioral = load_all(args.analysis_dir, "behavioral")
    representation = load_all(args.analysis_dir, "representation")
    connection = load_all(args.analysis_dir, "activation_behavior_connection")

    fig4_behavioral_corrected_effects(behavioral, args.output_dir)
    fig5_representation_cohesion(representation, args.output_dir)
    fig6_raw_asr_per_condition(behavioral, args.output_dir)
    fig7_z_behavior_forest(connection, args.output_dir)

    print(f"Saved 4 figures (.pdf + .png) to {args.output_dir}")


if __name__ == "__main__":
    main()
