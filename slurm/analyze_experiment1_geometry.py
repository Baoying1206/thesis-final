"""Experiment 1 (RQ1) geometry analysis (FINAL_STUDY_PROTOCOL.md Sec 4.3).
Loads the RAW per-condition activation `.pt` files produced by
`extract_experiment1_activations.py` -- this script never loads a model,
never touches GPU, and never reads generation/judge output (Sec 4.2's
firewall: direction estimation is never filtered by behavioral outcome).

Computes, at the model's frozen primary layer and `t_generation_boundary`
position (primary) plus `t_final_user_end` (sensitivity):
- per-mechanism placebo-calibrated direction `tilde_d_m = d_m - d_placebo`
  (Sec 4.2), `d_m` estimated from ALL loaded `direction_ids` instructions
- pairwise cosine among the 6 mechanisms' calibrated directions
- the CO/MG partition (read dynamically from
  `templates/imported/templates_wei_canonical.json` via
  `src/taxonomy_v2_loader.py` -- never hardcoded) vs. all 10 balanced
  3-vs-3 partitions of the 6 mechanisms, ranked
- split-half reliability per mechanism
- instruction-cluster bootstrap (Sec 8) of S_CO/S_MG/S_between/
  Delta_CO/Delta_MG/T and the canonical partition's rank
- per-mechanism diff-vector norm (mean/std/min/max) -- cosine alone is
  blind to effect size, so this is reported alongside it, not as a
  replacement
- a point-estimate-only (no bootstrap) sweep of the same S_CO/S_MG/
  S_between/Delta_CO/Delta_MG/T/canonical-rank stats at EVERY layer, not
  just the frozen primary layer -- a sensitivity check for whether the
  primary-layer finding is a one-off artifact of that layer, matching
  Sec 4.2's "full-layer activation extraction" framing (the extraction
  script already saves every layer; this was the first analysis to
  actually use them)

Every result reported per model separately (Sec 4.3) -- this script
processes one `--model-alias` per run; cross-model comparison happens by
running it 3 times and reading the 3 output files together, never by
averaging inside this script.
"""

import argparse
import json
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(SCRIPT_DIR, "..")
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from taxonomy_v2_loader import load_taxonomy_v2  # noqa: E402
from stats_shared import (  # noqa: E402
    aggregate, cos, all_2group_partitions, compute_2group_partition_stats,
    rank_partitions, split_half_reliability, load_instruction_clusters,
    pairwise_cosine_matrix, partition_T_from_matrix,
)

WEI_CANONICAL_PATH = os.path.join(REPO_ROOT, "templates", "imported", "templates_wei_canonical.json")
SAMPLED_PROMPTS_EN_ONLY_PATH = os.path.join(REPO_ROOT, "data", "source", "sampled_prompts_en_only.json")
DEFAULT_EXTRACTION_DIR = os.path.join(SCRIPT_DIR, "experiment1_output")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "experiment1_analysis")

PRIMARY_TOKEN_POSITION = "t_generation_boundary"
SECONDARY_TOKEN_POSITION = "t_final_user_end"
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260828
SPLIT_HALF_REPS = 500
SPLIT_HALF_SEED = 20260828


def load_condition_activations(extraction_dir, model_alias, condition, ids_key_check=True):
    path = os.path.join(extraction_dir, f"{model_alias}_{condition}_activations.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing extraction output: {path} -- run extract_experiment1_activations.py first")
    return torch.load(path, map_location="cpu")


def calibrated_diff_vecs(plain_acts, placebo_acts, mech_acts, ids, layer, position):
    """Returns {instruction_id: [hidden] tensor} of placebo-calibrated
    difference vectors for one mechanism, at one layer/position."""
    out = {}
    for i in ids:
        raw_diff = mech_acts[i][position][layer] - plain_acts[i][position][layer]
        placebo_diff = placebo_acts[i][position][layer] - plain_acts[i][position][layer]
        out[i] = raw_diff - placebo_diff
    return out


def bootstrap_geometry(all_diff_vecs, mechanisms, CO, MG, clusters, partitions, canonical_idx):
    import random
    rng = random.Random(BOOTSTRAP_SEED)
    n_clusters = len(clusters)

    S_CO_l, S_MG_l, S_between_l, Delta_CO_l, Delta_MG_l, T_l, canonical_ranks = [], [], [], [], [], [], []
    for _ in range(N_BOOTSTRAP):
        draws = [clusters[rng.randrange(n_clusters)] for _ in range(n_clusters)]
        resampled_ids = [i for cluster in draws for i in cluster]
        vecs = {m: aggregate(torch.stack([all_diff_vecs[m][i] for i in resampled_ids])) for m in mechanisms}
        # Precompute the 6x6 cosine matrix ONCE per replicate (vectorized) --
        # avoids 10 partitions x 6 pairwise cos() calls each on every replicate.
        matrix, index_of = pairwise_cosine_matrix(vecs, mechanisms)
        S_CO, S_MG, S_between, Delta_CO, Delta_MG, T = compute_2group_partition_stats(vecs, CO, MG)
        S_CO_l.append(S_CO); S_MG_l.append(S_MG); S_between_l.append(S_between)
        Delta_CO_l.append(Delta_CO); Delta_MG_l.append(Delta_MG); T_l.append(T)
        all_T = [partition_T_from_matrix(matrix, index_of, [a, b]) for a, b in partitions]
        ranks = rank_partitions(all_T)
        canonical_ranks.append(ranks[canonical_idx])

    def summarize(values):
        t = torch.tensor(values)
        return {"mean": t.mean().item(), "median": t.median().item(),
                "ci95": [t.quantile(0.025).item(), t.quantile(0.975).item()]}

    return {
        "S_CO": summarize(S_CO_l), "S_MG": summarize(S_MG_l), "S_between": summarize(S_between_l),
        "Delta_CO": summarize(Delta_CO_l), "Delta_MG": summarize(Delta_MG_l), "T": summarize(T_l),
        "P_Delta_CO_gt_0": sum(1 for x in Delta_CO_l if x > 0) / N_BOOTSTRAP,
        "P_Delta_MG_gt_0": sum(1 for x in Delta_MG_l if x > 0) / N_BOOTSTRAP,
        "P_canonical_rank_1": sum(1 for r in canonical_ranks if r == 1) / N_BOOTSTRAP,
        "n_bootstrap": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED,
        "note": "P(...) values are bootstrap resampling proportions, not formal p-values.",
    }


def mean_norms(all_diff_vecs, mechanisms, ids):
    """Magnitude signal alongside the direction-only cosine analysis --
    cosine is blind to effect size, so a mechanism whose 'cohesion' is
    driven by a near-zero, noisy diff vector looks identical to one with
    a strong, consistent effect unless norms are reported separately."""
    out = {}
    for m in mechanisms:
        norms = torch.stack([all_diff_vecs[m][i] for i in ids]).norm(dim=-1)
        out[m] = {"mean_norm": norms.mean().item(), "std_norm": norms.std().item(),
                  "min_norm": norms.min().item(), "max_norm": norms.max().item()}
    return out


def layerwise_sweep(plain_acts, placebo_acts, mech_acts, ids, mechanisms, CO, MG, position, n_layers_total):
    """Point-estimate ONLY (no bootstrap) S_CO/S_MG/S_between/Delta_CO/
    Delta_MG/T/canonical_partition_rank at EVERY layer -- a sensitivity
    sweep matching Sec 4.2's 'full-layer activation extraction' framing
    (the primary layer is pre-registered and fixed; this sweep checks
    whether the primary-layer finding is a one-off artifact of that
    specific layer or holds broadly). Bootstrap remains fixed-layer only
    (Sec 4.2) -- running 2000 reps at every layer would be needless
    compute for a sensitivity check, not the primary analysis."""
    rows = []
    partitions = all_2group_partitions(mechanisms)
    canonical_idx = next(i for i, (a, b) in enumerate(partitions) if set(a) == set(CO) or set(a) == set(MG))
    for layer in range(n_layers_total):
        diff_vecs_l = {m: calibrated_diff_vecs(plain_acts, placebo_acts, mech_acts[m], ids, layer, position) for m in mechanisms}
        vecs_l = {m: aggregate(torch.stack([diff_vecs_l[m][i] for i in ids])) for m in mechanisms}
        matrix, index_of = pairwise_cosine_matrix(vecs_l, mechanisms)
        S_CO, S_MG, S_between, Delta_CO, Delta_MG, T = compute_2group_partition_stats(vecs_l, CO, MG)
        all_T = [partition_T_from_matrix(matrix, index_of, [a, b]) for a, b in partitions]
        ranks = rank_partitions(all_T)
        rows.append({
            "layer": layer, "S_CO": S_CO, "S_MG": S_MG, "S_between": S_between,
            "Delta_CO": Delta_CO, "Delta_MG": Delta_MG, "T_taxonomy": T,
            "canonical_partition_rank": ranks[canonical_idx],
        })
    return rows


def run(model_alias, primary_layer, extraction_dir, output_dir, skip_layerwise_sweep=False):
    taxonomy = load_taxonomy_v2(WEI_CANONICAL_PATH)
    mechanisms = taxonomy["active_mechanisms"]
    CO, MG = taxonomy["CO_mechs"], taxonomy["MG_mechs"]

    plain_acts = load_condition_activations(extraction_dir, model_alias, "plain")
    placebo_acts = load_condition_activations(extraction_dir, model_alias, "placebo")
    mech_acts = {m: load_condition_activations(extraction_dir, model_alias, m) for m in mechanisms}

    ids = sorted(set(plain_acts.keys()) & set(placebo_acts.keys()) & set.intersection(*[set(mech_acts[m].keys()) for m in mechanisms]))
    if not ids:
        raise ValueError(f"no common instruction_ids across plain/placebo/{mechanisms} for {model_alias}")

    with open(SAMPLED_PROMPTS_EN_ONLY_PATH, "r", encoding="utf-8") as f:
        pool = json.load(f)
    instruction_en_by_id = {row["id"]: row["instruction_en"] for row in pool}
    clusters = load_instruction_clusters(ids, instruction_en_by_id)

    results_by_position = {}
    for position in (PRIMARY_TOKEN_POSITION, SECONDARY_TOKEN_POSITION):
        print(f"[{model_alias}] position={position}: computing calibrated diff vectors...", file=sys.stderr)
        all_diff_vecs = {m: calibrated_diff_vecs(plain_acts, placebo_acts, mech_acts[m], ids, primary_layer, position) for m in mechanisms}
        vecs = {m: aggregate(torch.stack([all_diff_vecs[m][i] for i in ids])) for m in mechanisms}

        cosine_matrix = {a: {b: cos(vecs[a], vecs[b]) for b in mechanisms} for a in mechanisms}
        partitions = all_2group_partitions(mechanisms)
        canonical_idx = next(i for i, (a, b) in enumerate(partitions) if set(a) == set(CO) or set(a) == set(MG))
        all_T = [compute_2group_partition_stats(vecs, a, b)[5] for a, b in partitions]
        ranks = rank_partitions(all_T)
        S_CO, S_MG, S_between, Delta_CO, Delta_MG, T = compute_2group_partition_stats(vecs, CO, MG)
        print(f"[{model_alias}] position={position}: point estimates done "
              f"(S_CO={S_CO:.4f} S_MG={S_MG:.4f} canonical_rank={ranks[canonical_idx]}/{len(partitions)})", file=sys.stderr)

        print(f"[{model_alias}] position={position}: split-half reliability ({SPLIT_HALF_REPS} reps x {len(mechanisms)} mechanisms)...", file=sys.stderr)
        reliability = {m: split_half_reliability(all_diff_vecs[m], ids, n_reps=SPLIT_HALF_REPS, seed=SPLIT_HALF_SEED) for m in mechanisms}
        n_layers_total = mech_acts[mechanisms[0]][ids[0]][position].shape[0]

        print(f"[{model_alias}] position={position}: bootstrap ({N_BOOTSTRAP} reps)...", file=sys.stderr)
        bootstrap = bootstrap_geometry(all_diff_vecs, mechanisms, CO, MG, clusters, partitions, canonical_idx)

        sweep = None
        if not skip_layerwise_sweep:
            print(f"[{model_alias}] position={position}: layerwise sweep ({n_layers_total} layers, point-estimate only)...", file=sys.stderr)
            sweep = layerwise_sweep(plain_acts, placebo_acts, mech_acts, ids, mechanisms, CO, MG, position, n_layers_total)
        else:
            print(f"[{model_alias}] position={position}: --skip-layerwise-sweep set, skipping ({n_layers_total} layers)", file=sys.stderr)

        results_by_position[position] = {
            "n_instructions": len(ids), "n_instruction_clusters": len(clusters),
            "cosine_matrix": cosine_matrix,
            "S_CO": S_CO, "S_MG": S_MG, "S_between": S_between,
            "Delta_CO": Delta_CO, "Delta_MG": Delta_MG, "T_taxonomy": T,
            "canonical_partition_rank": ranks[canonical_idx],
            "n_partitions": len(partitions),
            "mean_norms": mean_norms(all_diff_vecs, mechanisms, ids),
            "reliability": reliability,
            "bootstrap": bootstrap,
            "layerwise_sweep_point_estimate_only": sweep,
            "layerwise_sweep_skipped": skip_layerwise_sweep,
        }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_alias}_experiment1_geometry.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "result_status": "EXPERIMENT1_GEOMETRY_ANALYSIS",
            "model_alias": model_alias, "primary_layer": primary_layer,
            "taxonomy_version": taxonomy["taxonomy_version"], "CO_mechs": CO, "MG_mechs": MG,
            "primary_position": PRIMARY_TOKEN_POSITION, "sensitivity_position": SECONDARY_TOKEN_POSITION,
            "by_position": results_by_position,
        }, f, indent=2)
    print(json.dumps({"result_status": "EXPERIMENT1_ANALYSIS_DONE", "model_alias": model_alias, "output_path": out_path}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--primary-layer", type=int, required=True)
    parser.add_argument("--extraction-dir", default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-layerwise-sweep", action="store_true",
                         help="Skip the full-layer point-estimate sweep (Sec 4.2 sensitivity check) -- "
                              "much faster, gives the core bootstrap/reliability/cosine results only. "
                              "Re-run without this flag later to get the complete record.")
    args = parser.parse_args()
    run(args.model_alias, args.primary_layer, args.extraction_dir, args.output_dir, args.skip_layerwise_sweep)


if __name__ == "__main__":
    main()
