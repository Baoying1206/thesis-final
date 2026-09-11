"""Experiment 2 (RQ2) representation analysis (FINAL_STUDY_PROTOCOL.md
Sec 6/6.1/6.2). Loads RAW per-condition activation `.pt` files produced
by `extract_experiment2_activations.py` -- never loads a model, never
touches GPU.

`direction_ids` activations -> `d_m` per condition (9 non-neutral,
Sec 6's firewall: computed from ALL loaded direction_ids, never filtered
by behavioral outcome -- this script has no generation/judge code in it
at all), pairwise cosine, {CO, MG, Context} vs. all 280 alternative
3-way partitions of the 9 conditions (Sec 6.1), split-half reliability,
Context leave-one-out stability, and an explicit check of RQ2's 5
minimum-evidence criteria (Sec 6.1) -- reported as VALUES against each
criterion, not a single yes/no verdict this script invents; the human
judgement of whether a value counts as "reliable enough" is not
automated here beyond the operational proxies documented in the code.

`validation_ids` activations -> Sec 6.2's projection-vs-strict_success
correlation check ONLY: projects each validation instruction's
difference vector onto the ALREADY-FROZEN `d_m` (read-only, never
re-estimated here) and correlates the resulting scalar with
`strict_success` from `run_formal_behavioral.py`'s judge output for the
SAME model/condition/instruction. This is the only place this script
reads behavioral (judge) data, and it never feeds back into `d_m`.

Every result reported per model separately (one `--model-alias` per
run); cross-model replication (criterion 5) is evaluated by reading all
3 models' output files together, not inside this script.
"""

import argparse
import json
import os
import random
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(SCRIPT_DIR, "..")
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from stats_shared import (  # noqa: E402
    aggregate, cos, all_3group_partitions, compute_3group_partition_stats,
    rank_partitions, split_half_reliability, load_instruction_clusters,
    pairwise_cosine_matrix, partition_T_from_matrix,
)

SAMPLED_PROMPTS_EN_ONLY_PATH = os.path.join(REPO_ROOT, "data", "source", "sampled_prompts_en_only.json")
DEFAULT_EXTRACTION_DIR = os.path.join(SCRIPT_DIR, "experiment2_output")
DEFAULT_BEHAVIORAL_DIR = os.path.join(SCRIPT_DIR, "formal_behavioral_output")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "experiment2_analysis")

PRIMARY_TOKEN_POSITION = "t_generation_boundary"
SECONDARY_TOKEN_POSITION = "t_final_user_end"
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260828
SPLIT_HALF_REPS = 500
SPLIT_HALF_SEED = 20260828

CO = ["co_prefix_injection", "co_refusal_suppression", "co_persona_roleplay"]
MG = ["mg_encoding_obfuscation", "mg_payload_splitting", "mg_distractors_negated"]
CONTEXT = ["context_persona", "context_authority", "context_fictional"]
NON_NEUTRAL_CONDITIONS = CO + MG + CONTEXT


def load_condition_activations(extraction_dir, model_alias, ids_key, condition):
    path = os.path.join(extraction_dir, f"{model_alias}_{ids_key}_{condition}_activations.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing extraction output: {path} -- run extract_experiment2_activations.py first")
    return torch.load(path, map_location="cpu")


def diff_vecs(neutral_acts, cond_acts, ids, layer, position):
    return {i: cond_acts[i][position][layer] - neutral_acts[i][position][layer] for i in ids}


def mean_norms(all_diff_vecs, conditions, ids):
    """Magnitude signal alongside the direction-only cosine analysis --
    see analyze_experiment1_geometry.py's mean_norms for the rationale."""
    out = {}
    for c in conditions:
        norms = torch.stack([all_diff_vecs[c][i] for i in ids]).norm(dim=-1)
        out[c] = {"mean_norm": norms.mean().item(), "std_norm": norms.std().item(),
                  "min_norm": norms.min().item(), "max_norm": norms.max().item()}
    return out


def layerwise_sweep(neutral_acts, cond_acts, ids, position, n_layers_total, partitions, canonical_idx):
    """Point-estimate ONLY (no bootstrap) S_CO/S_MG/S_Context/S_between/T/
    canonical-partition-rank at EVERY layer -- see
    analyze_experiment1_geometry.py's layerwise_sweep for the rationale
    (Sec 4.2's 'full-layer' framing, reused here for Sec 6.1)."""
    rows = []
    for layer in range(n_layers_total):
        diff_vecs_l = {m: diff_vecs(neutral_acts, cond_acts[m], ids, layer, position) for m in NON_NEUTRAL_CONDITIONS}
        vecs_l = {m: aggregate(torch.stack([diff_vecs_l[m][i] for i in ids])) for m in NON_NEUTRAL_CONDITIONS}
        matrix, index_of = pairwise_cosine_matrix(vecs_l, NON_NEUTRAL_CONDITIONS)
        S_co, S_mg, S_ctx, S_between, T = compute_3group_partition_stats(vecs_l, CO, MG, CONTEXT)
        all_T = [partition_T_from_matrix(matrix, index_of, [a, b, c]) for a, b, c in partitions]
        ranks = rank_partitions(all_T)
        rows.append({
            "layer": layer, "S_CO": S_co, "S_MG": S_mg, "S_Context": S_ctx,
            "S_between": S_between, "T": T, "canonical_partition_rank": ranks[canonical_idx],
        })
    return rows


def bootstrap_representation(all_diff_vecs, clusters, partitions, canonical_idx):
    rng = random.Random(BOOTSTRAP_SEED)
    n_clusters = len(clusters)
    S_co_l, S_mg_l, S_ctx_l, S_between_l, T_l, canonical_ranks = [], [], [], [], [], []

    for _ in range(N_BOOTSTRAP):
        draws = [clusters[rng.randrange(n_clusters)] for _ in range(n_clusters)]
        resampled_ids = [i for cluster in draws for i in cluster]
        vecs = {m: aggregate(torch.stack([all_diff_vecs[m][i] for i in resampled_ids])) for m in NON_NEUTRAL_CONDITIONS}
        S_co, S_mg, S_ctx, S_between, T = compute_3group_partition_stats(vecs, CO, MG, CONTEXT)
        S_co_l.append(S_co); S_mg_l.append(S_mg); S_ctx_l.append(S_ctx); S_between_l.append(S_between); T_l.append(T)
        # Precompute the 9x9 cosine matrix ONCE per replicate (vectorized) --
        # the naive per-partition cos() approach (280 partitions x 2000 reps)
        # is too slow; matrix lookups make this tractable.
        matrix, index_of = pairwise_cosine_matrix(vecs, NON_NEUTRAL_CONDITIONS)
        all_T = [partition_T_from_matrix(matrix, index_of, [a, b, c]) for a, b, c in partitions]
        ranks = rank_partitions(all_T)
        canonical_ranks.append(ranks[canonical_idx])

    def summarize(values):
        t = torch.tensor(values)
        return {"mean": t.mean().item(), "median": t.median().item(),
                "ci95": [t.quantile(0.025).item(), t.quantile(0.975).item()]}

    return {
        "S_CO": summarize(S_co_l), "S_MG": summarize(S_mg_l), "S_Context": summarize(S_ctx_l),
        "S_between": summarize(S_between_l), "T": summarize(T_l),
        "P_canonical_rank_1": sum(1 for r in canonical_ranks if r == 1) / N_BOOTSTRAP,
        "n_bootstrap": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED,
    }


def context_leave_one_out(all_diff_vecs, ids):
    """For each pair of 2-of-3 Context conditions, cosine between their
    point-estimate directions, and cosine of that pair's centroid vs CO
    and vs MG centroids -- Sec 6.1's leave-one-out stability check."""
    co_vec = aggregate(torch.stack([torch.stack([all_diff_vecs[m][i] for i in ids]).mean(0) for m in CO]))
    mg_vec = aggregate(torch.stack([torch.stack([all_diff_vecs[m][i] for i in ids]).mean(0) for m in MG]))

    out = {}
    for held_out in CONTEXT:
        remaining = [m for m in CONTEXT if m != held_out]
        pair_vecs = [torch.stack([all_diff_vecs[m][i] for i in ids]).mean(0) for m in remaining]
        pair_cos = cos(pair_vecs[0], pair_vecs[1])
        pair_centroid = aggregate(torch.stack(pair_vecs))
        out[held_out] = {
            "remaining_pair": remaining,
            "pair_internal_cosine": pair_cos,
            "pair_centroid_vs_CO": cos(pair_centroid, co_vec),
            "pair_centroid_vs_MG": cos(pair_centroid, mg_vec),
        }
    return out


def evaluate_minimum_evidence(reliability, cosine_result, leave_one_out, canonical_rank):
    """Reports the 5 Sec 6.1 criteria as VALUES + an operational proxy
    pass/fail -- documented here, not frozen elsewhere, since Sec 6.1
    gives no numeric threshold. Do not treat `all_criteria_met` as an
    automatic final verdict without human review of the underlying
    values and the operational proxies used."""
    criterion_1 = {c: reliability[c]["split_half_cosine_ci95"][0] > 0 for c in CONTEXT}
    criterion_1_met = all(criterion_1.values())

    S_context = cosine_result["S_Context"]
    S_context_vs_CO = cosine_result["S_Context_vs_CO"]
    S_context_vs_MG = cosine_result["S_Context_vs_MG"]
    criterion_2_met = S_context > S_context_vs_CO and S_context > S_context_vs_MG

    criterion_3_met = all(
        v["pair_centroid_vs_CO"] < v["pair_internal_cosine"] and v["pair_centroid_vs_MG"] < v["pair_internal_cosine"]
        for v in leave_one_out.values()
    )

    criterion_4_met = canonical_rank == 1

    return {
        "criterion_1_all_context_reliable": {"per_condition": criterion_1, "met": criterion_1_met,
                                              "proxy": "split_half_cosine_ci95 lower bound > 0"},
        "criterion_2_context_internal_gt_between": {
            "S_Context": S_context, "S_Context_vs_CO": S_context_vs_CO, "S_Context_vs_MG": S_context_vs_MG,
            "met": criterion_2_met,
        },
        "criterion_3_leave_one_out_stable": {"per_holdout": leave_one_out, "met": criterion_3_met},
        "criterion_4_partition_best_rank": {"canonical_partition_rank": canonical_rank, "met": criterion_4_met},
        "criterion_5_replication_2_of_3_models": "NOT evaluated by this script -- read all 3 models' "
                                                  "output files together to assess this criterion.",
        "all_4_single_model_criteria_met": criterion_1_met and criterion_2_met and criterion_3_met and criterion_4_met,
    }


def run_representation(model_alias, primary_layer, extraction_dir, output_dir):
    with open(SAMPLED_PROMPTS_EN_ONLY_PATH, "r", encoding="utf-8") as f:
        pool = json.load(f)
    instruction_en_by_id = {row["id"]: row["instruction_en"] for row in pool}

    neutral_acts = load_condition_activations(extraction_dir, model_alias, "direction_ids", "neutral")
    cond_acts = {m: load_condition_activations(extraction_dir, model_alias, "direction_ids", m) for m in NON_NEUTRAL_CONDITIONS}

    ids = sorted(set(neutral_acts.keys()) & set.intersection(*[set(cond_acts[m].keys()) for m in NON_NEUTRAL_CONDITIONS]))
    if not ids:
        raise ValueError(f"no common instruction_ids across neutral/{NON_NEUTRAL_CONDITIONS} for {model_alias}")
    clusters = load_instruction_clusters(ids, instruction_en_by_id)

    by_position = {}
    for position in (PRIMARY_TOKEN_POSITION, SECONDARY_TOKEN_POSITION):
        all_diff_vecs = {m: diff_vecs(neutral_acts, cond_acts[m], ids, primary_layer, position) for m in NON_NEUTRAL_CONDITIONS}
        vecs = {m: aggregate(torch.stack([all_diff_vecs[m][i] for i in ids])) for m in NON_NEUTRAL_CONDITIONS}

        cosine_matrix = {a: {b: cos(vecs[a], vecs[b]) for b in NON_NEUTRAL_CONDITIONS} for a in NON_NEUTRAL_CONDITIONS}
        S_co, S_mg, S_ctx, S_between, T = compute_3group_partition_stats(vecs, CO, MG, CONTEXT)

        def between(g1, g2):
            pairs = [cos(vecs[x], vecs[y]) for x in g1 for y in g2]
            return sum(pairs) / len(pairs)

        cosine_result = {
            "S_CO": S_co, "S_MG": S_mg, "S_Context": S_ctx, "S_between_all": S_between, "T_partition": T,
            "S_Context_vs_CO": between(CONTEXT, CO), "S_Context_vs_MG": between(CONTEXT, MG),
            "S_CO_vs_MG": between(CO, MG),
        }

        partitions = all_3group_partitions(NON_NEUTRAL_CONDITIONS)
        canonical_idx = next(i for i, (a, b, c) in enumerate(partitions)
                              if {frozenset(a), frozenset(b), frozenset(c)} == {frozenset(CO), frozenset(MG), frozenset(CONTEXT)})
        all_T = [compute_3group_partition_stats(vecs, a, b, c)[4] for a, b, c in partitions]
        ranks = rank_partitions(all_T)
        canonical_rank = ranks[canonical_idx]

        reliability = {m: split_half_reliability(all_diff_vecs[m], ids, n_reps=SPLIT_HALF_REPS, seed=SPLIT_HALF_SEED) for m in NON_NEUTRAL_CONDITIONS}
        leave_one_out = context_leave_one_out(all_diff_vecs, ids)
        minimum_evidence = evaluate_minimum_evidence(reliability, cosine_result, leave_one_out, canonical_rank)
        bootstrap = bootstrap_representation(all_diff_vecs, clusters, partitions, canonical_idx)
        n_layers_total = cond_acts[NON_NEUTRAL_CONDITIONS[0]][ids[0]][position].shape[0]

        by_position[position] = {
            "n_instructions": len(ids), "n_instruction_clusters": len(clusters),
            "cosine_matrix": cosine_matrix, "cosine_result": cosine_result,
            "n_partitions": len(partitions), "canonical_partition_rank": canonical_rank,
            "mean_norms": mean_norms(all_diff_vecs, NON_NEUTRAL_CONDITIONS, ids),
            "reliability": reliability, "context_leave_one_out": leave_one_out,
            "minimum_evidence": minimum_evidence, "bootstrap": bootstrap,
            "layerwise_sweep_point_estimate_only": layerwise_sweep(
                neutral_acts, cond_acts, ids, position, n_layers_total, partitions, canonical_idx),
        }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_alias}_experiment2_representation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "result_status": "EXPERIMENT2_REPRESENTATION_ANALYSIS",
            "model_alias": model_alias, "primary_layer": primary_layer,
            "CO": CO, "MG": MG, "Context": CONTEXT,
            "primary_position": PRIMARY_TOKEN_POSITION, "sensitivity_position": SECONDARY_TOKEN_POSITION,
            "by_position": by_position,
        }, f, indent=2)
    print(json.dumps({"result_status": "EXPERIMENT2_REPRESENTATION_DONE", "model_alias": model_alias, "output_path": out_path}, indent=2))
    return out_path


def pearson_correlation(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x ** 0.5 * var_y ** 0.5)


def run_validation_correlation(model_alias, primary_layer, extraction_dir, behavioral_dir, output_dir):
    """Sec 6.2: project validation_ids activations onto the ALREADY-FROZEN
    d_m (from direction_ids, computed above) and correlate with
    strict_success from run_formal_behavioral.py's judge output. This is
    the SAME model's frozen d_m -- run run_representation() first."""
    neutral_direction = load_condition_activations(extraction_dir, model_alias, "direction_ids", "neutral")
    cond_direction = {m: load_condition_activations(extraction_dir, model_alias, "direction_ids", m) for m in NON_NEUTRAL_CONDITIONS}
    direction_ids = sorted(set(neutral_direction.keys()) & set.intersection(*[set(cond_direction[m].keys()) for m in NON_NEUTRAL_CONDITIONS]))

    neutral_val = load_condition_activations(extraction_dir, model_alias, "validation_ids", "neutral")
    cond_val = {m: load_condition_activations(extraction_dir, model_alias, "validation_ids", m) for m in NON_NEUTRAL_CONDITIONS}
    validation_ids = sorted(set(neutral_val.keys()) & set.intersection(*[set(cond_val[m].keys()) for m in NON_NEUTRAL_CONDITIONS]))

    judge_path = os.path.join(behavioral_dir, f"{model_alias}_judge_records.jsonl")
    gen_path = os.path.join(behavioral_dir, f"{model_alias}_generation_records.jsonl")
    if not os.path.exists(judge_path) or not os.path.exists(gen_path):
        raise FileNotFoundError(f"missing formal behavioral output: {judge_path} / {gen_path} -- run run_formal_behavioral.py first")

    gen_by_key = {}
    with open(gen_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            gen_by_key[row["generation_key"]] = row
    strict_success_by_key = {}
    with open(judge_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            strict_success_by_key[row["generation_key"]] = row["strict_success"]

    results = {}
    for position in (PRIMARY_TOKEN_POSITION, SECONDARY_TOKEN_POSITION):
        for m in NON_NEUTRAL_CONDITIONS:
            d_m = aggregate(torch.stack([
                cond_direction[m][i][position][primary_layer] - neutral_direction[i][position][primary_layer]
                for i in direction_ids
            ]))
            d_m_unit = d_m / (d_m.norm() + 1e-12)

            projections, outcomes = [], []
            for j in validation_ids:
                gen_key = next((k for k, g in gen_by_key.items()
                                 if g["instruction_id"] == j and g["condition"] == m), None)
                if gen_key is None or gen_key not in strict_success_by_key:
                    continue
                diff = cond_val[m][j][position][primary_layer] - neutral_val[j][position][primary_layer]
                projections.append(torch.dot(diff, d_m_unit).item())
                outcomes.append(1 if strict_success_by_key[gen_key] else 0)

            key = f"{position}::{m}"
            results[key] = {
                "condition": m, "position": position, "n_validation_matched": len(projections),
                "pearson_r_projection_vs_strict_success": pearson_correlation(projections, outcomes),
            }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_alias}_experiment2_validation_correlation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "result_status": "EXPERIMENT2_VALIDATION_CORRELATION_CHECK_ONLY",
            "model_alias": model_alias,
            "note": "Sec 6.2 firewall: d_m recomputed here from direction_ids only, read-only, "
                    "never re-estimated or influenced by validation_ids activations.",
            "results": results,
        }, f, indent=2)
    print(json.dumps({"result_status": "EXPERIMENT2_VALIDATION_CORRELATION_DONE", "model_alias": model_alias, "output_path": out_path}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--primary-layer", type=int, required=True)
    parser.add_argument("--extraction-dir", default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--behavioral-dir", default=DEFAULT_BEHAVIORAL_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-validation-correlation", action="store_true",
                         help="Only run the direction_ids representation analysis (Sec 6.1); "
                              "skip Sec 6.2 (requires run_formal_behavioral.py's output).")
    args = parser.parse_args()

    run_representation(args.model_alias, args.primary_layer, args.extraction_dir, args.output_dir)
    if not args.skip_validation_correlation:
        run_validation_correlation(args.model_alias, args.primary_layer, args.extraction_dir, args.behavioral_dir, args.output_dir)


if __name__ == "__main__":
    main()
