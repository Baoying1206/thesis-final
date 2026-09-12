"""Study B (RQ2, FINAL_STUDY_PROTOCOL.md Sec 5R, Round 17) analysis.
Loads the RAW per-condition activation `.pt` files produced by
`extract_study_b_activations.py` (both `direction_ids` and, if present,
`validation_ids`) plus Study A's (Experiment 1's) raw `.pt` files for
the frozen p_CO/p_MG reference directions (Sec 5R.4.7) -- never loads
a model, never touches GPU.

Computes, per family (persona/authority/fictional), at the model's
frozen primary layer:
- 5R.4.2 effect magnitude M[f,t] for t=1..4 (P vs N, `direction_ids`)
- 5R.4.3 turn-to-turn cosine consistency
- 5R.4.4 raw residual r_f (P stage-4 vs S) -- SECONDARY, confounded,
  reported alongside but never instead of:
- 5R.4.5 difference-in-differences I_f^repr = (P-N)-(S-C) -- PRIMARY
  representational estimand
- 5R.4.6 activation-behavior connection z[i] vs strict_success
  (`validation_ids` only, requires judge records from
  `extract_study_b_activations.py`'s `validation_ids` run)
- 5R.4.7 cos(I_f^repr, p_CO), cos(I_f^repr, p_MG) -- Study A's frozen
  directions, recomputed from Experiment 1's raw activations, never
  from Study A's JSON summary (which does not persist the raw vectors)
- 5R.5 behavioral I_f^ASR = (ASR_P-ASR_N)-(ASR_S-ASR_C), plus
  per-condition ASR/NRR/HRR/judge-parse-failure rate (`validation_ids`
  only)

Every bootstrap replicate fully recomputes every derived quantity from
its own resampled instruction-clusters (Sec 5R.2's frozen rule) -- see
`src/stats_shared.py`'s `bootstrap_did_vector` / `bootstrap_did_scalar`
/ `bootstrap_vector_diff` / `point_biserial_bootstrap`, all validated
against synthetic fixtures with known-sign effects before use here.

Reported per model separately (Sec 5R.6) -- run once per `--model-alias`;
cross-model comparison (>=2/3 models agreeing, CI not crossing zero) is
done by reading the 3 output files together, never inside this script.

**This script has never been run against real data (Study B has never
been extracted on GPU) -- only the underlying stats_shared functions
have been validated, against synthetic fixtures, not this script's own
data-loading/aggregation logic end to end.**
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

from stats_shared import (  # noqa: E402
    aggregate, cos, load_instruction_clusters, holm_correction,
    bootstrap_did_scalar, bootstrap_did_vector, bootstrap_vector_diff, point_biserial_bootstrap,
)
from study_b_loader import FAMILIES, condition_name  # noqa: E402

SAMPLED_PROMPTS_EN_ONLY_PATH = os.path.join(REPO_ROOT, "data", "source", "sampled_prompts_en_only.json")
DEFAULT_STUDY_B_DIR = os.path.join(SCRIPT_DIR, "study_b_output")
DEFAULT_EXPERIMENT1_DIR = os.path.join(SCRIPT_DIR, "experiment1_output")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "study_b_analysis")

PRIMARY_TOKEN_POSITION = "t_generation_boundary"
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260828

CO_MECHANISMS = ["prefix_injection", "refusal_suppression", "persona_roleplay"]
MG_MECHANISMS = ["encoding_obfuscation", "payload_splitting", "distractors_negated"]


def load_instruction_texts():
    with open(SAMPLED_PROMPTS_EN_ONLY_PATH, "r", encoding="utf-8") as f:
        pool = json.load(f)
    return {row["id"]: row["instruction_en"] for row in pool}


def load_study_b_activations(study_b_dir, model_alias, ids_key, family, form):
    path = os.path.join(study_b_dir, f"{model_alias}_{ids_key}_{condition_name(family, form)}_activations.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing Study B extraction output: {path} -- run extract_study_b_activations.py first")
    return torch.load(path, map_location="cpu")


def stage_vecs(raw_acts, stage_key, layer, ids):
    """raw_acts: {instruction_id: {'stage_t': [n_layers+1, hidden]} or {PRIMARY_TOKEN_POSITION: [...]}}."""
    out = {}
    for i in ids:
        if i not in raw_acts:
            continue
        entry = raw_acts[i]
        vec_all_layers = entry.get(stage_key) if stage_key in entry else entry.get(PRIMARY_TOKEN_POSITION)
        if vec_all_layers is None:
            continue
        out[i] = vec_all_layers[layer]
    return out


def load_experiment1_frozen_directions(experiment1_dir, model_alias, layer, position, ids):
    """Recomputes Study A's placebo-calibrated CO/MG group directions
    from Experiment 1's raw .pt files (Sec 5R.4.7) -- Study A's JSON
    summary does not persist the raw vectors, only scalar cosines, so
    this cannot be read from there."""
    def load_cond(name):
        path = os.path.join(experiment1_dir, f"{model_alias}_{name}_activations.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing Experiment 1 (Study A) output: {path}")
        return torch.load(path, map_location="cpu")

    plain_acts = load_cond("plain")
    placebo_acts = load_cond("placebo")

    def mech_direction(mech_name):
        mech_acts = load_cond(mech_name)
        diffs = []
        for i in ids:
            if i not in mech_acts or i not in plain_acts or i not in placebo_acts:
                continue
            raw_diff = mech_acts[i][position][layer] - plain_acts[i][position][layer]
            placebo_diff = placebo_acts[i][position][layer] - plain_acts[i][position][layer]
            diffs.append(raw_diff - placebo_diff)
        return torch.stack(diffs).mean(0)

    co_dirs = torch.stack([mech_direction(m) for m in CO_MECHANISMS])
    mg_dirs = torch.stack([mech_direction(m) for m in MG_MECHANISMS])
    return co_dirs.mean(0), mg_dirs.mean(0)


def run_representation(model_alias, primary_layer, study_b_dir, experiment1_dir, output_dir):
    print(f"[{model_alias}] loading direction_ids Study B activations...", file=sys.stderr)
    instruction_texts = load_instruction_texts()

    family_results = {}
    for family in FAMILIES:
        print(f"[{model_alias}] family={family}: loading P/N/S/C...", file=sys.stderr)
        acts = {form: load_study_b_activations(study_b_dir, model_alias, "direction_ids", family, form)
                for form in ("P", "N", "S", "C")}

        # ids present in ALL four conditions (P/N need all 4 stages; S/C are single-point)
        ids_p = set(acts["P"].keys())
        ids_common = ids_p & set(acts["N"].keys()) & set(acts["S"].keys()) & set(acts["C"].keys())
        ids_common = sorted(ids_common)
        clusters = load_instruction_clusters(ids_common, instruction_texts)
        print(f"[{model_alias}] family={family}: {len(ids_common)} common instructions, {len(clusters)} clusters", file=sys.stderr)

        # 5R.4.2/5R.4.3: per-stage P-vs-N effect magnitude and turn-to-turn consistency (point estimate only)
        per_stage_mean_diff = {}
        for t in (1, 2, 3, 4):
            key = f"stage_{t}"
            vp = stage_vecs(acts["P"], key, primary_layer, ids_common)
            vn = stage_vecs(acts["N"], key, primary_layer, ids_common)
            diffs = [vp[i] - vn[i] for i in ids_common if i in vp and i in vn]
            per_stage_mean_diff[t] = torch.stack(diffs).mean(0) if diffs else None

        M = {t: (per_stage_mean_diff[t].norm().item() if per_stage_mean_diff[t] is not None else None) for t in (1, 2, 3, 4)}
        consistency = {}
        for t in (1, 2, 3):
            a, b = per_stage_mean_diff[t], per_stage_mean_diff[t + 1]
            consistency[f"{t}_to_{t+1}"] = cos(a, b) if (a is not None and b is not None) else None

        # 5R.4.4: raw residual r_f (P stage-4 vs S) -- secondary, confounded
        vp4 = stage_vecs(acts["P"], "stage_4", primary_layer, ids_common)
        vs = stage_vecs(acts["S"], PRIMARY_TOKEN_POSITION, primary_layer, ids_common)
        print(f"[{model_alias}] family={family}: bootstrapping r_f (secondary, confounded)...", file=sys.stderr)
        r_f = bootstrap_vector_diff(vp4, vs, clusters, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)

        # 5R.4.5: primary DiD estimand
        vn4 = stage_vecs(acts["N"], "stage_4", primary_layer, ids_common)
        vc = stage_vecs(acts["C"], PRIMARY_TOKEN_POSITION, primary_layer, ids_common)
        print(f"[{model_alias}] family={family}: bootstrapping I_f^repr (primary DiD)...", file=sys.stderr)
        p_CO, p_MG = load_experiment1_frozen_directions(experiment1_dir, model_alias, primary_layer, PRIMARY_TOKEN_POSITION, ids_common)
        I_repr = bootstrap_did_vector(
            vp4, vn4, vs, vc, clusters, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED,
            extra_cos_targets={"p_CO": p_CO, "p_MG": p_MG},
        )

        family_results[family] = {
            "n_instructions": len(ids_common),
            "M_per_stage": M,
            "turn_to_turn_cosine_consistency": consistency,
            "r_f_secondary_confounded": r_f,
            "I_f_repr_primary_DiD": I_repr,
        }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_alias}_study_b_representation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "result_status": "STUDY_B_REPRESENTATION_ANALYSIS",
            "model_alias": model_alias, "primary_layer": primary_layer,
            "note": "5R.4.4's r_f is SECONDARY/confounded (raw context-length effect not controlled); 5R.4.5's I_f_repr is the PRIMARY causal estimand (difference-in-differences).",
            "by_family": family_results,
        }, f, indent=2, ensure_ascii=False)
    print(json.dumps({"result_status": "STUDY_B_REPRESENTATION_DONE", "model_alias": model_alias, "output_path": out_path}, indent=2))


def run_behavioral(model_alias, study_b_dir, output_dir):
    """Sec 5R.5/5R.4.6 -- requires extract_study_b_activations.py's
    `validation_ids` run (generation + WildGuard judging)."""
    judge_path = os.path.join(study_b_dir, f"{model_alias}_validation_ids_study_b_judge_records.jsonl")
    if not os.path.exists(judge_path):
        raise FileNotFoundError(f"missing judge records: {judge_path} -- run extract_study_b_activations.py --ids-key validation_ids first")

    instruction_texts = load_instruction_texts()
    with open(judge_path, "r", encoding="utf-8") as f:
        judge_records = [json.loads(line) for line in f]

    by_condition_instruction = {}
    parse_fail_by_condition = {}
    for r in judge_records:
        cond, iid = r["condition"], r["instruction_id"]
        by_condition_instruction.setdefault(cond, {})[iid] = r
        parse_fail_by_condition.setdefault(cond, [0, 0])
        parse_fail_by_condition[cond][1] += 1
        if not r.get("parse_success", True):
            parse_fail_by_condition[cond][0] += 1

    family_results = {}
    for family in FAMILIES:
        recs = {form: by_condition_instruction.get(condition_name(family, form), {}) for form in ("P", "N", "S", "C")}
        ids_common = sorted(set(recs["P"]) & set(recs["N"]) & set(recs["S"]) & set(recs["C"]))
        clusters = load_instruction_clusters(ids_common, instruction_texts)

        def strict_success_metric(form):
            return {i: (1.0 if recs[form][i].get("strict_success") else 0.0) for i in ids_common if i in recs[form]}

        metric_P, metric_N, metric_S, metric_C = (strict_success_metric(f) for f in ("P", "N", "S", "C"))
        print(f"[{model_alias}] family={family}: bootstrapping I_f^ASR...", file=sys.stderr)
        I_asr = bootstrap_did_scalar(metric_P, metric_N, metric_S, metric_C, clusters, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)

        per_condition_rates = {}
        for form in ("P", "N", "S", "C"):
            r = recs[form]
            n = len(r)
            if n == 0:
                per_condition_rates[form] = None
                continue
            per_condition_rates[form] = {
                "n": n,
                "strict_success_rate": sum(1 for v in r.values() if v.get("strict_success")) / n,
                "refusal_rate": sum(1 for v in r.values() if v.get("response_refusal") == 1) / n,
                "harmful_response_rate": sum(1 for v in r.values() if v.get("response_harmfulness") == 1) / n,
                "judge_parse_failure_rate": sum(1 for v in r.values() if not v.get("parse_success", True)) / n,
            }

        family_results[family] = {
            "n_instructions": len(ids_common),
            "I_f_ASR_primary_DiD": I_asr,
            "per_condition_rates": per_condition_rates,
        }

    # Sec 5R.6: Holm correction WITHIN this model, across its 3 families'
    # I_f^ASR p-values -- never pooled across models. Added this round
    # after real results showed borderline p-values (e.g. p=0.066) that
    # must not be read as "near-significant" without correction.
    named_pvalues = [(fam, family_results[fam]["I_f_ASR_primary_DiD"]["p_two_sided"])
                      for fam in FAMILIES if family_results[fam]["I_f_ASR_primary_DiD"]["p_two_sided"] is not None]
    adjusted = holm_correction(named_pvalues)
    for fam, p_holm in adjusted.items():
        family_results[fam]["I_f_ASR_primary_DiD"]["p_holm_adjusted"] = p_holm

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_alias}_study_b_behavioral.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "result_status": "STUDY_B_BEHAVIORAL_ANALYSIS",
            "model_alias": model_alias,
            "note": "I_f_ASR_primary_DiD.p_holm_adjusted is Holm-corrected across this model's 3 families (Sec 5R.6) -- never compare the raw p_two_sided across families without it.",
            "by_family": family_results,
        }, f, indent=2, ensure_ascii=False)
    print(json.dumps({"result_status": "STUDY_B_BEHAVIORAL_DONE", "model_alias": model_alias, "output_path": out_path}, indent=2))


def run_activation_behavior_connection(model_alias, primary_layer, study_b_dir, output_dir):
    """Sec 5R.4.6: is z[i] = <h[i,f,P,4]-h[i,f,N,4], d_hat[f]> predictive
    of validation_ids's strict_success? d_hat[f] is I_f^repr's direction,
    estimated from direction_ids ONLY (Sec 5R.7's firewall) -- read from
    the already-written representation-analysis JSON's point estimate,
    recomputed here from the raw direction_ids vectors (never from
    validation_ids) to keep the vector, not just its norm."""
    judge_path = os.path.join(study_b_dir, f"{model_alias}_validation_ids_study_b_judge_records.jsonl")
    if not os.path.exists(judge_path):
        raise FileNotFoundError(f"missing judge records: {judge_path}")
    with open(judge_path, "r", encoding="utf-8") as f:
        judge_records = [json.loads(line) for line in f]
    outcome_by_condition_instruction = {}
    for r in judge_records:
        outcome_by_condition_instruction.setdefault(r["condition"], {})[r["instruction_id"]] = 1 if r.get("strict_success") else 0

    instruction_texts = load_instruction_texts()
    family_results = {}
    for family in FAMILIES:
        # d_hat[f]: I_f^repr's direction from direction_ids (Sec 5R.4.5), unit-normalized
        dir_acts = {form: load_study_b_activations(study_b_dir, model_alias, "direction_ids", family, form) for form in ("P", "N", "S", "C")}
        dir_ids_common = sorted(set(dir_acts["P"]) & set(dir_acts["N"]) & set(dir_acts["S"]) & set(dir_acts["C"]))
        vp4 = stage_vecs(dir_acts["P"], "stage_4", primary_layer, dir_ids_common)
        vn4 = stage_vecs(dir_acts["N"], "stage_4", primary_layer, dir_ids_common)
        vs = stage_vecs(dir_acts["S"], PRIMARY_TOKEN_POSITION, primary_layer, dir_ids_common)
        vc = stage_vecs(dir_acts["C"], PRIMARY_TOKEN_POSITION, primary_layer, dir_ids_common)
        mean_p4, mean_n4 = aggregate(torch.stack([vp4[i] for i in dir_ids_common])), aggregate(torch.stack([vn4[i] for i in dir_ids_common]))
        mean_s, mean_c = aggregate(torch.stack([vs[i] for i in dir_ids_common])), aggregate(torch.stack([vc[i] for i in dir_ids_common]))
        I_repr_point = (mean_p4 - mean_n4) - (mean_s - mean_c)
        d_hat = I_repr_point / I_repr_point.norm()

        # z[i] on validation_ids: <h[i,P,4]-h[i,N,4], d_hat>
        val_acts_P = load_study_b_activations(study_b_dir, model_alias, "validation_ids", family, "P")
        val_acts_N = load_study_b_activations(study_b_dir, model_alias, "validation_ids", family, "N")
        val_ids_common = sorted(set(val_acts_P) & set(val_acts_N) & set(outcome_by_condition_instruction.get(condition_name(family, "P"), {})))
        z_by_id = {}
        for i in val_ids_common:
            vp_i = val_acts_P[i]["stage_4"][primary_layer]
            vn_i = val_acts_N[i]["stage_4"][primary_layer]
            z_by_id[i] = torch.dot(vp_i - vn_i, d_hat).item()
        outcome_by_id = {i: outcome_by_condition_instruction[condition_name(family, "P")][i] for i in val_ids_common}

        clusters = load_instruction_clusters(val_ids_common, instruction_texts)
        print(f"[{model_alias}] family={family}: bootstrapping z-vs-strict_success connection...", file=sys.stderr)
        connection = point_biserial_bootstrap(z_by_id, outcome_by_id, clusters, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)
        family_results[family] = {"n_instructions": len(val_ids_common), "z_vs_strict_success": connection}

    # Sec 5R.6: Holm correction WITHIN this model, across its 3 families'
    # r_p_two_sided (point-biserial correlation p-values) -- same
    # discipline as run_behavioral()'s I_f^ASR correction, added this
    # round after real borderline results (e.g. Llama's z-correlations
    # sitting right at the edge of significance) made this necessary.
    named_pvalues = [(fam, family_results[fam]["z_vs_strict_success"]["r_p_two_sided"])
                      for fam in FAMILIES if family_results[fam]["z_vs_strict_success"]["r_p_two_sided"] is not None]
    adjusted = holm_correction(named_pvalues)
    for fam, p_holm in adjusted.items():
        family_results[fam]["z_vs_strict_success"]["r_p_holm_adjusted"] = p_holm

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_alias}_study_b_activation_behavior_connection.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "result_status": "STUDY_B_ACTIVATION_BEHAVIOR_CONNECTION",
            "model_alias": model_alias, "primary_layer": primary_layer,
            "note": "d_hat[f] estimated from direction_ids ONLY (Sec 5R.7 firewall); z[i] and strict_success both from validation_ids. r_p_holm_adjusted is Holm-corrected across this model's 3 families (Sec 5R.6).",
            "by_family": family_results,
        }, f, indent=2, ensure_ascii=False)
    print(json.dumps({"result_status": "STUDY_B_ACTIVATION_BEHAVIOR_CONNECTION_DONE", "model_alias": model_alias, "output_path": out_path}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--primary-layer", type=int, required=True)
    parser.add_argument("--study-b-dir", default=DEFAULT_STUDY_B_DIR)
    parser.add_argument("--experiment1-dir", default=DEFAULT_EXPERIMENT1_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-representation", action="store_true")
    parser.add_argument("--skip-behavioral", action="store_true")
    parser.add_argument("--skip-connection", action="store_true", help="Skip 5R.4.6's activation-behavior connection (needs both direction_ids and validation_ids extracted).")
    args = parser.parse_args()

    if not args.skip_representation:
        run_representation(args.model_alias, args.primary_layer, args.study_b_dir, args.experiment1_dir, args.output_dir)
    if not args.skip_behavioral:
        run_behavioral(args.model_alias, args.study_b_dir, args.output_dir)
    if not args.skip_connection:
        run_activation_behavior_connection(args.model_alias, args.primary_layer, args.study_b_dir, args.output_dir)


if __name__ == "__main__":
    main()
