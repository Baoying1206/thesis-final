"""RQ2 Round 20 replacement design (history-augmented canonical CO/MG)
analysis. Single-stage design (FINAL_STUDY_PROTOCOL.md Sec 13 Round 20 --
confirmed: no discovery/confirmation split, `validation_ids` IS the one
and only confirmatory dataset, `test_ids` stays unused/sealed for this
design). Loads the RAW per-condition activation `.pt` files produced by
`extract_history_augmented_co_mg.py` (`direction_ids` and
`validation_ids`) plus Experiment 1's raw `.pt` files for the frozen
p_CO/p_MG reference directions -- never loads a model, never touches
GPU.

Behavioral (PRIMARY, `validation_ids`):
- per-mechanism ASR[m,multi], ASR[m,single] (WildGuard strict_success)
- E_CO = mean_{m in CO}(ASR[m,multi]-ASR[m,single]),
  E_MG analogously, E_N = neutral's own delta,
  Gamma = E_CO-E_MG, corrected_CO = E_CO-E_N, corrected_MG = E_MG-E_N
  (`stats_shared.bootstrap_history_augmented_effects`, validated
  against a synthetic fixture with a known injected +0.30 CO-only
  effect before being trusted here)
- Holm correction applied WITHIN this model across exactly the 3
  primary hypotheses (corrected_CO, corrected_MG, Gamma) -- this IS the
  final confirmatory result for this design; no second stage follows.

Representational (secondary, `direction_ids`):
- per-mechanism d_m = mean(h[m,multi,final_stage]) - mean(h[m,single]),
  with bootstrap CI on ||d_m|| and cos(d_m, p_CO)/cos(d_m, p_MG) where
  p_CO/p_MG are Experiment 1's own frozen, placebo-calibrated group
  directions (recomputed from Experiment 1's raw activations, never
  from its JSON summary)
- CO/MG internal cohesion: mean within-CO d_m cosine vs within-MG vs
  between (point estimate only, `stats_shared.compute_2group_partition_stats`
  reused as-is -- it only needs a {name: vector} dict and two 3-item
  group-name lists, which is exactly this design's 6 real mechanisms)

Activation-behavior connection (`validation_ids`, using `direction_ids`
-estimated directions only -- Sec 6's firewall, unchanged):
- per mechanism m, z[i] = <h[i,m,multi,final_stage]-h[i,m,single], d_hat_cat>
  where d_hat_cat is p_CO (if m in CO) or p_MG (if m in MG), unit-
  normalized; correlated against `validation_ids` strict_success via
  `point_biserial_bootstrap`. Holm-corrected across the 6 real
  mechanisms within this model.

Reported per model separately -- run once per `--model-alias`;
cross-model comparison is done by reading the 3 output files together,
never inside this script.

**This script has never been run against real data (this design has
never been extracted on GPU) -- only the underlying stats_shared
functions have been validated, against synthetic fixtures, not this
script's own data-loading/aggregation logic end to end.**
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
    cos, load_instruction_clusters, holm_correction,
    bootstrap_history_augmented_effects_multi_scaffold, bootstrap_vector_diff,
    compute_2group_partition_stats, point_biserial_bootstrap,
)
from history_augmented_co_mg_loader import (  # noqa: E402
    CO_MECHANISMS, MG_MECHANISMS, ALL_MECHANISM_GROUPS, MULTI_FORMS, FINAL_STAGE_KEY,
    condition_name, multi_form_to_scaffold_kind,
)

SAMPLED_PROMPTS_EN_ONLY_PATH = os.path.join(REPO_ROOT, "data", "source", "sampled_prompts_en_only.json")
DEFAULT_EXTRACTION_DIR = os.path.join(SCRIPT_DIR, "history_augmented_co_mg_output")
DEFAULT_EXPERIMENT1_DIR = os.path.join(SCRIPT_DIR, "experiment1_output")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "history_augmented_co_mg_analysis")

PRIMARY_TOKEN_POSITION = "t_generation_boundary"
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260828

REAL_MECHANISMS = list(CO_MECHANISMS) + list(MG_MECHANISMS)  # excludes 'neutral' -- no canonical text/direction for it


def load_instruction_texts():
    with open(SAMPLED_PROMPTS_EN_ONLY_PATH, "r", encoding="utf-8") as f:
        pool = json.load(f)
    return {row["id"]: row["instruction_en"] for row in pool}


def load_activations(extraction_dir, model_alias, ids_key, mechanism, form):
    path = os.path.join(extraction_dir, f"{model_alias}_{ids_key}_{condition_name(mechanism, form)}_activations.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing extraction output: {path} -- run extract_history_augmented_co_mg.py first")
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
    """Recomputes Experiment 1's placebo-calibrated CO/MG group
    directions from its raw .pt files -- Experiment 1's JSON summary
    does not persist the raw vectors, only scalar cosines, so this
    cannot be read from there. Identical logic to
    analyze_study_b.py's own helper (duplicated, not imported --
    Study B is SUPERSEDED and this design must not depend on it)."""
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


def run_behavioral(model_alias, extraction_dir, output_dir):
    judge_path = os.path.join(extraction_dir, f"{model_alias}_validation_ids_history_augmented_judge_records.jsonl")
    if not os.path.exists(judge_path):
        raise FileNotFoundError(f"missing judge records: {judge_path} -- run extract_history_augmented_co_mg.py --ids-key validation_ids first")

    instruction_texts = load_instruction_texts()
    with open(judge_path, "r", encoding="utf-8") as f:
        judge_records = [json.loads(line) for line in f]

    by_condition_instruction = {}
    for r in judge_records:
        by_condition_instruction.setdefault(r["condition"], {})[r["instruction_id"]] = r

    def strict_success_metric(mechanism, form):
        recs = by_condition_instruction.get(condition_name(mechanism, form), {})
        return {i: (1.0 if v.get("strict_success") else 0.0) for i, v in recs.items()}

    asr_single = {m: strict_success_metric(m, "single") for m in ALL_MECHANISM_GROUPS}
    asr_multi_by_kind = {}
    for form in MULTI_FORMS:
        kind = multi_form_to_scaffold_kind(form)
        asr_multi_by_kind[kind] = {m: strict_success_metric(m, form) for m in ALL_MECHANISM_GROUPS}

    ids_common = sorted(set.intersection(
        *[set(asr_single[m]) for m in ALL_MECHANISM_GROUPS],
        *[set(asr_multi_by_kind[kind][m]) for kind in asr_multi_by_kind for m in ALL_MECHANISM_GROUPS],
    ))
    clusters = load_instruction_clusters(ids_common, instruction_texts)
    print(f"[{model_alias}] behavioral: {len(ids_common)} instructions with all {len(ALL_MECHANISM_GROUPS) * len(MULTI_FORMS + ('single',))} conditions present, {len(clusters)} clusters", file=sys.stderr)

    asr_single_common = {m: {i: asr_single[m][i] for i in ids_common} for m in ALL_MECHANISM_GROUPS}
    asr_multi_by_kind_common = {
        kind: {m: {i: asr_multi_by_kind[kind][m][i] for i in ids_common} for m in ALL_MECHANISM_GROUPS}
        for kind in asr_multi_by_kind
    }

    print(f"[{model_alias}] bootstrapping E_CO/E_MG/E_N/Gamma/corrected_CO/corrected_MG for each scaffold kind + neutral-vs-progressive contrast...", file=sys.stderr)
    effects = bootstrap_history_augmented_effects_multi_scaffold(
        asr_single_common, asr_multi_by_kind_common, clusters, CO_MECHANISMS, MG_MECHANISMS,
        n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED,
    )

    # Sec 13 Round 20 follow-up: Holm correction WITHIN this model across
    # EXACTLY the primary hypotheses -- 3 per scaffold kind (corrected_CO,
    # corrected_MG, Gamma), pooled across the 2 kinds = 6 total -- this is
    # the final confirmatory result for this design (single-stage,
    # confirmed with the user: no discovery/confirmation split, no
    # second pass on test_ids). The neutral-vs-progressive CONTRAST is
    # reported separately, NOT folded into this same Holm family (it
    # answers a different question -- "does escalation beat neutral
    # small talk" -- from "does each scaffold beat single-turn").
    primary_pvalues = []
    for kind in effects["by_kind"]:
        for k in ("corrected_CO", "corrected_MG", "Gamma"):
            p = effects["by_kind"][kind][k]["p_two_sided"]
            if p is not None:
                primary_pvalues.append((f"{kind}.{k}", p))
    adjusted = holm_correction(primary_pvalues)
    for name, p_holm in adjusted.items():
        kind, k = name.split(".", 1)
        effects["by_kind"][kind][k]["p_holm_adjusted"] = p_holm

    per_condition_rates = {}
    for m in ALL_MECHANISM_GROUPS:
        for form in ("single",) + MULTI_FORMS:
            recs = by_condition_instruction.get(condition_name(m, form), {})
            n = len(recs)
            if n == 0:
                per_condition_rates[condition_name(m, form)] = None
                continue
            per_condition_rates[condition_name(m, form)] = {
                "n": n,
                "strict_success_rate": sum(1 for v in recs.values() if v.get("strict_success")) / n,
                "refusal_rate": sum(1 for v in recs.values() if v.get("response_refusal") == 1) / n,
                "harmful_response_rate": sum(1 for v in recs.values() if v.get("response_harmfulness") == 1) / n,
                "judge_parse_failure_rate": sum(1 for v in recs.values() if not v.get("parse_success", True)) / n,
            }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_alias}_history_augmented_behavioral.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "result_status": "HISTORY_AUGMENTED_BEHAVIORAL_ANALYSIS",
            "model_alias": model_alias, "n_instructions": len(ids_common),
            "note": "Single-stage design (Sec 13 Round 20 + follow-up) -- this IS the final confirmatory result, not a discovery stage. p_holm_adjusted is Holm-corrected across exactly {corrected_CO, corrected_MG, Gamma} x {neutral, progressive} = 6 hypotheses within this model, never pooled across models. effects.contrast (progressive-vs-neutral) is a separate, non-Holm-corrected secondary comparison -- a different question from the 6 primary hypotheses.",
            "effects": effects,
            "per_condition_rates": per_condition_rates,
        }, f, indent=2, ensure_ascii=False)
    print(json.dumps({"result_status": "HISTORY_AUGMENTED_BEHAVIORAL_DONE", "model_alias": model_alias, "output_path": out_path}, indent=2))


def run_representation(model_alias, primary_layer, extraction_dir, experiment1_dir, output_dir):
    """Memory note (real cluster incident, Sep 2026): an earlier version of
    this function (a) preloaded ALL 6 mechanisms' multi-form direction_ids
    .pt files into one dict simultaneously (each ~300 instructions x 6
    stored stages, large) and (b) called load_experiment1_frozen_directions
    -- which itself reloads Experiment 1's plain/placebo/6-mechanism .pt
    files -- once per (form, mechanism) = 12 times redundantly, since
    p_CO/p_MG do not depend on this design's scaffold kind at all. Combined,
    this got the CPU-partition analysis job OOM-killed by the kernel
    (`Killed`, no Python traceback) on a real run. Fixed by computing
    p_CO/p_MG exactly once, and loading/freeing one mechanism's activation
    files at a time instead of all 6 at once."""
    instruction_texts = load_instruction_texts()

    probe_acts = load_activations(extraction_dir, model_alias, "direction_ids", REAL_MECHANISMS[0], "single")
    all_direction_ids = sorted(probe_acts.keys())
    del probe_acts
    p_CO, p_MG = load_experiment1_frozen_directions(experiment1_dir, model_alias, primary_layer, PRIMARY_TOKEN_POSITION, all_direction_ids)

    by_kind_results = {}
    for form in MULTI_FORMS:
        kind = multi_form_to_scaffold_kind(form)

        d_m_point = {}
        mechanism_results = {}
        for m in REAL_MECHANISMS:
            acts_multi_m = load_activations(extraction_dir, model_alias, "direction_ids", m, form)
            acts_single_m = load_activations(extraction_dir, model_alias, "direction_ids", m, "single")
            ids_common = sorted(set(acts_multi_m) & set(acts_single_m))
            clusters = load_instruction_clusters(ids_common, instruction_texts)
            v_multi = stage_vecs(acts_multi_m, FINAL_STAGE_KEY, primary_layer, ids_common)
            v_single = stage_vecs(acts_single_m, PRIMARY_TOKEN_POSITION, primary_layer, ids_common)
            del acts_multi_m, acts_single_m

            print(f"[{model_alias}] kind={kind} mechanism={m}: bootstrapping d_m^history...", file=sys.stderr)
            d_m = bootstrap_vector_diff(v_multi, v_single, clusters, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED,
                                         extra_cos_targets={"p_CO": p_CO, "p_MG": p_MG})
            mechanism_results[m] = {"n_instructions": len(ids_common), "d_m_history": d_m}

            common_diffs = [v_multi[i] - v_single[i] for i in ids_common if i in v_multi and i in v_single]
            d_m_point[m] = torch.stack(common_diffs).mean(0) if common_diffs else None

        cohesion = None
        if all(d_m_point[m] is not None for m in REAL_MECHANISMS):
            S_co, S_mg, S_between, delta_co, delta_mg, T = compute_2group_partition_stats(d_m_point, list(CO_MECHANISMS), list(MG_MECHANISMS))
            cohesion = {
                "within_CO_mean_cosine": S_co, "within_MG_mean_cosine": S_mg, "between_CO_MG_mean_cosine": S_between,
                "CO_minus_between": delta_co, "MG_minus_between": delta_mg, "T_statistic": T,
                "note": "point estimate only (no bootstrap CI) -- descriptive cohesion check on the canonical CO/MG partition of the 6 d_m^history vectors, same statistic family as Experiment 1 Sec 4.3 but not re-run through its partition-ranking bootstrap this round.",
            }

        by_kind_results[kind] = {"by_mechanism": mechanism_results, "co_mg_cohesion": cohesion}

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_alias}_history_augmented_representation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "result_status": "HISTORY_AUGMENTED_REPRESENTATION_ANALYSIS",
            "model_alias": model_alias, "primary_layer": primary_layer,
            "by_scaffold_kind": by_kind_results,
        }, f, indent=2, ensure_ascii=False)
    print(json.dumps({"result_status": "HISTORY_AUGMENTED_REPRESENTATION_DONE", "model_alias": model_alias, "output_path": out_path}, indent=2))


def run_activation_behavior_connection(model_alias, primary_layer, extraction_dir, experiment1_dir, output_dir):
    """For each scaffold kind, z[i] = <h[i,m,multi_kind,final_stage]-h[i,m,single], d_hat_cat>,
    where d_hat_cat is Experiment 1's frozen p_CO (if m in CO) or p_MG
    (if m in MG), unit-normalized -- estimated from `direction_ids`
    ONLY (Sec 6/5R.7's firewall, unchanged), correlated against
    `validation_ids` strict_success. Holm-corrected across the 6 real
    mechanisms WITHIN each scaffold kind separately (not pooled across
    kinds -- each kind's 6 mechanisms is its own family)."""
    judge_path = os.path.join(extraction_dir, f"{model_alias}_validation_ids_history_augmented_judge_records.jsonl")
    if not os.path.exists(judge_path):
        raise FileNotFoundError(f"missing judge records: {judge_path}")
    with open(judge_path, "r", encoding="utf-8") as f:
        judge_records = [json.loads(line) for line in f]
    outcome_by_condition_instruction = {}
    for r in judge_records:
        outcome_by_condition_instruction.setdefault(r["condition"], {})[r["instruction_id"]] = 1 if r.get("strict_success") else 0

    instruction_texts = load_instruction_texts()

    # d_hat direction from direction_ids ONLY (fixed reference, Experiment 1's own directions) --
    # scaffold-independent, so estimated once using the single-form ids (any direction_ids form works here).
    all_dir_ids = set()
    for m in REAL_MECHANISMS:
        acts_single_m = load_activations(extraction_dir, model_alias, "direction_ids", m, "single")
        all_dir_ids |= set(acts_single_m.keys())
    p_CO, p_MG = load_experiment1_frozen_directions(experiment1_dir, model_alias, primary_layer, PRIMARY_TOKEN_POSITION, sorted(all_dir_ids))
    d_hat_CO = p_CO / p_CO.norm()
    d_hat_MG = p_MG / p_MG.norm()

    by_kind_results = {}
    for form in MULTI_FORMS:
        kind = multi_form_to_scaffold_kind(form)
        mechanism_results = {}
        for m in REAL_MECHANISMS:
            d_hat = d_hat_CO if m in CO_MECHANISMS else d_hat_MG
            acts_multi = load_activations(extraction_dir, model_alias, "validation_ids", m, form)
            acts_single = load_activations(extraction_dir, model_alias, "validation_ids", m, "single")
            outcome = outcome_by_condition_instruction.get(condition_name(m, form), {})
            ids_common = sorted(set(acts_multi) & set(acts_single) & set(outcome))

            z_by_id = {}
            for i in ids_common:
                v_multi_i = acts_multi[i][FINAL_STAGE_KEY][primary_layer]
                v_single_i = acts_single[i][PRIMARY_TOKEN_POSITION][primary_layer]
                z_by_id[i] = torch.dot(v_multi_i - v_single_i, d_hat).item()
            outcome_by_id = {i: outcome[i] for i in ids_common}

            clusters = load_instruction_clusters(ids_common, instruction_texts)
            print(f"[{model_alias}] kind={kind} mechanism={m}: bootstrapping z-vs-strict_success connection...", file=sys.stderr)
            connection = point_biserial_bootstrap(z_by_id, outcome_by_id, clusters, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)

            # Diagnostics (Sec 13 follow-up, real-data review Sep 2026): a
            # handful of small-|r| point-biserial correlations came back
            # Holm-significant despite n=72 -- unexpected under standard
            # correlation-test theory (SE(r) ~ 1/sqrt(n-3) ~ 0.12 there).
            # A synthetic check of point_biserial_bootstrap itself did NOT
            # reproduce this (wide CIs, p>>0.05 for small injected r), so
            # this is not a generic bug in the bootstrap function -- but it
            # was never checked against the REAL z distribution, because
            # the raw activation .pt tensors only exist on the cluster.
            # These summary stats (not the full z vector) are cheap enough
            # to persist here so this can be diagnosed without re-pulling
            # multi-GB tensor files: e.g. a few extreme z outliers that
            # happen to align with y could dominate resampling-with-
            # replacement and produce a spuriously one-sided bootstrap
            # distribution even though the bulk of the data has near-zero
            # correlation. Do NOT report any r as a reliable finding until
            # its z distribution here has been eyeballed for exactly this.
            z_vals = list(z_by_id.values())
            z_sorted = sorted(z_vals)
            n_z = len(z_sorted)
            z_mean = sum(z_vals) / n_z
            z_std = (sum((v - z_mean) ** 2 for v in z_vals) / n_z) ** 0.5
            diagnostics = {
                "z_mean": z_mean, "z_std": z_std,
                "z_min": z_sorted[0], "z_max": z_sorted[-1],
                "z_median": z_sorted[n_z // 2],
                "z_p05": z_sorted[int(0.05 * n_z)], "z_p95": z_sorted[min(int(0.95 * n_z), n_z - 1)],
                "note": "z summary stats only -- y class balance is now in z_vs_strict_success.{y_n_success,y_n_fail,reliable} (point_biserial_bootstrap computes this itself; see stats_shared.py's real-data incident note on this function).",
            }
            mechanism_results[m] = {"n_instructions": len(ids_common), "z_vs_strict_success": connection, "z_diagnostics": diagnostics}

        # Sep 2026 real-data incident: point_biserial_bootstrap's `reliable`
        # flag is False when class balance is too extreme (min(success,fail)
        # < 5) -- confirmed on real data that these degenerate to an
        # outlier-detection artifact, not a genuine correlation, regardless
        # of |r|. Excluded entirely from the Holm family, not just flagged --
        # including a known-degenerate p-value in a multiple-comparison
        # correction wastes correction budget without adding any real
        # information.
        named_pvalues = [(m, mechanism_results[m]["z_vs_strict_success"]["r_p_two_sided"])
                          for m in REAL_MECHANISMS
                          if mechanism_results[m]["z_vs_strict_success"]["r_p_two_sided"] is not None
                          and mechanism_results[m]["z_vs_strict_success"]["reliable"]]
        for m, p_holm in holm_correction(named_pvalues).items():
            mechanism_results[m]["z_vs_strict_success"]["r_p_holm_adjusted"] = p_holm
        for m in REAL_MECHANISMS:
            if not mechanism_results[m]["z_vs_strict_success"]["reliable"]:
                mechanism_results[m]["z_vs_strict_success"]["r_p_holm_adjusted"] = None

        by_kind_results[kind] = mechanism_results

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_alias}_history_augmented_activation_behavior_connection.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "result_status": "HISTORY_AUGMENTED_ACTIVATION_BEHAVIOR_CONNECTION",
            "model_alias": model_alias, "primary_layer": primary_layer,
            "note": "d_hat is Experiment 1's own frozen p_CO/p_MG (direction_ids-estimated only, Sec 6 firewall); z[i] and strict_success both from validation_ids. r_p_holm_adjusted is Holm-corrected across only the mechanisms with z_vs_strict_success.reliable==true WITHIN each scaffold kind, never pooled across kinds or models -- mechanisms with reliable==false (min(y_n_success,y_n_fail)<5, real-data incident Sep 2026: extreme class imbalance degenerates the bootstrap into an outlier-detection artifact regardless of |r|) are excluded from the Holm family entirely and have r_p_holm_adjusted=null; do not report their r or p as a finding.",
            "by_scaffold_kind": by_kind_results,
        }, f, indent=2, ensure_ascii=False)
    print(json.dumps({"result_status": "HISTORY_AUGMENTED_ACTIVATION_BEHAVIOR_CONNECTION_DONE", "model_alias": model_alias, "output_path": out_path}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--primary-layer", type=int, required=True)
    parser.add_argument("--extraction-dir", default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--experiment1-dir", default=DEFAULT_EXPERIMENT1_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-behavioral", action="store_true")
    parser.add_argument("--skip-representation", action="store_true")
    parser.add_argument("--skip-connection", action="store_true")
    args = parser.parse_args()

    if not args.skip_behavioral:
        run_behavioral(args.model_alias, args.extraction_dir, args.output_dir)
    if not args.skip_representation:
        run_representation(args.model_alias, args.primary_layer, args.extraction_dir, args.experiment1_dir, args.output_dir)
    if not args.skip_connection:
        run_activation_behavior_connection(args.model_alias, args.primary_layer, args.extraction_dir, args.experiment1_dir, args.output_dir)


if __name__ == "__main__":
    main()
