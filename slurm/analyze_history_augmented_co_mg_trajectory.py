"""Turn-wise (stage-by-stage) trajectory analysis for the history-
augmented canonical CO/MG design (FINAL_STUDY_PROTOCOL.md Sec 13
Round 20).

`analyze_history_augmented_co_mg.py`'s representation analysis
(`run_representation`) only reads FINAL_STAGE_KEY (stage_6, the payload
turn) and compares it against the single-turn baseline -- a
deliberately conservative choice (the thesis Method chapter's
"Turn-Wise Activation Trajectories" subsection defines t=1..6 but
reports only t=6, to avoid overclaiming turn-by-turn drift before it
had actually been checked against real data). `extract_history_augmented_co_mg.py`
saves EVERY stage's activation for the multi_* forms, not just the
final one (`activations_by_condition[condition][instruction_id]["stage_1"]`
through `"stage_6"`), so the 5 scaffold-turn vectors (stage_1_hook ..
stage_5_bridge) already exist in the direction_ids .pt files -- no
re-extraction needed -- but no analysis script had read them until
this one.

For each scaffold kind (neutral/progressive) and each of the 6 stages
(5 mechanism-free scaffold turns + the final mechanism-specific payload
turn), computes:
- per-mechanism d_m(stage) = mean(h[m,multi,stage]) - mean(h[m,single]),
  with bootstrap CI on ||d_m|| and cos(d_m, p_CO)/cos(d_m, p_MG)
  (Experiment 1's frozen, placebo-calibrated group directions) --
  identical statistic to `run_representation`, just swept over every
  stage instead of only the final one.
- CO/MG internal cohesion at that stage (within-CO vs within-MG vs
  between mean cosine of the 6 d_m(stage) point estimates), same
  statistic family as Experiment 1 Sec 4.3 and
  `analyze_history_augmented_co_mg.py`'s final-stage cohesion check.

Purpose: check whether the cross-model cohesion finding (within-CO >
within-MG, reported at the final stage only) is already present during
the mechanism-free scaffold turns -- before any mechanism-specific text
is introduced -- or only emerges once the payload text appears at
stage_6. The former would suggest the cohesion pattern is partly a
generic multi-turn/conversation-length artifact rather than a
mechanism-specific signal; the latter would support the
mechanism-specific interpretation the thesis currently uses. stage_6's
numbers here must exactly match `run_representation`'s output -- they
are the same statistic on the same data, just re-derived here inside
the stage sweep instead of standalone.

Reuses `analyze_history_augmented_co_mg.py`'s own loading/bootstrap
helpers directly (imported, not duplicated). `direction_ids` only,
representation only (Sec 6's firewall: `validation_ids`/`test_ids`
never touch direction estimation) -- this script does not touch
behavioral data or the activation-behavior connection.

CPU-only, no model, no GPU. Substantially more compute than
`run_representation`: 6 stages x 6 mechanisms x 2 scaffold kinds x
`--n-boot` bootstrap_vector_diff calls, vs. `run_representation`'s
1 stage -- budget SLURM walltime accordingly (roughly 6x); use
`--n-boot` to reduce if a first pass needs to run faster.

Never run against real data before this round -- smoke-test with a
low `--n-boot` before committing full cluster time to it.
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
    load_instruction_clusters, bootstrap_vector_diff, compute_2group_partition_stats,
)
from history_augmented_co_mg_loader import (  # noqa: E402
    CO_MECHANISMS, MG_MECHANISMS, MULTI_FORMS, STAGE_KEYS, FINAL_STAGE_KEY,
    multi_form_to_scaffold_kind,
)
from analyze_history_augmented_co_mg import (  # noqa: E402
    REAL_MECHANISMS, PRIMARY_TOKEN_POSITION, N_BOOTSTRAP, BOOTSTRAP_SEED,
    DEFAULT_EXTRACTION_DIR, DEFAULT_EXPERIMENT1_DIR, DEFAULT_OUTPUT_DIR,
    load_instruction_texts, load_activations, stage_vecs, load_experiment1_frozen_directions,
)

ALL_STAGE_KEYS = list(STAGE_KEYS) + [FINAL_STAGE_KEY]  # 6 stages: 5 scaffold turns + the payload turn


def run_trajectory(model_alias, primary_layer, extraction_dir, experiment1_dir, output_dir, n_boot):
    instruction_texts = load_instruction_texts()

    probe_acts = load_activations(extraction_dir, model_alias, "direction_ids", REAL_MECHANISMS[0], "single")
    all_direction_ids = sorted(probe_acts.keys())
    del probe_acts
    p_CO, p_MG = load_experiment1_frozen_directions(experiment1_dir, model_alias, primary_layer, PRIMARY_TOKEN_POSITION, all_direction_ids)

    by_kind_results = {}
    for form in MULTI_FORMS:
        kind = multi_form_to_scaffold_kind(form)
        by_stage_mechanism = {stage_key: {} for stage_key in ALL_STAGE_KEYS}
        d_m_point_by_stage = {stage_key: {} for stage_key in ALL_STAGE_KEYS}

        for m in REAL_MECHANISMS:
            # One load per mechanism (not per stage) -- every stage's vector
            # is already inside this same .pt file, same memory-safety
            # pattern as run_representation's OOM fix (load one mechanism
            # at a time, free before the next).
            acts_multi_m = load_activations(extraction_dir, model_alias, "direction_ids", m, form)
            acts_single_m = load_activations(extraction_dir, model_alias, "direction_ids", m, "single")
            ids_common = sorted(set(acts_multi_m) & set(acts_single_m))
            clusters = load_instruction_clusters(ids_common, instruction_texts)
            v_single = stage_vecs(acts_single_m, PRIMARY_TOKEN_POSITION, primary_layer, ids_common)

            for stage_key in ALL_STAGE_KEYS:
                v_multi_stage = stage_vecs(acts_multi_m, stage_key, primary_layer, ids_common)
                print(f"[{model_alias}] kind={kind} mechanism={m} stage={stage_key}: bootstrapping d_m^history...", file=sys.stderr)
                d_m = bootstrap_vector_diff(v_multi_stage, v_single, clusters, n_boot=n_boot, seed=BOOTSTRAP_SEED,
                                             extra_cos_targets={"p_CO": p_CO, "p_MG": p_MG})
                by_stage_mechanism[stage_key][m] = {"n_instructions": len(ids_common), "d_m_history": d_m}

                common_diffs = [v_multi_stage[i] - v_single[i] for i in ids_common if i in v_multi_stage and i in v_single]
                d_m_point_by_stage[stage_key][m] = torch.stack(common_diffs).mean(0) if common_diffs else None

            del acts_multi_m, acts_single_m

        by_stage_cohesion = {}
        for stage_key in ALL_STAGE_KEYS:
            d_m_point = d_m_point_by_stage[stage_key]
            if all(d_m_point[m] is not None for m in REAL_MECHANISMS):
                S_co, S_mg, S_between, delta_co, delta_mg, T = compute_2group_partition_stats(d_m_point, list(CO_MECHANISMS), list(MG_MECHANISMS))
                by_stage_cohesion[stage_key] = {
                    "within_CO_mean_cosine": S_co, "within_MG_mean_cosine": S_mg, "between_CO_MG_mean_cosine": S_between,
                    "CO_minus_between": delta_co, "MG_minus_between": delta_mg, "T_statistic": T,
                    "note": "point estimate only (no bootstrap CI) -- same statistic family as run_representation's final-stage cohesion and Experiment 1 Sec 4.3; the stage_6 entry here should numerically match run_representation's output for this model/kind.",
                }
            else:
                by_stage_cohesion[stage_key] = None

        by_kind_results[kind] = {
            "by_stage": {
                stage_key: {"by_mechanism": by_stage_mechanism[stage_key], "cohesion": by_stage_cohesion[stage_key]}
                for stage_key in ALL_STAGE_KEYS
            },
        }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_alias}_history_augmented_trajectory.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "result_status": "HISTORY_AUGMENTED_TRAJECTORY_ANALYSIS",
            "model_alias": model_alias, "primary_layer": primary_layer,
            "stage_order": ALL_STAGE_KEYS,
            "note": "Exploratory/diagnostic, not a confirmatory hypothesis test -- checks whether the within-CO > within-MG cohesion pattern (reported at the final stage only in the main analysis) is already present during the mechanism-free scaffold turns (stage_1_hook..stage_5_bridge) or only emerges once the mechanism-specific payload text appears (the last entry in stage_order, == FINAL_STAGE_KEY, same numbers as analyze_history_augmented_co_mg.py's run_representation).",
            "by_scaffold_kind": by_kind_results,
        }, f, indent=2, ensure_ascii=False)
    print(json.dumps({"result_status": "HISTORY_AUGMENTED_TRAJECTORY_DONE", "model_alias": model_alias, "output_path": out_path}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--primary-layer", type=int, required=True)
    parser.add_argument("--extraction-dir", default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--experiment1-dir", default=DEFAULT_EXPERIMENT1_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()

    run_trajectory(args.model_alias, args.primary_layer, args.extraction_dir, args.experiment1_dir, args.output_dir, args.n_boot)


if __name__ == "__main__":
    main()
