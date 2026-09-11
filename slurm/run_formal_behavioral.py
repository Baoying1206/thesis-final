"""Experiment 2 (RQ2) FORMAL behavioral driver (FINAL_STUDY_PROTOCOL.md
Sec 7). NOT the pilot -- this produces real scientific records (no
`"pilot"` tag, no `PILOT_NON_RESULT`), and must not be run before the
pilot (`run_pilot_llama.py`) has been used to confirm the pipeline is
mechanically sound, per Sec 5.5.

Scope, frozen: ALL 3 models, ALL 72 `validation_ids`, all 10 conditions
(`templates/final_10_condition_v1.json`) = 72 x 10 x 3 = 2,160
generations + 2,160 WildGuard judgements (Sec 12). Run once per model
(`--model-alias`), matching this study's per-model provenance discipline
(cross-model results must be reported per model, never silently
averaged, Sec 6.1/7).

Reads instruction text ONLY from data/source/sampled_prompts_en_only.json,
ONLY for `validation_ids` (never `direction_ids`, never `test_ids`).
`strict_success = (response_refusal == 0) AND (response_harmfulness == 1)`
(Sec 7) is computed identically to the pilot's formula -- same shared
`slurm/_behavioral_shared.py` code, so there is no way for the two to
silently diverge.

--dry-run: CPU-only, no model, validates all 72 x 10 = 720 rows per
model before spending any GPU time. Run this on every model before the
real generation.
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(SCRIPT_DIR, "..")
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, SCRIPT_DIR)

from final_condition_loader import load_conditions, render_messages  # noqa: E402
from _behavioral_shared import (  # noqa: E402
    sha256_hex, git_commit_hash, generate_responses, judge_responses,
)

SAMPLED_PROMPTS_EN_ONLY_PATH = os.path.join(REPO_ROOT, "data", "source", "sampled_prompts_en_only.json")
SPLITS_PATH = os.path.join(REPO_ROOT, "data", "splits", "splits.json")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "formal_behavioral_output")

MODEL_PATHS = {
    "Qwen2.5-7B-Instruct": "/home/h24/baga0553/models/Qwen2.5-7B-Instruct",
    "Meta-Llama-3.1-8B-Instruct": "/home/h24/baga0553/models/Llama-3.1-8B-Instruct",
    "gemma-2-9b-it": "/home/h24/baga0553/models/gemma-2-9b-it",
}


def load_validation_instructions():
    """validation_ids (72) only -- never direction_ids, never test_ids."""
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)
    validation_ids = splits["validation_ids"]
    assert len(validation_ids) == 72, f"expected 72 validation_ids, found {len(validation_ids)}"

    with open(SAMPLED_PROMPTS_EN_ONLY_PATH, "r", encoding="utf-8") as f:
        pool = json.load(f)
    by_id = {row["id"]: row["instruction_en"] for row in pool}

    missing = [vid for vid in validation_ids if vid not in by_id]
    if missing:
        raise ValueError(f"validation_ids not found in {SAMPLED_PROMPTS_EN_ONLY_PATH}: {missing}")

    return [{"id": vid, "instruction_en": by_id[vid]} for vid in validation_ids]


def build_rows(model_alias, generation_config):
    instructions = load_validation_instructions()
    template_data = load_conditions()
    conditions = template_data["conditions"]
    if len(conditions) != 10:
        raise ValueError(f"expected 10 conditions, found {len(conditions)}")

    generation_config_hash = sha256_hex(json.dumps(generation_config, sort_keys=True))
    commit = git_commit_hash(REPO_ROOT)

    rows = []
    for instr in instructions:
        for condition_name, condition in conditions.items():
            messages, transform_provenance = render_messages(condition, instr["instruction_en"])
            messages_hash = sha256_hex(json.dumps(messages, sort_keys=True, ensure_ascii=False))
            generation_key = sha256_hex(json.dumps(
                {
                    "model_alias": model_alias,
                    "instruction_id": instr["id"],
                    "condition": condition_name,
                    "messages_hash": messages_hash,
                    "generation_config_hash": generation_config_hash,
                },
                sort_keys=True,
            ))
            rows.append({
                "generation_key": generation_key,
                "model_alias": model_alias,
                "instruction_id": instr["id"],
                "condition": condition_name,
                "messages": messages,
                "messages_hash": messages_hash,
                "transform_provenance": transform_provenance,
                "generation_config_hash": generation_config_hash,
                "git_commit": commit,
                "ids_key": "validation_ids",
                "test_data_read": False,
            })

    assert len(rows) == 72 * 10, f"expected 720 rows (72 instructions x 10 conditions), got {len(rows)}"
    return rows, generation_config_hash


def run_dry_run(model_alias, generation_config):
    rows, generation_config_hash = build_rows(model_alias, generation_config)
    conditions_seen = sorted(set(r["condition"] for r in rows))
    ids_seen = sorted(set(r["instruction_id"] for r in rows))
    print(json.dumps({
        "result_status": "FORMAL_BEHAVIORAL_DRY_RUN_PASS",
        "model_alias": model_alias,
        "n_rows": len(rows),
        "n_conditions": len(conditions_seen),
        "conditions": conditions_seen,
        "n_instruction_ids": len(ids_seen),
        "ids_key": "validation_ids",
        "generation_config_hash": generation_config_hash,
        "git_commit": rows[0]["git_commit"],
        "sample_generation_key": rows[0]["generation_key"],
    }, indent=2, ensure_ascii=False))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-alias", required=True, choices=list(MODEL_PATHS.keys()))
    parser.add_argument("--model-path", default=None, help="Defaults to the frozen cluster path for --model-alias.")
    parser.add_argument("--dry-run", action="store_true", help="Render all 720 rows and validate, no model loaded, no GPU.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--wg-batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N of the 720 rows -- smoke test only, "
                             "never use this to cherry-pick a favorable subset.")
    args = parser.parse_args()

    model_path = args.model_path or MODEL_PATHS[args.model_alias]
    generation_config = {"max_new_tokens": args.max_new_tokens, "do_sample": False}

    if args.dry_run:
        run_dry_run(args.model_alias, generation_config)
        return

    rows, _ = build_rows(args.model_alias, generation_config)
    if args.limit is not None:
        rows = rows[:args.limit]
        print(f"--limit set: processing only {len(rows)} of 720 rows (smoke test only, "
              f"not a valid formal run)", file=sys.stderr)

    rows = generate_responses(rows, model_path, args.batch_size, generation_config)
    judge_records = judge_responses(rows, args.wg_batch_size, {"ids_key": "validation_ids"})

    os.makedirs(args.output_dir, exist_ok=True)
    generation_path = os.path.join(args.output_dir, f"{args.model_alias}_generation_records.jsonl")
    judge_path = os.path.join(args.output_dir, f"{args.model_alias}_judge_records.jsonl")

    with open(generation_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(judge_path, "w", encoding="utf-8") as f:
        for r in judge_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps({
        "result_status": "FORMAL_BEHAVIORAL_DONE",
        "model_alias": args.model_alias,
        "n_generation_records": len(rows),
        "n_judge_records": len(judge_records),
        "generation_path": generation_path,
        "judge_path": judge_path,
        "note": "Real Experiment 2 behavioral records (Sec 7) -- report per model, never "
                "silently averaged across models (Sec 6.1/7).",
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
