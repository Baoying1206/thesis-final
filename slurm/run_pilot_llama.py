"""Pilot driver for Experiment 2 (FINAL_STUDY_PROTOCOL.md Sec 5.5).

Frozen scope, exactly as specified: Meta-Llama-3.1-8B-Instruct only, the
30 pre-fixed `direction_ids` listed below, all 10 conditions
(`templates/final_10_condition_v1.json`), 300 generations + 300
WildGuard judgements. Every record this script produces is tagged
`"pilot": true` / `"result_status": "PILOT_NON_RESULT"` -- this is a
mechanical sanity check (rendering, generation, judging pipeline), NOT
a scientific result, and per Sec 5.5 its output must NEVER be used to
select, revise, or tune any condition's template wording.

Reads instruction text ONLY from data/source/sampled_prompts_en_only.json
(English-only, FINAL_STUDY_PROTOCOL.md Sec 2), ONLY for the 30 fixed ids
below. Never reads `data/source/sampled_prompts.json`'s `instructions`
multilingual field, never reads any `validation_ids` or `test_ids`.

Requires transformers + torch + GPU (the real Llama-3.1-8B-Instruct and
WildGuard weights); run this on the cluster, not a laptop. Run with
--dry-run first (no model loaded, CPU-only) to validate rendering for
all 300 rows before spending any GPU time.

The WildGuard prompt/parsing below is reused BYTE-IDENTICAL from
~/new_experiment/scripts/03_generate_and_label.py's WILDGUARD_PROMPT /
_parse_wildguard (read there, not migrated wholesale, per this study's
"reuse, don't fabricate" discipline for anything already named as a
frozen artifact -- FINAL_STUDY_PROTOCOL.md Sec 9.4's
judge_prompt_version: "wildguard_prompt_v1_from_03_generate_and_label"
identifier refers to exactly this prompt).
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
DEFAULT_MODEL_PATH = "/home/h24/baga0553/models/Llama-3.1-8B-Instruct"
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "pilot_output")
MODEL_ALIAS = "Meta-Llama-3.1-8B-Instruct"

# Frozen (FINAL_STUDY_PROTOCOL.md Sec 5.5): first 30 direction_ids in
# the order they appear in data/splits/splits.json. Never resampled,
# never chosen by any behavioral criterion.
PILOT_DIRECTION_IDS = [
    "p002", "p003", "p004", "p005", "p007", "p009", "p010", "p012", "p015", "p017",
    "p021", "p022", "p024", "p025", "p026", "p027", "p028", "p030", "p032", "p033",
    "p034", "p038", "p039", "p043", "p044", "p047", "p048", "p049", "p050", "p052",
]


def load_pilot_instructions():
    """Load ONLY the 30 fixed direction_ids' instruction_en text from the
    English-only derived pool file. Raises if any id is missing or if
    any id outside the frozen 30 is accidentally requested."""
    with open(SAMPLED_PROMPTS_EN_ONLY_PATH, "r", encoding="utf-8") as f:
        pool = json.load(f)
    by_id = {row["id"]: row["instruction_en"] for row in pool}

    missing = [pid for pid in PILOT_DIRECTION_IDS if pid not in by_id]
    if missing:
        raise ValueError(f"pilot direction_ids not found in {SAMPLED_PROMPTS_EN_ONLY_PATH}: {missing}")

    return [{"id": pid, "instruction_en": by_id[pid]} for pid in PILOT_DIRECTION_IDS]


def build_pilot_rows(generation_config):
    """Render all 30 x 10 = 300 (instruction, condition) pairs. Pure
    CPU, no model needed -- this is what --dry-run exercises."""
    instructions = load_pilot_instructions()
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
                    "model_alias": MODEL_ALIAS,
                    "instruction_id": instr["id"],
                    "condition": condition_name,
                    "messages_hash": messages_hash,
                    "generation_config_hash": generation_config_hash,
                },
                sort_keys=True,
            ))
            rows.append({
                "generation_key": generation_key,
                "model_alias": MODEL_ALIAS,
                "instruction_id": instr["id"],
                "condition": condition_name,
                "messages": messages,
                "messages_hash": messages_hash,
                "transform_provenance": transform_provenance,
                "generation_config_hash": generation_config_hash,
                "git_commit": commit,
                "ids_key": "direction_ids",
                "test_data_read": False,
                "pilot": True,
                "result_status": "PILOT_NON_RESULT",
            })

    assert len(rows) == 300, f"expected 300 rows (30 instructions x 10 conditions), got {len(rows)}"
    return rows, generation_config_hash


def run_dry_run(generation_config):
    rows, generation_config_hash = build_pilot_rows(generation_config)
    conditions_seen = sorted(set(r["condition"] for r in rows))
    ids_seen = sorted(set(r["instruction_id"] for r in rows))
    print(json.dumps({
        "result_status": "PILOT_DRY_RUN_PASS",
        "pilot": True,
        "n_rows": len(rows),
        "n_conditions": len(conditions_seen),
        "conditions": conditions_seen,
        "n_instruction_ids": len(ids_seen),
        "instruction_ids_match_frozen_list": ids_seen == sorted(PILOT_DIRECTION_IDS),
        "generation_config_hash": generation_config_hash,
        "git_commit": rows[0]["git_commit"],
        "sample_generation_key": rows[0]["generation_key"],
    }, indent=2, ensure_ascii=False))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Render all 300 rows and validate, no model loaded, no GPU.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--wg-batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N of the 300 rows -- for a quick end-to-end "
                             "smoke test before committing full GPU time. Never use this to cherry-pick.")
    args = parser.parse_args()

    generation_config = {"max_new_tokens": args.max_new_tokens, "do_sample": False}

    if args.dry_run:
        run_dry_run(generation_config)
        return

    rows, _ = build_pilot_rows(generation_config)
    if args.limit is not None:
        rows = rows[:args.limit]
        print(f"--limit set: processing only {len(rows)} of 300 rows (smoke test only, "
              f"not a valid pilot run)", file=sys.stderr)

    rows = generate_responses(rows, args.model_path, args.batch_size, generation_config)
    judge_records = judge_responses(rows, args.wg_batch_size, {"pilot": True, "result_status": "PILOT_NON_RESULT"})

    os.makedirs(args.output_dir, exist_ok=True)
    generation_path = os.path.join(args.output_dir, "pilot_generation_records.jsonl")
    judge_path = os.path.join(args.output_dir, "pilot_judge_records.jsonl")

    with open(generation_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(judge_path, "w", encoding="utf-8") as f:
        for r in judge_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps({
        "result_status": "PILOT_NON_RESULT",
        "pilot": True,
        "n_generation_records": len(rows),
        "n_judge_records": len(judge_records),
        "generation_path": generation_path,
        "judge_path": judge_path,
        "note": "PILOT_NON_RESULT -- mechanical sanity check only. Never use these numbers to "
                "select, revise, or tune any condition's template wording (Sec 5.5).",
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
