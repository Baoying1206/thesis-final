"""Experiment 2 (RQ2) activation extraction (FINAL_STUDY_PROTOCOL.md
Sec 6/6.2). Produces RAW per-(model, condition, instruction_id)
activation records only, for the 10-condition template
(`templates/final_10_condition_v1.json`) -- this script does NOT
compute `d_m`, the group-level two-stage aggregation (Sec 5.4), or any
Sec 6.1 statistic; that is a separate, later analysis step over this
script's output.

Parameterized over `--ids-key`, which must be exactly one of:
- `direction_ids` (300): the ONLY id set any direction is ever
  estimated from (Sec 6's firewall rule -- this script never filters by
  behavioral outcome, and doesn't even have generation/judge code in it
  at all to filter against).
- `validation_ids` (72): consumed ONLY by the Sec 6.2
  projection-vs-strict_success correlation check downstream. Every
  record produced with `--ids-key validation_ids` carries
  `"used_for_direction_estimation": false` (Sec 9.2) -- a structural,
  hard-coded assertion, not just a documented rule.

`test_ids` is not a valid `--ids-key` value and is rejected by argparse.

One forward pass per `(condition, instruction_id)` (`output_hidden_states=True`,
no `.generate()`) -- both `t_generation_boundary` and `t_final_user_end`
are read from the SAME forward pass, exactly as in
`slurm/extract_experiment1_activations.py` and
`audits/audit_real_tokenizer_boundary.py`. `direction_ids`: 300 x 10 x 3
= 9,000 forward passes. `validation_ids`: 72 x 10 x 3 = 2,160 forward
passes (Sec 12).

--dry-run: tokenizer-only (no model weights), validates rendering AND
real token-position location for all rows for the given `--ids-key`.
Run this BEFORE spending any GPU time.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(SCRIPT_DIR, "..")
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from final_condition_loader import load_conditions, render_messages  # noqa: E402

SAMPLED_PROMPTS_EN_ONLY_PATH = os.path.join(REPO_ROOT, "data", "source", "sampled_prompts_en_only.json")
SPLITS_PATH = os.path.join(REPO_ROOT, "data", "splits", "splits.json")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "experiment2_output")

PRIMARY_TOKEN_POSITION = "t_generation_boundary"
SECONDARY_TOKEN_POSITION = "t_final_user_end"

ALLOWED_IDS_KEYS = ("direction_ids", "validation_ids")
EXPECTED_COUNT = {"direction_ids": 300, "validation_ids": 72}

# Frozen (FINAL_STUDY_PROTOCOL.md Sec 4.2, reused for Sec 6 -- same 3
# models, same primary-layer formula).
MODEL_TOKENIZER_SOURCES = [
    ("Qwen2.5-7B-Instruct", "QWEN", "/home/h24/baga0553/models/Qwen2.5-7B-Instruct", 16),
    ("Meta-Llama-3.1-8B-Instruct", "LLAMA", "/home/h24/baga0553/models/Llama-3.1-8B-Instruct", 19),
    ("gemma-2-9b-it", "GEMMA", "/home/h24/baga0553/models/gemma-2-9b-it", 25),
]


def sha256_hex(s):
    if isinstance(s, str):
        s = s.encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def git_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def load_instructions(ids_key):
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)
    ids = splits[ids_key]
    expected = EXPECTED_COUNT[ids_key]
    assert len(ids) == expected, f"expected {expected} {ids_key}, found {len(ids)}"

    with open(SAMPLED_PROMPTS_EN_ONLY_PATH, "r", encoding="utf-8") as f:
        pool = json.load(f)
    by_id = {row["id"]: row["instruction_en"] for row in pool}

    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"{ids_key} not found in {SAMPLED_PROMPTS_EN_ONLY_PATH}: {missing[:5]}")

    return [{"id": i, "instruction_en": by_id[i]} for i in ids]


def locate_positions_for_messages(tokenizer, messages, final_user_text):
    """Same proven method as
    audits/audit_real_tokenizer_boundary.py / extract_experiment1_activations.py,
    adapted for a real (already-rendered) multi-turn messages list."""
    full_prompt_result = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    full_prompt_ids = full_prompt_result["input_ids"] if hasattr(full_prompt_result, "keys") else list(full_prompt_result)
    full_text_no_priming = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    ids_no_priming_result = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    ids_no_priming = ids_no_priming_result["input_ids"] if hasattr(ids_no_priming_result, "keys") else list(ids_no_priming_result)

    occurrences = full_text_no_priming.count(final_user_text)
    if occurrences != 1:
        raise ValueError(f"final_user content substring found {occurrences}x in rendered text (need exactly 1)")
    char_start = full_text_no_priming.rfind(final_user_text)
    char_end = char_start + len(final_user_text)

    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("tokenizer is not fast -- no offset mapping available")

    encoding = tokenizer(full_text_no_priming, add_special_tokens=False, return_offsets_mapping=True)
    if encoding["input_ids"] != list(ids_no_priming):
        raise ValueError("re-tokenization mismatch -- cannot trust offsets")

    if list(full_prompt_ids[:len(ids_no_priming)]) != list(ids_no_priming):
        raise ValueError("add_generation_prompt prefix mismatch -- index alignment invalid")

    candidates = [i for i, (s, e) in enumerate(encoding["offset_mapping"]) if char_start <= s < char_end]
    if not candidates:
        raise ValueError("no token overlaps the final_user content span")
    t_final_user_end = max(candidates)

    special_ids = set(tokenizer.all_special_ids) if hasattr(tokenizer, "all_special_ids") else set()
    if ids_no_priming[t_final_user_end] in special_ids:
        raise ValueError("resolved t_final_user_end token is a special/turn-marker token")

    t_generation_boundary = len(full_prompt_ids) - 1
    return full_prompt_ids, t_generation_boundary, t_final_user_end


def build_rows(instructions, conditions):
    rows = []
    for instr in instructions:
        for name, condition in conditions.items():
            messages, transform_provenance = render_messages(condition, instr["instruction_en"])
            rows.append({
                "instruction_id": instr["id"],
                "condition": name,
                "messages": messages,
                "transform_provenance": transform_provenance,
                "template_content_sha256": sha256_hex(json.dumps(
                    {k: condition[k] for k in ("setup_user", "assistant_acknowledgement", "final_user")},
                    sort_keys=True, ensure_ascii=False,
                )),
            })
    return rows


def run_dry_run(model_alias, tokenizer_path, ids_key):
    from transformers import AutoTokenizer
    print(f"[{model_alias}] loading tokenizer only: {tokenizer_path}", file=sys.stderr)
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(json.dumps({
            "result_status": "TOKENIZER_LOAD_FAILED", "model_alias": model_alias,
            "tokenizer_path": tokenizer_path, "load_error": str(e),
        }, indent=2, ensure_ascii=False))
        return False

    instructions = load_instructions(ids_key)
    template_data = load_conditions()
    conditions = template_data["conditions"]
    rows = build_rows(instructions, conditions)
    expected_n = EXPECTED_COUNT[ids_key] * 10
    assert len(rows) == expected_n, f"expected {expected_n} rows, got {len(rows)}"

    n_ok, failures = 0, []
    for row in rows:
        try:
            final_user_text = row["messages"][-1]["content"]
            _, t_gen, t_final = locate_positions_for_messages(tokenizer, row["messages"], final_user_text)
            if t_final >= t_gen:
                raise ValueError(f"t_final_user_end ({t_final}) >= t_generation_boundary ({t_gen})")
            n_ok += 1
        except Exception as e:
            failures.append({"instruction_id": row["instruction_id"], "condition": row["condition"], "reason": str(e)})

    print(json.dumps({
        "result_status": "EXPERIMENT2_DRY_RUN_PASS" if not failures else "EXPERIMENT2_DRY_RUN_FAIL",
        "model_alias": model_alias, "ids_key": ids_key,
        "n_rows": len(rows), "n_ok": n_ok, "n_failures": len(failures),
        "failures": failures[:10],
    }, indent=2, ensure_ascii=False))
    return len(failures) == 0


def run_extraction(model_alias, model_path, primary_layer_expected, ids_key, output_dir, limit=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[{model_alias}] loading tokenizer + model: {model_path}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    n_layers = model.config.num_hidden_layers
    primary_layer_computed = int(n_layers * 0.6)
    if primary_layer_computed != primary_layer_expected:
        raise ValueError(
            f"[{model_alias}] frozen primary layer {primary_layer_expected} does not match "
            f"floor(0.6*{n_layers})={primary_layer_computed} -- model config changed since freezing, stop"
        )

    instructions = load_instructions(ids_key)
    template_data = load_conditions()
    conditions = template_data["conditions"]
    rows = build_rows(instructions, conditions)
    if limit is not None:
        rows = rows[:limit]

    commit = git_commit_hash()
    used_for_direction_estimation = False if ids_key == "validation_ids" else None
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, f"{model_alias}_{ids_key}_experiment2_manifest.jsonl")
    condition_names = list(conditions.keys())
    activations_by_condition = {name: {} for name in condition_names}

    n_ok, n_fail = 0, 0
    with open(manifest_path, "w", encoding="utf-8") as manifest_f:
        for idx, row in enumerate(rows):
            final_user_text = row["messages"][-1]["content"]
            try:
                full_prompt_ids, t_gen, t_final = locate_positions_for_messages(tokenizer, row["messages"], final_user_text)
            except Exception as e:
                manifest_f.write(json.dumps({
                    "instruction_id": row["instruction_id"], "condition": row["condition"],
                    "model_alias": model_alias, "result_status": "EXTRACTION_FAIL",
                    "failure_reason": str(e), "ids_key": ids_key, "test_data_read": False,
                }, ensure_ascii=False) + "\n")
                n_fail += 1
                continue

            input_ids = torch.tensor([full_prompt_ids]).to(model.device)
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True)
            hidden_states = torch.stack(out.hidden_states, dim=0)  # [n_layers+1, 1, seq_len, hidden]
            gen_boundary_vec = hidden_states[:, 0, t_gen, :].to(torch.float16).cpu()
            final_user_vec = hidden_states[:, 0, t_final, :].to(torch.float16).cpu()

            activations_by_condition[row["condition"]][row["instruction_id"]] = {
                PRIMARY_TOKEN_POSITION: gen_boundary_vec,
                SECONDARY_TOKEN_POSITION: final_user_vec,
            }
            activation_path = os.path.join(output_dir, f"{model_alias}_{ids_key}_{row['condition']}_activations.pt")

            record = {
                "instruction_id": row["instruction_id"], "condition": row["condition"],
                "model_alias": model_alias, "layer_count": hidden_states.shape[0],
                "primary_layer": primary_layer_expected,
                PRIMARY_TOKEN_POSITION: {"index": t_gen, "token_id": int(full_prompt_ids[t_gen])},
                SECONDARY_TOKEN_POSITION: {"index": t_final, "token_id": int(full_prompt_ids[t_final])},
                "activation_path": activation_path, "activation_index": row["instruction_id"],
                "template_content_sha256": row["template_content_sha256"],
                "transform_provenance": row["transform_provenance"],
                "git_commit": commit, "ids_key": ids_key, "test_data_read": False,
                "result_status": "EXTRACTION_OK",
            }
            if ids_key == "validation_ids":
                record["used_for_direction_estimation"] = False
            manifest_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_ok += 1

            if (idx + 1) % 100 == 0:
                print(f"  [{model_alias}][{ids_key}] {idx + 1}/{len(rows)} (ok={n_ok} fail={n_fail})", file=sys.stderr)

    for name in condition_names:
        activation_path = os.path.join(output_dir, f"{model_alias}_{ids_key}_{name}_activations.pt")
        torch.save(activations_by_condition[name], activation_path)
        print(f"  saved {activation_path} ({len(activations_by_condition[name])} instructions)", file=sys.stderr)

    del model
    torch.cuda.empty_cache()

    print(json.dumps({
        "result_status": "EXPERIMENT2_EXTRACTION_DONE",
        "model_alias": model_alias, "ids_key": ids_key,
        "n_rows": len(rows), "n_ok": n_ok, "n_fail": n_fail,
        "manifest_path": manifest_path, "output_dir": output_dir,
    }, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-alias", required=True, choices=[m[0] for m in MODEL_TOKENIZER_SOURCES])
    parser.add_argument("--ids-key", required=True, choices=list(ALLOWED_IDS_KEYS))
    parser.add_argument("--model-path", default=None, help="Defaults to the frozen cluster path for --model-alias.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Tokenizer-only, no model weights, no GPU.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows -- smoke test only.")
    args = parser.parse_args()

    entry = next(m for m in MODEL_TOKENIZER_SOURCES if m[0] == args.model_alias)
    _, env_suffix, default_path, primary_layer = entry
    model_path = args.model_path or os.environ.get(f"THESIS_FINAL_TOKENIZER_PATH_{env_suffix}", default_path)

    if args.dry_run:
        ok = run_dry_run(args.model_alias, model_path, args.ids_key)
        sys.exit(0 if ok else 1)

    run_extraction(args.model_alias, model_path, primary_layer, args.ids_key, args.output_dir, args.limit)


if __name__ == "__main__":
    main()
