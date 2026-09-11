"""Experiment 1 (RQ1) activation extraction (FINAL_STUDY_PROTOCOL.md
Sec 4). Produces RAW per-(model, condition, instruction_id) activation
records only -- this script does NOT compute d_m, placebo calibration,
split-half reliability, or partition ranking; that is a separate,
later analysis step over this script's output (Sec 4.2/4.3), not part
of extraction.

Scope, frozen: 8 single-turn conditions (`plain`, `placebo`, and the 6
canonical mechanisms from templates/imported/templates_wei_canonical.json),
all 300 `direction_ids`, 3 models, full-layer hidden states at both
frozen token positions (`t_generation_boundary` primary,
`t_final_user_end` secondary). One forward pass per (condition,
instruction_id) with output_hidden_states=True -- both positions are
read from the SAME forward pass (t_final_user_end's position is a
prefix-shared index into the same hidden_states tensors as
t_generation_boundary, verified per-row, exactly as established in
audits/audit_real_tokenizer_boundary.py). No `.generate()` calls; this
is 300 x 8 x 3 = 7,200 forward passes, matching FINAL_STUDY_PROTOCOL.md
Sec 12's compute estimate.

Never filters by behavioral outcome (there is no such thing to filter
by here -- no generation happens in this script at all). Never reads
`validation_ids`/`test_ids`. Reads instruction text ONLY from
data/source/sampled_prompts_en_only.json.

--dry-run: tokenizer-only (no model weights), validates rendering AND
real token-position location for all 300 x 8 = 2,400 rows per model.
Run this BEFORE spending any GPU time.

Single-turn message structure (this study's own choice, not migrated
from the old exploratory study's hand-rolled prompt concatenation):
messages = [{"role": "user", "content": T_m(instruction)}], rendered
through the model's own tokenizer.apply_chat_template -- consistent
with this study's standing rule (FINAL_STUDY_PROTOCOL.md Sec 5.2) that
every rendered prompt goes through the real chat template, never a
hand-concatenated string. This means each model's own default system
prompt (if any) is included, exactly as a real single-turn user message
would be -- a known, flagged divergence from the old exploratory
single-turn study's hand-rolled templates (which had no system prompt),
see FINAL_STUDY_PROTOCOL.md Sec 13.
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

from canonical_transforms import encode_base64_utf8, split_payload_midpoint  # noqa: E402

SAMPLED_PROMPTS_EN_ONLY_PATH = os.path.join(REPO_ROOT, "data", "source", "sampled_prompts_en_only.json")
SPLITS_PATH = os.path.join(REPO_ROOT, "data", "splits", "splits.json")
WEI_CANONICAL_PATH = os.path.join(REPO_ROOT, "templates", "imported", "templates_wei_canonical.json")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "experiment1_output")

PRIMARY_TOKEN_POSITION = "t_generation_boundary"
SECONDARY_TOKEN_POSITION = "t_final_user_end"

# Frozen (FINAL_STUDY_PROTOCOL.md Sec 4.2): floor(0.6 * n_layers), asserted
# against each model's real config at load time, not just trusted here.
MODEL_TOKENIZER_SOURCES = [
    ("Qwen2.5-7B-Instruct", "QWEN", "/home/h24/baga0553/models/Qwen2.5-7B-Instruct", 16),
    ("Meta-Llama-3.1-8B-Instruct", "LLAMA", "/home/h24/baga0553/models/Llama-3.1-8B-Instruct", 19),
    ("gemma-2-9b-it", "GEMMA", "/home/h24/baga0553/models/gemma-2-9b-it", 25),
]

# Condition order frozen (FINAL_STUDY_PROTOCOL.md Sec 4.1).
CONDITION_ORDER = [
    "plain", "placebo", "prefix_injection", "refusal_suppression",
    "persona_roleplay", "encoding_obfuscation", "payload_splitting", "distractors_negated",
]
PLACEHOLDER_SCHEME_BY_CONDITION = {
    "plain": "instruction", "placebo": "instruction",
    "prefix_injection": "instruction", "refusal_suppression": "instruction",
    "persona_roleplay": "instruction", "distractors_negated": "instruction",
    "encoding_obfuscation": "instruction_b64", "payload_splitting": "payload_a_b",
}


def sha256_hex(s):
    if isinstance(s, str):
        s = s.encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def git_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def load_conditions():
    with open(WEI_CANONICAL_PATH, "r", encoding="utf-8") as f:
        wei = json.load(f)
    conditions = {"plain": "{instruction}"}
    for name in CONDITION_ORDER[1:]:
        conditions[name] = wei["templates"][name]["text"]
    assert list(conditions.keys()) == CONDITION_ORDER
    return conditions


def load_direction_instructions():
    """direction_ids (300) only -- never validation_ids/test_ids."""
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)
    direction_ids = splits["direction_ids"]
    assert len(direction_ids) == 300, f"expected 300 direction_ids, found {len(direction_ids)}"

    with open(SAMPLED_PROMPTS_EN_ONLY_PATH, "r", encoding="utf-8") as f:
        pool = json.load(f)
    by_id = {row["id"]: row["instruction_en"] for row in pool}

    missing = [did for did in direction_ids if did not in by_id]
    if missing:
        raise ValueError(f"direction_ids not found in {SAMPLED_PROMPTS_EN_ONLY_PATH}: {missing[:5]}")

    return [{"id": did, "instruction_en": by_id[did]} for did in direction_ids]


def render_final_user(condition_name, template_text, instruction_text):
    scheme = PLACEHOLDER_SCHEME_BY_CONDITION[condition_name]
    if scheme == "instruction":
        assert template_text.count("{instruction}") == 1
        return template_text.replace("{instruction}", instruction_text), None
    if scheme == "instruction_b64":
        b64_string, input_sha256, output_sha256 = encode_base64_utf8(instruction_text)
        return (
            template_text.replace("{instruction_b64}", b64_string),
            {"transform": "encode_base64_utf8", "input_sha256": input_sha256, "output_sha256": output_sha256},
        )
    if scheme == "payload_a_b":
        fragment_a, fragment_b, normalized_sha256, fragment_a_sha256, fragment_b_sha256 = split_payload_midpoint(instruction_text)
        rendered = template_text.replace("{payload_a}", fragment_a).replace("{payload_b}", fragment_b)
        return rendered, {
            "transform": "split_payload_midpoint", "normalized_sha256": normalized_sha256,
            "fragment_a_sha256": fragment_a_sha256, "fragment_b_sha256": fragment_b_sha256,
        }
    raise ValueError(f"unknown placeholder_scheme {scheme!r}")


def locate_positions(tokenizer, final_user_text):
    """Adapted from audits/audit_real_tokenizer_boundary.py's proven
    method: exact uniqueness-checked substring search + fast-tokenizer
    offset mapping + re-tokenization consistency check + special-token
    guard. Returns (full_prompt_ids, t_generation_boundary_index,
    t_final_user_end_index) or raises ValueError with a specific reason
    if the position cannot be uniquely located -- never a guess."""
    messages = [{"role": "user", "content": final_user_text}]

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
        for name in CONDITION_ORDER:
            final_user_text, transform_provenance = render_final_user(name, conditions[name], instr["instruction_en"])
            rows.append({
                "instruction_id": instr["id"],
                "condition": name,
                "final_user_text": final_user_text,
                "transform_provenance": transform_provenance,
                "template_content_sha256": sha256_hex(conditions[name]),
            })
    assert len(rows) == 300 * 8
    return rows


def run_dry_run(model_alias, tokenizer_path):
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

    instructions = load_direction_instructions()
    conditions = load_conditions()
    rows = build_rows(instructions, conditions)

    n_ok, failures = 0, []
    for row in rows:
        try:
            _, t_gen, t_final = locate_positions(tokenizer, row["final_user_text"])
            if t_final >= t_gen:
                raise ValueError(f"t_final_user_end ({t_final}) >= t_generation_boundary ({t_gen})")
            n_ok += 1
        except Exception as e:
            failures.append({"instruction_id": row["instruction_id"], "condition": row["condition"], "reason": str(e)})

    print(json.dumps({
        "result_status": "EXPERIMENT1_DRY_RUN_PASS" if not failures else "EXPERIMENT1_DRY_RUN_FAIL",
        "model_alias": model_alias,
        "n_rows": len(rows),
        "n_ok": n_ok,
        "n_failures": len(failures),
        "failures": failures[:10],
    }, indent=2, ensure_ascii=False))
    return len(failures) == 0


def run_extraction(model_alias, model_path, primary_layer_expected, output_dir, limit=None):
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

    instructions = load_direction_instructions()
    conditions = load_conditions()
    rows = build_rows(instructions, conditions)
    if limit is not None:
        rows = rows[:limit]

    commit = git_commit_hash()
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, f"{model_alias}_experiment1_manifest.jsonl")
    activations_by_condition = {name: {} for name in CONDITION_ORDER}

    n_ok, n_fail = 0, 0
    with open(manifest_path, "w", encoding="utf-8") as manifest_f:
        for idx, row in enumerate(rows):
            try:
                full_prompt_ids, t_gen, t_final = locate_positions(tokenizer, row["final_user_text"])
            except Exception as e:
                manifest_f.write(json.dumps({
                    **{k: v for k, v in row.items() if k != "final_user_text"},
                    "model_alias": model_alias, "result_status": "EXTRACTION_FAIL",
                    "failure_reason": str(e), "ids_key": "direction_ids", "test_data_read": False,
                }, ensure_ascii=False) + "\n")
                n_fail += 1
                continue

            input_ids = torch.tensor([full_prompt_ids]).to(model.device)
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True)
            # hidden_states: tuple of (n_layers+1) tensors, each [1, seq_len, hidden]
            hidden_states = torch.stack(out.hidden_states, dim=0)  # [n_layers+1, 1, seq_len, hidden]
            gen_boundary_vec = hidden_states[:, 0, t_gen, :].to(torch.float16).cpu()   # [n_layers+1, hidden]
            final_user_vec = hidden_states[:, 0, t_final, :].to(torch.float16).cpu()   # [n_layers+1, hidden]

            activations_by_condition[row["condition"]][row["instruction_id"]] = {
                PRIMARY_TOKEN_POSITION: gen_boundary_vec,
                SECONDARY_TOKEN_POSITION: final_user_vec,
            }
            activation_path = os.path.join(output_dir, f"{model_alias}_{row['condition']}_activations.pt")

            manifest_f.write(json.dumps({
                "instruction_id": row["instruction_id"], "condition": row["condition"],
                "model_alias": model_alias, "layer_count": hidden_states.shape[0],
                "primary_layer": primary_layer_expected,
                PRIMARY_TOKEN_POSITION: {"index": t_gen, "token_id": int(full_prompt_ids[t_gen])},
                SECONDARY_TOKEN_POSITION: {"index": t_final, "token_id": int(full_prompt_ids[t_final])},
                "activation_path": activation_path,
                "activation_index": row["instruction_id"],
                "template_content_sha256": row["template_content_sha256"],
                "transform_provenance": row["transform_provenance"],
                "git_commit": commit, "ids_key": "direction_ids", "test_data_read": False,
                "result_status": "EXTRACTION_OK",
            }, ensure_ascii=False) + "\n")
            n_ok += 1

            if (idx + 1) % 50 == 0:
                print(f"  [{model_alias}] {idx + 1}/{len(rows)} (ok={n_ok} fail={n_fail})", file=sys.stderr)

    for name in CONDITION_ORDER:
        activation_path = os.path.join(output_dir, f"{model_alias}_{name}_activations.pt")
        torch.save(activations_by_condition[name], activation_path)
        print(f"  saved {activation_path} ({len(activations_by_condition[name])} instructions)", file=sys.stderr)

    del model
    torch.cuda.empty_cache()

    print(json.dumps({
        "result_status": "EXPERIMENT1_EXTRACTION_DONE",
        "model_alias": model_alias, "n_rows": len(rows), "n_ok": n_ok, "n_fail": n_fail,
        "manifest_path": manifest_path, "output_dir": output_dir,
    }, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-alias", required=True, choices=[m[0] for m in MODEL_TOKENIZER_SOURCES])
    parser.add_argument("--model-path", default=None, help="Defaults to the frozen cluster path for --model-alias.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Tokenizer-only, no model weights, no GPU.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N of 2,400 rows -- smoke test only.")
    args = parser.parse_args()

    entry = next(m for m in MODEL_TOKENIZER_SOURCES if m[0] == args.model_alias)
    _, env_suffix, default_path, primary_layer = entry
    model_path = args.model_path or os.environ.get(f"THESIS_FINAL_TOKENIZER_PATH_{env_suffix}", default_path)

    if args.dry_run:
        ok = run_dry_run(args.model_alias, model_path)
        sys.exit(0 if ok else 1)

    run_extraction(args.model_alias, model_path, primary_layer, args.output_dir, args.limit)


if __name__ == "__main__":
    main()
