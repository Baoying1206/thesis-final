"""RQ2 Round 20 replacement design: history-augmented canonical CO/MG
activation extraction + (for validation_ids/test_ids) real generation
and judging (FINAL_STUDY_PROTOCOL.md Sec 13 Round 20; Sec 5R is
SUPERSEDED_BY_HISTORY_AUGMENTED_CANONICAL_CO_MG_RQ2).

14 conditions = 7 mechanism groups (prefix_injection,
refusal_suppression, persona_roleplay = CO; encoding_obfuscation,
payload_splitting, distractors_negated = MG; neutral) x {multi, single}
(`src/history_augmented_co_mg_loader.py`). "multi" is a genuine 6-stage
trajectory: 5 frozen, mechanism-free scaffold turns (real assistant
generation each turn, history carried forward -- the same
"fixed-policy interactive multi-turn protocol" discipline as the
superseded Study B design) followed by the UNMODIFIED canonical CO/MG
mechanism text (or the bare instruction, for neutral) as the 6th turn.
Scaffold length was revised up from an initial 3-turn draft (before any
real confirmatory extraction) to address a statistical-power concern:
a short, topic-unrelated scaffold might show near-zero effect on
well-aligned models regardless of whether a genuine multi-turn effect
exists. Stage count is read from `history_augmented_co_mg_loader.STAGE_KEYS`
-- not hardcoded here -- so a future scaffold-length change only needs
editing the template + that constant.
"single" is exactly Experiment 1's own single-turn condition -- one
turn, no scaffold, byte-identical canonical text.

Per the same compute-saving design as the superseded Study B script,
generation requirement differs by `--ids-key`:
- `direction_ids`: "multi" needs the scaffold stages REALLY generated
  (to build context for the later stages' boundary extraction) but NOT
  the final payload stage (no downstream use, no behavioral label
  wanted for direction estimation). "single" needs only its one forward
  pass, no generation at all.
- `validation_ids`/`test_ids`: "multi" needs every stage generated for
  real (the final stage's response is the judged behavioral outcome).
  "single" also needs one real generation (its response must exist to
  be judged). WildGuard judges every condition's FINAL exchange only.

Token position: `t_generation_boundary` only, same as the superseded
Study B script -- a length, not a substring search.

**This script has never been run against real GPU -- like every driver
in this repo, it needs `--dry-run` first, then a small `--limit` smoke
test on the cluster, before being trusted.**
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(SCRIPT_DIR, "..")
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from history_augmented_co_mg_loader import (  # noqa: E402
    ALL_MECHANISM_GROUPS, FORMS, STAGE_KEYS,
    load_template, load_canonical_texts, render_payload_text, render_single_messages, condition_name,
)
from _behavioral_shared import sha256_hex, git_commit_hash, judge_responses  # noqa: E402

SAMPLED_PROMPTS_EN_ONLY_PATH = os.path.join(REPO_ROOT, "data", "source", "sampled_prompts_en_only.json")
SPLITS_PATH = os.path.join(REPO_ROOT, "data", "splits", "splits.json")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "history_augmented_co_mg_output")

PRIMARY_TOKEN_POSITION = "t_generation_boundary"
ALLOWED_IDS_KEYS = ("direction_ids", "validation_ids", "test_ids")
EXPECTED_COUNT = {"direction_ids": 300, "validation_ids": 72, "test_ids": 200}
# test_ids was unsealed Round 19 for the (now-superseded) Study B design
# and remains unsealed -- this script inherits that unsealed status, it
# does not re-seal or re-unseal anything. Every OTHER driver in this
# repo still rejects test_ids as an --ids-key value.

GENERATION_CONFIG = {"max_new_tokens": 200, "do_sample": False}
WG_BATCH_SIZE = 16

MODEL_TOKENIZER_SOURCES = [
    ("Qwen2.5-7B-Instruct", "QWEN", "/home/h24/baga0553/models/Qwen2.5-7B-Instruct", 16),
    ("Meta-Llama-3.1-8B-Instruct", "LLAMA", "/home/h24/baga0553/models/Llama-3.1-8B-Instruct", 19),
    ("gemma-2-9b-it", "GEMMA", "/home/h24/baga0553/models/gemma-2-9b-it", 25),
]


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


def build_all_rows(instructions, template_data, canonical_texts, mechanisms=None):
    """One row per (instruction, mechanism, form). `mechanisms` restricts
    to a subset (analogous to Study B's `--families` scoping, e.g. for a
    future confirmatory-style run limited to one mechanism)."""
    mechanisms = mechanisms or list(ALL_MECHANISM_GROUPS)
    trajectory_rows, single_pass_rows = [], []
    for instr in instructions:
        for mechanism in mechanisms:
            trajectory_rows.append({
                "instruction_id": instr["id"], "instruction_text": instr["instruction_en"],
                "mechanism": mechanism, "form": "multi", "condition": condition_name(mechanism, "multi"),
                "messages": [],
            })
            messages, provenance = render_single_messages(template_data, canonical_texts, mechanism, instr["instruction_en"])
            single_pass_rows.append({
                "instruction_id": instr["id"], "instruction_text": instr["instruction_en"],
                "mechanism": mechanism, "form": "single", "condition": condition_name(mechanism, "single"),
                "messages": messages, "transform_provenance": provenance,
            })
    return trajectory_rows, single_pass_rows


def locate_generation_boundary(tokenizer, messages):
    result = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    full_prompt_ids = result["input_ids"] if hasattr(result, "keys") else list(result)
    full_prompt_ids = list(full_prompt_ids)
    return full_prompt_ids, len(full_prompt_ids) - 1


def run_dry_run(model_alias, tokenizer_path, ids_key):
    from transformers import AutoTokenizer
    print(f"[{model_alias}] loading tokenizer only: {tokenizer_path}", file=sys.stderr)
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(json.dumps({"result_status": "TOKENIZER_LOAD_FAILED", "model_alias": model_alias,
                           "tokenizer_path": tokenizer_path, "load_error": str(e)}, indent=2, ensure_ascii=False))
        return False

    instructions = load_instructions(ids_key)
    template_data = load_template()
    canonical_texts = load_canonical_texts()
    trajectory_rows, single_pass_rows = build_all_rows(instructions, template_data, canonical_texts)

    n_ok, failures = 0, []
    for row in trajectory_rows:
        try:
            stage1_text = template_data["scaffold_stages"]["stage_1_hook"]
            messages = [{"role": "user", "content": stage1_text}]
            _, t_gen = locate_generation_boundary(tokenizer, messages)
            if t_gen < 0:
                raise ValueError("t_generation_boundary < 0")
            n_ok += 1
        except Exception as e:
            failures.append({"instruction_id": row["instruction_id"], "condition": row["condition"], "reason": str(e)})
    for row in single_pass_rows:
        try:
            _, t_gen = locate_generation_boundary(tokenizer, row["messages"])
            if t_gen < 0:
                raise ValueError("t_generation_boundary < 0")
            n_ok += 1
        except Exception as e:
            failures.append({"instruction_id": row["instruction_id"], "condition": row["condition"], "reason": str(e)})

    total = len(trajectory_rows) + len(single_pass_rows)
    print(json.dumps({
        "result_status": "HISTORY_AUGMENTED_DRY_RUN_PASS" if not failures else "HISTORY_AUGMENTED_DRY_RUN_FAIL",
        "model_alias": model_alias, "ids_key": ids_key,
        "note": "validates 'multi' stage-1 rendering and full 'single' rendering only -- stages 2-4 of 'multi' require real generation, not exercised here",
        "n_rows": total, "n_ok": n_ok, "n_failures": len(failures), "failures": failures[:10],
    }, indent=2, ensure_ascii=False))
    return len(failures) == 0


def get_eos_ids(model, tokenizer):
    eos_ids = model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids)
    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if end_of_turn_id is not None and end_of_turn_id != tokenizer.unk_token_id and end_of_turn_id not in eos_ids:
        eos_ids.append(end_of_turn_id)
    return eos_ids


def extract_boundary_activation(model, tokenizer, messages):
    import torch
    full_prompt_ids, t_gen = locate_generation_boundary(tokenizer, messages)
    input_ids = torch.tensor([full_prompt_ids]).to(model.device)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
    hidden_states = torch.stack(out.hidden_states, dim=0)
    vec = hidden_states[:, 0, t_gen, :].to(torch.float16).cpu()
    return vec, t_gen, int(full_prompt_ids[t_gen]), len(full_prompt_ids)


def run_generation_wave(model, tokenizer, eos_ids, rows, batch_size):
    import torch
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = [tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True) for r in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                                  do_sample=GENERATION_CONFIG["do_sample"], pad_token_id=tokenizer.eos_token_id, eos_token_id=eos_ids)
        for j, gen_ids in enumerate(out):
            new_ids = gen_ids[enc.input_ids.shape[1]:]
            batch[j]["stage_response"] = tokenizer.decode(new_ids, skip_special_tokens=True)
        print(f"    generated {min(start + batch_size, len(rows))}/{len(rows)}", file=sys.stderr)


def run_extraction(model_alias, model_path, primary_layer_expected, ids_key, output_dir, batch_size, limit=None, mechanisms=None, pilot_ids=None):
    """`pilot_ids`, when given (a small fixed list of instruction ids),
    overrides the normal `ids_key`-driven instruction loading and forces
    FULL generation + judging at every stage regardless of `ids_key`,
    tagged `pilot: True`/`PILOT_NON_RESULT` throughout -- same discipline
    as `extract_study_b_activations.py`'s `pilot_ids` (Sec 5.5, extended
    to Study B by Sec 5R.8): exercise the exact generate+judge path the
    expensive real run will use, on a small set, never used to tune
    scaffold/mechanism wording."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[{model_alias}] loading tokenizer + model: {model_path}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    attn_kwargs = {"attn_implementation": "eager"} if "gemma" in model_path.lower() else {}
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", **attn_kwargs)
    model.eval()
    eos_ids = get_eos_ids(model, tokenizer)

    n_layers = model.config.num_hidden_layers
    primary_layer_computed = int(n_layers * 0.6)
    if primary_layer_computed != primary_layer_expected:
        raise ValueError(f"[{model_alias}] frozen primary layer {primary_layer_expected} does not match floor(0.6*{n_layers})={primary_layer_computed}")

    if pilot_ids is not None:
        with open(SAMPLED_PROMPTS_EN_ONLY_PATH, "r", encoding="utf-8") as f:
            pool = json.load(f)
        by_id = {row["id"]: row["instruction_en"] for row in pool}
        missing = [i for i in pilot_ids if i not in by_id]
        if missing:
            raise ValueError(f"pilot_ids not found in {SAMPLED_PROMPTS_EN_ONLY_PATH}: {missing}")
        instructions = [{"id": i, "instruction_en": by_id[i]} for i in pilot_ids]
    else:
        instructions = load_instructions(ids_key)
    if limit is not None:
        instructions = instructions[:limit]
    template_data = load_template()
    canonical_texts = load_canonical_texts()
    active_mechanisms = mechanisms or list(ALL_MECHANISM_GROUPS)
    trajectory_rows, single_pass_rows = build_all_rows(instructions, template_data, canonical_texts, mechanisms=active_mechanisms)

    is_pilot = pilot_ids is not None
    force_full_generation = is_pilot
    manifest_ids_key_label = "pilot" if is_pilot else ids_key
    needs_full_behavioral = ids_key in ("validation_ids", "test_ids") or force_full_generation
    test_data_read = (ids_key == "test_ids")

    commit = git_commit_hash(REPO_ROOT)
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, f"{model_alias}_{manifest_ids_key_label}_history_augmented_manifest.jsonl")
    responses_path = os.path.join(output_dir, f"{model_alias}_{manifest_ids_key_label}_history_augmented_responses.jsonl")
    activations_by_condition = {condition_name(m, f): {} for m in active_mechanisms for f in FORMS}
    judge_input_rows = []
    n_ok, n_fail = 0, 0

    manifest_f = open(manifest_path, "w", encoding="utf-8")
    responses_f = open(responses_path, "w", encoding="utf-8")

    def write_response(row, stage):
        responses_f.write(json.dumps({
            "instruction_id": row["instruction_id"], "condition": row["condition"], "stage": stage,
            "response": row.get("stage_response"),
            "response_len_chars": len(row["stage_response"]) if row.get("stage_response") else 0,
        }, ensure_ascii=False) + "\n")

    def write_record(row, stage, t_gen, tok_id, seq_len, result_status, failure_reason=None):
        nonlocal n_ok, n_fail
        record = {
            "instruction_id": row["instruction_id"], "condition": row["condition"],
            "mechanism": row["mechanism"], "form": row["form"], "stage": stage,
            "model_alias": model_alias, "primary_layer": primary_layer_expected,
            "result_status": ("PILOT_NON_RESULT" if is_pilot else result_status),
            "ids_key": ids_key, "test_data_read": test_data_read,
            "git_commit": commit,
        }
        if is_pilot:
            record["pilot"] = True
        if result_status == "EXTRACTION_OK":
            record[PRIMARY_TOKEN_POSITION] = {"index": t_gen, "token_id": tok_id, "seq_len": seq_len}
            n_ok += 1
        else:
            record["failure_reason"] = failure_reason
            n_fail += 1
        if ids_key in ("validation_ids", "test_ids"):
            record["used_for_direction_estimation"] = False
        manifest_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # --- single: single forward pass, optionally + real generation ---
    print(f"[{model_alias}] single-turn conditions: {len(single_pass_rows)} rows", file=sys.stderr)
    single_pass_ok_rows = []
    for row in single_pass_rows:
        try:
            vec, t_gen, tok_id, seq_len = extract_boundary_activation(model, tokenizer, row["messages"])
            activations_by_condition[row["condition"]][row["instruction_id"]] = {PRIMARY_TOKEN_POSITION: vec}
            write_record(row, "single", t_gen, tok_id, seq_len, "EXTRACTION_OK")
            single_pass_ok_rows.append(row)
        except Exception as e:
            write_record(row, "single", None, None, None, "EXTRACTION_FAIL", str(e))

    if needs_full_behavioral and single_pass_ok_rows:
        print(f"  generating real responses for {len(single_pass_ok_rows)} single rows (batched)", file=sys.stderr)
        run_generation_wave(model, tokenizer, eos_ids, single_pass_ok_rows, batch_size)
        for row in single_pass_ok_rows:
            write_response(row, "single")
            judge_input_rows.append({
                "generation_key": sha256_hex(f"{model_alias}|{row['condition']}|{row['instruction_id']}|single"),
                "messages": row["messages"] + [{"role": "assistant", "content": row["stage_response"]}],
                "response": row["stage_response"],
                "condition": row["condition"], "instruction_id": row["instruction_id"],
            })

    # --- multi: (n_scaffold_stages + 1)-wave real trajectory ---
    n_scaffold_stages = len(STAGE_KEYS)
    final_stage = n_scaffold_stages + 1
    stage_key_for = dict(enumerate(STAGE_KEYS, start=1))
    print(f"[{model_alias}] multi-turn conditions: {len(trajectory_rows)} rows x {final_stage} stages", file=sys.stderr)
    for stage_idx in range(1, final_stage + 1):
        print(f"  stage {stage_idx}: appending user turn + extracting boundary", file=sys.stderr)
        for row in trajectory_rows:
            if stage_idx <= n_scaffold_stages:
                text = template_data["scaffold_stages"][stage_key_for[stage_idx]]
            else:
                text, provenance = render_payload_text(template_data, canonical_texts, row["mechanism"], row["instruction_text"])
                row["transform_provenance"] = provenance
            row["messages"].append({"role": "user", "content": text})
            try:
                vec, t_gen, tok_id, seq_len = extract_boundary_activation(model, tokenizer, row["messages"])
                activations_by_condition[row["condition"]].setdefault(row["instruction_id"], {})[f"stage_{stage_idx}"] = vec
                write_record(row, stage_idx, t_gen, tok_id, seq_len, "EXTRACTION_OK")
            except Exception as e:
                write_record(row, stage_idx, None, None, None, "EXTRACTION_FAIL", str(e))

        need_generation = (stage_idx < final_stage) or needs_full_behavioral
        if need_generation:
            print(f"  stage {stage_idx}: generating real responses ({len(trajectory_rows)} rows)", file=sys.stderr)
            run_generation_wave(model, tokenizer, eos_ids, trajectory_rows, batch_size)
            for row in trajectory_rows:
                write_response(row, stage_idx)
                row["messages"].append({"role": "assistant", "content": row["stage_response"]})
                if stage_idx == final_stage and needs_full_behavioral:
                    judge_input_rows.append({
                        "generation_key": sha256_hex(f"{model_alias}|{row['condition']}|{row['instruction_id']}|final_stage"),
                        "messages": row["messages"], "response": row["stage_response"],
                        "condition": row["condition"], "instruction_id": row["instruction_id"],
                    })

    manifest_f.close()
    responses_f.close()

    for name, per_instruction in activations_by_condition.items():
        activation_path = os.path.join(output_dir, f"{model_alias}_{manifest_ids_key_label}_{name}_activations.pt")
        torch.save(per_instruction, activation_path)
        print(f"  saved {activation_path} ({len(per_instruction)} instructions)", file=sys.stderr)

    # Free the main model before loading WildGuard -- same ordering fix
    # as extract_study_b_activations.py (a real cluster run showed
    # "parameters on the meta device" when WildGuard loaded while the
    # main model was still resident).
    del model
    torch.cuda.empty_cache()

    judge_path = None
    if needs_full_behavioral and judge_input_rows:
        print(f"[{model_alias}] judging {len(judge_input_rows)} final responses with WildGuard", file=sys.stderr)
        judge_extra_tags = {"ids_key": ids_key, "study": "history_augmented_co_mg"}
        if is_pilot:
            judge_extra_tags.update({"pilot": True, "result_status": "PILOT_NON_RESULT"})
        judge_records = judge_responses(judge_input_rows, WG_BATCH_SIZE, judge_extra_tags)
        judge_path = os.path.join(output_dir, f"{model_alias}_{manifest_ids_key_label}_history_augmented_judge_records.jsonl")
        with open(judge_path, "w", encoding="utf-8") as jf:
            for jr, src in zip(judge_records, judge_input_rows):
                jr["condition"] = src["condition"]
                jr["instruction_id"] = src["instruction_id"]
                jf.write(json.dumps(jr, ensure_ascii=False) + "\n")

    print(json.dumps({
        "result_status": ("HISTORY_AUGMENTED_PILOT_DONE" if is_pilot else "HISTORY_AUGMENTED_EXTRACTION_DONE"),
        "model_alias": model_alias, "ids_key": ids_key, "pilot": is_pilot,
        "n_instructions": len(instructions), "n_ok": n_ok, "n_fail": n_fail,
        "manifest_path": manifest_path, "responses_path": responses_path, "judge_path": judge_path, "output_dir": output_dir,
    }, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-alias", required=True, choices=[m[0] for m in MODEL_TOKENIZER_SOURCES])
    parser.add_argument("--ids-key", required=True, choices=list(ALLOWED_IDS_KEYS))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N instructions -- smoke test only.")
    parser.add_argument("--mechanisms", nargs="+", default=None, choices=list(ALL_MECHANISM_GROUPS),
                         help="Restrict to specific mechanism groups (default: all 7). REQUIRED (no default) for --ids-key test_ids.")
    args = parser.parse_args()

    if args.ids_key == "test_ids" and args.mechanisms is None and not args.dry_run:
        raise SystemExit("--ids-key test_ids requires --mechanisms to be explicitly specified -- no default, by design (matches the superseded Study B script's discipline).")

    entry = next(m for m in MODEL_TOKENIZER_SOURCES if m[0] == args.model_alias)
    _, env_suffix, default_path, primary_layer = entry
    model_path = args.model_path or os.environ.get(f"THESIS_FINAL_TOKENIZER_PATH_{env_suffix}", default_path)

    if args.dry_run:
        ok = run_dry_run(args.model_alias, model_path, args.ids_key)
        sys.exit(0 if ok else 1)

    run_extraction(args.model_alias, model_path, primary_layer, args.ids_key, args.output_dir, args.batch_size, args.limit, mechanisms=args.mechanisms)


if __name__ == "__main__":
    main()
