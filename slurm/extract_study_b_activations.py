"""Study B (RQ2, FINAL_STUDY_PROTOCOL.md Sec 5R, Round 17) activation
extraction + (for validation_ids only) real generation and judging.

12 conditions = 3 families (persona/authority/fictional) x {P, N, S, C}
(`src/study_b_loader.py`). P/N are genuine 4-stage trajectories: the
model REALLY generates a response after every stage, and that real
response is appended to history before the next stage's user turn is
built ("fixed-policy interactive multi-turn protocol", Sec 5R.1.1) --
the user-side stage scripts are frozen, never adapted to what the
model said. S/C are single-turn (mechanically derived from P/N,
`src/study_b_loader.py`'s compression rule).

Per Sec 5R.8, generation requirement differs by `--ids-key`:
- `direction_ids`: P/N need stages 1-3 REALLY generated (to build
  context for stages 2/3/4's boundary extraction) but NOT stage 4
  (no downstream use, no behavioral label wanted for direction
  estimation -- Sec 6's firewall). S/C need only their one forward
  pass, no generation at all.
- `validation_ids`: P/N need all 4 stages generated for real (stage
  4's response is the judged behavioral outcome). S/C also need one
  real generation each (their response must exist to be judged).
  After all generation, WildGuard judges every condition's FINAL
  exchange only (stage 4 for P/N, the single turn for S/C) --
  `_behavioral_shared.judge_responses`, reused as-is.

Token position: `t_generation_boundary` only (last input token before
that stage's generation) -- no `t_final_user_end` this round (Sec
5R.3). Unlike Sec 6's extraction, this only needs the LENGTH of the
tokenized, generation-primed prompt, not a substring/offset search --
there is no ambiguity to resolve since we are not locating a span
inside the text.

**This script has never been run against real GPU or even validated
end-to-end -- like every driver in this repo, it needs `--dry-run`
first, then a small `--limit` smoke test on the cluster, before being
trusted. Do not assume correctness from a read-through alone.**
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

from study_b_loader import (  # noqa: E402
    FAMILIES, ALL_FORMS, PROGRESSIVE_FORMS, COMPRESSED_FORMS,
    load_template, render_stage_user_text, render_compressed_messages, condition_name,
)
from _behavioral_shared import sha256_hex, git_commit_hash, judge_responses  # noqa: E402

SAMPLED_PROMPTS_EN_ONLY_PATH = os.path.join(REPO_ROOT, "data", "source", "sampled_prompts_en_only.json")
SPLITS_PATH = os.path.join(REPO_ROOT, "data", "splits", "splits.json")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "study_b_output")

PRIMARY_TOKEN_POSITION = "t_generation_boundary"
ALLOWED_IDS_KEYS = ("direction_ids", "validation_ids")
EXPECTED_COUNT = {"direction_ids": 300, "validation_ids": 72}

GENERATION_CONFIG = {"max_new_tokens": 200, "do_sample": False}
WG_BATCH_SIZE = 16

# Same frozen sources as Sec 4/6 (extract_experiment2_activations.py) --
# this study reuses the same 3 models, same primary-layer formula.
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


def build_all_rows(instructions, template_data):
    """One row per (instruction, family, form) -- 12 conditions x
    len(instructions). `form` in {P,N} gets 'messages': [] (built up
    wave by wave); `form` in {S,C} gets its full single-message list
    immediately (nothing to build up)."""
    trajectory_rows, single_pass_rows = [], []
    for instr in instructions:
        for family in FAMILIES:
            for form in PROGRESSIVE_FORMS:
                trajectory_rows.append({
                    "instruction_id": instr["id"], "instruction_text": instr["instruction_en"],
                    "family": family, "form": form, "condition": condition_name(family, form),
                    "messages": [],
                })
            for form in COMPRESSED_FORMS:
                messages = render_compressed_messages(template_data, family, form, instr["instruction_en"])
                single_pass_rows.append({
                    "instruction_id": instr["id"], "instruction_text": instr["instruction_en"],
                    "family": family, "form": form, "condition": condition_name(family, form),
                    "messages": messages,
                })
    return trajectory_rows, single_pass_rows


def locate_generation_boundary(tokenizer, messages):
    """Only `t_generation_boundary` this round (Sec 5R.3) -- a length,
    not a substring search, so no offset-mapping machinery needed
    (contrast with Sec 6's `locate_positions_for_messages`, which also
    resolves `t_final_user_end`)."""
    result = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    full_prompt_ids = result["input_ids"] if hasattr(result, "keys") else list(result)
    full_prompt_ids = list(full_prompt_ids)
    t_gen = len(full_prompt_ids) - 1
    return full_prompt_ids, t_gen


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
    template_data = load_template()
    trajectory_rows, single_pass_rows = build_all_rows(instructions, template_data)

    n_ok, failures = 0, []

    # Dry-run scope limitation (stated plainly, not hidden): stage 2-4
    # of P/N depend on real generation from earlier stages, which this
    # tokenizer-only dry-run cannot produce. It validates stage-1
    # rendering for P/N and full rendering for S/C only -- real
    # stage-2/3/4 boundary detection is exercised only in a real GPU run.
    for row in trajectory_rows:
        try:
            stages = template_data["strategies"][row["family"]][row["form"]]
            stage_1_text = render_stage_user_text(stages, 1, row["instruction_text"])
            messages = [{"role": "user", "content": stage_1_text}]
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
        "result_status": "STUDY_B_DRY_RUN_PASS" if not failures else "STUDY_B_DRY_RUN_FAIL",
        "model_alias": model_alias, "ids_key": ids_key,
        "note": "validates P/N stage-1 rendering and full S/C rendering only -- stages 2-4 of P/N require real generation, not exercised here",
        "n_rows": total, "n_ok": n_ok, "n_failures": len(failures),
        "failures": failures[:10],
    }, indent=2, ensure_ascii=False))
    return len(failures) == 0


def get_eos_ids(model, tokenizer):
    """Same fix as _behavioral_shared.generate_responses (Gemma-2's
    real end-of-turn token can differ from tokenizer.eos_token_id)."""
    eos_ids = model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids)
    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if end_of_turn_id is not None and end_of_turn_id != tokenizer.unk_token_id and end_of_turn_id not in eos_ids:
        eos_ids.append(end_of_turn_id)
    return eos_ids


def extract_boundary_activation(model, tokenizer, messages):
    """Row-at-a-time forward pass (output_hidden_states=True, no
    .generate()) -- matches Sec 6's extraction pattern. Cheap relative
    to generation, so not batched; batching is reserved for the
    generation waves (Sec 5R.8's real cost driver)."""
    import torch
    full_prompt_ids, t_gen = locate_generation_boundary(tokenizer, messages)
    input_ids = torch.tensor([full_prompt_ids]).to(model.device)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
    hidden_states = torch.stack(out.hidden_states, dim=0)  # [n_layers+1, 1, seq_len, hidden]
    vec = hidden_states[:, 0, t_gen, :].to(torch.float16).cpu()
    return vec, t_gen, int(full_prompt_ids[t_gen]), len(full_prompt_ids)


def run_generation_wave(model, tokenizer, eos_ids, rows, batch_size):
    """Batched real generation for one stage -- rows must already have
    this stage's user turn appended to `messages`. Mutates each row,
    adding 'stage_response'. Same eos_ids / eager-attention discipline
    as _behavioral_shared.generate_responses, but operates on an
    already-loaded model (this driver keeps the model loaded across
    all 4 waves, unlike run_formal_behavioral.py which loads once per
    script invocation -- same idea, different call shape)."""
    import torch
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
            for r in batch
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                do_sample=GENERATION_CONFIG["do_sample"],
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=eos_ids,
            )
        for j, gen_ids in enumerate(out):
            new_ids = gen_ids[enc.input_ids.shape[1]:]
            batch[j]["stage_response"] = tokenizer.decode(new_ids, skip_special_tokens=True)
        print(f"    generated {min(start + batch_size, len(rows))}/{len(rows)}", file=sys.stderr)


def run_extraction(model_alias, model_path, primary_layer_expected, ids_key, output_dir, batch_size, limit=None, pilot_ids=None):
    """`pilot_ids`, when given (a small fixed list of instruction ids),
    overrides the normal `ids_key`-driven instruction loading and forces
    FULL generation + judging at every stage regardless of `ids_key`
    (i.e. behaves like `validation_ids` for generation purposes) --
    the point of the pilot (Sec 5.5's discipline, extended to Study B
    per Sec 5R.8) is to exercise the exact same generation+judging path
    the expensive real run will use, on a small instruction set, tagged
    `pilot: True` / `PILOT_NON_RESULT` throughout and never used to
    tune stage wording."""
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
        raise ValueError(
            f"[{model_alias}] frozen primary layer {primary_layer_expected} does not match "
            f"floor(0.6*{n_layers})={primary_layer_computed} -- model config changed since freezing, stop"
        )

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
        instructions = instructions[:limit]  # --limit N = first N INSTRUCTIONS (each yields 12 condition-rows), not N rows
    template_data = load_template()
    trajectory_rows, single_pass_rows = build_all_rows(instructions, template_data)

    is_pilot = pilot_ids is not None
    force_full_generation = is_pilot  # pilot always exercises the full generate+judge path, regardless of ids_key
    manifest_ids_key_label = "pilot" if is_pilot else ids_key

    commit = git_commit_hash(REPO_ROOT)
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, f"{model_alias}_{manifest_ids_key_label}_study_b_manifest.jsonl")
    activations_by_condition = {condition_name(f, form): {} for f in FAMILIES for form in ALL_FORMS}
    judge_input_rows = []  # only ever populated for validation_ids or pilot
    n_ok, n_fail = 0, 0

    manifest_f = open(manifest_path, "w", encoding="utf-8")

    def write_record(row, stage, t_gen, tok_id, seq_len, result_status, failure_reason=None):
        nonlocal n_ok, n_fail
        record = {
            "instruction_id": row["instruction_id"], "condition": row["condition"],
            "family": row["family"], "form": row["form"], "stage": stage,
            "model_alias": model_alias, "primary_layer": primary_layer_expected,
            "result_status": ("PILOT_NON_RESULT" if is_pilot else result_status),
            "ids_key": ids_key, "test_data_read": False,
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
        if ids_key == "validation_ids":
            record["used_for_direction_estimation"] = False
        manifest_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # --- S / C: single forward pass, optionally + real generation (validation_ids or pilot) ---
    print(f"[{model_alias}] single-turn conditions (S/C): {len(single_pass_rows)} rows", file=sys.stderr)
    single_pass_ok_rows = []
    for row in single_pass_rows:
        try:
            vec, t_gen, tok_id, seq_len = extract_boundary_activation(model, tokenizer, row["messages"])
            activations_by_condition[row["condition"]][row["instruction_id"]] = {PRIMARY_TOKEN_POSITION: vec}
            write_record(row, "single", t_gen, tok_id, seq_len, "EXTRACTION_OK")
            single_pass_ok_rows.append(row)
        except Exception as e:
            write_record(row, "single", None, None, None, "EXTRACTION_FAIL", str(e))

    if (ids_key == "validation_ids" or force_full_generation) and single_pass_ok_rows:
        print(f"  generating real responses for {len(single_pass_ok_rows)} S/C rows (batched)", file=sys.stderr)
        run_generation_wave(model, tokenizer, eos_ids, single_pass_ok_rows, batch_size)
        for row in single_pass_ok_rows:
            judge_input_rows.append({
                "generation_key": sha256_hex(f"{model_alias}|{row['condition']}|{row['instruction_id']}|single"),
                "messages": row["messages"] + [{"role": "assistant", "content": row["stage_response"]}],
                "response": row["stage_response"],
                "condition": row["condition"], "instruction_id": row["instruction_id"],
            })

    # --- P / N: 4-wave real trajectory ---
    print(f"[{model_alias}] trajectory conditions (P/N): {len(trajectory_rows)} rows x 4 stages", file=sys.stderr)
    for stage_idx in (1, 2, 3, 4):
        print(f"  stage {stage_idx}: appending user turn + extracting boundary", file=sys.stderr)
        for row in trajectory_rows:
            stages = template_data["strategies"][row["family"]][row["form"]]
            text = render_stage_user_text(stages, stage_idx, row["instruction_text"])
            row["messages"].append({"role": "user", "content": text})
            try:
                vec, t_gen, tok_id, seq_len = extract_boundary_activation(model, tokenizer, row["messages"])
                activations_by_condition[row["condition"]].setdefault(row["instruction_id"], {})[f"stage_{stage_idx}"] = vec
                write_record(row, stage_idx, t_gen, tok_id, seq_len, "EXTRACTION_OK")
            except Exception as e:
                write_record(row, stage_idx, None, None, None, "EXTRACTION_FAIL", str(e))

        need_generation = (stage_idx < 4) or (ids_key == "validation_ids") or force_full_generation
        if need_generation:
            print(f"  stage {stage_idx}: generating real responses ({len(trajectory_rows)} rows)", file=sys.stderr)
            run_generation_wave(model, tokenizer, eos_ids, trajectory_rows, batch_size)
            for row in trajectory_rows:
                row["messages"].append({"role": "assistant", "content": row["stage_response"]})
                if stage_idx == 4 and (ids_key == "validation_ids" or force_full_generation):
                    judge_input_rows.append({
                        "generation_key": sha256_hex(f"{model_alias}|{row['condition']}|{row['instruction_id']}|stage4"),
                        "messages": row["messages"],
                        "response": row["stage_response"],
                        "condition": row["condition"], "instruction_id": row["instruction_id"],
                    })

    manifest_f.close()

    for name, per_instruction in activations_by_condition.items():
        activation_path = os.path.join(output_dir, f"{model_alias}_{manifest_ids_key_label}_{name}_activations.pt")
        torch.save(per_instruction, activation_path)
        print(f"  saved {activation_path} ({len(per_instruction)} instructions)", file=sys.stderr)

    judge_path = None
    if (ids_key == "validation_ids" or force_full_generation) and judge_input_rows:
        print(f"[{model_alias}] judging {len(judge_input_rows)} final responses with WildGuard", file=sys.stderr)
        judge_extra_tags = {"ids_key": ids_key, "study": "study_b"}
        if is_pilot:
            judge_extra_tags.update({"pilot": True, "result_status": "PILOT_NON_RESULT"})
        judge_records = judge_responses(judge_input_rows, WG_BATCH_SIZE, judge_extra_tags)
        judge_path = os.path.join(output_dir, f"{model_alias}_{manifest_ids_key_label}_study_b_judge_records.jsonl")
        with open(judge_path, "w", encoding="utf-8") as jf:
            for jr, src in zip(judge_records, judge_input_rows):
                jr["condition"] = src["condition"]
                jr["instruction_id"] = src["instruction_id"]
                jf.write(json.dumps(jr, ensure_ascii=False) + "\n")

    del model
    torch.cuda.empty_cache()

    print(json.dumps({
        "result_status": ("STUDY_B_PILOT_DONE" if is_pilot else "STUDY_B_EXTRACTION_DONE"),
        "model_alias": model_alias, "ids_key": ids_key, "pilot": is_pilot,
        "n_instructions": len(instructions), "n_ok": n_ok, "n_fail": n_fail,
        "manifest_path": manifest_path, "judge_path": judge_path, "output_dir": output_dir,
    }, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-alias", required=True, choices=[m[0] for m in MODEL_TOKENIZER_SOURCES])
    parser.add_argument("--ids-key", required=True, choices=list(ALLOWED_IDS_KEYS))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N instructions (each yields 12 condition-rows) -- smoke test only.")
    args = parser.parse_args()

    entry = next(m for m in MODEL_TOKENIZER_SOURCES if m[0] == args.model_alias)
    _, env_suffix, default_path, primary_layer = entry
    model_path = args.model_path or os.environ.get(f"THESIS_FINAL_TOKENIZER_PATH_{env_suffix}", default_path)

    if args.dry_run:
        ok = run_dry_run(args.model_alias, model_path, args.ids_key)
        sys.exit(0 if ok else 1)

    run_extraction(args.model_alias, model_path, primary_layer, args.ids_key, args.output_dir, args.batch_size, args.limit)


if __name__ == "__main__":
    main()
