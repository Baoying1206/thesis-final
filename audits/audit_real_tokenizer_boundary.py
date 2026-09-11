"""Real-tokenizer boundary audit for the 10 frozen conditions
(FINAL_STUDY_PROTOCOL.md Sec 5/9, Round 5). Loads each model's REAL
tokenizer (never model weights -- only `AutoTokenizer.from_pretrained`,
never `AutoModel*`) and calls the official
`tokenizer.apply_chat_template(..., add_generation_prompt=True)` for
each of the 10 conditions, to verify and record the two frozen token
positions:

- t_generation_boundary (primary): the last index of the full prompt's
  input_ids (i.e. len(full_prompt_ids) - 1), after apply_chat_template
  has appended its generation-priming tokens.
- t_final_user_end (secondary sensitivity): the index of the LAST token
  whose text belongs to the actual final_user content -- located via an
  exact, uniqueness-checked substring search plus a fast-tokenizer
  offset mapping, never assumed to be "N tokens before the end" and
  never allowed to resolve to a turn/role special token.

Any of the following is a hard per-condition FAILURE (never a guess):
condition missing from the template, render_messages() raising,
transform inconsistency, the final_user substring not found or not
unique in the rendered text, a non-fast tokenizer (no offset mapping
available), the string-path re-tokenization not matching the canonical
tokenize=True path (would indicate hidden truncation/divergence), no
token found overlapping the content span, or the resolved
t_final_user_end token being one of the tokenizer's own special tokens.
A tokenizer that fails to load at all (e.g. a gated HF repo with no
local access) fails that entire model, not just one condition.

Does not require attack conditions to match neutral's token length and
never pads anything -- lengths are reported, not equalized. Never reads
data/source/sampled_prompts.json or any split file; uses one fixed,
clearly-fake placeholder instruction for every condition. No GPU, no
`AutoModel`, no generation, no WildGuard, no test_ids.

Tokenizer source resolution (path + a "version" proxy) per model:
1. An environment-variable override, if set:
   THESIS_FINAL_TOKENIZER_PATH_<ALIAS> (ALIAS = QWEN / LLAMA / GEMMA).
2. Otherwise the frozen cluster path from MODEL_TOKENIZER_SOURCES below
   (matches src/defence_metrics_reference.py's MODEL_PATHS, duplicated
   here rather than imported to avoid pulling in that file's unrelated
   old Exp3 constants).
"""

import hashlib
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(SCRIPT_DIR, "..")
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from final_condition_loader import load_conditions, render_messages  # noqa: E402

DUMMY_INSTRUCTION = "AUDIT_PLACEHOLDER_INSTRUCTION_TEXT_NOT_REAL_DATA_ROUND5_TOKEN_BOUNDARY_CHECK"

# (model_alias, env_override_suffix, frozen cluster path) -- matches
# FINAL_STUDY_PROTOCOL.md Sec 3 and src/defence_metrics_reference.py's
# MODEL_PATHS.
MODEL_TOKENIZER_SOURCES = [
    ("Qwen2.5-7B-Instruct", "QWEN", "/home/h24/baga0553/models/Qwen2.5-7B-Instruct"),
    ("Meta-Llama-3.1-8B-Instruct", "LLAMA", "/home/h24/baga0553/models/Llama-3.1-8B-Instruct"),
    ("gemma-2-9b-it", "GEMMA", "/home/h24/baga0553/models/gemma-2-9b-it"),
]

PRIMARY_TOKEN_POSITION = "t_generation_boundary"
SECONDARY_TOKEN_POSITION = "t_final_user_end"


def resolve_tokenizer_source(alias, env_suffix, default_path):
    env_var = f"THESIS_FINAL_TOKENIZER_PATH_{env_suffix}"
    override = os.environ.get(env_var)
    if override:
        return override, f"env override {env_var}"
    return default_path, "frozen cluster path (MODEL_TOKENIZER_SOURCES)"


def sha256_hex(s):
    if isinstance(s, str):
        s = s.encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def content_hash(condition):
    return sha256_hex(json.dumps(
        {
            "setup_user": condition["setup_user"],
            "assistant_acknowledgement": condition["assistant_acknowledgement"],
            "final_user": condition["final_user"],
        },
        sort_keys=True, ensure_ascii=False,
    ))


def tokenizer_version_proxy(tokenizer, source_path):
    if os.path.isdir(source_path):
        for fname in ("tokenizer_config.json", "tokenizer.json"):
            fpath = os.path.join(source_path, fname)
            if os.path.exists(fpath):
                with open(fpath, "rb") as f:
                    return {"kind": "local_file_sha256", "file": fname, "sha256": hashlib.sha256(f.read()).hexdigest()}
        return {"kind": "local_dir_no_tokenizer_config_found", "sha256": None}
    try:
        from huggingface_hub import model_info
        info = model_info(source_path)
        return {"kind": "hf_hub_resolved_revision", "sha": info.sha}
    except Exception as e:
        return {"kind": "hf_hub_revision_lookup_failed", "error": str(e)}


def load_tokenizer(source_path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(source_path)
    return tokenizer


def extract_input_ids(result):
    """apply_chat_template(tokenize=True) return type is not stable
    across transformers versions: some return a flat list[int], others
    (e.g. transformers>=5) return a BatchEncoding/dict with an
    'input_ids' key. Normalize to a flat list[int] either way rather
    than assuming one shape."""
    if hasattr(result, "keys"):
        return list(result["input_ids"])
    if hasattr(result, "input_ids"):
        return list(result.input_ids)
    return list(result)


def audit_one_condition(tokenizer, name, condition):
    result = {"condition": name, "result_status": None, "failures": []}

    try:
        messages, transform_provenance = render_messages(condition, DUMMY_INSTRUCTION)
    except Exception as e:
        result["failures"].append(f"render_messages raised: {e}")
        result["result_status"] = "TOKEN_AUDIT_FAIL"
        return result

    final_user_rendered = messages[2]["content"]
    result["template_content_sha256"] = content_hash(condition)
    result["transform_provenance"] = transform_provenance

    try:
        full_prompt_ids = extract_input_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        full_text_no_priming = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        ids_no_priming = extract_input_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    except Exception as e:
        result["failures"].append(f"apply_chat_template raised: {e}")
        result["result_status"] = "TOKEN_AUDIT_FAIL"
        return result

    occurrences = full_text_no_priming.count(final_user_rendered)
    if occurrences != 1:
        result["failures"].append(
            f"final_user content substring found {occurrences}x in rendered text (need exactly 1) -- "
            "position not uniquely locatable"
        )
        result["result_status"] = "TOKEN_AUDIT_FAIL"
        return result
    char_start = full_text_no_priming.rfind(final_user_rendered)
    char_end = char_start + len(final_user_rendered)

    if not getattr(tokenizer, "is_fast", False):
        result["failures"].append("tokenizer is not a fast tokenizer -- no offset mapping available, cannot uniquely locate t_final_user_end")
        result["result_status"] = "TOKEN_AUDIT_FAIL"
        return result

    encoding = tokenizer(full_text_no_priming, add_special_tokens=False, return_offsets_mapping=True)
    retokenized_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    if retokenized_ids != list(ids_no_priming):
        result["failures"].append(
            "re-tokenizing the string form of apply_chat_template does not match the canonical "
            "tokenize=True id sequence -- cannot trust offsets (possible truncation or template divergence)"
        )
        result["result_status"] = "TOKEN_AUDIT_FAIL"
        return result

    if list(full_prompt_ids[:len(ids_no_priming)]) != list(ids_no_priming):
        result["failures"].append(
            "add_generation_prompt=True prefix does not match the add_generation_prompt=False sequence -- "
            "index alignment between the two calls is invalid"
        )
        result["result_status"] = "TOKEN_AUDIT_FAIL"
        return result

    candidate_indices = [i for i, (s, e) in enumerate(offsets) if char_start <= s < char_end]
    if not candidate_indices:
        result["failures"].append("no token found overlapping the final_user content span -- position not uniquely locatable")
        result["result_status"] = "TOKEN_AUDIT_FAIL"
        return result
    t_final_user_end = max(candidate_indices)

    special_ids = set(tokenizer.all_special_ids) if hasattr(tokenizer, "all_special_ids") else set()
    if ids_no_priming[t_final_user_end] in special_ids:
        result["failures"].append(
            f"resolved t_final_user_end token (id={ids_no_priming[t_final_user_end]}) is a special/turn-marker "
            "token -- refusing to misuse a turn-end marker as content end"
        )
        result["result_status"] = "TOKEN_AUDIT_FAIL"
        return result

    t_generation_boundary = len(full_prompt_ids) - 1
    total_token_count = len(full_prompt_ids)

    result["result_status"] = "TOKEN_AUDIT_PASS"
    result["total_token_count"] = total_token_count
    result[PRIMARY_TOKEN_POSITION] = {
        "index": t_generation_boundary,
        "token_id": int(full_prompt_ids[t_generation_boundary]),
    }
    result[SECONDARY_TOKEN_POSITION] = {
        "index": t_final_user_end,
        "token_id": int(ids_no_priming[t_final_user_end]),
    }
    result["positions_coincide"] = (t_generation_boundary == t_final_user_end)
    return result


def audit_one_model(model_alias, source_path, source_origin, template_data):
    model_result = {
        "model_alias": model_alias,
        "tokenizer_path": source_path,
        "tokenizer_path_origin": source_origin,
        "result_status": None,
        "pilot_forbidden": True,
        "conditions": {},
    }

    try:
        tokenizer = load_tokenizer(source_path)
    except Exception as e:
        model_result["result_status"] = "TOKENIZER_LOAD_FAILED"
        model_result["load_error"] = str(e)
        return model_result

    import transformers
    model_result["tokenizer_class"] = type(tokenizer).__name__
    model_result["tokenizer_name_or_path"] = getattr(tokenizer, "name_or_path", None)
    model_result["transformers_version"] = transformers.__version__
    model_result["tokenizer_version_proxy"] = tokenizer_version_proxy(tokenizer, source_path)

    conditions = template_data["conditions"]
    neutral_tokens = None
    condition_order = list(conditions.keys())
    if "neutral" in condition_order:
        condition_order.remove("neutral")
        condition_order = ["neutral"] + condition_order

    all_pass = True
    for name in condition_order:
        condition = conditions[name]
        cond_result = audit_one_condition(tokenizer, name, condition)
        if name == "neutral" and cond_result["result_status"] == "TOKEN_AUDIT_PASS":
            neutral_tokens = cond_result["total_token_count"]
        if cond_result["result_status"] != "TOKEN_AUDIT_PASS":
            all_pass = False
        model_result["conditions"][name] = cond_result

    for name, cond_result in model_result["conditions"].items():
        if cond_result["result_status"] == "TOKEN_AUDIT_PASS" and neutral_tokens is not None:
            cond_result["delta_tokens_vs_neutral"] = cond_result["total_token_count"] - neutral_tokens
        else:
            cond_result["delta_tokens_vs_neutral"] = None

    if all_pass and len(model_result["conditions"]) == 10 and neutral_tokens is not None:
        model_result["result_status"] = "TOKEN_AUDIT_PASS"
        model_result["pilot_forbidden"] = False
    else:
        model_result["result_status"] = "TOKEN_AUDIT_FAIL"
        model_result["pilot_forbidden"] = True

    return model_result


def audit():
    template_data = load_conditions()
    if len(template_data["conditions"]) != 10:
        return {
            "result_status": "TOKEN_AUDIT_FAIL",
            "pilot_forbidden": True,
            "failures": [f"expected 10 conditions in template, found {len(template_data['conditions'])}"],
            "models": {},
        }

    models = {}
    for alias, env_suffix, default_path in MODEL_TOKENIZER_SOURCES:
        source_path, origin = resolve_tokenizer_source(alias, env_suffix, default_path)
        models[alias] = audit_one_model(alias, source_path, origin, template_data)

    all_models_pass = all(m["result_status"] == "TOKEN_AUDIT_PASS" for m in models.values())
    return {
        "result_status": "TOKEN_AUDIT_PASS" if all_models_pass else "TOKEN_AUDIT_FAIL",
        "pilot_forbidden": not all_models_pass,
        "template_path": os.path.join(REPO_ROOT, "templates", "final_10_condition_v2.json"),
        "dummy_instruction_used": DUMMY_INSTRUCTION,
        "models": models,
    }


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["result_status"] == "TOKEN_AUDIT_PASS" else 1)
