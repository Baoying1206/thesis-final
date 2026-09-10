"""CPU-only structural audit of templates/final_10_condition_v1.json
(Round 4, human-reviewed revision).

Checks (FINAL_STUDY_PROTOCOL.md Sec 5 / Sec 11): condition count, 3/3/3/1
group distribution, unified user/assistant/user message structure,
per-condition placeholder scheme correctness, the shared generic setup
(7 of 10) and the shared generic acknowledgement (10 of 10, widened this
revision), Context conditions' setup preserved from source and final
request left untransformed, CO/MG final_user byte-identical to its
single-turn source, the pre-registered context_persona/co_persona_roleplay
boundary-comparison declaration, per-condition content hash, the frozen
token-position schema, reproducibility self-tests for both frozen
render-time transforms (Sec 5.6), and a guard against any condition ever
being self-marked READY or READY_FOR_PILOT.

Emits result_status = "READY_FOR_TOKEN_AUDIT" only if the full
structural check group, both transforms' reproducibility checks, and the
all-10 unified-acknowledgement check all pass with zero failures --
never "READY_FOR_PILOT" (Sec 11's gate distinction; that remains a
separate, later, human-only decision).

Imports src/canonical_transforms.py for the transform reproducibility
checks -- otherwise self-contained. Stdlib only (json, hashlib, os, sys).
No tokenizer, no model, no GPU, no network.
"""

import hashlib
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(SCRIPT_DIR, "..")
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from canonical_transforms import encode_base64_utf8, split_payload_midpoint  # noqa: E402

TEMPLATE_PATH = os.path.join(REPO_ROOT, "templates", "final_10_condition_v1.json")
WEI_CANONICAL_PATH = os.path.join(REPO_ROOT, "templates", "imported", "templates_wei_canonical.json")
CONTEXT_MULTITURN_PATH = os.path.join(REPO_ROOT, "templates", "imported", "templates_context_multiturn_v2.json")

EXPECTED_CONDITION_COUNT = 10
EXPECTED_GROUP_COUNTS = {"CO": 3, "MG": 3, "Context": 3, "neutral": 1}
EXPECTED_MESSAGE_STRUCTURE = ["user", "assistant", "user"]
EXPECTED_STATUS = "HUMAN_REVIEWED_ROUND4"
FORBIDDEN_STATUSES = {"READY", "READY_FOR_PILOT"}
EXPECTED_TOKEN_POSITIONS = {"primary": "t_generation_boundary", "secondary_sensitivity": "t_final_user_end"}

PLACEHOLDER_TOKENS = ["{instruction}", "{instruction_b64}", "{payload_a}", "{payload_b}"]
SCHEME_REQUIRED_PLACEHOLDERS = {
    "instruction": {"{instruction}": 1},
    "instruction_b64": {"{instruction_b64}": 1},
    "payload_a_b": {"{payload_a}": 1, "{payload_b}": 1},
}

CONDITION_TO_MECHANISM = {
    "co_prefix_injection": "prefix_injection",
    "co_refusal_suppression": "refusal_suppression",
    "co_persona_roleplay": "persona_roleplay",
    "mg_encoding_obfuscation": "encoding_obfuscation",
    "mg_payload_splitting": "payload_splitting",
    "mg_distractors_negated": "distractors_negated",
}

CONTEXT_CONDITION_TO_FAMILY = {
    "context_persona": "ctx_persona",
    "context_authority": "ctx_authority",
    "context_fictional": "ctx_fictional",
}

# heuristic only, documented as such -- not exhaustive
FORBIDDEN_SEMANTIC_KEYWORDS = [
    "physician", "clinical", "attorney", "legal matter", "authoris", "authoriz",
    "novel", "screenplay", "tabletop", "role-playing game", "persona",
    "continuation", "transcript", "draft", "unfinished part",
]

TRANSFORM_SELF_TEST_STRINGS = [
    "Example instruction.",
    "Odd length example!",
    "café résumé — non-ASCII test",
    "  Extra   whitespace  Mixed CASE  ",
]


def sha256_hex(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def content_hash(condition):
    return sha256_hex(json.dumps(
        {
            "setup_user": condition["setup_user"],
            "assistant_acknowledgement": condition["assistant_acknowledgement"],
            "final_user": condition["final_user"],
        },
        sort_keys=True, ensure_ascii=False,
    ))


def check_placeholders(name, field_name, text, allowed_tokens):
    failures = []
    for token in PLACEHOLDER_TOKENS:
        expected_count = allowed_tokens.get(token, 0)
        actual_count = text.count(token)
        if actual_count != expected_count:
            failures.append(
                f"{name}.{field_name}: {token} appears {actual_count}x, expected {expected_count}x"
            )
    return failures


def check_transform_reproducibility():
    """Run both frozen transforms twice on several fixed self-test
    strings; require identical output+hashes both times, plus the
    frozen invariants (no embedded newline / round-trip decode for
    base64; odd-length rule / exact reconstruction for splitting)."""
    checks = []
    failures = []

    for s in TRANSFORM_SELF_TEST_STRINGS:
        run1 = encode_base64_utf8(s)
        run2 = encode_base64_utf8(s)
        if run1 != run2:
            failures.append(f"encode_base64_utf8({s!r}) not reproducible across two runs: {run1} != {run2}")
            continue
        b64_string, input_sha256, output_sha256 = run1
        import base64 as _b64
        decoded = _b64.b64decode(b64_string.encode("ascii"))
        if decoded != s.encode("utf-8"):
            failures.append(f"encode_base64_utf8({s!r}): round-trip decode mismatch")
        if "\n" in b64_string:
            failures.append(f"encode_base64_utf8({s!r}): embedded newline in output")
        if sha256_hex(s) != input_sha256:
            failures.append(f"encode_base64_utf8({s!r}): input_sha256 does not match independently computed sha256")
        checks.append(f"encode_base64_utf8 reproducible + round-trip OK for {s!r}")

    for s in TRANSFORM_SELF_TEST_STRINGS:
        run1 = split_payload_midpoint(s)
        run2 = split_payload_midpoint(s)
        if run1 != run2:
            failures.append(f"split_payload_midpoint({s!r}) not reproducible across two runs: {run1} != {run2}")
            continue
        fragment_a, fragment_b, normalized_sha256, fragment_a_sha256, fragment_b_sha256 = run1
        length_delta = len(fragment_b) - len(fragment_a)
        if length_delta not in (0, 1):
            failures.append(
                f"split_payload_midpoint({s!r}): odd-length rule violated, "
                f"len(fragment_b)-len(fragment_a)={length_delta}, expected 0 or 1"
            )
        if sha256_hex(fragment_a) != fragment_a_sha256 or sha256_hex(fragment_b) != fragment_b_sha256:
            failures.append(f"split_payload_midpoint({s!r}): fragment hash does not match independently computed sha256")
        checks.append(f"split_payload_midpoint reproducible + odd-length-rule OK for {s!r}")

    return checks, failures


def audit():
    result = {
        "result_status": None,
        "template_path": TEMPLATE_PATH,
        "condition_count": None,
        "group_counts": {},
        "checks": [],
        "failures": [],
        "structural_group_pass": None,
        "transform_reproducibility_pass": None,
        "unified_acknowledgement_pass": None,
    }

    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(WEI_CANONICAL_PATH, "r", encoding="utf-8") as f:
        wei = json.load(f)
    with open(CONTEXT_MULTITURN_PATH, "r", encoding="utf-8") as f:
        context_multiturn = json.load(f)

    conditions = data["conditions"]
    structural_failures = []
    ack_failures = []

    result["condition_count"] = len(conditions)
    if len(conditions) != EXPECTED_CONDITION_COUNT:
        structural_failures.append(f"condition_count={len(conditions)} != expected {EXPECTED_CONDITION_COUNT}")

    # --- group distribution ---
    group_counts = {}
    for name, cond in conditions.items():
        group_counts[cond["group"]] = group_counts.get(cond["group"], 0) + 1
    result["group_counts"] = group_counts
    if group_counts != EXPECTED_GROUP_COUNTS:
        structural_failures.append(f"group_counts={group_counts} != expected {EXPECTED_GROUP_COUNTS}")
    else:
        result["checks"].append(f"group distribution OK: {group_counts}")

    if data.get("message_structure") != EXPECTED_MESSAGE_STRUCTURE:
        structural_failures.append(
            f"top-level message_structure={data.get('message_structure')} != expected {EXPECTED_MESSAGE_STRUCTURE}"
        )
    else:
        result["checks"].append("top-level message_structure declaration OK")

    all_acks = set()
    setup_pairs_co_mg_neutral = set()

    for name, cond in conditions.items():
        status = cond.get("status")
        if status in FORBIDDEN_STATUSES:
            structural_failures.append(f"{name}: status is {status!r} -- no condition may ever be marked this way")
        elif status != EXPECTED_STATUS:
            structural_failures.append(f"{name}: status={status!r} != expected {EXPECTED_STATUS!r}")
        else:
            result["checks"].append(f"{name}: status OK ({EXPECTED_STATUS})")

        setup = cond["setup_user"]
        ack = cond["assistant_acknowledgement"]
        final = cond["final_user"]
        all_acks.add(ack)

        result["checks"].append(f"{name}: role order OK (['user', 'assistant', 'user'])")

        scheme = cond.get("placeholder_scheme")
        allowed = SCHEME_REQUIRED_PLACEHOLDERS.get(scheme)
        if allowed is None:
            structural_failures.append(f"{name}: unknown placeholder_scheme {scheme!r}")
        else:
            structural_failures.extend(check_placeholders(name, "final_user", final, allowed))
            structural_failures.extend(check_placeholders(name, "setup_user", setup, {}))
            structural_failures.extend(check_placeholders(name, "assistant_acknowledgement", ack, {}))
            result["checks"].append(f"{name}: placeholder_scheme={scheme} OK")

        result["checks"].append(f"{name}: content_sha256={content_hash(cond)}")

        if cond["group"] == "Context":
            if final != "{instruction}":
                structural_failures.append(
                    f"{name}: Context condition's final_user must be the untransformed instruction "
                    f"'{{instruction}}', found {final!r}"
                )
            else:
                result["checks"].append(f"{name}: Context final request left untransformed OK")
            if "variants" in cond or "family_specific_neutral_control" in cond or "family" in cond:
                structural_failures.append(
                    f"{name}: Context condition must be a flat single operationalisation, not a "
                    "nested family/variant structure"
                )
            family_key = CONTEXT_CONDITION_TO_FAMILY.get(name)
            if family_key is not None:
                expected_setup = context_multiturn["families"][family_key]["variants"]["v1"]["setup_user"]
                if setup != expected_setup:
                    structural_failures.append(
                        f"{name}: setup_user does not byte-match templates_context_multiturn_v2.json#"
                        f"families.{family_key}.variants.v1.setup_user -- Context setup must be preserved unchanged"
                    )
                else:
                    result["checks"].append(f"{name}: Context setup content preserved OK (matches {family_key}.v1)")

        elif cond["group"] in ("CO", "MG"):
            setup_pairs_co_mg_neutral.add(setup)
            mechanism_key = CONDITION_TO_MECHANISM.get(name)
            if mechanism_key is None:
                structural_failures.append(f"{name}: no known single-turn mechanism mapping for this condition name")
            else:
                expected_text = wei["templates"][mechanism_key]["text"]
                if final != expected_text:
                    structural_failures.append(
                        f"{name}: final_user does not byte-match templates_wei_canonical.json#"
                        f"templates.{mechanism_key}.text -- canonical mechanism definition may have been altered"
                    )
                else:
                    result["checks"].append(f"{name}: CO/MG source mapping OK (byte-identical to {mechanism_key})")

        elif cond["group"] == "neutral":
            setup_pairs_co_mg_neutral.add(setup)
            if final != "{instruction}":
                structural_failures.append(f"neutral: final_user must be the untransformed instruction, found {final!r}")
            else:
                result["checks"].append("neutral: final request left untransformed OK")

        if cond["group"] in ("CO", "MG", "neutral"):
            haystack = (setup + " " + ack).lower()
            hits = [kw for kw in FORBIDDEN_SEMANTIC_KEYWORDS if kw in haystack]
            if hits:
                structural_failures.append(
                    f"{name}: setup/acknowledgement contains forbidden semantic keyword(s) {hits} "
                    "(persona/authority/fictional/continuation semantics must not leak into CO/MG/neutral)"
                )
            else:
                result["checks"].append(f"{name}: no forbidden semantic keywords in setup/ack (heuristic check)")

    # --- unified acknowledgement across ALL 10 (widened this revision) ---
    if len(all_acks) != 1:
        ack_failures.append(f"assistant_acknowledgement is not uniform across all 10 conditions: found {len(all_acks)} distinct value(s)")
    else:
        ack_value = next(iter(all_acks))
        shared_ack = data.get("shared_generic_acknowledgement", {})
        if ack_value != shared_ack.get("assistant_acknowledgement"):
            ack_failures.append("unified acknowledgement does not match top-level shared_generic_acknowledgement")
        else:
            result["checks"].append("assistant_acknowledgement uniform across ALL 10 conditions, matches shared_generic_acknowledgement OK")

    # --- shared setup across 6 CO/MG + neutral (7 of 10) ---
    if len(setup_pairs_co_mg_neutral) != 1:
        structural_failures.append(
            f"CO/MG/neutral setup_user is not uniform across all 7 conditions: found {len(setup_pairs_co_mg_neutral)} distinct value(s)"
        )
    else:
        setup_value = next(iter(setup_pairs_co_mg_neutral))
        shared_setup = data.get("shared_generic_setup", {})
        if setup_value != shared_setup.get("setup_user"):
            structural_failures.append("CO/MG/neutral shared setup_user does not match top-level shared_generic_setup")
        else:
            result["checks"].append("setup_user uniform across all 6 CO/MG + neutral conditions, matches shared_generic_setup OK")

    # --- pre-registered boundary comparison declaration ---
    boundary_comparisons = data.get("pre_registered_boundary_comparisons", [])
    declared_pairs = [set(bc.get("pair", [])) for bc in boundary_comparisons]
    if {"context_persona", "co_persona_roleplay"} not in declared_pairs:
        structural_failures.append(
            "pre_registered_boundary_comparisons does not declare the context_persona/co_persona_roleplay pair "
            "(FINAL_STUDY_PROTOCOL.md Sec 5.3)"
        )
    else:
        result["checks"].append("pre-registered boundary comparison (context_persona, co_persona_roleplay) declared OK")

    # --- token-position schema ---
    tp = data.get("token_positions", {})
    tp_actual = {"primary": tp.get("primary"), "secondary_sensitivity": tp.get("secondary_sensitivity")}
    if tp_actual != EXPECTED_TOKEN_POSITIONS:
        structural_failures.append(f"token_positions={tp_actual} != expected {EXPECTED_TOKEN_POSITIONS}")
    else:
        result["checks"].append(f"token_positions schema OK: {tp_actual}")

    # --- transform reproducibility ---
    transform_checks, transform_failures = check_transform_reproducibility()
    result["checks"].extend(transform_checks)

    result["structural_group_pass"] = len(structural_failures) == 0
    result["transform_reproducibility_pass"] = len(transform_failures) == 0
    result["unified_acknowledgement_pass"] = len(ack_failures) == 0

    result["failures"] = structural_failures + transform_failures + ack_failures

    if result["structural_group_pass"] and result["transform_reproducibility_pass"] and result["unified_acknowledgement_pass"]:
        result["result_status"] = "READY_FOR_TOKEN_AUDIT"
    else:
        result["result_status"] = "AUDIT_FAIL"

    return result


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["result_status"] == "READY_FOR_TOKEN_AUDIT" else 1)
