"""Loader and messages-renderer for the frozen 10-condition Experiment 2 template.

See FINAL_STUDY_PROTOCOL.md Sec 5 for the design this file implements.
Stdlib only (json, os) plus canonical_transforms (also stdlib-only) for
the two non-plain placeholder schemes. No tokenizer, no model, no torch
import anywhere in this file -- callers pass an already-loaded tokenizer
to render_prompt_with_tokenizer() if/when they need apply_chat_template,
but this module itself never imports one.

Round 4 revision: all 10 conditions now carry real, human-reviewed
(status="HUMAN_REVIEWED_ROUND4") template text and one of three
placeholder schemes (Sec 5.2/5.6). "Text populated" and "human-reviewed"
are NOT the same thing as "authorized to generate" -- this module never
gates on, or asserts, generation authorization; that is a separate,
later decision (FINAL_STUDY_PROTOCOL.md Sec 5.5 pilot) outside this
module's scope.
"""

import json
import os

from canonical_transforms import encode_base64_utf8, split_payload_midpoint

DEFAULT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "templates", "final_10_condition_v2.json",
)
# v1 (superseded Round 14): Context conditions' setup_user under v1 produced
# no significant strict_success uplift in real Qwen/Llama data. Kept
# unmodified at templates/final_10_condition_v1.json as the historical
# Round-1 record -- see FINAL_STUDY_PROTOCOL.md Sec 13 and v2's top-level
# "note" field for the full disclosure of this post-hoc deviation.

EXPECTED_CONDITION_COUNT = 10
EXPECTED_GROUPS = {"CO", "MG", "Context", "neutral"}
EXPECTED_MESSAGE_STRUCTURE = ["user", "assistant", "user"]
EXPECTED_STATUS = "HUMAN_REVIEWED_ROUND4"

PRIMARY_TOKEN_POSITION = "t_generation_boundary"
SECONDARY_TOKEN_POSITION = "t_final_user_end"

PLACEHOLDER_SCHEMES = ("instruction", "instruction_b64", "payload_a_b")


def load_conditions(path=None):
    """Load the 10-condition template file and do minimal structural
    validation (count, group labels). Does not require any particular
    condition status -- callers that need to gate on review/authorization
    state should check the `status` field themselves."""
    path = path or DEFAULT_TEMPLATE_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conditions = data["conditions"]
    if len(conditions) != EXPECTED_CONDITION_COUNT:
        raise ValueError(
            f"expected {EXPECTED_CONDITION_COUNT} conditions, found {len(conditions)} in {path}"
        )

    groups_seen = {c["group"] for c in conditions.values()}
    if not groups_seen.issubset(EXPECTED_GROUPS):
        raise ValueError(f"unexpected group label(s) {groups_seen - EXPECTED_GROUPS} in {path}")

    return data


def has_populated_text(condition):
    """Purely structural: are setup_user/assistant_acknowledgement/
    final_user all non-null? This is NOT a generation-authorization
    check -- see module docstring."""
    return (
        condition.get("setup_user") is not None
        and condition.get("assistant_acknowledgement") is not None
        and condition.get("final_user") is not None
    )


def render_messages(condition, instruction_text):
    """Build the real 3-turn messages list for one condition + one
    source instruction, substituting whichever placeholder(s) the
    condition's placeholder_scheme declares.

    instruction_text is the RAW source instruction (before any
    transform); this function does not read from
    data/source/sampled_prompts.json or any split file itself -- the
    caller is responsible for supplying instruction_text (and for never
    doing so with a test_ids-sourced instruction outside an explicitly
    authorized round, per FINAL_STUDY_PROTOCOL.md Sec 2). Returns
    (messages, transform_provenance) where transform_provenance is None
    for the plain "instruction" scheme, and a dict of input/output
    hashes (FINAL_STUDY_PROTOCOL.md Sec 5.6) for the other two schemes.
    """
    if not has_populated_text(condition):
        raise ValueError("condition has one or more null text fields; cannot render")

    scheme = condition.get("placeholder_scheme")
    final_user = condition["final_user"]
    transform_provenance = None

    if scheme == "instruction":
        if final_user.count("{instruction}") != 1:
            raise ValueError(
                f"final_user must contain exactly one {{instruction}} placeholder, "
                f"found {final_user.count('{instruction}')}"
            )
        rendered_final_user = final_user.replace("{instruction}", instruction_text)

    elif scheme == "instruction_b64":
        if final_user.count("{instruction_b64}") != 1:
            raise ValueError(
                f"final_user must contain exactly one {{instruction_b64}} placeholder, "
                f"found {final_user.count('{instruction_b64}')}"
            )
        b64_string, input_sha256, output_sha256 = encode_base64_utf8(instruction_text)
        rendered_final_user = final_user.replace("{instruction_b64}", b64_string)
        transform_provenance = {
            "transform": "encode_base64_utf8",
            "input_sha256": input_sha256,
            "output_sha256": output_sha256,
        }

    elif scheme == "payload_a_b":
        if final_user.count("{payload_a}") != 1 or final_user.count("{payload_b}") != 1:
            raise ValueError(
                f"final_user must contain exactly one {{payload_a}} and one {{payload_b}} "
                f"placeholder, found {final_user.count('{payload_a}')} / {final_user.count('{payload_b}')}"
            )
        fragment_a, fragment_b, normalized_sha256, fragment_a_sha256, fragment_b_sha256 = (
            split_payload_midpoint(instruction_text)
        )
        rendered_final_user = final_user.replace("{payload_a}", fragment_a).replace("{payload_b}", fragment_b)
        transform_provenance = {
            "transform": "split_payload_midpoint",
            "normalized_sha256": normalized_sha256,
            "fragment_a_sha256": fragment_a_sha256,
            "fragment_b_sha256": fragment_b_sha256,
        }

    else:
        raise ValueError(f"unknown placeholder_scheme {scheme!r}")

    messages = [
        {"role": "user", "content": condition["setup_user"]},
        {"role": "assistant", "content": condition["assistant_acknowledgement"]},
        {"role": "user", "content": rendered_final_user},
    ]
    return messages, transform_provenance


def render_prompt_with_tokenizer(tokenizer, condition, instruction_text, add_generation_prompt=True):
    """Convenience wrapper: render_messages() then tokenizer.apply_chat_template().
    Callers must pass an already-loaded tokenizer; this module never
    loads one itself. Not exercised by the CPU-only audit (no tokenizer
    available there) -- exercised only once a real driver is written.
    Returns (rendered_prompt_string, transform_provenance)."""
    messages, transform_provenance = render_messages(condition, instruction_text)
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    return rendered, transform_provenance


if __name__ == "__main__":
    data = load_conditions()
    conditions = data["conditions"]
    populated = [name for name, c in conditions.items() if has_populated_text(c)]
    human_reviewed = [name for name, c in conditions.items() if c.get("status") == EXPECTED_STATUS]
    print(f"loaded {len(conditions)} conditions from {DEFAULT_TEMPLATE_PATH}")
    print(f"text populated ({len(populated)}): {populated}")
    print(f"status=={EXPECTED_STATUS} ({len(human_reviewed)}): {human_reviewed}")

    for name in ["context_persona", "mg_encoding_obfuscation", "mg_payload_splitting"]:
        messages, provenance = render_messages(conditions[name], "EXAMPLE_INSTRUCTION_TEXT_NOT_REAL_DATA")
        print(f"\nsample rendered messages ({name}):")
        print(json.dumps({"messages": messages, "transform_provenance": provenance}, indent=2, ensure_ascii=False))
