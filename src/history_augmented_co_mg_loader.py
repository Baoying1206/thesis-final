"""Loader for RQ2's Round 20 replacement design: history-augmented
canonical CO/MG multi-turn delivery (FINAL_STUDY_PROTOCOL.md Sec 13
Round 20; Sec 5R is SUPERSEDED_BY_HISTORY_AUGMENTED_CANONICAL_CO_MG_RQ2).

7 mechanism groups (prefix_injection, refusal_suppression,
persona_roleplay = CO; encoding_obfuscation, payload_splitting,
distractors_negated = MG; neutral) x 3 delivery forms (single,
multi_neutral, multi_progressive) = 21 conditions. Each "multi_*" form
is a 5-turn frozen scaffold followed by the canonical mechanism text as
the 6th turn; "multi_neutral"'s scaffold is topic-unrelated small talk,
"multi_progressive"'s scaffold is a generic (non-instruction-specific)
escalating "security research" framing -- added after a literature
check found published high-ASR multi-turn jailbreaks rely on CONTENT
shaping, not structure alone, so a pure structure-only comparison
(multi_neutral) risked being underpowered by design. Both scaffolds are
5 turns (revised up from an initial 3-turn neutral-only draft to
address a statistical-power concern raised before any real confirmatory
extraction). Both are compared against the SAME "single" baseline.

The canonical mechanism TEXT is read live from
`templates/imported/templates_wei_canonical.json` (Experiment 1's own
frozen, unmodified source) -- this module does NOT duplicate that text
anywhere, so byte-identity to Experiment 1's single-turn definitions is
enforced by construction, not by copy-paste that could silently drift.
Placeholder substitution ({instruction}/{instruction_b64}/
{payload_a}+{payload_b}) reuses `canonical_transforms.py` verbatim --
the same frozen, hash-recording transforms Sec 5.6 specifies, not
reimplemented here.

Stdlib only (+ canonical_transforms, also stdlib-only). No tokenizer,
no model, no torch.
"""

import json
import os

from canonical_transforms import encode_base64_utf8, split_payload_midpoint

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(SCRIPT_DIR, "..")

DEFAULT_TEMPLATE_PATH = os.path.join(REPO_ROOT, "templates", "history_augmented_co_mg_v1.json")
DEFAULT_CANONICAL_PATH = os.path.join(REPO_ROOT, "templates", "imported", "templates_wei_canonical.json")

CO_MECHANISMS = ("prefix_injection", "refusal_suppression", "persona_roleplay")
MG_MECHANISMS = ("encoding_obfuscation", "payload_splitting", "distractors_negated")
ALL_MECHANISM_GROUPS = CO_MECHANISMS + MG_MECHANISMS + ("neutral",)
SCAFFOLD_KINDS = ("neutral", "progressive")
MULTI_FORMS = tuple(f"multi_{k}" for k in SCAFFOLD_KINDS)  # ("multi_neutral", "multi_progressive")
FORMS = ("single",) + MULTI_FORMS
STAGE_KEYS = ("stage_1_hook", "stage_2_reinforce", "stage_3_followup", "stage_4_extend", "stage_5_bridge")
FINAL_STAGE_KEY = f"stage_{len(STAGE_KEYS) + 1}"  # e.g. "stage_6" -- the payload turn, one past the scaffold


def multi_form_to_scaffold_kind(form):
    if not form.startswith("multi_"):
        raise ValueError(f"not a multi form: {form!r}")
    kind = form[len("multi_"):]
    if kind not in SCAFFOLD_KINDS:
        raise ValueError(f"unknown scaffold kind {kind!r} in form {form!r}")
    return kind


def load_template(path=None):
    path = path or DEFAULT_TEMPLATE_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    groups_seen = set(data["mechanism_groups"].keys())
    if groups_seen != set(ALL_MECHANISM_GROUPS):
        raise ValueError(f"expected mechanism groups {ALL_MECHANISM_GROUPS}, found {groups_seen} in {path}")
    scaffolds_seen = set(data["scaffolds"].keys())
    if scaffolds_seen != set(SCAFFOLD_KINDS):
        raise ValueError(f"expected scaffold kinds {SCAFFOLD_KINDS}, found {scaffolds_seen} in {path}")
    for kind in SCAFFOLD_KINDS:
        for key in STAGE_KEYS:
            if key not in data["scaffolds"][kind]:
                raise ValueError(f"scaffolds.{kind} missing {key} in {path}")
    return data


def load_canonical_texts(path=None):
    """{mechanism_name: raw_text_with_placeholders} for the 6 real
    mechanisms, read live from templates_wei_canonical.json -- never
    copied elsewhere. 'neutral' is not in this file; handled separately."""
    path = path or DEFAULT_CANONICAL_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for m in CO_MECHANISMS + MG_MECHANISMS:
        out[m] = data["templates"][m]["text"]
    return out


def render_payload(mechanism, placeholder_scheme, canonical_text, instruction_text):
    """Returns (rendered_text, transform_provenance). transform_provenance
    is None for the plain 'instruction' scheme, a dict of hashes for the
    other two -- identical logic/semantics to
    src/final_condition_loader.py's render_messages, reused independently
    here rather than imported, since that module's render_messages() is
    coupled to the OLD (Study-B-predecessor) 10-condition condition
    schema, not this one's 14-condition/mechanism-group schema."""
    if placeholder_scheme == "instruction":
        if canonical_text.count("{instruction}") != 1:
            raise ValueError(f"{mechanism}: expected exactly one {{instruction}} placeholder")
        return canonical_text.replace("{instruction}", instruction_text), None

    if placeholder_scheme == "instruction_b64":
        if canonical_text.count("{instruction_b64}") != 1:
            raise ValueError(f"{mechanism}: expected exactly one {{instruction_b64}} placeholder")
        b64_string, input_sha256, output_sha256 = encode_base64_utf8(instruction_text)
        rendered = canonical_text.replace("{instruction_b64}", b64_string)
        return rendered, {"transform": "encode_base64_utf8", "input_sha256": input_sha256, "output_sha256": output_sha256}

    if placeholder_scheme == "payload_a_b":
        if canonical_text.count("{payload_a}") != 1 or canonical_text.count("{payload_b}") != 1:
            raise ValueError(f"{mechanism}: expected exactly one {{payload_a}} and one {{payload_b}} placeholder")
        fragment_a, fragment_b, normalized_sha256, fragment_a_sha256, fragment_b_sha256 = split_payload_midpoint(instruction_text)
        rendered = canonical_text.replace("{payload_a}", fragment_a).replace("{payload_b}", fragment_b)
        return rendered, {
            "transform": "split_payload_midpoint", "normalized_sha256": normalized_sha256,
            "fragment_a_sha256": fragment_a_sha256, "fragment_b_sha256": fragment_b_sha256,
        }

    raise ValueError(f"unknown placeholder_scheme {placeholder_scheme!r}")


def render_payload_text(template_data, canonical_texts, mechanism, instruction_text):
    """Returns (text, transform_provenance) for the final-turn payload --
    the canonical mechanism text (byte-identical to Experiment 1) for
    the 6 real mechanisms, or the bare instruction for 'neutral'."""
    group = template_data["mechanism_groups"][mechanism]
    if mechanism == "neutral":
        override = group["final_user_override"]
        if override.count("{instruction}") != 1:
            raise ValueError("neutral's final_user_override must contain exactly one {instruction} placeholder")
        return override.replace("{instruction}", instruction_text), None
    canonical_text = canonical_texts[mechanism]
    return render_payload(mechanism, group["placeholder_scheme"], canonical_text, instruction_text)


def render_single_messages(template_data, canonical_texts, mechanism, instruction_text):
    """'single' form: exactly Experiment 1's own single-turn condition
    -- one user turn, no scaffold. Returns (messages, transform_provenance)."""
    text, provenance = render_payload_text(template_data, canonical_texts, mechanism, instruction_text)
    return [{"role": "user", "content": text}], provenance


def condition_name(mechanism, form):
    return f"{mechanism}_{form}"


def parse_condition_name(name):
    for mechanism in ALL_MECHANISM_GROUPS:
        prefix = mechanism + "_"
        if name.startswith(prefix):
            form = name[len(prefix):]
            if form in FORMS:
                return mechanism, form
    raise ValueError(f"unrecognized condition name {name!r}")


if __name__ == "__main__":
    template_data = load_template()
    canonical_texts = load_canonical_texts()
    print(f"loaded template: status={template_data.get('status')}")
    for kind in SCAFFOLD_KINDS:
        print(f"scaffold[{kind}] stages: {list(template_data['scaffolds'][kind].keys())}")
    print(f"mechanism groups ({len(ALL_MECHANISM_GROUPS)}): {ALL_MECHANISM_GROUPS}")
    print(f"forms ({len(FORMS)}): {FORMS}")

    example_instruction = "EXAMPLE_INSTRUCTION_TEXT_NOT_REAL_DATA"
    for mechanism in ALL_MECHANISM_GROUPS:
        text, prov = render_payload_text(template_data, canonical_texts, mechanism, example_instruction)
        print(f"  {mechanism} final_stage/single: {text[:90]}{'...' if len(text) > 90 else ''}  (transform={prov['transform'] if prov else None})")
