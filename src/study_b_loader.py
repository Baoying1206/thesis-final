"""Loader for Study B's 12-condition 2x2 progressive-multi-turn design
(FINAL_STUDY_PROTOCOL.md Sec 5R, Round 17). Reads
`templates/study_b_progressive_multiturn_v1.json` (status
DRAFT_NOT_HUMAN_REVIEWED -- this loader does not gate on that status;
callers that need to enforce a frozen/reviewed template must check
`status` themselves, same discipline as `final_condition_loader.py`).

12 conditions = 3 families (persona/authority/fictional) x {P, N, S, C}.
P/N are hand-authored (4-stage hook/reinforce/bridge/payload -- stages
1-3 byte-identical across all instructions in a family, only stage 4
substitutes {instruction}). S/C are NOT stored as text in the template
file at all -- both are deterministically derived at render time from
P's/N's stage_1/2/3 respectively, via the identical compression rule
(Sec 5R.1). This module is the ONLY place that implements that
compression rule -- do not reimplement it elsewhere, or S/C risk
drifting from their source P/N text.

Stdlib only. No tokenizer, no model, no torch import anywhere in this
file.
"""

import json
import os

DEFAULT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "templates", "study_b_progressive_multiturn_v1.json",
)

FAMILIES = ("persona", "authority", "fictional")
PROGRESSIVE_FORMS = ("P", "N")
COMPRESSED_FORMS = ("S", "C")
ALL_FORMS = ("P", "N", "S", "C")
ALL_CONDITIONS = tuple(f"{fam}_{form}" for fam in FAMILIES for form in ALL_FORMS)  # 12
STAGE_KEYS = ("stage_1_hook", "stage_2_reinforce", "stage_3_bridge", "stage_4_payload")

# Sec 5R.1: identical connective for S (from P) and C (from N), across
# all 3 families -- a single frozen constant, not per-family text.
FIXED_CONNECTIVE = "Now, here's what I'd like to ask: "


def load_template(path=None):
    """Structural validation only (Sec 5R.1's 12-condition shape) --
    does not gate on `status`, same discipline as
    final_condition_loader.load_conditions()."""
    path = path or DEFAULT_TEMPLATE_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    families_seen = set(data["strategies"].keys())
    if families_seen != set(FAMILIES):
        raise ValueError(f"expected families {FAMILIES}, found {families_seen} in {path}")

    for fam in FAMILIES:
        for form in PROGRESSIVE_FORMS:
            if form not in data["strategies"][fam]:
                raise ValueError(f"{fam} missing form {form!r} in {path}")
            stages = data["strategies"][fam][form]
            missing = [k for k in STAGE_KEYS if k not in stages]
            if missing:
                raise ValueError(f"{fam}/{form} missing stage keys {missing} in {path}")

    return data


def derive_compressed_text(progressive_stages, instruction_text):
    """Deterministic compression rule (Sec 5R.1): concatenate
    stage_1_hook + stage_2_reinforce + stage_3_bridge (verbatim, joined
    by single spaces), then the fixed connective, then the instruction.
    Used IDENTICALLY for S (called with P's stages) and C (called with
    N's stages) -- this function takes no branch on which one it's
    being used for, so S/C cannot drift apart from their source P/N
    text by construction."""
    for k in ("stage_1_hook", "stage_2_reinforce", "stage_3_bridge"):
        if k not in progressive_stages:
            raise ValueError(f"missing {k} -- cannot derive compressed text")
    parts = [
        progressive_stages["stage_1_hook"],
        progressive_stages["stage_2_reinforce"],
        progressive_stages["stage_3_bridge"],
    ]
    return " ".join(parts) + " " + FIXED_CONNECTIVE + instruction_text


def render_stage_user_text(progressive_stages, stage_idx, instruction_text):
    """stage_idx in {1,2,3,4}. Stages 1-3 are byte-identical fixed text
    (no substitution, ignores instruction_text); stage 4 substitutes
    {instruction}."""
    if stage_idx not in (1, 2, 3, 4):
        raise ValueError(f"stage_idx must be 1-4, got {stage_idx}")
    key = STAGE_KEYS[stage_idx - 1]
    text = progressive_stages[key]
    if stage_idx == 4:
        n = text.count("{instruction}")
        if n != 1:
            raise ValueError(f"stage_4_payload must contain exactly one {{instruction}} placeholder, found {n}")
        return text.replace("{instruction}", instruction_text)
    return text


def render_compressed_messages(template_data, family, form, instruction_text):
    """form in {'S','C'}. Returns a 1-message list:
    [{"role": "user", "content": ...}]. S derives from P, C derives
    from N -- both via the identical derive_compressed_text()."""
    if form not in COMPRESSED_FORMS:
        raise ValueError(f"render_compressed_messages is only for S/C, got form={form!r}")
    source_form = "P" if form == "S" else "N"
    source_stages = template_data["strategies"][family][source_form]
    text = derive_compressed_text(source_stages, instruction_text)
    return [{"role": "user", "content": text}]


def condition_name(family, form):
    return f"{family}_{form}"


def parse_condition_name(name):
    family, form = name.rsplit("_", 1)
    if family not in FAMILIES or form not in ALL_FORMS:
        raise ValueError(f"unrecognized condition name {name!r}")
    return family, form


if __name__ == "__main__":
    data = load_template()
    print(f"loaded template: status={data.get('status')} revision={data.get('revision')}")
    print(f"conditions ({len(ALL_CONDITIONS)}): {list(ALL_CONDITIONS)}")

    example_instruction = "EXAMPLE_INSTRUCTION_TEXT_NOT_REAL_DATA"
    for fam in FAMILIES:
        p_stages = data["strategies"][fam]["P"]
        for stage_idx in (1, 2, 3, 4):
            text = render_stage_user_text(p_stages, stage_idx, example_instruction)
            print(f"  {fam}/P/stage_{stage_idx}: {text[:80]}{'...' if len(text) > 80 else ''}")
        s_messages = render_compressed_messages(data, fam, "S", example_instruction)
        print(f"  {fam}/S (derived from P): {s_messages[0]['content'][:100]}...")
        c_messages = render_compressed_messages(data, fam, "C", example_instruction)
        print(f"  {fam}/C (derived from N): {c_messages[0]['content'][:100]}...")
