"""Study B pilot (RQ2, FINAL_STUDY_PROTOCOL.md Sec 5R.8, Round 17).
Llama-only, mechanical-soundness check ONLY -- same discipline as Sec
5.5's original pilot. All 12 conditions, on the SAME 30 pre-fixed
`direction_ids` Sec 5.5 already uses (reused for consistency across
both studies' pilots, per Sec 5R.8's own note).

Unlike Sec 5R.8's cost estimate (which assumed the cheaper
direction_ids-role call pattern -- no stage-4 generation, no judging),
this driver forces the FULL generate+judge path at every stage
(`extract_study_b_activations.run_extraction(..., pilot_ids=...)`,
which internally treats pilot mode like `validation_ids` for
generation purposes) -- because the pilot's whole point is to catch
real generation/judging bugs (this session's Gemma-2 empty-response
and left-padding bugs were both only caught this way, never by a
forward-pass-only check) before the expensive real run, matching the
ORIGINAL Sec 5.5 pilot's actual purpose more faithfully than the
cheaper estimate in Sec 5R.8. This is a deliberate, noted deviation
from that estimate, not an oversight -- flag it if reviewing.

Every record is tagged `"pilot": true`, `"result_status":
"PILOT_NON_RESULT"` (extract_study_b_activations.py's pilot-mode
logic). Output must NEVER be used to tune stage wording (Sec 5.5's
standing rule, unchanged).

**Never run against real GPU -- needs the same dry-run-first,
small-scale validation as every other driver in this repo.**
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from extract_study_b_activations import run_extraction, MODEL_TOKENIZER_SOURCES, DEFAULT_OUTPUT_DIR  # noqa: E402

MODEL_ALIAS = "Meta-Llama-3.1-8B-Instruct"

# Same 30 ids as Sec 5.5's original pilot (FINAL_STUDY_PROTOCOL.md) --
# reused verbatim for cross-study pilot consistency (Sec 5R.8's note).
PILOT_IDS = [
    "p002", "p003", "p004", "p005", "p007", "p009", "p010", "p012", "p015", "p017",
    "p021", "p022", "p024", "p025", "p026", "p027", "p028", "p030", "p032", "p033",
    "p034", "p038", "p039", "p043", "p044", "p047", "p048", "p049", "p050", "p052",
]

DEFAULT_PILOT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "study_b_pilot_output")


def main():
    entry = next(m for m in MODEL_TOKENIZER_SOURCES if m[0] == MODEL_ALIAS)
    _, env_suffix, default_path, primary_layer = entry
    model_path = os.environ.get(f"THESIS_FINAL_TOKENIZER_PATH_{env_suffix}", default_path)
    output_dir = os.environ.get("STUDY_B_PILOT_OUTPUT_DIR", DEFAULT_PILOT_OUTPUT_DIR)

    # ids_key is nominally "direction_ids" here (PILOT_IDS is a subset of
    # that split) but is functionally irrelevant to what gets generated --
    # pilot_ids= forces the full generate+judge path regardless (see
    # module docstring).
    run_extraction(
        MODEL_ALIAS, model_path, primary_layer, ids_key="direction_ids",
        output_dir=output_dir, batch_size=8, pilot_ids=PILOT_IDS,
    )


if __name__ == "__main__":
    main()
