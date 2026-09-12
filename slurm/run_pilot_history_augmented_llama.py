"""History-augmented canonical CO/MG pilot (RQ2 Round 20 replacement
design). Llama-only, mechanical-soundness check ONLY -- same discipline
as Sec 5.5's original pilot and Study B's `run_pilot_study_b_llama.py`.
All 14 conditions, on the SAME 30 pre-fixed `direction_ids` both prior
pilots used (reused verbatim for cross-study pilot consistency).

Forces the FULL generate+judge path at every stage
(`extract_history_augmented_co_mg.run_extraction(..., pilot_ids=...)`,
which internally treats pilot mode like `validation_ids` for
generation purposes) -- the pilot's whole point is to catch real
generation/judging bugs (Gemma-2 empty-response, left-padding) before
the expensive real run.

Every record is tagged `"pilot": true`, `"result_status":
"PILOT_NON_RESULT"`. Output must NEVER be used to tune scaffold or
mechanism wording.

**Never run against real GPU -- needs the same dry-run-first,
small-scale validation as every other driver in this repo.**
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from extract_history_augmented_co_mg import run_extraction, MODEL_TOKENIZER_SOURCES, DEFAULT_OUTPUT_DIR  # noqa: E402

MODEL_ALIAS = "Meta-Llama-3.1-8B-Instruct"

# Same 30 ids as Sec 5.5's original pilot / Study B's pilot -- reused
# verbatim for cross-study pilot consistency.
PILOT_IDS = [
    "p002", "p003", "p004", "p005", "p007", "p009", "p010", "p012", "p015", "p017",
    "p021", "p022", "p024", "p025", "p026", "p027", "p028", "p030", "p032", "p033",
    "p034", "p038", "p039", "p043", "p044", "p047", "p048", "p049", "p050", "p052",
]

DEFAULT_PILOT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "history_augmented_co_mg_pilot_output")


def main():
    entry = next(m for m in MODEL_TOKENIZER_SOURCES if m[0] == MODEL_ALIAS)
    _, env_suffix, default_path, primary_layer = entry
    model_path = os.environ.get(f"THESIS_FINAL_TOKENIZER_PATH_{env_suffix}", default_path)
    output_dir = os.environ.get("HISTORY_AUGMENTED_PILOT_OUTPUT_DIR", DEFAULT_PILOT_OUTPUT_DIR)

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
