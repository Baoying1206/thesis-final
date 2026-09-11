# thesis_final

Independent, final-stage thesis experiment. Separate Git history from
`~/new_experiment` (the prior, still-frozen exploratory work) -- this
repo does not share commits, objects, or working files with it.
`~/new_experiment` is treated as strictly **read-only** reference
material from here on: no code, data, commits, or output in that repo
are modified, moved, or deleted by anything in this directory.

## What this is

Two research questions only (see `FINAL_STUDY_PROTOCOL.md` for the full
design):

- **RQ1**: Does Wei et al.'s competing-objectives (CO) / mismatched-
  generalisation (MG) taxonomy correspond to a consistent internal
  activation geometry across three models?
- **RQ2**: Does multi-turn contextual reconfiguration show a stable
  internal representation and behavioral effect that CANNOT be explained
  by CO or MG alone -- i.e. does it constitute a candidate supplementary
  category? (Not assumed true; several other outcomes are explicitly
  anticipated -- see the protocol.)

## Status

**Round 4 (human-reviewed revision): Experiment 2's 10-condition design
is FROZEN -- template text, the 10-condition set, the unified
acknowledgement, the Base64/payload-splitting transform rules, and the
two token-position definitions must not be further optimized.** No
model has been loaded and no GPU/generation/WildGuard has run as of the
end of Round 4. This study **reuses** `~/new_experiment`'s existing
572-item English instruction pool and its existing frozen
`direction_ids`/`validation_ids`/`test_ids` (300/72/200) split verbatim
-- no new split has been created, no dataset independence from that
pool is claimed (see `FINAL_STUDY_PROTOCOL.md` Sec 2). `test_ids`
remains sealed -- its instruction text has not been read.

Experiment 2's 10 conditions (3 CO mechanisms + 3 MG mechanisms + 3
Context operationalisations + 1 shared neutral,
`templates/final_10_condition_v1.json`) all carry real,
human-reviewed template text (`status: HUMAN_REVIEWED_ROUND4`) -- every
condition shares one byte-identical acknowledgement
("Understood — go ahead with your request."); the 3 Context conditions'
setup framing and the 6 CO/MG conditions' canonical mechanism text
(applied to the final turn, byte-identical to the single-turn study) are
unchanged from the source templates. The frozen render-time transforms
for `mg_encoding_obfuscation` (UTF-8 + standard Base64) and
`mg_payload_splitting` (deterministic normalized-text midpoint split)
live in `src/canonical_transforms.py`. The study is **English-only** --
instructions are read from `data/source/sampled_prompts_en_only.json`
(`id`/`category`/`instruction_en` only, derived losslessly in Round 7).
The original `data/source/sampled_prompts.json` (which carried the full
9-language `instructions` field inherited from the source pool, never
read by any code here) was **deleted in Round 10** -- recoverable from
`~/new_experiment` or this repo's git history if ever needed again
(`FINAL_STUDY_PROTOCOL.md` Sec 2). See `FINAL_STUDY_PROTOCOL.md` Sec 13
for remaining open items.

**Round 6**: the repo was simplified to just `data/`, `templates/`,
`src/`, `slurm/` plus this file, `FINAL_STUDY_PROTOCOL.md`, and
`MIGRATION_MANIFEST.json` -- the CPU-only structural and real-tokenizer
audit scripts (`audits/`), `config/OUTPUT_SCHEMA.md`, and the 2
reference-only source docs (`legacy_reference/`) were deleted. What they
verified is preserved in writing in `FINAL_STUDY_PROTOCOL.md` Sec 9/11;
the structural audit script remains recoverable from this repo's git
history if needed again.

**Round 7**: `audits/audit_real_tokenizer_boundary.py` was restored and
is kept permanently -- it's needed to actually run the real-tokenizer
boundary check (`FINAL_STUDY_PROTOCOL.md` Sec 11.1). Locally it was only
validated against Qwen2.5-7B-Instruct (via HF Hub).

**Round 8**: run for real on the cluster against all 3 frozen local
tokenizer paths -- **`TOKEN_AUDIT_PASS` for Qwen2.5-7B-Instruct,
Meta-Llama-3.1-8B-Instruct, and gemma-2-9b-it**, 10/10 conditions each
(`FINAL_STUDY_PROTOCOL.md` Sec 11.1,
`data/manifests/real_tokenizer_boundary_audit_cluster_round8.json`).
**`READY_FOR_PILOT` confirmed 2026-09-11** (`FINAL_STUDY_PROTOCOL.md`
Sec 5.5) -- the first time this status has been set in this repo.

**Round 9**: `slurm/run_pilot_llama.py` written -- loads
Meta-Llama-3.1-8B-Instruct + WildGuard via plain `transformers`, renders
the frozen 30 ids x 10 conditions, generates, judges, writes
`pilot_generation_records.jsonl`/`pilot_judge_records.jsonl` (every
record tagged `"pilot": true`/`"result_status": "PILOT_NON_RESULT"`,
never to be used to tune templates). `--dry-run` (CPU-only, no model)
validated locally: 300/300 rows render correctly, including the
Base64/payload-splitting transforms against real pool text.

`slurm/extract_experiment1_activations.py` also written -- Experiment 1
(RQ1) raw activation extraction, one forward pass per
`(condition, instruction_id)` (no `.generate()`), both frozen token
positions read from the same pass, full-layer hidden states saved per
`(model, condition)`. `--dry-run` validated 2,400/2,400 rows for
Qwen2.5-7B-Instruct (via HF Hub); Llama/Gemma untested here (gated).
This script produces raw activation records only -- it does not compute
`d_m`, calibration, or any Sec 4.3 statistic.

**Round 10**: Experiment 2's two remaining drivers written --
`slurm/extract_experiment2_activations.py` (representation extraction,
parameterized `--ids-key direction_ids|validation_ids`; `--dry-run`
validated 3,000/3,000 and 720/720 rows respectively for
Qwen2.5-7B-Instruct) and `slurm/run_formal_behavioral.py` (the FORMAL
behavioral run -- all 3 models, all 72 `validation_ids`, all 10
conditions, 2,160 generations + 2,160 judgements; `--dry-run` validated
720/720 rows for all 3 models). Generation/judging code was factored
out of `run_pilot_llama.py` into `slurm/_behavioral_shared.py` so the
pilot and the formal run share identical logic and cannot silently
diverge.

**Round 12**: `slurm/extract_experiment1_activations.py` validated on
real GPU (`slurm-node-gpu-01`, L40S) for all 3 models -- `--limit 8`
(all 8 conditions) `8/8` real forward passes each, `.pt` files populated
correctly including the two conditions with special placeholder
transforms. The other 3 drivers (Exp2 extraction, pilot, formal
behavioral) have not been run against real GPU yet -- that's the next
concrete step (`FINAL_STUDY_PROTOCOL.md` Sec 4.2/13).

Statistical analysis code also now exists: `src/stats_shared.py`
(bootstrap, Holm correction, partition enumeration/ranking, split-half
reliability -- validated against hand-computed examples) and
`slurm/analyze_experiment1_geometry.py` /
`slurm/analyze_experiment2_representation.py` /
`slurm/analyze_experiment2_behavioral.py`, each validated end-to-end
against fabricated synthetic fixtures only (never real activations or
real generations -- those don't exist yet). Every driver and every
analysis script exists in code now; **only the tokenizer audit has
actually run against real model data.** RQ1/RQ2 cannot be answered yet.

**`sbatch/`**: real SLURM submission scripts, one per `slurm/`/`audits/`
driver, conventions matched to `~/new_experiment/slurm/*.sh`
(`partition=cpu`/`gpu`, `account=slurm-students`, the project's shared
venv, `MODEL_IDX`-array submission for per-model jobs). None submitted
yet -- `exp1_smoketest.sh` (Qwen, 4 rows) is the recommended first real
GPU run, to validate the extraction code path before the full array.

## Relationship to `~/new_experiment`

Every file under `data/source/`, `data/splits/`, `templates/imported/`,
and 4 files under `src/` was **copied byte-identical** from that repo
(verified source-vs-destination SHA-256 for every one -- see
`MIGRATION_MANIFEST.json`). `~/new_experiment` itself was never
modified, moved, or deleted -- this is a copy, not a move. Everything
else (the 12-item code-reuse audit in `FINAL_STUDY_PROTOCOL.md` Sec 9)
is either flagged `REWRITE` (deferred to a future round, not copied) or
already copied as noted above -- no scientific results, generations,
judgements, or tensors were migrated. (`MIGRATION_MANIFEST.json` also
records 3 provenance files and 2 reference docs migrated to
`audits/imported/`/`legacy_reference/` at the time -- those destination
paths no longer exist after Round 6's simplification, see
`FINAL_STUDY_PROTOCOL.md`'s Round 6 note.)
