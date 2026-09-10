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
(`id`/`category`/`instruction_en` only, derived losslessly this round).
`data/source/sampled_prompts.json` still carries the full 9-language
`instructions` field inherited from the source pool; it is kept
unmodified purely as the byte-identical migration-provenance record and
no code here reads it (`FINAL_STUDY_PROTOCOL.md` Sec 2). `READY_FOR_PILOT`
remains a separate, later, human-only decision
pending a real-tokenizer boundary check on the cluster and the pilot
itself (`FINAL_STUDY_PROTOCOL.md` Sec 5.5). See `FINAL_STUDY_PROTOCOL.md`
Sec 13 for remaining open items.

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
boundary check (`FINAL_STUDY_PROTOCOL.md` Sec 11.1). Locally it has only
been validated against Qwen2.5-7B-Instruct (via HF Hub, `TOKEN_AUDIT_PASS`,
10/10 conditions); Llama-3.1-8B-Instruct and gemma-2-9b-it are gated on
HF and untested here -- **running this script on the cluster is the
next concrete step**.

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
