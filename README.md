# thesis_final

Independent, final-stage thesis experiment. Separate Git history from
`~/new_experiment` (the prior, still-frozen exploratory work) -- this
repo does not share commits, objects, or working files with it.
`~/new_experiment` is treated as strictly **read-only** reference
material from here on: no code, data, commits, or output in that repo
are modified, moved, or deleted by anything in this directory.

## Current final results (read this first)

Both research questions now have complete, real, 3-model GPU data.

**RQ1**: Wei et al.'s CO (competing-objectives) / MG (mismatched-
generalisation) jailbreak taxonomy corresponds to a consistent internal
activation geometry across Qwen2.5-7B-Instruct, Meta-Llama-3.1-8B-
Instruct, and gemma-2-9b-it. In all 3 models the canonical CO/MG
partition is the best- or second-best-ranked way (out of 10 possible
2-group splits of the 6 mechanisms) to divide the mechanisms by
within-vs-between-group cosine similarity, and every mechanism's
direction is highly split-half reliable (~0.99+). See
`FINAL_STUDY_PROTOCOL.md` Sec 4, `slurm/experiment1_analysis/`, figures
in `slurm/experiment1_analysis/figures/`.

**RQ2 (current mainline)**: history-augmented canonical CO/MG design
(FINAL_STUDY_PROTOCOL.md Sec 13 Round 20 + follow-up) -- 7 mechanism
groups (3 CO + 3 MG + neutral) x 3 delivery forms (single-turn,
5-turn-neutral-scaffold-then-payload, 5-turn-progressive-scaffold-
then-payload) = 21 conditions, real GPU behavioral + representation
data on all 3 models. Key findings:
- **Neutral (topic-unrelated) multi-turn history has no measurable
  effect** on canonical CO/MG jailbreak success, in any of the 3
  models -- a control condition not tested in prior multi-turn
  jailbreak literature (Crescendo, Deceptive Delight, etc. all shape
  content, not pure structure).
- **Progressive (escalating "security research" framing) multi-turn
  history does move behavior, but the way it moves is model-specific**:
  Qwen shows a broad, mechanism-independent compliance shift; Llama
  shows an effect concentrated in one MG mechanism
  (`encoding_obfuscation`); Gemma shows both CO and MG moving (with a
  caveat: Gemma's neutral-mechanism baseline is a true floor, 0/216,
  which degenerates the correction that isolates mechanism-specific
  effects from generic uplift in the other two models). No universal
  cross-model behavioral rule.
- **The one finding that IS cross-model consistent**: in all 3 models
  and both scaffold kinds, the 3 CO mechanisms' history-induced
  representational shift vectors cohere strongly with each other
  (mean cosine ~0.6-0.86); the 3 MG mechanisms' shift vectors do not
  (mean cosine close to or below the CO/MG between-group baseline).
  This extends RQ1's static geometric finding into a claim about
  shared dynamic/causal structure: CO mechanisms appear to share a
  common internal "compliance framing" lever that responds coherently
  to any delivery change; MG mechanisms behave as independent,
  idiosyncratic exploits with no such shared lever.
- **Activation shift predicts individual-instance success for exactly
  2 model-mechanism pairs, not generally**: Llama's `payload_splitting`
  (both scaffold kinds, r=+0.46/+0.39) and Gemma's
  `refusal_suppression` (progressive only, r=-0.38). A first pass found
  14 apparently-significant correlations, but investigation traced 11
  of them to a real statistical artifact -- extreme class imbalance
  (1-2 successes out of 72) degenerates the cluster bootstrap into an
  outlier-detection test regardless of the reported r's magnitude. Now
  fixed in `src/stats_shared.py` (`point_biserial_bootstrap`'s
  `reliable` flag, excluded from the Holm family when
  min(successes, failures) < 5) and confirmed on a synthetic
  single-outlier reproduction. Do not trust any r from this analysis
  without checking `z_vs_strict_success.reliable`.

See `FINAL_STUDY_PROTOCOL.md` Sec 13 Round 20 for the full history and
rationale, `slurm/history_augmented_co_mg_analysis/` for the raw
per-model JSON results, and
`slurm/history_augmented_co_mg_analysis/figures/` for the publication
figures (fig4: behavioral corrected effects; fig5: representation
cohesion; fig6: raw per-condition ASR; fig7: activation-behavior
connection forest plot, gray = excluded as unreliable).

Superseded prior RQ2 designs (Round 1-14's static contextual framing,
Round 16-19's persona/authority/fictional "Study B") are retained in
`FINAL_STUDY_PROTOCOL.md` and under `slurm/study_b_*`/
`templates/study_b_*`/`slurm/experiment2_*` for provenance and
reproducibility only -- neither feeds the current RQ2 results above.
See the "Historical development log" section below for what each
superseded design was and why it was replaced; nothing described there
was deleted or moved, it is all still present in this repo exactly
where it was written.

## What this is

Two research questions (see `FINAL_STUDY_PROTOCOL.md` for the full
design/statistics):

- **RQ1**: Does Wei et al.'s competing-objectives (CO) / mismatched-
  generalisation (MG) taxonomy correspond to a consistent internal
  activation geometry across three models?
- **RQ2**: Does history-augmented canonical CO/MG multi-turn delivery
  -- the same, unmodified canonical CO/MG mechanism text from RQ1,
  delivered after a frozen scaffold instead of as a single turn --
  produce a behavioral and/or representational effect beyond the same
  mechanism's single-turn delivery, net of a matched neutral-scaffold
  baseline, and does that effect (if any) differ between CO and MG?

## Historical development log (superseded designs, kept for provenance)

Nothing in this section was deleted, archived, or moved -- every file
it references is still in this repo's working tree exactly where it
always was; git history is the only "archive" this project uses. This
section is a compressed index into `FINAL_STUDY_PROTOCOL.md`'s much
more detailed round-by-round record (Sec 13), not a replacement for it.

**RQ1 build-out (Rounds 4-13)**: the 8-condition single-turn design
was frozen (Round 4), the tokenizer-boundary audit passed on real
tokenizers for all 3 models (Round 8), the pilot behavioral run and
extraction drivers were written and validated (Rounds 9-10), and
`extract_experiment1_activations.py` was run for real on all 3 models
(Round 12) -- this is the data behind RQ1's result above. Along the
way, `audits/`, `config/OUTPUT_SCHEMA.md`, and 2 reference-only docs
were deleted (Round 6) after what they verified was written into
`FINAL_STUDY_PROTOCOL.md` Sec 9/11; `audits/audit_real_tokenizer_boundary.py`
was restored afterward (Round 7) because it's needed to actually rerun
the boundary check and is kept permanently.

**RQ2, first design (Rounds 1-14, `slurm/experiment2_*`,
`templates/final_10_condition_v1.json`)**: a static 10-condition design
(3 CO + 3 MG + 3 "Context" framings -- persona/authority/fictional --
+ 1 shared neutral). Run for real on all 3 models; both the original
and a strengthened v2 template of the Context conditions showed no
significant behavioral effect. This null result is what motivated the
Round 16 redesign below -- the static design's real data is untouched
and still readable under `slurm/experiment2_analysis/`.

**RQ2, second design ("Study B", Rounds 16-19, `slurm/study_b_*`,
`templates/study_b_progressive_multiturn_v1.json`)**: reframed the
Context conditions as a 2x2 progressive/compressed x positive/neutral
difference-in-differences design across the same persona/authority/
fictional families (not the canonical CO/MG mechanisms). Real
discovery-stage data was collected on all 3 models; only Qwen+fictional
was significant after Holm correction. A single pre-registered
confirmatory test on previously-sealed `test_ids` was run for
Qwen+fictional (Round 19) -- also real, also complete. **Superseded by
the current RQ2 design (Round 20)** because it never combined this
progressive/compressed structure axis with the canonical CO/MG
mechanisms RQ1 characterizes -- Study B's own "positive/neutral" payload
was persona/authority/fictional-specific framing, not Wei et al.'s
6 mechanisms. `FINAL_STUDY_PROTOCOL.md` Sec 5R carries an explicit
`SUPERSEDED_BY_HISTORY_AUGMENTED_CANONICAL_CO_MG_RQ2` status marker;
its data, code, and results are all still present and unmodified.

**RQ2, current design (Round 20 + follow-up, "history-augmented
canonical CO/MG")**: see "Current final results" above. Built as a true
2-factor design (7 mechanism groups, including all 6 of RQ1's canonical
CO/MG mechanisms x delivery form), closing the gap Study B left open.
Initially a 3-turn neutral-only scaffold; lengthened to 5 turns and a
second "progressive" scaffold kind was added (7 mechanisms x 3 forms =
21 conditions) after (a) a statistical-power concern about a short,
topic-unrelated scaffold, and (b) a literature check showing published
high-ASR multi-turn jailbreaks all rely on content shaping, not
structure alone -- a pure "unrelated history" vs. "escalating history"
control was not found in prior work. Validated end-to-end (dry-run,
smoketest, pilot) before any real confirmatory extraction; three real
bugs were found and fixed against real cluster runs, not just code
review: an OOM (Gemma's float32 logit-softcapping scaling with batch
size and sequence length), a memory-management bug in the analysis
script (redundant reloading of Experiment 1's reference directions),
and a statistical artifact in the activation-behavior connection
(extreme class imbalance degenerating the bootstrap into an
outlier-detection test regardless of the reported correlation's
magnitude -- see "Current final results" above).

## `sbatch/`

Real SLURM submission scripts, one per `slurm/`/`audits/` driver,
conventions matched to `~/new_experiment/slurm/*.sh` (`partition=cpu`/
`gpu`, `account=slurm-students`, the project's shared venv,
`MODEL_IDX`-array submission for per-model jobs).

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
paths no longer exist after Round 6's simplification, see the
historical log above.)
