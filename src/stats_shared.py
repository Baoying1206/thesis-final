"""Shared statistical primitives for Experiment 1 (Sec 4.3) and
Experiment 2 (Sec 6.1/6.2/7) analysis. Reimplemented fresh for this
study, with the core algorithms (partition enumeration, cosine-based
partition scoring, rank-with-ties, split-half reliability, bootstrap,
Holm correction) matching the proven math in
~/new_experiment/scripts/33_canonical_taxonomy_geometry.py and
57_behavioral_test_bootstrap_analysis.py (read there, not migrated
wholesale -- those files are GPU/old-repo-path coupled and were flagged
REWRITE in this study's original code-reuse audit; only their algorithms
are reused here).

CPU-only. Vector functions accept torch tensors (activation extraction
output is torch); scalar/bootstrap functions are pure Python, no torch
dependency, so they can run wherever generation/judge JSONL lives even
without torch installed.
"""

import random
from itertools import combinations

import torch
import torch.nn.functional as F

from axis_manifest import normalize_text  # noqa: E402


# ── vector geometry (Sec 4.2/4.3, 6/6.1) ────────────────────────────────

def aggregate(vecs, method="mean", trim_frac=0.1):
    """vecs: [n, d] tensor. Returns [d] tensor."""
    if method == "mean":
        return vecs.mean(0)
    if method == "median":
        return vecs.median(0).values
    if method == "trimmed_mean":
        n = vecs.shape[0]
        k = int(n * trim_frac)
        if k == 0:
            return vecs.mean(0)
        sorted_vecs, _ = torch.sort(vecs, dim=0)
        return sorted_vecs[k:n - k].mean(0)
    raise ValueError(f"unknown aggregation method: {method}")


def cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()


def all_2group_partitions(items):
    """items: list of 6 -- all ways to split into two groups of 3.
    A|B and B|A are the same partition (deduplicated). Returns list of
    (group_a_sorted, group_b_sorted) tuples. Used for Experiment 1's
    CO/MG partition-ranking (Sec 4.3): 10 partitions for 6 mechanisms."""
    assert len(items) == 6, f"all_2group_partitions expects exactly 6 items, got {len(items)}"
    items_set = set(items)
    seen, partitions = set(), []
    for combo in combinations(items, 3):
        a = frozenset(combo)
        b = frozenset(items_set - a)
        key = frozenset([a, b])
        if key in seen:
            continue
        seen.add(key)
        partitions.append((sorted(a), sorted(b)))
    return partitions


def all_3group_partitions(items):
    """items: list of 9 -- all ways to split into three UNLABELED groups
    of 3 each. Used for Experiment 2's {CO, MG, Context} partition-
    ranking (Sec 6.1): compares the canonical partition against
    alternatives among the same 9 conditions. 9 items into 3 groups of 3
    (unlabeled groups): 9!/(3!^3 * 3!) = 280 partitions."""
    assert len(items) == 9, f"all_3group_partitions expects exactly 9 items, got {len(items)}"
    items_set = set(items)
    seen, partitions = set(), []
    for combo_a in combinations(items, 3):
        a = frozenset(combo_a)
        remaining = items_set - a
        for combo_b in combinations(sorted(remaining), 3):
            b = frozenset(combo_b)
            c = frozenset(remaining - b)
            key = frozenset([a, b, c])
            if key in seen:
                continue
            seen.add(key)
            partitions.append((sorted(a), sorted(b), sorted(c)))
    return partitions


def compute_2group_partition_stats(vecs, group_a, group_b):
    """vecs: {name: [d] tensor}. Returns S_a, S_b, S_between, delta_a, delta_b, T
    (T = mean within-group similarity minus between-group similarity)."""
    within_a = [cos(vecs[group_a[i]], vecs[group_a[j]]) for i in range(3) for j in range(i + 1, 3)]
    within_b = [cos(vecs[group_b[i]], vecs[group_b[j]]) for i in range(3) for j in range(i + 1, 3)]
    between = [cos(vecs[a], vecs[b]) for a in group_a for b in group_b]
    S_a = sum(within_a) / len(within_a)
    S_b = sum(within_b) / len(within_b)
    S_between = sum(between) / len(between)
    return S_a, S_b, S_between, S_a - S_between, S_b - S_between, (S_a + S_b) / 2 - S_between


def compute_3group_partition_stats(vecs, group_a, group_b, group_c):
    """vecs: {name: [d] tensor}. 3-group generalization for Experiment 2's
    {CO, MG, Context} partition (Sec 6.1). Returns S_a, S_b, S_c,
    S_between (mean of all 3 cross-group pairwise similarities), T (mean
    within-group similarity minus S_between)."""
    def within(group):
        pairs = [cos(vecs[group[i]], vecs[group[j]]) for i in range(3) for j in range(i + 1, 3)]
        return sum(pairs) / len(pairs)

    def between(g1, g2):
        pairs = [cos(vecs[x], vecs[y]) for x in g1 for y in g2]
        return sum(pairs) / len(pairs)

    S_a, S_b, S_c = within(group_a), within(group_b), within(group_c)
    S_ab, S_ac, S_bc = between(group_a, group_b), between(group_a, group_c), between(group_b, group_c)
    S_between = (S_ab + S_ac + S_bc) / 3
    T = (S_a + S_b + S_c) / 3 - S_between
    return S_a, S_b, S_c, S_between, T


def pairwise_cosine_matrix(vecs, names):
    """vecs: {name: [d] tensor}. Vectorized -- one normalize + one matmul,
    not len(names)^2 individual cos() calls. Returns (matrix [n,n] tensor,
    index_of: {name: row/col index}) so bootstrap loops over many
    partitions can look up similarities from a precomputed matrix instead
    of recomputing cosine per partition per replicate (the naive
    per-partition cos() approach is too slow for Experiment 2's 280
    alternative partitions x 2000 bootstrap reps)."""
    stacked = torch.stack([vecs[n] for n in names])
    normed = F.normalize(stacked, dim=-1)
    matrix = normed @ normed.T
    index_of = {n: i for i, n in enumerate(names)}
    return matrix, index_of


def partition_T_from_matrix(matrix, index_of, groups):
    """groups: list of 2 or 3 name-lists (each length 3). Returns T = mean
    within-group similarity minus mean between-group similarity, read
    from a precomputed pairwise_cosine_matrix -- no cos() calls."""
    def within(group):
        idxs = [index_of[n] for n in group]
        pairs = [matrix[idxs[i], idxs[j]].item() for i in range(3) for j in range(i + 1, 3)]
        return sum(pairs) / len(pairs)

    def between(g1, g2):
        idxs1 = [index_of[n] for n in g1]
        idxs2 = [index_of[n] for n in g2]
        pairs = [matrix[i, j].item() for i in idxs1 for j in idxs2]
        return sum(pairs) / len(pairs)

    within_scores = [within(g) for g in groups]
    between_pairs = [between(groups[i], groups[j]) for i in range(len(groups)) for j in range(i + 1, len(groups))]
    S_between = sum(between_pairs) / len(between_pairs)
    T = sum(within_scores) / len(within_scores) - S_between
    return T


def rank_partitions(scores, tol=1e-12):
    """Minimum-rank method with a tie tolerance: ties within `tol` share
    the best (lowest) rank number. 1 = best (highest score). Returns a
    list of ranks, same length/order as `scores`."""
    n = len(scores)
    return [1 + sum(1 for j in range(n) if scores[j] > scores[i] + tol) for i in range(n)]


# ── split-half reliability (Sec 4.3/6.1) ────────────────────────────────

def split_half_reliability(vecs_by_id, ids, n_reps=500, seed=20260828):
    """vecs_by_id: {instruction_id: [d] tensor} for ONE condition's
    already-calibrated (or raw) difference vectors. Returns split-half
    cosine mean/median/95%-CI over n_reps random halvings -- how
    reliably this condition's own mean direction reproduces across
    independent subsamples of instructions."""
    rng = random.Random(seed)
    n = len(ids)
    half = n // 2
    sh_cos = torch.zeros(n_reps)
    for r in range(n_reps):
        shuffled = list(ids)
        rng.shuffle(shuffled)
        vecs_a = torch.stack([vecs_by_id[i] for i in shuffled[:half]])
        vecs_b = torch.stack([vecs_by_id[i] for i in shuffled[half:2 * half]])
        dir_a, dir_b = vecs_a.mean(0), vecs_b.mean(0)
        sh_cos[r] = F.cosine_similarity(dir_a.unsqueeze(0), dir_b.unsqueeze(0), dim=-1)
    return {
        "split_half_cosine_mean": sh_cos.mean().item(),
        "split_half_cosine_median": sh_cos.median().item(),
        "split_half_cosine_ci95": [sh_cos.quantile(0.025).item(), sh_cos.quantile(0.975).item()],
        "n_reps": n_reps, "seed": seed,
    }


# ── instruction-cluster resampling (Sec 8, frozen) ──────────────────────

def load_instruction_clusters(ids, instruction_en_by_id):
    """ids: list of instruction ids. instruction_en_by_id: {id: text}.
    Groups ids sharing the same normalize_text() output into clusters --
    the frozen bootstrap resampling unit (Sec 8): resample CLUSTERS with
    replacement, never raw ids directly, so a known duplicate-text pair
    doesn't get double-counted as two independent bootstrap draws."""
    text_to_ids = {}
    for i in ids:
        norm = normalize_text(instruction_en_by_id[i])
        text_to_ids.setdefault(norm, []).append(i)
    return list(text_to_ids.values())


# ── bootstrap p-values and multiple-comparison correction (Sec 7/8) ────

def bootstrap_two_sided_p(replicate_deltas):
    """Doubled-tail-proportion two-sided bootstrap p-value (frozen, Sec 8):
    p = 2 * min(P(delta<=0), P(delta>=0)), capped at 1.0."""
    n = len(replicate_deltas)
    if n == 0:
        return None
    p_le = sum(1 for d in replicate_deltas if d <= 0) / n
    p_ge = sum(1 for d in replicate_deltas if d >= 0) / n
    return min(1.0, 2 * min(p_le, p_ge))


def bootstrap_one_sided_p(replicate_deltas, direction):
    """One-sided bootstrap p-value -- ONLY valid when the direction was
    frozen BEFORE seeing this data, as part of a pre-registered analysis
    plan (e.g. a single confirmatory hypothesis test on a previously
    sealed dataset). Never choose `direction` post-hoc based on which
    side the point estimate favors -- that silently converts this into
    an anti-conservative, effectively-uncorrected two-sided test.
    direction='greater': H1 is true delta > 0; p = P(delta <= 0).
    direction='less': H1 is true delta < 0; p = P(delta >= 0)."""
    n = len(replicate_deltas)
    if n == 0:
        return None
    if direction == "greater":
        return sum(1 for d in replicate_deltas if d <= 0) / n
    if direction == "less":
        return sum(1 for d in replicate_deltas if d >= 0) / n
    raise ValueError(f"direction must be 'greater' or 'less', got {direction!r}")


def holm_correction(named_pvalues):
    """named_pvalues: list of (name, p) tuples. Returns {name: adjusted_p}."""
    m = len(named_pvalues)
    if m == 0:
        return {}
    order = sorted(range(m), key=lambda i: named_pvalues[i][1])
    adjusted = [None] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        p = named_pvalues[idx][1]
        adj = min((m - rank) * p, 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return {named_pvalues[i][0]: adjusted[i] for i in range(m)}


def bootstrap_did_scalar(metric_P, metric_N, metric_S, metric_C, clusters, n_boot=2000, seed=20260828):
    """Difference-in-differences bootstrap for Study B's behavioral
    estimand (FINAL_STUDY_PROTOCOL.md Sec 5R.4.5/5R.5, Round 17):
    I = (P-N) - (S-C), paired by instruction id, resampling
    instruction-normalized-text clusters (Sec 8's frozen unit).
    metric_P/N/S/C: {instruction_id: 0/1 float}. Every bootstrap
    replicate recomputes I from that replicate's own resampled ids
    (Sec 5R.2's frozen rule) -- there is no fixed component reused
    across replicates."""
    rng = random.Random(seed)
    n_clusters = len(clusters)
    all_ids = list(metric_N.keys())

    def value_for(metric, ids):
        vals = [metric[i] for i in ids if metric.get(i) is not None]
        return sum(vals) / len(vals) if vals else None

    def did(ids):
        p, n, s, c = value_for(metric_P, ids), value_for(metric_N, ids), value_for(metric_S, ids), value_for(metric_C, ids)
        if None in (p, n, s, c):
            return None
        return (p - n) - (s - c)

    point = did(all_ids)
    deltas = []
    for _ in range(n_boot):
        draws = [clusters[rng.randrange(n_clusters)] for _ in range(n_clusters)]
        resampled_ids = [i for cluster in draws for i in cluster]
        d = did(resampled_ids)
        if d is not None:
            deltas.append(d)

    if not deltas:
        return {"point_I": point, "ci_2_5": None, "ci_97_5": None, "p_two_sided": None,
                "n_boot_valid": 0, "n_boot": n_boot, "resample_unit": "instruction_normalized_text_cluster"}

    deltas_sorted = sorted(deltas)
    n = len(deltas_sorted)
    return {
        "point_I": point,
        "ci_2_5": deltas_sorted[int(0.025 * n)], "ci_97_5": deltas_sorted[min(int(0.975 * n), n - 1)],
        "p_two_sided": bootstrap_two_sided_p(deltas),
        "p_one_sided_greater": bootstrap_one_sided_p(deltas, "greater"),
        "p_one_sided_less": bootstrap_one_sided_p(deltas, "less"),
        "n_boot_valid": n, "n_boot": n_boot,
        "resample_unit": "instruction_normalized_text_cluster",
    }


def bootstrap_did_vector(vecs_P, vecs_N, vecs_S, vecs_C, clusters, n_boot=2000, seed=20260828,
                          extra_cos_targets=None):
    """Representational difference-in-differences bootstrap (Sec
    5R.4.5, Round 17): I_vec = (mean(P)-mean(N)) - (mean(S)-mean(C)),
    paired by instruction id. Reports the point estimate vector, a
    bootstrap CI on ||I_vec|| (magnitude -- NOT a formal test that the
    vector differs from a null/permutation baseline; that is a scope
    limitation, not implemented this round), and, if
    `extra_cos_targets` is given ({name: unit_vector}), the bootstrap
    distribution of cos(I_vec_replicate, target) for each target --
    e.g. Study A's frozen p_CO/p_MG directions (Sec 5R.4.7). Every
    bootstrap replicate recomputes I_vec AND every cosine from that
    replicate's own resampled ids (Sec 5R.2's frozen rule) -- the
    target directions passed in `extra_cos_targets` are themselves
    fixed external references (Study A is not re-run), so only the
    LEFT-hand side of each cosine is resampled, which is correct: the
    frozen CO/MG directions are constants here, not something Study B
    estimates."""
    rng = random.Random(seed)
    n_clusters = len(clusters)
    all_ids = list(vecs_N.keys())
    extra_cos_targets = extra_cos_targets or {}

    def mean_vec(vecs, ids):
        present = [vecs[i] for i in ids if i in vecs]
        if not present:
            return None
        return torch.stack(present).mean(0)

    def compute_I(ids):
        p, n, s, c = mean_vec(vecs_P, ids), mean_vec(vecs_N, ids), mean_vec(vecs_S, ids), mean_vec(vecs_C, ids)
        if p is None or n is None or s is None or c is None:
            return None
        return (p - n) - (s - c)

    point_I = compute_I(all_ids)
    norms, cos_dists = [], {name: [] for name in extra_cos_targets}
    for _ in range(n_boot):
        draws = [clusters[rng.randrange(n_clusters)] for _ in range(n_clusters)]
        resampled_ids = [i for cluster in draws for i in cluster]
        I_rep = compute_I(resampled_ids)
        if I_rep is None:
            continue
        norms.append(I_rep.norm().item())
        for name, target in extra_cos_targets.items():
            cos_dists[name].append(cos(I_rep, target))

    result = {
        "point_I_norm": point_I.norm().item() if point_I is not None else None,
        "n_boot_valid": len(norms), "n_boot": n_boot,
        "resample_unit": "instruction_normalized_text_cluster",
    }
    if norms:
        norms_sorted = sorted(norms)
        n = len(norms_sorted)
        result["norm_ci_2_5"] = norms_sorted[int(0.025 * n)]
        result["norm_ci_97_5"] = norms_sorted[min(int(0.975 * n), n - 1)]
    for name, dists in cos_dists.items():
        if not dists:
            continue
        dists_sorted = sorted(dists)
        n = len(dists_sorted)
        result[f"cos_{name}"] = {
            "point": cos(point_I, extra_cos_targets[name]) if point_I is not None else None,
            "ci_2_5": dists_sorted[int(0.025 * n)], "ci_97_5": dists_sorted[min(int(0.975 * n), n - 1)],
        }
    return result


def bootstrap_vector_diff(vecs_a, vecs_b, clusters, n_boot=2000, seed=20260828, extra_cos_targets=None):
    """Simple paired vector-difference bootstrap: d = mean(a) - mean(b),
    reporting ||d||'s point estimate and CI. Used for Study B's `r_f`
    (Sec 5R.4.4 -- the raw, NOT-DiD-adjusted residual, explicitly kept
    only as a secondary/confounded quantity alongside the primary
    `bootstrap_did_vector` estimand), and for the history-augmented
    canonical CO/MG design's per-mechanism representational shift
    `d_m^history = mean(h[m,multi,stage4]) - mean(h[m,single])` (Sec 13
    Round 20), where `extra_cos_targets` (e.g. Experiment 1's frozen
    p_CO/p_MG directions) gives the bootstrap distribution of
    cos(d_replicate, target) alongside the norm, same pattern as
    `bootstrap_did_vector`'s `extra_cos_targets`. Every replicate
    recomputes from its own resampled ids, same rule as the rest of
    this module; `extra_cos_targets` are fixed external references, not
    resampled."""
    rng = random.Random(seed)
    n_clusters = len(clusters)
    all_ids = list(vecs_b.keys())
    extra_cos_targets = extra_cos_targets or {}

    def mean_vec(vecs, ids):
        present = [vecs[i] for i in ids if i in vecs]
        return torch.stack(present).mean(0) if present else None

    def diff(ids):
        a, b = mean_vec(vecs_a, ids), mean_vec(vecs_b, ids)
        return (a - b) if (a is not None and b is not None) else None

    point = diff(all_ids)
    norms, cos_dists = [], {name: [] for name in extra_cos_targets}
    for _ in range(n_boot):
        draws = [clusters[rng.randrange(n_clusters)] for _ in range(n_clusters)]
        resampled_ids = [i for cluster in draws for i in cluster]
        d = diff(resampled_ids)
        if d is None:
            continue
        norms.append(d.norm().item())
        for name, target in extra_cos_targets.items():
            cos_dists[name].append(cos(d, target))

    result = {"point_norm": point.norm().item() if point is not None else None,
              "n_boot_valid": len(norms), "n_boot": n_boot,
              "resample_unit": "instruction_normalized_text_cluster"}
    if norms:
        norms_sorted = sorted(norms)
        n = len(norms_sorted)
        result["norm_ci_2_5"] = norms_sorted[int(0.025 * n)]
        result["norm_ci_97_5"] = norms_sorted[min(int(0.975 * n), n - 1)]
    for name, dists in cos_dists.items():
        if not dists:
            continue
        dists_sorted = sorted(dists)
        n = len(dists_sorted)
        result[f"cos_{name}"] = {
            "point": cos(point, extra_cos_targets[name]) if point is not None else None,
            "ci_2_5": dists_sorted[int(0.025 * n)], "ci_97_5": dists_sorted[min(int(0.975 * n), n - 1)],
        }
    return result


def point_biserial_bootstrap(z_by_id, outcome_by_id, clusters, n_boot=2000, seed=20260828):
    """Sec 5R.4.6's activation-behavior connection: does the projection
    z[i] predict strict_success[i]? z_by_id/outcome_by_id:
    {instruction_id: float / 0-or-1}. Reports the point-biserial
    correlation (Pearson correlation between z and the binary outcome)
    with a bootstrap CI, plus the mean-z difference between the
    success and failure groups (also bootstrapped) -- two of the three
    checks Sec 5R.4.6 asks for; logistic regression is not implemented
    here (would need a numerical solver dependency this module
    otherwise avoids) -- flagged as a scope limitation, not silently
    dropped."""
    rng = random.Random(seed)
    n_clusters = len(clusters)

    def corr_and_group_diff(ids):
        zs = [z_by_id[i] for i in ids if i in z_by_id and i in outcome_by_id]
        ys = [outcome_by_id[i] for i in ids if i in z_by_id and i in outcome_by_id]
        n = len(zs)
        if n < 2:
            return None, None
        mean_z, mean_y = sum(zs) / n, sum(ys) / n
        cov = sum((zs[k] - mean_z) * (ys[k] - mean_y) for k in range(n)) / n
        var_z = sum((v - mean_z) ** 2 for v in zs) / n
        var_y = sum((v - mean_y) ** 2 for v in ys) / n
        corr = cov / ((var_z * var_y) ** 0.5) if var_z > 0 and var_y > 0 else None
        succ = [zs[k] for k in range(n) if ys[k] == 1]
        fail = [zs[k] for k in range(n) if ys[k] == 0]
        group_diff = (sum(succ) / len(succ) - sum(fail) / len(fail)) if succ and fail else None
        return corr, group_diff

    all_ids = list(outcome_by_id.keys())
    point_corr, point_group_diff = corr_and_group_diff(all_ids)

    corrs, group_diffs = [], []
    for _ in range(n_boot):
        draws = [clusters[rng.randrange(n_clusters)] for _ in range(n_clusters)]
        resampled_ids = [i for cluster in draws for i in cluster]
        c, g = corr_and_group_diff(resampled_ids)
        if c is not None:
            corrs.append(c)
        if g is not None:
            group_diffs.append(g)

    def ci(vals):
        if not vals:
            return None, None
        s = sorted(vals)
        n = len(s)
        return s[int(0.025 * n)], s[min(int(0.975 * n), n - 1)]

    corr_lo, corr_hi = ci(corrs)
    diff_lo, diff_hi = ci(group_diffs)
    return {
        "point_biserial_r": point_corr, "r_ci_2_5": corr_lo, "r_ci_97_5": corr_hi,
        "r_p_two_sided": bootstrap_two_sided_p(corrs) if corrs else None,
        "point_group_diff": point_group_diff, "group_diff_ci_2_5": diff_lo, "group_diff_ci_97_5": diff_hi,
        "group_diff_p_two_sided": bootstrap_two_sided_p(group_diffs) if group_diffs else None,
        "n_boot_valid_r": len(corrs), "n_boot_valid_group_diff": len(group_diffs), "n_boot": n_boot,
        "note": "logistic regression not implemented (scope limitation) -- point-biserial correlation and success/failure group z-mean difference only",
    }


def bootstrap_history_augmented_effects(asr_multi_by_mechanism, asr_single_by_mechanism, clusters,
                                         co_mechanisms, mg_mechanisms, n_boot=2000, seed=20260828):
    """RQ2 Round 20 replacement design's primary estimand (history-
    augmented canonical CO/MG, FINAL_STUDY_PROTOCOL.md Sec 13 Round 20):
    for each mechanism m (including 'neutral'), delta_m =
    ASR[m,multi] - ASR[m,single]; E_CO = mean_{m in CO}(delta_m);
    E_MG = mean_{m in MG}(delta_m); E_N = delta_neutral;
    Gamma = E_CO - E_MG; corrected_CO = E_CO - E_N;
    corrected_MG = E_MG - E_N. Algebraically corrected_CO - corrected_MG
    == Gamma (E_N cancels), so Gamma is reported once, not twice.

    asr_multi_by_mechanism / asr_single_by_mechanism:
    {mechanism_name: {instruction_id: 0/1}}, both must include every
    name in co_mechanisms + mg_mechanisms + ['neutral']. clusters: from
    load_instruction_clusters() (Sec 8's frozen resampling unit).

    Every bootstrap replicate recomputes delta_m, E_CO, E_MG, E_N,
    Gamma, corrected_CO, corrected_MG entirely from that replicate's own
    resampled ids -- no fixed component is reused across replicates
    (the same discipline as bootstrap_did_scalar/bootstrap_did_vector).

    The 3 quantities intended as this design's primary confirmatory
    hypotheses are corrected_CO, corrected_MG, and Gamma -- Holm-correct
    those 3 p-values WITHIN each model (never pooled across models),
    same as every other family in this module. E_CO/E_MG/E_N are
    reported as descriptive/secondary, not separately Holm-corrected."""
    rng = random.Random(seed)
    n_clusters = len(clusters)
    all_mechanisms = list(co_mechanisms) + list(mg_mechanisms) + ["neutral"]

    def value_for(metric, ids):
        vals = [metric[i] for i in ids if metric.get(i) is not None]
        return sum(vals) / len(vals) if vals else None

    def deltas_for_ids(ids):
        delta_m = {}
        for m in all_mechanisms:
            multi_v = value_for(asr_multi_by_mechanism[m], ids)
            single_v = value_for(asr_single_by_mechanism[m], ids)
            if multi_v is None or single_v is None:
                return None
            delta_m[m] = multi_v - single_v
        e_co = sum(delta_m[m] for m in co_mechanisms) / len(co_mechanisms)
        e_mg = sum(delta_m[m] for m in mg_mechanisms) / len(mg_mechanisms)
        e_n = delta_m["neutral"]
        return {
            "E_CO": e_co, "E_MG": e_mg, "E_N": e_n,
            "Gamma": e_co - e_mg,
            "corrected_CO": e_co - e_n, "corrected_MG": e_mg - e_n,
        }

    all_ids = list(asr_single_by_mechanism["neutral"].keys())
    point = deltas_for_ids(all_ids)

    keys = ("E_CO", "E_MG", "E_N", "Gamma", "corrected_CO", "corrected_MG")
    replicate_series = {k: [] for k in keys}
    for _ in range(n_boot):
        draws = [clusters[rng.randrange(n_clusters)] for _ in range(n_clusters)]
        resampled_ids = [i for cluster in draws for i in cluster]
        rep = deltas_for_ids(resampled_ids)
        if rep is None:
            continue
        for k in keys:
            replicate_series[k].append(rep[k])

    def summarize(k):
        vals = replicate_series[k]
        if not vals:
            return {"point": point[k] if point else None, "ci_2_5": None, "ci_97_5": None,
                     "p_two_sided": None, "n_boot_valid": 0}
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        return {
            "point": point[k],
            "ci_2_5": vals_sorted[int(0.025 * n)], "ci_97_5": vals_sorted[min(int(0.975 * n), n - 1)],
            "p_two_sided": bootstrap_two_sided_p(vals),
            "n_boot_valid": n,
        }

    result = {k: summarize(k) for k in keys}
    result["n_boot"] = n_boot
    result["resample_unit"] = "instruction_normalized_text_cluster"
    result["co_mechanisms"] = list(co_mechanisms)
    result["mg_mechanisms"] = list(mg_mechanisms)
    result["primary_hypotheses"] = ["corrected_CO", "corrected_MG", "Gamma"]
    return result


def paired_bootstrap_delta(metric_pos, metric_neutral, clusters, n_boot=2000, seed=20260828):
    """metric_pos/metric_neutral: {instruction_id: 0/1 or float or None}.
    clusters: from load_instruction_clusters(). Resamples clusters with
    replacement (Sec 8). Returns point delta, 95% CI, and the two-sided
    bootstrap p-value."""
    rng = random.Random(seed)
    n_clusters = len(clusters)
    all_ids = list(metric_neutral.keys())

    def value_for(metric, ids):
        vals = [metric[i] for i in ids if metric.get(i) is not None]
        return sum(vals) / len(vals) if vals else None

    point_pos = value_for(metric_pos, all_ids)
    point_neutral = value_for(metric_neutral, all_ids)
    point_delta = (point_pos - point_neutral) if (point_pos is not None and point_neutral is not None) else None

    deltas = []
    for _ in range(n_boot):
        draws = [clusters[rng.randrange(n_clusters)] for _ in range(n_clusters)]
        resampled_ids = [i for cluster in draws for i in cluster]
        a = value_for(metric_pos, resampled_ids)
        b = value_for(metric_neutral, resampled_ids)
        if a is not None and b is not None:
            deltas.append(a - b)

    if not deltas:
        return {"point_delta": point_delta, "ci_2_5": None, "ci_97_5": None, "p_two_sided": None,
                "n_boot_valid": 0, "n_boot": n_boot, "resample_unit": "instruction_normalized_text_cluster"}

    deltas_sorted = sorted(deltas)
    n = len(deltas_sorted)
    return {
        "point_delta": point_delta, "point_pos": point_pos, "point_neutral": point_neutral,
        "ci_2_5": deltas_sorted[int(0.025 * n)], "ci_97_5": deltas_sorted[min(int(0.975 * n), n - 1)],
        "p_two_sided": bootstrap_two_sided_p(deltas),
        "n_boot_valid": n, "n_boot": n_boot,
        "resample_unit": "instruction_normalized_text_cluster",
    }
