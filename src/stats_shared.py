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
