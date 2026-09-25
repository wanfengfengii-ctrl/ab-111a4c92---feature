"""Unit tests for the exhaustive deconvolution solver."""

from decimal import Decimal

import pytest

from app.solver import (
    VERDICT_AMBIGUOUS,
    VERDICT_UNIQUE,
    VERDICT_UNRESOLVED,
    Deconvolver,
    Peak,
    SearchSpaceExceededError,
)

S = Decimal("1.003355")


def make_peaks(spec: list[tuple[str, int]]) -> list[Peak]:
    return [Peak(index=i, mz=Decimal(mz), intensity=it) for i, (mz, it) in enumerate(spec)]


def solve(spec, charges, tol):
    return Deconvolver(make_peaks(spec), charges, Decimal(tol)).solve()


def test_simple_chain_is_unique():
    result = solve(
        [
            ("400.000000", 500),
            ("500.000000", 1000),
            ("501.003355", 800),
            ("502.006710", 600),
            ("503.010065", 400),
            ("700.000000", 50),
        ],
        [1],
        "0.0005",
    )
    assert result.verdict == VERDICT_UNIQUE
    assert result.explained_intensity == 2800
    assert result.explained_peak_count == 4
    assert result.cluster_count == 1
    assert result.secondary is None
    assert result.primary[0].peak_indices == (1, 2, 3, 4)
    assert result.primary[0].charge == 1


def test_intensity_is_optimised_before_peak_count():
    # X = {1,3} at z=1 explains 200 over 2 peaks; Y = {0,1,2} at z=3 explains
    # 102 over 3 peaks.  They conflict on peak 1, and Y's leftover peaks
    # cannot recombine, so the intensity-first lexicographic order must
    # prefer X even though Y explains more peaks.
    result = solve(
        [
            ("500.6689033", 1),
            ("501.003355", 100),
            ("501.3378067", 1),
            ("502.006710", 100),
        ],
        [1, 2, 3],
        "0.0001",
    )
    assert result.verdict == VERDICT_UNIQUE
    assert (result.explained_intensity, result.explained_peak_count, result.cluster_count) == (200, 2, 1)
    assert result.primary[0].peak_indices == (1, 3)
    assert result.primary[0].charge == 1


def test_peak_count_is_optimised_before_cluster_count():
    # Two disjoint pairs (4 peaks, 2 clusters) must beat one pair (2 peaks).
    result = solve(
        [
            ("500.000000", 10),
            ("501.003355", 10),
            ("600.000000", 10),
            ("601.003355", 10),
        ],
        [1],
        "0.0001",
    )
    assert result.verdict == VERDICT_UNIQUE
    assert result.explained_peak_count == 4
    assert result.cluster_count == 2


def test_fewest_clusters_breaks_ties():
    # A 4-chain as one cluster beats two pairs covering the same 4 peaks.
    result = solve(
        [
            ("500.000000", 10),
            ("501.003355", 10),
            ("502.006710", 10),
            ("503.010065", 10),
        ],
        [1],
        "0.0001",
    )
    assert result.verdict == VERDICT_UNIQUE
    assert result.cluster_count == 1
    assert result.primary[0].peak_indices == (0, 1, 2, 3)


def test_ambiguous_tie_yields_second_witness():
    result = solve(
        [("300.0", 100), ("300.6", 200), ("301.0", 200)],
        [1],
        "0.5",
    )
    assert result.verdict == VERDICT_AMBIGUOUS
    assert result.secondary is not None
    primary_sets = {c.peak_indices for c in result.primary}
    secondary_sets = {c.peak_indices for c in result.secondary}
    assert primary_sets != secondary_sets
    assert primary_sets | secondary_sets == {(0, 1), (0, 2)}
    for witness in (result.primary, result.secondary):
        assert sum(c.explained_intensity for c in witness) == result.explained_intensity
        assert len(witness) == result.cluster_count


def test_unresolved_when_no_legal_cluster():
    result = solve(
        [("100.0", 10), ("100.3", 20), ("100.6", 30)],
        [1],
        "0.001",
    )
    assert result.verdict == VERDICT_UNRESOLVED
    assert result.primary == ()
    assert result.secondary is None
    assert (result.explained_intensity, result.explained_peak_count, result.cluster_count) == (0, 0, 0)


def test_tolerance_boundary_is_inclusive_and_exact():
    spec = [("400.000000", 5), ("401.003855", 7)]  # deviation exactly 0.0005
    assert solve(spec, [1], "0.0005").verdict == VERDICT_UNIQUE
    assert solve(spec, [1], "0.0004999").verdict == VERDICT_UNRESOLVED


def test_charge_two_spacing():
    spec = [("700.0000000", 10), ("700.5016775", 20)]
    result = solve(spec, [2], "0.0000001")
    assert result.verdict == VERDICT_UNIQUE
    assert result.primary[0].charge == 2
    assert solve(spec, [1], "0.0000001").verdict == VERDICT_UNRESOLVED


def test_cluster_size_capped_at_six():
    spec = [(str(Decimal("900.000000") + i * S), 1) for i in range(7)]
    result = solve(spec, [1], "0.0001")
    assert result.explained_peak_count == 7
    assert result.cluster_count == 2
    assert all(c.size <= 6 for c in result.primary)
    # Several equal splits of a 7-chain into two clusters exist.
    assert result.verdict == VERDICT_AMBIGUOUS
    assert result.secondary is not None


def test_clusters_are_mutually_disjoint():
    spec = [(str(Decimal("500.000000") + i * S), 5) for i in range(6)]
    result = solve(spec, [1, 2, 3], "0.001")
    seen: list[int] = []
    for cluster in result.primary:
        seen.extend(cluster.peak_indices)
    assert len(seen) == len(set(seen))


def test_solver_is_deterministic():
    spec = [("300.0", 100), ("300.6", 200), ("301.0", 200)]
    first = solve(spec, [1], "0.5")
    second = solve(spec, [1], "0.5")
    assert first == second


def test_same_peaks_matching_two_charges_are_ambiguous():
    # A pair whose spacing fits both z=1 and (via a wide tolerance) nothing
    # else: same explained intensity/count -> charge assignment is ambiguous.
    spec = [("500.000000", 10), ("501.003355", 10)]
    result = solve(spec, [1], "0.0001")
    assert result.verdict == VERDICT_UNIQUE
    # With charges {1, 2} and a tolerance that lets the pair match both
    # spacings, two equal-quality assignments exist.
    wide = solve(spec, [1, 2], "0.6")
    assert wide.verdict == VERDICT_AMBIGUOUS
    charges = {c.charge for c in wide.primary} | {c.charge for c in wide.secondary}
    assert charges == {1, 2}


def test_search_budget_guard():
    # A tiny work budget must abort with an explicit error, never a wrong
    # or partial verdict.
    spec = [(str(Decimal("500.000000") + i * S), 1) for i in range(12)]
    with pytest.raises(SearchSpaceExceededError):
        Deconvolver(make_peaks(spec), [1], Decimal("0.0001"), max_search_ops=1).solve()
