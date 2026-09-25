"""Unit tests for the exact tolerance-sensitivity spectrum."""

from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from app.sensitivity import (
    BOUNDARY_DISPLAY_DIGITS,
    SensitivityScanner,
    SensitivityScanError,
    critical_tolerances,
    format_boundary,
)
from app.solver import Deconvolver, Peak

S = Decimal("1.003355")


def make_peaks(spec: list[tuple[str, int]]) -> list[Peak]:
    return [Peak(index=i, mz=Decimal(mz), intensity=it) for i, (mz, it) in enumerate(spec)]


# The 3-peak transition fixture: d(0,1)=0.6, d(0,2)=1.0, d(1,2)=0.4.
TRANSITION_SPEC = [("300.0", 100), ("300.6", 200), ("301.0", 200)]


def scan(spec, charges, lower, upper, budget=20_000_000):
    return SensitivityScanner(
        make_peaks(spec), charges, Decimal(lower), Decimal(upper), budget
    ).run()


def test_critical_tolerances_are_exact_rationals():
    critical = critical_tolerances(make_peaks(TRANSITION_SPEC), [1])
    assert critical == (
        Fraction(3355, 10**6),
        Fraction(403355, 10**6),
        Fraction(603355, 10**6),
    )
    assert all(isinstance(value, Fraction) for value in critical)


def test_critical_tolerances_derive_from_pair_differences_and_charges():
    peaks = make_peaks([("700.0000000", 10), ("700.5016775", 20)])
    # z=2: spacing is exactly 1.003355/2 -> critical tolerance 0.
    # z=1: |0.5016775 - 1.003355| = 0.5016775.
    assert critical_tolerances(peaks, [2]) == (Fraction(0),)
    assert critical_tolerances(peaks, [1]) == (Fraction(5016775, 10**7),)
    assert critical_tolerances(peaks, [1, 2]) == (
        Fraction(0),
        Fraction(5016775, 10**7),
    )


def test_non_terminating_critical_tolerance_stays_exact():
    # Pair spacing 1.103355 at z=3: crit = |3*1.103355 - 1.003355| / 3
    # = 2.306710/3, a non-terminating decimal.
    peaks = make_peaks([("400", 5), ("401.103355", 7)])
    (critical,) = critical_tolerances(peaks, [3])
    assert critical == Fraction(2306710, 3000000)
    assert critical == Fraction(230671, 300000)


def test_scanner_evaluates_only_endpoints_and_critical_points():
    spectrum = scan(TRANSITION_SPEC, [1], "0", "0.7")
    assert spectrum.evaluation_points == (
        Fraction(0),
        Fraction(3355, 10**6),
        Fraction(403355, 10**6),
        Fraction(603355, 10**6),
        Fraction(7, 10),
    )
    assert spectrum.critical_points == (
        Fraction(3355, 10**6),
        Fraction(403355, 10**6),
        Fraction(603355, 10**6),
    )


def test_no_fixed_step_sampling_when_no_critical_points_inside():
    # A range containing no critical tolerance (the only one for this pair,
    # |0.3 - 1.003355| = 0.703355, lies far above it) needs exactly two
    # evaluations — the endpoints — and yields a single segment.
    spectrum = scan([("100.0", 10), ("100.3", 20)], [1], "0", "0.001")
    assert spectrum.evaluation_points == (Fraction(0), Fraction(1, 1000))
    assert len(spectrum.segments) == 1
    segment = spectrum.segments[0]
    assert segment.adjudication.verdict == "UNRESOLVED"
    assert segment.upper_inclusive is True


def test_three_peak_transition_spectrum():
    spectrum = scan(TRANSITION_SPEC, [1], "0", "0.7")
    assert len(spectrum.segments) == 4

    seg = spectrum.segments[0]
    assert (seg.lower, seg.upper, seg.upper_inclusive) == (
        Fraction(0),
        Fraction(3355, 10**6),
        False,
    )
    assert seg.adjudication.verdict == "UNRESOLVED"

    seg = spectrum.segments[1]
    assert (seg.lower, seg.upper, seg.upper_inclusive) == (
        Fraction(3355, 10**6),
        Fraction(403355, 10**6),
        False,
    )
    assert seg.adjudication.verdict == "UNIQUE"
    assert [c.peak_indices for c in seg.adjudication.primary] == [(0, 2)]

    seg = spectrum.segments[2]
    assert (seg.lower, seg.upper, seg.upper_inclusive) == (
        Fraction(403355, 10**6),
        Fraction(603355, 10**6),
        False,
    )
    assert seg.adjudication.verdict == "AMBIGUOUS"
    assert seg.adjudication.secondary is not None

    seg = spectrum.segments[3]
    assert (seg.lower, seg.upper, seg.upper_inclusive) == (
        Fraction(603355, 10**6),
        Fraction(7, 10),
        True,
    )
    assert seg.adjudication.verdict == "UNIQUE"
    assert [c.peak_indices for c in seg.adjudication.primary] == [(0, 1, 2)]
    assert seg.adjudication.explained_intensity == 500


def test_segments_are_contiguous_and_only_last_is_upper_inclusive():
    spectrum = scan(TRANSITION_SPEC, [1], "0", "0.7")
    segments = spectrum.segments
    assert segments[0].lower == Fraction(0)
    for left, right in zip(segments, segments[1:]):
        assert left.upper == right.lower
        assert left.upper_inclusive is False
    assert segments[-1].upper == Fraction(7, 10)
    assert segments[-1].upper_inclusive is True


def test_adjacent_identical_adjudications_are_merged():
    # Equal-spacing 4-chain: the only critical tolerance inside (0, 0.001]
    # is 0.0005 (pairs two apart), but the adjudication is identical on both
    # sides of it, so the spectrum must be one merged segment.
    spec = [(str(Decimal("500.000000") + i * S), 10) for i in range(4)]
    spectrum = scan(spec, [1], "0", "0.001")
    assert len(spectrum.segments) == 1
    segment = spectrum.segments[0]
    assert (segment.lower, segment.upper) == (Fraction(0), Fraction(1, 1000))
    assert segment.upper_inclusive is True
    assert segment.adjudication.verdict == "UNIQUE"
    assert [c.peak_indices for c in segment.adjudication.primary] == [(0, 1, 2, 3)]


def test_degenerate_range_yields_single_point_segment():
    spectrum = scan(TRANSITION_SPEC, [1], "0.5", "0.5")
    assert len(spectrum.segments) == 1
    segment = spectrum.segments[0]
    assert segment.lower == segment.upper == Fraction(1, 2)
    assert segment.upper_inclusive is True
    assert segment.adjudication.verdict == "AMBIGUOUS"


def test_critical_tolerance_exactly_at_upper_bound():
    spectrum = scan(TRANSITION_SPEC, [1], "0", "0.003355")
    assert [ (s.lower, s.upper, s.upper_inclusive) for s in spectrum.segments ] == [
        (Fraction(0), Fraction(3355, 10**6), False),
        (Fraction(3355, 10**6), Fraction(3355, 10**6), True),
    ]
    assert spectrum.segments[0].adjudication.verdict == "UNRESOLVED"
    assert spectrum.segments[1].adjudication.verdict == "UNIQUE"


def test_critical_tolerances_at_range_endpoints_are_not_interior():
    spectrum = scan(TRANSITION_SPEC, [1], "0.003355", "0.603355")
    assert spectrum.critical_points == (Fraction(403355, 10**6),)
    assert spectrum.evaluation_points == (
        Fraction(3355, 10**6),
        Fraction(403355, 10**6),
        Fraction(603355, 10**6),
    )
    assert [s.adjudication.verdict for s in spectrum.segments] == [
        "UNIQUE",
        "AMBIGUOUS",
        "UNIQUE",
    ]


def test_segment_adjudication_matches_independent_midpoint_solve():
    # Cross-check: an independent deconvolution at an interior point of every
    # non-degenerate segment must reproduce the segment's adjudication.
    spectrum = scan(TRANSITION_SPEC, [1], "0", "0.7")
    peaks = make_peaks(TRANSITION_SPEC)
    for segment in spectrum.segments:
        if segment.lower == segment.upper:
            continue
        midpoint = (segment.lower + segment.upper) / 2
        independent = Deconvolver(peaks, [1], midpoint, 20_000_000).solve()
        assert independent == segment.adjudication


def test_scan_budget_exhaustion_fails_explicitly_without_partial_spectrum():
    spec = [(str(Decimal("500.000000") + i * S), 1) for i in range(12)]
    with pytest.raises(SensitivityScanError, match="no partial spectrum"):
        scan(spec, [1], "0", "0.001", budget=1)


def test_format_boundary_renders_terminating_decimals_exactly():
    assert format_boundary(Fraction(0)) == "0"
    assert format_boundary(Fraction(3)) == "3"
    assert format_boundary(Fraction(3355, 10**6)) == "0.003355"
    assert format_boundary(Fraction(1, 200)) == "0.005"
    assert format_boundary(Fraction(25, 2)) == "12.5"
    # 1/2**40 terminates but needs 40 fractional digits: no rounding allowed.
    assert format_boundary(Fraction(1, 2**40)) == "0.0000000000009094947017729282379150390625"


def test_format_boundary_rounds_non_terminating_to_display_digits():
    rendered = format_boundary(Fraction(1, 3))
    assert rendered == "0." + "3" * BOUNDARY_DISPLAY_DIGITS
    # The rounded display must sit within one last-place unit of the exact
    # value (reference computed at higher precision than the display).
    with localcontext() as context:
        context.prec = BOUNDARY_DISPLAY_DIGITS + 10
        reference = Decimal(1) / Decimal(3)
    assert abs(Decimal(rendered) - reference) < Decimal("1e-39")


def test_format_boundary_of_critical_point_is_used_for_segments():
    peaks = make_peaks([("400", 5), ("401.103355", 7)])
    spectrum = SensitivityScanner(
        peaks, [3], Decimal("0"), Decimal("0.8"), 20_000_000
    ).run()
    boundary_strings = [
        (format_boundary(s.lower), format_boundary(s.upper)) for s in spectrum.segments
    ]
    assert boundary_strings[0][1] == "0.7689033333333333333333333333333333333333"
    assert boundary_strings[1][0] == boundary_strings[0][1]
    # The verdict flips exactly at the non-terminating critical tolerance.
    assert spectrum.segments[0].adjudication.verdict == "UNRESOLVED"
    assert spectrum.segments[1].adjudication.verdict == "UNIQUE"
