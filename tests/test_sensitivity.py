"""Tests for the tolerance sensitivity spectrum (engine + endpoint)."""

from decimal import Decimal
from fractions import Fraction

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.sensitivity import (
    SensitivityScanFailedError,
    critical_tolerances,
    format_tolerance,
    scan_spectrum,
)
from app.solver import Deconvolver, Peak

client = TestClient(app)

S = Decimal("1.003355")


def make_peaks(spec: list[tuple[str, int]]) -> list[Peak]:
    return [Peak(index=i, mz=Decimal(mz), intensity=it) for i, (mz, it) in enumerate(spec)]


def post_spectrum(payload: dict):
    return client.post("/api/v1/deconvolve/sensitivity", json=payload)


# ---------------------------------------------------------------------- #
# Critical-tolerance derivation (exact, decimal precision)
# ---------------------------------------------------------------------- #


def test_critical_tolerances_are_exact():
    peaks = make_peaks([("400.000000", 5), ("401.003855", 7)])
    # deviation is exactly 0.0005 at z=1.
    assert critical_tolerances(peaks, [1]) == (Fraction(1, 2000),)


def test_critical_tolerances_cover_every_pair_and_charge():
    peaks = make_peaks([("500.000000", 1), ("500.334452", 1)])
    # z=3: deviation is exactly 1e-6 -> 1/3000000, a non-terminating decimal
    # kept as an exact fraction.
    assert critical_tolerances(peaks, [3]) == (Fraction(1, 3_000_000),)
    crits = critical_tolerances(peaks, [1, 3])
    assert crits == tuple(sorted(crits))
    assert Fraction(1, 3_000_000) in crits
    assert Fraction("0.668903") in crits  # |0.334452 - 1.003355| at z=1


def test_critical_tolerances_deduplicate():
    # Both adjacent pairs of an exact 3-chain share critical tolerance 0.
    peaks = make_peaks([("500.000000", 1), ("501.003355", 1), ("502.006710", 1)])
    crits = critical_tolerances(peaks, [1])
    assert crits == (Fraction(0), Fraction("1.003355"))


def test_format_tolerance_exact_strings():
    assert format_tolerance(Fraction(0)) == "0"
    assert format_tolerance(Fraction(5)) == "5"
    assert format_tolerance(Fraction(1, 2)) == "0.5"
    assert format_tolerance(Fraction(7, 8)) == "0.875"
    assert format_tolerance(Fraction(1, 2000)) == "0.0005"
    assert format_tolerance(Fraction(20171, 2000)) == "10.0855"
    assert format_tolerance(Fraction(1, 3)) == "1/3"
    assert format_tolerance(Fraction(1, 3_000_000)) == "1/3000000"


# ---------------------------------------------------------------------- #
# Spectrum scan (engine)
# ---------------------------------------------------------------------- #


def test_scan_splits_exactly_at_critical_tolerance():
    peaks = make_peaks([("400.000000", 5), ("401.003855", 7)])
    spectrum = scan_spectrum(
        peaks=peaks, charges=[1], lower=Decimal("0.0001"), upper=Decimal("0.001")
    )
    assert spectrum.critical_tolerances == (Fraction(1, 2000),)
    assert spectrum.evaluation_count == 2
    assert len(spectrum.segments) == 2
    lo, hi = spectrum.segments
    assert (lo.lower, lo.upper, lo.upper_inclusive) == (
        Fraction(1, 10000),
        Fraction(1, 2000),
        False,
    )
    assert lo.result.verdict == "UNRESOLVED"
    assert (hi.lower, hi.upper, hi.upper_inclusive) == (
        Fraction(1, 2000),
        Fraction(1, 1000),
        True,
    )
    assert hi.result.verdict == "UNIQUE"
    assert hi.result.primary[0].peak_indices == (0, 1)


def test_scan_merges_adjacent_identical_verdicts():
    # Exact 3-chain: pair (0,2) becomes eligible at 1.003355 but {0,2} never
    # beats the full chain, so the verdict is identical on both sides.
    peaks = make_peaks([("500.000000", 10), ("501.003355", 10), ("502.006710", 10)])
    spectrum = scan_spectrum(
        peaks=peaks, charges=[1], lower=Decimal("0"), upper=Decimal("2")
    )
    assert spectrum.critical_tolerances == (Fraction("1.003355"),)
    assert spectrum.evaluation_count == 2
    assert len(spectrum.segments) == 1
    seg = spectrum.segments[0]
    assert (seg.lower, seg.upper, seg.upper_inclusive) == (Fraction(0), Fraction(2), True)
    assert seg.result.verdict == "UNIQUE"
    assert (seg.result.explained_intensity, seg.result.explained_peak_count) == (30, 3)


def test_scan_degenerate_closed_range():
    peaks = make_peaks([("400.000000", 5), ("401.003855", 7)])
    spectrum = scan_spectrum(
        peaks=peaks, charges=[1], lower=Decimal("0.0005"), upper=Decimal("0.0005")
    )
    assert spectrum.evaluation_count == 1
    assert len(spectrum.segments) == 1
    seg = spectrum.segments[0]
    assert seg.lower == seg.upper == Fraction(1, 2000)
    assert seg.upper_inclusive is True
    # The boundary is inclusive: deviation == tolerance still matches.
    assert seg.result.verdict == "UNIQUE"


def test_scan_upper_bound_exactly_on_critical_tolerance():
    # The closed range end coincides with the critical tolerance: the final
    # segment is the inclusive single point carrying the relaxed verdict.
    peaks = make_peaks([("400.000000", 5), ("401.003855", 7)])
    spectrum = scan_spectrum(
        peaks=peaks, charges=[1], lower=Decimal("0.0001"), upper=Decimal("0.0005")
    )
    assert spectrum.evaluation_count == 2
    assert len(spectrum.segments) == 2
    lo, hi = spectrum.segments
    assert (lo.upper, lo.upper_inclusive, lo.result.verdict) == (
        Fraction(1, 2000),
        False,
        "UNRESOLVED",
    )
    assert (hi.lower, hi.upper, hi.upper_inclusive, hi.result.verdict) == (
        Fraction(1, 2000),
        Fraction(1, 2000),
        True,
        "UNIQUE",
    )


def test_scan_evaluates_only_endpoints_and_critical_points():
    # No fixed-step sampling: one evaluation at the lower endpoint plus one
    # per in-range critical tolerance.
    peaks = make_peaks([(str(Decimal("500.000000") + i * S), 5) for i in range(6)])
    spectrum = scan_spectrum(
        peaks=peaks, charges=[1, 2], lower=Decimal("0"), upper=Decimal("2")
    )
    assert spectrum.evaluation_count == 1 + len(spectrum.critical_tolerances)
    boundaries = {Fraction(0), Fraction(2)} | set(spectrum.critical_tolerances)
    for seg in spectrum.segments:
        assert seg.lower in boundaries
        assert seg.upper in boundaries


def test_scan_segments_match_fresh_deconvolutions():
    # Each segment's verdict equals a from-scratch deconvolution at its
    # (inclusive) lower boundary — nothing is carried over between points.
    peaks = make_peaks([("300.0", 100), ("300.6", 200), ("301.0", 200)])
    spectrum = scan_spectrum(
        peaks=peaks, charges=[1], lower=Decimal("0.1"), upper=Decimal("0.7")
    )
    assert [s.result.verdict for s in spectrum.segments] == [
        "UNIQUE",
        "AMBIGUOUS",
        "UNIQUE",
    ]
    for seg in spectrum.segments:
        assert seg.result == Deconvolver(peaks, [1], seg.lower).solve()


def test_scan_is_deterministic_and_stateless():
    peaks = make_peaks([("300.0", 100), ("300.6", 200), ("301.0", 200)])
    kwargs = dict(peaks=peaks, charges=[1], lower=Decimal("0.1"), upper=Decimal("0.6"))
    assert scan_spectrum(**kwargs) == scan_spectrum(**kwargs)


def test_scan_budget_exhaustion_fails_explicitly():
    # 12-peak chain: the exhaustive DP needs more than the remaining budget.
    peaks = make_peaks([(str(Decimal("500.000000") + i * S), 1) for i in range(12)])
    with pytest.raises(SensitivityScanFailedError) as excinfo:
        scan_spectrum(
            peaks=peaks,
            charges=[1],
            lower=Decimal("0.0001"),
            upper=Decimal("0.0001"),
            max_total_ops=140,
        )
    message = str(excinfo.value)
    assert "budget" in message
    assert "no spectrum was produced" in message


def test_scan_budget_counts_cluster_generation():
    peaks = make_peaks([("400.000000", 5), ("401.003855", 7)])
    with pytest.raises(SensitivityScanFailedError):
        scan_spectrum(
            peaks=peaks,
            charges=[1],
            lower=Decimal("0.0001"),
            upper=Decimal("0.001"),
            max_total_ops=1,
        )


def test_scan_rejects_inverted_range():
    peaks = make_peaks([("400.000000", 5), ("401.003855", 7)])
    with pytest.raises(ValueError):
        scan_spectrum(
            peaks=peaks, charges=[1], lower=Decimal("0.01"), upper=Decimal("0.001")
        )


# ---------------------------------------------------------------------- #
# Endpoint
# ---------------------------------------------------------------------- #


def test_endpoint_boundary_split():
    resp = post_spectrum(
        {
            "peaks": [
                {"mz": "400.000000", "intensity": 5},
                {"mz": "401.003855", "intensity": 7},
            ],
            "charges": [1],
            "tolerance_range": {"lower": "0.0001", "upper": "0.001"},
        }
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["critical_tolerances"] == ["0.0005"]
    assert body["evaluation_count"] == 2
    segments = body["segments"]
    assert [s["lower"] for s in segments] == ["0.0001", "0.0005"]
    assert [s["upper"] for s in segments] == ["0.0005", "0.001"]
    assert [s["upper_inclusive"] for s in segments] == [False, True]
    assert [s["verdict"] for s in segments] == ["UNRESOLVED", "UNIQUE"]
    assert segments[0]["clusters"] == []
    assert [p["index"] for p in segments[0]["unexplained_peaks"]] == [0, 1]
    assert segments[1]["clusters"][0]["peak_indices"] == [0, 1]
    assert segments[1]["objectives"] == {
        "explained_intensity": 12,
        "explained_peak_count": 2,
        "cluster_count": 1,
    }
    assert body["input_summary"] == {
        "peak_count": 2,
        "charges": [1],
        "tolerance_range": {"lower": "0.0001", "upper": "0.001"},
        "isotope_spacing": "1.003355",
    }


def test_endpoint_three_regimes_with_witness():
    resp = post_spectrum(
        {
            "peaks": [
                {"mz": "300.0", "intensity": 100},
                {"mz": "300.6", "intensity": 200},
                {"mz": "301.0", "intensity": 200},
            ],
            "charges": [1],
            "tolerance_range": {"lower": "0.1", "upper": "0.7"},
        }
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    segments = body["segments"]
    assert [s["verdict"] for s in segments] == ["UNIQUE", "AMBIGUOUS", "UNIQUE"]
    # The ambiguous segment carries a distinct second witness.
    witness = segments[1]["second_witness"]
    assert witness is not None
    primary_sets = {tuple(c["peak_indices"]) for c in segments[1]["clusters"]}
    witness_sets = {tuple(c["peak_indices"]) for c in witness["clusters"]}
    assert primary_sets != witness_sets
    # Segments are contiguous and ascending; only the last includes its upper.
    for left, right in zip(segments, segments[1:]):
        assert left["upper"] == right["lower"]
        assert left["upper_inclusive"] is False
    assert segments[-1]["upper_inclusive"] is True


def test_endpoint_non_terminating_critical_tolerance():
    resp = post_spectrum(
        {
            "peaks": [
                {"mz": "500.000000", "intensity": 10},
                {"mz": "500.334452", "intensity": 20},
            ],
            "charges": [3],
            "tolerance_range": {"lower": "0.0000001", "upper": "0.000001"},
        }
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["critical_tolerances"] == ["1/3000000"]
    segments = body["segments"]
    assert [s["lower"] for s in segments] == ["0.0000001", "1/3000000"]
    assert [s["upper"] for s in segments] == ["1/3000000", "0.000001"]
    assert [s["verdict"] for s in segments] == ["UNRESOLVED", "UNIQUE"]
    assert segments[1]["clusters"][0]["charge"] == 3


def test_endpoint_merges_identical_verdicts():
    resp = post_spectrum(
        {
            "peaks": [
                {"mz": "500.000000", "intensity": 10},
                {"mz": "501.003355", "intensity": 10},
                {"mz": "502.006710", "intensity": 10},
            ],
            "charges": [1],
            "tolerance_range": {"lower": "0", "upper": "2"},
        }
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["critical_tolerances"] == ["1.003355"]
    assert body["evaluation_count"] == 2
    assert len(body["segments"]) == 1
    seg = body["segments"][0]
    assert (seg["lower"], seg["upper"], seg["upper_inclusive"]) == ("0", "2", True)
    assert seg["verdict"] == "UNIQUE"


def test_endpoint_is_deterministic():
    payload = {
        "peaks": [
            {"mz": "300.0", "intensity": 100},
            {"mz": "300.6", "intensity": 200},
            {"mz": "301.0", "intensity": 200},
        ],
        "charges": [1],
        "tolerance_range": {"lower": "0.1", "upper": "0.6"},
    }
    assert {post_spectrum(payload).text for _ in range(3)} == {post_spectrum(payload).text}


def test_endpoint_consistent_with_deconvolve_endpoint():
    # The segment verdicts agree with the plain deconvolution endpoint when
    # sampled inside each segment.
    peaks = [
        {"mz": "300.0", "intensity": 100},
        {"mz": "300.6", "intensity": 200},
        {"mz": "301.0", "intensity": 200},
    ]
    body = post_spectrum(
        {"peaks": peaks, "charges": [1], "tolerance_range": {"lower": "0.1", "upper": "0.7"}}
    ).json()
    for seg, tol in zip(body["segments"], ["0.1", "0.403355", "0.603355"]):
        direct = client.post(
            "/api/v1/deconvolve",
            json={"peaks": peaks, "charges": [1], "tolerance": tol},
        ).json()
        assert seg["verdict"] == direct["verdict"]
        assert seg["objectives"] == direct["objectives"]


def test_scan_budget_exceeded_returns_503_without_partial_spectrum(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "MAX_SENSITIVITY_OPS", 1)
    resp = post_spectrum(
        {
            "peaks": [
                {"mz": "400.000000", "intensity": 5},
                {"mz": "401.003855", "intensity": 7},
            ],
            "charges": [1],
            "tolerance_range": {"lower": "0.0001", "upper": "0.001"},
        }
    )
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"]["code"] == "SENSITIVITY_SCAN_FAILED"
    assert "budget" in body["error"]["message"]
    assert "segments" not in body


GOOD_PEAKS = [
    {"mz": "500.000000", "intensity": 100},
    {"mz": "501.003355", "intensity": 90},
]
GOOD_RANGE = {"lower": "0.0001", "upper": "0.01"}


@pytest.mark.parametrize(
    "payload,expected_loc",
    [
        (
            {"peaks": GOOD_PEAKS, "charges": [1], "tolerance_range": {"lower": "0.01", "upper": "0.001"}},
            "tolerance_range",
        ),
        (
            {"peaks": GOOD_PEAKS, "charges": [1], "tolerance_range": {"lower": "-0.1", "upper": "0.1"}},
            "tolerance_range.lower",
        ),
        (
            {"peaks": GOOD_PEAKS, "charges": [1], "tolerance_range": {"lower": "0.1"}},
            "tolerance_range.upper",
        ),
        ({"peaks": GOOD_PEAKS, "charges": [1]}, "tolerance_range"),
        (
            {
                "peaks": [
                    {"mz": "501.003355", "intensity": 90},
                    {"mz": "500.000000", "intensity": 100},
                ],
                "charges": [1],
                "tolerance_range": GOOD_RANGE,
            },
            "peaks",
        ),
        (
            {"peaks": GOOD_PEAKS, "charges": [1, 1], "tolerance_range": GOOD_RANGE},
            "charges",
        ),
        (
            {"peaks": GOOD_PEAKS, "charges": [1], "tolerance_range": GOOD_RANGE, "debug": True},
            "debug",
        ),
    ],
)
def test_invalid_scan_request_is_422_located_and_no_partial_spectrum(payload, expected_loc):
    resp = post_spectrum(payload)
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert "segments" not in body
    fields = body["error"]["fields"]
    assert fields, "expected at least one located field error"
    locs = [f["loc"] for f in fields]
    assert any(loc == expected_loc or loc.startswith(expected_loc + ".") for loc in locs), (
        f"expected loc {expected_loc!r} in {locs}"
    )
