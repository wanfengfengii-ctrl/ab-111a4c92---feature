"""API-level tests for the tolerance sensitivity-spectrum endpoint."""

from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

SPECTRUM_URL = "/api/v1/sensitivity-spectrum"

GOOD_PEAKS = [
    {"mz": "500.000000", "intensity": 100},
    {"mz": "501.003355", "intensity": 90},
]

# The 3-peak transition fixture from the acceptance suite.
TRANSITION_PAYLOAD = {
    "peaks": [
        {"mz": "300.0", "intensity": 100},
        {"mz": "300.6", "intensity": 200},
        {"mz": "301.0", "intensity": 200},
    ],
    "charges": [1],
    "tolerance_range": {"lower": "0", "upper": "0.7"},
}


def post(payload: dict):
    return client.post(SPECTRUM_URL, json=payload)


def test_spectrum_endpoint_full_transition():
    resp = post(TRANSITION_PAYLOAD)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    segments = body["segments"]
    assert [
        (s["lower"], s["upper"], s["upper_inclusive"]) for s in segments
    ] == [
        ("0", "0.003355", False),
        ("0.003355", "0.403355", False),
        ("0.403355", "0.603355", False),
        ("0.603355", "0.7", True),
    ]
    verdicts = [s["adjudication"]["verdict"] for s in segments]
    assert verdicts == ["UNRESOLVED", "UNIQUE", "AMBIGUOUS", "UNIQUE"]

    # The AMBIGUOUS region carries a distinct witness.
    ambiguous = segments[2]["adjudication"]
    assert ambiguous["second_witness"] is not None
    assert ambiguous["second_witness"] != {
        "clusters": ambiguous["clusters"],
        "unexplained_peaks": ambiguous["unexplained_peaks"],
    }

    # Only the final segment includes its upper boundary.
    assert [s["upper_inclusive"] for s in segments] == [False, False, False, True]

    # Exactly the two range endpoints plus the three interior critical points.
    assert body["evaluation_point_count"] == 5
    assert body["critical_point_count"] == 3
    summary = body["input_summary"]
    assert summary["tolerance_range"] == {"lower": "0", "upper": "0.7"}
    assert summary["isotope_spacing"] == "1.003355"


def test_spectrum_segments_are_contiguous_and_sorted():
    body = post(TRANSITION_PAYLOAD).json()
    lowers = [Decimal(s["lower"]) for s in body["segments"]]
    uppers = [Decimal(s["upper"]) for s in body["segments"]]
    assert lowers == sorted(lowers)
    for left_upper, right_lower in zip(uppers, lowers[1:]):
        assert left_upper == right_lower


def test_spectrum_segment_adjudication_is_full_solution():
    body = post(TRANSITION_PAYLOAD).json()
    final = body["segments"][-1]["adjudication"]
    assert final["objectives"] == {
        "explained_intensity": 500,
        "explained_peak_count": 3,
        "cluster_count": 1,
    }
    assert final["clusters"][0]["peak_indices"] == [0, 1, 2]
    assert final["unexplained_peaks"] == []


def test_spectrum_matches_single_deconvolve_endpoint():
    # The adjudication at the range upper bound must equal the deconvolve
    # endpoint's verdict for the same tolerance.
    spectrum = post(
        {
            "peaks": TRANSITION_PAYLOAD["peaks"],
            "charges": [1],
            "tolerance_range": {"lower": "0.5", "upper": "0.5"},
        }
    ).json()
    at_point = spectrum["segments"][0]["adjudication"]

    deconv = client.post(
        "/api/v1/deconvolve",
        json={
            "peaks": TRANSITION_PAYLOAD["peaks"],
            "charges": [1],
            "tolerance": "0.5",
        },
    ).json()
    assert at_point["verdict"] == deconv["verdict"]
    assert at_point["objectives"] == deconv["objectives"]
    assert [
        (c["charge"], c["peak_indices"]) for c in at_point["clusters"]
    ] == [(c["charge"], c["peak_indices"]) for c in deconv["clusters"]]


def test_spectrum_merges_adjacent_identical_regions():
    payload = {
        "peaks": [
            {"mz": str(Decimal("500.000000") + i * Decimal("1.003355")), "intensity": 10}
            for i in range(4)
        ],
        "charges": [1],
        "tolerance_range": {"lower": "0", "upper": "0.001"},
    }
    body = post(payload).json()
    assert len(body["segments"]) == 1
    segment = body["segments"][0]
    assert (segment["lower"], segment["upper"], segment["upper_inclusive"]) == (
        "0",
        "0.001",
        True,
    )


def test_spectrum_is_deterministic():
    bodies = {post(TRANSITION_PAYLOAD).text for _ in range(3)}
    assert len(bodies) == 1


def test_spectrum_non_terminating_boundary_is_rendered_and_splits_verdict():
    # crit = 0.768903... (230671/300000, non-terminating) at z=3.
    payload = {
        "peaks": [
            {"mz": "400", "intensity": 5},
            {"mz": "401.103355", "intensity": 7},
        ],
        "charges": [3],
        "tolerance_range": {"lower": "0", "upper": "0.8"},
    }
    body = post(payload).json()
    assert len(body["segments"]) == 2
    boundary = body["segments"][0]["upper"]
    assert boundary == "0.7689033333333333333333333333333333333333"
    assert body["segments"][1]["lower"] == boundary
    assert [s["adjudication"]["verdict"] for s in body["segments"]] == [
        "UNRESOLVED",
        "UNIQUE",
    ]


def test_spectrum_search_budget_exceeded_returns_503(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "MAX_SEARCH_OPS", 1)
    peaks = [
        {"mz": str(Decimal("500.000000") + Decimal("1.003355") * i), "intensity": 1}
        for i in range(12)
    ]
    resp = post(
        {
            "peaks": peaks,
            "charges": [1],
            "tolerance_range": {"lower": "0", "upper": "0.001"},
        }
    )
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"]["code"] == "SENSITIVITY_SCAN_EXCEEDED"
    assert "segments" not in body
    assert "verdict" not in body


@pytest.mark.parametrize(
    "payload,expected_loc",
    [
        # Peak / charge violations reuse the deconvolve validation rules.
        ({"peaks": GOOD_PEAKS[:1], "charges": [1], "tolerance_range": {"lower": "0", "upper": "1"}}, "peaks"),
        ({"peaks": GOOD_PEAKS, "charges": [], "tolerance_range": {"lower": "0", "upper": "1"}}, "charges"),
        ({"peaks": GOOD_PEAKS, "charges": [1, 1], "tolerance_range": {"lower": "0", "upper": "1"}}, "charges"),
        # Range-specific violations.
        ({"peaks": GOOD_PEAKS, "charges": [1], "tolerance_range": {"lower": "-0.1", "upper": "1"}}, "lower"),
        ({"peaks": GOOD_PEAKS, "charges": [1], "tolerance_range": {"lower": "0.5", "upper": "0.4"}}, "upper"),
        ({"peaks": GOOD_PEAKS, "charges": [1], "tolerance_range": {"lower": "0"}}, "upper"),
        # Unknown fields remain forbidden.
        ({"peaks": GOOD_PEAKS, "charges": [1], "tolerance_range": {"lower": "0", "upper": "1"}, "debug": True}, "debug"),
        ({"peaks": GOOD_PEAKS, "charges": [1], "tolerance_range": {"lower": "0", "upper": "1", "debug": True}}, "debug"),
    ],
)
def test_invalid_spectrum_input_is_422_with_located_field(payload, expected_loc):
    resp = post(payload)
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert "segments" not in body
    fields = body["error"]["fields"]
    assert fields, "expected at least one located field error"
    locs = [f["loc"] for f in fields]
    assert any(
        loc == expected_loc
        or loc.startswith(expected_loc + ".")
        or loc.endswith("." + expected_loc)
        for loc in locs
    ), f"expected loc containing {expected_loc!r} in {locs}"
