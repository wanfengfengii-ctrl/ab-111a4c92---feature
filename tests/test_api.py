"""API-level tests via FastAPI's TestClient."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

GOOD_PEAKS = [
    {"mz": "500.000000", "intensity": 100},
    {"mz": "501.003355", "intensity": 90},
]


def post(payload: dict):
    return client.post("/api/v1/deconvolve", json=payload)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_unique_verdict_end_to_end():
    resp = post(
        {
            "peaks": [
                {"mz": "400.000000", "intensity": 500},
                {"mz": "500.000000", "intensity": 1000},
                {"mz": "501.003355", "intensity": 800},
                {"mz": "502.006710", "intensity": 600},
                {"mz": "503.010065", "intensity": 400},
                {"mz": "700.000000", "intensity": 50},
            ],
            "charges": [1],
            "tolerance": "0.0005",
        }
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "UNIQUE"
    assert body["objectives"] == {
        "explained_intensity": 2800,
        "explained_peak_count": 4,
        "cluster_count": 1,
    }
    assert body["clusters"][0]["peak_indices"] == [1, 2, 3, 4]
    assert body["clusters"][0]["charge"] == 1
    assert [p["index"] for p in body["unexplained_peaks"]] == [0, 5]
    assert body["second_witness"] is None
    assert body["input_summary"]["isotope_spacing"] == "1.003355"


def test_ambiguous_returns_distinct_second_witness():
    resp = post(
        {
            "peaks": [
                {"mz": "300.0", "intensity": 100},
                {"mz": "300.6", "intensity": 200},
                {"mz": "301.0", "intensity": 200},
            ],
            "charges": [1],
            "tolerance": "0.5",
        }
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "AMBIGUOUS"
    witness = body["second_witness"]
    assert witness is not None
    primary_sets = {tuple(c["peak_indices"]) for c in body["clusters"]}
    witness_sets = {tuple(c["peak_indices"]) for c in witness["clusters"]}
    assert primary_sets != witness_sets
    assert primary_sets | witness_sets == {(0, 1), (0, 2)}


def test_unresolved():
    resp = post(
        {
            "peaks": [
                {"mz": "100.0", "intensity": 10},
                {"mz": "100.3", "intensity": 20},
            ],
            "charges": [1],
            "tolerance": "0.001",
        }
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "UNRESOLVED"
    assert body["clusters"] == []
    assert [p["index"] for p in body["unexplained_peaks"]] == [0, 1]


def test_response_is_deterministic():
    payload = {"peaks": GOOD_PEAKS, "charges": [1], "tolerance": "0.001"}
    assert {post(payload).text for _ in range(3)} == {post(payload).text}


def test_search_budget_exceeded_returns_503(monkeypatch):
    from decimal import Decimal

    import app.main as main_module

    monkeypatch.setattr(main_module, "MAX_SEARCH_OPS", 1)
    peaks = [
        {"mz": str(Decimal("500.000000") + Decimal("1.003355") * i), "intensity": 1}
        for i in range(12)
    ]
    resp = post({"peaks": peaks, "charges": [1], "tolerance": "0.0001"})
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"]["code"] == "SEARCH_SPACE_EXCEEDED"
    assert "verdict" not in body


@pytest.mark.parametrize(
    "payload,expected_loc",
    [
        ({"peaks": GOOD_PEAKS[:1], "charges": [1], "tolerance": "0.001"}, "peaks"),
        (
            {
                "peaks": [{"mz": str(100 + i), "intensity": 1} for i in range(37)],
                "charges": [1],
                "tolerance": "0.001",
            },
            "peaks",
        ),
        (
            {
                "peaks": [
                    {"mz": "501.003355", "intensity": 90},
                    {"mz": "500.000000", "intensity": 100},
                ],
                "charges": [1],
                "tolerance": "0.001",
            },
            "peaks",
        ),
        (
            {"peaks": [GOOD_PEAKS[0], {"mz": "501.003355", "intensity": 0}], "charges": [1], "tolerance": "0.001"},
            "peaks.1.intensity",
        ),
        (
            {"peaks": [GOOD_PEAKS[0], {"mz": "501.003355", "intensity": 1.5}], "charges": [1], "tolerance": "0.001"},
            "peaks.1.intensity",
        ),
        (
            {"peaks": [{"mz": "0", "intensity": 1}, {"mz": "1.003355", "intensity": 1}], "charges": [1], "tolerance": "0.001"},
            "peaks.0.mz",
        ),
        ({"peaks": GOOD_PEAKS, "charges": [], "tolerance": "0.001"}, "charges"),
        ({"peaks": GOOD_PEAKS, "charges": [0], "tolerance": "0.001"}, "charges.0"),
        ({"peaks": GOOD_PEAKS, "charges": [1, 1], "tolerance": "0.001"}, "charges"),
        ({"peaks": GOOD_PEAKS, "charges": [1], "tolerance": "-0.1"}, "tolerance"),
        ({"peaks": GOOD_PEAKS, "charges": [1]}, "tolerance"),
        ({"peaks": GOOD_PEAKS, "charges": [1], "tolerance": "0.001", "debug": True}, "debug"),
    ],
)
def test_invalid_input_is_422_with_located_field_and_no_verdict(payload, expected_loc):
    resp = post(payload)
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert "verdict" not in body
    fields = body["error"]["fields"]
    assert fields, "expected at least one located field error"
    locs = [f["loc"] for f in fields]
    assert any(loc == expected_loc or loc.startswith(expected_loc + ".") for loc in locs), (
        f"expected loc {expected_loc!r} in {locs}"
    )
