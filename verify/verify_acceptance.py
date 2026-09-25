#!/usr/bin/env python3
"""One-shot acceptance suite against a live Isotope Deconvolution API.

Usage (Docker Compose, from the repository root):

    docker compose run --rm verify

or against any reachable instance:

    API_BASE_URL=http://localhost:8000 python verify/verify_acceptance.py

Every check talks to the real HTTP API.  The process exits 0 only if all
checks pass.
"""

from __future__ import annotations

import os
import sys
import time

import httpx

BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
HEALTH_URL = f"{BASE_URL}/health"
DECONVOLVE_URL = f"{BASE_URL}/api/v1/deconvolve"
SENSITIVITY_URL = f"{BASE_URL}/api/v1/deconvolve/sensitivity"

_checks = 0
_failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global _checks
    _checks += 1
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}" + (f"  -- {detail}" if detail else ""))
        _failures.append(name)


def wait_for_api(timeout_s: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(HEALTH_URL, timeout=3.0)
            if resp.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    return False


def post(client: httpx.Client, payload: dict) -> httpx.Response:
    return client.post(DECONVOLVE_URL, json=payload, timeout=30.0)


def post_sensitivity(client: httpx.Client, payload: dict) -> httpx.Response:
    return client.post(SENSITIVITY_URL, json=payload, timeout=60.0)


def cluster_index_sets(solution_clusters: list[dict]) -> set[tuple[int, ...]]:
    return {tuple(c["peak_indices"]) for c in solution_clusters}


# ---------------------------------------------------------------------- #
# Scenarios
# ---------------------------------------------------------------------- #


def scenario_health(client: httpx.Client) -> None:
    print("[health]")
    resp = client.get(HEALTH_URL, timeout=5.0)
    check("GET /health returns 200", resp.status_code == 200, f"got {resp.status_code}")
    body = resp.json() if resp.status_code == 200 else {}
    check("health body reports ok", body.get("status") == "ok", repr(body))


def scenario_unique(client: httpx.Client) -> dict:
    print("[unique verdict]")
    payload = {
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
    resp = post(client, payload)
    check("unique: 200", resp.status_code == 200, f"got {resp.status_code}: {resp.text}")
    body = resp.json()
    check("unique: verdict UNIQUE", body.get("verdict") == "UNIQUE", body.get("verdict", ""))
    obj = body.get("objectives", {})
    check(
        "unique: objectives (2800 / 4 / 1)",
        (obj.get("explained_intensity"), obj.get("explained_peak_count"), obj.get("cluster_count"))
        == (2800, 4, 1),
        repr(obj),
    )
    clusters = body.get("clusters", [])
    check("unique: one cluster", len(clusters) == 1, repr(clusters))
    if clusters:
        check(
            "unique: cluster is charge 1 over peaks [1,2,3,4]",
            clusters[0].get("charge") == 1 and clusters[0].get("peak_indices") == [1, 2, 3, 4],
            repr(clusters[0]),
        )
    unexplained = [p["index"] for p in body.get("unexplained_peaks", [])]
    check("unique: unexplained peaks are [0, 5]", unexplained == [0, 5], repr(unexplained))
    check("unique: no second witness", body.get("second_witness") is None)
    return payload


def scenario_determinism(client: httpx.Client, payload: dict) -> None:
    print("[determinism]")
    bodies = {post(client, payload).text for _ in range(3)}
    check("identical input yields byte-identical responses", len(bodies) == 1)


def scenario_ambiguous(client: httpx.Client) -> None:
    print("[ambiguous verdict + second witness]")
    # d(0,1)=0.6 and d(0,2)=1.0 both lie within 0.5 of 1.003355, d(1,2)=0.4 does
    # not.  Clusters {0,1} and {0,2} tie on (intensity 300, 2 peaks, 1 cluster).
    payload = {
        "peaks": [
            {"mz": "300.0", "intensity": 100},
            {"mz": "300.6", "intensity": 200},
            {"mz": "301.0", "intensity": 200},
        ],
        "charges": [1],
        "tolerance": "0.5",
    }
    resp = post(client, payload)
    check("ambiguous: 200", resp.status_code == 200, f"got {resp.status_code}: {resp.text}")
    body = resp.json()
    check("ambiguous: verdict AMBIGUOUS", body.get("verdict") == "AMBIGUOUS", body.get("verdict", ""))
    witness = body.get("second_witness")
    check("ambiguous: second witness present", witness is not None)
    if witness:
        primary_sets = cluster_index_sets(body.get("clusters", []))
        witness_sets = cluster_index_sets(witness.get("clusters", []))
        check(
            "ambiguous: witness differs from primary",
            primary_sets != witness_sets,
            f"primary={primary_sets} witness={witness_sets}",
        )
        check(
            "ambiguous: witnesses are {{0,1}} and {{0,2}}",
            primary_sets | witness_sets == {(0, 1), (0, 2)},
            f"primary={primary_sets} witness={witness_sets}",
        )
        w_intensity = sum(c["explained_intensity"] for c in witness.get("clusters", []))
        obj = body.get("objectives", {})
        check(
            "ambiguous: witness matches primary objectives",
            w_intensity == obj.get("explained_intensity")
            and len(witness.get("clusters", [])) == obj.get("cluster_count"),
            f"witness_intensity={w_intensity} objectives={obj}",
        )


def scenario_unresolved(client: httpx.Client) -> None:
    print("[unresolved verdict]")
    payload = {
        "peaks": [
            {"mz": "100.0", "intensity": 10},
            {"mz": "100.3", "intensity": 20},
            {"mz": "100.6", "intensity": 30},
        ],
        "charges": [1],
        "tolerance": "0.001",
    }
    resp = post(client, payload)
    check("unresolved: 200", resp.status_code == 200, f"got {resp.status_code}: {resp.text}")
    body = resp.json()
    check("unresolved: verdict UNRESOLVED", body.get("verdict") == "UNRESOLVED", body.get("verdict", ""))
    check("unresolved: no clusters", body.get("clusters") == [])
    check(
        "unresolved: all peaks unexplained",
        [p["index"] for p in body.get("unexplained_peaks", [])] == [0, 1, 2],
    )
    obj = body.get("objectives", {})
    check(
        "unresolved: zero objectives",
        (obj.get("explained_intensity"), obj.get("explained_peak_count"), obj.get("cluster_count"))
        == (0, 0, 0),
        repr(obj),
    )


def scenario_intensity_before_peak_count(client: httpx.Client) -> None:
    print("[lexicographic objectives: intensity beats peak count]")
    # X = {1,3} at z=1 explains 200 over 2 peaks; Y = {0,1,2} at z=3 explains
    # 102 over 3 peaks.  They conflict on peak 1 and Y's leftover peaks cannot
    # recombine, so the intensity-first lexicographic order must prefer X.
    payload = {
        "peaks": [
            {"mz": "500.6689033", "intensity": 1},
            {"mz": "501.003355", "intensity": 100},
            {"mz": "501.3378067", "intensity": 1},
            {"mz": "502.006710", "intensity": 100},
        ],
        "charges": [1, 2, 3],
        "tolerance": "0.0001",
    }
    resp = post(client, payload)
    check("lexico: 200", resp.status_code == 200, f"got {resp.status_code}: {resp.text}")
    body = resp.json()
    obj = body.get("objectives", {})
    check(
        "lexico: intensity-first optimum (200 / 2 / 1)",
        (obj.get("explained_intensity"), obj.get("explained_peak_count"), obj.get("cluster_count"))
        == (200, 2, 1),
        repr(obj),
    )
    clusters = body.get("clusters", [])
    check(
        "lexico: chosen cluster is {1,3} at z=1",
        len(clusters) == 1
        and clusters[0].get("peak_indices") == [1, 3]
        and clusters[0].get("charge") == 1,
        repr(clusters),
    )
    check("lexico: verdict UNIQUE", body.get("verdict") == "UNIQUE", body.get("verdict", ""))
    check(
        "lexico: unexplained peaks are [0, 2]",
        [p["index"] for p in body.get("unexplained_peaks", [])] == [0, 2],
    )


def scenario_charge_two(client: httpx.Client) -> None:
    print("[charge-state spacing]")
    base = {
        "peaks": [
            {"mz": "700.0000000", "intensity": 10},
            {"mz": "700.5016775", "intensity": 20},
        ],
        "tolerance": "0.0000001",
    }
    resp = post(client, {**base, "charges": [2]})
    body = resp.json()
    clusters = body.get("clusters", [])
    check(
        "z=2: pair at 1.003355/2 spacing forms a cluster",
        resp.status_code == 200
        and body.get("verdict") == "UNIQUE"
        and len(clusters) == 1
        and clusters[0].get("charge") == 2
        and clusters[0].get("peak_indices") == [0, 1],
        resp.text,
    )
    resp = post(client, {**base, "charges": [1]})
    check(
        "z=1: same pair is not an isotope spacing",
        resp.status_code == 200 and resp.json().get("verdict") == "UNRESOLVED",
        resp.text,
    )


def scenario_tolerance_boundary(client: httpx.Client) -> None:
    print("[tolerance boundary is inclusive]")
    base = {
        "peaks": [
            {"mz": "400.000000", "intensity": 5},
            {"mz": "401.003855", "intensity": 7},  # deviation exactly 0.0005
        ],
        "charges": [1],
    }
    resp = post(client, {**base, "tolerance": "0.0005"})
    check(
        "boundary: deviation == tolerance is accepted",
        resp.status_code == 200 and resp.json().get("verdict") == "UNIQUE",
        resp.text,
    )
    resp = post(client, {**base, "tolerance": "0.0004999"})
    check(
        "boundary: deviation > tolerance is rejected",
        resp.status_code == 200 and resp.json().get("verdict") == "UNRESOLVED",
        resp.text,
    )


def scenario_cluster_size_cap(client: httpx.Client) -> None:
    print("[cluster size capped at 6]")
    spacing = "1.003355"
    from decimal import Decimal

    mz = Decimal("900.000000")
    peaks = []
    for i in range(7):
        peaks.append({"mz": str(mz), "intensity": 1})
        mz += Decimal(spacing)
    payload = {"peaks": peaks, "charges": [1], "tolerance": "0.0001"}
    resp = post(client, payload)
    body = resp.json()
    obj = body.get("objectives", {})
    check(
        "7-chain: all 7 peaks explained by 2 clusters (cap at 6 forces a split)",
        resp.status_code == 200
        and (obj.get("explained_peak_count"), obj.get("cluster_count")) == (7, 2),
        resp.text,
    )
    sizes = [len(c["peak_indices"]) for c in body.get("clusters", [])]
    check("7-chain: every cluster has at most 6 peaks", all(s <= 6 for s in sizes), repr(sizes))
    check(
        "7-chain: multiple equal splits exist -> AMBIGUOUS with witness",
        body.get("verdict") == "AMBIGUOUS" and body.get("second_witness") is not None,
        body.get("verdict", ""),
    )


def scenario_full_scale(client: httpx.Client) -> None:
    print("[36 peaks, 6 disjoint chains]")
    from decimal import Decimal

    spacing = Decimal("1.003355")
    peaks = []
    for k in range(6):
        mz = Decimal(200 + 100 * k)
        for i in range(6):
            peaks.append({"mz": str(mz), "intensity": 100 + 10 * k + i})
            mz += spacing
    payload = {"peaks": peaks, "charges": [1], "tolerance": "0.0001"}
    expected_intensity = sum(p["intensity"] for p in peaks)
    resp = post(client, payload)
    body = resp.json()
    obj = body.get("objectives", {})
    check(
        "36 peaks: UNIQUE, 6 clusters, everything explained",
        resp.status_code == 200
        and body.get("verdict") == "UNIQUE"
        and (obj.get("explained_intensity"), obj.get("explained_peak_count"), obj.get("cluster_count"))
        == (expected_intensity, 36, 6)
        and body.get("unexplained_peaks") == [],
        resp.text[:400],
    )


def scenario_validation(client: httpx.Client) -> None:
    print("[validation: field-locatable 422, never a verdict]")
    good_peaks = [
        {"mz": "500.000000", "intensity": 100},
        {"mz": "501.003355", "intensity": 90},
    ]
    cases = {
        "too few peaks": {"peaks": good_peaks[:1], "charges": [1], "tolerance": "0.001"},
        "too many peaks": {
            "peaks": [{"mz": str(100 + i), "intensity": 1} for i in range(37)],
            "charges": [1],
            "tolerance": "0.001",
        },
        "mz not increasing": {
            "peaks": [
                {"mz": "501.003355", "intensity": 90},
                {"mz": "500.000000", "intensity": 100},
            ],
            "charges": [1],
            "tolerance": "0.001",
        },
        "zero intensity": {
            "peaks": [good_peaks[0], {"mz": "501.003355", "intensity": 0}],
            "charges": [1],
            "tolerance": "0.001",
        },
        "fractional intensity": {
            "peaks": [good_peaks[0], {"mz": "501.003355", "intensity": 1.5}],
            "charges": [1],
            "tolerance": "0.001",
        },
        "non-positive mz": {
            "peaks": [{"mz": "0", "intensity": 1}, {"mz": "1.003355", "intensity": 1}],
            "charges": [1],
            "tolerance": "0.001",
        },
        "empty charges": {"peaks": good_peaks, "charges": [], "tolerance": "0.001"},
        "zero charge": {"peaks": good_peaks, "charges": [0], "tolerance": "0.001"},
        "duplicate charges": {"peaks": good_peaks, "charges": [1, 1], "tolerance": "0.001"},
        "negative tolerance": {"peaks": good_peaks, "charges": [1], "tolerance": "-0.1"},
        "missing tolerance": {"peaks": good_peaks, "charges": [1]},
        "unknown field": {"peaks": good_peaks, "charges": [1], "tolerance": "0.001", "debug": True},
    }
    for name, payload in cases.items():
        resp = post(client, payload)
        ok_status = resp.status_code == 422
        body = resp.json() if ok_status else {}
        fields = body.get("error", {}).get("fields", [])
        located = all(f.get("loc") for f in fields) and len(fields) > 0
        check(
            f"validation[{name}]: 422 with located fields, no verdict",
            ok_status and located and "verdict" not in body,
            f"status={resp.status_code} body={resp.text[:300]}",
        )


# ---------------------------------------------------------------------- #
# Sensitivity spectrum scenarios
# ---------------------------------------------------------------------- #


def scenario_sensitivity_boundary_split(client: httpx.Client) -> dict:
    print("[sensitivity: spectrum splits exactly at the critical tolerance]")
    # Pair deviation is exactly 0.0005 at z=1, so the verdict must flip at
    # the derived critical tolerance 0.0005 — not on any sampled grid.
    payload = {
        "peaks": [
            {"mz": "400.000000", "intensity": 5},
            {"mz": "401.003855", "intensity": 7},
        ],
        "charges": [1],
        "tolerance_range": {"lower": "0.0001", "upper": "0.001"},
    }
    resp = post_sensitivity(client, payload)
    check("spectrum: 200", resp.status_code == 200, f"got {resp.status_code}: {resp.text}")
    body = resp.json()
    segments = body.get("segments", [])
    check(
        "spectrum: critical tolerance derived exactly as 0.0005",
        body.get("critical_tolerances") == ["0.0005"],
        repr(body.get("critical_tolerances")),
    )
    check(
        "spectrum: recomputed only at range endpoint + critical point",
        body.get("evaluation_count") == 2,
        repr(body.get("evaluation_count")),
    )
    check("spectrum: two segments", len(segments) == 2, repr(segments))
    if len(segments) == 2:
        lo_seg, hi_seg = segments
        check(
            "spectrum: [0.0001, 0.0005) is UNRESOLVED, upper excluded",
            lo_seg.get("lower") == "0.0001"
            and lo_seg.get("upper") == "0.0005"
            and lo_seg.get("upper_inclusive") is False
            and lo_seg.get("verdict") == "UNRESOLVED"
            and lo_seg.get("clusters") == []
            and [p["index"] for p in lo_seg.get("unexplained_peaks", [])] == [0, 1],
            repr(lo_seg),
        )
        obj = hi_seg.get("objectives", {})
        check(
            "spectrum: [0.0005, 0.001] is UNIQUE (12/2/1), upper included",
            hi_seg.get("lower") == "0.0005"
            and hi_seg.get("upper") == "0.001"
            and hi_seg.get("upper_inclusive") is True
            and hi_seg.get("verdict") == "UNIQUE"
            and (obj.get("explained_intensity"), obj.get("explained_peak_count"), obj.get("cluster_count"))
            == (12, 2, 1)
            and [c["peak_indices"] for c in hi_seg.get("clusters", [])] == [[0, 1]]
            and hi_seg.get("second_witness") is None,
            repr(hi_seg),
        )
        check(
            "spectrum: segments contiguous and ascending",
            lo_seg.get("upper") == hi_seg.get("lower"),
            f"{lo_seg.get('upper')} != {hi_seg.get('lower')}",
        )
    # The plain deconvolution endpoint must agree on both sides.
    at = post(client, {"peaks": payload["peaks"], "charges": [1], "tolerance": "0.0005"})
    below = post(client, {"peaks": payload["peaks"], "charges": [1], "tolerance": "0.0004999"})
    check(
        "spectrum: deconvolve agrees at and below the critical tolerance",
        at.status_code == 200
        and at.json().get("verdict") == "UNIQUE"
        and below.status_code == 200
        and below.json().get("verdict") == "UNRESOLVED",
        f"at={at.text} below={below.text}",
    )
    return payload


def scenario_sensitivity_determinism(client: httpx.Client, payload: dict) -> None:
    print("[sensitivity: determinism]")
    bodies = {post_sensitivity(client, payload).text for _ in range(3)}
    check("identical scan yields byte-identical spectra", len(bodies) == 1)


def scenario_sensitivity_merges_unchanged_verdicts(client: httpx.Client) -> None:
    print("[sensitivity: adjacent identical verdicts are merged]")
    # Exact 3-chain: pair (0,2) becomes eligible at 1.003355 but {0,2} never
    # beats the full chain, so the verdict is identical on both sides of the
    # critical point and the two intervals must collapse into one.
    payload = {
        "peaks": [
            {"mz": "500.000000", "intensity": 10},
            {"mz": "501.003355", "intensity": 10},
            {"mz": "502.006710", "intensity": 10},
        ],
        "charges": [1],
        "tolerance_range": {"lower": "0", "upper": "2"},
    }
    resp = post_sensitivity(client, payload)
    check("merge: 200", resp.status_code == 200, f"got {resp.status_code}: {resp.text}")
    body = resp.json()
    segments = body.get("segments", [])
    check(
        "merge: critical 1.003355 derived, two evaluations performed",
        body.get("critical_tolerances") == ["1.003355"] and body.get("evaluation_count") == 2,
        repr({k: body.get(k) for k in ("critical_tolerances", "evaluation_count")}),
    )
    check("merge: identical verdicts collapse to one segment", len(segments) == 1, repr(segments))
    if segments:
        seg = segments[0]
        obj = seg.get("objectives", {})
        check(
            "merge: single segment [0, 2] UNIQUE (30/3/1), upper included",
            seg.get("lower") == "0"
            and seg.get("upper") == "2"
            and seg.get("upper_inclusive") is True
            and seg.get("verdict") == "UNIQUE"
            and (obj.get("explained_intensity"), obj.get("explained_peak_count"), obj.get("cluster_count"))
            == (30, 3, 1)
            and [c["peak_indices"] for c in seg.get("clusters", [])] == [[0, 1, 2]],
            repr(seg),
        )


def scenario_sensitivity_fractional_boundary(client: httpx.Client) -> None:
    print("[sensitivity: non-terminating critical tolerance stays exact]")
    # z=3: deviation is exactly 0.000001, so the critical tolerance is
    # 1/3000000 = 0.000000333... — reported as an exact fraction string.
    payload = {
        "peaks": [
            {"mz": "500.000000", "intensity": 10},
            {"mz": "500.334452", "intensity": 20},
        ],
        "charges": [3],
        "tolerance_range": {"lower": "0.0000001", "upper": "0.000001"},
    }
    resp = post_sensitivity(client, payload)
    check("fraction: 200", resp.status_code == 200, f"got {resp.status_code}: {resp.text}")
    body = resp.json()
    segments = body.get("segments", [])
    check(
        "fraction: critical tolerance is exactly 1/3000000",
        body.get("critical_tolerances") == ["1/3000000"],
        repr(body.get("critical_tolerances")),
    )
    check("fraction: two segments", len(segments) == 2, repr(segments))
    if len(segments) == 2:
        lo_seg, hi_seg = segments
        check(
            "fraction: exact boundaries and inclusive flags",
            lo_seg.get("lower") == "0.0000001"
            and lo_seg.get("upper") == "1/3000000"
            and lo_seg.get("upper_inclusive") is False
            and lo_seg.get("verdict") == "UNRESOLVED"
            and hi_seg.get("lower") == "1/3000000"
            and hi_seg.get("upper") == "0.000001"
            and hi_seg.get("upper_inclusive") is True
            and hi_seg.get("verdict") == "UNIQUE"
            and hi_seg.get("clusters", [{}])[0].get("charge") == 3,
            repr(segments),
        )
    above = post(client, {"peaks": payload["peaks"], "charges": [3], "tolerance": "0.0000004"})
    below = post(client, {"peaks": payload["peaks"], "charges": [3], "tolerance": "0.0000003"})
    check(
        "fraction: deconvolve agrees on both sides of 1/3000000",
        above.status_code == 200
        and above.json().get("verdict") == "UNIQUE"
        and below.status_code == 200
        and below.json().get("verdict") == "UNRESOLVED",
        f"above={above.text} below={below.text}",
    )


def scenario_sensitivity_validation(client: httpx.Client) -> None:
    print("[sensitivity: invalid range/peaks -> 422, never a partial spectrum]")
    good_peaks = [
        {"mz": "500.000000", "intensity": 100},
        {"mz": "501.003355", "intensity": 90},
    ]
    good_range = {"lower": "0.0001", "upper": "0.01"}
    cases = {
        "lower above upper": (
            {"peaks": good_peaks, "charges": [1], "tolerance_range": {"lower": "0.01", "upper": "0.001"}},
            "tolerance_range",
        ),
        "negative lower": (
            {"peaks": good_peaks, "charges": [1], "tolerance_range": {"lower": "-0.1", "upper": "0.1"}},
            "tolerance_range.lower",
        ),
        "missing upper": (
            {"peaks": good_peaks, "charges": [1], "tolerance_range": {"lower": "0.1"}},
            "tolerance_range.upper",
        ),
        "missing range": (
            {"peaks": good_peaks, "charges": [1]},
            "tolerance_range",
        ),
        "mz not increasing": (
            {
                "peaks": [dict(good_peaks[1]), dict(good_peaks[0])],
                "charges": [1],
                "tolerance_range": good_range,
            },
            "peaks",
        ),
        "unknown field": (
            {"peaks": good_peaks, "charges": [1], "tolerance_range": good_range, "debug": True},
            "debug",
        ),
    }
    for name, (payload, want_loc) in cases.items():
        resp = post_sensitivity(client, payload)
        ok_status = resp.status_code == 422
        body = resp.json() if ok_status else {}
        fields = body.get("error", {}).get("fields", [])
        locs = [f.get("loc", "") for f in fields]
        located = any(loc == want_loc or loc.startswith(want_loc + ".") for loc in locs)
        check(
            f"spectrum validation[{name}]: 422 located at {want_loc}, no partial spectrum",
            ok_status and located and "segments" not in body,
            f"status={resp.status_code} locs={locs} body={resp.text[:300]}",
        )


# ---------------------------------------------------------------------- #


def main() -> int:
    print(f"Acceptance target: {BASE_URL}")
    if not wait_for_api():
        print("FATAL: API did not become healthy within 60s")
        return 1
    with httpx.Client() as client:
        scenario_health(client)
        payload = scenario_unique(client)
        scenario_determinism(client, payload)
        scenario_ambiguous(client)
        scenario_unresolved(client)
        scenario_intensity_before_peak_count(client)
        scenario_charge_two(client)
        scenario_tolerance_boundary(client)
        scenario_cluster_size_cap(client)
        scenario_full_scale(client)
        scenario_validation(client)
        scan_payload = scenario_sensitivity_boundary_split(client)
        scenario_sensitivity_determinism(client, scan_payload)
        scenario_sensitivity_merges_unchanged_verdicts(client)
        scenario_sensitivity_fractional_boundary(client)
        scenario_sensitivity_validation(client)
    print(f"\n{_checks - len(_failures)}/{_checks} checks passed")
    if _failures:
        print("FAILED checks:")
        for name in _failures:
            print(f"  - {name}")
        return 1
    print("ALL ACCEPTANCE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
