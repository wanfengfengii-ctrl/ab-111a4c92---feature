"""Tolerance sensitivity spectrum for the deconvolution verdict.

Before a cluster review, an analyst submits the same peaks and allowed
charges as for a plain deconvolution, plus a *closed* tolerance range
``[lower, upper]``.  This module derives — with exact decimal/rational
arithmetic — every *critical tolerance* at which the legal cluster set can
change, recomputes the existing global deconvolution only at the range
endpoints and those critical points (never on a fixed-step grid, never from
cached results of earlier requests), and merges adjacent intervals whose
normalised verdict is identical.

A peak pair ``i < j`` is adjacent-eligible at charge ``z`` iff
``|Δmz·z − 1.003355| ≤ tolerance·z``, so the legal cluster set can only
change at ``t = |Δmz·z − 1.003355| / z`` for some peak pair and charge.
Because the spacing test is inclusive, the verdict is constant on each
half-open interval ``[t_k, t_{k+1})`` between consecutive critical points;
evaluating exactly *at* each critical point therefore characterises the
whole range.  Critical tolerances are kept as :class:`fractions.Fraction`
because e.g. ``z = 3`` yields non-terminating decimals that no finite
decimal string could represent exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Iterable, Sequence

from .solver import (
    ISOTOPE_SPACING,
    DeconvolutionResult,
    Deconvolver,
    Peak,
    SearchSpaceExceededError,
)

#: Default total work budget for one sensitivity scan.  The budget covers
#: critical-tolerance derivation, every per-tolerance cluster generation,
#: and all exhaustive searches across the scan; exceeding it aborts the
#: scan with :class:`SensitivityScanFailedError` instead of returning a
#: partial spectrum.
DEFAULT_MAX_SENSITIVITY_OPS = 20_000_000


class SensitivityScanFailedError(RuntimeError):
    """The sensitivity scan could not complete within the work budget.

    Raised instead of returning a partial spectrum: no critical conclusion
    is ever silently omitted.
    """


@dataclass(frozen=True)
class SpectrumSegment:
    """One tolerance interval on which the normalised verdict is constant.

    ``lower`` is always inclusive; ``upper`` is inclusive only for the
    final segment (the closed end of the scanned range).
    """

    lower: Fraction
    upper: Fraction
    upper_inclusive: bool
    result: DeconvolutionResult


@dataclass(frozen=True)
class SensitivitySpectrum:
    """Merged segments plus the critical tolerances found inside the range."""

    segments: tuple[SpectrumSegment, ...]
    #: Critical tolerances in ``(lower, upper]``; each triggered a
    #: recomputation of the global deconvolution.
    critical_tolerances: tuple[Fraction, ...]
    #: How many full deconvolutions were recomputed (range endpoint +
    #: in-range critical points — never a fixed-step sample).
    evaluation_count: int


def critical_tolerances(
    peaks: Sequence[Peak], charges: Iterable[int]
) -> tuple[Fraction, ...]:
    """Every tolerance at which the legal cluster set can change.

    For each peak pair ``i < j`` (spacing ``Δmz``) and charge ``z`` the pair
    becomes adjacent-eligible exactly at ``|Δmz·z − 1.003355| / z``.  The
    result is sorted and de-duplicated; every value is an exact fraction
    derived from the decimal peak differences.
    """
    mzs = [Fraction(p.mz) for p in peaks]
    spacing = Fraction(ISOTOPE_SPACING)
    criticals: set[Fraction] = set()
    for z in sorted(set(charges)):
        for i in range(len(mzs)):
            for j in range(i + 1, len(mzs)):
                deviation = (mzs[j] - mzs[i]) * z - spacing
                criticals.add(abs(deviation) / z)
    return tuple(sorted(criticals))


def format_tolerance(value: Fraction) -> str:
    """Canonical exact string for a tolerance boundary.

    Terminating rationals render as plain decimals (``"0.0005"``);
    non-terminating ones render as an exact reduced fraction
    (``"1/3000000"``).  Both forms round-trip through ``Fraction(...)``.
    """
    if value.denominator == 1:
        return str(value.numerator)
    denominator = value.denominator
    twos = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    fives = 0
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        return f"{value.numerator}/{value.denominator}"
    # Terminating: value = scaled / 10**k with scaled not divisible by 10.
    k = max(twos, fives)
    scaled = value.numerator * (2 ** (k - twos)) * (5 ** (k - fives))
    sign = "-" if scaled < 0 else ""
    digits = str(abs(scaled))
    if len(digits) <= k:
        digits = "0" * (k + 1 - len(digits)) + digits
    return f"{sign}{digits[:-k]}.{digits[-k:]}"


def scan_spectrum(
    *,
    peaks: Sequence[Peak],
    charges: Iterable[int],
    lower: Decimal,
    upper: Decimal,
    max_total_ops: int = DEFAULT_MAX_SENSITIVITY_OPS,
) -> SensitivitySpectrum:
    """Compute the verdict's sensitivity spectrum over ``[lower, upper]``.

    The global deconvolution is recomputed from scratch at ``lower`` and at
    every critical tolerance inside the range — nowhere else.  Adjacent
    segments whose normalised verdict (verdict, objectives, clusters,
    unexplained peaks, ambiguity witness) is identical are merged.  Raises
    :class:`SensitivityScanFailedError` if the shared work budget is
    exhausted; no partial spectrum is ever returned.
    """
    lo = Fraction(lower)
    hi = Fraction(upper)
    if lo > hi:
        raise ValueError("lower must not exceed upper")
    peaks = tuple(peaks)
    charges = tuple(sorted(set(charges)))

    # Cluster generation costs one pair-scan per charge; charge it to the
    # same budget as the exhaustive search so pathological inputs (huge
    # charge sets, wide ranges) abort explicitly instead of hanging.
    pair_count = len(peaks) * (len(peaks) - 1) // 2
    generation_cost = max(1, pair_count * len(charges))

    ops_used = 0

    def spend(amount: int, detail: str) -> None:
        nonlocal ops_used
        ops_used += amount
        if ops_used > max_total_ops:
            raise SensitivityScanFailedError(
                "sensitivity scan aborted: work budget "
                f"({max_total_ops} operations) exhausted {detail}; "
                "no spectrum was produced — narrow the tolerance range or "
                "the charge set"
            )

    # 1. Derive every critical tolerance from the peak differences, then
    #    keep those strictly inside the range as recomputation points.
    spend(generation_cost, "while deriving critical tolerances")
    in_range = tuple(t for t in critical_tolerances(peaks, charges) if lo < t <= hi)
    evaluation_points = (lo,) + in_range

    # 2. Recompute the existing global deconvolution at each point.
    results: list[DeconvolutionResult] = []
    for position, point in enumerate(evaluation_points):
        spend(
            generation_cost,
            f"during tolerance evaluation {position + 1} "
            f"of {len(evaluation_points)}",
        )
        solver = Deconvolver(
            peaks,
            charges,
            point,
            max_search_ops=max_total_ops - ops_used,
        )
        try:
            results.append(solver.solve())
        except SearchSpaceExceededError as exc:
            raise SensitivityScanFailedError(
                "sensitivity scan aborted: exact deconvolution exceeded the "
                f"remaining work budget at tolerance {format_tolerance(point)} "
                f"(evaluation {position + 1} of {len(evaluation_points)}); "
                "no spectrum was produced"
            ) from exc
        ops_used += solver.search_ops

    # 3. Materialise the constant-verdict intervals: [t_k, t_{k+1}) and a
    #    closed final segment ending at the range's upper bound.
    segments: list[SpectrumSegment] = []
    last = len(evaluation_points) - 1
    for position, point in enumerate(evaluation_points):
        is_final = position == last
        segments.append(
            SpectrumSegment(
                lower=point,
                upper=hi if is_final else evaluation_points[position + 1],
                upper_inclusive=is_final,
                result=results[position],
            )
        )

    # 4. Merge adjacent segments whose normalised verdict is identical.
    merged: list[SpectrumSegment] = []
    for segment in segments:
        if merged and merged[-1].result == segment.result:
            previous = merged[-1]
            merged[-1] = SpectrumSegment(
                lower=previous.lower,
                upper=segment.upper,
                upper_inclusive=segment.upper_inclusive,
                result=previous.result,
            )
        else:
            merged.append(segment)

    return SensitivitySpectrum(
        segments=tuple(merged),
        critical_tolerances=in_range,
        evaluation_count=len(evaluation_points),
    )
