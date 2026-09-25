"""Exact tolerance-sensitivity spectrum for the deconvolution verdict.

For a fixed peak list and charge set, the legal-cluster universe changes only
when the tolerance crosses the *critical tolerance* of some adjacent peak
pair: the exact value

    crit(i, j, z) = |Δmz(i,j)·z − 1.003355| / z

at which the spacing test ``|Δmz·z − 1.003355| ≤ tol·z`` flips from false to
true (the test is inclusive, so the edge — and every cluster chain born with
it — appears exactly *at* its critical tolerance).

The scanner therefore evaluates the existing exhaustive deconvolution only at
the requested range endpoints and at every critical tolerance strictly inside
the range — never at fixed-step samples, and never reusing results across
requests.  Between two consecutive evaluation points the legal-cluster set is
provably constant, so each point's adjudication holds over the whole
half-open region up to the next point.  Adjacent regions whose *normalised*
adjudication (verdict, objectives, clusters, unexplained peaks, ambiguity
witness) is identical are then merged before the spectrum is returned.

All boundary arithmetic uses :class:`fractions.Fraction`, so critical values
such as ``0.1/3`` that have no finite decimal expansion are still compared and
ordered exactly.  Boundaries are rendered as decimal strings: exact when the
value is a terminating decimal, otherwise rounded to 40 significant digits
(the exact rational value is always used internally; only the display string
is rounded).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Iterable, Sequence

from .solver import (
    ISOTOPE_SPACING,
    DeconvolutionResult,
    Deconvolver,
    Peak,
    SearchSpaceExceededError,
)

#: Significant digits used when a boundary has no exact terminating decimal
#: representation (e.g. the critical tolerance 0.1/3).  Internal comparisons
#: always use the exact rational value; this only affects display strings.
BOUNDARY_DISPLAY_DIGITS = 40


class SensitivityScanError(RuntimeError):
    """The sensitivity scan could not complete; no partial spectrum exists."""


@dataclass(frozen=True)
class SpectrumSegment:
    """One maximal tolerance region with a single constant adjudication.

    ``lower``/``upper`` are exact rationals; ``upper_inclusive`` marks
    whether the upper boundary itself belongs to the segment (true only for
    the final segment, which closes at the requested range upper bound).
    """

    lower: Fraction
    upper: Fraction
    upper_inclusive: bool
    adjudication: DeconvolutionResult


@dataclass(frozen=True)
class SensitivitySpectrum:
    """Complete scan outcome: ordered segments plus the evaluated points."""

    segments: tuple[SpectrumSegment, ...]
    #: Distinct tolerances at which the deconvolution was recomputed:
    #: the range endpoints and every in-range critical tolerance.
    evaluation_points: tuple[Fraction, ...]
    #: Distinct critical tolerances lying strictly inside the range.
    critical_points: tuple[Fraction, ...]


def critical_tolerances(
    peaks: Sequence[Peak], charges: Iterable[int]
) -> tuple[Fraction, ...]:
    """Every tolerance at which the legal-cluster universe can change.

    Derived exactly from adjacent-peak differences and the allowed charges:
    for each peak pair ``i < j`` and charge ``z`` the critical value is
    ``|Δmz·z − 1.003355| / z``.  Each returned value is the birth tolerance
    of at least the 2-peak cluster ``(i, j)`` at charge ``z``, so the set is
    precisely the set of tolerances where the legal-cluster universe changes.
    """
    mzs = [Fraction(p.mz) for p in peaks]
    spacing = Fraction(ISOTOPE_SPACING)
    critical: set[Fraction] = set()
    for charge in sorted(set(charges)):
        for i in range(len(mzs)):
            for j in range(i + 1, len(mzs)):
                critical.add(abs((mzs[j] - mzs[i]) * charge - spacing) / charge)
    return tuple(sorted(critical))


def normalised_adjudication_key(
    result: DeconvolutionResult, peak_count: int
) -> tuple:
    """Canonical identity of an adjudication for segment merging.

    Two segments merge only if verdict, objectives, primary clusters,
    unexplained peaks and the ambiguity witness are all identical.
    """

    def solution_key(clusters: tuple | None) -> tuple | None:
        if clusters is None:
            return None
        return tuple((c.charge, c.peak_indices) for c in clusters)

    explained = {idx for cluster in result.primary for idx in cluster.peak_indices}
    unexplained = tuple(idx for idx in range(peak_count) if idx not in explained)
    return (
        result.verdict,
        result.explained_intensity,
        result.explained_peak_count,
        result.cluster_count,
        solution_key(result.primary),
        unexplained,
        solution_key(result.secondary),
    )


class SensitivityScanner:
    """Computes the tolerance-sensitivity spectrum over a closed range.

    The scan is recomputed from scratch on every call: no state is shared
    between requests and no fixed-step sampling is used.
    """

    def __init__(
        self,
        peaks: Sequence[Peak],
        charges: Iterable[int],
        lower: Decimal,
        upper: Decimal,
        max_search_ops: int,
    ) -> None:
        self._peaks = tuple(peaks)
        self._charges = tuple(sorted(set(charges)))
        self._lower = Fraction(lower)
        self._upper = Fraction(upper)
        self._max_search_ops = max_search_ops

    def run(self) -> SensitivitySpectrum:
        critical = critical_tolerances(self._peaks, self._charges)
        interior = tuple(
            value for value in critical if self._lower < value < self._upper
        )
        # Recompute the global deconvolution exactly at the range endpoints
        # and at every critical tolerance strictly inside the range.
        evaluation_points = tuple(
            sorted({self._lower, self._upper, *interior})
        )
        adjudications = [self._adjudicate(point) for point in evaluation_points]
        regions = self._point_regions(evaluation_points, adjudications)
        segments = self._merge(regions)
        return SensitivitySpectrum(
            segments=tuple(segments),
            evaluation_points=evaluation_points,
            critical_points=interior,
        )

    # ------------------------------------------------------------------ #
    # Deconvolution at one tolerance
    # ------------------------------------------------------------------ #

    def _adjudicate(self, tolerance: Fraction) -> DeconvolutionResult:
        try:
            return Deconvolver(
                peaks=self._peaks,
                charges=self._charges,
                tolerance=tolerance,
                max_search_ops=self._max_search_ops,
            ).solve()
        except SearchSpaceExceededError as exc:
            raise SensitivityScanError(
                "sensitivity scan failed: the exact deconvolution at "
                f"tolerance {format_boundary(tolerance)} exceeded the "
                f"configured work budget ({self._max_search_ops} operations); "
                "no partial spectrum was produced — narrow the tolerance "
                "range or the charge set"
            ) from exc

    # ------------------------------------------------------------------ #
    # Region construction and merging
    # ------------------------------------------------------------------ #

    def _point_regions(
        self,
        points: tuple[Fraction, ...],
        adjudications: list[DeconvolutionResult],
    ) -> list[SpectrumSegment]:
        """One region per evaluation point, before merging.

        Region k covers ``[points[k], points[k+1])``; its adjudication is
        constant inside because no critical tolerance lies strictly between
        consecutive evaluation points.  The final region is the closed point
        ``[upper, upper]`` at the requested range upper bound.  A degenerate
        range (lower == upper) yields exactly that single point region.
        """
        regions = [
            SpectrumSegment(
                lower=points[k],
                upper=points[k + 1],
                upper_inclusive=False,
                adjudication=adjudications[k],
            )
            for k in range(len(points) - 1)
        ]
        regions.append(
            SpectrumSegment(
                lower=points[-1],
                upper=points[-1],
                upper_inclusive=True,
                adjudication=adjudications[-1],
            )
        )
        return regions

    def _merge(self, regions: list[SpectrumSegment]) -> list[SpectrumSegment]:
        """Merge adjacent regions whose normalised adjudication is identical."""
        merged: list[SpectrumSegment] = []
        for region in regions:
            if merged and normalised_adjudication_key(
                merged[-1].adjudication, len(self._peaks)
            ) == normalised_adjudication_key(region.adjudication, len(self._peaks)):
                previous = merged[-1]
                merged[-1] = SpectrumSegment(
                    lower=previous.lower,
                    upper=region.upper,
                    upper_inclusive=region.upper_inclusive,
                    adjudication=previous.adjudication,
                )
            else:
                merged.append(region)
        return merged


def format_boundary(value: Fraction) -> str:
    """Render an exact rational boundary as a decimal string.

    Terminating decimals are rendered exactly (no rounding, no trailing
    zeros); non-terminating values are rounded to
    ``BOUNDARY_DISPLAY_DIGITS`` significant digits.
    """
    if value.denominator == 1:
        return str(value.numerator)
    # Factor the denominator: only 2s and 5s means a terminating expansion.
    rest = value.denominator
    twos = 0
    while rest % 2 == 0:
        rest //= 2
        twos += 1
    fives = 0
    while rest % 5 == 0:
        rest //= 5
        fives += 1
    sign = "-" if value < 0 else ""
    if rest == 1:
        # Terminating: value = scaled / 10^k with k = max(twos, fives); the
        # scaled integer gives the exact expansion with no rounding at all.
        k = max(twos, fives)
        scaled = abs(value.numerator) * 2 ** (k - twos) * 5 ** (k - fives)
        digits = str(scaled).zfill(k + 1)
        text = digits[:-k] + "." + digits[-k:] if k else digits
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return sign + text
    # Non-terminating: round to BOUNDARY_DISPLAY_DIGITS significant digits.
    # This branch is explicitly approximate, so a Decimal division under a
    # sufficiently wide local context is the clearest exact-specified rounding.
    with localcontext() as context:
        context.prec = BOUNDARY_DISPLAY_DIGITS
        approx = Decimal(value.numerator) / Decimal(value.denominator)
    text = format(approx, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text
