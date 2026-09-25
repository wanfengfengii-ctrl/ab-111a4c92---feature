"""Exact, deterministic deconvolution of overlapping isotope peak clusters.

A *cluster* is a set of 2-6 peaks sharing one charge state ``z`` whose
adjacent m/z spacings each deviate from ``1.003355 / z`` by no more than the
given tolerance.  Every peak may belong to at most one cluster.

This module performs an *exhaustive* search over all legal combinations of
mutually disjoint clusters and optimises the objectives lexicographically:

1. maximise the total explained intensity;
2. maximise the number of explained peaks;
3. minimise the number of clusters.

No greedy nearest-peak or strongest-candidate-first heuristics are used: a
dynamic program over peak bitmasks explores the complete search space, and
ties on all three objectives are detected by enumerating a second optimal
witness.  All arithmetic is exact — :class:`decimal.Decimal` for API
requests, and :class:`fractions.Fraction` for the tolerance sensitivity
scan (see :mod:`app.sensitivity`), whose critical tolerances may be
non-terminating decimals — so results are exact and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Iterable, Sequence

#: Exact isotope spacing (mass difference of 13C vs 12C) in Dalton.
ISOTOPE_SPACING = Decimal("1.003355")

MIN_CLUSTER_SIZE = 2
MAX_CLUSTER_SIZE = 6

VERDICT_UNIQUE = "UNIQUE"
VERDICT_AMBIGUOUS = "AMBIGUOUS"
VERDICT_UNRESOLVED = "UNRESOLVED"

#: Default budget of cluster-expansion operations for the exact search.  The
#: search is always exhaustive; the budget only guards the service against
#: pathological inputs (e.g. a tolerance as wide as the isotope spacing
#: itself) where the NP-hard set-packing search space explodes.  Exceeding it
#: raises :class:`SearchSpaceExceededError` instead of hanging.
DEFAULT_MAX_SEARCH_OPS = 20_000_000


class SearchSpaceExceededError(RuntimeError):
    """The exact search exceeded the configured work budget."""

# Objective tuples are (explained_intensity, explained_peak_count, -cluster_count).
# Plain tuple comparison then implements the required lexicographic order:
# intensity first, then explained peak count, then fewest clusters.
Objective = tuple[int, int, int]


@dataclass(frozen=True)
class Peak:
    """One input peak. ``index`` is its position in the submitted list."""

    index: int
    mz: Decimal
    intensity: int


@dataclass(frozen=True)
class Cluster:
    """A legal isotope cluster: 2-6 peaks at one charge state."""

    charge: int
    peak_indices: tuple[int, ...]
    mask: int
    explained_intensity: int

    @property
    def size(self) -> int:
        return len(self.peak_indices)

    def canonical_key(self) -> tuple:
        return (self.peak_indices[0], self.charge, self.peak_indices)


@dataclass(frozen=True)
class DeconvolutionResult:
    verdict: str
    explained_intensity: int
    explained_peak_count: int
    cluster_count: int
    #: Canonically sorted clusters of the primary optimal solution.
    primary: tuple[Cluster, ...]
    #: A second, distinct optimal solution (only for AMBIGUOUS verdicts).
    secondary: tuple[Cluster, ...] | None


class Deconvolver:
    """Exhaustive exact solver over disjoint isotope-peak clusters."""

    def __init__(
        self,
        peaks: Sequence[Peak],
        charges: Iterable[int],
        tolerance: Decimal | Fraction,
        max_search_ops: int = DEFAULT_MAX_SEARCH_OPS,
    ) -> None:
        peaks = tuple(peaks)
        if not peaks:
            raise ValueError("at least one peak is required")
        charges = tuple(sorted(set(charges)))
        if not charges:
            raise ValueError("at least one charge state is required")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        self._peaks = peaks
        self._charges = charges
        self._tolerance = tolerance
        self._max_search_ops = max_search_ops
        self._search_ops = 0
        self._full_mask = (1 << len(peaks)) - 1
        self.clusters: tuple[Cluster, ...] = tuple(self._generate_clusters())
        by_peak: list[list[Cluster]] = [[] for _ in peaks]
        for cluster in self.clusters:
            for idx in cluster.peak_indices:
                by_peak[idx].append(cluster)
        # Deterministic bucket order so witness enumeration is reproducible.
        self._clusters_by_peak: tuple[tuple[Cluster, ...], ...] = tuple(
            tuple(sorted(bucket, key=lambda c: (c.charge, c.peak_indices)))
            for bucket in by_peak
        )
        self._best_memo: dict[int, Objective] = {}

    @property
    def search_ops(self) -> int:
        """Cluster-expansion operations consumed so far (budget accounting)."""
        return self._search_ops

    # ------------------------------------------------------------------ #
    # Cluster generation
    # ------------------------------------------------------------------ #

    def _generate_clusters(self) -> list[Cluster]:
        """Enumerate every legal cluster for every allowed charge state.

        The spacing test is evaluated exactly: ``|Δmz·z − 1.003355| ≤ tol·z``
        is equivalent to ``|Δmz − 1.003355/z| ≤ tol`` but needs no division.
        When ``tolerance`` is a :class:`~fractions.Fraction` (a critical
        tolerance from the sensitivity scan, possibly a non-terminating
        decimal) the m/z values are promoted to exact fractions as well, so
        the boundary comparison stays exact.
        """
        if isinstance(self._tolerance, Fraction):
            mzs: list = [Fraction(p.mz) for p in self._peaks]
            spacing = Fraction(ISOTOPE_SPACING)
        else:
            mzs = [p.mz for p in self._peaks]
            spacing = ISOTOPE_SPACING
        intensities = [p.intensity for p in self._peaks]
        n = len(mzs)
        clusters: list[Cluster] = []
        for charge in self._charges:
            threshold = self._tolerance * charge
            # adjacency[i] = peaks j > i whose spacing from i matches 1.003355/z.
            adjacency: list[list[int]] = [[] for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    delta = mzs[j] - mzs[i]
                    deviation = delta * charge - spacing
                    if deviation > threshold:
                        break  # m/z strictly increasing: later j deviate even more
                    if deviation >= -threshold:
                        adjacency[i].append(j)
            # A cluster is a chain i1 < i2 < ... < ik (2 <= k <= 6) of
            # adjacent matches; extend chains depth-first.
            chain: list[int] = []

            def visit() -> None:
                if len(chain) >= MIN_CLUSTER_SIZE:
                    mask = 0
                    total = 0
                    for idx in chain:
                        mask |= 1 << idx
                        total += intensities[idx]
                    clusters.append(
                        Cluster(
                            charge=charge,
                            peak_indices=tuple(chain),
                            mask=mask,
                            explained_intensity=total,
                        )
                    )
                if len(chain) == MAX_CLUSTER_SIZE:
                    return
                for nxt in adjacency[chain[-1]]:
                    chain.append(nxt)
                    visit()
                    chain.pop()

            for start in range(n):
                chain.append(start)
                visit()
                chain.pop()
        clusters.sort(key=lambda c: (c.peak_indices[0], c.charge, c.peak_indices))
        return clusters

    # ------------------------------------------------------------------ #
    # Exhaustive optimisation (exact DP over used-peak bitmasks)
    # ------------------------------------------------------------------ #

    def _best(self, used_mask: int) -> Objective:
        """Best achievable objective over the peaks still free in ``used_mask``."""
        memo = self._best_memo
        cached = memo.get(used_mask)
        if cached is not None:
            return cached
        free = self._full_mask & ~used_mask
        if free == 0:
            result: Objective = (0, 0, 0)
        else:
            first = (free & -free).bit_length() - 1
            # Option A: leave `first` unexplained.
            best = self._best(used_mask | (1 << first))
            # Option B: cover `first` with each legal cluster that fits.
            for cluster in self._clusters_by_peak[first]:
                self._search_ops += 1
                if self._search_ops > self._max_search_ops:
                    raise SearchSpaceExceededError(
                        "exact deconvolution search exceeded the configured "
                        f"work budget ({self._max_search_ops} operations); "
                        "narrow the tolerance or the charge set"
                    )
                if cluster.mask & used_mask:
                    continue
                ci, cp, cc = self._best(used_mask | cluster.mask)
                candidate = (
                    ci + cluster.explained_intensity,
                    cp + cluster.size,
                    cc - 1,
                )
                if candidate > best:
                    best = candidate
            result = best
        memo[used_mask] = result
        return result

    # ------------------------------------------------------------------ #
    # Optimal-witness enumeration (for tie detection / second witness)
    # ------------------------------------------------------------------ #

    def _witnesses(self, used_mask: int, limit: int) -> list[tuple[Cluster, ...]]:
        """Up to ``limit`` distinct optimal solutions from ``used_mask``."""
        if limit <= 0:
            return []
        target = self._best(used_mask)
        free = self._full_mask & ~used_mask
        if free == 0:
            return [()]
        first = (free & -free).bit_length() - 1
        found: list[tuple[Cluster, ...]] = []
        skip_mask = used_mask | (1 << first)
        if self._best(skip_mask) == target:
            for tail in self._witnesses(skip_mask, limit):
                found.append(tail)
                if len(found) >= limit:
                    return found[:limit]
        for cluster in self._clusters_by_peak[first]:
            if cluster.mask & used_mask:
                continue
            ci, cp, cc = self._best(used_mask | cluster.mask)
            if (ci + cluster.explained_intensity, cp + cluster.size, cc - 1) != target:
                continue
            for tail in self._witnesses(used_mask | cluster.mask, limit - len(found)):
                found.append((cluster,) + tail)
                if len(found) >= limit:
                    return found[:limit]
        return found

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def solve(self) -> DeconvolutionResult:
        intensity, peak_count, neg_clusters = self._best(0)
        if peak_count == 0:
            return DeconvolutionResult(
                verdict=VERDICT_UNRESOLVED,
                explained_intensity=0,
                explained_peak_count=0,
                cluster_count=0,
                primary=(),
                secondary=None,
            )
        canonical = {self._canon(w) for w in self._witnesses(0, 2)}
        ordered = sorted(canonical, key=self._solution_key)
        primary = ordered[0]
        secondary = ordered[1] if len(ordered) > 1 else None
        return DeconvolutionResult(
            verdict=VERDICT_UNIQUE if secondary is None else VERDICT_AMBIGUOUS,
            explained_intensity=intensity,
            explained_peak_count=peak_count,
            cluster_count=-neg_clusters,
            primary=primary,
            secondary=secondary,
        )

    @staticmethod
    def _canon(solution: tuple[Cluster, ...]) -> tuple[Cluster, ...]:
        return tuple(sorted(solution, key=lambda c: c.canonical_key()))

    @staticmethod
    def _solution_key(solution: tuple[Cluster, ...]) -> tuple:
        return tuple((c.peak_indices, c.charge) for c in solution)
