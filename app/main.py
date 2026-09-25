"""FastAPI application exposing the versioned deconvolution endpoint."""

from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import __version__
from .schemas import (
    ClusterOut,
    DeconvolutionRequest,
    DeconvolutionResponse,
    InputSummaryOut,
    ObjectivesOut,
    PeakOut,
    SensitivityInputSummaryOut,
    SensitivityRequest,
    SensitivityResponse,
    SensitivitySegmentOut,
    SolutionOut,
    ToleranceRangeOut,
)
from .sensitivity import (
    SensitivityScanFailedError,
    SpectrumSegment,
    format_tolerance,
    scan_spectrum,
)
from .solver import (
    ISOTOPE_SPACING,
    Cluster,
    DeconvolutionResult,
    Deconvolver,
    Peak,
    SearchSpaceExceededError,
)

# Safety valve for pathological search spaces (see app.solver).  The search
# remains fully exhaustive below this budget.
MAX_SEARCH_OPS = int(os.environ.get("DECONVOLVER_MAX_SEARCH_OPS", "20000000"))

# Total work budget for one tolerance sensitivity scan (critical-tolerance
# derivation plus every recomputed deconvolution along the range).
MAX_SENSITIVITY_OPS = int(os.environ.get("DECONVOLVER_MAX_SENSITIVITY_OPS", "20000000"))

app = FastAPI(
    title="Isotope Peak Deconvolution Service",
    version=__version__,
    description=(
        "Deterministic, exhaustive deconvolution of overlapping isotope peak "
        "clusters for high-resolution mass spectrometry review. Objectives are "
        "optimised lexicographically: (1) maximise explained total intensity, "
        "(2) maximise explained peak count, (3) minimise cluster count."
    ),
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return field-locatable errors; invalid input never yields a verdict."""
    fields = []
    for err in exc.errors():
        loc = [str(part) for part in err.get("loc", ()) if part != "body"]
        fields.append(
            {
                "loc": ".".join(loc) if loc else "body",
                "message": err.get("msg", ""),
                "type": err.get("type", ""),
            }
        )
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid input; no deconvolution verdict was produced.",
                "fields": fields,
            }
        },
    )


@app.exception_handler(SearchSpaceExceededError)
async def search_space_exception_handler(
    request: Request, exc: SearchSpaceExceededError
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "code": "SEARCH_SPACE_EXCEEDED",
                "message": str(exc),
                "fields": [],
            }
        },
    )


@app.exception_handler(SensitivityScanFailedError)
async def sensitivity_scan_exception_handler(
    request: Request, exc: SensitivityScanFailedError
) -> JSONResponse:
    """Budget exhaustion fails the whole scan: never a partial spectrum."""
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "code": "SENSITIVITY_SCAN_FAILED",
                "message": str(exc),
                "fields": [],
            }
        },
    )


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "service": "isotope-deconvolver", "version": __version__}


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "service": "isotope-deconvolver",
        "version": __version__,
        "endpoints": {
            "deconvolve": "POST /api/v1/deconvolve",
            "sensitivity": "POST /api/v1/deconvolve/sensitivity",
            "health": "GET /health",
            "docs": "GET /docs",
        },
    }


@app.post(
    "/api/v1/deconvolve",
    response_model=DeconvolutionResponse,
    tags=["v1"],
    summary="Deterministically deconvolve overlapping isotope peaks",
)
def deconvolve(payload: DeconvolutionRequest) -> DeconvolutionResponse:
    peaks = [
        Peak(index=i, mz=p.mz, intensity=p.intensity)
        for i, p in enumerate(payload.peaks)
    ]
    result = Deconvolver(
        peaks=peaks,
        charges=payload.charges,
        tolerance=payload.tolerance,
        max_search_ops=MAX_SEARCH_OPS,
    ).solve()
    return _build_response(payload, peaks, result)


@app.post(
    "/api/v1/deconvolve/sensitivity",
    response_model=SensitivityResponse,
    tags=["v1"],
    summary="Tolerance sensitivity spectrum of the deconvolution verdict",
)
def sensitivity(payload: SensitivityRequest) -> SensitivityResponse:
    """Scan a closed tolerance range and report where the verdict changes.

    Critical tolerances are derived exactly from the peak differences and
    allowed charges; the global deconvolution is recomputed only at the
    range endpoints and those critical points.  Adjacent segments with
    identical normalised verdicts are merged.
    """
    peaks = [
        Peak(index=i, mz=p.mz, intensity=p.intensity)
        for i, p in enumerate(payload.peaks)
    ]
    charges = sorted(set(payload.charges))
    spectrum = scan_spectrum(
        peaks=peaks,
        charges=charges,
        lower=payload.tolerance_range.lower,
        upper=payload.tolerance_range.upper,
        max_total_ops=MAX_SENSITIVITY_OPS,
    )
    segments = [
        _segment_out(segment, peaks) for segment in spectrum.segments
    ]
    return SensitivityResponse(
        segments=segments,
        critical_tolerances=[
            format_tolerance(t) for t in spectrum.critical_tolerances
        ],
        evaluation_count=spectrum.evaluation_count,
        input_summary=SensitivityInputSummaryOut(
            peak_count=len(peaks),
            charges=charges,
            tolerance_range=ToleranceRangeOut(
                lower=str(payload.tolerance_range.lower),
                upper=str(payload.tolerance_range.upper),
            ),
            isotope_spacing=str(ISOTOPE_SPACING),
        ),
    )


def _segment_out(segment: SpectrumSegment, peaks: list[Peak]) -> SensitivitySegmentOut:
    primary = _solution_out(segment.result.primary, peaks)
    secondary = (
        _solution_out(segment.result.secondary, peaks)
        if segment.result.secondary is not None
        else None
    )
    return SensitivitySegmentOut(
        lower=format_tolerance(segment.lower),
        upper=format_tolerance(segment.upper),
        upper_inclusive=segment.upper_inclusive,
        verdict=segment.result.verdict,
        objectives=ObjectivesOut(
            explained_intensity=segment.result.explained_intensity,
            explained_peak_count=segment.result.explained_peak_count,
            cluster_count=segment.result.cluster_count,
        ),
        clusters=primary.clusters,
        unexplained_peaks=primary.unexplained_peaks,
        second_witness=secondary,
    )


def _build_response(
    payload: DeconvolutionRequest,
    peaks: list[Peak],
    result: DeconvolutionResult,
) -> DeconvolutionResponse:
    primary = _solution_out(result.primary, peaks)
    secondary = (
        _solution_out(result.secondary, peaks) if result.secondary is not None else None
    )
    return DeconvolutionResponse(
        verdict=result.verdict,
        objectives=ObjectivesOut(
            explained_intensity=result.explained_intensity,
            explained_peak_count=result.explained_peak_count,
            cluster_count=result.cluster_count,
        ),
        clusters=primary.clusters,
        unexplained_peaks=primary.unexplained_peaks,
        second_witness=secondary,
        input_summary=InputSummaryOut(
            peak_count=len(peaks),
            charges=sorted(set(payload.charges)),
            tolerance=str(payload.tolerance),
            isotope_spacing=str(ISOTOPE_SPACING),
        ),
    )


def _solution_out(clusters: tuple[Cluster, ...], peaks: list[Peak]) -> SolutionOut:
    explained: set[int] = set()
    out_clusters: list[ClusterOut] = []
    for cluster in clusters:
        explained.update(cluster.peak_indices)
        out_clusters.append(
            ClusterOut(
                charge=cluster.charge,
                peak_indices=list(cluster.peak_indices),
                explained_intensity=cluster.explained_intensity,
                peaks=[_peak_out(peaks[i]) for i in cluster.peak_indices],
            )
        )
    unexplained = [_peak_out(p) for p in peaks if p.index not in explained]
    return SolutionOut(clusters=out_clusters, unexplained_peaks=unexplained)


def _peak_out(peak: Peak) -> PeakOut:
    return PeakOut(index=peak.index, mz=str(peak.mz), intensity=peak.intensity)
