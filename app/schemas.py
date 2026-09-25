"""Request/response schemas for the versioned deconvolution API."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

# Strict: JSON floats such as 1.5 or numeric strings must not be accepted
# where a positive integer is required.
StrictPositiveInt = Annotated[int, Field(strict=True, gt=0)]


class PeakInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mz: Decimal = Field(description="Mass-to-charge ratio; finite decimal > 0.")
    intensity: StrictPositiveInt = Field(description="Positive integer intensity.")

    @field_validator("mz")
    @classmethod
    def _mz_finite_positive(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("mz must be a finite decimal number")
        if value <= 0:
            raise ValueError("mz must be greater than 0")
        return value


class _PeaksAndCharges(BaseModel):
    """Shared peak-list / charge-set validation for all analysis requests."""

    model_config = ConfigDict(extra="forbid")

    peaks: list[PeakInput] = Field(
        min_length=2,
        max_length=36,
        description="2 to 36 peaks, strictly increasing in mz.",
    )
    charges: list[StrictPositiveInt] = Field(
        min_length=1,
        description="Allowed charge states (a set: positive, unique integers).",
    )

    @field_validator("charges")
    @classmethod
    def _charges_unique(cls, value: list[int]) -> list[int]:
        if len(set(value)) != len(value):
            raise ValueError("charges must be a set: duplicate values are not allowed")
        return value

    @field_validator("peaks")
    @classmethod
    def _peaks_strictly_increasing(cls, value: list[PeakInput]) -> list[PeakInput]:
        for i in range(1, len(value)):
            if value[i].mz <= value[i - 1].mz:
                raise ValueError(
                    "peaks must be strictly increasing in mz: "
                    f"peaks[{i}].mz={value[i].mz} is not greater than "
                    f"peaks[{i - 1}].mz={value[i - 1].mz}"
                )
        return value


class DeconvolutionRequest(_PeaksAndCharges):
    tolerance: Decimal = Field(
        description="Non-negative decimal m/z tolerance applied to 1.003355/z."
    )

    @field_validator("tolerance")
    @classmethod
    def _tolerance_valid(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("tolerance must be a finite decimal number")
        if value < 0:
            raise ValueError("tolerance must be greater than or equal to 0")
        return value


class ToleranceRangeInput(BaseModel):
    """Closed tolerance interval for a sensitivity scan."""

    model_config = ConfigDict(extra="forbid")

    lower: Decimal = Field(description="Range lower bound; finite decimal >= 0.")
    upper: Decimal = Field(
        description="Range upper bound; finite decimal >= lower (closed interval)."
    )

    @field_validator("lower", "upper")
    @classmethod
    def _bound_finite_non_negative(cls, value: Decimal, info: ValidationInfo) -> Decimal:
        if not value.is_finite():
            raise ValueError(f"tolerance_range.{info.field_name} must be a finite decimal number")
        if value < 0:
            raise ValueError(
                f"tolerance_range.{info.field_name} must be greater than or equal to 0"
            )
        return value

    @field_validator("upper")
    @classmethod
    def _upper_not_below_lower(cls, value: Decimal, info: ValidationInfo) -> Decimal:
        lower = info.data.get("lower")
        if lower is not None and value < lower:
            raise ValueError(
                "tolerance_range.upper must be greater than or equal to "
                "tolerance_range.lower (the range is a closed interval)"
            )
        return value


class SensitivitySpectrumRequest(_PeaksAndCharges):
    tolerance_range: ToleranceRangeInput = Field(
        description="Closed tolerance interval [lower, upper] to scan."
    )


# ---------------------------------------------------------------------- #
# Responses
# ---------------------------------------------------------------------- #


class PeakOut(BaseModel):
    index: int
    mz: str
    intensity: int


class ClusterOut(BaseModel):
    charge: int
    peak_indices: list[int]
    explained_intensity: int
    peaks: list[PeakOut]


class SolutionOut(BaseModel):
    clusters: list[ClusterOut]
    unexplained_peaks: list[PeakOut]


class ObjectivesOut(BaseModel):
    explained_intensity: int
    explained_peak_count: int
    cluster_count: int


class InputSummaryOut(BaseModel):
    peak_count: int
    charges: list[int]
    tolerance: str
    isotope_spacing: str


class DeconvolutionResponse(BaseModel):
    verdict: Literal["UNIQUE", "AMBIGUOUS", "UNRESOLVED"]
    objectives: ObjectivesOut
    clusters: list[ClusterOut]
    unexplained_peaks: list[PeakOut]
    second_witness: SolutionOut | None
    input_summary: InputSummaryOut


class AdjudicationOut(BaseModel):
    """The full deconvolution adjudication that is constant over a segment."""

    verdict: Literal["UNIQUE", "AMBIGUOUS", "UNRESOLVED"]
    objectives: ObjectivesOut
    clusters: list[ClusterOut]
    unexplained_peaks: list[PeakOut]
    second_witness: SolutionOut | None


class SpectrumSegmentOut(BaseModel):
    """One maximal tolerance region sharing a single adjudication.

    The segment covers ``[lower, upper)``; ``upper_inclusive`` is true only
    for the final segment, which closes at the requested range upper bound.
    Boundary strings are exact decimals, or rounded to 40 significant digits
    when the exact critical tolerance has no finite decimal expansion.
    """

    lower: str
    upper: str
    upper_inclusive: bool
    adjudication: AdjudicationOut


class SensitivityInputSummaryOut(BaseModel):
    peak_count: int
    charges: list[int]
    tolerance_range: dict[str, str]
    isotope_spacing: str


class SensitivitySpectrumResponse(BaseModel):
    segments: list[SpectrumSegmentOut]
    #: Distinct tolerances at which the deconvolution was recomputed.
    evaluation_point_count: int
    #: Distinct critical tolerances found strictly inside the range.
    critical_point_count: int
    input_summary: SensitivityInputSummaryOut
