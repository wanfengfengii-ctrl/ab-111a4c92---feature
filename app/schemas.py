"""Request/response schemas for the versioned deconvolution API."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

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


class PeaksChargesInput(BaseModel):
    """Shared base: the peak list and the allowed charge states."""

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


class DeconvolutionRequest(PeaksChargesInput):
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
    """Closed tolerance interval ``[lower, upper]`` for a sensitivity scan."""

    model_config = ConfigDict(extra="forbid")

    lower: Decimal = Field(
        description="Inclusive lower bound; finite decimal >= 0."
    )
    upper: Decimal = Field(
        description="Inclusive upper bound; finite decimal >= 0."
    )

    @field_validator("lower", "upper")
    @classmethod
    def _bound_valid(cls, value: Decimal, info: ValidationInfo) -> Decimal:
        if not value.is_finite():
            raise ValueError(f"{info.field_name} must be a finite decimal number")
        if value < 0:
            raise ValueError(f"{info.field_name} must be greater than or equal to 0")
        return value

    @model_validator(mode="after")
    def _range_ordered(self) -> "ToleranceRangeInput":
        if self.lower > self.upper:
            raise ValueError(
                f"lower ({self.lower}) must not exceed upper ({self.upper})"
            )
        return self


class SensitivityRequest(PeaksChargesInput):
    tolerance_range: ToleranceRangeInput = Field(
        description=(
            "Closed tolerance interval to scan; the verdict spectrum is "
            "recomputed at its endpoints and at every critical tolerance "
            "derived from the peak differences inside it."
        )
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


# ---------------------------------------------------------------------- #
# Sensitivity spectrum
# ---------------------------------------------------------------------- #


class SensitivitySegmentOut(BaseModel):
    """One tolerance interval on which the normalised verdict is constant."""

    lower: str = Field(
        description=(
            "Inclusive lower boundary; exact decimal string, or 'p/q' when "
            "the critical tolerance is a non-terminating decimal."
        )
    )
    upper: str = Field(description="Upper boundary; same exact format as lower.")
    upper_inclusive: bool = Field(
        description=(
            "Whether the upper boundary belongs to this segment. Only the "
            "final segment (ending at the closed range end) is inclusive."
        )
    )
    verdict: Literal["UNIQUE", "AMBIGUOUS", "UNRESOLVED"]
    objectives: ObjectivesOut
    clusters: list[ClusterOut]
    unexplained_peaks: list[PeakOut]
    second_witness: SolutionOut | None


class ToleranceRangeOut(BaseModel):
    lower: str
    upper: str


class SensitivityInputSummaryOut(BaseModel):
    peak_count: int
    charges: list[int]
    tolerance_range: ToleranceRangeOut
    isotope_spacing: str


class SensitivityResponse(BaseModel):
    segments: list[SensitivitySegmentOut] = Field(
        description=(
            "Verdict-constant tolerance intervals, sorted by increasing "
            "tolerance; adjacent intervals with identical normalised "
            "verdicts are merged."
        )
    )
    critical_tolerances: list[str] = Field(
        description=(
            "Every critical tolerance inside the scanned range (i.e. in "
            "(lower, upper]) at which the legal cluster set changes; each "
            "triggered a recomputation of the global deconvolution."
        )
    )
    evaluation_count: int = Field(
        description=(
            "Number of full deconvolutions recomputed: the range's lower "
            "endpoint plus one per in-range critical tolerance (never a "
            "fixed-step sample)."
        )
    )
    input_summary: SensitivityInputSummaryOut
