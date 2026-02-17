from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from typing import Optional


class SafeBase(BaseModel):
    """Base that coerces None → default for list/dict fields (LLMs love returning null)."""

    @model_validator(mode="before")
    @classmethod
    def _coerce_nulls(cls, values):
        if not isinstance(values, dict):
            return values
        for name, field in cls.model_fields.items():
            if name in values and values[name] is None:
                if hasattr(field.annotation, "__origin__"):
                    origin = field.annotation.__origin__
                    if origin is list:
                        values[name] = []
                    elif origin is dict:
                        values[name] = {}
        return values


# ── Job Description Models ──────────────────────────────────────────────

class HardRequirements(SafeBase):
    required_skills: list[str] = Field(default_factory=list)
    min_experience_years: int = 0
    required_degree: Optional[str] = None
    required_locations: list[str] = Field(default_factory=list)
    custom_musts: list[str] = Field(default_factory=list)


class PreferredCriteria(SafeBase):
    preferred_skills: list[str] = Field(default_factory=list)
    preferred_experience: Optional[str] = None
    preferred_certifications: list[str] = Field(default_factory=list)
    custom_preferences: list[str] = Field(default_factory=list)


class JobRequirements(SafeBase):
    title: str = ""
    summary: str = ""
    seniority_level: str = ""  # e.g. "junior", "mid", "senior", "lead"
    hard: HardRequirements = Field(default_factory=HardRequirements)
    preferred: PreferredCriteria = Field(default_factory=PreferredCriteria)


# ── Candidate Models ────────────────────────────────────────────────────

class ExperienceEntry(SafeBase):
    title: str = ""
    company: str = ""
    duration_years: float = 0.0
    responsibilities: list[str] = Field(default_factory=list)


class EducationEntry(SafeBase):
    degree: str = ""
    field: str = ""
    institution: str = ""
    year: Optional[int] = None


class CandidateProfile(SafeBase):
    name: str = "Unknown"
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    total_experience_years: float = 0.0
    current_title: str = ""
    skills: list[str] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    source_file: str = ""  # original PDF filename
    raw_text: str = ""     # full extracted text (kept for re-scoring)


# ── Evaluation Models ───────────────────────────────────────────────────

class DimensionScore(SafeBase):
    score: float = 0.0        # 0-100
    evidence: str = ""        # short human-readable reason
    matched: list[str] = Field(default_factory=list)   # what matched
    missing: list[str] = Field(default_factory=list)    # what didn't


class FilterResult(SafeBase):
    passed: bool = True
    failures: list[str] = Field(default_factory=list)  # reasons for failure


class CandidateEvaluation(SafeBase):
    candidate: CandidateProfile
    filter_result: FilterResult = Field(default_factory=FilterResult)
    scores: dict[str, DimensionScore] = Field(default_factory=dict)
    composite_score: float = 0.0
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    summary: str = ""
    rank: int = 0
