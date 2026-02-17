from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


# ── Job Description Models ──────────────────────────────────────────────

class HardRequirements(BaseModel):
    required_skills: list[str] = Field(default_factory=list)
    min_experience_years: int = 0
    required_degree: Optional[str] = None
    required_locations: list[str] = Field(default_factory=list)
    custom_musts: list[str] = Field(default_factory=list)


class PreferredCriteria(BaseModel):
    preferred_skills: list[str] = Field(default_factory=list)
    preferred_experience: Optional[str] = None
    preferred_certifications: list[str] = Field(default_factory=list)
    custom_preferences: list[str] = Field(default_factory=list)


class JobRequirements(BaseModel):
    title: str = ""
    summary: str = ""
    seniority_level: str = ""  # e.g. "junior", "mid", "senior", "lead"
    hard: HardRequirements = Field(default_factory=HardRequirements)
    preferred: PreferredCriteria = Field(default_factory=PreferredCriteria)


# ── Candidate Models ────────────────────────────────────────────────────

class ExperienceEntry(BaseModel):
    title: str = ""
    company: str = ""
    duration_years: float = 0.0
    responsibilities: list[str] = Field(default_factory=list)


class EducationEntry(BaseModel):
    degree: str = ""
    field: str = ""
    institution: str = ""
    year: Optional[int] = None


class CandidateProfile(BaseModel):
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

class DimensionScore(BaseModel):
    score: float = 0.0        # 0-100
    evidence: str = ""        # short human-readable reason
    matched: list[str] = Field(default_factory=list)   # what matched
    missing: list[str] = Field(default_factory=list)    # what didn't


class FilterResult(BaseModel):
    passed: bool = True
    failures: list[str] = Field(default_factory=list)  # reasons for failure


class CandidateEvaluation(BaseModel):
    candidate: CandidateProfile
    filter_result: FilterResult = Field(default_factory=FilterResult)
    scores: dict[str, DimensionScore] = Field(default_factory=dict)
    composite_score: float = 0.0
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    summary: str = ""
    rank: int = 0
