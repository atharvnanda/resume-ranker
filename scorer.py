"""
scorer.py — Hard-requirement filtering, multi-dimension scoring, and ranking.
Pure logic — no UI, no IO.
"""
from __future__ import annotations

import config
from models import (
    JobRequirements,
    CandidateProfile,
    CandidateEvaluation,
    DimensionScore,
    FilterResult,
)
from utils import Embedder, skill_matches


# ── Hard Filter ──────────────────────────────────────────────────────────

def apply_hard_filter(candidate: CandidateProfile, job: JobRequirements) -> FilterResult:
    """Check if candidate meets ALL hard requirements. Fail fast."""
    failures: list[str] = []

    # Unparseable resume
    if candidate.name in ("UNPARSEABLE", "PARSE_ERROR"):
        return FilterResult(passed=False, failures=["Resume could not be parsed"])

    # Minimum experience
    if job.hard.min_experience_years > 0:
        if candidate.total_experience_years < job.hard.min_experience_years:
            failures.append(
                f"Experience {candidate.total_experience_years:.1f}y "
                f"< required {job.hard.min_experience_years}y"
            )

    # Required degree
    if job.hard.required_degree:
        deg_req = job.hard.required_degree.lower()
        has_degree = any(
            deg_req in e.degree.lower() or deg_req in e.field.lower()
            for e in candidate.education
        )
        if not has_degree:
            failures.append(f"Missing required degree: {job.hard.required_degree}")

    # Custom musts (simple keyword check in raw text)
    for must in job.hard.custom_musts:
        if must.lower() not in candidate.raw_text.lower():
            failures.append(f"Missing hard requirement: {must}")

    return FilterResult(passed=len(failures) == 0, failures=failures)


# ── Dimension Scorers ────────────────────────────────────────────────────

def score_skills(
    candidate: CandidateProfile,
    job: JobRequirements,
    embedder: Embedder,
) -> DimensionScore:
    """Score based on how many required skills the candidate has."""
    required = job.hard.required_skills
    if not required:
        return DimensionScore(score=100.0, evidence="No required skills specified")

    matched, missing = [], []
    for skill in required:
        if skill_matches(skill, candidate.skills, embedder):
            matched.append(skill)
        else:
            missing.append(skill)

    score = (len(matched) / len(required)) * 100
    evidence = f"{len(matched)}/{len(required)} required skills matched"
    return DimensionScore(score=score, evidence=evidence, matched=matched, missing=missing)


def score_experience(candidate: CandidateProfile, job: JobRequirements) -> DimensionScore:
    """Score based on years of experience relative to the requirement."""
    required = job.hard.min_experience_years
    actual = candidate.total_experience_years

    if required == 0:
        # No requirement — give credit for having any experience, cap at 100
        score = min(actual * 20, 100.0)  # 5+ years = 100
    else:
        ratio = actual / required
        score = min(ratio * 100, 100.0)

    evidence = f"{actual:.1f} years (required: {required})"
    return DimensionScore(score=score, evidence=evidence)


def score_seniority(candidate: CandidateProfile, job: JobRequirements) -> DimensionScore:
    """Score based on title alignment to the target seniority level."""
    target = job.seniority_level.lower()
    if not target:
        return DimensionScore(score=100.0, evidence="No seniority level specified")

    LEVEL_MAP = {"intern": 0, "junior": 1, "mid": 2, "senior": 3, "lead": 4, "principal": 5, "executive": 5}

    target_level = LEVEL_MAP.get(target, 2)

    # Guess candidate level from their title keywords
    title_lower = candidate.current_title.lower()
    candidate_level = 2  # default to mid
    for keyword, level in LEVEL_MAP.items():
        if keyword in title_lower:
            candidate_level = level
            break

    # Boost with experience heuristic
    if candidate.total_experience_years >= 10:
        candidate_level = max(candidate_level, 3)
    elif candidate.total_experience_years >= 5:
        candidate_level = max(candidate_level, 2)

    diff = abs(target_level - candidate_level)
    score = max(100 - diff * 25, 0)
    evidence = f"Target: {target}, Candidate title: {candidate.current_title}"
    return DimensionScore(score=float(score), evidence=evidence)


def score_education(candidate: CandidateProfile, job: JobRequirements) -> DimensionScore:
    """Score based on education level."""
    DEGREE_RANK = {"phd": 4, "doctorate": 4, "master": 3, "m.s.": 3, "m.a.": 3, "mba": 3,
                   "bachelor": 2, "b.s.": 2, "b.a.": 2, "b.tech": 2, "b.e.": 2,
                   "associate": 1, "diploma": 1, "high school": 0}

    best_rank = 0
    best_degree = "None"
    for edu in candidate.education:
        deg_lower = edu.degree.lower()
        for keyword, rank in DEGREE_RANK.items():
            if keyword in deg_lower and rank > best_rank:
                best_rank = rank
                best_degree = edu.degree

    score = min(best_rank * 33.3, 100.0)
    evidence = f"Highest: {best_degree}" if best_rank > 0 else "No education found"
    return DimensionScore(score=score, evidence=evidence)


def score_preferred(
    candidate: CandidateProfile,
    job: JobRequirements,
    embedder: Embedder,
) -> DimensionScore:
    """Score based on preferred/nice-to-have criteria."""
    all_preferred = (
        job.preferred.preferred_skills
        + job.preferred.preferred_certifications
    )
    if not all_preferred:
        return DimensionScore(score=100.0, evidence="No preferred criteria specified")

    all_candidate = candidate.skills + candidate.certifications
    matched, missing = [], []
    for item in all_preferred:
        if skill_matches(item, all_candidate, embedder):
            matched.append(item)
        else:
            missing.append(item)

    score = (len(matched) / len(all_preferred)) * 100
    evidence = f"{len(matched)}/{len(all_preferred)} preferred criteria met"
    return DimensionScore(score=score, evidence=evidence, matched=matched, missing=missing)


# ── Full Evaluation Pipeline ─────────────────────────────────────────────

def evaluate_candidate(
    candidate: CandidateProfile,
    job: JobRequirements,
    embedder: Embedder,
    weights: dict[str, float] | None = None,
) -> CandidateEvaluation:
    """Run the full evaluation: filter → score → summarize."""
    weights = weights or config.DEFAULT_WEIGHTS

    evaluation = CandidateEvaluation(candidate=candidate)

    # Hard filter
    evaluation.filter_result = apply_hard_filter(candidate, job)
    if not evaluation.filter_result.passed:
        evaluation.summary = "Filtered out: " + "; ".join(evaluation.filter_result.failures)
        return evaluation

    # Score each dimension
    evaluation.scores = {
        "skills":     score_skills(candidate, job, embedder),
        "experience": score_experience(candidate, job),
        "seniority":  score_seniority(candidate, job),
        "education":  score_education(candidate, job),
        "preferred":  score_preferred(candidate, job, embedder),
    }

    # Composite score (weighted average)
    total_weight = sum(weights.values())
    evaluation.composite_score = sum(
        evaluation.scores[dim].score * (weights.get(dim, 0) / total_weight)
        for dim in evaluation.scores
    )

    # Strengths & gaps
    for dim, ds in evaluation.scores.items():
        if ds.score >= 80:
            evaluation.strengths.append(f"{dim}: {ds.evidence}")
        elif ds.score < 50:
            evaluation.gaps.append(f"{dim}: {ds.evidence}")

    # Summary
    evaluation.summary = (
        f"Score: {evaluation.composite_score:.1f}/100 | "
        f"Strengths: {', '.join(evaluation.strengths) or 'None'} | "
        f"Gaps: {', '.join(evaluation.gaps) or 'None'}"
    )

    return evaluation


def rank_candidates(
    candidates: list[CandidateProfile],
    job: JobRequirements,
    embedder: Embedder,
    weights: dict[str, float] | None = None,
) -> list[CandidateEvaluation]:
    """Evaluate and rank all candidates. Filtered-out candidates go to the bottom."""
    evaluations = [
        evaluate_candidate(c, job, embedder, weights) for c in candidates
    ]

    # Sort: passed candidates by score (desc), then failed ones
    passed = [e for e in evaluations if e.filter_result.passed]
    failed = [e for e in evaluations if not e.filter_result.passed]

    passed.sort(key=lambda e: e.composite_score, reverse=True)

    ranked = passed + failed
    for i, ev in enumerate(ranked, 1):
        ev.rank = i

    return ranked
