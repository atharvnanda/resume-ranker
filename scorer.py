"""
scorer.py — Hard-requirement filtering, hybrid scoring (rule-based + LLM-as-judge), and ranking.
"""
from __future__ import annotations

import config
from models import (
    JobRequirements,
    CandidateProfile,
    CandidateEvaluation,
    DimensionScore,
    FilterResult,
    LLMJudgment,
)
from utils import LLM, Embedder, skill_matches, build_search_pool


# ── LLM Evaluation Prompt ────────────────────────────────────────────────

EVAL_SYSTEM_PROMPT = """\
You are a strict technical hiring evaluator. You will be given a job description and a candidate's resume.
Evaluate the candidate on EXACTLY these three dimensions using the fixed rubrics below.

Return a JSON object with this exact structure — nothing else:
{
  "skills_depth": {
    "score": 0-100,
    "evidence": "2-3 sentence justification"
  },
  "project_relevance": {
    "score": 0-100,
    "evidence": "2-3 sentence justification"
  },
  "overall_fit": {
    "score": 0-100,
    "evidence": "2-3 sentence justification"
  },
  "strengths": ["strength1", "strength2"],
  "gaps": ["gap1", "gap2"]
}

═══ DIMENSION 1: skills_depth (How well does the candidate ACTUALLY know the required skills?) ═══

Measure these 4 factors, then average them into one score:

A. COVERAGE (0-25): What fraction of required skills appear in their resume?
   - 25 = all required skills present
   - 0 = none present

B. DEMONSTRATED USE (0-25): Are skills backed by real work, or just listed?
   - 25 = every skill appears in project/work descriptions with specific outcomes
   - 15 = most skills mentioned in context of actual work
   - 5 = skills are only listed in a skills section with no supporting work
   - 0 = skills not present at all

C. DEPTH (0-25): Does their usage show surface-level or deep knowledge?
   - 25 = advanced patterns (architecture decisions, optimization, scaling, mentoring others)
   - 15 = solid intermediate use (built features, integrated systems, solved problems)
   - 5 = basic use (tutorials, simple CRUD, coursework only)
   - 0 = no evidence of actual use

D. RECENCY (0-25): Were skills used in recent roles or only years ago?
   - 25 = used in current/most recent role
   - 15 = used in last 2-3 years
   - 5 = used only in older roles (3+ years ago)
   - 0 = cannot determine when used

═══ DIMENSION 2: project_relevance (How relevant are their projects to THIS specific job?) ═══

Measure these 4 factors, then average them into one score:

A. DOMAIN MATCH (0-25): Does their project domain match the job's domain?
   - 25 = same domain (e.g. both are fintech, both are e-commerce)
   - 15 = related domain
   - 5 = different but transferable
   - 0 = completely unrelated

B. TECH STACK OVERLAP (0-25): Do their projects use the same technologies the job requires?
   - 25 = projects use most/all of the required tech stack
   - 15 = projects use some of the required tech stack
   - 5 = projects use different but related technologies
   - 0 = no overlap

C. COMPLEXITY (0-25): How complex are their projects?
   - 25 = production systems, multi-service architectures, real user traffic
   - 15 = complete applications with multiple features, CI/CD, deployment
   - 5 = simple apps, single-feature projects, coursework
   - 0 = no projects found

D. IMPACT (0-25): Did the projects produce measurable results?
   - 25 = quantified impact (users served, performance improved, revenue generated)
   - 15 = clear impact described but not quantified
   - 5 = impact unclear or minimal
   - 0 = no impact described

═══ DIMENSION 3: overall_fit (Holistic assessment as a tiebreaker) ═══

Consider everything together — skills, projects, experience trajectory, preferred/nice-to-have criteria, certifications.
Score 0-100 for how well this candidate would perform in this specific role.

═══ RULES ═══
- Base scores ONLY on what is explicitly written in the resume. Do NOT assume or infer skills not mentioned.
- If a skill is only listed but never demonstrated in projects or work, score it under DEMONSTRATED USE as 5, not 25.
- Be consistent: two candidates with identical evidence must get identical scores.
- strengths: list 2-4 specific things that make this candidate stand out for THIS role.
- gaps: list 2-4 specific things missing or weak relative to THIS role's requirements.
"""


def _build_eval_prompt(candidate: CandidateProfile, job: JobRequirements) -> str:
    """Build the user prompt for LLM evaluation."""
    # Format JD requirements concisely
    jd_section = f"""JOB: {job.title} ({job.seniority_level} level)
Summary: {job.summary}
Required skills: {', '.join(job.hard.required_skills) or 'None specified'}
Min experience: {job.hard.min_experience_years} years
Preferred skills: {', '.join(job.preferred.preferred_skills) or 'None'}
Preferred certs: {', '.join(job.preferred.preferred_certifications) or 'None'}"""

    # Send the full resume text — let the LLM read everything
    return f"""{jd_section}

═══ CANDIDATE RESUME ═══
{candidate.raw_text}"""


def llm_evaluate(
    candidate: CandidateProfile, job: JobRequirements, llm: LLM
) -> LLMJudgment:
    """Ask the LLM to evaluate a candidate against the JD. Returns structured judgment."""
    prompt = _build_eval_prompt(candidate, job)
    try:
        data = llm.extract_json(EVAL_SYSTEM_PROMPT, prompt)
        return LLMJudgment(**data)
    except Exception:
        return LLMJudgment()


# ── Hard Filter ──────────────────────────────────────────────────────────

def apply_hard_filter(
    candidate: CandidateProfile, job: JobRequirements, embedder: Embedder
) -> FilterResult:
    """Check if candidate meets ALL hard requirements. Fail fast."""
    failures: list[str] = []

    if candidate.name in ("UNPARSEABLE", "PARSE_ERROR"):
        return FilterResult(passed=False, failures=["Resume could not be parsed"])

    if job.hard.min_experience_years > 0:
        if candidate.total_experience_years < job.hard.min_experience_years:
            failures.append(
                f"Experience {candidate.total_experience_years:.1f}y "
                f"< required {job.hard.min_experience_years}y"
            )

    if job.hard.required_degree:
        deg_req = job.hard.required_degree.lower()
        has_degree = any(
            deg_req in e.degree.lower() or deg_req in e.field.lower()
            for e in candidate.education
        )
        if not has_degree:
            failures.append(f"Missing required degree: {job.hard.required_degree}")

    for must in job.hard.custom_musts:
        search_pool = (
            candidate.skills
            + candidate.certifications
            + [resp for exp in candidate.experience for resp in exp.responsibilities]
        )
        if not skill_matches(must, search_pool, embedder):
            must_words = must.lower().split()
            key_terms = [w for w in must_words if len(w) > 3]
            raw_lower = candidate.raw_text.lower()
            if not any(term in raw_lower for term in key_terms):
                failures.append(f"Missing hard requirement: {must}")

    return FilterResult(passed=len(failures) == 0, failures=failures)


# ── Rule-Based Scorers (deterministic) ───────────────────────────────────

def score_experience(candidate: CandidateProfile, job: JobRequirements) -> DimensionScore:
    """Score based on years of experience relative to the requirement."""
    required = job.hard.min_experience_years
    actual = candidate.total_experience_years

    if required == 0:
        score = min(actual * 20, 100.0)
    else:
        score = min((actual / required) * 100, 100.0)

    evidence = f"{actual:.1f} years (required: {required})"
    return DimensionScore(score=score, evidence=evidence)


def score_seniority(candidate: CandidateProfile, job: JobRequirements) -> DimensionScore:
    """Score based on title alignment to the target seniority level."""
    target = job.seniority_level.lower()
    if not target:
        return DimensionScore(score=100.0, evidence="No seniority level specified")

    LEVEL_MAP = {"intern": 0, "junior": 1, "mid": 2, "senior": 3, "lead": 4, "principal": 5, "executive": 5}
    target_level = LEVEL_MAP.get(target, 2)

    title_lower = candidate.current_title.lower()
    candidate_level = 2
    for keyword, level in LEVEL_MAP.items():
        if keyword in title_lower:
            candidate_level = level
            break

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


# ── Full Evaluation Pipeline ─────────────────────────────────────────────

def evaluate_candidate(
    candidate: CandidateProfile,
    job: JobRequirements,
    embedder: Embedder,
    llm: LLM,
    weights: dict[str, float] | None = None,
) -> CandidateEvaluation:
    """Run the full evaluation: filter → rule scores → LLM judgment → combine."""
    weights = weights or config.DEFAULT_WEIGHTS

    evaluation = CandidateEvaluation(candidate=candidate)

    # Hard filter
    evaluation.filter_result = apply_hard_filter(candidate, job, embedder)
    if not evaluation.filter_result.passed:
        evaluation.summary = "Filtered out: " + "; ".join(evaluation.filter_result.failures)
        return evaluation

    # Rule-based scores
    evaluation.scores = {
        "experience": score_experience(candidate, job),
        "seniority":  score_seniority(candidate, job),
        "education":  score_education(candidate, job),
    }

    # LLM judgment
    judgment = llm_evaluate(candidate, job, llm)
    evaluation.llm_judgment = judgment
    evaluation.scores["skills_depth"] = judgment.skills_depth
    evaluation.scores["project_relevance"] = judgment.project_relevance
    evaluation.scores["overall_fit"] = judgment.overall_fit

    # Composite score (weighted average)
    total_weight = sum(weights.get(dim, 0) for dim in evaluation.scores)
    if total_weight > 0:
        evaluation.composite_score = sum(
            evaluation.scores[dim].score * (weights.get(dim, 0) / total_weight)
            for dim in evaluation.scores
        )

    # Strengths & gaps — merge rule-based + LLM
    evaluation.strengths = list(judgment.strengths)
    evaluation.gaps = list(judgment.gaps)
    for dim in ("experience", "seniority", "education"):
        ds = evaluation.scores[dim]
        if ds.score >= 80:
            evaluation.strengths.append(f"{dim}: {ds.evidence}")
        elif ds.score < 50:
            evaluation.gaps.append(f"{dim}: {ds.evidence}")

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
    llm: LLM,
    weights: dict[str, float] | None = None,
) -> list[CandidateEvaluation]:
    """Evaluate and rank all candidates. Filtered-out candidates go to the bottom."""
    evaluations = [
        evaluate_candidate(c, job, embedder, llm, weights) for c in candidates
    ]

    passed = [e for e in evaluations if e.filter_result.passed]
    failed = [e for e in evaluations if not e.filter_result.passed]

    passed.sort(key=lambda e: e.composite_score, reverse=True)

    ranked = passed + failed
    for i, ev in enumerate(ranked, 1):
        ev.rank = i

    return ranked
