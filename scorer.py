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
from utils import LLM, Embedder, skill_matches, build_search_pool, compute_skills_coverage


# ── LLM Evaluation Prompt ────────────────────────────────────────────────

EVAL_SYSTEM_PROMPT = """\
You are a strict technical hiring evaluator. You will be given a job description, a candidate's full work history (jobs AND projects merged together), and a PRE-COMPUTED skills checklist.

Your job is to evaluate the candidate's DEPTH of skill usage and the relevance of their TOTAL WORK (jobs + projects combined) to this specific role.

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

═══ DIMENSION 1: skills_depth (How DEEPLY does the candidate demonstrate required skills?) ═══

IMPORTANT: You are given a pre-computed checklist of which required skills are PRESENT or MISSING.
Use it as ground truth. Do NOT override it. A candidate missing 5 out of 9 required skills CANNOT score above 40.

Measure these 3 factors (coverage is already computed separately):

A. DEMONSTRATED USE (0-35): Are the PRESENT skills backed by real work, or just listed?
   - 35 = every present skill appears in [JOB] or [PROJECT] entries with specific outcomes
   - 20 = most present skills mentioned in context of actual work
   - 10 = skills are only listed in a skills section with no supporting work evidence
   - 0 = skills not demonstrated at all

B. DEPTH (0-35): For the skills that ARE present, does usage show deep knowledge?
   - 35 = advanced patterns (architecture decisions, performance optimization, scaling)
   - 20 = solid intermediate use (built features end-to-end, integrated systems)
   - 10 = basic use (tutorials, simple CRUD, academic coursework only)
   - 0 = no evidence of actual hands-on use

C. RECENCY (0-30): Were the present skills used in recent work?
   - 30 = used in current/most recent [JOB] or recent [PROJECT]
   - 20 = used in last 2-3 years
   - 10 = used only in older roles (3+ years ago)
   - 0 = cannot determine when used

CRITICAL: If the skills checklist shows many required skills MISSING, this dimension MUST score low.

═══ DIMENSION 2: project_relevance (How relevant is the candidate's TOTAL WORK to THIS role?) ═══

You are given a merged list of ALL the candidate's work, tagged as [JOB] or [PROJECT].
Evaluate ALL of them together — jobs AND projects both count equally.

Measure these 4 factors, then average into one score:

A. DOMAIN MATCH (0-25): Is their work in a similar domain as this role?
   - 25 = same domain (e.g. both are web apps, both are SaaS platforms)
   - 15 = related domain
   - 5 = different but transferable
   - 0 = completely unrelated

B. TECH STACK OVERLAP (0-25): Does their work use the technologies this JD requires?
   Look at BOTH [JOB] responsibilities AND [PROJECT] technologies.
   - 25 = work uses most/all of the required tech stack
   - 15 = work uses some of the required tech stack
   - 5 = different but related technologies
   - 0 = no overlap at all

C. COMPLEXITY & SCALE (0-25): How complex is their demonstrated work?
   - 25 = production systems, multi-service architectures, real users
   - 15 = complete applications with multiple features, deployment
   - 5 = simple apps, single-feature projects, coursework
   - 0 = no substantial work found

D. IMPACT & OWNERSHIP (0-25): Did they build/own/lead, or just participate?
   - 25 = quantified impact, led development, owned systems end-to-end
   - 15 = clear contributions described but not quantified
   - 5 = participation described vaguely
   - 0 = no impact described

CRITICAL RULES:
- If the candidate has NO [JOB] entries AND NO [PROJECT] entries, score 0-10.
- If work exists but uses NONE of the required technologies, score 10-30.
- If some work uses some required tech, score 30-60.
- If multiple items closely match the role's tech stack and domain, score 60-85.
- Only score 85+ if work demonstrates production-level systems with the exact required stack.

═══ DIMENSION 3: overall_fit (Holistic tiebreaker) ═══

Consider everything together — skills coverage, depth, work relevance, experience trajectory, preferred criteria, certifications.
Score 0-100. This should correlate with the other scores — do NOT give a high overall_fit if skills_depth and project_relevance are low.

═══ RULES ═══
- Base scores ONLY on what is explicitly written. Do NOT assume or infer.
- If a required skill is MISSING from the checklist, treat it as genuinely absent.
- If a skill is only listed but never demonstrated in [JOB] or [PROJECT], score DEMONSTRATED USE as 10, not 35.
- Be harsh and consistent.
- strengths: list 2-4 specific things for THIS role. Cite actual [JOB] or [PROJECT] entries.
- gaps: list 2-4 specific things missing or weak for THIS role.
"""


def _build_eval_prompt(
    candidate: CandidateProfile,
    job: JobRequirements,
    matched_skills: list[str],
    missing_skills: list[str],
) -> str:
    """Build the user prompt for LLM evaluation.
    Merges experience + projects into one unified 'All Work' section so the LLM
    can evaluate project_relevance based on everything the candidate has done."""

    # JD summary
    jd_section = f"""JOB: {job.title} ({job.seniority_level} level)
Summary: {job.summary}
Required skills: {', '.join(job.hard.required_skills) or 'None specified'}
Min experience: {job.hard.min_experience_years} years
Preferred skills: {', '.join(job.preferred.preferred_skills) or 'None'}
Preferred certs: {', '.join(job.preferred.preferred_certifications) or 'None'}"""

    # Pre-computed skills checklist
    total = len(matched_skills) + len(missing_skills)
    checklist = f"""═══ PRE-COMPUTED SKILLS CHECKLIST ({len(matched_skills)}/{total} required skills found) ═══
✅ FOUND: {', '.join(matched_skills) if matched_skills else 'None'}
❌ MISSING: {', '.join(missing_skills) if missing_skills else 'None'}
(This checklist was computed deterministically. Treat it as ground truth.)"""

    # Build unified work section — merge jobs + projects with clear tags
    work_items = []
    for exp in candidate.experience:
        header = exp.title
        if exp.company:
            header += f" at {exp.company}"
        if exp.duration_years:
            header += f" ({exp.duration_years}y)"
        bullets = "\n".join(f"  - {r}" for r in exp.responsibilities) if exp.responsibilities else "  - No details provided"
        work_items.append(f"[JOB] {header}\n{bullets}")

    for proj in candidate.projects:
        tech = f" [{', '.join(proj.technologies)}]" if proj.technologies else ""
        desc = proj.description or "No description provided"
        url = f" | {proj.url}" if proj.url else ""
        work_items.append(f"[PROJECT] {proj.name}{tech}{url}\n  - {desc}")

    work_section = "\n\n".join(work_items) if work_items else "No work experience or projects found."

    # Candidate summary
    candidate_info = f"""═══ CANDIDATE: {candidate.name} ═══
Current Title: {candidate.current_title or 'Not specified'}
Total Experience: {candidate.total_experience_years} years
Skills: {', '.join(candidate.skills) or 'None extracted'}
Education: {', '.join(f'{e.degree} {e.field} — {e.institution}' for e in candidate.education) or 'None'}
Certifications: {', '.join(candidate.certifications) or 'None'}"""

    return f"""{jd_section}

{checklist}

{candidate_info}

═══ ALL WORK (Jobs + Projects merged) ═══
{work_section}"""


def llm_evaluate(
    candidate: CandidateProfile,
    job: JobRequirements,
    llm: LLM,
    matched_skills: list[str],
    missing_skills: list[str],
) -> LLMJudgment:
    """Ask the LLM to evaluate a candidate against the JD. Returns structured judgment."""
    prompt = _build_eval_prompt(candidate, job, matched_skills, missing_skills)
    try:
        data = llm.extract_json(EVAL_SYSTEM_PROMPT, prompt)
        return LLMJudgment(**data)
    except Exception:
        return LLMJudgment()


# ── Hard Filter ──────────────────────────────────────────────────────────

def apply_hard_filter(
    candidate: CandidateProfile,
    job: JobRequirements,
    embedder: Embedder,
    skills_coverage_pct: float = 100.0,
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

    # Check required skills coverage threshold
    if skills_coverage_pct < config.REQUIRED_SKILLS_FAIL_THRESHOLD * 100:
        failures.append(
            f"Skills coverage {skills_coverage_pct:.0f}% "
            f"< minimum {config.REQUIRED_SKILLS_FAIL_THRESHOLD * 100:.0f}%"
        )

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
    DEGREE_RANK = {
        # Doctoral
        "phd": 4, "ph.d": 4, "doctorate": 4, "doctoral": 4,
        # Masters
        "master": 4, "m.s.": 3, "m.s": 3, "ms ": 3, "m.a.": 3, "m.a": 3,
        "mba": 3, "m.tech": 3, "m.e.": 3, "mca": 3, "m.sc": 3, "msc": 3,
        # Bachelors
        "bachelor": 2, "b.s.": 2, "b.s": 2, "bs ": 2, "b.a.": 2, "b.a": 2,
        "b.tech": 2, "btech": 2, "b.e.": 2, "b.e": 2, "bca": 2,
        "b.sc": 2, "bsc": 2, "b.com": 2, "bcom": 2,
        # Associate / Diploma
        "associate": 1, "diploma": 1,
        # High school
        "high school": 0, "12th": 0, "hsc": 0,
    }

    best_rank = 0
    best_degree = "None"
    for edu in candidate.education:
        deg_lower = edu.degree.lower().strip()
        for keyword, rank in DEGREE_RANK.items():
            if keyword in deg_lower and rank > best_rank:
                best_rank = rank
                best_degree = edu.degree

    # Fallback: check raw text for degree mentions if LLM missed education
    if best_rank == 0 and candidate.raw_text:
        raw_lower = candidate.raw_text.lower()
        for keyword, rank in DEGREE_RANK.items():
            if keyword in raw_lower and rank > best_rank:
                best_rank = rank
                best_degree = f"(detected from resume text: {keyword})"

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
    """Run the full evaluation: skills coverage → filter → rule scores → LLM judgment → combine."""
    weights = weights or config.DEFAULT_WEIGHTS

    evaluation = CandidateEvaluation(candidate=candidate)

    # Step 0: Deterministic skills coverage (runs before everything else)
    search_pool = build_search_pool(candidate)
    coverage_pct, matched_skills, missing_skills = compute_skills_coverage(
        job.hard.required_skills, search_pool, embedder
    )
    evaluation.scores["skills_coverage"] = DimensionScore(
        score=coverage_pct,
        evidence=f"{len(matched_skills)}/{len(matched_skills) + len(missing_skills)} required skills found",
        matched=matched_skills,
        missing=missing_skills,
    )

    # Hard filter (now includes skills coverage check)
    evaluation.filter_result = apply_hard_filter(candidate, job, embedder, coverage_pct)
    if not evaluation.filter_result.passed:
        evaluation.summary = "Filtered out: " + "; ".join(evaluation.filter_result.failures)
        return evaluation

    # Rule-based scores
    evaluation.scores.update({
        "experience": score_experience(candidate, job),
        "seniority":  score_seniority(candidate, job),
        "education":  score_education(candidate, job),
    })

    # LLM judgment (receives pre-computed skills checklist to ground its evaluation)
    judgment = llm_evaluate(candidate, job, llm, matched_skills, missing_skills)
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
