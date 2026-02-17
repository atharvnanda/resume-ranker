"""
parsers.py — PDF text extraction and LLM-based structured extraction.
Handles both Job Descriptions (pasted text) and Resume PDFs.
"""
from __future__ import annotations
import pdfplumber

import config
from models import JobRequirements, CandidateProfile
from utils import LLM


# ── PDF Text Extraction ─────────────────────────────────────────────────

def extract_text_from_pdf(file) -> str:
    """
    Extract text from an uploaded PDF file object (Streamlit UploadedFile).
    Returns raw text or empty string on failure.
    """
    try:
        with pdfplumber.open(file) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages).strip()
    except Exception:
        return ""


# ── LLM Prompts ─────────────────────────────────────────────────────────

JD_SYSTEM_PROMPT = """\
You are a precise job-description parser.  
Extract the following fields as a JSON object — nothing else.

{
  "title": "job title",
  "summary": "1-2 sentence summary of the role",
  "seniority_level": "junior | mid | senior | lead | executive",
  "hard": {
    "required_skills": ["skill1", "skill2"],
    "min_experience_years": 0,
    "required_degree": "degree or null",
    "required_locations": ["location1"],
    "custom_musts": ["any other hard requirements"]
  },
  "preferred": {
    "preferred_skills": ["skill1"],
    "preferred_experience": "description or null",
    "preferred_certifications": ["cert1"],
    "custom_preferences": ["any other nice-to-haves"]
  }
}

CRITICAL routing rules — follow these exactly:
- Any technology, tool, language, framework, or technical concept → required_skills (e.g. "React.js", "system design", "REST APIs", "Git").
- Any mention of years of experience → min_experience_years (extract the number only). Do NOT put experience requirements in custom_musts.
- Any degree requirement → required_degree.
- custom_musts is ONLY for non-skill, non-experience, non-degree requirements like "must have work authorization" or "must pass background check". If nothing fits, leave it as an empty list.
- Only include what is explicitly stated; don't invent requirements.
- If a field isn't mentioned, use the default (empty list, 0, null).
- Skills should be individual items (split "Python and Java" into two).
"""

RESUME_SYSTEM_PROMPT = """\
You are a precise resume parser.  
Given raw resume text, extract the following as a JSON object — nothing else.

{
  "name": "full name",
  "email": "email or null",
  "phone": "phone or null",
  "location": "city/state/country or null",
  "total_experience_years": 0.0,
  "current_title": "most recent job title",
  "skills": ["skill1", "skill2"],
  "experience": [
    {
      "title": "job title",
      "company": "company name",
      "duration_years": 1.5,
      "responsibilities": ["did X", "built Y"]
    }
  ],
  "education": [
    {
      "degree": "B.S.",
      "field": "Computer Science",
      "institution": "MIT",
      "year": 2020
    }
  ],
  "certifications": ["AWS SAA", "PMP"]
}

CRITICAL rules for skill extraction:
- Extract EVERY technology, tool, language, framework, library, database, concept, and methodology mentioned ANYWHERE in the resume — not just the skills section.
- Scan the summary, experience bullets, project descriptions, and education for skills.
- Include both specific tools AND general concepts (e.g. "MySQL", "SQL", "system design", "API design", "REST APIs", "authentication", "microservices", "Redux", "Docker").
- If a responsibility says "built REST APIs with JWT authentication", extract: "REST APIs", "JWT", "authentication".
- If it says "used Redux for state management", extract: "Redux", "state management".
- Do NOT collapse related skills — list them individually (e.g. list both "SQL" and "MySQL" if both are implied).

Other rules:
- Calculate total_experience_years by summing durations. Estimate if only dates given.
- If data is missing or ambiguous, use the default (null, 0, empty list).
- Do NOT fabricate information not present in the text.
"""


# ── Parsing Functions ────────────────────────────────────────────────────

def parse_job_description(jd_text: str, llm: LLM) -> JobRequirements:
    """Parse a pasted job description into structured requirements."""
    if not jd_text.strip():
        return JobRequirements()

    data = llm.extract_json(JD_SYSTEM_PROMPT, jd_text.strip())
    return JobRequirements(**data)


def parse_resume(file, llm: LLM) -> CandidateProfile:
    """
    Extract text from a PDF, then parse it into a structured profile.
    Handles edge cases: empty PDFs, files too short, parse failures.
    """
    filename = getattr(file, "name", "unknown.pdf")

    # Extract raw text
    raw_text = extract_text_from_pdf(file)

    if len(raw_text) < config.MIN_RESUME_TEXT_LENGTH:
        return CandidateProfile(
            name="UNPARSEABLE",
            source_file=filename,
            raw_text=raw_text,
        )

    # LLM extraction
    try:
        data = llm.extract_json(RESUME_SYSTEM_PROMPT, raw_text)
        profile = CandidateProfile(**data)
    except Exception:
        profile = CandidateProfile(name="PARSE_ERROR")

    profile.source_file = filename
    profile.raw_text = raw_text
    return profile
