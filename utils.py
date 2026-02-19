from __future__ import annotations
import json
import time
import logging

from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import numpy as np

import config

logger = logging.getLogger(__name__)


# ── LLM Client ──────────────────────────────────────────────────────────

class LLM:
    """Thin wrapper around Groq. Only used for structured extraction."""

    def __init__(self):
        if not config.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not found. Add it to your .env file."
            )
        self._client = Groq(api_key=config.GROQ_API_KEY)

    def extract_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Send a prompt and return parsed JSON. Retries on failure."""
        for attempt in range(config.LLM_MAX_RETRIES):
            try:
                resp = self._client.chat.completions.create(
                    model=config.GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=config.LLM_TEMPERATURE,
                    seed=config.LLM_SEED,
                )
                return json.loads(resp.choices[0].message.content)
            except json.JSONDecodeError:
                logger.warning(f"LLM returned invalid JSON (attempt {attempt + 1}/{config.LLM_MAX_RETRIES})")
                if attempt == config.LLM_MAX_RETRIES - 1:
                    raise ValueError("LLM returned invalid JSON after retries.")
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{config.LLM_MAX_RETRIES}): {e}")
                if attempt == config.LLM_MAX_RETRIES - 1:
                    raise
                time.sleep(1)
        return {}


# ── Embeddings ───────────────────────────────────────────────────────────

class Embedder:
    """Loads a sentence-transformer model once and provides similarity helpers."""

    def __init__(self):
        self._model = SentenceTransformer(config.EMBEDDING_MODEL)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of strings into embedding vectors."""
        if not texts:
            return np.array([])
        return self._model.encode(texts, normalize_embeddings=True)

    def best_match_score(self, query: str, candidates: list[str]) -> tuple[float, str]:
        """Return the highest similarity score and the best-matching candidate string."""
        if not candidates:
            return 0.0, ""
        query_emb = self.encode([query])
        cand_embs = self.encode(candidates)
        sims = cosine_similarity(query_emb, cand_embs)[0]
        best_idx = int(np.argmax(sims))
        return float(sims[best_idx]), candidates[best_idx]


# ── Skill Matching ───────────────────────────────────────────────────────

# Common skill aliases: maps a required skill to alternative phrasings
# that should also count as a match. This handles conceptual skills that
# candidates describe differently than JDs.
SKILL_ALIASES: dict[str, list[str]] = {
    "system design":        ["system architecture", "software architecture", "architectural design", "high-level design", "hld"],
    "web architecture":     ["system architecture", "software architecture", "web design patterns"],
    "rest apis":            ["restful", "rest api", "api development", "api design", "web services", "http apis"],
    "rest api":             ["restful", "rest apis", "api development", "api design", "web services", "http apis"],
    "ci/cd":                ["ci cd", "cicd", "continuous integration", "continuous deployment", "jenkins", "github actions", "gitlab ci"],
    "microservices":        ["micro services", "microservice architecture", "service-oriented", "soa"],
    "version control":      ["git", "github", "gitlab", "bitbucket", "svn"],
    "git":                  ["github", "gitlab", "bitbucket", "version control"],
    "javascript":           ["js", "es6", "es6+", "ecmascript"],
    "typescript":           ["ts"],
    "node.js":              ["nodejs", "node js", "express.js", "expressjs", "express"],
    "react.js":             ["reactjs", "react js", "react"],
    "nosql":                ["mongodb", "dynamodb", "cassandra", "redis", "firestore", "couchdb", "no-sql"],
    "sql":                  ["mysql", "postgresql", "postgres", "sqlite", "mssql", "sql server", "oracle db"],
    "containerization":     ["docker", "containers", "podman", "container"],
    "docker":               ["containerization", "containers", "dockerfile", "docker-compose"],
    "cloud platforms":      ["aws", "gcp", "azure", "cloud computing", "cloud services"],
    "aws":                  ["amazon web services", "ec2", "s3", "lambda", "cloud"],
}


def _check_aliases(required: str, candidate_skills: list[str]) -> bool:
    """Check if any alias of the required skill appears in candidate skills."""
    req_lower = required.lower().strip()
    aliases = SKILL_ALIASES.get(req_lower, [])
    if not aliases:
        return False
    for alias in aliases:
        for skill in candidate_skills:
            skill_lower = skill.lower().strip()
            if not skill_lower:
                continue
            if alias in skill_lower or skill_lower in alias:
                return True
    return False


def _substring_match(required: str, candidate_skill: str) -> bool:
    """Check if one string contains the other (catches sql ⊂ mysql, api ⊂ api design)."""
    req = required.lower().strip()
    cand = candidate_skill.lower().strip()
    # Either direction: "sql" in "mysql" or "rest api" in "rest apis"
    return req in cand or cand in req


def build_search_pool(candidate) -> list[str]:
    """
    Build a broad pool of matchable strings from a candidate profile.
    Includes skills, certifications, experience, projects, and raw text fragments.
    This ensures we don't miss skills the LLM forgot to extract.
    """
    pool = list(candidate.skills)
    pool += candidate.certifications
    pool += [exp.title for exp in candidate.experience if exp.title]
    for exp in candidate.experience:
        pool += exp.responsibilities
    # Include project technologies and descriptions
    for proj in getattr(candidate, "projects", []):
        if proj.name:
            pool.append(proj.name)
        if proj.description:
            pool.append(proj.description)
        pool += proj.technologies
    # Add raw text split into meaningful chunks (lines) as fallback
    if candidate.raw_text:
        lines = [line.strip() for line in candidate.raw_text.split("\n") if len(line.strip()) > 10]
        pool += lines
    return pool


def skill_matches(required: str, candidate_skills: list[str], embedder: Embedder) -> bool:
    """
    Check if a required skill is present in the candidate's skill list.
    Four-layer matching:
      0. Alias expansion (system design → system architecture, etc.)
      1. Substring containment (sql ⊂ mysql, api ⊂ rest api)
      2. Fuzzy string match (React vs React.js vs ReactJS)
      3. Semantic similarity fallback (CI/CD vs Jenkins pipelines)
    """
    req_lower = required.lower().strip()
    if not candidate_skills:
        return False

    # 0. Alias expansion (catches "system design" when resume says "system architecture")
    if _check_aliases(required, candidate_skills):
        return True

    for skill in candidate_skills:
        skill_lower = skill.lower().strip()
        if not skill_lower:
            continue

        # 1. Substring containment (fast, catches sql/mysql, api/rest api)
        if _substring_match(req_lower, skill_lower):
            return True

        # 2. Fuzzy string match (handles React vs React.js vs ReactJS)
        if fuzz.token_sort_ratio(req_lower, skill_lower) >= config.FUZZY_MATCH_THRESHOLD:
            return True

    # 3. Semantic similarity fallback (handles "CI/CD" vs "Jenkins pipelines")
    score, _ = embedder.best_match_score(required, candidate_skills)
    if score >= config.SIMILARITY_THRESHOLD:
        return True

    return False


# ── Deterministic Skills Coverage ────────────────────────────────────────

def compute_skills_coverage(
    required_skills: list[str],
    candidate_pool: list[str],
    embedder: Embedder,
) -> tuple[float, list[str], list[str]]:
    """
    Check each required skill against the full candidate pool.
    Returns (score 0-100, matched_skills, missing_skills).
    Purely deterministic — no LLM involved.
    """
    if not required_skills:
        return 100.0, [], []

    matched: list[str] = []
    missing: list[str] = []

    for skill in required_skills:
        if skill_matches(skill, candidate_pool, embedder):
            matched.append(skill)
        else:
            missing.append(skill)

    score = (len(matched) / len(required_skills)) * 100.0
    return score, matched, missing
