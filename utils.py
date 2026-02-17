from __future__ import annotations
import json
import time

from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import numpy as np

import config


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
                )
                return json.loads(resp.choices[0].message.content)
            except json.JSONDecodeError:
                if attempt == config.LLM_MAX_RETRIES - 1:
                    raise ValueError("LLM returned invalid JSON after retries.")
                time.sleep(0.5)
            except Exception as e:
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

def skill_matches(required: str, candidate_skills: list[str], embedder: Embedder) -> bool:
    """
    Check if a required skill is present in the candidate's skill list.
    Uses fuzzy string matching first (fast), then semantic similarity as fallback.
    """
    req_lower = required.lower().strip()

    # 1. Fuzzy string match (handles React vs React.js vs ReactJS)
    for skill in candidate_skills:
        if fuzz.token_sort_ratio(req_lower, skill.lower().strip()) >= config.FUZZY_MATCH_THRESHOLD:
            return True

    # 2. Semantic similarity fallback (handles "CI/CD" vs "Jenkins pipelines")
    if candidate_skills:
        score, _ = embedder.best_match_score(required, candidate_skills)
        if score >= config.SIMILARITY_THRESHOLD:
            return True

    return False
