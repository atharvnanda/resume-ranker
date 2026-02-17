import os
from dotenv import load_dotenv

load_dotenv()

# --- Groq API ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0
LLM_MAX_RETRIES = 3

# --- Embeddings ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.75  # minimum cosine similarity to count as a skill match
FUZZY_MATCH_THRESHOLD = 80   # minimum rapidfuzz score (0-100) for string matching

# --- Scoring weights (defaults, HR can adjust in UI) ---
DEFAULT_WEIGHTS = {
    "skills":     0.35,
    "experience": 0.30,
    "seniority":  0.15,
    "education":  0.10,
    "preferred":  0.10,
}

# --- Upload limits ---
MAX_FILE_SIZE_MB = 10
MAX_RESUMES = 200

# --- Resume parsing ---
MIN_RESUME_TEXT_LENGTH = 50  # characters; below this, flag as unparseable
