import os
from dotenv import load_dotenv

load_dotenv()

# --- Groq API ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "openai/gpt-oss-120b"       # 120B, 500 tok/s — used for parsing + evaluation
LLM_TEMPERATURE = 0
LLM_SEED = 42                            # fixed seed for reproducible outputs
LLM_MAX_RETRIES = 3

# --- Embeddings ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.55   # cosine similarity — lowered to catch related skills (sql/mysql, api design/rest apis)
FUZZY_MATCH_THRESHOLD = 75    # rapidfuzz score — lowered to catch abbreviation variants

# --- Scoring weights (defaults, HR can adjust in UI) ---
DEFAULT_WEIGHTS = {
    "skills_coverage":    0.25,   # deterministic: what fraction of required skills are present
    "skills_depth":       0.20,   # LLM: how deeply are skills demonstrated
    "project_relevance":  0.20,   # LLM: how relevant is their work to this role
    "experience":         0.15,
    "seniority":          0.05,
    "education":          0.05,
    "overall_fit":        0.10,
}

# --- Hard filter ---
REQUIRED_SKILLS_FAIL_THRESHOLD = 0.4  # fail if candidate matches fewer than 40% of required skills

# --- Upload limits ---
MAX_FILE_SIZE_MB = 10
MAX_RESUMES = 200

# --- Resume parsing ---
MIN_RESUME_TEXT_LENGTH = 50  # characters; below this, flag as unparseable
