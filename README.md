# Resume Ranker

An intelligent resume screening system for HR teams that ranks candidates against job requirements using semantic understanding — not just keyword matching.

Paste a job description, upload a batch of resume PDFs, and get a ranked shortlist with transparent, per-dimension scoring and exportable results — all in one click.

---

## What It Does

1. **Parses** a pasted job description into structured hard requirements and preferred criteria using an LLM.
2. **Extracts** text from uploaded resume PDFs and converts each into a structured candidate profile.
3. **Filters** candidates against hard requirements (must-have skills, minimum experience, required degree, custom musts). Candidates who don't pass are flagged and moved to the bottom — no wasted scoring compute.
4. **Scores** each passing candidate across five weighted dimensions:
   - **Skills** — fuzzy string match + semantic embedding similarity against required skills.
   - **Experience** — years of experience relative to the requirement.
   - **Seniority** — title-level alignment to the target seniority (intern → executive).
   - **Education** — highest degree attained.
   - **Preferred** — match rate on nice-to-have skills and certifications.
5. **Ranks** candidates by weighted composite score.
6. **Exports** the ranked table as an Excel file.

---

## How It's Better Than Typical ATS Checkers

| Typical ATS Checker | Resume Ranker |
|---|---|
| Scores 1 resume vs 1 JD | Ranks N candidates against each other |
| Keyword overlap % | Semantic understanding of skills |
| No hard constraints | Enforces must-have requirements first |
| Easy to game with keywords | Validates actual experience context via LLM |
| Black-box score | Transparent per-dimension breakdown with evidence |
| Fixed scoring | Adjustable dimension weights per role |

---

## Tech Stack

| Layer | Tool | Role |
|---|---|---|
| UI | **Streamlit** | Web interface, file upload, interactive controls |
| LLM | **Groq API** (Llama 3.1 8B Instant) | Structured extraction only (JD + resume parsing) |
| Embeddings | **Sentence Transformers** (all-MiniLM-L6-v2) | Local semantic skill matching — no API calls for scoring |
| PDF parsing | **pdfplumber** | Text extraction from resume PDFs |
| Skill matching | **rapidfuzz** + cosine similarity | Two-layer match: fuzzy string first, then embedding fallback |
| Data models | **Pydantic** | Strict schema validation for all structured data |
| Export | **pandas** + **openpyxl** | Excel export of ranked results |

---

## Project Structure

```
resume-ranking/
├── app.py              # Streamlit UI — inputs, progress, results display, Excel export
├── config.py           # All configuration constants (API keys, model names, weights, limits)
├── models.py           # Pydantic data models (JD, candidate, evaluation, scores)
├── parsers.py          # PDF text extraction + LLM-based structured parsing
├── scorer.py           # Hard filter → 5-dimension scoring → weighted ranking
├── utils.py            # Groq LLM wrapper, embedding helper, skill matching logic
├── requirements.txt    # Python dependencies
├── .env                # GROQ_API_KEY (not committed)
└── .gitignore
```

---

## Pipeline Architecture

```
┌──────────────┐     ┌──────────────┐
│  Paste JD    │     │ Upload PDFs  │
└──────┬───────┘     └──────┬───────┘
       │                    │
       ▼                    ▼
  ┌─────────┐        ┌───────────┐
  │ LLM     │        │ pdfplumber│ ── extract text
  │ parse   │        │ + LLM     │ ── structured extraction
  └────┬────┘        └─────┬─────┘
       │                   │
       ▼                   ▼
  JobRequirements    CandidateProfile[]
       │                   │
       └─────────┬─────────┘
                 ▼
        ┌────────────────┐
        │  Hard Filter   │ ── experience, degree, custom musts
        └────────┬───────┘
                 ▼
        ┌────────────────┐
        │  Score (×5)    │ ── skills, experience, seniority, education, preferred
        └────────┬───────┘
                 ▼
        ┌────────────────┐
        │  Rank & Export │ ── sorted table + Excel download
        └────────────────┘
```

---

## Setup

### 1. Clone

```bash
git clone https://github.com/atharvnanda/resume-ranker.git
cd resume-ranker
```

### 2. Virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

### 5. Run

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Usage

1. **Paste a job description** into the left panel.
2. **Upload resume PDFs** in the right panel (up to 200 files, 10 MB each).
3. **Adjust scoring weights** in the sidebar if needed (defaults: Skills 35%, Experience 30%, Seniority 15%, Education 10%, Preferred 10%).
4. Click **🚀 Rank Candidates**.
5. Review the ranked table, expand individual candidate cards for detailed breakdowns, and download the results as Excel.

---

## Configuration

All tunable parameters live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model for LLM extraction |
| `LLM_TEMPERATURE` | `0` | Deterministic extraction |
| `LLM_MAX_RETRIES` | `3` | Retry attempts on LLM/JSON failures |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer for semantic matching |
| `SIMILARITY_THRESHOLD` | `0.75` | Minimum cosine similarity for a skill match |
| `FUZZY_MATCH_THRESHOLD` | `80` | Minimum rapidfuzz score (0–100) for string matching |
| `MAX_FILE_SIZE_MB` | `10` | Per-file upload limit |
| `MAX_RESUMES` | `200` | Maximum resumes per batch |
| `MIN_RESUME_TEXT_LENGTH` | `50` | Characters below which a PDF is flagged as unparseable |
| `DEFAULT_WEIGHTS` | Skills .35, Exp .30, Sen .15, Edu .10, Pref .10 | Scoring dimension weights (adjustable in UI) |

---

## Edge Cases Handled

- **Empty or corrupt PDFs** → flagged as `UNPARSEABLE`, ranked last with clear message.
- **LLM returns invalid JSON** → retried up to 3 times, then flagged as `PARSE_ERROR`.
- **Oversized files** → skipped with a warning before processing.
- **Too many uploads** → truncated to the limit with an error message.
- **All weights set to zero** → blocked with a validation error.
- **Missing API key** → clear error on startup, not a silent crash.
- **Skill name variations** (React vs React.js vs ReactJS) → caught by fuzzy string matching.
- **Semantic skill gaps** (CI/CD vs Jenkins pipelines) → caught by embedding similarity fallback.
- **No hard requirements in JD** → filter passes everyone, scoring still applies.
- **Missing candidate fields** → Pydantic defaults prevent crashes.

---

## Dependencies

```
streamlit
groq
pdfplumber
pydantic
sentence-transformers
scikit-learn
python-dotenv
rapidfuzz
pandas
openpyxl
```

---

## License

MIT
