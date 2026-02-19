# 📄 Resume Ranker — Complete Project Deep Dive

> **What is this project?**
> It's an AI-powered tool for HR teams. You paste a job description, upload a bunch of resumes (PDFs), and it **ranks all candidates** from best to worst — with transparent scores and explanations. Think of it as a very smart, automated first-round screener.

---

## 🗂️ Project Structure

```
resume-ranker/
├── app.py          ← The web UI (what you see in the browser)
├── config.py       ← All settings/constants in one place
├── models.py       ← Data "shapes" / blueprints for all data
├── parsers.py      ← Reads PDFs & extracts structured info using AI
├── scorer.py       ← The brain: filters, scores, and ranks candidates
├── utils.py        ← Shared helpers: AI client, skill matching, embeddings
├── requirements.txt← Python packages to install
└── .env            ← Your secret API key (not in the repo)
```

---

## 🔁 How The Whole Thing Works — End to End

Here's the full journey from click to ranked results:

```
You paste a JD + upload PDFs
        ↓
[app.py] User clicks "Rank Candidates"
        ↓
[parsers.py] JD text → LLM → JobRequirements object
[parsers.py] Each PDF → extract text → LLM → CandidateProfile object
        ↓
[scorer.py] For each candidate:
    1. Compute skills coverage (deterministic — no AI)
    2. Hard filter: does candidate meet minimum bar?
    3. Rule-based scores: experience, seniority, education
    4. LLM judge: skills depth, project relevance, overall fit
    5. Weighted average → composite score (0-100)
        ↓
[scorer.py] Sort by score (failed candidates go to bottom)
        ↓
[app.py] Display ranked table + per-candidate breakdown + Excel export
```

---

## 📁 File-by-File Explanation

---

### 1. `config.py` — The Settings File
> **Simple explanation:** Like a control panel. Every magic number or setting lives here. Nothing is hardcoded in other files.

**What each setting does:**

| Setting | What it means |
|---|---|
| `GROQ_API_KEY` | Your API key to use the Groq AI service. Read from `.env` file. |
| `GROQ_MODEL` | Which AI model to use. The 120B is a very large, capable model. |
| `LLM_TEMPERATURE = 0` | **Temperature** = how creative/random the AI is. `0` means fully deterministic — same input always gives same output. Good for consistency. |
| `LLM_SEED = 42` | Another reproducibility setting. Like a random seed in any program — fixes the "dice roll" so results are repeatable. |
| `LLM_MAX_RETRIES = 3` | If the AI gives bad output, retry up to 3 times before giving up. |
| `EMBEDDING_MODEL` | The local AI model used for **semantic skill matching** (explained later). `all-MiniLM-L6-v2` is a small, fast model that runs on your machine — no API calls needed. |
| `SIMILARITY_THRESHOLD = 0.55` | **Cosine similarity** (a math score from 0 to 1 measuring how "similar" two things are in meaning) — skills scoring above 0.55 are considered a match. Lowered to catch related terms like "SQL" matching "MySQL". |
| `FUZZY_MATCH_THRESHOLD = 75` | **Fuzzy matching** (approximate string matching — like "React" matching "ReactJS") — a score of 75/100 is needed to count as a match. |
| `DEFAULT_WEIGHTS` | How much each scoring dimension counts toward the final score (as fractions — they're normalized). |
| `REQUIRED_SKILLS_FAIL_THRESHOLD = 0.4` | A candidate must match at least **40%** of required skills to pass the hard filter. Below that = auto-rejected. |
| `MAX_FILE_SIZE_MB / MAX_RESUMES` | Safety limits to prevent crashes. |
| `MIN_RESUME_TEXT_LENGTH = 50` | If a PDF produces fewer than 50 characters of text, it's considered unparseable. |

---

### 2. `models.py` — The Data Blueprints
> **Simple explanation:** Defines "what does a job description look like as data?" and "what does a candidate look like as data?" Uses **Pydantic** — a library that validates data shapes and types automatically (like a strict form that won't accept wrong inputs).

**Key concept — Pydantic Models:** Imagine a template. You say "a candidate has a name (string), years of experience (number), skills (list of strings)…" and Pydantic enforces this. If the AI returns `null` for a list field, Pydantic auto-converts it to `[]` instead of crashing.

There's a clever base class called `SafeBase` that silently converts `null` values returned by the AI to safe defaults (`[]`, `""`, `0`) instead of crashing.

**The data models in `models.py` are:**

| Model | Represents | Key fields |
|---|---|---|
| `HardRequirements` | Must-have job requirements | `required_skills`, `min_experience_years`, `required_degree`, `custom_musts` |
| `PreferredCriteria` | Nice-to-have requirements | `preferred_skills`, `preferred_certifications` |
| `JobRequirements` | The full job description | `title`, `seniority_level`, `hard`, `preferred` |
| `ExperienceEntry` | One job in a resume | `title`, `company`, `duration_years`, `responsibilities` |
| `EducationEntry` | One degree/school | `degree`, `field`, `institution`, `year` |
| `CandidateProfile` | A full parsed resume | `name`, `skills`, `experience[]`, `education[]`, `total_experience_years`, `raw_text` |
| `DimensionScore` | One scoring dimension result | `score` (0-100), `evidence` (text explanation), `matched`, `missing` |
| `LLMJudgment` | The AI's verdict on a candidate | `skills_depth`, `project_relevance`, `overall_fit`, `strengths`, `gaps` |
| `FilterResult` | Did the candidate pass the hard filter? | `passed` (True/False), `failures` (list of reasons) |
| `CandidateEvaluation` | The full result for one candidate | `candidate`, `scores{}`, `filter_result`, `composite_score`, `rank` |

---

### 3. `utils.py` — Shared Helpers
> **Simple explanation:** The toolbox. Has the AI client wrapper, the embedding model, and all the skill-matching logic.

#### `LLM` class — the AI caller

- Wraps the **Groq API** (a fast cloud AI service).
- Sends a **system prompt** (instructions to the AI) + a **user prompt** (the actual data).
- Forces the response to be **JSON** — no prose, just structured data.
- Retries up to 3 times on failure.

#### `Embedder` class — semantic understanding

> **What is an embedding?** A way to convert text into a list of numbers (a vector) that captures its *meaning*. Similar concepts (like "CI/CD" and "Jenkins pipelines") will have vectors that are close together in space. This lets the system match skills by *meaning*, not just spelling.

- Loads `all-MiniLM-L6-v2` — a small but powerful local **sentence transformer** model.
- `encode()` turns text into meaning-vectors.
- `best_match_score()` computes the **cosine similarity** (how close two vectors are — 1.0 = identical meaning, 0.0 = completely unrelated) between a required skill and a list of candidate skills.

#### Skill Matching — the 4-layer system

The key function `skill_matches()` checks whether a required skill is present in a candidate's profile using **four layers**:

```
Layer 0 — Alias expansion
  "system design" → also checks "system architecture", "distributed systems", etc.

Layer 1 — Substring containment (fast)
  "SQL" found inside "MySQL" ✓
  "REST API" found inside "REST APIs" ✓

Layer 2 — Fuzzy string matching
  "React" matches "React.js" or "ReactJS" (rapidfuzz score ≥ 75) ✓

Layer 3 — Semantic similarity (AI-based)
  "CI/CD" semantically matches "Jenkins pipelines" (cosine sim ≥ 0.55) ✓
```

This cascading approach is smart: fast checks first, expensive AI only as fallback.

---

### 4. `parsers.py` — Reading & Extracting Data
> **Simple explanation:** Takes messy raw inputs (a blob of JD text, a PDF file) and turns them into clean, structured Python objects.

#### Step 1 — PDF text extraction

Uses **pdfplumber** — a library that reads PDF files and extracts the text from each page. Resets the file pointer first (`file.seek(0)`) because Streamlit's file uploader can leave it at the end.

#### Step 2 — LLM-based structured extraction

After getting raw text, a carefully engineered prompt is sent to the AI. For example, the **Job Description Parser** prompt instructs the AI:

> *"Extract the following fields as a JSON object. Put all technologies into `required_skills`. Put years mentioned into `min_experience_years`. Leave `custom_musts` only for non-skill, non-experience requirements."*

The AI returns structured JSON, which is then validated into a `JobRequirements` Pydantic model.

Same approach for resumes: raw PDF text → AI → `CandidateProfile`.

**Error handling:** If a PDF has fewer than 50 characters of text, the candidate is flagged as `UNPARSEABLE` and ranked last. If the AI returns invalid JSON after 3 retries, the candidate is flagged as `PARSE_ERROR`.

---

### 5. `scorer.py` — The Brain
> **Simple explanation:** Takes the parsed job and candidates, and produces a ranked list with scores. This is the most complex file.

The scoring pipeline for each candidate has **4 steps**:

---

#### Step 1 — Skills Coverage (Deterministic — no AI)

Checks what fraction of the required skills the candidate has, using the 4-layer matching system from `utils.py`.

```
Required: ["Python", "React", "PostgreSQL", "Docker", "AWS"]
Candidate has: ["Python", "ReactJS", "MySQL", "Kubernetes"]

Matches:
  Python ✓ (exact)
  ReactJS ✓ (fuzzy match for "React")
  MySQL ✓ (substring: "sql" in "mysql")
  Docker ✗
  AWS ✗

Coverage: 3/5 = 60%
```

This is **deterministic** — pure logic, no AI, always the same result. Stored as `skills_coverage` score.

---

#### Step 2 — Hard Filter (Fail Fast)

If a candidate fails any hard requirement, they're **immediately marked as failed** and skipped for the expensive AI scoring. They still appear in results, just at the bottom with reasons listed.

Checks performed:
- Minimum years of experience
- Required degree
- Skills coverage below 40% threshold
- Any custom hard requirements (e.g. "must have work authorization")

---

#### Step 3 — Rule-Based Scores (Deterministic)

Three dimensions scored with pure logic:

**Experience score:**
```
Required: 5 years, Candidate has: 7 years
Score = min((7/5) * 100, 100) = 100 ✓

Required: 5 years, Candidate has: 2 years
Score = min((2/5) * 100, 100) = 40
```

**Seniority score:**
Maps job titles to a level (intern=0, junior=1, mid=2, senior=3, lead=4, principal/executive=5). Computes the level difference and deducts 25 points per level of mismatch.
```
Job wants: Senior (level 3)
Candidate title: "Software Engineer" → Mid (level 2)
Diff = 1 → Score = 100 - 25 = 75
```

Also uses years of experience as a fallback (10+ years → at least Senior, 5+ → at least Mid).

**Education score:**
Maps degree names to a rank (High School=0, Associate=1, Bachelor=2, Master=3, PhD=4) and scores accordingly (rank × 33.3, capped at 100). Has a fallback that scans raw resume text for degree keywords if the AI missed them.

---

#### Step 4 — LLM-as-Judge (AI Scoring)

The most powerful part. An AI is given:
1. The full job description
2. The candidate's parsed resume
3. **A pre-computed checklist of which required skills are present/missing** (this anchors the AI and prevents it from hallucinating)

The AI is asked to score three dimensions:

| Dimension | What it measures | Sub-factors |
|---|---|---|
| `skills_depth` | Are present skills backed by real work, or just listed? | Demonstrated use (0-35), Depth of knowledge (0-35), Recency (0-30) |
| `project_relevance` | How relevant is their actual work to this role? | Domain match (0-25), Tech stack overlap (0-25), Complexity (0-25), Impact (0-25) |
| `overall_fit` | Holistic tiebreaker | Everything combined |

The AI also returns:
- `strengths`: A list of what the candidate does well
- `gaps`: A list of what they're missing

**Why give the AI a pre-computed checklist?** Because LLMs can be inconsistent. By telling the AI "here's exactly which skills matched", the project prevents the AI from contradicting the rule-based check. The AI focuses on *quality and depth*, not re-doing the *presence check*.

---

#### Final Score Calculation

All dimension scores are combined into a weighted average:

```
composite_score = Σ (score[dim] × weight[dim]) / Σ weight[dim]
```

Default weights:
- Skills Coverage: **25%**
- Skills Depth (LLM): **20%**
- Project Relevance (LLM): **20%**
- Experience: **15%**
- Overall Fit (LLM): **10%**
- Seniority: **5%**
- Education: **5%**

Candidates who **passed** the hard filter are sorted by this score (high → low). Candidates who **failed** are appended after, also sorted among themselves.

---

### 6. `app.py` — The Web Interface
> **Simple explanation:** The thing you see in your browser. Built with **Streamlit** — a Python library that turns Python scripts into web apps without writing any HTML/CSS/JS.

**Key sections:**

**Sidebar (left panel)** — 7 sliders, one per scoring dimension, to adjust weights. HR can tune these per role (e.g. weight education higher for research roles).

**Main area** — Two columns:
- Left: text box to paste job description
- Right: file uploader for resume PDFs

**"Rank Candidates" button** — Triggers the whole pipeline:
1. Validates inputs (JD not empty, files uploaded, weights sum > 0)
2. Initializes models (lazy — only loads on first run, then **cached** with `@st.cache_resource`)
3. Parses the JD
4. Parses each resume (with a progress bar)
5. Calls `rank_candidates()` from `scorer.py`
6. Displays results

**Results display:**
- A summary table (rank, name, score, filter status, key matches/misses)
- Expandable cards per candidate with full dimension scores and evidence
- **Excel download button** — exports everything to an `.xlsx` file using **pandas** + **openpyxl**

**Caching:**
`@st.cache_resource` means: load the embedding model (which is slow, ~2 seconds) only once and reuse it across all interactions. Without this, every button click would reload the model from disk.

---

## 🔧 The Tech Stack — Quick Reference

| Tool | What it is | Used for |
|---|---|---|
| **Streamlit** | Python → web app library | The whole UI |
| **Groq API** | Cloud AI inference service | Parsing JDs & resumes, LLM judging |
| **Sentence Transformers** | Local AI model library | Semantic skill matching (runs on your machine) |
| **pdfplumber** | PDF reading library | Extracting text from resume PDFs |
| **Pydantic** | Data validation library | Enforcing data shapes, safe defaults |
| **rapidfuzz** | Fast fuzzy string matching | Catching "React" vs "ReactJS" variations |
| **scikit-learn** | ML utilities | Computing cosine similarity |
| **pandas + openpyxl** | Data table + Excel file library | Building and exporting results as Excel |
| **python-dotenv** | Environment variable loader | Reading the `.env` file for secrets |

---

## 🚧 Edge Cases The Code Handles

| Situation | What happens |
|---|---|
| PDF is blank or corrupt | Flagged as `UNPARSEABLE`, ranked last |
| AI returns invalid JSON | Retried up to 3 times, then flagged as `PARSE_ERROR` |
| File is > 10 MB | Skipped with a warning |
| More than 200 resumes | Truncated to 200 |
| All weights set to 0 | Blocked before processing |
| Missing API key | Clear error on startup |
| "React" vs "ReactJS" vs "React.js" | Caught by fuzzy matching |
| "CI/CD" vs "Jenkins pipelines" | Caught by semantic embedding similarity |
| No hard requirements in JD | Filter passes everyone, scoring still applies |
| AI misses education section | Falls back to scanning raw resume text for degree keywords |

---

## 💡 What Makes This Better Than Simple Keyword Matching

A typical ATS (Applicant Tracking System — the software companies use to filter resumes) just counts keyword overlaps. This project goes much further:

1. **Understands skill variants** — "React", "ReactJS", "React.js" are all the same skill
2. **Understands semantics** — "CI/CD experience" and "Jenkins pipelines" are related
3. **Checks quality, not just presence** — An AI verifies if skills are actually *demonstrated in real work* or just listed
4. **Enforces hard limits first** — Doesn't waste AI compute on unqualified candidates
5. **Transparent scoring** — Every score comes with a text explanation
6. **Tunable** — HR can adjust weights per role
7. **Ranks the whole pool** — Not just "pass/fail" but an ordered shortlist