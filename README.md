# Resume Ranker

An intelligent resume screening system for HR teams that ranks candidates against job requirements using semantic understanding — not just keyword matching.

## What It Does

- Parses batches of resume PDFs into structured candidate profiles
- Enforces hard requirements (must-have skills, min experience, location)
- Scores candidates across multiple dimensions using semantic similarity
- Produces a ranked shortlist with clear explanations for each candidate
- Works for any role — just change the job description

## Tech Stack

- **Python** + **Streamlit** (UI)
- **Groq API** — Llama 3.1 8B Instant (resume/JD parsing only)
- **Sentence Transformers** — all-MiniLM-L6-v2 (local semantic scoring)
- **pdfplumber** (PDF text extraction)
- **Pydantic** (data validation)

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key_here
```

Run:

```bash
streamlit run app.py
```

## How It's Better Than Typical ATS Checkers

| Typical ATS Checker | Resume Ranker |
|---|---|
| Scores 1 resume vs 1 JD | Ranks N candidates against each other |
| Keyword overlap % | Semantic understanding of skills |
| No hard constraints | Enforces must-have requirements |
| Easy to game with keywords | Validates actual experience context |
| Black-box score | Transparent per-dimension breakdown |
