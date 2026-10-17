# Prompt Studio

AutoGen-powered prompt generator + reverse-engineering UI.

## Run
```
python -m venv .venv
.venv\Scripts\pip install -r backend\requirements.txt
copy .env.example .env      # add OPENAI_API_KEY or ANTHROPIC_API_KEY
cd backend
..\.venv\Scripts\python -m uvicorn main:app --reload
```
Open http://127.0.0.1:8000. Without a key it runs in offline template mode.

## How it works
- **Generate**: `drafter` + `reviewer` AutoGen agents (RoundRobinGroupChat) iterate until the reviewer says APPROVE.
- **Reverse engineer** (runs when you click a prompt): `analyst` infers role/context/input/output, strengths and gaps; `rewriter` produces the improved prompt. Result is cached; use Re-run to refresh.
