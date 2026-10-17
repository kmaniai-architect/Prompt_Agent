from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from typing import Literal

import agents
import store
from strategies import STRATEGIES, public_catalog


@asynccontextmanager
async def lifespan(_: FastAPI):
    store.init()
    yield


app = FastAPI(title="Prompt Studio", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class GenerateIn(BaseModel):
    # Which of these the UI sends depends on the strategy; at least one must be filled in.
    role: str = ""
    context: str = ""
    input: str = ""
    output: str = ""
    title: str = ""
    strategy: Literal[tuple(STRATEGIES)] = "auto"  # type: ignore[valid-type]
    length: Literal["concise", "balanced", "detailed"] = "balanced"
    examples: str = ""


@app.get("/api/catalog")
def catalog():
    return public_catalog()


@app.get("/api/health")
def health():
    return {"provider": agents.provider(), "model": agents.model_name(), "live": agents.has_key()}


@app.get("/api/prompts")
def list_prompts():
    return store.list_all()


@app.get("/api/prompts/{pid}")
def get_prompt(pid: int):
    p = store.get(pid)
    if not p:
        raise HTTPException(404, "Prompt not found")
    return p


@app.post("/api/prompts/generate")
async def generate(body: GenerateIn):
    if not any(s.strip() for s in (body.role, body.context, body.input, body.output)):
        raise HTTPException(422, "Fill in at least one field")
    try:
        result = await agents.generate_prompt(
            body.role, body.context, body.input, body.output, body.strategy, body.length, body.examples
        )
    except Exception as e:  # surface LLM/auth errors to the UI
        raise HTTPException(502, f"Agent error: {e}")
    first = next((s for s in (body.role, body.context, body.input, body.output) if s.strip()), "")
    title = body.title.strip() or first.strip()[:50]
    return store.create({**body.model_dump(), "title": title, "length": body.length, **result})


@app.post("/api/prompts/{pid}/reverse")
async def reverse(pid: int, force: bool = False):
    p = store.get(pid)
    if not p:
        raise HTTPException(404, "Prompt not found")
    if p["reverse"] and not force:
        return p["reverse"]
    try:
        result = await agents.reverse_engineer(p["prompt"])
    except Exception as e:
        raise HTTPException(502, f"Agent error: {e}")
    store.save_reverse(pid, result)
    return result


@app.delete("/api/prompts/{pid}")
def delete(pid: int):
    if not store.delete(pid):
        raise HTTPException(404, "Prompt not found")
    return {"ok": True}


app.mount(
    "/", StaticFiles(directory=Path(__file__).parent.parent / "frontend", html=True), name="ui"
)
