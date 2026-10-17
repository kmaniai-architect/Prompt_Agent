"""AutoGen multi-agent logic.

generate_prompt: Drafter <-> Reviewer loop builds a customised prompt from role/context/input/output.
reverse_engineer: Analyst deconstructs an existing prompt, Rewriter produces a constructive version.
Falls back to deterministic templates when no API key is configured.
"""
import json
import os
import re

from dotenv import load_dotenv

from strategies import LENGTHS, PROMPT_RULES, STRATEGIES, word_count

load_dotenv()

MODEL_INFO = {
    "vision": False,
    "function_calling": False,
    "json_output": False,
    "structured_output": False,
    "family": "unknown",
}


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "openai").lower()


def has_key() -> bool:
    key = "ANTHROPIC_API_KEY" if provider() == "anthropic" else "OPENAI_API_KEY"
    return bool(os.getenv(key))


def model_name() -> str:
    default = "claude-sonnet-5-5" if provider() == "anthropic" else "gpt-4o-mini"
    return os.getenv("LLM_MODEL", default)


def _make_client():
    if provider() == "anthropic":
        from autogen_ext.models.anthropic import AnthropicChatCompletionClient

        extra = {}
        if os.getenv("ANTHROPIC_WORKSPACE_ID"):  # needed for keys not scoped to a workspace
            extra["default_headers"] = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]}
        client = AnthropicChatCompletionClient(
            model=model_name(), api_key=os.getenv("ANTHROPIC_API_KEY"), model_info=MODEL_INFO, **extra
        )
        # AutoGen always sends `temperature`, which newer Anthropic models/SDKs reject.
        messages = client._client.messages
        original_create = messages.create

        def create_without_temperature(*args, **kwargs):
            kwargs.pop("temperature", None)
            return original_create(*args, **kwargs)

        messages.create = create_without_temperature
        return client
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    extra = {"base_url": os.environ["LLM_BASE_URL"]} if os.getenv("LLM_BASE_URL") else {}
    return OpenAIChatCompletionClient(
        model=model_name(), api_key=os.getenv("OPENAI_API_KEY"), model_info=MODEL_INFO, **extra
    )


def _tag(text: str, name: str) -> str | None:
    m = re.search(rf"<{name}>\s*(.*?)\s*</{name}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


async def _run_team(agents, task: str, max_messages: int, stop_text: str | None = None):
    from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
    from autogen_agentchat.teams import RoundRobinGroupChat

    term = MaxMessageTermination(max_messages)
    if stop_text:
        term = term | TextMentionTermination(stop_text)
    team = RoundRobinGroupChat(agents, termination_condition=term)
    result = await team.run(task=task)
    return [
        {"agent": m.source, "content": m.content}
        for m in result.messages
        if m.source != "user" and isinstance(getattr(m, "content", None), str)
    ]


def drafter_message(strategy: str, budget: int) -> str:
    return f"""You are Drafter, an expert prompt engineer. Write a production-quality prompt that a user
will paste into an LLM.

STRATEGY: {STRATEGIES[strategy]["instruction"]}

{PROMPT_RULES}

HARD LIMIT: the prompt must be at most {budget} words. Shorter is better if nothing is lost.
Use short markdown headings only if they help; do not force a fixed section template.

Reply in exactly this format:
<strategy>the strategy name you used, e.g. zero-shot</strategy>
<prompt>
the full prompt
</prompt>
When the Reviewer gives feedback, apply it and reply in the same format."""


def reviewer_message(strategy: str, budget: int) -> str:
    return f"""You are Reviewer, a strict prompt-quality critic. Check the Drafter's prompt against the
user's role, context, input and required output, and against these rules:

{PROMPT_RULES}

Also check that it genuinely follows the requested strategy ({strategy}) and is at most {budget} words
(estimate; reject anything clearly over). Flag ambiguity, filler, repetition and a vague output format.
If changes are needed, give at most 4 short bullet fixes. If it is good, reply with exactly: APPROVE"""


TRIMMER = """You are Trimmer. Shorten the given prompt to the word limit without losing any requirement,
placeholder, example format or output specification. Remove filler, repetition and generic rules first.
Reply with only <prompt>...</prompt>."""

ANALYST = """You are Analyst. Reverse-engineer the given prompt: infer what the author was trying to
achieve and how the prompt is built. Reply ONLY with JSON inside <analysis>...</analysis> using keys:
inferred_role, inferred_context, inferred_input, inferred_output, intent (one sentence),
techniques (list), strengths (list), weaknesses (list), missing (list)."""

REWRITER = """You are Rewriter. Using the Analyst's findings and the original prompt, produce a
constructive, improved prompt that fixes every weakness and fills every gap while preserving intent.
Follow these rules:

""" + PROMPT_RULES + """

Do NOT bloat the prompt: it must not exceed 130% of the original length unless a missing element
makes that unavoidable, and every added sentence must fix a named weakness. Keep the original
strategy (zero-shot, few-shot, etc.) unless it is clearly wrong for the task. Return it in
<prompt>...</prompt>, then a short <changes> list of what you improved and why, one bullet per line
starting with '-'."""


async def generate_prompt(
    role: str, context: str, input_: str, output: str,
    strategy: str = "auto", length: str = "balanced", examples: str = "",
) -> dict:
    budget = LENGTHS[length]["words"]
    if not has_key():
        return _mock_generate(role, context, input_, output, strategy, examples)
    from autogen_agentchat.agents import AssistantAgent

    client = _make_client()
    try:
        drafter = AssistantAgent(
            "drafter", model_client=client, system_message=drafter_message(strategy, budget)
        )
        reviewer = AssistantAgent(
            "reviewer", model_client=client, system_message=reviewer_message(strategy, budget)
        )
        given = [
            (label, val.strip())
            for label, val in (
                ("Role", role), ("Context", context),
                ("Input the prompt will receive", input_), ("Required output", output),
            )
            if val.strip()
        ]
        missing = [l for l, v in (("role", role), ("context", context), ("input", input_), ("output format", output)) if not v.strip()]
        task = (
            "\n".join(f"{l}: {v}" for l, v in given) + "\n"
            + (f"User-provided examples:\n{examples}\n" if examples.strip() else "")
            + (f"Not provided (infer sensible defaults or omit if unnecessary): {', '.join(missing)}\n" if missing else "")
            + "\nDraft the prompt."
        )
        # REVIEW_ROUNDS=0 skips the reviewer (1 model call); each extra round adds up to 2 calls.
        rounds = max(0, int(os.getenv("REVIEW_ROUNDS", "2")))
        if rounds == 0:
            transcript = await _run_team([drafter], task, 2)
        else:
            transcript = await _run_team([drafter, reviewer], task, 2 * rounds + 2, "APPROVE")
        final = next(
            (t for t in reversed(transcript) if t["agent"] == "drafter"), {"content": ""}
        )["content"]
        prompt = _tag(final, "prompt") or final.strip()
        used = (_tag(final, "strategy") or strategy).strip().lower()

        # Guarantee the budget: one compression pass if the prompt is well over the limit.
        if word_count(prompt) > budget * 1.2:
            trimmer = AssistantAgent("trimmer", model_client=client, system_message=TRIMMER)
            trimmed = await _run_team(
                [trimmer], f"Word limit: {budget}\n\n<prompt>\n{prompt}\n</prompt>", 2
            )
            transcript += trimmed
            prompt = _tag(trimmed[-1]["content"], "prompt") or prompt
    finally:
        await client.close()
    return {
        "prompt": prompt,
        "transcript": transcript,
        "mode": f"autogen/{provider()}",
        "strategy": used if used in STRATEGIES else strategy,
        "length": length,
    }


async def reverse_engineer(prompt: str) -> dict:
    if not has_key():
        return _mock_reverse(prompt)
    from autogen_agentchat.agents import AssistantAgent

    client = _make_client()
    try:
        analyst = AssistantAgent("analyst", model_client=client, system_message=ANALYST)
        rewriter = AssistantAgent("rewriter", model_client=client, system_message=REWRITER)
        transcript = await _run_team([analyst, rewriter], f"Original prompt:\n\n{prompt}", 3)
    finally:
        await client.close()
    by = {t["agent"]: t["content"] for t in transcript}
    try:
        analysis = json.loads(_tag(by.get("analyst", ""), "analysis") or "{}")
    except json.JSONDecodeError:
        analysis = {}
    rw = by.get("rewriter", "")
    changes = [
        c.lstrip("- ").strip() for c in (_tag(rw, "changes") or "").splitlines() if c.strip()
    ]
    return {
        "analysis": analysis,
        "improved_prompt": _tag(rw, "prompt") or rw.strip(),
        "changes": changes,
        "transcript": transcript,
        "mode": f"autogen/{provider()}",
    }


# ---------- offline fallbacks (no API key) ----------

def _mock_generate(role, context, input_, output, strategy="auto", examples="") -> dict:
    s = "zero-shot" if strategy == "auto" else strategy
    role = role.strip() or "an expert assistant"
    context = context.strip() or "the task described in the input"
    input_ = input_.strip() or "{{input}}"
    output = output.strip() or "a clear, well-structured answer"
    miss = "If information is missing, say so instead of guessing."
    if s == "role-based":
        prompt = (f"You are {role}. Your audience: {context}\n\nRespond to the input below.\n\n"
                  f"Input: {input_}\n\nOutput: {output}\n\n{miss}")
    elif s == "task-based":
        prompt = (f"Objective: act as {role} for this context: {context}\n\nSteps:\n"
                  f"1. Read the input: {input_}\n2. Work out the answer using only that input.\n"
                  f"3. Produce the deliverable: {output}\n\nDone when the output matches the format exactly. {miss}")
    elif s == "few-shot":
        ex = examples.strip() or f"Input: <sample input>\nOutput: <sample output: {output}>"
        prompt = (f"As {role} ({context}), produce: {output}\n\nExamples:\n{ex}\n\n"
                  f"Now do the same.\nInput: {input_}\nOutput:")
    elif s == "chain-of-thought":
        prompt = (f"You are {role}. Context: {context}\n\nInput: {input_}\n\nThink through the problem "
                  f"step by step first. Then give the final answer under 'Answer:' as: {output}\n\n{miss}")
    elif s == "structured":
        prompt = (f"<task>Act as {role}. Context: {context}</task>\n<input>{input_}</input>\n"
                  f"<rules>Use only the input. {miss}</rules>\n<output_format>{output}</output_format>")
    else:  # zero-shot
        prompt = (f"As {role}, handle the input below. Context: {context}\n\nInput: {input_}\n\n"
                  f"Output: {output}\n\n{miss}")
    return {"prompt": prompt, "transcript": [], "mode": "offline-template", "strategy": s}


def _mock_reverse(prompt: str) -> dict:
    low = prompt.lower()
    sections = {
        "role": any(k in low for k in ("you are", "role")),
        "context": "context" in low or "background" in low,
        "output format": any(k in low for k in ("output", "format", "respond")),
        "constraints": any(k in low for k in ("must", "do not", "don't", "constraint", "never")),
        "examples": "example" in low,
    }
    present = [k for k, v in sections.items() if v]
    missing = [k for k, v in sections.items() if not v]
    additions = {
        "examples": "## Examples\nInput: <sample input>\nOutput: <ideal output in the required format>",
        "constraints": "## Constraints\n- Do not invent facts.\n- Ask for missing information.",
        "output format": "## Output Format\nSpecify structure, length and tone.",
        "context": "## Context\nState the background and the goal.",
        "role": "## Role\nDefine who the model is.",
    }
    body = prompt.strip()
    if "## " in body:  # already structured: keep it, append only what is missing
        improved = body + "".join("\n\n" + additions[m] for m in missing)
    else:
        improved = "## Role\nDefine who the model is.\n\n## Task\n" + body + "".join(
            "\n\n" + additions[m] for m in missing if m != "role"
        )
    improved += "\n\n## Self-check\nBefore answering, verify the output follows the format and every claim is grounded in the input."
    role_m = re.search(r"you are ([^.\n]+)", prompt, re.I)
    sentence = re.sub(r"^#+\s*\w+\s*", "", body).strip().split("\n")[0]
    return {
        "analysis": {
            "inferred_role": role_m.group(1).strip() if role_m else "not stated",
            "intent": sentence[:160] or "unclear",
            "techniques": present,
            "strengths": [f"Includes {p}" for p in present],
            "weaknesses": [f"No explicit {m}" for m in missing],
            "missing": missing,
        },
        "improved_prompt": improved,
        "changes": [f"Added {m} section" for m in missing] + ["Added a self-check step"],
        "transcript": [],
        "mode": "offline-template",
    }
