"""Prompting strategies, length budgets and the shared prompt-writing rules."""

STRATEGIES = {
    "auto": {
        "label": "Auto",
        "icon": "✨",
        "desc": "The agent picks the best strategy for your task.",
        "best_for": "Not sure? Start here.",
        "instruction": (
            "Choose the single best strategy for this task from: zero-shot, few-shot, role-based, "
            "task-based, chain-of-thought, structured. Prefer the simplest one that reliably works, "
            "then write the prompt in that strategy's style."
        ),
    },
    "zero-shot": {
        "label": "Zero-shot",
        "icon": "🎯",
        "desc": "A direct instruction with no examples.",
        "best_for": "Simple, well-known tasks. Cheapest in tokens.",
        "instruction": (
            "Write a ZERO-SHOT prompt: one clear instruction, the input, and the output format. "
            "No examples, no persona padding. Keep it as short as possible."
        ),
    },
    "few-shot": {
        "label": "Few-shot",
        "icon": "🧩",
        "desc": "A short instruction plus 2-3 input/output examples.",
        "best_for": "Fixed formats, tone matching, classification.",
        "instruction": (
            "Write a FEW-SHOT prompt: a one-or-two-line instruction, then 2-3 compact, realistic "
            "input -> output examples that demonstrate the exact required format, then the live input "
            "placeholder. Use the user's examples if provided; otherwise invent short ones. Keep each "
            "example brief."
        ),
    },
    "role-based": {
        "label": "Role-based",
        "icon": "🎭",
        "desc": "Assigns an expert persona, audience and tone.",
        "best_for": "Advice, teaching, writing and review tasks.",
        "instruction": (
            "Write a ROLE-BASED prompt: open with a precise expert persona (expertise, audience, tone) "
            "in one or two sentences, then the task, input and output format. Avoid generic claims "
            "like 'world-class expert'."
        ),
    },
    "task-based": {
        "label": "Task-based",
        "icon": "📋",
        "desc": "Breaks the job into numbered steps with a clear deliverable.",
        "best_for": "Multi-step work and repeatable workflows.",
        "instruction": (
            "Write a TASK-BASED prompt: state the objective, list the numbered steps the model must "
            "follow, define the deliverable and acceptance criteria. Steps must be concrete and ordered."
        ),
    },
    "chain-of-thought": {
        "label": "Chain-of-thought",
        "icon": "🧠",
        "desc": "Asks the model to reason step by step before answering.",
        "best_for": "Logic, maths, analysis, debugging, test design.",
        "instruction": (
            "Write a CHAIN-OF-THOUGHT prompt: instruct the model to reason step by step first, then give "
            "the final answer in a clearly separated section in the required format. Keep the reasoning "
            "instruction short and specific to the task."
        ),
    },
    "structured": {
        "label": "Structured (XML)",
        "icon": "🧱",
        "desc": "Delimited tags for inputs and a strict output schema.",
        "best_for": "Pipelines, JSON output, anything parsed by code.",
        "instruction": (
            "Write a STRUCTURED prompt: wrap the instruction, input and rules in XML-style tags "
            "(e.g. <task>, <input>, <rules>, <output_format>) and specify a strict, machine-parseable "
            "output schema."
        ),
    },
}

LENGTHS = {
    "concise": {"label": "Concise", "words": 120, "desc": "~120 words"},
    "balanced": {"label": "Balanced", "words": 250, "desc": "~250 words"},
    "detailed": {"label": "Detailed", "words": 450, "desc": "~450 words"},
}

# Hard rules both agents must follow.
PROMPT_RULES = """PROMPT-WRITING RULES (follow all of them):
1. Put the main instruction first; start with a verb.
2. One task per prompt. Be specific; remove ambiguity and filler.
3. Say what to do, not only what to avoid (positive instructions).
4. Give context only when it changes the answer.
5. Delimit variable input clearly and keep the user's {{placeholders}} unchanged.
6. State the exact output format (structure, length, tone) the model must return.
7. Add constraints only where failure is likely; never pad with generic rules.
8. If information is missing, tell the model to say so instead of guessing.
9. Every sentence must earn its tokens: no repetition, no motivational fluff, no meta commentary.
10. Stay within the word budget."""


def public_catalog() -> dict:
    return {
        "strategies": [
            {"id": k, **{f: v[f] for f in ("label", "icon", "desc", "best_for")}}
            for k, v in STRATEGIES.items()
        ],
        "lengths": [{"id": k, **v} for k, v in LENGTHS.items()],
    }


def word_count(text: str) -> int:
    return len(text.split())
