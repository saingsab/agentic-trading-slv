"""Builds and runs the Phase 5 CodeAgent.

Model: local Ollama via LiteLLMModel (confirmed with the user 2026-08-24
-- litellm has native Anthropic support too, so PLAN.md's "local model
first, then swap to Claude and compare" is a model_id change later, not a
different model class).

Sandboxing: Docker by default (confirmed with the user 2026-08-24 --
PLAN.md's own warning: "it executes generated Python and this machine
holds your API keys"). data/ is mounted read-only into the container so
tools.py's functions (which run inside the sandbox -- see tools.py's
docstring) can reach slv.db without slv itself being installed there.
sandbox=False (local executor, same process as this CLI) exists for fast
iteration on tool logic, not for routine use.
"""
from __future__ import annotations

from smolagents import CodeAgent, LiteLLMModel

from slv import config
from slv.agent.tools import get_indicators, get_regime, search_journal

MODEL_ID = "ollama_chat/llama3.1:8b"
API_BASE = "http://127.0.0.1:11434"

# Must match the "/data/slv.db" literal hardcoded in each tools.py
# function -- the sandbox can't import slv.config to share this constant,
# so the two sides of this contract just have to be kept in sync by hand.
CONTAINER_DATA_DIR = "/data"

TOOLS = [get_indicators, get_regime, search_journal]


def build_agent(sandbox: bool = True) -> CodeAgent:
    model = LiteLLMModel(model_id=MODEL_ID, api_base=API_BASE, num_ctx=8192)

    if not sandbox:
        return CodeAgent(tools=TOOLS, model=model)

    return CodeAgent(
        tools=TOOLS,
        model=model,
        executor_type="docker",
        executor_kwargs={
            # Reuse the cached jupyter-kernel image if it exists; smolagents
            # falls back to building it if it doesn't, so this is safe on
            # a fresh machine too -- just slower the first time.
            "build_new_image": False,
            "container_run_kwargs": {
                "volumes": {str(config.DATA_DIR): {"bind": CONTAINER_DATA_DIR, "mode": "ro"}}
            },
        },
    )


def ask(question: str, sandbox: bool = True) -> str:
    """Run one question through the agent and return its final answer.

    Uses the agent as a context manager so the Docker container (if any)
    is always stopped and removed on exit -- CodeAgent doesn't clean that
    up on its own otherwise (confirmed live: a plain agent.run() without
    `with` leaves the container running).
    """
    with build_agent(sandbox=sandbox) as agent:
        return agent.run(question)
