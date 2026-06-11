"""Agent entrypoint for one-step Tic-Tac-Toe tool-call rollouts."""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are a careful Tic-Tac-Toe player. You play X. "
    "Choose exactly one legal square by calling the choose_square tool."
)

CHOOSE_SQUARE_TOOL = {
    "type": "function",
    "function": {
        "name": "choose_square",
        "description": "Choose the next Tic-Tac-Toe square for X.",
        "parameters": {
            "type": "object",
            "properties": {
                "square": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 9,
                    "description": "The square number to place X in.",
                }
            },
            "required": ["square"],
            "additionalProperties": False,
        },
    },
}


async def run_agent(ctx, batch) -> None:
    """Run one tool-call model request for each board."""

    try:
        from openai import AsyncOpenAI
        import httpx
    except ImportError as exc:
        raise RuntimeError("The Tic-Tac-Toe agentic example requires `openai` and `httpx`. Install them with `pip install openai`.") from exc

    items = list(batch.iter_samples())
    logger.info("Tic-Tac-Toe agent start requests=%d max_running_prompts=%d", len(items), ctx.max_running_prompts)
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=300.0,
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)

    async def run_one(item) -> None:
        await client.chat.completions.create(
            model="policy",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item.prompt},
            ],
            tools=[CHOOSE_SQUARE_TOOL],
            tool_choice={"type": "function", "function": {"name": "choose_square"}},
            stream=False,
        )

    try:
        await asyncio.gather(*(run_one(item) for item in items))
    finally:
        await client.close()
