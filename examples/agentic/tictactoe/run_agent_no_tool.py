"""Agent entrypoint for one-step Tic-Tac-Toe XML rollouts."""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are a careful Tic-Tac-Toe player. You play X. "
    "Answer with exactly one XML tag such as <move>5</move>."
)


async def run_agent(ctx, batch) -> None:
    """Run one XML-response model request for each board."""

    try:
        from openai import AsyncOpenAI
        import httpx
    except ImportError as exc:
        raise RuntimeError("The Tic-Tac-Toe agentic example requires `openai` and `httpx`. Install them with `pip install openai`.") from exc

    items = list(batch.iter_samples())
    logger.info("Tic-Tac-Toe XML agent start requests=%d max_running_prompts=%d", len(items), ctx.max_running_prompts)
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
            stream=False,
        )

    try:
        await asyncio.gather(*(run_one(item) for item in items))
    finally:
        await client.close()
