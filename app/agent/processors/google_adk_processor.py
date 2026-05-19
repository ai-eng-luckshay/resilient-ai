"""
GoogleADKProcessor — extension point for Google Agent Development Kit.

STATUS: Functional stub.

What this would wire:
    from google.adk import Agent, Runner
    from google.adk.sessions import InMemorySessionService

    agent = Agent(
        name="gateway-agent",
        model="gemini-2.0-flash",
        tools=ALL_TOOLS,
        instruction=request.system_prompt,
    )
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, session_service=session_service, ...)

    async for event in runner.run_async(user_id=..., session_id=..., message=...):
        if event.is_final_response():
            yield ndjson("TEXT", event.text)

    The tool binding pattern is identical to LangGraphProcessor — tools from
    assistant_tools.py are passed directly into the ADK Agent constructor.

How to activate:
    1. pip install google-adk
    2. Set AGENT_PROCESSOR=GOOGLE_ADK in .env
    3. Set GEMINI_API_KEY in .env
    4. Implement the session lifecycle and event mapping below.
"""
import json
import logging
from typing import AsyncIterator

from app.agent.processors.base import BaseProcessor
from app.config.logger import get_trace_id
from app.models.schemas import ChatRequest

logger = logging.getLogger(__name__)

_INFO_MESSAGE = (
    "Google ADK processor is configured as an extension point. "
    "Install google-adk, implement session lifecycle in google_adk_processor.py, "
    "and set AGENT_PROCESSOR=GOOGLE_ADK to activate."
)


class GoogleADKProcessor(BaseProcessor):
    async def stream(self, request: ChatRequest, history: list) -> AsyncIterator[str]:
        trace = get_trace_id()
        logger.info("[%s] GoogleADKProcessor invoked (stub)", trace)

        yield json.dumps({"type": "STATUS", "content": "GOOGLE_ADK_STUB"}) + "\n"
        yield json.dumps({"type": "TEXT", "content": _INFO_MESSAGE}) + "\n"
        yield json.dumps({"type": "STATUS", "content": "COMPLETE"}) + "\n"
        # In production: replace above with ADK runner.run_async() event loop
        raise NotImplementedError(
            "GoogleADKProcessor is a stub. See docstring for wiring instructions."
        )
