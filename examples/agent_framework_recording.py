"""Record a Microsoft Agent Framework run into a cassette.

The agent's three tools (fetch, extract, write) are Agent Framework tools that record their
own name, arguments and result. A scripted chat client stands in for the LLM, so the example
runs offline with no API key.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any, Sequence

from agent_framework import BaseChatClient, ChatResponse, Content, FunctionInvocationLayer, Message, tool

from trajectory_replay import ToolCall, make_tools, save

FIXTURES = {"https://shop.example/item": "PRICE: 12.50"}

PLAN = [
    ("fetch", {"url": "https://shop.example/item"}),
    ("extract", {"text": "PRICE: 12.50", "field": "PRICE"}),
    ("write", {"key": "price", "value": "12.50"}),
]


class ScriptedChatClient(FunctionInvocationLayer, BaseChatClient):
    """Answers with the planned tool calls, one per turn, then a final text answer."""

    def __init__(self, plan: list[tuple[str, dict[str, Any]]]) -> None:
        super().__init__()
        self._plan = list(plan)
        self._turn = 0

    async def _inner_get_response(self, *, messages: Sequence[Message], stream: bool,
                                  options: Any, **kwargs: Any) -> ChatResponse:
        self._turn += 1
        if self._plan:
            name, arguments = self._plan.pop(0)
            content = Content.from_function_call(f"call_{self._turn}", name, arguments=arguments)
            return ChatResponse(messages=[Message(role="assistant", contents=[content])])
        return ChatResponse(messages=[Message(role="assistant", contents=["done"])])


class Recorder:
    """Keeps each tool call, with its arguments and the result the agent saw, in order."""

    def __init__(self) -> None:
        self.calls: list[ToolCall] = []

    def wrap(self, name: str, fn):
        def call(**arguments: Any) -> str:
            try:
                result = fn(**arguments)
            except Exception as exc:  # a tool error is part of the trajectory
                self.calls.append(ToolCall(name, dict(arguments), error=str(exc)))
                raise
            self.calls.append(ToolCall(name, dict(arguments), result=result))
            return result
        return call


def build_agent(recorder: Recorder, plan, store: dict[str, str]):
    from agent_framework import Agent

    tools = make_tools(FIXTURES, store)
    fetch_fn = recorder.wrap("fetch", tools["fetch"])
    extract_fn = recorder.wrap("extract", tools["extract"])
    write_fn = recorder.wrap("write", tools["write"])

    @tool(name="fetch", description="Fetch a fixture page by URL.")
    def fetch(url: Annotated[str, "The page URL"]) -> str:
        return fetch_fn(url=url)

    @tool(name="extract", description="Extract the value of a field from text.")
    def extract(text: Annotated[str, "The text"], field: Annotated[str, "The field name"]) -> str:
        return extract_fn(text=text, field=field)

    @tool(name="write", description="Store a value under a key.")
    def write(key: Annotated[str, "The key"], value: Annotated[str, "The value"]) -> str:
        return write_fn(key=key, value=value)

    return Agent(client=ScriptedChatClient(plan), name="price-agent",
                 instructions="Get the price and store it.", tools=[fetch, extract, write])


async def _record_async(cassette_path: Path) -> list[ToolCall]:
    recorder = Recorder()
    agent = build_agent(recorder, list(PLAN), {})
    await agent.run("get the price")
    return recorder.calls


def record(cassette_path: Path) -> list[ToolCall]:
    calls = asyncio.run(_record_async(cassette_path))
    save(calls, cassette_path)
    return calls


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "cassettes/agent-framework-price.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    for call in record(out):
        print(call.name, "->", call.result)
