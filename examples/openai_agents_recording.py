"""Record an OpenAI Agents SDK run into a cassette.

The agent uses the three test tools (fetch, extract, write) as SDK function tools. Each tool
records its own name, arguments and result as the agent saw them. A scripted ``Model`` stands
in for the LLM, so the example runs offline with no API key.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from agents import Agent, Model, ModelResponse, Runner, Usage, function_tool
from openai.types.responses import ResponseFunctionToolCall

from trajectory_replay import ToolCall, make_tools, save

FIXTURES = {"https://shop.example/item": "PRICE: 12.50"}


class ScriptedModel(Model):
    """Returns the planned tool calls one per turn, then a final answer."""

    def __init__(self, plan: list[tuple[str, dict[str, Any]]]) -> None:
        self._plan = list(plan)
        self._n = 0

    async def get_response(self, system_instructions, input, model_settings, tools, output_schema,
                           handoffs, tracing, *, previous_response_id, conversation_id, prompt):
        self._n += 1
        if self._plan:
            name, arguments = self._plan.pop(0)
            item = ResponseFunctionToolCall(
                type="function_call", call_id=f"call_{self._n}", name=name,
                arguments=json.dumps(arguments), status="completed",
            )
            return ModelResponse(output=[item], usage=Usage(), response_id=None)
        from openai.types.responses import ResponseOutputMessage, ResponseOutputText
        msg = ResponseOutputMessage(
            id=f"msg_{self._n}", type="message", role="assistant", status="completed",
            content=[ResponseOutputText(type="output_text", text="done", annotations=[])],
        )
        return ModelResponse(output=[msg], usage=Usage(), response_id=None)

    async def stream_response(self, *args, **kwargs):
        raise NotImplementedError("the example does not stream")


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


def build_agent(store: dict[str, str], plan, recorder: Recorder) -> Agent:
    tools = make_tools(FIXTURES, store)
    fetch_fn = recorder.wrap("fetch", tools["fetch"])
    extract_fn = recorder.wrap("extract", tools["extract"])
    write_fn = recorder.wrap("write", tools["write"])

    @function_tool
    def fetch(url: str) -> str:
        """Fetch a fixture page by URL."""
        return fetch_fn(url=url)

    @function_tool
    def extract(text: str, field: str) -> str:
        """Extract the value of a field from text."""
        return extract_fn(text=text, field=field)

    @function_tool
    def write(key: str, value: str) -> str:
        """Store a value under a key."""
        return write_fn(key=key, value=value)

    return Agent(name="price-agent", instructions="Get the price and store it.",
                 tools=[fetch, extract, write], model=ScriptedModel(plan))


PLAN = [
    ("fetch", {"url": "https://shop.example/item"}),
    ("extract", {"text": "PRICE: 12.50", "field": "PRICE"}),
    ("write", {"key": "price", "value": "12.50"}),
]


def record(cassette_path: Path) -> list[ToolCall]:
    store: dict[str, str] = {}
    recorder = Recorder()
    agent = build_agent(store, list(PLAN), recorder)
    Runner.run_sync(agent, "get the price", max_turns=6)
    save(recorder.calls, cassette_path)
    return recorder.calls


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "cassettes/openai-agents-price.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    for call in record(out):
        print(call.name, "->", call.result)
