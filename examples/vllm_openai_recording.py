"""Record a tool-using run against an OpenAI-compatible server such as vLLM.

vLLM serves an OpenAI-compatible ``/v1/chat/completions`` endpoint. This example talks to
that endpoint with the ``openai`` client and runs the three test tools (fetch, extract, write)
through the standard tool-calling loop. Each executed call is recorded with the result text the
model received, so the cassette replays exactly.

Point ``base_url`` at a real vLLM server (for example ``http://localhost:8000/v1`` after
``vllm serve Qwen/...``) to record a live run. The tests use a small local stub server that
speaks the same wire format, so they run offline with no GPU and no model weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from trajectory_replay import ToolCall, make_tools, save

FIXTURES = {"https://shop.example/item": "PRICE: 12.50"}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "fetch",
            "description": "Fetch a fixture page by URL.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract",
            "description": "Extract the value of a field from text.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "field": {"type": "string"}},
                "required": ["text", "field"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Store a value under a key.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                "required": ["key", "value"],
            },
        },
    },
]

PLAN = [
    ("fetch", {"url": "https://shop.example/item"}),
    ("extract", {"text": "PRICE: 12.50", "field": "PRICE"}),
    ("write", {"key": "price", "value": "12.50"}),
]

# Our own loop sends this text to the model when a tool raises. It is recorded as the result.
ERROR_PREFIX = "error: "


def run(base_url: str, model: str, task: str, tools: dict[str, Any], max_steps: int = 8,
        api_key: str = "EMPTY") -> list[ToolCall]:
    """Run the chat loop until the model stops asking for tools, and return the trajectory."""
    client = OpenAI(base_url=base_url, api_key=api_key)
    messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
    trajectory: list[ToolCall] = []
    for _ in range(max_steps):
        response = client.chat.completions.create(model=model, messages=messages, tools=TOOL_SCHEMAS)
        message = response.choices[0].message
        if not message.tool_calls:
            return trajectory
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in message.tool_calls
            ],
        })
        for call in message.tool_calls:
            name = call.function.name
            arguments = json.loads(call.function.arguments or "{}")
            try:
                result = tools[name](**arguments)
            except Exception as exc:  # a tool error is part of the trajectory
                text = f"{ERROR_PREFIX}{exc}"
                trajectory.append(ToolCall(name, arguments, result=text))
            else:
                text = result
                trajectory.append(ToolCall(name, arguments, result=result))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
    return trajectory


def record(base_url: str, cassette_path: Path, model: str = "Qwen/Qwen2.5-7B-Instruct",
           task: str = "get the price") -> list[ToolCall]:
    store: dict[str, str] = {}
    calls = run(base_url, model, task, make_tools(FIXTURES, store))
    save(calls, cassette_path)
    return calls


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url", help="for example http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--out", default="cassettes/vllm-price.json")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for call in record(args.base_url, out, args.model):
        print(call.name, "->", call.result)
