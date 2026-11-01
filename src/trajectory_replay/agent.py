"""A minimal tool-using agent loop driven by any model callable."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .cassette import ReplayMismatch, ToolCall

# A model takes the chat history and returns either
# {"tool": name, "arguments": {...}} or {"final": answer}.
ModelFn = Callable[[list[dict[str, Any]]], dict[str, Any]]


@dataclass(frozen=True)
class RunResult:
    answer: str | None
    trajectory: list[ToolCall]
    stopped: str  # "final" or "max_steps"


def run_agent(
    model: ModelFn,
    task: str,
    tools: dict[str, Callable[..., str]],
    max_steps: int = 8,
) -> RunResult:
    """Run the model until it answers, or until max_steps is reached.

    Tool errors are recorded in the trajectory and returned to the model.
    ReplayMismatch is not caught, so a replay stops at the first divergence.
    """
    history: list[dict[str, Any]] = [{"role": "user", "content": task}]
    trajectory: list[ToolCall] = []
    for _ in range(max_steps):
        action = model(history)
        if "final" in action:
            return RunResult(answer=action["final"], trajectory=trajectory, stopped="final")

        name = action.get("tool", "")
        arguments = action.get("arguments", {})
        if name not in tools:
            trajectory.append(ToolCall(name, arguments, error="unknown tool"))
            history.append({"role": "tool", "name": name, "content": "error: unknown tool"})
            continue
        try:
            result = tools[name](**arguments)
        except ReplayMismatch:
            raise
        except Exception as exc:  # tool failures are data for the agent, not crashes
            trajectory.append(ToolCall(name, arguments, error=str(exc)))
            history.append({"role": "tool", "name": name, "content": f"error: {exc}"})
            continue
        trajectory.append(ToolCall(name, arguments, result=result))
        history.append({"role": "tool", "name": name, "content": result})
    return RunResult(answer=None, trajectory=trajectory, stopped="max_steps")
