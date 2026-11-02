"""Record tool calls to a cassette file and replay them deterministically."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation: what was asked, and what came back."""

    name: str
    arguments: dict[str, Any]
    result: str | None = None
    error: str | None = None
    # Identifier the recording agent gave this call, if any. It is for linking a step to other
    # evidence (a ledger claim, a log line). Replay and scoring ignore it.
    call_id: str | None = None


class ReplayMismatch(Exception):
    """Raised when a replayed run diverges from the recorded cassette."""


class Replayer:
    """Serves recorded results in order and refuses any call that differs."""

    def __init__(self, cassette: list[ToolCall]) -> None:
        self._steps = list(cassette)
        self._pos = 0

    def tools(self) -> dict[str, Callable[..., str]]:
        """Return replay tools for every tool name in the cassette."""
        names = {step.name for step in self._steps}
        return {name: self._make(name) for name in names}

    def async_tools(self) -> dict[str, Callable[..., Awaitable[str]]]:
        """Like :meth:`tools`, for agents whose tools are coroutines. Same order and mismatch rules."""
        def wrap(fn: Callable[..., str]) -> Callable[..., Awaitable[str]]:
            async def call(**arguments: Any) -> str:
                return fn(**arguments)

            return call

        return {name: wrap(fn) for name, fn in self.tools().items()}

    def _make(self, name: str) -> Callable[..., str]:
        def call(**arguments: Any) -> str:
            if self._pos >= len(self._steps):
                raise ReplayMismatch(f"unexpected extra call: {name}{arguments}")
            step = self._steps[self._pos]
            if step.name != name or step.arguments != arguments:
                raise ReplayMismatch(
                    f"step {self._pos}: recorded {step.name}{step.arguments}, "
                    f"got {name}{arguments}"
                )
            self._pos += 1
            if step.error is not None:
                raise RuntimeError(step.error)
            assert step.result is not None
            return step.result

        return call

    def assert_exhausted(self) -> None:
        """Fail if the run finished before every recorded call was replayed."""
        if self._pos != len(self._steps):
            raise ReplayMismatch(
                f"run ended after {self._pos} of {len(self._steps)} recorded calls"
            )


def save(calls: list[ToolCall], path: str | Path) -> None:
    """Write a cassette as sorted, indented JSON."""
    data = []
    for call in calls:
        item = asdict(call)
        if item["call_id"] is None:  # keep cassettes written before call ids existed byte-identical
            del item["call_id"]
        data.append(item)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load(path: str | Path) -> list[ToolCall]:
    """Read a cassette written by :func:`save`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ToolCall(**item) for item in data]
