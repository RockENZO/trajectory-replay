"""Turn another agent's tool log into a cassette.

The agent only has to expose, per executed call, a name, its arguments and the text it got back.
Records may be mappings or objects with those attributes. Nothing here imports any agent.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .cassette import ToolCall


def _get(record: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(record, Mapping):
            if name in record:
                return record[name]
        elif hasattr(record, name):
            return getattr(record, name)
    return default


def calls_from_records(records: Iterable[Any], *, include_rejected: bool = False) -> list[ToolCall]:
    """Convert logged tool executions to ``ToolCall`` steps, in order.

    Field names read from each record: ``name`` (or ``tool``), ``arguments`` (or ``args``),
    ``result`` (or ``output``), and optionally ``call_id`` (or ``id``) and ``rejected``.

    The result is replayed as the text the agent saw, including for a call that failed:
    an agent that gets an error back as text must get the same text back on replay.
    A call that a person refused was never executed, so it is left out unless
    ``include_rejected`` is true.
    """
    calls: list[ToolCall] = []
    for index, record in enumerate(records):
        if _get(record, "rejected", default=False) and not include_rejected:
            continue
        name = _get(record, "name", "tool")
        if not name:
            raise ValueError(f"record {index} has no tool name")
        arguments = _get(record, "arguments", "args", default={})
        result = _get(record, "result", "output", default="")
        call_id = _get(record, "call_id", "id")
        calls.append(ToolCall(
            name=str(name),
            arguments=dict(arguments or {}),
            result="" if result is None else str(result),
            call_id=None if call_id is None else str(call_id),
        ))
    return calls
