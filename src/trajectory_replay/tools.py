"""Three deterministic tools for the test agent. No network access."""

from __future__ import annotations

import re
from collections.abc import Callable


class ToolError(Exception):
    """A tool could not complete the request."""


def make_tools(fixtures: dict[str, str], store: dict[str, str]) -> dict[str, Callable[..., str]]:
    """Build fetch, extract and write over a fixture map and an in-memory store."""

    def fetch(url: str) -> str:
        if url not in fixtures:
            raise ToolError(f"unknown url: {url}")
        return fixtures[url]

    def extract(text: str, field: str) -> str:
        pattern = re.compile(rf"^{re.escape(field)}:\s*(.+)$", re.MULTILINE)
        match = pattern.search(text)
        if match is None:
            raise ToolError(f"field not found: {field}")
        return match.group(1).strip()

    def write(key: str, value: str) -> str:
        store[key] = value
        return "ok"

    return {"fetch": fetch, "extract": extract, "write": write}
