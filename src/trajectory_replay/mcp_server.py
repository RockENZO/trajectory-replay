"""Expose the deterministic test tools as a real MCP server, and record MCP sessions.

Run the server over stdio with ``python -m trajectory_replay.mcp_server``. ``record_session``
connects to any stdio MCP server as a client, calls tools in order, and returns the calls as
``ToolCall`` records. Those can be saved as a cassette and replayed with ``Replayer``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

from .cassette import ToolCall
from .tools import ToolError, make_tools

DEMO_FIXTURES = {"https://shop.example/item": "PRICE: 12.50"}


def create_server(fixtures: dict[str, str] | None = None, store: dict[str, str] | None = None):
    """Build a FastMCP server whose tools are fetch, extract and write."""
    from mcp.server.fastmcp import FastMCP

    tools = make_tools(DEMO_FIXTURES if fixtures is None else fixtures, {} if store is None else store)
    server = FastMCP("trajectory-replay-tools")
    for name in ("fetch", "extract", "write"):
        server.add_tool(tools[name], name=name)
    return server


async def record_session(
    command: Sequence[str],
    calls: Sequence[tuple[str, dict[str, Any]]],
    env: dict[str, str] | None = None,
) -> list[ToolCall]:
    """Call tools on a stdio MCP server in order and record what each returned.

    A tool error reported by the server is recorded with ``error`` set, as the agent saw it.
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=command[0], args=list(command[1:]), env=env)
    recorded: list[ToolCall] = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for name, arguments in calls:
                result = await session.call_tool(name, arguments)
                text = "".join(getattr(part, "text", "") for part in result.content)
                if result.isError:
                    recorded.append(ToolCall(name, dict(arguments), error=text or "tool error"))
                else:
                    recorded.append(ToolCall(name, dict(arguments), result=text))
    return recorded


def main() -> None:
    create_server().run()


if __name__ == "__main__":
    sys.exit(main())
