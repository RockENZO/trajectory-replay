import asyncio
import os
import sys
from pathlib import Path

from trajectory_replay import Replayer, ReplayMismatch, ToolCall, score, score_outputs
from trajectory_replay.mcp_server import record_session

SERVER = [sys.executable, "-m", "trajectory_replay.mcp_server"]
# The server subprocess must import this package from the same source tree as the tests.
SERVER_ENV = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}


def _record(calls):
    return asyncio.run(record_session(SERVER, calls, env=SERVER_ENV))

CALLS = [
    ("fetch", {"url": "https://shop.example/item"}),
    ("extract", {"text": "PRICE: 12.50", "field": "PRICE"}),
    ("write", {"key": "price", "value": "12.50"}),
]


def test_recorded_mcp_session_replays_exactly():
    live = _record(CALLS)
    assert [c.result for c in live] == ["PRICE: 12.50", "12.50", "ok"]

    # Replay the recorded calls with the same tool names against a replayer, not the server.
    replayer = Replayer(live)
    tools = replayer.tools()
    replayed = [
        ToolCall(name, dict(args), result=tools[name](**args))
        for name, args in CALLS
    ]
    replayer.assert_exhausted()
    assert score(replayed, live).exact
    assert score_outputs(replayed, live).exact == 3


def test_server_errors_are_recorded_as_the_agent_saw_them():
    live = _record([("fetch", {"url": "https://unknown.example"})])
    assert live[0].error is not None
    assert "unknown url" in live[0].error
    assert live[0].result is None


def test_replay_refuses_a_different_argument_type():
    replayer = Replayer([ToolCall("write", {"key": "k", "value": True}, result="ok")])
    try:
        replayer.tools()["write"](key="k", value=1)
    except ReplayMismatch:
        return
    raise AssertionError("expected ReplayMismatch")
