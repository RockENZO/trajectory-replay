import asyncio
from dataclasses import dataclass

import pytest

from trajectory_replay import (
    ReplayMismatch,
    Replayer,
    ToolCall,
    calls_from_records,
    load,
    save,
    score,
)


@dataclass
class Record:  # shaped like an agent's own log entry
    call_id: str
    name: str
    arguments: dict
    result: str
    ok: bool = True
    rejected: bool = False


def test_converts_objects_and_mappings_in_order():
    records = [
        Record("c1", "read", {"path": "a"}, "alpha"),
        {"tool": "bash", "args": {"command": "ls"}, "output": "a.txt", "id": "c2"},
    ]
    assert calls_from_records(records) == [
        ToolCall("read", {"path": "a"}, result="alpha", call_id="c1"),
        ToolCall("bash", {"command": "ls"}, result="a.txt", call_id="c2"),
    ]


def test_a_failed_call_replays_as_the_text_the_agent_saw():
    [call] = calls_from_records([Record("c1", "bash", {"command": "exit 3"}, "[exit code 3]", ok=False)])
    assert call.result == "[exit code 3]" and call.error is None


def test_rejected_calls_were_never_run_so_they_are_skipped_by_default():
    records = [Record("c1", "write", {"path": "x"}, "Rejected", rejected=True), Record("c2", "read", {"path": "x"}, "ok")]
    assert [c.call_id for c in calls_from_records(records)] == ["c2"]
    assert [c.call_id for c in calls_from_records(records, include_rejected=True)] == ["c1", "c2"]


def test_a_record_without_a_name_is_an_error():
    with pytest.raises(ValueError, match="record 0"):
        calls_from_records([{"arguments": {}}])


def test_call_ids_roundtrip_but_do_not_affect_matching_or_scoring(tmp_path):
    with_ids = [ToolCall("fetch", {"url": "u"}, result="r", call_id="abc")]
    without_ids = [ToolCall("fetch", {"url": "u"}, result="r")]
    path = tmp_path / "c.json"
    save(with_ids, path)
    assert load(path) == with_ids
    assert score(with_ids, without_ids).exact
    assert Replayer(with_ids).tools()["fetch"](url="u") == "r"


def test_cassettes_without_ids_are_written_exactly_as_before(tmp_path):
    path = tmp_path / "c.json"
    save([ToolCall("fetch", {"url": "u"}, result="r")], path)
    assert "call_id" not in path.read_text()


def test_async_tools_replay_in_order_and_refuse_divergence():
    replayer = Replayer([ToolCall("a", {"x": 1}, result="one"), ToolCall("b", {}, result="two")])
    tools = replayer.async_tools()

    async def go():
        assert await tools["a"](x=1) == "one"
        with pytest.raises(ReplayMismatch):
            await tools["a"](x=1)  # the next recorded step is "b"
        assert await tools["b"]() == "two"

    asyncio.run(go())
    replayer.assert_exhausted()
