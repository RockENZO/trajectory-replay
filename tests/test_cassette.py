import pytest

from trajectory_replay import ReplayMismatch, Replayer, ToolCall, load, save

GOLD = [
    ToolCall("fetch", {"url": "u"}, result="PRICE: 3"),
    ToolCall("write", {"key": "k", "value": "3"}, result="ok"),
]


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "cassette.json"
    save(GOLD, path)
    assert load(path) == GOLD


def test_replay_returns_recorded_results():
    replayer = Replayer(GOLD)
    tools = replayer.tools()
    assert tools["fetch"](url="u") == "PRICE: 3"
    assert tools["write"](key="k", value="3") == "ok"
    replayer.assert_exhausted()


def test_replay_rejects_different_arguments():
    tools = Replayer(GOLD).tools()
    with pytest.raises(ReplayMismatch):
        tools["fetch"](url="other")


def test_replay_rejects_extra_calls():
    replayer = Replayer(GOLD)
    tools = replayer.tools()
    tools["fetch"](url="u")
    tools["write"](key="k", value="3")
    with pytest.raises(ReplayMismatch):
        tools["write"](key="k", value="3")


def test_replay_reports_unfinished_runs():
    replayer = Replayer(GOLD)
    replayer.tools()["fetch"](url="u")
    with pytest.raises(ReplayMismatch):
        replayer.assert_exhausted()


def test_recorded_errors_are_replayed_as_errors():
    cassette = [ToolCall("fetch", {"url": "x"}, error="unknown url: x")]
    with pytest.raises(RuntimeError, match="unknown url"):
        Replayer(cassette).tools()["fetch"](url="x")
