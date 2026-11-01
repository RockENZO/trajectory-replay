import pytest

from trajectory_replay import ReplayMismatch, Replayer, make_tools, run_agent, score

FIXTURES = {"https://shop.example/item": "PRICE: 12.50\nNAME: Lamp"}

PLAN = [
    {"tool": "fetch", "arguments": {"url": "https://shop.example/item"}},
    {"tool": "extract", "arguments": {"text": "PRICE: 12.50\nNAME: Lamp", "field": "PRICE"}},
    {"tool": "write", "arguments": {"key": "price", "value": "12.50"}},
    {"final": "saved 12.50"},
]


def scripted(actions):
    """A fake model that replays a fixed list of actions."""
    iterator = iter(actions)

    def model(history):
        return next(iterator)

    return model


def test_agent_completes_task_and_records_trajectory():
    store = {}
    result = run_agent(scripted(PLAN), "get the price", make_tools(FIXTURES, store))
    assert result.stopped == "final"
    assert result.answer == "saved 12.50"
    assert store == {"price": "12.50"}
    assert [call.name for call in result.trajectory] == ["fetch", "extract", "write"]
    assert result.trajectory[0].result == "PRICE: 12.50\nNAME: Lamp"


def test_tool_errors_are_recorded_and_the_loop_continues():
    plan = [
        {"tool": "fetch", "arguments": {"url": "https://nope.example"}},
        {"final": "gave up"},
    ]
    result = run_agent(scripted(plan), "task", make_tools(FIXTURES, {}))
    assert result.stopped == "final"
    assert result.trajectory[0].error == "unknown url: https://nope.example"


def test_unknown_tools_are_recorded_as_errors():
    plan = [{"tool": "delete_everything", "arguments": {}}, {"final": "stopped"}]
    result = run_agent(scripted(plan), "task", make_tools(FIXTURES, {}))
    assert result.trajectory[0].error == "unknown tool"


def test_max_steps_stops_runaway_loops():
    def looping(history):
        return {"tool": "fetch", "arguments": {"url": "https://shop.example/item"}}

    result = run_agent(looping, "task", make_tools(FIXTURES, {}), max_steps=3)
    assert result.stopped == "max_steps"
    assert result.answer is None
    assert len(result.trajectory) == 3


def test_replay_reproduces_the_live_trajectory():
    live = run_agent(scripted(PLAN), "get the price", make_tools(FIXTURES, {}))
    replayer = Replayer(live.trajectory)
    replayed = run_agent(scripted(PLAN), "get the price", replayer.tools())
    replayer.assert_exhausted()
    assert replayed.answer == live.answer
    assert score(replayed.trajectory, live.trajectory).exact


def test_replay_stops_at_the_first_regression():
    live = run_agent(scripted(PLAN), "get the price", make_tools(FIXTURES, {}))
    regressed = [
        PLAN[0],
        {"tool": "write", "arguments": {"key": "price", "value": "12.50"}},
        {"final": "saved 12.50"},
    ]
    with pytest.raises(ReplayMismatch):
        run_agent(scripted(regressed), "get the price", Replayer(live.trajectory).tools())
