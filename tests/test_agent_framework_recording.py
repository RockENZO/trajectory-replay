import importlib.util
from pathlib import Path

from trajectory_replay import Replayer, ToolCall, load, score, score_outputs

_SPEC = importlib.util.spec_from_file_location(
    "agent_framework_recording",
    Path(__file__).resolve().parents[1] / "examples" / "agent_framework_recording.py",
)
example = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)


def test_agent_framework_run_records_exact_calls_and_results(tmp_path: Path) -> None:
    out = tmp_path / "price.json"
    recorded = example.record(out)
    assert [(c.name, c.arguments) for c in recorded] == example.PLAN
    assert [c.result for c in recorded] == ["PRICE: 12.50", "12.50", "ok"]
    assert load(out) == recorded


def test_recorded_run_replays_exactly(tmp_path: Path) -> None:
    out = tmp_path / "price.json"
    live = example.record(out)
    replayer = Replayer(load(out))
    tools = replayer.tools()
    replayed = [ToolCall(name, dict(args), result=tools[name](**args)) for name, args in example.PLAN]
    replayer.assert_exhausted()
    assert score(replayed, live).exact
    assert score_outputs(replayed, live).exact == 3


def test_failed_tool_records_what_the_model_saw(tmp_path: Path, monkeypatch) -> None:
    seen = []
    original = example.ScriptedChatClient._inner_get_response

    async def spy(self, *, messages, stream, options, **kwargs):
        seen.append(list(messages))
        return await original(self, messages=messages, stream=stream, options=options, **kwargs)

    monkeypatch.setattr(example.ScriptedChatClient, "_inner_get_response", spy)
    out = tmp_path / "missing.json"
    plan = [("fetch", {"url": "https://missing.example/item"})]
    recorded = example.record(out, plan=plan)
    assert len(recorded) == 1
    assert recorded[0].error is None
    assert recorded[0].result == example.FAILURE_TEXT
    # The text the model received for the failed call matches the recorded result.
    model_saw = [
        c.result for msgs in seen for m in msgs for c in m.contents if c.type == "function_result"
    ]
    assert model_saw == [example.FAILURE_TEXT]
    replayer = Replayer(load(out))
    assert replayer.tools()["fetch"](url="https://missing.example/item") == example.FAILURE_TEXT
