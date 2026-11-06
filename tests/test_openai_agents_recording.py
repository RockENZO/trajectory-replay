import importlib.util
from pathlib import Path

import pytest

from trajectory_replay import Replayer, load, score, score_outputs

# Skip only when the optional agents extra is not installed; CI installs it.
pytest.importorskip("agents")

_SPEC = importlib.util.spec_from_file_location(
    "openai_agents_recording",
    Path(__file__).resolve().parents[1] / "examples" / "openai_agents_recording.py",
)
example = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)


def test_agents_run_records_exact_calls_and_results(tmp_path: Path) -> None:
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
    replayed = [example.ToolCall(name, dict(args), result=tools[name](**args)) for name, args in example.PLAN]
    replayer.assert_exhausted()
    assert score(replayed, live).exact
    assert score_outputs(replayed, live).exact == 3
