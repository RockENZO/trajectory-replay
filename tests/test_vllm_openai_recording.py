"""Tests for the OpenAI-compatible recorder.

A small stub server speaks the ``/v1/chat/completions`` wire format and returns scripted tool
calls, so these tests run offline. They do not exercise a real vLLM server or model weights.
"""

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("openai")  # the optional openai extra; CI installs it

from trajectory_replay import Replayer, ReplayMismatch, ToolCall, load, score, score_outputs

_SPEC = importlib.util.spec_from_file_location(
    "vllm_openai_recording",
    Path(__file__).resolve().parents[1] / "examples" / "vllm_openai_recording.py",
)
example = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)


def _completion(message: dict, finish_reason: str) -> bytes:
    body = {
        "id": "cmpl-stub",
        "object": "chat.completion",
        "created": 0,
        "model": "stub",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
    return json.dumps(body).encode()


def _serve(plan: list[tuple[str, dict]]):
    """Return a running stub server. Request i returns plan[i] as a tool call, then a final answer."""
    state = {"turn": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server naming
            length = int(self.headers.get("Content-Length", 0))
            json.loads(self.rfile.read(length) or b"{}")  # the request must be valid JSON
            turn = state["turn"]
            state["turn"] += 1
            if turn < len(plan):
                name, arguments = plan[turn]
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": f"call_{turn}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }],
                }
                payload = _completion(message, "tool_calls")
            else:
                payload = _completion({"role": "assistant", "content": "done"}, "stop")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):  # keep test output quiet
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def stub(request):
    plan = request.param
    server = _serve(plan)
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()
    server.server_close()


@pytest.mark.parametrize("stub", [example.PLAN], indirect=True)
def test_openai_compatible_run_records_exact_calls_and_results(stub, tmp_path: Path) -> None:
    out = tmp_path / "price.json"
    recorded = example.record(stub, out)
    assert [(c.name, c.arguments) for c in recorded] == example.PLAN
    assert [c.result for c in recorded] == ["PRICE: 12.50", "12.50", "ok"]
    assert load(out) == recorded


@pytest.mark.parametrize("stub", [example.PLAN], indirect=True)
def test_recorded_run_replays_exactly(stub, tmp_path: Path) -> None:
    out = tmp_path / "price.json"
    live = example.record(stub, out)
    replayer = Replayer(load(out))
    tools = replayer.tools()
    replayed = [ToolCall(name, dict(args), result=tools[name](**args)) for name, args in example.PLAN]
    replayer.assert_exhausted()
    assert score(replayed, live).exact
    assert score_outputs(replayed, live).exact == 3


@pytest.mark.parametrize("stub", [[("fetch", {"url": "https://missing.example/item"})]], indirect=True)
def test_failed_tool_records_the_text_the_model_received(stub, tmp_path: Path) -> None:
    out = tmp_path / "missing.json"
    recorded = example.record(stub, out)
    assert len(recorded) == 1
    assert recorded[0].error is None
    assert recorded[0].result == "error: unknown url: https://missing.example/item"
    replayer = Replayer(load(out))
    assert replayer.tools()["fetch"](url="https://missing.example/item") == recorded[0].result



def test_replay_mismatch_stops_the_run_instead_of_reaching_the_model(tmp_path: Path) -> None:
    out = tmp_path / "price.json"
    recording_server = _serve(example.PLAN)
    try:
        example.record(f"http://127.0.0.1:{recording_server.server_address[1]}/v1", out)
    finally:
        recording_server.shutdown()
        recording_server.server_close()

    replayer = Replayer(load(out))
    # This model asks for a different URL than the cassette recorded first.
    diverging_server = _serve([("fetch", {"url": "https://other.example/item"})])
    try:
        with pytest.raises(ReplayMismatch):
            example.run(f"http://127.0.0.1:{diverging_server.server_address[1]}/v1",
                        "Qwen/Qwen2.5-7B-Instruct", "get the price", replayer.tools())
    finally:
        diverging_server.shutdown()
        diverging_server.server_close()
