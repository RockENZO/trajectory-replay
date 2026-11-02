# trajectory-replay

Deterministic replay and trajectory regression tests for tool-using LLM agents.

Final-answer checks miss regressions in how an agent reaches its answer. An agent can still
return the right value after calling the wrong tool, repeating a step, or skipping a check.
This project records each run as a cassette of tool calls, replays it without a live model
or network, and scores the agent's trajectory against a gold trajectory.

## What it measures

- **Replay.** A recorded run is served back call by call. Any call that differs from the
  cassette (name or arguments) stops the run with `ReplayMismatch`.
- **Trajectory match.** Predicted and gold tool calls are compared in order, using a longest
  common subsequence on tool name and arguments. Results are ignored. The score reports
  matched steps, precision, recall, F1 and exact match.

Planned, not in this release: latency and token cost per step, and frozen task suites run
against local models.

## Status

This is a skeleton. It contains no model results. The results table below stays empty until a
run has been measured, and each row will record the device, model digest and package versions.

| Suite | Model | Device | Trajectory F1 | Exact match | Date |
| --- | --- | --- | --- | --- | --- |
| _none yet_ | | | | | |

## Quickstart

```bash
uv sync --extra dev
uv run pytest -q
```

Python 3.10 or newer. No API keys, GPU or network access are needed for the tests.

## Example

```python
from trajectory_replay import Replayer, make_tools, run_agent, save, score

fixtures = {"https://shop.example/item": "PRICE: 12.50"}
store = {}

live = run_agent(model, "get the price", make_tools(fixtures, store))  # model: your agent
save(live.trajectory, "cassettes/price.json")

replayer = Replayer(live.trajectory)
replayed = run_agent(model, "get the price", replayer.tools())
replayer.assert_exhausted()
assert score(replayed.trajectory, live.trajectory).exact
```

A model is any callable that takes the chat history and returns either
`{"tool": name, "arguments": {...}}` or `{"final": answer}`.

## Recording another agent's tool log

Any agent that logs each executed tool call (name, arguments, the text it got back, optionally an id)
can be turned into a cassette without importing this package into the agent:

```python
from trajectory_replay import calls_from_records, save

save(calls_from_records(agent_log), "cassettes/task-01.json")
```

Records can be mappings or objects (`name`/`tool`, `arguments`/`args`, `result`/`output`, `call_id`/`id`,
`rejected`). A call that failed is replayed as the text the agent saw; a call a person refused is left out.
For agents whose tools are coroutines, `Replayer.async_tools()` returns async replacements with the same
ordering and mismatch rules. `ToolCall.call_id` is carried through the cassette for linking a step to other
evidence; matching and scoring ignore it.

## Layout

- `src/trajectory_replay/cassette.py`: `ToolCall`, `Replayer`, `save`, `load`
- `src/trajectory_replay/trajectory.py`: `score` and `TrajectoryScore`
- `src/trajectory_replay/agent.py`: `run_agent`, a minimal tool loop
- `src/trajectory_replay/adapters.py`: `calls_from_records`, tool log to cassette
- `src/trajectory_replay/tools.py`: three deterministic tools (fetch, extract, write)
- `tests/`: unit tests, using a scripted fake model

## Limits

- The tools are deterministic test doubles. They do not show how real web pages behave.
- Matching is on exact arguments. Paraphrased or equivalent arguments count as mismatches.
- Trajectory scores do not measure whether the final answer is correct. Use them alongside
  answer checks.
- This complements observability tools such as Phoenix, Langfuse and MLflow. It does not
  replace them.

## License

MIT. See [LICENSE](LICENSE).

## Contributing

Issues and pull requests are welcome. Please keep changes small and include tests. Pull
requests that were generated without human review of the diff will be closed.
