from trajectory_replay import ToolCall, score


def c(name: str, result: str = "x", **arguments: str) -> ToolCall:
    return ToolCall(name, dict(arguments), result=result)


def test_exact_match():
    gold = [c("fetch", url="a"), c("extract", text="t", field="price")]
    s = score(gold, gold)
    assert s.exact
    assert s.f1 == 1.0


def test_partial_match_keeps_gold_order():
    gold = [c("fetch", url="a"), c("extract", text="t", field="price"), c("write", key="k", value="v")]
    predicted = [c("fetch", url="a"), c("write", key="k", value="v")]
    s = score(predicted, gold)
    assert s.matched == 2
    assert s.precision == 1.0
    assert abs(s.recall - 2 / 3) < 1e-9
    assert not s.exact


def test_reordered_calls_do_not_count_as_matches():
    gold = [c("fetch", url="a"), c("extract", text="t", field="price")]
    predicted = [c("extract", text="t", field="price"), c("fetch", url="a")]
    assert score(predicted, gold).matched == 1


def test_argument_differences_are_mismatches():
    assert score([c("fetch", url="b")], [c("fetch", url="a")]).matched == 0


def test_results_are_ignored():
    gold = [ToolCall("fetch", {"url": "a"}, result="old")]
    predicted = [ToolCall("fetch", {"url": "a"}, result="new")]
    assert score(predicted, gold).exact


def test_empty_trajectories_are_exact():
    s = score([], [])
    assert s.exact
    assert s.f1 == 1.0


def test_extra_calls_lower_precision():
    gold = [c("fetch", url="a")]
    predicted = [c("fetch", url="a"), c("fetch", url="a")]
    s = score(predicted, gold)
    assert s.matched == 1
    assert s.precision == 0.5
    assert s.recall == 1.0
