from trajectory_replay import ToolCall, aligned_pairs, output_similarity, score, score_outputs


def c(name: str, result: str | None = "x", **arguments: str) -> ToolCall:
    return ToolCall(name, dict(arguments), result=result)


def test_identical_results_are_exact():
    gold = [c("fetch", "PRICE: 12.50", url="a")]
    s = score_outputs(gold, gold)
    assert (s.compared, s.exact, s.mean_similarity) == (1, 1, 1.0)
    assert s.exact_rate == 1.0


def test_whitespace_and_case_do_not_count_as_differences():
    gold = [c("fetch", "Price:  12.50\n", url="a")]
    pred = [c("fetch", "price: 12.50", url="a")]
    s = score_outputs(pred, gold)
    assert s.exact == 1
    assert s.mean_similarity == 1.0


def test_different_results_on_matched_calls_lower_similarity():
    gold = [c("fetch", "PRICE: 12.50", url="a")]
    pred = [c("fetch", "PRICE: 12.05", url="a")]
    s = score_outputs(pred, gold)
    assert s.compared == 1
    assert s.exact == 0
    assert 0.0 < s.mean_similarity < 1.0


def test_unaligned_calls_are_not_compared():
    gold = [c("fetch", "a", url="a"), c("extract", "b", text="t")]
    pred = [c("fetch", "zzz", url="other")]
    s = score_outputs(pred, gold)
    assert s.compared == 0
    assert s.mean_similarity == 0.0
    assert score(pred, gold).matched == 0


def test_empty_and_missing_results():
    assert output_similarity(None, "") == 1.0
    assert output_similarity("", "abc") == 0.0


def test_aligned_pairs_follow_gold_order():
    gold = [c("fetch", url="a"), c("extract", text="t", field="p"), c("write", key="k", value="v")]
    pred = [c("fetch", url="a"), c("write", key="k", value="v")]
    assert aligned_pairs(pred, gold) == [(0, 0), (1, 2)]
