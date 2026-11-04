"""Compare an agent's tool-call trajectory with a gold trajectory.

Calls match on tool name and arguments only. Results are ignored, so a
trajectory is judged on what the agent asked for, in what order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .cassette import ToolCall


@dataclass(frozen=True)
class TrajectoryScore:
    matched: int
    predicted_len: int
    gold_len: int

    @property
    def precision(self) -> float:
        if self.predicted_len == 0:
            return 1.0 if self.gold_len == 0 else 0.0
        return self.matched / self.predicted_len

    @property
    def recall(self) -> float:
        if self.gold_len == 0:
            return 1.0
        return self.matched / self.gold_len

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 0.0 if p + r == 0 else 2 * p * r / (p + r)

    @property
    def exact(self) -> bool:
        return self.matched == self.predicted_len == self.gold_len


def _key(call: ToolCall) -> tuple[str, str]:
    return call.name, json.dumps(call.arguments, sort_keys=True)


def aligned_pairs(predicted: list[ToolCall], gold: list[ToolCall]) -> list[tuple[int, int]]:
    """Return (predicted index, gold index) pairs of a longest common subsequence on calls.

    Pairs are in increasing order on both sides. Calls match on name and arguments only.
    """
    p_keys = [_key(c) for c in predicted]
    g_keys = [_key(c) for c in gold]
    rows, cols = len(p_keys) + 1, len(g_keys) + 1
    table = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        for j in range(1, cols):
            if p_keys[i - 1] == g_keys[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    pairs: list[tuple[int, int]] = []
    i, j = len(p_keys), len(g_keys)
    while i > 0 and j > 0:
        if p_keys[i - 1] == g_keys[j - 1]:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif table[i - 1][j] >= table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def score(predicted: list[ToolCall], gold: list[ToolCall]) -> TrajectoryScore:
    """Count gold-order matches using a longest common subsequence."""
    return TrajectoryScore(
        matched=len(aligned_pairs(predicted, gold)),
        predicted_len=len(predicted),
        gold_len=len(gold),
    )
