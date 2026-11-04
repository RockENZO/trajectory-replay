"""Compare the tool results of aligned calls, in addition to the call trajectory.

Trajectory scoring ignores results. This module scores what a tool returned for the
calls that matched the gold trajectory. It is a text comparison, not a judgement of
whether the agent's final answer is correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from .cassette import ToolCall
from .trajectory import aligned_pairs


def normalize_output(text: str | None) -> str:
    """Casefold and collapse runs of whitespace. ``None`` becomes the empty string."""
    return " ".join((text or "").split()).casefold()


def output_similarity(predicted: str | None, gold: str | None) -> float:
    """Similarity in [0, 1] between two normalized tool results (1.0 means identical)."""
    a, b = normalize_output(predicted), normalize_output(gold)
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


@dataclass(frozen=True)
class OutputScore:
    """Result comparison over the calls that were aligned with the gold trajectory."""

    compared: int
    exact: int
    mean_similarity: float

    @property
    def exact_rate(self) -> float:
        return self.exact / self.compared if self.compared else 0.0


def score_outputs(predicted: list[ToolCall], gold: list[ToolCall]) -> OutputScore:
    """Score results of calls aligned by :func:`trajectory.aligned_pairs`.

    Calls that the trajectory score does not match contribute nothing here, so this
    score is only meaningful alongside the trajectory score.
    """
    pairs = aligned_pairs(predicted, gold)
    if not pairs:
        return OutputScore(compared=0, exact=0, mean_similarity=0.0)
    sims = []
    exact = 0
    for i, j in pairs:
        sim = output_similarity(predicted[i].result, gold[j].result)
        sims.append(sim)
        if normalize_output(predicted[i].result) == normalize_output(gold[j].result):
            exact += 1
    return OutputScore(compared=len(pairs), exact=exact, mean_similarity=sum(sims) / len(sims))
