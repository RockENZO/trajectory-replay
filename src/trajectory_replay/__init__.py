"""Deterministic replay and trajectory regression tests for tool-using LLM agents."""

from .adapters import calls_from_records
from .agent import RunResult, run_agent
from .cassette import ReplayMismatch, Replayer, ToolCall, load, save
from .tools import ToolError, make_tools
from .outputs import OutputScore, output_similarity, score_outputs
from .trajectory import TrajectoryScore, aligned_pairs, score

__all__ = [
    "calls_from_records",
    "ReplayMismatch",
    "Replayer",
    "RunResult",
    "ToolCall",
    "ToolError",
    "TrajectoryScore",
    "OutputScore",
    "aligned_pairs",
    "load",
    "make_tools",
    "output_similarity",
    "run_agent",
    "save",
    "score",
    "score_outputs",
]

__version__ = "0.1.0"
