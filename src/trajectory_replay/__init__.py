"""Deterministic replay and trajectory regression tests for tool-using LLM agents."""

from .adapters import calls_from_records
from .agent import RunResult, run_agent
from .cassette import ReplayMismatch, Replayer, ToolCall, load, save
from .tools import ToolError, make_tools
from .trajectory import TrajectoryScore, score

__all__ = [
    "calls_from_records",
    "ReplayMismatch",
    "Replayer",
    "RunResult",
    "ToolCall",
    "ToolError",
    "TrajectoryScore",
    "load",
    "make_tools",
    "run_agent",
    "save",
    "score",
]

__version__ = "0.1.0"
