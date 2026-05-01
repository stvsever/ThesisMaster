"""Prompt templates and loader helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

PROMPTS_DIR: Path = Path(__file__).resolve().parent

PART_TO_PROMPT_FILE: Dict[str, str] = {
    "part1": "part1_prompt.md",
    "part2": "part2_prompt.md",
    "part3": "part3_prompt.md",
    "part4": "part4_prompt.md",
    "part5": "part5_prompt.md",
}


def load_prompt(part: str) -> str:
    """Read a part-specific prompt template (Markdown) from disk."""
    if part not in PART_TO_PROMPT_FILE:
        raise ValueError(f"Unknown part {part!r}")
    return (PROMPTS_DIR / PART_TO_PROMPT_FILE[part]).read_text(encoding="utf-8")


def load_system_prompt() -> str:
    return (PROMPTS_DIR / "system_prompt.md").read_text(encoding="utf-8")


__all__ = ["PROMPTS_DIR", "load_prompt", "load_system_prompt"]
