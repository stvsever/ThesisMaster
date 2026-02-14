from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass
class PromptSection:
    name: str
    text: str
    priority: int
    min_chars: int = 0


@dataclass
class PromptPackResult:
    text: str
    estimated_tokens: int
    max_tokens: int
    included_sections: List[str]
    truncated_sections: List[str]
    section_token_estimates: Dict[str, int]


def _estimate_tokens_heuristic(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 4.0))


def estimate_tokens(text: str, model: str = "gpt-5-nano") -> int:
    try:
        import tiktoken  # type: ignore

        try:
            encoding = tiktoken.encoding_for_model(model)
        except Exception:
            encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(text))
    except Exception:
        return _estimate_tokens_heuristic(text)


def _truncate_to_chars(text: str, target_chars: int) -> str:
    if target_chars <= 0:
        return ""
    if len(text) <= target_chars:
        return text
    if target_chars < 24:
        return text[:target_chars]
    return text[: target_chars - 12].rstrip() + "\n...[TRUNCATED]"


def pack_prompt_sections(
    sections: Sequence[PromptSection],
    *,
    max_tokens: int,
    reserve_tokens: int = 2500,
    model: str = "gpt-5-nano",
    section_delimiter: str = "\n\n",
) -> PromptPackResult:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    sorted_sections = sorted(sections, key=lambda item: int(item.priority))
    token_budget = max(256, int(max_tokens) - int(max(0, reserve_tokens)))

    assembled: List[str] = []
    included: List[str] = []
    truncated: List[str] = []
    section_token_estimates: Dict[str, int] = {}

    for section in sorted_sections:
        formatted = f"[{section.name}]\n{section.text.strip()}" if section.text.strip() else f"[{section.name}]"
        candidate = section_delimiter.join([*assembled, formatted]) if assembled else formatted
        candidate_tokens = estimate_tokens(candidate, model=model)
        if candidate_tokens <= token_budget:
            assembled = [*assembled, formatted]
            included.append(section.name)
            section_token_estimates[section.name] = estimate_tokens(formatted, model=model)
            continue

        if not assembled:
            target_chars = max(int(token_budget * 4), section.min_chars)
            maybe_truncated = _truncate_to_chars(formatted, target_chars=target_chars)
            assembled = [maybe_truncated]
            included.append(section.name)
            truncated.append(section.name)
            section_token_estimates[section.name] = estimate_tokens(maybe_truncated, model=model)
            continue

        remaining_tokens = max(0, token_budget - estimate_tokens(section_delimiter.join(assembled), model=model))
        if remaining_tokens <= 24:
            truncated.append(section.name)
            continue
        target_chars = max(section.min_chars, int(remaining_tokens * 4))
        maybe_truncated = _truncate_to_chars(formatted, target_chars=target_chars)
        if maybe_truncated.strip():
            assembled.append(maybe_truncated)
            included.append(section.name)
            truncated.append(section.name)
            section_token_estimates[section.name] = estimate_tokens(maybe_truncated, model=model)
        else:
            truncated.append(section.name)

    final_text = section_delimiter.join(assembled)
    return PromptPackResult(
        text=final_text,
        estimated_tokens=estimate_tokens(final_text, model=model),
        max_tokens=max_tokens,
        included_sections=included,
        truncated_sections=truncated,
        section_token_estimates=section_token_estimates,
    )
