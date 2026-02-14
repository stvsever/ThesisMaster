from __future__ import annotations

from pathlib import Path
from string import Template
import json
from typing import Any, Dict, Mapping


def _agentic_root() -> Path:
    return Path(__file__).resolve().parents[1]


PROMPTS_ROOT = _agentic_root() / "prompts"
PROMPTS_MANIFEST_PATH = PROMPTS_ROOT / "prompts_manifest.json"


def load_prompts_manifest() -> Dict[str, Any]:
    if not PROMPTS_MANIFEST_PATH.exists():
        return {"contract_version": "1.0.0", "prompts": {}}
    payload = json.loads(PROMPTS_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Invalid prompts manifest: expected object")
    return payload


def load_prompt(prompt_filename: str) -> str:
    path = PROMPTS_ROOT / prompt_filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    manifest = load_prompts_manifest()
    prompt_meta = (manifest.get("prompts", {}) or {}).get(prompt_filename, {})
    if prompt_meta and str(prompt_meta.get("active", True)).lower() in {"false", "0", "no"}:
        raise ValueError(f"Prompt disabled in manifest: {prompt_filename}")
    return path.read_text(encoding="utf-8")


def render_prompt(prompt_template: str, values: Mapping[str, str]) -> str:
    template = Template(prompt_template)
    return template.safe_substitute({str(k): str(v) for k, v in values.items()})
