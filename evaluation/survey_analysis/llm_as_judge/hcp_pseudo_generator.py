"""
Realistic Pseudo HCP Output Generator
======================================
Generates pseudo expert-HCP outputs for all 10 cases via Gemini Flash
through OpenRouter.  Each output simulates a Dutch-speaking licensed HCP
who has reviewed the Qualtrics survey material.

The HCP persona differs from PHOENIX:
- More clinical jargon and abbreviations
- Slightly different treatment prioritisation (less network-optimised)
- Coaching messages with a more therapist-like tone
- Natural variation in label length and style

Outputs are saved to:
    llm_as_judge/data/expert_outputs/hcp_outputs.json
    data/02_parsed/hcp_outputs.json   (pipeline canonical location)

Canonical output format (identical to PHOENIX):
    Part 1  {"items": [{"label": "..."}]}
    Part 2  {"items": [{"label": "..."}]}
    Part 3  {"ranking": [{"rank": 1, "option_id": "BO-X"}]}
    Part 4  {"selected_options": ["..."]}
    Part 5  {"message": "..."}
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_HERE         = Path(__file__).resolve().parent               # .../llm_as_judge/
_SURVEY_ROOT  = _HERE.parent                                  # .../evaluation/survey_analysis/
_EVAL_ROOT    = _SURVEY_ROOT.parent                           # .../evaluation/
_PHOENIX_DATA = _EVAL_ROOT / "phoenix_outputs" / "data"       # .../evaluation/phoenix_outputs/data/

CASE_INPUTS_PATH: Path = _PHOENIX_DATA / "inputs" / "qualtrics_case_inputs.json"
CASE_CONTEXTS_PATH: Path = _PHOENIX_DATA / "inputs" / "case_contexts_for_judge.json"

EXPERT_OUTPUTS_DIR: Path = _HERE / "data" / "expert_outputs"
PIPELINE_PARSED_DIR: Path = _SURVEY_ROOT / "data" / "02_parsed"

DEFAULT_MODEL: str = "google/gemini-3.1-flash-lite-preview"
DEFAULT_MAX_WORKERS: int = 10

# ─────────────────────────────────────────────────────────────────────────────
# HCP persona system prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Je bent een ervaren klinisch psycholoog (GZ-psycholoog) met 10 jaar praktijk-"
    "ervaring in ambulante GGZ. Je beoordeelt patiëntcasussen en geeft klinisch "
    "gefundeerde antwoorden op het format van een Qualtrics-survey. Antwoord altijd "
    "in valide JSON. Geen uitleg, geen proza, alleen JSON. Gebruik Nederlandse "
    "klinische vakterm. Geef antwoorden die klinisch realistisch maar licht "
    "afwijkend van een AI-systeem zijn: focus op jouw klinische intuïtie en "
    "praktijkervaring."
)

# ─────────────────────────────────────────────────────────────────────────────
# Part-specific prompt builders (HCP perspective)
# ─────────────────────────────────────────────────────────────────────────────

def _build_part1_prompt(
    case_inputs: Dict[str, Any],
    case_context: Dict[str, Any],
) -> str:
    vignette = case_inputs["part1"]["vignette"]
    return (
        "TAAK: Identificeer 3 tot 6 compacte symptoomlabels op basis van de "
        "klachtenomschrijving van een patiënt.\n\n"
        "Richtlijnen:\n"
        "- Labels zijn symptomen of klachtdimensies (geen behandeldoelen, "
        "diagnosen of oorzaken)\n"
        "- Gebruik korte klinische termen (2-5 woorden) die je in je klinische "
        "praktijk zou hanteren\n"
        "- Dek alle relevante klachtendomeinen\n"
        "- Vermijd overlap tussen labels\n"
        "- Taal: Nederlands\n\n"
        f"Klachtenomschrijving:\n{vignette}\n\n"
        'JSON-formaat: {"items": [{"label": "..."}, {"label": "..."}]}'
    )


def _build_part2_prompt(
    case_inputs: Dict[str, Any],
    case_context: Dict[str, Any],
) -> str:
    symptoms = case_inputs["part2"]["standardized_symptoms"]
    summary = case_inputs["part2"]["case_summary"]
    return (
        "TAAK: Genereer 3 tot 5 aanpasbare behandelingsopties voor deze patiënt.\n\n"
        "Een behandelingsoptie is iets wat de patiënt concreet kan aanpassen "
        "(gedrag, routine of vaardigheid) en dagelijks meetbaar is via een app.\n\n"
        "Richtlijnen:\n"
        "- Klinisch relevante opties die aansluiten op de beschreven symptomen\n"
        "- Hanteerbaar voor de patiënt in de ambulante setting\n"
        "- Geen symptoomlabels of diagnosen als opties\n"
        "- Taal: Nederlands\n\n"
        f"Casus: {summary}\n"
        f"Symptomen: {json.dumps(symptoms, ensure_ascii=False)}\n\n"
        'JSON-formaat: {"items": [{"label": "..."}, {"label": "..."}]}'
    )


def _build_part3_prompt(
    case_inputs: Dict[str, Any],
    case_context: Dict[str, Any],
) -> str:
    options = case_context["standardized_treatment_options"]
    monitoring = case_context["network_summary"]["monitoring_summary"]
    edges = case_context["network_summary"]["edges"]
    return (
        "TAAK: Rangschik de vijf behandelingsopties op behandelingsprioriteit "
        "(rang 1 = hoogste prioriteit) vanuit jouw klinische perspectief.\n\n"
        "Gebruik de monitoringdata en netwerkrelaties als ondersteuning, maar "
        "weeg ook je klinische intuïtie mee.\n\n"
        f"Monitoring (21 dagen): {monitoring}\n\n"
        f"Opties: {json.dumps(options, ensure_ascii=False)}\n\n"
        f"Netwerkrelaties: {json.dumps(edges, ensure_ascii=False)}\n\n"
        "Rang alle vijf opties precies één keer.\n"
        '{"ranking": [{"rank": 1, "option_id": "BO-X"}, {"rank": 2, "option_id": "BO-Y"}, '
        '{"rank": 3, "option_id": "BO-Z"}, {"rank": 4, "option_id": "BO-W"}, '
        '{"rank": 5, "option_id": "BO-V"}]}'
    )


def _build_part4_prompt(
    case_inputs: Dict[str, Any],
    case_context: Dict[str, Any],
) -> str:
    targets = case_context["treatment_targets"]
    items = case_context["candidate_ema_items"]
    items_block = "\n".join(f"  - {item['label']}" for item in items)
    return (
        "TAAK: Kies precies 6 EMA-monitoringitems: 2 per behandeldoel.\n\n"
        f"Behandeldoelen: {json.dumps(targets, ensure_ascii=False)}\n\n"
        "Regels:\n"
        "- Kies uitsluitend items uit de lijst (exacte label overnemen)\n"
        "- 2 items per doel, totaal 6\n"
        "- Kies de klinisch meest informatieve items\n\n"
        f"Kandidatenlijst:\n{items_block}\n\n"
        '{"selected_options": ["...", "...", "...", "...", "...", "..."]}'
    )


def _build_part5_prompt(
    case_inputs: Dict[str, Any],
    case_context: Dict[str, Any],
) -> str:
    problem = case_context["primary_problem"]
    goal = case_context["treatment_goal"]
    barrier = case_context["barrier"]
    coping = case_context["coping_strategy"]
    return (
        "TAAK: Schrijf een kort coachingbericht (2-4 zinnen) voor de smartphone-app.\n\n"
        "Schrijf als een betrokken behandelaar: warm, direct, respectvol. "
        "Gebruik jij/je. Geen diagnostisch taalgebruik. Sluit aan op de "
        "specifieke casus. Geef één kleine, haalbare actie mee.\n\n"
        f"Probleem: {problem}\n"
        f"Doel: {goal}\n"
        f"Barrière: {barrier}\n"
        f"Coping: {coping}\n\n"
        '{"message": "Jouw bericht hier."}'
    )


PROMPT_BUILDERS = {
    "part1": _build_part1_prompt,
    "part2": _build_part2_prompt,
    "part3": _build_part3_prompt,
    "part4": _build_part4_prompt,
    "part5": _build_part5_prompt,
}

# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing + validation (same helpers as phoenix_engine_runner)
# ─────────────────────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        start = 1 if lines[0].startswith("```") else 0
        end = -1 if lines and lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end]).strip()
    return text


def _first_json_block(text: str) -> str:
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        idx = text.find(open_ch)
        if idx == -1:
            continue
        depth = 0
        for i, ch in enumerate(text[idx:], idx):
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[idx: i + 1]
    return text


def _parse_llm_json(text: str) -> Dict[str, Any]:
    cleaned = _strip_fences(text)
    snippet = _first_json_block(cleaned)
    return json.loads(snippet)


def _canonicalize(part: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    if part in ("part1", "part2"):
        items = raw.get("items", [])
        if not isinstance(items, list) or not items:
            raise ValueError(f"{part}: empty items list")
        return {
            "items": [
                {"label": str(item.get("label", item) if isinstance(item, dict) else item).strip()}
                for item in items
            ]
        }
    if part == "part3":
        ranking = raw.get("ranking", [])
        if not isinstance(ranking, list) or len(ranking) != 5:
            raise ValueError(f"part3: need exactly 5 ranking entries, got {len(ranking) if isinstance(ranking, list) else type(ranking)}")
        ranks_seen, ids_seen = set(), set()
        canonical = []
        for r in ranking:
            rank = int(r["rank"])
            oid = str(r["option_id"]).strip()
            if rank in ranks_seen or oid in ids_seen:
                raise ValueError(f"part3: duplicate rank={rank} or option_id={oid!r}")
            ranks_seen.add(rank)
            ids_seen.add(oid)
            canonical.append({"rank": rank, "option_id": oid})
        canonical.sort(key=lambda x: x["rank"])
        if set(ranks_seen) != {1, 2, 3, 4, 5}:
            raise ValueError(f"part3: ranks must be 1–5, got {sorted(ranks_seen)}")
        return {"ranking": canonical}
    if part == "part4":
        selected = raw.get("selected_options", [])
        if not isinstance(selected, list):
            raise ValueError("part4: selected_options must be a list")
        return {"selected_options": [str(s).strip() for s in selected]}
    if part == "part5":
        msg = str(raw.get("message", "")).strip()
        if not msg:
            raise ValueError("part5: message must be non-empty")
        return {"message": msg}
    raise ValueError(f"Unknown part: {part!r}")


# ─────────────────────────────────────────────────────────────────────────────
# LLM call (OpenRouter)
# ─────────────────────────────────────────────────────────────────────────────

def _llm_call(
    messages: List[Dict[str, str]],
    api_key: str,
    model: str,
    max_retries: int = 3,
) -> str:
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/PHOENIX-eval/masterproef",
                "X-Title": "PHOENIX HCP Pseudo Generator",
            },
            timeout=90.0,
        )
        use_sdk = True
    except ImportError:
        use_sdk = False

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            if use_sdk:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.8,   # slightly higher variance for HCP realism
                    max_tokens=1024,
                )
                return (resp.choices[0].message.content or "").strip()
            else:
                import urllib.request
                body = json.dumps({
                    "model": model,
                    "messages": messages,
                    "temperature": 0.8,
                    "max_tokens": 1024,
                }).encode()
                req = urllib.request.Request(
                    "https://openrouter.ai/api/v1/chat/completions",
                    data=body,
                    method="POST",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(req, timeout=90) as r:
                    payload = json.loads(r.read().decode())
                return (payload["choices"][0]["message"].get("content") or "").strip()
        except Exception as exc:
            last_exc = exc
            wait = min(30.0, 1.5 ** attempt)
            logger.warning("LLM attempt %d/%d failed: %s — %.1fs", attempt, max_retries, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"LLM failed after {max_retries} attempts: {last_exc!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Single task runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_task(
    case_id: str,
    part: str,
    case_inputs: Dict[str, Any],
    case_context: Dict[str, Any],
    api_key: str,
    model: str,
    json_retries: int = 2,
) -> Tuple[str, str, Dict[str, Any]]:
    builder = PROMPT_BUILDERS[part]
    user_msg = builder(case_inputs, case_context)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    last_exc: Optional[Exception] = None
    last_text = ""
    for attempt in range(json_retries + 1):
        try:
            text = _llm_call(messages, api_key=api_key, model=model)
            last_text = text
            raw = _parse_llm_json(text)
            canonical = _canonicalize(part, raw)
            logger.info("✓ HCP %s / %s", case_id, part)
            return (case_id, part, canonical)
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            last_exc = exc
            logger.warning("Parse failed HCP %s/%s attempt %d: %s", case_id, part, attempt + 1, exc)
            if attempt < json_retries:
                messages = messages + [
                    {"role": "assistant", "content": last_text},
                    {
                        "role": "user",
                        "content": (
                            "Vorige respons bevatte geen valide JSON. "
                            "Geef uitsluitend het JSON-object terug, geen proza."
                        ),
                    },
                ]
        except Exception as exc:
            last_exc = exc
            logger.error("LLM error HCP %s/%s: %s", case_id, part, exc)
            raise

    raise RuntimeError(
        f"HCP {case_id}/{part}: failed after {json_retries + 1} attempts: {last_exc!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_pseudo_hcp_outputs(
    case_ids: Optional[List[str]] = None,
    parts: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_workers: int = DEFAULT_MAX_WORKERS,
    output_path: Optional[Path] = None,
    also_copy_to_pipeline: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Generate pseudo HCP outputs for all requested cases and parts.

    Returns
    -------
    dict
        ``{case_id: {part: canonical_output}}``
    """
    _api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not _api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY must be set or passed as api_key=..."
        )

    if not CASE_INPUTS_PATH.exists():
        raise FileNotFoundError(f"Case inputs not found: {CASE_INPUTS_PATH}")
    if not CASE_CONTEXTS_PATH.exists():
        raise FileNotFoundError(f"Case contexts not found: {CASE_CONTEXTS_PATH}")

    case_inputs_bundle = json.loads(CASE_INPUTS_PATH.read_text(encoding="utf-8"))
    case_contexts_bundle = json.loads(CASE_CONTEXTS_PATH.read_text(encoding="utf-8"))

    all_cases = list(case_inputs_bundle["cases"].keys())
    all_parts = list(PROMPT_BUILDERS.keys())

    selected_cases = case_ids or all_cases
    selected_parts = parts or all_parts

    tasks = [
        (cid, part)
        for cid in selected_cases
        for part in selected_parts
        if cid in case_inputs_bundle["cases"] and cid in case_contexts_bundle
    ]

    logger.info(
        "HCP pseudo generator: %d tasks   model=%s   max_workers=%d",
        len(tasks), model, max_workers,
    )

    results: Dict[str, Dict[str, Any]] = {cid: {} for cid in selected_cases}
    errors: List[Tuple[str, str, str]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_task = {
            pool.submit(
                _run_task,
                cid,
                part,
                case_inputs_bundle["cases"][cid],
                case_contexts_bundle[cid],
                _api_key,
                model,
            ): (cid, part)
            for cid, part in tasks
        }
        for future in as_completed(future_to_task):
            cid, part = future_to_task[future]
            try:
                _, _, canonical = future.result()
                results[cid][part] = canonical
            except Exception as exc:
                logger.error("FAILED HCP %s/%s: %s", cid, part, exc)
                errors.append((cid, part, str(exc)))

    if errors:
        for cid, part, msg in errors:
            logger.error("  ✗ HCP %s/%s → %s", cid, part, msg)
        raise RuntimeError(
            f"{len(errors)} HCP task(s) failed. First: {errors[0]}"
        )

    payload = json.dumps(results, ensure_ascii=False, indent=2)

    # Primary output
    out_path = output_path or (EXPERT_OUTPUTS_DIR / "hcp_outputs.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload, encoding="utf-8")
    logger.info("Saved pseudo HCP outputs → %s", out_path)

    # Mirror to pipeline parsed dir
    if also_copy_to_pipeline:
        dst = PIPELINE_PARSED_DIR / "hcp_outputs.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(payload, encoding="utf-8")
        logger.info("Copied → %s (pipeline canonical)", dst)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Generate pseudo HCP outputs via LLM.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--parts", nargs="+", default=None)
    args = parser.parse_args()

    results = generate_pseudo_hcp_outputs(
        case_ids=args.cases,
        parts=args.parts,
        model=args.model,
        max_workers=args.workers,
    )
    print(f"\n✓ Generated pseudo HCP outputs for {len(results)} cases.")
