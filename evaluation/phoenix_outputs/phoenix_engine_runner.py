"""
PHOENIX LLM Engine Runner
=========================
Generates PHOENIX system outputs for all 10 survey cases by calling
Gemini Flash via OpenRouter.  One LLM call per (case, part) pair; all
calls run concurrently through ThreadPoolExecutor.

Canonical output shapes (identical to the HCP judge schema):

    Part 1  {"items": [{"label": "..."}]}
    Part 2  {"items": [{"label": "..."}]}
    Part 3  {"ranking": [{"rank": 1, "option_id": "BO-X"}]}
    Part 4  {"selected_options": ["..."]}
    Part 5  {"message": "..."}

Usage
-----
    python evaluation/phoenix_outputs/phoenix_engine_runner.py

    or via run.py:
    python evaluation/phoenix_outputs/run.py run-engine [--model ...] [--workers N]
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

_HERE = Path(__file__).resolve().parent
CASE_INPUTS_PATH: Path = _HERE / "data" / "inputs" / "qualtrics_case_inputs.json"
CASE_CONTEXTS_PATH: Path = _HERE / "data" / "inputs" / "case_contexts_for_judge.json"
LLM_OUTPUTS_PATH: Path = _HERE / "data" / "outputs" / "system_outputs_llm.json"

# Also written here so the survey-analysis pipeline can use it directly.
_SURVEY_ROOT = _HERE.parent / "survey_analysis"
PIPELINE_SYSTEM_DIR: Path = _SURVEY_ROOT / "data" / "03_system"
JUDGE_PHOENIX_DIR: Path = _SURVEY_ROOT / "llm_as_judge" / "data" / "phoenix_outputs"

DEFAULT_MODEL: str = "google/gemini-3.1-flash-lite-preview"
DEFAULT_MAX_WORKERS: int = 10

# ─────────────────────────────────────────────────────────────────────────────
# System prompt (PHOENIX persona)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Je bent PHOENIX, een klinisch AI-systeem voor gepersonaliseerde digitale "
    "geestelijke gezondheidszorg. Je analyseert patiëntcasussen en genereert "
    "klinisch gefundeerde, evidence-based outputs. Optimaliseer voor compacte "
    "taakvaliditeit, klinische precisie, EMA-meetbaarheid, netwerklogica, "
    "gedragsveranderingspotentieel en veilige persoonsgerichte communicatie. "
    "Geef altijd een valide JSON-antwoord in het exacte gevraagde formaat. "
    "Geen proza, geen uitleg, alleen JSON. Gebruik uitsluitend de verstrekte "
    "informatie."
)

# ─────────────────────────────────────────────────────────────────────────────
# Part-specific prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_part1_prompt(
    case_inputs: Dict[str, Any],
    case_context: Dict[str, Any],
) -> str:
    vignette = case_inputs["part1"]["vignette"]
    return (
        "TAAK: Identificeer 3 tot 6 beknopte symptoomlabels op basis van de "
        "onderstaande klachtenomschrijving.\n\n"
        "Regels:\n"
        "- Labels zijn symptomen of toestandsdimensies (geen behandelingen, "
        "diagnosen, externe oorzaken of trekken)\n"
        "- Elk label is een compacte klinische term (2-5 woorden) die een andere "
        "zorgverlener direct zou begrijpen\n"
        "- Dek alle voornaamste klachtendomeinen uit het vignette\n"
        "- Kies labels die als dagelijkse EMA-variabele voorstelbaar zijn\n"
        "- Geen doublures of overlappende constructen\n"
        "- Taal: Nederlands\n\n"
        f"Klachtenomschrijving:\n{vignette}\n\n"
        'Geef antwoord als: {"items": [{"label": "Symptoom A"}, {"label": "Symptoom B"}]}'
    )


def _build_part2_prompt(
    case_inputs: Dict[str, Any],
    case_context: Dict[str, Any],
) -> str:
    symptoms = case_inputs["part2"]["standardized_symptoms"]
    summary = case_inputs["part2"]["case_summary"]
    return (
        "TAAK: Genereer 3 tot 5 beknopte labels van aanpasbare behandelingsopties "
        "voor deze patiënt.\n\n"
        "Definitie behandelingsoptie: een gedrag, routine of strategie die de "
        "patiënt of therapeut realistisch kan veranderen en dagelijks via een app "
        "(ja/nee, tellen, minuten, schaal 0-10) meetbaar is.\n\n"
        "Regels:\n"
        "- Geen symptoomlabels, diagnosen of brede gezondheidsbegrippen\n"
        "- Elke optie is causaal verbonden met een of meer van de symptomen\n"
        "- De set bevat complementaire opties met verschillende verandermechanismen\n"
        "- Elk label is compact genoeg om later in een netwerk en EMA-itemset te gebruiken\n"
        "- Opties zijn complementair (geen doublures)\n"
        "- Taal: Nederlands\n\n"
        f"Casussamenvatting: {summary}\n\n"
        f"Gestandaardiseerde symptomen:\n{json.dumps(symptoms, ensure_ascii=False)}\n\n"
        'Geef antwoord als: {"items": [{"label": "Behandelingsoptie A"}, '
        '{"label": "Behandelingsoptie B"}]}'
    )


def _build_part3_prompt(
    case_inputs: Dict[str, Any],
    case_context: Dict[str, Any],
) -> str:
    options = case_context["standardized_treatment_options"]
    monitoring = case_context["network_summary"]["monitoring_summary"]
    edges = case_context["network_summary"]["edges"]
    edge_interp = case_context["network_summary"].get("edge_interpretation", {})
    return (
        "TAAK: Rangschik alle vijf behandelingsopties op behandelingsprioriteit "
        "(rang 1 = hoogste prioriteit).\n\n"
        "Criterium: combineer (a) de sterkte van de netwerkrelaties, "
        "(b) de huidige 21-daagse EMA-burdendata, en (c) de aanpasbaarheid.\n\n"
        "Rangschikkingslogica:\n"
        "- Prioriteer opties met sterke absolute edge-gewichten naar centrale symptomen\n"
        "- Behandel risk-edges als aangrijpingspunt voor reductie en protective-edges als aangrijpingspunt voor versterking\n"
        "- Weeg huidige EMA-last mee wanneer meerdere opties vergelijkbaar netwerkgewicht hebben\n"
        "- Behoud een complete en coherente 1-5 rangorde zonder doublures\n\n"
        "Let op edge-richting:\n"
        f"  risk-edge: {edge_interp.get('positive', 'hogere optie → meer symptoomlast')}\n"
        f"  protective-edge: {edge_interp.get('negative', 'hogere optie → minder symptoomlast')}\n\n"
        f"Monitoringsamenvatting: {monitoring}\n\n"
        f"Behandelingsopties:\n{json.dumps(options, ensure_ascii=False)}\n\n"
        f"Netwerkrelaties:\n{json.dumps(edges, ensure_ascii=False)}\n\n"
        "Geef antwoord als:\n"
        '{"ranking": [{"rank": 1, "option_id": "BO-X"}, {"rank": 2, "option_id": "BO-Y"}, '
        '{"rank": 3, "option_id": "BO-Z"}, {"rank": 4, "option_id": "BO-W"}, '
        '{"rank": 5, "option_id": "BO-V"}]}\n'
        "Rangschik alle vijf opties exact één keer. Gebruik de opgegeven option_id's."
    )


def _build_part4_prompt(
    case_inputs: Dict[str, Any],
    case_context: Dict[str, Any],
) -> str:
    targets = case_context["treatment_targets"]
    items = case_context["candidate_ema_items"]
    items_block = "\n".join(
        f"  - {item['label']}" for item in items
    )
    return (
        "TAAK: Selecteer exact 6 EMA-items uit de kandidatenlijst: 2 items per "
        "behandeldoel.\n\n"
        f"Behandeldoelen:\n{json.dumps(targets, ensure_ascii=False)}\n\n"
        "Selectieregels:\n"
        "1. Kies uitsluitend items uit de lijst (kopieer de label exact, geen "
        "parafrasering)\n"
        "2. Exact 2 items per behandeldoel (totaal 6)\n"
        "3. Kies de meest directe, informatieve items\n"
        "4. Zorg dat elke selectie concrete dagelijkse variatie kan registreren\n"
        "5. Beperk meetbelasting door geen overlappende of vrijwel identieke items te kiezen\n"
        "6. Geen doublures\n\n"
        f"Kandidatenlijst (20 items):\n{items_block}\n\n"
        'Geef antwoord als: {"selected_options": ["Label item 1", "Label item 2", '
        '"Label item 3", "Label item 4", "Label item 5", "Label item 6"]}'
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
        "TAAK: Schrijf een mobiel coachingbericht van 2 tot 4 zinnen.\n\n"
        "Eisen:\n"
        "- Spreek de patiënt direct aan (tweede persoon, jij/je)\n"
        "- Warm, direct en professioneel (geen klinisch jargon)\n"
        "- Bevat één concrete, kleine, tijdsgebonden actie voor vandaag/vanavond\n"
        "- Erkent of werkt rond de hoofdbarrière\n"
        "- Sluit expliciet aan bij het behandeldoel en de copingstrategie\n"
        "- Formuleer als self-efficacy ondersteunend micro-commitment\n"
        "- Veilig, niet-veroordelend en zonder beloftes over snelle verbetering\n"
        "- Maximaal 4 zinnen, geschikt voor een mobiel scherm\n"
        "- Taal: Nederlands\n\n"
        f"Hoofdprobleem: {problem}\n"
        f"Behandeldoel: {goal}\n"
        f"Barrière: {barrier}\n"
        f"Copingstrategie: {coping}\n\n"
        'Geef antwoord als: {"message": "Jouw bericht hier."}'
    )


PROMPT_BUILDERS = {
    "part1": _build_part1_prompt,
    "part2": _build_part2_prompt,
    "part3": _build_part3_prompt,
    "part4": _build_part4_prompt,
    "part5": _build_part5_prompt,
}

# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first and last lines if they are fences
        start = 1 if lines[0].startswith("```") else 0
        end = -1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end]).strip()
    return text


def _find_json_object(text: str) -> str:
    """Return the first {...} or [...] block found in text."""
    for start_ch, end_ch in (("{", "}"), ("[", "]")):
        idx = text.find(start_ch)
        if idx == -1:
            continue
        # Find the matching closing bracket
        depth = 0
        for i, ch in enumerate(text[idx:], idx):
            if ch == start_ch:
                depth += 1
            elif ch == end_ch:
                depth -= 1
                if depth == 0:
                    return text[idx: i + 1]
    return text


def _parse_llm_json(text: str) -> Dict[str, Any]:
    """Extract and parse JSON from an LLM response."""
    cleaned = _strip_markdown_fences(text)
    snippet = _find_json_object(cleaned)
    return json.loads(snippet)


# ─────────────────────────────────────────────────────────────────────────────
# Output validation
# ─────────────────────────────────────────────────────────────────────────────

def _canonicalize(part: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and return canonical output dict for a given part."""
    if part in ("part1", "part2"):
        items = raw.get("items", [])
        if not isinstance(items, list) or len(items) == 0:
            raise ValueError(f"{part}: 'items' must be a non-empty list, got {type(items)}")
        canonical_items = []
        for item in items:
            if isinstance(item, dict):
                label = str(item.get("label", "")).strip()
            else:
                label = str(item).strip()
            if not label:
                raise ValueError(f"{part}: empty label in items")
            canonical_items.append({"label": label})
        return {"items": canonical_items}

    if part == "part3":
        ranking = raw.get("ranking", [])
        if not isinstance(ranking, list) or len(ranking) != 5:
            raise ValueError(
                f"part3: ranking must have exactly 5 entries, got "
                f"{len(ranking) if isinstance(ranking, list) else type(ranking)}"
            )
        seen_ranks, seen_ids = set(), set()
        canonical_ranking = []
        for r in ranking:
            rank = int(r["rank"])
            oid = str(r["option_id"]).strip()
            if rank in seen_ranks:
                raise ValueError(f"part3: duplicate rank {rank}")
            if oid in seen_ids:
                raise ValueError(f"part3: duplicate option_id {oid!r}")
            seen_ranks.add(rank)
            seen_ids.add(oid)
            canonical_ranking.append({"rank": rank, "option_id": oid})
        canonical_ranking.sort(key=lambda x: x["rank"])
        if set(seen_ranks) != {1, 2, 3, 4, 5}:
            raise ValueError(f"part3: ranks must be {{1..5}}, got {sorted(seen_ranks)}")
        return {"ranking": canonical_ranking}

    if part == "part4":
        selected = raw.get("selected_options", [])
        if not isinstance(selected, list):
            raise ValueError(f"part4: 'selected_options' must be a list")
        return {"selected_options": [str(s).strip() for s in selected]}

    if part == "part5":
        message = str(raw.get("message", "")).strip()
        if not message:
            raise ValueError("part5: 'message' must be non-empty")
        return {"message": message}

    raise ValueError(f"Unknown part: {part!r}")


# ─────────────────────────────────────────────────────────────────────────────
# LLM call (OpenRouter via OpenAI SDK)
# ─────────────────────────────────────────────────────────────────────────────

def _llm_call(
    messages: List[Dict[str, str]],
    api_key: str,
    model: str,
    max_retries: int = 3,
) -> str:
    """Make a single chat completion call with retries."""
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/PHOENIX-eval/masterproef",
                "X-Title": "PHOENIX Engine Runner",
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
                    temperature=0.7,
                    max_tokens=1024,
                )
                return (resp.choices[0].message.content or "").strip()
            else:
                import urllib.request
                body = json.dumps({
                    "model": model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 1024,
                }).encode("utf-8")
                req = urllib.request.Request(
                    "https://openrouter.ai/api/v1/chat/completions",
                    data=body,
                    method="POST",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/PHOENIX-eval/masterproef",
                        "X-Title": "PHOENIX Engine Runner",
                    },
                )
                with urllib.request.urlopen(req, timeout=90) as r:
                    payload = json.loads(r.read().decode("utf-8"))
                return (payload["choices"][0]["message"].get("content") or "").strip()
        except Exception as exc:
            last_exc = exc
            wait = min(30.0, 1.5 ** attempt)
            logger.warning(
                "LLM attempt %d/%d failed: %s — sleeping %.1fs",
                attempt, max_retries, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(
        f"LLM call failed after {max_retries} attempts: {last_exc!r}"
    )


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
    """
    Run one (case_id, part) LLM task.

    Returns ``(case_id, part, canonical_output)``.
    """
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
            logger.info("✓ %s / %s", case_id, part)
            return (case_id, part, canonical)
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            last_exc = exc
            logger.warning(
                "JSON parse/validate failed %s/%s attempt %d: %s",
                case_id, part, attempt + 1, exc,
            )
            if attempt < json_retries:
                messages = messages + [
                    {"role": "assistant", "content": last_text},
                    {
                        "role": "user",
                        "content": (
                            "Vorige respons bevatte geen valide JSON. "
                            "Geef uitsluitend het gevraagde JSON-object terug, "
                            "geen proza, geen markdown."
                        ),
                    },
                ]
        except Exception as exc:
            last_exc = exc
            logger.error("LLM call error %s/%s: %s", case_id, part, exc)
            raise

    raise RuntimeError(
        f"{case_id}/{part}: parsing failed after {json_retries + 1} attempts: "
        f"{last_exc!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_phoenix_engine(
    case_ids: Optional[List[str]] = None,
    parts: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_workers: int = DEFAULT_MAX_WORKERS,
    case_inputs_path: Optional[Path] = None,
    case_contexts_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    also_copy_to_pipeline: bool = True,
    also_copy_to_judge_dir: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Run the PHOENIX engine for all requested cases and parts.

    Parameters
    ----------
    case_ids : list[str] | None
        Subset of case ids to run (default: all 10).
    parts : list[str] | None
        Subset of parts to run (default: all 5).
    api_key : str | None
        OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
    model : str
        OpenRouter model id (default: google/gemini-3.1-flash-lite-preview).
    max_workers : int
        Thread-pool concurrency (default: 10).
    case_inputs_path : Path | None
        Exact Qualtrics-derived inputs. Defaults to data/inputs/qualtrics_case_inputs.json.
    case_contexts_path : Path | None
        Shared judge context. Defaults to data/inputs/case_contexts_for_judge.json.
    output_path : Path | None
        Where to save system_outputs_llm.json (default: data/outputs/).
    also_copy_to_pipeline : bool
        If True, also write to data/03_system/system_outputs.json so that
        the survey-analysis pipeline picks it up.
    also_copy_to_judge_dir : bool
        If True, also write to llm_as_judge/data/phoenix_outputs/.

    Returns
    -------
    dict
        ``{case_id: {part: canonical_output}}``
    """
    _api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not _api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY must be set in the environment or passed as "
            "api_key=... to run_phoenix_engine()."
        )

    # Load inputs
    inputs_path = case_inputs_path or CASE_INPUTS_PATH
    contexts_path = case_contexts_path or CASE_CONTEXTS_PATH
    if not inputs_path.exists():
        raise FileNotFoundError(f"Case inputs not found: {inputs_path}")
    if not contexts_path.exists():
        raise FileNotFoundError(f"Case contexts not found: {contexts_path}")

    case_inputs_bundle = json.loads(inputs_path.read_text(encoding="utf-8"))
    case_contexts_bundle = json.loads(contexts_path.read_text(encoding="utf-8"))

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

    missing = [
        cid for cid in selected_cases
        if cid not in case_inputs_bundle["cases"] or cid not in case_contexts_bundle
    ]
    if missing:
        logger.warning("Cases not found in input data, skipping: %s", missing)

    logger.info(
        "PHOENIX engine: %d tasks   model=%s   max_workers=%d",
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
                logger.error("FAILED %s/%s: %s", cid, part, exc)
                errors.append((cid, part, str(exc)))

    if errors:
        for cid, part, msg in errors:
            logger.error("  ✗ %s/%s → %s", cid, part, msg)
        raise RuntimeError(
            f"{len(errors)} task(s) failed. First: {errors[0]}"
        )

    payload = json.dumps(results, ensure_ascii=False, indent=2)

    # Primary output
    out_path = output_path or LLM_OUTPUTS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload, encoding="utf-8")
    logger.info("Saved PHOENIX engine outputs → %s", out_path)

    # Mirror to survey-analysis pipeline dir
    if also_copy_to_pipeline:
        dst = PIPELINE_SYSTEM_DIR / "system_outputs.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(payload, encoding="utf-8")
        logger.info("Copied → %s", dst)

    # Mirror to judge data dir
    if also_copy_to_judge_dir:
        dst2 = JUDGE_PHOENIX_DIR / "system_outputs.json"
        dst2.parent.mkdir(parents=True, exist_ok=True)
        dst2.write_text(payload, encoding="utf-8")
        logger.info("Copied → %s", dst2)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run PHOENIX LLM engine for all cases.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--parts", nargs="+", default=None)
    args = parser.parse_args()

    outputs = run_phoenix_engine(
        case_ids=args.cases,
        parts=args.parts,
        model=args.model,
        max_workers=args.workers,
    )
    print(f"\n✓ Generated outputs for {len(outputs)} cases.")
