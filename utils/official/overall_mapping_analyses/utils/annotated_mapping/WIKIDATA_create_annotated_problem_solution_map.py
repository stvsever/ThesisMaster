#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WIKIDATA_create_annotated_problem_solution_map.py

Goal
----
Automate creation of an "annotated problem → solution" mapping for mental health:

- Columns: mental health diagnoses + symptoms (problems)
- Rows: treatments (solutions)
- Cell: 1 if treatment is linked/indicated for that problem in Wikidata (directly or via inference), else 0

Key improvements
----------------
1) Less pharmacology-dominated treatment set (BIO–PSYCHO–SOCIAL balance)
   - Pull direct disorder→treatment from both:
       * P2176 ("drug or therapy used for treatment")
       * P924  ("possible treatment")
   - Enrich treatment metadata (ATC + psychotherapy + medication hierarchy)
   - Balanced sampling across fine-grained domains (bio_pharm, bio_procedure, psycho, social, lifestyle, other)
     with caps on pharmacology share and minimums for non-pharm domains.

   - problem_solution_edges.tsv with one row per (treatment, problem) edge (direct + inferred),
     so you’re not forced to work with mostly-zeros matrices.

3) Explicit BIO/PSYCHO/SOCIAL/X domain layer for downstream analysis
   - Script defines a manual dictionary with 4 keys: BIO, PSYCHO, SOCIAL, X
   - All specified treatment_key entries are explicitly assigned to one of those 4 domains
   - A robust override is applied by QID (parsed from treatment_key), so label formatting differences
     in Wikidata do not break the assignment.

Data source (API)
-----------------
Wikidata SPARQL Query Service:
  - Disorder → treatment via:
      P2176 (drug or therapy used for treatment)
      P924  (possible treatment)
  - Disorder → symptom   via:
      P780  (symptoms and signs)

Inference
---------
We infer symptom → treatment using co-occurrence:
  symptom S is linked to treatment T if there exists at least N disorders D
  such that D has symptom S AND D has treatment T.

Outputs (written to out_dir)
----------------------------
Wide matrix:
- 01_create_annotated_problem_solution_map.csv      (treatment metadata columns + binary columns for problems)

Dense edge-list:
- problem_solution_edges.tsv                        (one row per edge; much less sparse than a matrix)

Lookups:
- problems_lookup.tsv                               (problem_key → label → QID, kind=DX/SYM)
- treatments_lookup.tsv                             (treatment_key → label → QID + degrees + domain_bpsx + domain_fine + flags)

Raw / inferred pairs:
- raw_pairs_disorder_treatment.tsv                  (direct pairs from Wikidata, incl. which property produced it)
- raw_pairs_disorder_symptom.tsv                    (direct pairs from Wikidata)
- inferred_pairs_symptom_treatment.tsv              (support counts for inferred symptom→treatment edges)

Repro:
- sparql_queries.txt                                (exact SPARQL)
- provenance.json                                   (run parameters + summary stats)
- run_log.txt                                       (human-readable summary)

Usage
-----
python 01_create_annotated_problem_solution_map.py

Optional knobs:
python 01_create_annotated_problem_solution_map.py \
  --max-disorders 150 --max-treatments 180 --max-symptoms 120 \
  --min-symptom-support 2 --page-size 5000 --sleep-s 1.0 --max-pages 60 \
  --treatments-balance-mode balanced \
  --max-pharm-fraction 0.35 --min-psycho 35 --min-social 35 --min-lifestyle 15 --min-bio-procedure 10

Notes
-----
- Requires: Python 3.9+, pandas, numpy, requests
- Internet required (Wikidata SPARQL endpoint)
- This mapping is plausibility-oriented (community-curated Wikidata links), not a clinical guideline.

TECHNICAL SUMMARY
-----------------
This script constructs a Wikidata-derived mental health Problem → Solution mapping with
explicit BIO–PSYCHO–SOCIAL balance and dual output representations.

Core pipeline:
1. Retrieve mental disorders (Q12135 subtree).
2. Retrieve disorder → treatment edges using:
   - P2176 (drug or therapy used for treatment)
   - P924  (possible treatment)
   Each edge records its source property.
3. Retrieve disorder → symptom edges via P780 (symptoms and signs).
4. Enrich treatments with Wikidata-derived flags:
   - has_atc (P267) → pharmacological signal
   - is_medication_item (Q12140 hierarchy)
   - is_psychotherapy (Q183257 hierarchy)
5. Assign each treatment two domain layers:
   - domain_fine: {bio_pharm, bio_procedure, psycho, social, lifestyle, other}
     (used for balancing and pharm cap)
   - domain_bpsx: {BIO, PSYCHO, SOCIAL, X}
     (used for downstream biopsychosocial analysis)
   A manual override dictionary pins domain_bpsx for a specified treatment list (by QID).
6. Select treatments using a balanced quota strategy:
   - cap pharmacology (bio_pharm) by max_pharm_fraction
   - enforce minimum counts for psycho, social, lifestyle, bio_procedure (if available)
   - fill remaining slots by disorder-degree while respecting caps.
7. Infer symptom → treatment edges via co-occurrence:
   a symptom S is linked to treatment T if ≥ N disorders have both S and T.
8. Build two synchronized outputs:
   a) Wide binary matrix (treatments × problems) for ML workflows.
   b) Dense edge list with explicit evidence_type (direct vs inferred) and support counts.
9. Export full provenance, lookup tables, raw pairs, and run metadata.

Important notes:
- Direct edges reflect community-curated Wikidata assertions, not clinical guidelines.
- Inferred symptom → treatment edges represent weak, co-occurrence-based evidence.
- Domain/category labels are heuristic+enrichment and may misclassify some interventions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from math import floor
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError as e:
    raise SystemExit("Missing dependency 'requests'. Install with: pip install requests") from e


# ---------------------------
# Configuration (your default output dir)
# ---------------------------
DEFAULT_OUT_DIR = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "overall_mapping_analyses/results/annotated_maps/WIKIDATA"
)

WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# Be polite & identifiable (Wikidata asks for this)
DEFAULT_USER_AGENT = "MentalHealthProblemSolutionMap/2.1 (contact: stijn.vanseveren@gmail.com)"

# Useful ontology anchors:
# - mental disorder: Q12135
# - psychotherapy:   Q183257
# - medication:      Q12140
QID_MENTAL_DISORDER = "Q12135"
QID_PSYCHOTHERAPY = "Q183257"
QID_MEDICATION = "Q12140"

# Properties
PID_TREATMENT = "P2176"          # drug or therapy used for treatment
PID_POSSIBLE_TREATMENT = "P924"  # possible treatment
PID_SYMPTOM = "P780"             # symptoms and signs
PID_ATC = "P267"                 # ATC code


# =====================================================================================
# EXPLICIT BIO/PSYCHO/SOCIAL/X DOMAIN DICTIONARY (manual assignment for specified set)
# =====================================================================================
# Requirement: start with a dictionary with 4 KEYS: BIO, PSYCHO, SOCIAL, X;
# each value is a list containing the provided treatment_key strings.
TREATMENT_DOMAINS_BPSX: Dict[str, List[str]] = {
    "BIO": [
        "TRT__Bupropion__Q72513765",
        "TRT__Cannabis_indica__Q2936421",
        "TRT__E_chlorprothixene_Z_according_to_WHO_INN_RL_40_conflated_wikidata__Q423809",
        "TRT__Gonadotropin_releasing_hormone_analogue__Q905999",
        "TRT__Lavandula_angustifolia__Q42081",
        "TRT__Matricaria_chamomilla__Q28437",
        "TRT__Melissa_officinalis__Q148396",
        "TRT__Panax_ginseng__Q182881",
        "TRT__Passiflora_incarnata__Q128939",
        "TRT__RS_baclofen__Q413717",
        "TRT__RS_citalopram__Q409672",
        "TRT__RS_fenfluramine__Q418928",
        "TRT__RS_ketamine__Q243547",
        "TRT__R_R_asenapine__Q50824517",
        "TRT__R_amphetamine__Q2506823",
        "TRT__R_zopiclone__Q27124198",
        "TRT__S_duloxetine__Q411932",
        "TRT__Tilia__Q127849",
        "TRT__Valeriana_officinalis__Q157819",
        "TRT__Z_thiothixene__Q2608288",
        "TRT__acetylcholinesterase_inhibitor__Q2592323",
        "TRT__alprazolam__Q319877",
        "TRT__amputation__Q477415",
        "TRT__antidepressant__Q76560",
        "TRT__antipsychotics__Q208144",
        "TRT__aripiprazole__Q411188",
        "TRT__armodafinil__Q418913",
        "TRT__artificial_pacemaker__Q372713",
        "TRT__buprenorphine__Q407721",
        "TRT__bupropion__Q834280",
        "TRT__butabarbital__Q410608",
        "TRT__carbamazepin__Q410412",
        "TRT__carphenazine__Q5045786",
        "TRT__chlorpromazine__Q407972",
        "TRT__cidoxepin__Q416791",
        "TRT__clomipramine__Q58713",
        "TRT__clonazepam__Q407988",
        "TRT__clonidine_hydrochloride__Q27292429",
        "TRT__clozapine__Q221361",
        "TRT__continuous_positive_airway_pressure__Q5165502",
        "TRT__dapoxetine_hydrochloride__Q27290682",
        "TRT__desipramine__Q423288",
        "TRT__diazepam__Q210402",
        "TRT__dihydro_ergocryptine__Q905717",
        "TRT__donepezil__Q415081",
        "TRT__escitalopram__Q423757",
        "TRT__ethinamate__Q410225",
        "TRT__feeding_tube__Q1087035",
        "TRT__fluoxetine__Q422244",
        "TRT__flupentixol__Q27164953",
        "TRT__flutoprazepam__Q5462983",
        "TRT__fluvoxamine__Q409236",
        "TRT__full_spectrum_light__Q2532663",
        "TRT__gabapentin__Q410352",
        "TRT__gender_affirming_surgery__Q1053501",
        "TRT__haloperidol__Q251347",
        "TRT__hormone_therapy__Q1628266",
        "TRT__imipramine__Q58396",
        "TRT__lamotrigine__Q410346",
        "TRT__levomethadyl__Q27287577",
        "TRT__levomilnacipran__Q6535779",
        "TRT__light_therapy__Q243570",
        "TRT__lisdexamfetamine_dimesylate__Q27289243",
        "TRT__lithium__Q152763",
        "TRT__liver_transplantation__Q1368191",
        "TRT__lorazepam__Q408265",
        "TRT__maprotiline__Q418361",
        "TRT__mechanical_ventilation__Q3766250",
        "TRT__medication__Q12140",
        "TRT__memantine__Q412189",
        "TRT__mesocarb__Q905058",
        "TRT__methylphenidate__Q422112",
        "TRT__mirtazapine__Q421930",
        "TRT__modafinil__Q410441",
        "TRT__naltrexone__Q409587",
        "TRT__nicotine__Q28086552",
        "TRT__nortriptyline__Q61387",
        "TRT__olanzapine__Q201872",
        "TRT__paroxetine__Q408471",
        "TRT__perlapine__Q27260239",
        "TRT__perphenazine__Q423520",
        "TRT__pimozide__Q144085",
        "TRT__piperacetazine__Q3905512",
        "TRT__psychiatric_medication__Q1572854",
        "TRT__psychoactive_drug__Q3706669",
        "TRT__psychotropic_drug__Q12830468",
        "TRT__quetiapine__Q408535",
        "TRT__risperidone__Q412443",
        "TRT__selective_serotonin_reuptake_inhibitor__Q334477",
        "TRT__sertraline__Q407617",
        "TRT__sodium_oxybate__Q7553347",
        "TRT__stimulant__Q211036",
        "TRT__temazepam__Q414796",
        "TRT__thiobutabarbital__Q906220",
        "TRT__topiramate__Q221174",
        "TRT__transclopenthixol__Q27278051",
        "TRT__transcranial_magnetic_stimulation__Q263962",
        "TRT__trazodone__Q411457",
        "TRT__venlafaxine__Q898407",
        "TRT__wake_therapy__Q7961010",
        "TRT__ziprasidone__Q205517",
        "TRT__zolpidem__Q218842",
    ],
    "PSYCHO": [
        "TRT__Discrete_trial_training__Q5326835",
        "TRT__Lifespan_Integration__Q46325854",
        "TRT__Relationship_Development_Intervention__Q7310740",
        "TRT__TEACCH_approach__Q954190",
        "TRT__Transcendental_Meditation__Q558571",
        "TRT__applied_behavior_analysis__Q621607",
        "TRT__art_therapy__Q928865",
        "TRT__assertiveness_training__Q67135667",
        "TRT__autism_therapy__Q3333688",
        "TRT__behavior_modification__Q2883473",
        "TRT__behavior_therapy__Q16262180",
        "TRT__biofeedback__Q864329",
        "TRT__cognitive_behavioral_therapy__Q1147152",
        "TRT__cognitive_behavioral_therapy_for_insomnia__Q5141192",
        "TRT__cognitive_processing_therapy__Q5141234",
        "TRT__cognitive_remediation_therapy__Q3424681",
        "TRT__cognitive_restructuring__Q849826",
        "TRT__counseling__Q4390239",
        "TRT__dialectical_behavior_therapy__Q1208421",
        "TRT__exposure_and_response_prevention__Q5421633",
        "TRT__eye_movement_desensitization_and_reprocessing__Q1385716",
        "TRT__gender_affirming_therapy__Q15748953",
        "TRT__grief_counseling__Q1118919",
        "TRT__habit_reversal_training__Q1566908",
        "TRT__health_behavior_change__Q4880695",
        "TRT__metacognitive_therapy__Q21011288",
        "TRT__music_therapy__Q209642",
        "TRT__psychodynamic_psychotherapy__Q2279219",
        "TRT__psychoeducation__Q2115980",
        "TRT__psychomotor_education__Q2116083",
        "TRT__psychotherapy__Q183257",
        "TRT__prolonged_exposure_therapy__Q2412412",
        "TRT__sensory_integration_therapy__Q7451132",
        "TRT__supportive_psychotherapy__Q7644630",
        "TRT__trauma_focused_cognitive_behavioral_therapy__Q18354078",

    ],
    "SOCIAL": [
        "TRT__augmentative_and_alternative_communication__Q781083",
        "TRT__occupational_therapy__Q380141",
        "TRT__palliative_care__Q29483",
        "TRT__speech_therapy__Q11888973",
        "TRT__supportive_care__Q3488952",
    ],
    "X": [
        "TRT__transitioning__Q1085588",
        "TRT__Christian_naturism__Q3241587",
        "TRT__Q24198226__Q24198226",
        "TRT__alternative_medicine__Q31338769",
        "TRT__end__Q12769393",
        "TRT__non_existence__Q3877969",
        "TRT__pharmacology__Q128406",
        "TRT__physical_exercise__Q219067",
        "TRT__preventive_medicine__Q1773974",
        "TRT__sleep_hygiene__Q1364783",
        "TRT__superglue__Q107407155",
        "TRT__symptomatic_treatment__Q621558",
        "TRT__treatment_of_paedophilia__Q12034607",
        "TRT__weight_loss__Q718113",
        "TRT__drug_harm_reduction__Q124744973",
        "TRT__early_childhood_intervention__Q3064851",
        "TRT__substance_abuse_treatment__Q1260022",
    ],
}


# ---------------------------
# Logging / timing
# ---------------------------
def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def ts() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str, level: str = "INFO") -> None:
    print(f"[{ts()}] {level}: {msg}", flush=True)


@dataclass
class Timer:
    name: str
    t0: float = 0.0

    def __enter__(self):
        self.t0 = time.time()
        log(f"START: {self.name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        dt_s = time.time() - self.t0
        if exc is None:
            log(f"END:   {self.name} (took {dt_s:.2f}s)")
        else:
            log(f"FAIL:  {self.name} after {dt_s:.2f}s ({exc_type.__name__}: {exc})", level="ERROR")


# ---------------------------
# File utilities
# ---------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def write_json(path: str, obj) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def file_size_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return 0.0


# ---------------------------
# String / ID helpers
# ---------------------------
def qid_from_uri(uri: str) -> str:
    if not isinstance(uri, str):
        return ""
    m = re.search(r"/(Q\d+)$", uri.strip())
    return m.group(1) if m else ""


def pid_from_uri(uri: str) -> str:
    if not isinstance(uri, str):
        return ""
    m = re.search(r"/(P\d+)$", uri.strip())
    return m.group(1) if m else ""


def qid_from_treatment_key(tkey: str) -> str:
    """
    Robust extraction of QID from a treatment_key like 'TRT__Something__Q123'.
    """
    if not isinstance(tkey, str):
        return ""
    m = re.search(r"__(Q\d+)$", tkey.strip())
    return m.group(1) if m else ""


_slug_re = re.compile(r"[^A-Za-z0-9]+")
def slugify(s: str, max_len: int = 80) -> str:
    s = (s or "").strip()
    s = _slug_re.sub("_", s)
    s = s.strip("_")
    if not s:
        return "UNKNOWN"
    return s[:max_len]


def safe_label(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def build_manual_bpsx_override_map(domain_dict: Dict[str, List[str]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns:
      - qid_to_domain_bpsx: QID -> one of {BIO, PSYCHO, SOCIAL, X}
      - key_to_domain_bpsx: treatment_key -> one of {BIO, PSYCHO, SOCIAL, X}

    Validates:
      - only allowed keys
      - no duplicates across domains
      - all listed keys contain a QID suffix
    """
    allowed = {"BIO", "PSYCHO", "SOCIAL", "X"}
    bad_keys = set(domain_dict.keys()) - allowed
    if bad_keys:
        raise ValueError(f"Manual domain dict contains invalid keys: {sorted(bad_keys)} (allowed={sorted(allowed)})")

    all_keys: List[str] = []
    for dom in allowed:
        all_keys.extend(domain_dict.get(dom, []))

    # duplicates
    seen = set()
    dups = []
    for k in all_keys:
        if k in seen:
            dups.append(k)
        seen.add(k)
    if dups:
        raise ValueError(f"Manual domain dict contains duplicate treatment_keys: {dups[:20]}")

    key_to_dom: Dict[str, str] = {}
    qid_to_dom: Dict[str, str] = {}

    qid_dups: List[str] = []
    for dom, keys in domain_dict.items():
        for k in keys:
            key_to_dom[k] = dom
            q = qid_from_treatment_key(k)
            if not q:
                raise ValueError(f"Manual domain dict has treatment_key without QID suffix: {k}")
            if q in qid_to_dom and qid_to_dom[q] != dom:
                qid_dups.append(q)
            qid_to_dom[q] = dom

    if qid_dups:
        raise ValueError(f"Same QID appears in multiple domains in manual dict: {sorted(set(qid_dups))}")

    return qid_to_dom, key_to_dom


MANUAL_QID_TO_BPSX, MANUAL_TKEY_TO_BPSX = build_manual_bpsx_override_map(TREATMENT_DOMAINS_BPSX)


# ---------------------------
# SPARQL client (Session + retries + caching + paging)
# ---------------------------
def make_session(user_agent: str) -> requests.Session:
    sess = requests.Session()
    sess.headers.update(
        {
            "Accept": "application/sparql-results+json",
            "User-Agent": user_agent,
        }
    )
    retry = Retry(
        total=6,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


def sparql_get_json(session: requests.Session, query: str, timeout_s: int = 60) -> dict:
    r = session.get(WIKIDATA_SPARQL_ENDPOINT, params={"query": query}, timeout=timeout_s)
    if r.status_code >= 400:
        txt = (r.text or "")[:300].replace("\n", " ")
        raise requests.HTTPError(f"HTTP {r.status_code}: {txt}", response=r)
    return r.json()


def fetch_paged_sparql(
    session: requests.Session,
    base_query_no_limit: str,
    cache_dir: str,
    cache_prefix: str,
    page_size: int = 5000,
    sleep_s: float = 1.0,
    max_pages: int = 60,
) -> List[dict]:
    """
    Fetch multiple pages using LIMIT/OFFSET. Cache each page JSON.
    """
    ensure_dir(cache_dir)
    all_rows: List[dict] = []

    for page in range(max_pages):
        offset = page * page_size
        query = f"{base_query_no_limit}\nLIMIT {page_size}\nOFFSET {offset}\n"
        cache_path = os.path.join(cache_dir, f"{cache_prefix}__page{page:03d}__off{offset}.json")

        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            source = "cache"
        else:
            data = sparql_get_json(session=session, query=query)
            write_json(cache_path, data)
            source = "web"
            time.sleep(max(0.0, float(sleep_s)))

        bindings = data.get("results", {}).get("bindings", [])
        n = len(bindings)
        log(f"{cache_prefix}: page={page:03d} offset={offset} rows={n} source={source}")

        if not bindings:
            break

        all_rows.extend(bindings)

        if n < page_size:
            break

    log(f"{cache_prefix}: total bindings collected = {len(all_rows)}")
    return all_rows


def bindings_to_df(bindings: List[dict], cols: List[str]) -> pd.DataFrame:
    if not bindings:
        return pd.DataFrame(columns=cols)
    rows = [{c: b.get(c, {}).get("value", "") for c in cols} for b in bindings]
    return pd.DataFrame(rows)


def chunks(lst: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def wd_values_block(qids: List[str]) -> str:
    parts = " ".join([f"wd:{q}" for q in qids if q and q.startswith("Q")])
    return parts


# ---------------------------
# SPARQL queries
# ---------------------------
def make_query_disorder_treatment(
    language: str = "en",
    treatment_pids: Optional[List[str]] = None,
) -> str:
    """
    Mental disorders (subclass/instance-of mental disorder) → treatments via chosen properties.
    Returns which property produced the edge (prop).
    """
    if not treatment_pids:
        treatment_pids = [PID_TREATMENT, PID_POSSIBLE_TREATMENT]

    pid_list = ", ".join([f"wdt:{pid}" for pid in treatment_pids])
    return f"""
SELECT ?disorder ?disorderLabel ?treatment ?treatmentLabel ?prop WHERE {{
  {{
    ?disorder wdt:P31/wdt:P279* wd:{QID_MENTAL_DISORDER} .
  }}
  UNION
  {{
    ?disorder wdt:P279* wd:{QID_MENTAL_DISORDER} .
  }}
  FILTER(?disorder != wd:{QID_MENTAL_DISORDER})
  ?disorder ?prop ?treatment .
  FILTER(?prop IN ({pid_list}))
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{language}". }}
}}
ORDER BY ?disorder ?treatment ?prop
""".strip()


def make_query_disorder_symptom(language: str = "en") -> str:
    return f"""
SELECT ?disorder ?disorderLabel ?symptom ?symptomLabel WHERE {{
  {{
    ?disorder wdt:P31/wdt:P279* wd:{QID_MENTAL_DISORDER} .
  }}
  UNION
  {{
    ?disorder wdt:P279* wd:{QID_MENTAL_DISORDER} .
  }}
  FILTER(?disorder != wd:{QID_MENTAL_DISORDER})
  ?disorder wdt:{PID_SYMPTOM} ?symptom .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{language}". }}
}}
ORDER BY ?disorder ?symptom
""".strip()


def make_query_treatment_enrichment(treatment_qids: List[str]) -> str:
    """
    For a set of treatment QIDs, retrieve boolean flags:
      - has_atc: has ATC code (P267)
      - is_psychotherapy: in psychotherapy hierarchy (Q183257)
      - is_medication_item: in medication hierarchy (Q12140) even without ATC
    """
    vals = wd_values_block(treatment_qids)
    return f"""
SELECT ?treatment ?has_atc ?is_psychotherapy ?is_medication_item WHERE {{
  VALUES ?treatment {{ {vals} }}
  BIND(EXISTS {{ ?treatment wdt:{PID_ATC} ?atc . }} AS ?has_atc)
  BIND(
    EXISTS {{ ?treatment wdt:P279* wd:{QID_PSYCHOTHERAPY} . }} ||
    EXISTS {{ ?treatment wdt:P31/wdt:P279* wd:{QID_PSYCHOTHERAPY} . }}
    AS ?is_psychotherapy
  )
  BIND(
    EXISTS {{ ?treatment wdt:P279* wd:{QID_MEDICATION} . }} ||
    EXISTS {{ ?treatment wdt:P31/wdt:P279* wd:{QID_MEDICATION} . }}
    AS ?is_medication_item
  )
}}
""".strip()


# ---------------------------
# Treatment domain/category logic
# ---------------------------
def _contains_any(text: str, needles: List[str]) -> bool:
    t = (text or "").lower()
    return any(n.lower() in t for n in needles)


BIO_PROCEDURE_KEYWORDS = [
    "electroconvulsive", "ect",
    "transcranial magnetic", "tms",
    "transcranial direct", "tdcs",
    "deep brain stimulation", "dbs",
    "vagus nerve stimulation", "vns",
    "neurostimulation", "neuromodulation",
    "light therapy", "phototherapy",
]

PSYCHO_KEYWORDS = [
    "psychotherapy", "therapy", "counsel", "counselling",
    "cognitive behavioral", "cognitive-behavioral", "cbt",
    "dialectical", "dbt",
    "acceptance and commitment", "act",
    "interpersonal therapy", "ipt",
    "exposure", "emdr",
    "psychoeducation",
    "motivational interviewing",
    "mindfulness-based", "mbct", "mbsr",
]

SOCIAL_KEYWORDS = [
    "supported employment", "housing", "case management", "assertive community",
    "community treatment", "social support", "peer support",
    "family therapy", "family intervention", "multifamily",
    "social skills training", "vocational", "school", "education",
    "rehabilitation", "occupational", "financial assistance",
]

LIFESTYLE_KEYWORDS = [
    "exercise", "physical activity", "sleep", "sleep hygiene",
    "diet", "nutrition", "weight", "yoga", "meditation", "relaxation",
    "breathing", "mindfulness", "stress management",
]

PHARM_KEYWORDS = [
    "antidepress", "antipsych", "ssri", "snri", "benzodia",
    "mood stabil", "lithium", "valpro", "lamotrig",
]


def infer_domain_fine(label: str, has_atc: bool, is_psychotherapy: bool, is_medication_item: bool) -> str:
    """
    Fine-grained domain used for balancing and pharmacology cap.
    Returns one of:
      {bio_pharm, bio_procedure, psycho, social, lifestyle, other}
    """
    l = (label or "").lower()

    if has_atc or is_medication_item:
        return "bio_pharm"
    if is_psychotherapy:
        return "psycho"

    if _contains_any(l, BIO_PROCEDURE_KEYWORDS):
        return "bio_procedure"
    if _contains_any(l, PSYCHO_KEYWORDS):
        return "psycho"
    if _contains_any(l, SOCIAL_KEYWORDS):
        return "social"
    if _contains_any(l, LIFESTYLE_KEYWORDS):
        return "lifestyle"
    if _contains_any(l, PHARM_KEYWORDS):
        return "bio_pharm"

    return "other"


def fine_to_bpsx_domain(domain_fine: str) -> str:
    """
    Collapse fine domain into {BIO, PSYCHO, SOCIAL, X}.
    """
    d = (domain_fine or "").strip().lower()
    if d in {"bio_pharm", "bio_procedure"}:
        return "BIO"
    if d == "psycho":
        return "PSYCHO"
    if d == "social":
        return "SOCIAL"
    return "X"


def infer_category(domain_bpsx: str, domain_fine: str) -> str:
    """
    Compact category label aligned to BIO/PSYCHO/SOCIAL/X.
    """
    if domain_bpsx == "BIO":
        return "medication" if domain_fine == "bio_pharm" else "biological_procedure"
    if domain_bpsx == "PSYCHO":
        return "psychotherapy"
    if domain_bpsx == "SOCIAL":
        return "psychosocial"
    return "other"


def enrich_treatment_flags(
    session: requests.Session,
    treatment_qids: List[str],
    cache_dir: str,
    batch_size: int = 200,
    sleep_s: float = 0.5,
) -> pd.DataFrame:
    """
    Returns DataFrame columns:
      treatment_qid, has_atc (bool), is_psychotherapy (bool), is_medication_item (bool)
    Cached for reproducibility.
    """
    ensure_dir(cache_dir)
    cache_path = os.path.join(cache_dir, "treatment_enrichment_flags.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        log("Treatment enrichment: loaded cached flags", level="INFO")
        bindings = data.get("results", {}).get("bindings", [])
        df = bindings_to_df(bindings, ["treatment", "has_atc", "is_psychotherapy", "is_medication_item"])
    else:
        all_bindings: List[dict] = []
        uniq = sorted(set([q for q in treatment_qids if q and q.startswith("Q")]))
        for i, qchunk in enumerate(chunks(uniq, batch_size), start=1):
            q = make_query_treatment_enrichment(qchunk)
            log(f"Treatment enrichment: batch {i} (n={len(qchunk)})")
            data_chunk = sparql_get_json(session=session, query=q)
            all_bindings.extend(data_chunk.get("results", {}).get("bindings", []))
            time.sleep(max(0.0, float(sleep_s)))

        data = {
            "head": {"vars": ["treatment", "has_atc", "is_psychotherapy", "is_medication_item"]},
            "results": {"bindings": all_bindings},
        }
        write_json(cache_path, data)
        df = bindings_to_df(all_bindings, ["treatment", "has_atc", "is_psychotherapy", "is_medication_item"])
        log("Treatment enrichment: wrote cached flags", level="INFO")

    if df.empty:
        return pd.DataFrame(columns=["treatment_qid", "has_atc", "is_psychotherapy", "is_medication_item"])

    df["treatment_qid"] = df["treatment"].apply(qid_from_uri)
    df["has_atc"] = df["has_atc"].astype(str).str.lower().eq("true")
    df["is_psychotherapy"] = df["is_psychotherapy"].astype(str).str.lower().eq("true")
    df["is_medication_item"] = df["is_medication_item"].astype(str).str.lower().eq("true")
    df = df[["treatment_qid", "has_atc", "is_psychotherapy", "is_medication_item"]].drop_duplicates("treatment_qid")
    return df


def select_treatments_balanced(
    candidates: pd.DataFrame,
    total_max: int,
    max_pharm_fraction: float,
    min_bio_procedure: int,
    min_psycho: int,
    min_social: int,
    min_lifestyle: int,
) -> List[str]:
    """
    Select treatments with diversity and a cap on pharmacology share.
    candidates must contain: qid, label, n_disorders, domain_fine

    Strategy:
      1) Apply minimum quotas for non-pharm domains (and bio_procedure).
      2) Apply max cap for bio_pharm.
      3) Fill remaining slots by degree across remaining pool, respecting bio_pharm cap.
    """
    if candidates.empty:
        return []

    df = candidates.copy()
    df["n_disorders"] = df["n_disorders"].fillna(0).astype(int)

    total_max = max(1, int(total_max))
    max_pharm = max(0, int(floor(total_max * float(max_pharm_fraction))))

    mins = {
        "bio_procedure": max(0, int(min_bio_procedure)),
        "psycho": max(0, int(min_psycho)),
        "social": max(0, int(min_social)),
        "lifestyle": max(0, int(min_lifestyle)),
    }

    sum_mins = sum(mins.values())
    if sum_mins > total_max:
        log(
            f"Sum of minimum domain quotas ({sum_mins}) exceeds max_treatments ({total_max}). Scaling down.",
            level="WARNING",
        )
        scale = total_max / max(1, sum_mins)
        for k in list(mins.keys()):
            mins[k] = max(0, int(floor(mins[k] * scale)))
        for k in ["psycho", "social", "lifestyle"]:
            if total_max >= 3 and mins[k] == 0:
                mins[k] = 1
        if sum(mins.values()) > total_max:
            mins = {k: 0 for k in mins}

    selected: List[str] = []
    selected_set = set()

    def _take_top(domain_fine: str, k: int) -> None:
        if k <= 0:
            return
        sub = df[(df["domain_fine"] == domain_fine) & (~df["qid"].isin(selected_set))].copy()
        if sub.empty:
            return
        sub = sub.sort_values(["n_disorders", "label"], ascending=[False, True]).head(k)
        for q in sub["qid"].tolist():
            if q not in selected_set:
                selected.append(q)
                selected_set.add(q)

    for dom in ["bio_procedure", "psycho", "social", "lifestyle"]:
        _take_top(dom, mins.get(dom, 0))

    _take_top("bio_pharm", max_pharm)

    remaining = total_max - len(selected)
    if remaining > 0:
        rest = df[~df["qid"].isin(selected_set)].copy()
        rest = rest.sort_values(["n_disorders", "label"], ascending=[False, True])

        n_pharm_sel = df[df["qid"].isin(selected_set) & (df["domain_fine"] == "bio_pharm")].shape[0]

        for qid, dom in zip(rest["qid"].tolist(), rest["domain_fine"].tolist()):
            if len(selected) >= total_max:
                break
            if dom == "bio_pharm" and n_pharm_sel >= max_pharm:
                continue
            selected.append(qid)
            selected_set.add(qid)
            if dom == "bio_pharm":
                n_pharm_sel += 1

    return selected[:total_max]


# ---------------------------
# Core build logic
# ---------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--user-agent", type=str, default=DEFAULT_USER_AGENT)
    parser.add_argument("--language", type=str, default="en", help="Label language for Wikidata label service.")

    parser.add_argument("--max-disorders", type=int, default=140, help="Keep top-N disorders by treatment-degree.")
    parser.add_argument("--max-treatments", type=int, default=160, help="Keep top-N treatments (balanced by domain if enabled).")
    parser.add_argument("--max-symptoms", type=int, default=120, help="Keep top-N symptoms by disorder-frequency (within kept disorders).")
    parser.add_argument("--min-symptom-support", type=int, default=2, help="Min number of supporting disorders to infer symptom→treatment link.")

    parser.add_argument("--page-size", type=int, default=5000)
    parser.add_argument("--sleep-s", type=float, default=1.0)
    parser.add_argument("--max-pages", type=int, default=60)

    parser.add_argument("--enrich-treatment-metadata", type=int, default=1, help="1=use ATC/psychotherapy/medication enrichment, 0=heuristics only.")

    # Balance controls (fine-grained)
    parser.add_argument("--treatments-balance-mode", type=str, default="balanced", choices=["balanced", "degree"],
                        help="balanced=enforce diversity; degree=top-N by disorder-degree.")
    parser.add_argument("--max-pharm-fraction", type=float, default=0.35, help="Max fraction of pharmacological treatments (bio_pharm) in kept treatments.")
    parser.add_argument("--min-bio-procedure", type=int, default=10, help="Minimum bio_procedure treatments to keep (if available).")
    parser.add_argument("--min-psycho", type=int, default=35, help="Minimum psycho treatments to keep (if available).")
    parser.add_argument("--min-social", type=int, default=35, help="Minimum social treatments to keep (if available).")
    parser.add_argument("--min-lifestyle", type=int, default=15, help="Minimum lifestyle treatments to keep (if available).")

    parser.add_argument("--treatment-pids", type=str, default=f"{PID_TREATMENT},{PID_POSSIBLE_TREATMENT}",
                        help="Comma-separated list of Wikidata PIDs for disorder→treatment edges (default: P2176,P924).")

    parser.add_argument("--dtype", type=str, default="uint8", choices=["uint8", "int8", "int16"], help="Matrix dtype to reduce file size.")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    ensure_dir(out_dir)
    cache_dir = os.path.join(out_dir, "_cache_wikidata")
    ensure_dir(cache_dir)

    session = make_session(args.user_agent)

    # File targets
    out_csv = os.path.join(out_dir, "01_create_annotated_problem_solution_map.csv")
    out_edges = os.path.join(out_dir, "problem_solution_edges.tsv")

    f_pairs_dt = os.path.join(out_dir, "raw_pairs_disorder_treatment.tsv")
    f_pairs_ds = os.path.join(out_dir, "raw_pairs_disorder_symptom.tsv")
    f_pairs_st = os.path.join(out_dir, "inferred_pairs_symptom_treatment.tsv")
    f_lookup_problems = os.path.join(out_dir, "problems_lookup.tsv")
    f_lookup_treatments = os.path.join(out_dir, "treatments_lookup.tsv")
    f_queries = os.path.join(out_dir, "sparql_queries.txt")
    f_prov = os.path.join(out_dir, "provenance.json")
    f_log = os.path.join(out_dir, "run_log.txt")

    # Parse treatment PIDs
    treatment_pids = [p.strip() for p in str(args.treatment_pids).split(",") if p.strip().startswith("P")]
    if not treatment_pids:
        treatment_pids = [PID_TREATMENT, PID_POSSIBLE_TREATMENT]

    log("MENTAL HEALTH PROBLEM→SOLUTION MAP (Wikidata SPARQL)")
    log(f"run_time: {now_iso()}")
    log(f"out_dir:  {out_dir}")
    log(f"endpoint: {WIKIDATA_SPARQL_ENDPOINT}")
    log(f"seed mental disorder: wd:{QID_MENTAL_DISORDER}")
    log(f"direct disorder→treatment PIDs: {','.join(treatment_pids)}")
    log(f"params: max_disorders={args.max_disorders}, max_treatments={args.max_treatments}, max_symptoms={args.max_symptoms}, min_symptom_support={args.min_symptom_support}")
    log(f"balance: mode={args.treatments_balance_mode}, max_pharm_fraction={args.max_pharm_fraction}, mins(bio_proc/psycho/social/lifestyle)={args.min_bio_procedure}/{args.min_psycho}/{args.min_social}/{args.min_lifestyle}")
    log(f"paging: page_size={args.page_size}, sleep_s={args.sleep_s}, max_pages={args.max_pages}")
    log(f"manual BPSX overrides loaded: {len(MANUAL_QID_TO_BPSX):,} QIDs")

    # ---------------------------
    # 1) Fetch disorder→treatment
    # ---------------------------
    with Timer("Fetch disorder→treatment (selected properties)"):
        q_dt = make_query_disorder_treatment(language=args.language, treatment_pids=treatment_pids)
        dt_bindings = fetch_paged_sparql(
            session=session,
            base_query_no_limit=q_dt,
            cache_dir=cache_dir,
            cache_prefix="disorder_treatment",
            page_size=args.page_size,
            sleep_s=args.sleep_s,
            max_pages=args.max_pages,
        )
    df_dt = bindings_to_df(dt_bindings, ["disorder", "disorderLabel", "treatment", "treatmentLabel", "prop"])
    if df_dt.empty:
        raise SystemExit("No disorder→treatment rows retrieved. Check internet access or query constraints.")
    log(f"Raw disorder→treatment rows (before normalize): {len(df_dt):,}")

    with Timer("Normalize disorder→treatment"):
        df_dt["disorder_qid"] = df_dt["disorder"].map(qid_from_uri)
        df_dt["treatment_qid"] = df_dt["treatment"].map(qid_from_uri)
        df_dt["source_pid"] = df_dt["prop"].map(pid_from_uri)
        df_dt["disorder_label"] = df_dt["disorderLabel"].map(safe_label)
        df_dt["treatment_label"] = df_dt["treatmentLabel"].map(safe_label)

        df_dt = df_dt[df_dt["disorder_qid"].str.startswith("Q") & df_dt["treatment_qid"].str.startswith("Q")]
        df_dt = df_dt.drop_duplicates(subset=["disorder_qid", "treatment_qid", "source_pid"]).reset_index(drop=True)

    log(f"Normalized disorder→treatment unique triples (disorder,treatment,prop): {len(df_dt):,}")

    # ---------------------------
    # 2) Fetch disorder→symptom
    # ---------------------------
    with Timer("Fetch disorder→symptom (P780)"):
        q_ds = make_query_disorder_symptom(language=args.language)
        ds_bindings = fetch_paged_sparql(
            session=session,
            base_query_no_limit=q_ds,
            cache_dir=cache_dir,
            cache_prefix="disorder_symptom",
            page_size=args.page_size,
            sleep_s=args.sleep_s,
            max_pages=args.max_pages,
        )
    df_ds = bindings_to_df(ds_bindings, ["disorder", "disorderLabel", "symptom", "symptomLabel"])

    with Timer("Normalize disorder→symptom"):
        if not df_ds.empty:
            df_ds["disorder_qid"] = df_ds["disorder"].map(qid_from_uri)
            df_ds["symptom_qid"] = df_ds["symptom"].map(qid_from_uri)
            df_ds["disorder_label"] = df_ds["disorderLabel"].map(safe_label)
            df_ds["symptom_label"] = df_ds["symptomLabel"].map(safe_label)
            df_ds = df_ds[df_ds["disorder_qid"].str.startswith("Q") & df_ds["symptom_qid"].str.startswith("Q")]
            df_ds = df_ds.drop_duplicates(subset=["disorder_qid", "symptom_qid"]).reset_index(drop=True)
        else:
            df_ds = pd.DataFrame(columns=["disorder_qid", "symptom_qid", "disorder_label", "symptom_label"])

    log(f"Normalized disorder→symptom unique pairs: {len(df_ds):,}")

    # ---------------------------
    # 3) Select top disorders
    # ---------------------------
    with Timer("Select top disorders"):
        dd = (
            df_dt.groupby(["disorder_qid", "disorder_label"])["treatment_qid"]
            .nunique()
            .reset_index(name="n_treatments")
            .sort_values("n_treatments", ascending=False)
        )
        keep_disorders = dd.head(int(args.max_disorders))["disorder_qid"].tolist()

    df_dt_k0 = df_dt[df_dt["disorder_qid"].isin(keep_disorders)].copy()
    df_ds_k0 = df_ds[df_ds["disorder_qid"].isin(keep_disorders)].copy()

    log(f"Kept disorders: {len(keep_disorders):,}")
    log(f"Pairs (kept disorders): disorder→treatment triples={len(df_dt_k0):,}, disorder→symptom pairs={len(df_ds_k0):,}")

    # ---------------------------
    # 4) Candidate treatments + degrees
    # ---------------------------
    with Timer("Compute candidate treatment degrees"):
        treatment_degree = (
            df_dt_k0.groupby(["treatment_qid", "treatment_label"])["disorder_qid"]
            .nunique()
            .reset_index(name="n_disorders")
            .sort_values("n_disorders", ascending=False)
            .reset_index(drop=True)
        )
        candidates = treatment_degree.copy()

    # ---------------------------
    # 5) Enrich treatment metadata + assign domain_fine + domain_bpsx (with manual override)
    # ---------------------------
    with Timer("Build treatment metadata (candidates)"):
        treatments_df = candidates.rename(columns={"treatment_qid": "qid", "treatment_label": "label"}).copy()
        treatments_df["treatment_key"] = treatments_df.apply(lambda r: f"TRT__{slugify(r['label'])}__{r['qid']}", axis=1)

        if int(args.enrich_treatment_metadata) == 1:
            with Timer("Enrich treatment flags (ATC / psychotherapy / medication hierarchy)"):
                flags_df = enrich_treatment_flags(
                    session=session,
                    treatment_qids=treatments_df["qid"].tolist(),
                    cache_dir=cache_dir,
                    batch_size=200,
                    sleep_s=0.5,
                )
            treatments_df = treatments_df.merge(flags_df, left_on="qid", right_on="treatment_qid", how="left")
            treatments_df.drop(columns=["treatment_qid"], inplace=True, errors="ignore")
            treatments_df["has_atc"] = treatments_df["has_atc"].fillna(False).astype(bool)
            treatments_df["is_psychotherapy"] = treatments_df["is_psychotherapy"].fillna(False).astype(bool)
            treatments_df["is_medication_item"] = treatments_df["is_medication_item"].fillna(False).astype(bool)
        else:
            treatments_df["has_atc"] = False
            treatments_df["is_psychotherapy"] = False
            treatments_df["is_medication_item"] = False

        # Fine-grained domain (for balancing and pharm cap)
        treatments_df["domain_fine"] = treatments_df.apply(
            lambda r: infer_domain_fine(
                label=str(r.get("label", "")),
                has_atc=bool(r.get("has_atc", False)),
                is_psychotherapy=bool(r.get("is_psychotherapy", False)),
                is_medication_item=bool(r.get("is_medication_item", False)),
            ),
            axis=1,
        )

        # BPSX domain (for downstream analysis) with manual override by QID
        def _bpsx_domain(row) -> Tuple[str, str]:
            q = str(row.get("qid", ""))
            if q in MANUAL_QID_TO_BPSX:
                return MANUAL_QID_TO_BPSX[q], "manual_override"
            return fine_to_bpsx_domain(str(row.get("domain_fine", ""))), "inferred_from_fine"

        tmp = treatments_df.apply(_bpsx_domain, axis=1, result_type="expand")
        treatments_df["domain_bpsx"] = tmp[0]
        treatments_df["domain_bpsx_source"] = tmp[1]

        # Compact category aligned to BPSX (still uses domain_fine to distinguish meds vs procedures inside BIO)
        treatments_df["category"] = treatments_df.apply(
            lambda r: infer_category(str(r.get("domain_bpsx", "X")), str(r.get("domain_fine", "other"))),
            axis=1,
        )

        treatments_df = treatments_df.sort_values(
            ["n_disorders", "domain_fine", "label"], ascending=[False, True, True]
        ).reset_index(drop=True)

        n_manual_used = int((treatments_df["domain_bpsx_source"] == "manual_override").sum())
        log(f"BPSX manual override applied to {n_manual_used:,} treatments present in candidate set")

    # ---------------------------
    # 6) Select treatments (balanced vs degree) using domain_fine
    # ---------------------------
    with Timer("Select treatments (balanced vs degree)"):
        if str(args.treatments_balance_mode).lower() == "degree":
            keep_treatments = treatments_df.head(int(args.max_treatments))["qid"].tolist()
        else:
            keep_treatments = select_treatments_balanced(
                candidates=treatments_df[["qid", "label", "n_disorders", "domain_fine"]].copy(),
                total_max=int(args.max_treatments),
                max_pharm_fraction=float(args.max_pharm_fraction),
                min_bio_procedure=int(args.min_bio_procedure),
                min_psycho=int(args.min_psycho),
                min_social=int(args.min_social),
                min_lifestyle=int(args.min_lifestyle),
            )

        keep_treatments = [q for q in keep_treatments if isinstance(q, str) and q.startswith("Q")]
        keep_treatments = list(dict.fromkeys(keep_treatments))  # preserve order, unique

        treatments_df = treatments_df[treatments_df["qid"].isin(keep_treatments)].copy()
        order_map = {qid: i for i, qid in enumerate(keep_treatments)}
        treatments_df["__ord"] = treatments_df["qid"].map(order_map).fillna(10**9).astype(int)
        treatments_df = treatments_df.sort_values("__ord", ascending=True).drop(columns=["__ord"]).reset_index(drop=True)

    df_dt_k = df_dt_k0[df_dt_k0["treatment_qid"].isin(keep_treatments)].copy()

    # Symptoms selection
    with Timer("Select top symptoms"):
        if not df_ds_k0.empty:
            sd = (
                df_ds_k0.groupby(["symptom_qid", "symptom_label"])["disorder_qid"]
                .nunique()
                .reset_index(name="n_disorders")
                .sort_values("n_disorders", ascending=False)
            )
            keep_symptoms = sd.head(int(args.max_symptoms))["symptom_qid"].tolist()
            df_ds_k = df_ds_k0[df_ds_k0["symptom_qid"].isin(keep_symptoms)].copy()
        else:
            keep_symptoms = []
            df_ds_k = df_ds_k0.copy()

    log(f"Kept: disorders={len(keep_disorders):,}  treatments={len(keep_treatments):,}  symptoms={len(keep_symptoms):,}")
    log(f"Kept pairs: disorder→treatment triples={len(df_dt_k):,}  disorder→symptom pairs={len(df_ds_k):,}")

    # Composition logs
    dom_fine_counts = treatments_df["domain_fine"].value_counts(dropna=False).to_dict()
    dom_bpsx_counts = treatments_df["domain_bpsx"].value_counts(dropna=False).to_dict()
    pharm_n = int(dom_fine_counts.get("bio_pharm", 0))
    total_n = int(len(treatments_df))
    pharm_frac = (pharm_n / total_n) if total_n else 0.0
    log(f"Treatment domain_fine composition: {dom_fine_counts}")
    log(f"Treatment domain_bpsx composition: {dom_bpsx_counts}")
    log(f"Pharmacology share (domain_fine=bio_pharm): {pharm_n}/{total_n} = {pharm_frac:.3f}")

    # Label maps
    disorder_label_map: Dict[str, str] = (
        df_dt_k.drop_duplicates("disorder_qid").set_index("disorder_qid")["disorder_label"].to_dict()
        if not df_dt_k.empty else
        df_dt_k0.drop_duplicates("disorder_qid").set_index("disorder_qid")["disorder_label"].to_dict()
    )
    treatment_label_map: Dict[str, str] = treatments_df.set_index("qid")["label"].to_dict()
    symptom_label_map: Dict[str, str] = (
        df_ds_k.drop_duplicates("symptom_qid").set_index("symptom_qid")["symptom_label"].to_dict()
        if not df_ds_k.empty
        else {}
    )

    # ---------------------------
    # 7) Build lookup tables
    # ---------------------------
    with Timer("Build lookup tables (problems/treatments)"):
        dx_keys: List[Tuple[str, str, str, str]] = []
        for qid in sorted(disorder_label_map.keys()):
            lab = disorder_label_map[qid]
            dx_keys.append((f"DX__{slugify(lab)}__{qid}", "DX", lab, qid))

        sym_keys: List[Tuple[str, str, str, str]] = []
        for qid in sorted(symptom_label_map.keys()):
            lab = symptom_label_map[qid]
            sym_keys.append((f"SYM__{slugify(lab)}__{qid}", "SYM", lab, qid))

        problems = dx_keys + sym_keys
        problems_df = pd.DataFrame(problems, columns=["problem_key", "kind", "label", "qid"])
        problems_df.to_csv(f_lookup_problems, sep="\t", index=False)

        treatments_lookup = treatments_df.rename(columns={"n_disorders": "n_disorders_direct"}).copy()
        treatments_lookup = treatments_lookup[
            [
                "treatment_key", "label", "qid", "n_disorders_direct",
                "domain_bpsx", "domain_bpsx_source",
                "domain_fine", "category",
                "has_atc", "is_psychotherapy", "is_medication_item",
            ]
        ].copy()
        treatments_lookup.to_csv(f_lookup_treatments, sep="\t", index=False)

    # ---------------------------
    # 8) Save raw direct pairs
    # ---------------------------
    with Timer("Write raw direct pair files"):
        df_dt_out = df_dt_k[["disorder_qid", "disorder_label", "treatment_qid", "treatment_label", "source_pid"]].copy()
        df_dt_out.to_csv(f_pairs_dt, sep="\t", index=False)

        if not df_ds_k.empty:
            df_ds_out = df_ds_k[["disorder_qid", "disorder_label", "symptom_qid", "symptom_label"]].copy()
            df_ds_out.to_csv(f_pairs_ds, sep="\t", index=False)
        else:
            write_text(f_pairs_ds, "No disorder→symptom data retrieved; file intentionally empty.\n")

    # ---------------------------
    # 9) Infer SYM→TRT edges via co-occurrence
    # ---------------------------
    df_inferred = pd.DataFrame(columns=["symptom_qid", "treatment_qid", "supporting_disorders_count"])
    min_support = int(args.min_symptom_support)

    with Timer("Infer symptom→treatment edges via co-occurrence"):
        if (not df_ds_k.empty) and (not df_dt_k.empty) and len(keep_symptoms) > 0:
            dt_pairs_unique = df_dt_k[["disorder_qid", "treatment_qid"]].drop_duplicates()

            merged = df_ds_k[["disorder_qid", "symptom_qid"]].merge(
                dt_pairs_unique,
                on="disorder_qid",
                how="inner",
            )
            support_df = (
                merged.groupby(["symptom_qid", "treatment_qid"])
                .size()
                .reset_index(name="supporting_disorders_count")
            )
            support_df = support_df[
                (support_df["supporting_disorders_count"] >= min_support)
                & (support_df["symptom_qid"].isin(keep_symptoms))
                & (support_df["treatment_qid"].isin(keep_treatments))
            ].copy()

            df_inferred = support_df.sort_values(
                ["supporting_disorders_count", "symptom_qid", "treatment_qid"],
                ascending=[False, True, True],
            ).reset_index(drop=True)

            df_inferred_out = df_inferred.copy()
            df_inferred_out["symptom_label"] = df_inferred_out["symptom_qid"].map(symptom_label_map).fillna("")
            df_inferred_out["treatment_label"] = df_inferred_out["treatment_qid"].map(treatment_label_map).fillna("")
            df_inferred_out["min_support_threshold"] = min_support
            df_inferred_out = df_inferred_out[
                ["symptom_qid", "symptom_label", "treatment_qid", "treatment_label", "supporting_disorders_count", "min_support_threshold"]
            ]
            df_inferred_out.to_csv(f_pairs_st, sep="\t", index=False)
            log(f"Inferred symptom→treatment edges kept: {len(df_inferred):,}")
        else:
            write_text(f_pairs_st, "No inferred symptom→treatment edges met threshold (or missing symptom data).\n")
            log("No inferred symptom→treatment edges computed (insufficient data).", level="WARNING")

    # ---------------------------
    # 10) Build final matrix + write CSV
    # ---------------------------
    with Timer("Build final matrix + write CSV"):
        problem_cols = problems_df["problem_key"].tolist()

        dx_qid_to_key = {qid: key for key, kind, lab, qid in dx_keys}
        sym_qid_to_key = {qid: key for key, kind, lab, qid in sym_keys}
        trt_qid_to_key = {qid: f"TRT__{slugify(treatment_label_map[qid])}__{qid}" for qid in treatment_label_map.keys()}

        treatment_keys_ordered = treatments_df["treatment_key"].tolist()
        trtkey_to_row = {k: i for i, k in enumerate(treatment_keys_ordered)}
        pkey_to_col = {k: j for j, k in enumerate(problem_cols)}

        dtype_map = {"uint8": np.uint8, "int8": np.int8, "int16": np.int16}
        mat_dtype = dtype_map[args.dtype]
        X = np.zeros((len(treatment_keys_ordered), len(problem_cols)), dtype=mat_dtype)

        dt_pairs = df_dt_k[["disorder_qid", "treatment_qid"]].drop_duplicates().copy()
        dt_pairs["problem_key"] = dt_pairs["disorder_qid"].map(dx_qid_to_key)
        dt_pairs["treatment_key"] = dt_pairs["treatment_qid"].map(trt_qid_to_key)
        dt_pairs = dt_pairs.dropna(subset=["problem_key", "treatment_key"])

        dt_pairs["row"] = dt_pairs["treatment_key"].map(trtkey_to_row)
        dt_pairs["col"] = dt_pairs["problem_key"].map(pkey_to_col)
        dt_pairs = dt_pairs.dropna(subset=["row", "col"])

        rows = dt_pairs["row"].astype(int).to_numpy()
        cols = dt_pairs["col"].astype(int).to_numpy()
        X[rows, cols] = 1

        n_direct = len(dt_pairs)
        log(f"Matrix fill: direct disorder→treatment edges set = {n_direct:,}")

        n_inferred = 0
        if not df_inferred.empty:
            inf = df_inferred.copy()
            inf["problem_key"] = inf["symptom_qid"].map(sym_qid_to_key)
            inf["treatment_key"] = inf["treatment_qid"].map(trt_qid_to_key)
            inf = inf.dropna(subset=["problem_key", "treatment_key"])

            inf["row"] = inf["treatment_key"].map(trtkey_to_row)
            inf["col"] = inf["problem_key"].map(pkey_to_col)
            inf = inf.dropna(subset=["row", "col"])

            irows = inf["row"].astype(int).to_numpy()
            icols = inf["col"].astype(int).to_numpy()
            X[irows, icols] = 1
            n_inferred = len(inf)
            log(f"Matrix fill: inferred symptom→treatment edges set = {n_inferred:,}")

        matrix_df = pd.DataFrame(X, columns=problem_cols)
        out = pd.concat([treatments_lookup.reset_index(drop=True), matrix_df], axis=1)
        out.to_csv(out_csv, index=False)

        density = float(X.mean()) if X.size else 0.0
        log(f"Matrix: rows={X.shape[0]:,} cols={X.shape[1]:,} density(mean)={density:.6f}")
        log(f"Wrote CSV: {out_csv} ({file_size_mb(out_csv):.2f} MB)")

    # ---------------------------
    # 11) Write dense edge-list
    # ---------------------------
    with Timer("Write dense edge-list (problem_solution_edges.tsv)"):
        # Direct edges (DX only)
        direct_edges = dt_pairs.copy()
        direct_edges["value"] = 1
        direct_edges["evidence_type"] = "direct_disorder_treatment"
        direct_edges["problem_kind"] = "DX"
        direct_edges["problem_qid"] = direct_edges["disorder_qid"]
        direct_edges["problem_label"] = direct_edges["problem_qid"].map(disorder_label_map).fillna("")
        direct_edges["treatment_label"] = direct_edges["treatment_qid"].map(treatment_label_map).fillna("")

        # Inferred edges (SYM only)
        if not df_inferred.empty:
            inferred_edges = df_inferred.copy()
            inferred_edges["problem_qid"] = inferred_edges["symptom_qid"]
            inferred_edges["problem_label"] = inferred_edges["problem_qid"].map(symptom_label_map).fillna("")
            inferred_edges["problem_kind"] = "SYM"
            inferred_edges["problem_key"] = inferred_edges["symptom_qid"].map(sym_qid_to_key)
            inferred_edges["treatment_key"] = inferred_edges["treatment_qid"].map(trt_qid_to_key)
            inferred_edges["value"] = 1
            inferred_edges["evidence_type"] = "inferred_symptom_treatment"
            inferred_edges["treatment_label"] = inferred_edges["treatment_qid"].map(treatment_label_map).fillna("")
        else:
            inferred_edges = pd.DataFrame(columns=[
                "treatment_qid", "treatment_key", "treatment_label",
                "problem_qid", "problem_key", "problem_label", "problem_kind",
                "value", "evidence_type", "supporting_disorders_count"
            ])

        direct_export = pd.DataFrame({
            "treatment_key": direct_edges["treatment_key"],
            "treatment_label": direct_edges["treatment_label"],
            "treatment_qid": direct_edges["treatment_qid"],
            "problem_key": direct_edges["problem_key"],
            "problem_label": direct_edges["problem_label"],
            "problem_qid": direct_edges["problem_qid"],
            "problem_kind": direct_edges["problem_kind"],
            "value": direct_edges["value"],
            "evidence_type": direct_edges["evidence_type"],
            "supporting_disorders_count": np.nan,
            "min_support_threshold": np.nan,
        })

        inferred_export = pd.DataFrame({
            "treatment_key": inferred_edges.get("treatment_key", pd.Series(dtype=str)),
            "treatment_label": inferred_edges.get("treatment_label", pd.Series(dtype=str)),
            "treatment_qid": inferred_edges.get("treatment_qid", pd.Series(dtype=str)),
            "problem_key": inferred_edges.get("problem_key", pd.Series(dtype=str)),
            "problem_label": inferred_edges.get("problem_label", pd.Series(dtype=str)),
            "problem_qid": inferred_edges.get("problem_qid", pd.Series(dtype=str)),
            "problem_kind": inferred_edges.get("problem_kind", pd.Series(dtype=str)),
            "value": inferred_edges.get("value", pd.Series(dtype=int)),
            "evidence_type": inferred_edges.get("evidence_type", pd.Series(dtype=str)),
            "supporting_disorders_count": inferred_edges.get("supporting_disorders_count", pd.Series(dtype=float)),
            "min_support_threshold": float(min_support) if not df_inferred.empty else np.nan,
        })

        edges = pd.concat([direct_export, inferred_export], axis=0, ignore_index=True)

        # Add treatment metadata (BPSX + fine)
        tmeta = treatments_lookup.set_index("treatment_key")[["domain_bpsx", "domain_fine", "category"]].to_dict(orient="index")
        edges["treatment_domain_bpsx"] = edges["treatment_key"].map(lambda k: (tmeta.get(k, {}) or {}).get("domain_bpsx", ""))
        edges["treatment_domain_fine"] = edges["treatment_key"].map(lambda k: (tmeta.get(k, {}) or {}).get("domain_fine", ""))
        edges["treatment_category"] = edges["treatment_key"].map(lambda k: (tmeta.get(k, {}) or {}).get("category", ""))

        edges.to_csv(out_edges, sep="\t", index=False)
        log(f"Wrote edge-list: {out_edges} ({file_size_mb(out_edges):.2f} MB), rows={len(edges):,}")

    # ---------------------------
    # 12) Provenance + log + queries
    # ---------------------------
    with Timer("Write provenance + run log + SPARQL queries"):
        stats = {
            "run_time": now_iso(),
            "out_dir": out_dir,
            "wikidata_endpoint": WIKIDATA_SPARQL_ENDPOINT,
            "seed_class_mental_disorder_qid": QID_MENTAL_DISORDER,
            "direct_treatment_properties": {pid: pid for pid in treatment_pids},
            "properties_used": {
                PID_TREATMENT: "drug or therapy used for treatment",
                PID_POSSIBLE_TREATMENT: "possible treatment",
                PID_SYMPTOM: "symptoms and signs",
                PID_ATC: "ATC code (used only for treatment categorization)",
            },
            "parameters": {
                "max_disorders": int(args.max_disorders),
                "max_treatments": int(args.max_treatments),
                "max_symptoms": int(args.max_symptoms),
                "min_symptom_support": int(args.min_symptom_support),
                "page_size": int(args.page_size),
                "sleep_s": float(args.sleep_s),
                "max_pages": int(args.max_pages),
                "language": str(args.language),
                "enrich_treatment_metadata": int(args.enrich_treatment_metadata),
                "treatments_balance_mode": str(args.treatments_balance_mode),
                "max_pharm_fraction": float(args.max_pharm_fraction),
                "min_bio_procedure": int(args.min_bio_procedure),
                "min_psycho": int(args.min_psycho),
                "min_social": int(args.min_social),
                "min_lifestyle": int(args.min_lifestyle),
                "matrix_dtype": str(args.dtype),
            },
            "manual_bpsx_override": {
                "n_manual_qids": int(len(MANUAL_QID_TO_BPSX)),
                "domain_counts_in_manual_dict": {k: int(len(v)) for k, v in TREATMENT_DOMAINS_BPSX.items()},
                "n_manual_applied_in_candidate_set": int((treatments_df["domain_bpsx_source"] == "manual_override").sum()),
            },
            "retrieved_counts": {
                "dt_triples_raw_total": int(df_dt.shape[0]),
                "ds_pairs_raw_total": int(df_ds.shape[0]),
                "dt_triples_kept": int(df_dt_k.shape[0]),
                "ds_pairs_kept": int(df_ds_k.shape[0]),
                "n_disorders_kept": int(len(keep_disorders)),
                "n_treatments_kept": int(len(keep_treatments)),
                "n_symptoms_kept": int(len(keep_symptoms)),
                "n_inferred_symptom_treatment_edges": int(df_inferred.shape[0]) if not df_inferred.empty else 0,
            },
            "treatments_domain_fine_counts": dom_fine_counts,
            "treatments_domain_bpsx_counts": dom_bpsx_counts,
            "matrix": {
                "rows_treatments": int(len(treatments_df)),
                "cols_problems_total": int(len(problems_df)),
                "cols_dx": int(len(dx_keys)),
                "cols_sym": int(len(sym_keys)),
                "density_fraction_ones": float(X.mean()) if X.size else 0.0,
            },
            "outputs": {
                "wide_matrix_csv": os.path.basename(out_csv),
                "dense_edge_list_tsv": os.path.basename(out_edges),
            },
            "warnings": [
                "This mapping is plausibility-oriented (community-curated Wikidata links), not a clinical guideline.",
                "Symptom→treatment edges are inferred via co-occurring disorders; interpret as weak evidence unless validated.",
                "Domain/category labels are heuristic+enrichment and may misclassify some interventions.",
            ],
        }
        write_json(f_prov, stats)

        write_text(
            f_queries,
            "=== disorder→treatment (selected properties) ===\n"
            + q_dt
            + "\n\n=== disorder→symptom (P780) ===\n"
            + q_ds
            + "\n"
        )

        lines: List[str] = []
        lines.append("MENTAL HEALTH PROBLEM→SOLUTION MAP (Wikidata SPARQL)")
        lines.append("-" * 88)
        lines.append(f"run_time: {stats['run_time']}")
        lines.append(f"out_dir:  {out_dir}")
        lines.append(f"out_csv:  {out_csv}")
        lines.append(f"out_edges:{out_edges}")
        lines.append("")
        lines.append("Kept set sizes:")
        lines.append(f"  disorders:  {stats['retrieved_counts']['n_disorders_kept']}")
        lines.append(f"  treatments: {stats['retrieved_counts']['n_treatments_kept']}")
        lines.append(f"  symptoms:   {stats['retrieved_counts']['n_symptoms_kept']}")
        lines.append(f"  inferred symptom→treatment edges: {stats['retrieved_counts']['n_inferred_symptom_treatment_edges']}")
        lines.append("")
        lines.append("Treatment domain_fine composition:")
        lines.append(json.dumps(dom_fine_counts, indent=2))
        lines.append("")
        lines.append("Treatment domain_bpsx composition:")
        lines.append(json.dumps(dom_bpsx_counts, indent=2))
        lines.append("")
        lines.append("Matrix:")
        lines.append(f"  rows (treatments): {stats['matrix']['rows_treatments']}")
        lines.append(f"  cols (problems):   {stats['matrix']['cols_problems_total']} (DX={stats['matrix']['cols_dx']}, SYM={stats['matrix']['cols_sym']})")
        lines.append(f"  density (mean cell value): {stats['matrix']['density_fraction_ones']:.6f}")
        lines.append("")
        lines.append("Top 15 treatments by direct disorder-degree:")
        top15 = treatments_df.head(15)[["label", "qid", "n_disorders", "domain_bpsx", "domain_fine", "category"]]
        lines.append(top15.to_csv(sep="\t", index=False))
        lines.append("")
        lines.append("Output files:")
        for p in [out_csv, out_edges, f_lookup_problems, f_lookup_treatments, f_pairs_dt, f_pairs_ds, f_pairs_st, f_queries, f_prov, f_log]:
            lines.append(f"  - {os.path.basename(p)} ({file_size_mb(p):.2f} MB)")
        lines.append("")
        lines.append("Warnings:")
        for w in stats["warnings"]:
            lines.append(f"  - {w}")

        write_text(f_log, "\n".join(lines))

    # ---------------------------
    # 13) Console summary
    # ---------------------------
    log("-" * 88)
    log("RUN SUMMARY")
    log(f"Wide CSV matrix: {out_csv}")
    log(f"Dense edge-list: {out_edges}")
    log(f"  rows(treatments): {len(treatments_df):,}")
    log(f"  cols(problems):   {len(problems_df):,} (DX={len(dx_keys):,}, SYM={len(sym_keys):,})")
    log(f"  dtype:            {args.dtype}")
    log(f"  direct edges:     filled={n_direct:,}")
    log(f"  inferred edges:   filled={n_inferred:,}")
    log("Output structure:")
    log(f"  - {os.path.basename(out_csv)} : treatment metadata + binary columns (problem_key)")
    log(f"  - {os.path.basename(out_edges)} : dense edge-list (treatment_key, problem_key, evidence, support)")
    log(f"  - {os.path.basename(f_lookup_problems)} : problem_key ↔ (DX/SYM) ↔ label ↔ QID")
    log(f"  - {os.path.basename(f_lookup_treatments)} : treatment_key ↔ label ↔ QID + degrees + domain_bpsx + domain_fine + flags")
    log(f"  - {os.path.basename(f_pairs_dt)} : raw disorder↔treatment triples (kept) incl. source_pid")
    log(f"  - {os.path.basename(f_pairs_ds)} : raw disorder↔symptom (kept)")
    log(f"  - {os.path.basename(f_pairs_st)} : inferred symptom↔treatment with support counts")
    log(f"  - {os.path.basename(f_prov)} : parameters + summary stats")
    log(f"  - {os.path.basename(f_queries)} : SPARQL used")
    log(f"  - {os.path.basename(f_log)} : human-readable run log")
    log("-" * 88)
    log("[OK] Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted by user (Ctrl+C).", level="WARNING")
        sys.exit(130)

