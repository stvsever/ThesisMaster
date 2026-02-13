#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
APA_ARCHIVE_create_annotated_problem_solution_map.py
(APA_ARCHIVE edition; BIO/PSYCHO/SOCIAL hierarchy + OPTIONAL MHO alignment + VERY verbose logging)

Key fixes vs current run (THIS version)
--------------------------------------------
FIX 1 — Matrix build crash (ValueError: Columns must be same length as key)
    Root cause:
      - duplicate problem column keys were possible (slug collisions / truncation)
      - and Problem.key could change after edges were created (upsert_problem mutated key/norm),
        so edges referenced keys that no longer existed.
      - pandas fails on df[problem_cols] assignment when problem_cols contains duplicates.

    Solution:
      - Problem and Treatment keys are now *stable + collision-proof* using a deterministic hash suffix.
      - Keys NEVER change after creation (upsert no longer mutates key/norm).
      - Matrix is built using a preallocated NumPy array (fast, no fragmentation).

FIX 2 — PerformanceWarning: highly fragmented DataFrame
    Root cause:
      - repeatedly doing df[c] = 0 for hundreds/thousands of columns.

    Solution:
      - allocate the full binary matrix as a NumPy array once, then concat.

Other minor improvements
-----------------------
- Added explicit uniqueness checks and safer mapping logic in matrix build.
- Kept outputs/backward behavior intact except for column keys now being hash-stabilized
  (lookup tables map keys -> labels, so this is usually a net win).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import sys
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin

# -----------------------------
# Dependency checks (graceful)
# -----------------------------
def _require(pkg: str, import_name: Optional[str] = None) -> None:
    try:
        __import__(import_name or pkg)
    except Exception as e:
        raise SystemExit(
            f"Missing dependency: {pkg}\n"
            f"Install with:\n"
            f"  pip install {pkg}\n"
        ) from e


_require("pandas")
_require("numpy")
_require("requests")
_require("bs4", "bs4")

# PDF extraction: prefer pypdf; fallback to pdfminer.six if present
try:
    import pypdf  # noqa: F401
except Exception:
    try:
        import pdfminer  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "Missing PDF extractor. Install one of:\n"
            "  pip install pypdf\n"
            "or\n"
            "  pip install pdfminer.six\n"
        ) from e

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Optional ontology alignment (MHO)
HAS_RDFLIB = True
try:
    import rdflib  # type: ignore
    from rdflib.namespace import RDF, RDFS, OWL  # type: ignore
except Exception:
    HAS_RDFLIB = False


# -----------------------------
# Defaults / paths
# -----------------------------
DEFAULT_OUT_DIR = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/overall_mapping_analyses/"
    "results/evaluate_LLM_performance/annotated_maps/APA_ARCHIVE"
)

# WHO mhGAP v2.0 PDF URLs (mirrors; we keep provenance)
MHGAP_V2_PDF_URLS = [
    "https://www.globalwn.org/files/2018-02/mhGAP%202.0%20English.pdf",
    "https://www.jstor.org/stable/pdf/resrep27913.1.pdf",
]

# WHO mhGAP v1.0 PDF URLs (fallbacks)
MHGAP_V1_PDF_URLS = [
    "https://apps.who.int/iris/bitstream/handle/10665/44406/9789241548069_eng.pdf?sequence=1",
    "https://www.cfpc.ca/uploadedFiles/Directories/Committees_List/WHO%20mhGAP%20intervention%20guide.pdf",
]

# SCP listing pages (under /resource/...) that contain “Read more” links
SCP_LISTING_BASE = "https://societyofclinicalpsychology.org/resource/psychological-treatments-archive/"
SCP_LISTING_PAGE_FMT = "https://societyofclinicalpsychology.org/resource/psychological-treatments-archive/page/{page}/?et_blog="

# SCP actual treatment posts (this is what we want to parse)
SCP_TREATMENT_PREFIX = "https://societyofclinicalpsychology.org/psychological-treatments-archive/"

# GALENOS Mental Health Ontology (MHO) artifact
MHO_OWL_URL = "https://raw.githubusercontent.com/galenos-project/mental-health-ontology/main/gmho_with_imports.owl"

DEFAULT_USER_AGENT = "AnnotatedProblemSolutionMap/4.2 (contact: stijn.vanseveren@gmail.com)"


# -----------------------------
# Logging (very verbose)
# -----------------------------
def _ts() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str, verbose: bool = True, level: str = "INFO") -> None:
    if verbose:
        print(f"[{_ts()}] [{level}] {msg}", flush=True)


def log_kv(title: str, kv: Dict[str, object], verbose: bool = True) -> None:
    if not verbose:
        return
    log(title, verbose=verbose)
    for k, v in kv.items():
        print(f"    - {k}: {v}", flush=True)


# -----------------------------
# Utilities
# -----------------------------
def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


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


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(s: str, n: int = 10) -> str:
    """
    Deterministic short hash for stable, collision-resistant IDs.
    Used to prevent slug collisions and to keep keys stable even if labels vary.
    """
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


_slug_re = re.compile(r"[^A-Za-z0-9]+")


def slugify(s: str, max_len: int = 80) -> str:
    s = (s or "").strip()
    s = _slug_re.sub("_", s)
    s = s.strip("_")
    return (s[:max_len] if s else "UNKNOWN")


def norm_key(s: str) -> str:
    """Normalization for de-duplication (case/punct/space-insensitive)."""
    s = (s or "").strip().lower()
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_label(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize(s: str) -> List[str]:
    s = norm_key(s)
    toks = [t for t in s.split(" ") if t]
    stop = {"and", "or", "the", "of", "to", "a", "an", "in", "for", "with", "by"}
    return [t for t in toks if t not in stop and len(t) >= 3]


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


# -----------------------------
# Requests session (retries)
# -----------------------------
def make_session(user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent})

    try:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry = Retry(
            total=5,
            backoff_factor=0.6,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
    except Exception:
        pass

    return s


# -----------------------------
# BIO/PSYCHO/SOCIAL vocab (editable)
# -----------------------------
BIO_KEYWORDS = {
    "medication", "pharmacotherapy", "drug", "antidepress", "antipsych", "benzodia",
    "ssri", "snri", "lithium", "valproate", "carbamazepine", "haloperidol", "risperidone",
    "olanzapine", "clozapine", "fluoxetine", "sertraline", "paroxetine", "venlafaxine",
    "diazepam", "lorazepam", "methadone", "buprenorphine", "naltrexone",
    "detox", "withdrawal management", "substitution therapy", "opioid agonist",
    "electroconvulsive", "ect", "tms", "brain stimulation", "neuromodulation"
}

PSYCHO_KEYWORDS = {
    "psychotherapy", "therapy", "cbt", "cognitive", "behaviour", "behavior", "exposure",
    "emdr", "dbt", "acceptance", "act", "mindfulness-based", "mbrp", "mbct",
    "interpersonal", "ipt", "psychodynamic", "supportive psychotherapy",
    "motivational interviewing", "mi", "problem solving", "pst",
    "family therapy", "couples therapy", "parent training",
    "trauma-focused", "tf-cbt", "relapse prevention (psychological)",
    "contingency management (behavioral)"
}

SOCIAL_KEYWORDS = {
    "psychoeducation", "social support", "peer support", "support group", "case management",
    "rehabilitation", "supported employment", "housing", "school", "community",
    "family support", "caregiver", "social skills", "vocational", "assertive community treatment",
    "home visit", "community-based", "intervention", "harm reduction",
    "sleep hygiene", "exercise", "diet", "yoga", "relaxation", "stress management",
}

SECONDARY_BUCKETS = {
    "BIO": [
        ("medication", {"medication", "pharmacotherapy", "antidepress", "antipsych", "benzodia", "ssri", "snri", "lithium"}),
        ("substance_use_medication", {"methadone", "buprenorphine", "naltrexone", "opioid"}),
        ("somatic_neuromodulation", {"electroconvulsive", "ect", "tms", "neuromodulation"}),
        ("other_biomedical", set()),
    ],
    "PSYCHO": [
        ("cognitive_behavioral", {"cbt", "cognitive", "behaviour", "behavior", "exposure", "tf-cbt"}),
        ("trauma_focused", {"emdr", "trauma-focused", "tf-cbt"}),
        ("third_wave", {"acceptance", "act", "mindfulness", "mbct"}),
        ("interpersonal", {"interpersonal", "ipt"}),
        ("family_systems", {"family therapy", "parent training", "couples"}),
        ("motivational_problem_solving", {"motivational", "problem solving", "pst"}),
        ("other_psychotherapy", set()),
    ],
    "SOCIAL": [
        ("psychosocial_support", {"psychoeducation", "peer", "support group", "family support", "caregiver"}),
        ("case_management_rehab", {"case management", "rehabilitation", "supported employment", "housing", "community"}),
        ("harm_reduction", {"harm reduction"}),
        ("lifestyle_behavioral", {"sleep", "exercise", "diet", "yoga", "relaxation", "stress management", "sleep hygiene"}),
        ("other_social", set()),
    ],
}


def classify_treatment_type(label: str) -> str:
    l = (label or "").lower()
    if any(k in l for k in ["therapy", "psychotherapy", "counsel", "cbt", "dbt", "emdr", "exposure",
                            "interpersonal", "behavio", "family therapy", "motivational", "problem solving"]):
        return "psychotherapy"
    if any(k in l for k in ["antidepress", "antipsych", "benzodia", "mood stabil", "lithium",
                            "valpro", "carbamaz", "ssri", "snri", "haloper", "risper", "olanzap",
                            "clozap", "fluox", "sertral", "parox", "venlaf", "diazep", "loraz",
                            "methadone", "buprenorphine", "naltrexone"]):
        return "medication"
    if any(k in l for k in ["psychoeducation", "support group", "peer", "family support",
                            "social", "rehabilitation", "case management", "supported employment",
                            "housing", "community", "skills training"]):
        return "psychosocial"
    if any(k in l for k in ["sleep", "exercise", "mindfulness", "diet", "yoga", "relaxation", "stress management", "sleep hygiene"]):
        return "lifestyle"
    if any(k in l for k in ["brief intervention", "harm reduction", "detox", "withdrawal",
                            "opioid substitution"]):
        return "substance_use_intervention"
    return "other"


# -----------------------------
# Downloading / caching
# -----------------------------
def download_file(session: requests.Session, url: str, out_path: str, timeout_s: int = 60, verbose: bool = True) -> bool:
    ensure_dir(os.path.dirname(out_path))
    try:
        log(f"Downloading: {url}", verbose=verbose)
        r = session.get(url, timeout=timeout_s)
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}")
        with open(out_path, "wb") as f:
            f.write(r.content)
        log(f"Saved file: {out_path} ({len(r.content):,} bytes)", verbose=verbose)
        return True
    except Exception as e:
        log(f"Download FAILED for {url} -> {e}", verbose=verbose, level="WARN")
        return False


def download_first_working(session: requests.Session, urls: List[str], out_path: str, sleep_s: float = 0.6, verbose: bool = True) -> Tuple[Optional[str], bool]:
    for u in urls:
        ok = download_file(session, u, out_path, verbose=verbose)
        if ok:
            time.sleep(sleep_s)
            return (u, True)
    return (None, False)


# -----------------------------
# PDF text extraction (page-aware)
# -----------------------------
def extract_pdf_pages_text(pdf_path: str, verbose: bool = True) -> List[str]:
    """
    Returns list of page texts.
    - With pypdf: real page splits
    - With pdfminer fallback: single-item list
    """
    log(f"Extracting PDF text (page-aware): {pdf_path}", verbose=verbose)
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(pdf_path)
        n_pages = len(reader.pages)
        log(f"PDF pages detected (pypdf): {n_pages}", verbose=verbose)
        pages: List[str] = []
        for i, p in enumerate(reader.pages):
            if verbose and (i % 20 == 0):
                log(f"  extracting page {i+1}/{n_pages}", verbose=verbose)
            t = p.extract_text() or ""
            pages.append(t)
        total_chars = sum(len(x) for x in pages)
        log(f"Extracted characters (pypdf): {total_chars:,}", verbose=verbose)
        return pages
    except Exception as e:
        log(f"pypdf extraction failed -> {e} ; trying pdfminer.six (no page splits)", verbose=verbose, level="WARN")

    try:
        from pdfminer.high_level import extract_text  # type: ignore
        txt = extract_text(pdf_path) or ""
        log(f"Extracted characters (pdfminer): {len(txt):,}", verbose=verbose)
        return [txt]
    except Exception as e:
        raise RuntimeError(f"Failed to extract PDF text from: {pdf_path}") from e


def join_pages_with_markers(pages: List[str]) -> str:
    out = []
    for i, t in enumerate(pages, start=1):
        out.append(f"\n\n<<<PAGE {i}>>>\n")
        out.append(t or "")
    return "".join(out)


# -----------------------------
# Data model (FIXED: stable keys)
# -----------------------------
@dataclass
class Problem:
    kind: str  # DX or SYM
    label: str
    sources: Set[str] = field(default_factory=set)
    provenances: Set[str] = field(default_factory=set)

    key: str = field(init=False)
    norm: str = field(init=False)
    _id: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.label = safe_label(self.label)
        self.norm = norm_key(self.label)

        # Stable ID based on kind+norm (never changes)
        self._id = stable_hash(f"{self.kind}|{self.norm}", n=10)

        # Readable but collision-resistant key (never changes)
        # Slug uses norm (stable), plus hashed suffix.
        self.key = f"{self.kind}__{slugify(self.norm, max_len=60)}__{self._id}"


@dataclass
class Treatment:
    label: str
    ttype: str
    sources: Set[str] = field(default_factory=set)
    urls: Set[str] = field(default_factory=set)
    evidence: Set[str] = field(default_factory=set)

    # Enrichment
    primary_domain: str = "UNKNOWN"        # BIO / PSYCHO / SOCIAL
    secondary_domain: str = "UNKNOWN"

    mho_match_iri: str = ""
    mho_match_label: str = ""
    mho_match_score: float = 0.0
    mho_ancestor_labels: str = ""

    key: str = field(init=False)
    norm: str = field(init=False)
    _id: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.label = safe_label(self.label)
        self.norm = norm_key(self.label)
        self._id = stable_hash(f"TRT|{self.norm}", n=10)
        self.key = f"TRT__{slugify(self.norm, max_len=60)}__{self._id}"


def upsert_problem(
    problems_by_norm: Dict[Tuple[str, str], Problem],
    kind: str,
    label: str,
    source: str,
    provenance: str,
) -> Problem:
    """
    FIXED:
      - NEVER mutate Problem.key or Problem.norm after insertion.
      - This prevents edge tables referencing keys that later change.
    """
    tmp = Problem(kind=kind, label=label)
    k = (tmp.kind, tmp.norm)
    if k not in problems_by_norm:
        tmp.sources.add(source)
        tmp.provenances.add(provenance)
        problems_by_norm[k] = tmp
        return tmp

    p = problems_by_norm[k]
    p.sources.add(source)
    p.provenances.add(provenance)

    # You may update the display label, but do NOT change norm/key/id.
    # Only accept label improvement if it is "better" but refers to same normalized form.
    new_lab = safe_label(label)
    if new_lab and len(new_lab) > len(p.label):
        if not re.search(r"\bfollow[- ]?up\b", new_lab, flags=re.IGNORECASE):
            p.label = new_lab

    return p


def upsert_treatment(
    treatments_by_norm: Dict[str, Treatment],
    label: str,
    source: str,
    url: Optional[str] = None,
    evidence: Optional[str] = None,
) -> Treatment:
    """
    Deduplicate treatments by normalized label, while keeping stable keys.
    """
    t_lab = safe_label(label)
    nk = norm_key(t_lab)
    t = treatments_by_norm.get(nk)
    if not t:
        t = Treatment(label=t_lab, ttype=classify_treatment_type(t_lab))
        treatments_by_norm[nk] = t

    t.sources.add(source)
    if url:
        t.urls.add(url)
    if evidence:
        t.evidence.add(evidence)
    return t


# -----------------------------
# MHO loading + lexical alignment (optional)
# -----------------------------
@dataclass
class MhoTerm:
    iri: str
    label: str
    norm: str
    synonyms: List[str] = field(default_factory=list)


def load_mho_terms(owl_path: str, verbose: bool = True) -> Tuple[List[MhoTerm], Dict[str, List[int]], Optional["rdflib.Graph"]]:
    if not HAS_RDFLIB:
        log("rdflib not installed -> MHO alignment will be skipped. Install: pip install rdflib", verbose=verbose, level="WARN")
        return ([], {}, None)

    log(f"Loading MHO OWL into rdflib: {owl_path}", verbose=verbose)
    g = rdflib.Graph()
    try:
        g.parse(owl_path)
    except Exception as e:
        log(f"Failed to parse OWL with rdflib -> {e} ; skipping MHO alignment", verbose=verbose, level="WARN")
        return ([], {}, None)

    log(f"MHO triples loaded: {len(g):,}", verbose=verbose)

    OBOINOWL = rdflib.Namespace("http://www.geneontology.org/formats/oboInOwl#")
    SKOS = rdflib.Namespace("http://www.w3.org/2004/02/skos/core#")

    terms: List[MhoTerm] = []

    classes = set(g.subjects(RDF.type, OWL.Class))
    log(f"MHO owl:Class count: {len(classes):,}", verbose=verbose)

    def _get_literals(s, p) -> List[str]:
        vals = []
        for o in g.objects(s, p):
            if isinstance(o, rdflib.term.Literal):
                vals.append(str(o))
        return vals

    for c in classes:
        labels = _get_literals(c, RDFS.label) + _get_literals(c, SKOS.prefLabel)
        if not labels:
            continue
        label = safe_label(labels[0])
        if not label:
            continue

        syns = []
        syns += _get_literals(c, SKOS.altLabel)
        syns += _get_literals(c, OBOINOWL.hasExactSynonym)
        syns += _get_literals(c, OBOINOWL.hasRelatedSynonym)
        syns = [safe_label(s) for s in syns if safe_label(s)]
        syns = list(dict.fromkeys(syns))

        terms.append(MhoTerm(iri=str(c), label=label, norm=norm_key(label), synonyms=syns))

    log(f"MHO labeled class terms extracted: {len(terms):,}", verbose=verbose)

    token_index: Dict[str, List[int]] = {}
    for idx, t in enumerate(terms):
        toks = set(tokenize(t.label))
        for s in t.synonyms:
            toks |= set(tokenize(s))
        for tok in toks:
            token_index.setdefault(tok, []).append(idx)

    log(f"MHO token index size: {len(token_index):,} tokens", verbose=verbose)
    return (terms, token_index, g)


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def match_to_mho(
    label: str,
    terms: List[MhoTerm],
    token_index: Dict[str, List[int]],
    min_score: float = 0.88,
    max_candidates: int = 700,
) -> Tuple[str, str, float]:
    if not terms:
        return ("", "", 0.0)

    nlabel = norm_key(label)
    if not nlabel:
        return ("", "", 0.0)

    toks = tokenize(label)
    cand_idxs: Set[int] = set()
    for tok in toks:
        cand_idxs |= set(token_index.get(tok, []))
        if len(cand_idxs) > max_candidates:
            break

    if not cand_idxs:
        return ("", "", 0.0)

    for idx in cand_idxs:
        if terms[idx].norm == nlabel:
            return (terms[idx].iri, terms[idx].label, 1.0)

    best = ("", "", 0.0)
    for idx in cand_idxs:
        t = terms[idx]
        s = _ratio(nlabel, t.norm)
        if s > best[2]:
            best = (t.iri, t.label, s)
        for syn in t.synonyms[:8]:
            s2 = _ratio(nlabel, norm_key(syn))
            if s2 > best[2]:
                best = (t.iri, t.label, s2)

    if best[2] >= min_score:
        return best
    return ("", "", 0.0)


def mho_ancestor_labels(
    g: Optional["rdflib.Graph"],
    iri: str,
    max_depth: int = 6,
    max_labels: int = 12,
) -> str:
    if not HAS_RDFLIB or g is None or not iri:
        return ""
    try:
        node = rdflib.URIRef(iri)
    except Exception:
        return ""

    seen: Set[rdflib.term.Node] = set()
    frontier: List[Tuple[rdflib.term.Node, int]] = [(node, 0)]
    labels: List[str] = []

    while frontier:
        cur, d = frontier.pop(0)
        if cur in seen or d > max_depth:
            continue
        seen.add(cur)

        for o in g.objects(cur, RDFS.label):
            if isinstance(o, rdflib.term.Literal):
                lab = safe_label(str(o))
                if lab and lab not in labels:
                    labels.append(lab)
                    if len(labels) >= max_labels:
                        return " | ".join(labels)

        for parent in g.objects(cur, RDFS.subClassOf):
            if parent and parent not in seen:
                frontier.append((parent, d + 1))

    return " | ".join(labels)


# -----------------------------
# BIO/PSYCHO/SOCIAL assignment
# -----------------------------
def infer_bps_primary_secondary(
    treatment_label: str,
    treatment_type: str,
    mho_label: str = "",
    ancestor_labels: str = ""
) -> Tuple[str, str, str]:
    text = " ".join([treatment_label or "", mho_label or "", ancestor_labels or ""]).lower()

    if treatment_type == "medication":
        primary = "BIO"
    elif treatment_type == "psychotherapy":
        primary = "PSYCHO"
    elif treatment_type in ("psychosocial", "lifestyle"):
        primary = "SOCIAL"
    elif treatment_type == "substance_use_intervention":
        if any(k in text for k in ["methadone", "buprenorphine", "naltrexone", "detox", "withdrawal"]):
            primary = "BIO"
        elif any(k in text for k in ["motivational", "cbt", "therapy", "interviewing"]):
            primary = "PSYCHO"
        else:
            primary = "SOCIAL"
    else:
        primary = "UNKNOWN"

    bio_hit = any(k in text for k in BIO_KEYWORDS)
    psycho_hit = any(k in text for k in PSYCHO_KEYWORDS)
    social_hit = any(k in text for k in SOCIAL_KEYWORDS)

    if primary == "UNKNOWN":
        scores = {
            "BIO": 1 if bio_hit else 0,
            "PSYCHO": 1 if psycho_hit else 0,
            "SOCIAL": 1 if social_hit else 0,
        }
        primary = max(scores, key=scores.get)
        if scores[primary] == 0:
            primary = "SOCIAL"

    rationale = []
    if bio_hit:
        rationale.append("bio_kw")
    if psycho_hit:
        rationale.append("psycho_kw")
    if social_hit:
        rationale.append("social_kw")
    if mho_label:
        rationale.append("mho_match")
    rationale_short = ",".join(rationale) if rationale else "type_only"

    sec = "UNKNOWN"
    for name, keys in SECONDARY_BUCKETS.get(primary, []):
        if not keys:
            sec = name
            break
        if any(k in text for k in keys):
            sec = name
            break

    return primary, sec, rationale_short


# -----------------------------
# SCP scraping (collect real treatment pages)
# -----------------------------
def save_html_snapshot(html: str, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def is_scp_treatment_post_url(href: str) -> bool:
    if not href:
        return False
    if not href.startswith("http"):
        return False
    if not href.startswith(SCP_TREATMENT_PREFIX):
        return False
    if href.rstrip("/") == SCP_TREATMENT_PREFIX.rstrip("/"):
        return False
    if "/page/" in href:
        return False
    if "filter" in href:
        return False
    if "category" in href:
        return False
    return True


def scp_collect_treatment_urls(
    session: requests.Session,
    max_pages: int,
    snapshots_dir: str,
    sleep_s: float,
    verbose: bool = True
) -> List[str]:
    urls: List[str] = []
    seen: Set[str] = set()

    log(f"SCP: collecting *treatment post* URLs from listing pages (max_pages={max_pages})", verbose=verbose)
    consecutive_no_new = 0

    for page in range(1, max_pages + 1):
        page_url = SCP_LISTING_BASE if page == 1 else SCP_LISTING_PAGE_FMT.format(page=page)
        log(f"SCP: fetching listing page {page}: {page_url}", verbose=verbose)

        try:
            r = session.get(page_url, timeout=45)
            if r.status_code >= 400:
                log(f"SCP: stopping pagination at page {page} (HTTP {r.status_code})", verbose=verbose, level="WARN")
                break

            snap_path = os.path.join(snapshots_dir, f"scp_listing_page_{page:03d}.html")
            save_html_snapshot(r.text, snap_path)

            soup = BeautifulSoup(r.text, "html.parser")

            before = len(urls)
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                href = urljoin(page_url, href)
                if is_scp_treatment_post_url(href) and href not in seen:
                    seen.add(href)
                    urls.append(href)

            added = len(urls) - before
            log(f"SCP: +{added} treatment urls (total={len(urls)}) from listing page {page}", verbose=verbose)

            if added == 0:
                consecutive_no_new += 1
            else:
                consecutive_no_new = 0

            if consecutive_no_new >= 2 and page >= 3:
                log("SCP: stopping pagination (2 consecutive pages with 0 new treatment URLs)", verbose=verbose)
                break

            time.sleep(max(0.2, sleep_s))
        except Exception as e:
            log(f"SCP: listing page fetch failed (page={page}) -> {e}", verbose=verbose, level="WARN")
            if page >= 3 and len(urls) > 0:
                break

    return urls


def _split_dx_items(text: str) -> List[str]:
    t = safe_label(text)
    if not t:
        return []
    t = t.replace("\u2022", "\n").replace("•", "\n")
    t = re.sub(r"(?i)^\s*related diagnosis\s*[:\-]?\s*", "", t).strip()
    parts = re.split(r"\s*(?:\n+|;|/|,|\band\b|\bor\b)\s*", t, flags=re.IGNORECASE)
    out: List[str] = []
    for p in parts:
        p = safe_label(p)
        if not p or len(p) < 3:
            continue
        if re.search(r"(?i)\b(all treatments|follow us|menu|search|copyright)\b", p):
            continue
        p = p.strip(" .;:-—–")
        if len(p) < 3:
            continue
        out.append(p)
    seen: Set[str] = set()
    cleaned: List[str] = []
    for x in out:
        nk = norm_key(x)
        if nk and nk not in seen:
            seen.add(nk)
            cleaned.append(x)
    return cleaned


def _scp_find_heading_node(soup: BeautifulSoup, heading_text: str) -> Optional[object]:
    target = norm_key(heading_text)
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "strong", "b", "p", "div", "span"]):
        txt = safe_label(tag.get_text(" ", strip=True))
        if not txt:
            continue
        txt_norm = norm_key(txt).replace(" :", ":")
        txt_norm2 = re.sub(r"\s*:\s*$", "", txt_norm).strip()
        if txt_norm2 == target:
            return tag
        if txt_norm2.startswith(target) and len(txt_norm2) <= len(target) + 8:
            return tag
    return None


def _scp_extract_section_text_after_heading(heading_node: object, max_items: int = 40) -> str:
    collected: List[str] = []
    for sib in getattr(heading_node, "next_siblings", []):
        if sib is None:
            continue
        if isinstance(sib, str):
            if not sib.strip():
                continue
            continue
        if not hasattr(sib, "get_text"):
            continue

        txt = safe_label(sib.get_text(" ", strip=True))
        if not txt:
            continue

        txt_l = txt.lower()
        if re.search(r"(?i)\b(status|est status|treatment|all treatments|references|overview|description)\b\s*:?\s*$", txt):
            break
        if re.search(r"(?i)^(status|est status)\b", txt_l):
            break
        if re.search(r"(?i)\bfollow us\b|\bsearch\b|\bcopyright\b", txt_l):
            break
        if getattr(sib, "name", "").lower() in {"h1", "h2", "h3", "h4"}:
            break

        if getattr(sib, "name", "").lower() in {"ul", "ol"}:
            for li in sib.find_all("li"):
                t = safe_label(li.get_text(" ", strip=True))
                if t:
                    collected.append(t)
                    if len(collected) >= max_items:
                        return "\n".join(collected)
        else:
            collected.append(txt)

        if len(collected) >= max_items:
            break

    return "\n".join(collected).strip()


def _scp_extract_status(soup: BeautifulSoup) -> str:
    txt = soup.get_text("\n")
    m = re.search(r"(?mi)^\s*Status\s*:\s*(.+?)\s*$", txt)
    if m:
        return safe_label(m.group(1))

    for tag in soup.find_all(["p", "div", "span", "strong", "b"]):
        t = safe_label(tag.get_text(" ", strip=True))
        if t and t.lower().startswith("status:"):
            return safe_label(t.split(":", 1)[1])
    return ""


def scp_parse_treatment_post(
    session: requests.Session,
    url: str,
    snapshots_dir: str,
    verbose: bool = True
) -> Tuple[Optional[str], List[str], Optional[str]]:
    r = session.get(url, timeout=60)
    r.raise_for_status()

    safe_fn = slugify(url, max_len=140)
    snap_path = os.path.join(snapshots_dir, f"scp_post__{safe_fn}.html")
    save_html_snapshot(r.text, snap_path)

    soup = BeautifulSoup(r.text, "html.parser")

    title = None
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)
    if not title and soup.title:
        title = soup.title.get_text(strip=True)

    if not title:
        return (None, [], None)

    evidence = _scp_extract_status(soup) or None

    related_dx: List[str] = []
    head = _scp_find_heading_node(soup, "Related Diagnosis")
    if head is not None:
        block_txt = _scp_extract_section_text_after_heading(head)
        related_dx = _split_dx_items(block_txt)
    else:
        flat = soup.get_text("\n")
        m = re.search(
            r"(?si)Related Diagnosis\s*(?:\n|:)\s*(.+?)(?:\n\s*(?:Status|EST Status|All Treatments|References|Overview)\b|\Z)",
            flat
        )
        if m:
            related_dx = _split_dx_items(m.group(1))

    related_dx = sorted(set(related_dx), key=lambda x: x.lower())

    if verbose:
        log(f"SCP: parsed treatment='{title}' | related_dx={len(related_dx)} | status='{evidence or ''}'", verbose=verbose)

    return (title, related_dx, evidence)


# -----------------------------
# mhGAP parsing (WHO PDF)
# -----------------------------
MHGAP_CODES_BASE = ["DEP", "PSY", "EPI", "CMH", "DEM", "SUB", "SUI", "OTH"]
MHGAP_CODE_TO_DX_BASE = {
    "DEP": "Depression",
    "PSY": "Psychosis",
    "EPI": "Epilepsy",
    "CMH": "Child and adolescent mental and behavioural disorders",
    "DEM": "Dementia",
    "SUB": "Disorders due to substance use",
    "SUI": "Self-harm / Suicide",
    "OTH": "Other significant mental health complaints",
}

RX_MHGAP_CODE_LINE = re.compile(r"^\s*([A-Z]{2,4})\b\s*(.*)\s*$", re.IGNORECASE)

POSITIVE_VERBS = ["consider", "offer", "provide", "use", "recommend", "start", "initiate", "prescribe", "administer"]
NEGATIVE_CUES = ["do not", "don't", "avoid", "contraindicated", "not recommended"]

TREATMENT_HINTS = [
    "therapy", "psychotherapy", "counselling", "counseling", "psychoeducation",
    "antidepress", "antipsych", "benzodia", "mood stabil", "lithium",
    "withdrawal", "detox", "substitution", "methadone", "buprenorphine", "naltrexone",
    "brief intervention", "motivational", "problem solving", "interpersonal",
    "family", "support", "rehabilitation", "case management",
    "sleep hygiene", "relaxation", "stress management",
]

INTERVENTION_VERBS = [
    "teach", "address", "offer", "provide", "recommend", "refer", "encourage", "advise",
    "support", "help", "assist", "arrange", "monitor", "follow", "schedule", "assess",
    "initiate", "start", "use", "consider", "deliver", "ensure",
]


def _text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, norm_key(a), norm_key(b)).ratio()


def mhgap_detect_module_start_pages(
    pages: List[str],
    code_to_title: Dict[str, str],
    verbose: bool = True
) -> Dict[str, int]:
    best: Dict[str, Tuple[float, int, str]] = {}

    for page_idx, ptxt in enumerate(pages):
        raw_lines = (ptxt or "").splitlines()

        lines: List[str] = []
        for ln in raw_lines:
            s = ln.strip()
            if not s:
                continue
            lines.append(s)
            if len(lines) >= 60:
                break

        for li, ln in enumerate(lines):
            m = RX_MHGAP_CODE_LINE.match(ln)
            if not m:
                continue
            code = (m.group(1) or "").upper()
            rest = safe_label(m.group(2) or "")
            if code not in code_to_title:
                continue

            if "»" in ln or rest.startswith("»"):
                continue

            rest2 = re.sub(r"^\d+(\.\d+)*\s*", "", rest).strip(" -–—:").strip()
            cand = rest2 if rest2 else code_to_title[code]

            base = _text_similarity(cand, code_to_title[code])
            bonus = 0.10 if li <= 10 else (0.05 if li <= 25 else 0.0)
            penalty = 0.0
            if "follow" in cand.lower():
                penalty += 0.20
            if cand.strip().isdigit():
                penalty += 0.30
            if len(cand) <= 3:
                penalty += 0.20

            score = base + bonus - penalty

            prev = best.get(code)
            if prev is None or score > prev[0]:
                best[code] = (score, page_idx, cand)

    out: Dict[str, int] = {}
    for code in sorted(code_to_title.keys()):
        if code in best:
            out[code] = best[code][1]

    if verbose:
        log("mhGAP: detected module start pages (best guesses):", verbose=verbose)
        for code in sorted(code_to_title.keys()):
            if code in best:
                sc, pg, cand = best[code]
                log(f"  - {code}: page={pg+1} (score={sc:.3f}) cand='{cand}' canonical='{code_to_title[code]}'", verbose=verbose)
            else:
                log(f"  - {code}: NOT FOUND (will fall back later)", verbose=verbose, level="WARN")

    return out


def mhgap_extract_treatments_from_segment(seg_lines: List[str]) -> List[str]:
    out: List[str] = []
    joined = "\n".join(seg_lines)
    sentences = re.split(r"(?<=[\.\?\!])\s+", joined.replace("\n", " ").strip())

    for s in sentences:
        s_l = s.lower()
        if not s or len(s) < 20:
            continue
        if any(nc in s_l for nc in NEGATIVE_CUES):
            continue
        if not any(h in s_l for h in TREATMENT_HINTS):
            continue
        if not any(v in s_l for v in POSITIVE_VERBS):
            continue

        for v in POSITIVE_VERBS:
            idx = s_l.find(v)
            if idx >= 0:
                phrase = s[idx + len(v):].strip(" :;-")
                phrase = phrase.split(".")[0].strip()
                if phrase:
                    parts = re.split(r"\s*(?:,|;|\bor\b|\band\b)\s*", phrase)
                    for p in parts:
                        p = safe_label(p)
                        if len(p) < 4:
                            continue
                        if any(x in p.lower() for x in ["the person", "a person", "people", "clients", "patients"]):
                            continue
                        p = re.sub(r"^(an?|the)\s+", "", p, flags=re.IGNORECASE).strip()
                        if any(h in p.lower() for h in TREATMENT_HINTS):
                            out.append(p)
                break

    for ln in seg_lines:
        l = (ln or "").strip()
        if len(l) < 6 or len(l) > 190:
            continue
        l_l = l.lower()
        if any(nc in l_l for nc in NEGATIVE_CUES):
            continue
        if not any(h in l_l for h in TREATMENT_HINTS):
            continue
        if l.startswith(("•", "‣", "»", "-", "–", "*", "·")):
            l = l.lstrip("•‣»-*–· ").strip()
            l = re.sub(r"\s+»\s*[A-Z]{2,4}\s*\d+(\.\d+)?\s*$", "", l)
            out.append(safe_label(l))

    cleaned: List[str] = []
    seen: Set[str] = set()
    for c in out:
        c = safe_label(c)
        if not c:
            continue
        c = re.sub(r"\([^)]{0,80}\)", "", c).strip()
        c = re.sub(r"^(consider|offer|provide|use|recommend|start|initiate|prescribe)\s+", "", c, flags=re.IGNORECASE).strip()
        if len(c) < 4:
            continue
        nk = norm_key(c)
        if nk in seen:
            continue
        seen.add(nk)
        cleaned.append(c)

    return cleaned


SYM_SECTION_TRIGGERS = [
    r"signs and symptoms",
    r"symptoms and signs",
    r"presenting complaints",
    r"common presentations",
    r"key symptoms",
    r"clinical features",
    r"core symptoms",
    r"typical symptoms",
    r"common symptoms",
]
SYM_INLINE_TRIGGERS = [
    r"\bassess for\b",
    r"\bask about\b",
    r"\blook for\b",
    r"\bsymptoms include\b",
    r"\binclude(s)?\b\s*:",
    r"\binclude(s)?\b",
]
RX_HEADINGISH = re.compile(r"^\s*(\d+(\.\d+)*)?\s*[A-Z][A-Z \-&/]{4,}\s*$")


def _looks_like_intervention(line: str) -> bool:
    l = (line or "").strip().lower()
    if not l or len(l) < 4:
        return True
    if any((v + " ") in l for v in INTERVENTION_VERBS):
        return True
    if any(h in l for h in TREATMENT_HINTS):
        return True
    if re.search(r"(?i)\b(recommend|offer|provide|refer|counsel|treat|prescribe)\b", l):
        return True
    return False


def _normalize_lines_join_wrapped_bullets(lines: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    n = len(lines)

    def is_bullet(s: str) -> bool:
        s = (s or "").lstrip()
        return s.startswith(("•", "-", "–", "*", "·"))

    def is_headingish(s: str) -> bool:
        s2 = (s or "").strip()
        if not s2:
            return False
        if RX_HEADINGISH.match(s2):
            return True
        if len(s2) <= 60 and re.match(r"^\d+(\.\d+)*\s+[A-Z][A-Za-z].+$", s2):
            return True
        if len(s2) <= 60 and re.match(r"^[A-Z][A-Za-z].+$", s2) and s2.upper() == s2:
            return True
        return False

    while i < n:
        cur = (lines[i] or "").rstrip()
        if not cur.strip():
            i += 1
            continue

        if is_bullet(cur):
            merged = cur.strip()
            j = i + 1
            while j < n and j <= i + 3:
                nxt = (lines[j] or "").strip()
                if not nxt:
                    j += 1
                    continue
                if is_bullet(nxt) or is_headingish(nxt):
                    break
                merged += " " + nxt
                j += 1
            out.append(merged)
            i = j
            continue

        out.append(cur.strip())
        i += 1

    return out


def mhgap_extract_symptoms_from_segment(seg_lines: List[str], verbose: bool = False) -> List[str]:
    seg_lines = _normalize_lines_join_wrapped_bullets(seg_lines)
    out: List[str] = []
    n = len(seg_lines)

    def collect_bullets(start_i: int, max_ahead: int = 140) -> None:
        end_i = min(n, start_i + max_ahead)
        started = False
        for j in range(start_i, end_i):
            ln = (seg_lines[j] or "").strip()
            if not ln:
                continue

            if started and (RX_HEADINGISH.match(ln) or ln.strip().startswith(("###", "##"))):
                break

            if ln.startswith(("•", "-", "–", "*", "·")):
                started = True
                item = ln.lstrip("•-–*· ").strip()
                if not item or len(item) > 240:
                    continue
                if item.endswith("?"):
                    continue
                parts = re.split(r"\s*;\s*|\s*,\s*", item)
                for p in parts:
                    p = safe_label(p)
                    if len(p) < 4:
                        continue
                    if _looks_like_intervention(p):
                        continue
                    out.append(p)

    def collect_inline_includes(line: str) -> None:
        l = safe_label(line)
        if not l or len(l) < 8:
            return
        m = re.search(r"(?i)\b(symptoms|signs)\s+include\s*:\s*(.+)$", l)
        if not m:
            m = re.search(r"(?i)\binclude\s*:\s*(.+)$", l)
        if not m:
            return
        tail = m.group(2) if len(m.groups()) >= 2 else m.group(1)
        tail = safe_label(tail)
        if not tail:
            return
        parts = re.split(r"\s*(?:;|,|\band\b|\bor\b)\s*", tail, flags=re.IGNORECASE)
        for p in parts:
            p = safe_label(p)
            if len(p) < 4:
                continue
            if _looks_like_intervention(p):
                continue
            if len(p.split()) == 1 and len(p) <= 3:
                continue
            out.append(p)

    for i, ln in enumerate(seg_lines):
        l = (ln or "").strip().lower()
        if any(re.search(pat, l) for pat in SYM_SECTION_TRIGGERS):
            collect_bullets(i + 1, max_ahead=180)

    for i, ln in enumerate(seg_lines):
        l = (ln or "").strip().lower()
        if any(re.search(pat, l) for pat in SYM_INLINE_TRIGGERS):
            collect_bullets(i + 1, max_ahead=70)
            collect_inline_includes(ln)

    for ln in seg_lines:
        if "include" in (ln or "").lower() and ":" in (ln or ""):
            collect_inline_includes(ln)

    seen: Set[str] = set()
    cleaned: List[str] = []
    for s in out:
        nk = norm_key(s)
        if nk and nk not in seen and len(s) >= 4:
            seen.add(nk)
            cleaned.append(s)

    if verbose:
        log(f"mhGAP symptom extraction: raw={len(out)} dedup={len(cleaned)}", verbose=True)

    return cleaned


DX_KEYWORDS = {
    "disorder", "disorders", "depression", "depressive", "psychosis", "psychotic",
    "epilepsy", "dementia", "suicide", "self-harm", "substance", "alcohol", "drug",
    "anxiety", "panic", "phobia", "post-traumatic", "ptsd", "stress", "grief",
    "behavior", "behaviour", "conduct", "adhd", "autism", "insomnia", "sleep",
    "somatic", "somatoform", "complaint", "complaints",
}

DX_HEADING_BLOCKLIST = re.compile(
    r"(?i)\b(assessment|management|follow[- ]?up|algorithm|table|chart|general principles|principles|implementation|notes|references|annex)\b"
)

RX_TITLEISH = re.compile(r"^\s*(?:\d+(\.\d+)*\s+)?[A-Z][A-Za-z0-9 ,/&\-\(\)]{2,90}\s*$")


def _is_subcondition_heading(line: str, mho_terms: List[MhoTerm], mho_token_index: Dict[str, List[int]],
                             use_mho_validation: bool, mho_min_score: float) -> Tuple[bool, str, float]:
    s = safe_label(line)
    if not s or len(s) < 4 or len(s) > 90:
        return (False, "", 0.0)
    if DX_HEADING_BLOCKLIST.search(s):
        return (False, "", 0.0)
    if not RX_TITLEISH.match(s):
        return (False, "", 0.0)

    s_l = s.lower()
    if not any(k in s_l for k in DX_KEYWORDS):
        return (False, "", 0.0)

    if not use_mho_validation or not mho_terms:
        s2 = re.sub(r"^\d+(\.\d+)*\s*", "", s).strip(" -–—:")
        return (True, s2, 0.0)

    iri, lab, score = match_to_mho(s, mho_terms, mho_token_index, min_score=mho_min_score, max_candidates=700)
    if score >= mho_min_score and lab:
        return (True, lab, float(score))
    return (False, "", float(score))


def mhgap_split_subcondition_sections(
    seg_lines: List[str],
    mho_terms: List[MhoTerm],
    mho_token_index: Dict[str, List[int]],
    enable: bool,
    use_mho_validation: bool,
    mho_min_score: float,
    max_sections: int,
    min_lines_per_section: int,
) -> List[Tuple[str, List[str], str]]:
    if not enable:
        return []

    lines = _normalize_lines_join_wrapped_bullets(seg_lines)
    idxs: List[Tuple[int, str, float]] = []

    for i, ln in enumerate(lines):
        ok, dx_lab, sc = _is_subcondition_heading(
            ln, mho_terms, mho_token_index, use_mho_validation=use_mho_validation, mho_min_score=mho_min_score
        )
        if ok and dx_lab:
            idxs.append((i, dx_lab, sc))

    filtered: List[Tuple[int, str, float]] = []
    seen_norm: Set[str] = set()
    for i, lab, sc in idxs:
        nk = norm_key(lab)
        if nk in seen_norm:
            continue
        seen_norm.add(nk)
        filtered.append((i, lab, sc))

    filtered = sorted(filtered, key=lambda x: x[0])[:max_sections]
    if not filtered:
        return []

    sections: List[Tuple[str, List[str], str]] = []
    for si, (start_i, lab, sc) in enumerate(filtered):
        end_i = (filtered[si + 1][0] - 1) if (si + 1 < len(filtered)) else (len(lines) - 1)
        chunk = lines[start_i:end_i + 1]
        if len([x for x in chunk if x.strip()]) < min_lines_per_section:
            continue
        prov = f"subcondition_heading='{lab}';mho_score={sc:.3f}" if sc > 0 else f"subcondition_heading='{lab}'"
        sections.append((lab, chunk, prov))

    return sections


# -----------------------------
# Build matrix (FIXED + FAST)
# -----------------------------
def build_binary_matrix(
    treatments: List[Treatment],
    problems: List[Problem],
    dx_trt_edges: List[Dict],
    sym_trt_edges: List[Dict],
) -> pd.DataFrame:
    """
    FIXED:
      - Avoid DataFrame fragmentation by preallocating a NumPy matrix.
      - Avoid pandas duplicate-key assignment failure by guaranteeing unique stable keys.
      - Never do df[problem_cols] = df[problem_cols].astype(int) (not needed; matrix is uint8).
    """
    # Ensure unique keys (should always hold with stable_hash design)
    problem_cols = [p.key for p in problems]
    if len(problem_cols) != len(set(problem_cols)):
        # This should never happen now; raise with diagnostics if it does.
        from collections import Counter
        dup = [k for k, v in Counter(problem_cols).items() if v > 1][:20]
        raise ValueError(f"Duplicate problem keys detected (first 20): {dup}")

    rows = []
    for t in treatments:
        rows.append(
            {
                "treatment_key": t.key,
                "treatment_label": t.label,
                "treatment_type": t.ttype,
                "primary_domain": t.primary_domain,
                "secondary_domain": t.secondary_domain,
                "sources": "|".join(sorted(t.sources)),
                "evidence": "|".join(sorted(t.evidence)),
                "urls": "|".join(sorted(t.urls)),
                "mho_match_label": t.mho_match_label,
                "mho_match_iri": t.mho_match_iri,
                "mho_match_score": float(t.mho_match_score),
                "mho_ancestor_labels": t.mho_ancestor_labels,
            }
        )

    df_meta = pd.DataFrame(rows).drop_duplicates(subset=["treatment_key"]).reset_index(drop=True)

    n_rows = df_meta.shape[0]
    n_cols = len(problem_cols)

    # Preallocate dense binary matrix
    mat = np.zeros((n_rows, n_cols), dtype=np.uint8)

    tkey_to_row = {k: i for i, k in enumerate(df_meta["treatment_key"].tolist())}
    pkey_to_col = {k: j for j, k in enumerate(problem_cols)}

    def _apply(edges: List[Dict]) -> None:
        for e in edges:
            tkey = e.get("treatment_key", "")
            pkey = e.get("problem_key", "")
            i = tkey_to_row.get(tkey)
            j = pkey_to_col.get(pkey)
            if i is not None and j is not None:
                mat[i, j] = 1

    _apply(dx_trt_edges)
    _apply(sym_trt_edges)

    df_bin = pd.DataFrame(mat, columns=problem_cols)
    df = pd.concat([df_meta, df_bin], axis=1)
    return df


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    ap.add_argument("--user-agent", type=str, default=DEFAULT_USER_AGENT)

    ap.add_argument("--include-symptoms", type=int, default=1)
    ap.add_argument("--min-symptom-support", type=int, default=2)
    ap.add_argument("--mhgap-prefer-v2", type=int, default=1)

    ap.add_argument("--enable-mhgap-subconditions", type=int, default=1)
    ap.add_argument("--mhgap-subcondition-max-per-module", type=int, default=40)
    ap.add_argument("--mhgap-subcondition-min-lines", type=int, default=25)
    ap.add_argument("--mho-validate-subconditions", type=int, default=1)
    ap.add_argument("--mhgap-subcondition-mho-min-score", type=float, default=0.92)

    ap.add_argument("--max-scp-pages", type=int, default=25)
    ap.add_argument("--max-scp-posts", type=int, default=900)
    ap.add_argument("--sleep-s", type=float, default=0.5)

    ap.add_argument("--verbose", type=int, default=1)

    ap.add_argument("--mho-min-score", type=float, default=0.88)
    ap.add_argument("--mho-max-candidates", type=int, default=700)

    args = ap.parse_args()
    verbose = bool(int(args.verbose))

    out_dir = os.path.abspath(args.out_dir)
    ensure_dir(out_dir)

    sources_dir = os.path.join(out_dir, "sources")
    ensure_dir(sources_dir)

    snapshots_dir = os.path.join(sources_dir, "snapshots_scp")
    ensure_dir(snapshots_dir)

    session = make_session(args.user_agent)

    # Output files
    f_matrix = os.path.join(out_dir, "01_create_annotated_problem_solution_map.csv")
    f_problems = os.path.join(out_dir, "problems_lookup.tsv")
    f_treatments = os.path.join(out_dir, "treatments_lookup.tsv")
    f_dx_trt = os.path.join(out_dir, "raw_pairs_dx_treatment.tsv")
    f_dx_sym = os.path.join(out_dir, "raw_pairs_dx_symptom.tsv")
    f_sym_trt = os.path.join(out_dir, "inferred_pairs_symptom_treatment.tsv")
    f_prov = os.path.join(out_dir, "provenance.json")
    f_log = os.path.join(out_dir, "run_log.txt")

    log("=" * 110, verbose=verbose)
    log("RUN START: Annotated Problem→Solution Map (APA_ARCHIVE) [DOM SCP dx + mhGAP subconditions + improved symptoms]", verbose=verbose)
    log_kv("CONFIG", {
        "out_dir": out_dir,
        "include_symptoms": int(args.include_symptoms),
        "min_symptom_support": int(args.min_symptom_support),
        "mhgap_prefer_v2": int(args.mhgap_prefer_v2),
        "enable_mhgap_subconditions": int(args.enable_mhgap_subconditions),
        "mhgap_subcondition_max_per_module": int(args.mhgap_subcondition_max_per_module),
        "mhgap_subcondition_min_lines": int(args.mhgap_subcondition_min_lines),
        "mho_validate_subconditions": int(args.mho_validate_subconditions),
        "mhgap_subcondition_mho_min_score": float(args.mhgap_subcondition_mho_min_score),
        "max_scp_pages": int(args.max_scp_pages),
        "max_scp_posts": int(args.max_scp_posts),
        "sleep_s": float(args.sleep_s),
        "HAS_RDFLIB (MHO alignment)": HAS_RDFLIB,
        "mho_min_score": float(args.mho_min_score),
        "mho_max_candidates": int(args.mho_max_candidates),
        "SCP_LISTING_BASE": SCP_LISTING_BASE,
        "SCP_TREATMENT_PREFIX": SCP_TREATMENT_PREFIX,
    }, verbose=verbose)

    # ---------------------------
    # 0) Download MHO ontology artifact
    # ---------------------------
    mho_path = os.path.join(sources_dir, "gmho_with_imports.owl")
    if not os.path.exists(mho_path):
        log("Step 0: Downloading MHO OWL...", verbose=verbose)
        ok = download_file(session, MHO_OWL_URL, mho_path, verbose=verbose)
        if not ok:
            log("MHO download failed -> continuing without MHO alignment (still produces matrix).", verbose=verbose, level="WARN")
    else:
        log(f"Step 0: MHO OWL already present (cache): {mho_path}", verbose=verbose)

    log("Step 0b: Building MHO lexical index (optional)...", verbose=verbose)
    mho_terms, mho_token_index, mho_graph = load_mho_terms(mho_path, verbose=verbose) if os.path.exists(mho_path) else ([], {}, None)

    # ---------------------------
    # 1) Download mhGAP PDF
    # ---------------------------
    mhgap_pdf_path = os.path.join(sources_dir, "mhgap_intervention_guide.pdf")
    mhgap_url_used: Optional[str] = None

    if not os.path.exists(mhgap_pdf_path):
        log("Step 1: Downloading mhGAP PDF (prefer v2; fallback v1)...", verbose=verbose)
        urls_try = (MHGAP_V2_PDF_URLS if int(args.mhgap_prefer_v2) == 1 else []) + MHGAP_V1_PDF_URLS
        mhgap_url_used, ok = download_first_working(session, urls_try, mhgap_pdf_path, sleep_s=args.sleep_s, verbose=verbose)
        if not ok:
            raise SystemExit(
                "Failed to download mhGAP PDF from all configured URLs.\n"
                f"Place a local PDF at:\n  {mhgap_pdf_path}\n"
                "and re-run."
            )
    else:
        mhgap_url_used = "local_cache"
        log(f"Step 1: mhGAP PDF already present (cache): {mhgap_pdf_path}", verbose=verbose)

    # Extract text (page-aware)
    log("Step 1b: Extracting mhGAP text (page-aware)...", verbose=verbose)
    mhgap_pages = extract_pdf_pages_text(mhgap_pdf_path, verbose=verbose)
    mhgap_text = join_pages_with_markers(mhgap_pages)
    mhgap_text_path = os.path.join(sources_dir, "mhgap_extracted_text.txt")
    write_text(mhgap_text_path, mhgap_text[:2_000_000])
    log(f"mhGAP extracted text saved: {mhgap_text_path}", verbose=verbose)

    # ---------------------------
    # 2) Parse mhGAP into DX->TRT and DX->SYM
    # ---------------------------
    log("Step 2: Parsing mhGAP into edges (DX->TRT, optional DX->SYM)...", verbose=verbose)

    problems_by_norm: Dict[Tuple[str, str], Problem] = {}
    treatments_by_norm: Dict[str, Treatment] = {}

    dx_trt_edges: List[Dict] = []
    dx_sym_edges: List[Dict] = []

    # Use canonical module mapping only (reduces spurious “FOR/IN/OF...” pseudo-modules)
    code_to_title: Dict[str, str] = dict(MHGAP_CODE_TO_DX_BASE)

    starts = mhgap_detect_module_start_pages(mhgap_pages, code_to_title, verbose=verbose)

    if len(starts) < len(code_to_title):
        log("mhGAP: some module starts missing -> applying fallback scan to fill gaps", verbose=verbose, level="WARN")
        for code in list(code_to_title.keys()):
            if code in starts:
                continue
            for pi, ptxt in enumerate(mhgap_pages):
                if re.search(rf"(?m)^\s*{re.escape(code)}\b", ptxt or "", flags=re.IGNORECASE):
                    starts[code] = pi
                    log(f"  fallback start: {code} -> page {pi+1}", verbose=verbose)
                    break

    module_order = sorted([(code, starts.get(code, 10**9)) for code in code_to_title.keys()], key=lambda x: x[1])
    log(f"mhGAP module order by detected start page: {module_order}", verbose=verbose)

    for idx, (code, start_page) in enumerate(module_order):
        if start_page >= 10**8:
            continue

        end_page = (module_order[idx + 1][1] - 1) if (idx + 1 < len(module_order) and module_order[idx + 1][1] < 10**8) else (len(mhgap_pages) - 1)
        if end_page < start_page:
            end_page = start_page

        seg_text = "\n".join(mhgap_pages[start_page:end_page + 1])
        seg_lines = seg_text.splitlines()

        module_dx_label = code_to_title.get(code, code)
        module_dx = upsert_problem(
            problems_by_norm,
            kind="DX",
            label=module_dx_label,
            source="WHO_mhGAP_IG",
            provenance=f"module_code={code};pages={start_page+1}-{end_page+1};pdf={os.path.basename(mhgap_pdf_path)}"
        )

        log(f"mhGAP module {code} '{module_dx_label}': pages {start_page+1}-{end_page+1} | seg_lines={len(seg_lines):,}", verbose=verbose)

        # module-level treatment extraction
        trt_cands = mhgap_extract_treatments_from_segment(seg_lines)
        log(f"mhGAP module {code} '{module_dx_label}': extracted treatments={len(trt_cands)}", verbose=verbose)

        for t_lab in trt_cands:
            t = upsert_treatment(
                treatments_by_norm,
                label=t_lab,
                source="WHO_mhGAP_IG",
                url=mhgap_url_used,
                evidence="Guideline_recommendation_text_mined",
            )

            dx_trt_edges.append(
                {
                    "problem_kind": "DX",
                    "problem_label": module_dx.label,
                    "problem_key": module_dx.key,
                    "treatment_label": t.label,
                    "treatment_key": t.key,
                    "source": "WHO_mhGAP_IG",
                    "evidence": "Guideline_recommendation_text_mined",
                    "url": mhgap_url_used or "",
                    "extraction": f"module_segment_pages(code={code};pages={start_page+1}-{end_page+1})",
                }
            )

        # module-level symptom extraction
        if int(args.include_symptoms) == 1:
            sym_cands = mhgap_extract_symptoms_from_segment(seg_lines, verbose=False)
            log(f"mhGAP module {code} '{module_dx_label}': extracted symptoms={len(sym_cands)}", verbose=verbose)
            for s_lab in sym_cands:
                sym = upsert_problem(
                    problems_by_norm,
                    kind="SYM",
                    label=s_lab,
                    source="WHO_mhGAP_IG",
                    provenance=f"linked_from_module={code};pages={start_page+1}-{end_page+1}"
                )
                dx_sym_edges.append(
                    {
                        "dx_label": module_dx.label,
                        "dx_key": module_dx.key,
                        "symptom_label": sym.label,
                        "symptom_key": sym.key,
                        "source": "WHO_mhGAP_IG",
                        "url": mhgap_url_used or "",
                        "extraction": f"symptom_triggers_in_module_pages(code={code};pages={start_page+1}-{end_page+1})",
                    }
                )

        # sub-condition extraction
        sub_sections = mhgap_split_subcondition_sections(
            seg_lines=seg_lines,
            mho_terms=mho_terms,
            mho_token_index=mho_token_index,
            enable=bool(int(args.enable_mhgap_subconditions)),
            use_mho_validation=bool(int(args.mho_validate_subconditions)),
            mho_min_score=float(args.mhgap_subcondition_mho_min_score),
            max_sections=clamp_int(int(args.mhgap_subcondition_max_per_module), 0, 200),
            min_lines_per_section=clamp_int(int(args.mhgap_subcondition_min_lines), 5, 200),
        )

        if sub_sections:
            log(f"mhGAP module {code}: detected sub-conditions={len(sub_sections)} (will map treatments within sections)", verbose=verbose)

        for dx_lab, section_lines, prov_tag in sub_sections:
            sub_dx = upsert_problem(
                problems_by_norm,
                kind="DX",
                label=dx_lab,
                source="WHO_mhGAP_IG",
                provenance=f"module_code={code};{prov_tag};pages={start_page+1}-{end_page+1}"
            )

            trt2 = mhgap_extract_treatments_from_segment(section_lines)
            for t_lab in trt2:
                t = upsert_treatment(
                    treatments_by_norm,
                    label=t_lab,
                    source="WHO_mhGAP_IG",
                    url=mhgap_url_used,
                    evidence="Guideline_recommendation_text_mined(subcondition_section)",
                )

                dx_trt_edges.append(
                    {
                        "problem_kind": "DX",
                        "problem_label": sub_dx.label,
                        "problem_key": sub_dx.key,
                        "treatment_label": t.label,
                        "treatment_key": t.key,
                        "source": "WHO_mhGAP_IG",
                        "evidence": "Guideline_recommendation_text_mined(subcondition_section)",
                        "url": mhgap_url_used or "",
                        "extraction": f"subcondition_section({prov_tag};module={code};pages={start_page+1}-{end_page+1})",
                    }
                )

            if int(args.include_symptoms) == 1:
                sym2 = mhgap_extract_symptoms_from_segment(section_lines, verbose=False)
                for s_lab in sym2:
                    sym = upsert_problem(
                        problems_by_norm,
                        kind="SYM",
                        label=s_lab,
                        source="WHO_mhGAP_IG",
                        provenance=f"linked_from_subcondition={dx_lab};module={code};pages={start_page+1}-{end_page+1}"
                    )
                    dx_sym_edges.append(
                        {
                            "dx_label": sub_dx.label,
                            "dx_key": sub_dx.key,
                            "symptom_label": sym.label,
                            "symptom_key": sym.key,
                            "source": "WHO_mhGAP_IG",
                            "url": mhgap_url_used or "",
                            "extraction": f"symptom_triggers_in_subcondition_section({prov_tag};module={code};pages={start_page+1}-{end_page+1})",
                        }
                    )

    log_kv("mhGAP extraction summary", {
        "dx_trt_edges": len(dx_trt_edges),
        "dx_sym_edges": len(dx_sym_edges),
        "unique_treatments_so_far": len(treatments_by_norm),
        "unique_problems_so_far": len(problems_by_norm),
        "unique_dx_so_far": len([p for p in problems_by_norm.values() if p.kind == "DX"]),
        "unique_sym_so_far": len([p for p in problems_by_norm.values() if p.kind == "SYM"]),
    }, verbose=verbose)

    # ---------------------------
    # 3) Scrape SCP archive (DOM-based Related Diagnosis extraction)
    # ---------------------------
    log("Step 3: Scraping SCP Psychological Treatments Archive (DOM extraction for Related Diagnosis)...", verbose=verbose)

    scp_treatment_urls = scp_collect_treatment_urls(
        session=session,
        max_pages=int(args.max_scp_pages),
        snapshots_dir=snapshots_dir,
        sleep_s=float(args.sleep_s),
        verbose=verbose
    )
    scp_treatment_urls = scp_treatment_urls[: int(args.max_scp_posts)]
    log(f"SCP treatment URLs to process (capped): {len(scp_treatment_urls)}", verbose=verbose)

    scp_edges: List[Dict] = []
    scp_fail = 0

    for i, u in enumerate(scp_treatment_urls, start=1):
        log(f"SCP: processing {i}/{len(scp_treatment_urls)} -> {u}", verbose=verbose)
        try:
            title, related_dx, evidence = scp_parse_treatment_post(
                session=session,
                url=u,
                snapshots_dir=snapshots_dir,
                verbose=verbose
            )
            if not title:
                continue

            t = upsert_treatment(
                treatments_by_norm,
                label=title,
                source="SCP_Chambless_Archive",
                url=u,
                evidence=(evidence or ""),
            )

            for dx_label in related_dx:
                dxp = upsert_problem(
                    problems_by_norm,
                    kind="DX",
                    label=dx_label,
                    source="SCP_Chambless_Archive",
                    provenance=u
                )
                scp_edges.append(
                    {
                        "problem_kind": "DX",
                        "problem_label": dxp.label,
                        "problem_key": dxp.key,
                        "treatment_label": t.label,
                        "treatment_key": t.key,
                        "source": "SCP_Chambless_Archive",
                        "evidence": evidence or "",
                        "url": u,
                        "extraction": "Related_Diagnosis_section(DOM)",
                    }
                )

        except Exception as e:
            scp_fail += 1
            log(f"SCP: FAILED parsing post -> {e}", verbose=verbose, level="WARN")
            continue

        time.sleep(float(args.sleep_s))

    dx_trt_edges.extend(scp_edges)

    log_kv("SCP scrape summary (DOM)", {
        "scp_urls_processed": len(scp_treatment_urls),
        "scp_failures": scp_fail,
        "scp_dx_trt_edges_added": len(scp_edges),
        "dx_trt_edges_total": len(dx_trt_edges),
        "unique_problems_total_now": len(problems_by_norm),
    }, verbose=verbose)

    # ---------------------------
    # 4) Infer SYM -> TRT via shared diagnoses
    # ---------------------------
    inferred_sym_trt: List[Dict] = []
    sym_trt_edges: List[Dict] = []

    if int(args.include_symptoms) == 1 and dx_sym_edges:
        log("Step 4: Inferring SYM->TRT edges via shared DX support...", verbose=verbose)

        dx_to_syms: Dict[str, Set[str]] = {}
        for e in dx_sym_edges:
            dx_to_syms.setdefault(e["dx_key"], set()).add(e["symptom_key"])

        dx_to_trts: Dict[str, Set[str]] = {}
        for e in dx_trt_edges:
            if e.get("problem_kind") != "DX":
                continue
            dx_to_trts.setdefault(e["problem_key"], set()).add(e["treatment_key"])

        sym_to_dxs: Dict[str, Set[str]] = {}
        for dxk, syms in dx_to_syms.items():
            for sk in syms:
                sym_to_dxs.setdefault(sk, set()).add(dxk)

        min_support = int(args.min_symptom_support)
        log(f"SYM->TRT inference threshold (min_support): {min_support}", verbose=verbose)

        prob_by_key = {p.key: p for p in problems_by_norm.values()}
        trt_by_key = {t.key: t for t in treatments_by_norm.values()}

        for symk, dxs in sym_to_dxs.items():
            support: Dict[str, int] = {}
            for dxk in dxs:
                for tk in dx_to_trts.get(dxk, set()):
                    support[tk] = support.get(tk, 0) + 1

            for tk, cnt in support.items():
                if cnt >= min_support:
                    sym_obj = prob_by_key.get(symk)
                    trt_obj = trt_by_key.get(tk)
                    if not sym_obj or not trt_obj:
                        continue

                    sym_trt_edges.append(
                        {
                            "problem_kind": "SYM",
                            "problem_label": sym_obj.label,
                            "problem_key": sym_obj.key,
                            "treatment_label": trt_obj.label,
                            "treatment_key": trt_obj.key,
                            "source": "Inferred_via_shared_DX",
                            "evidence": f"support_dx_count={cnt};threshold={min_support}",
                            "url": "",
                            "extraction": "SYM->(DX)->TRT",
                        }
                    )
                    inferred_sym_trt.append(
                        {
                            "symptom_key": sym_obj.key,
                            "symptom_label": sym_obj.label,
                            "treatment_key": trt_obj.key,
                            "treatment_label": trt_obj.label,
                            "supporting_dx_count": cnt,
                            "threshold": min_support,
                        }
                    )

        log_kv("SYM inference summary", {
            "symptoms_with_any_dx_support": len(sym_to_dxs),
            "inferred_sym_trt_edges": len(sym_trt_edges),
        }, verbose=verbose)
    else:
        log("Step 4: SYM inference skipped (include_symptoms=0 or no DX->SYM edges).", verbose=verbose)

    # ---------------------------
    # 5) MHO alignment + BIO/PSYCHO/SOCIAL enrichment
    # ---------------------------
    log("Step 5: Enriching treatments with MHO alignment + BIO/PSYCHO/SOCIAL hierarchy...", verbose=verbose)

    for i, t in enumerate(treatments_by_norm.values(), start=1):
        if verbose and (i % 50 == 0):
            log(f"  enriching treatments... {i}/{len(treatments_by_norm)}", verbose=verbose)

        if mho_terms:
            iri, lab, score = match_to_mho(
                t.label,
                mho_terms,
                mho_token_index,
                min_score=float(args.mho_min_score),
                max_candidates=int(args.mho_max_candidates)
            )
            t.mho_match_iri = iri
            t.mho_match_label = lab
            t.mho_match_score = float(score)
            if iri and mho_graph is not None:
                t.mho_ancestor_labels = mho_ancestor_labels(mho_graph, iri)

        primary, secondary, rationale = infer_bps_primary_secondary(
            treatment_label=t.label,
            treatment_type=t.ttype,
            mho_label=t.mho_match_label,
            ancestor_labels=t.mho_ancestor_labels
        )
        t.primary_domain = primary
        t.secondary_domain = secondary

        t.evidence.add(f"BPS_classification({rationale})")

    log("Step 5 done: treatment enrichment completed.", verbose=verbose)

    # ---------------------------
    # 6) Finalize problems + treatments; write lookups + raw edges
    # ---------------------------
    log("Step 6: Writing lookup tables and raw edge tables...", verbose=verbose)

    problems = list(problems_by_norm.values())
    problems.sort(key=lambda p: (p.kind, p.label.lower()))

    treatments = list(treatments_by_norm.values())
    treatments.sort(key=lambda t: (t.primary_domain, t.secondary_domain, t.ttype, t.label.lower()))

    df_problems = pd.DataFrame(
        [
            {
                "problem_key": p.key,
                "kind": p.kind,
                "label": p.label,
                "sources": "|".join(sorted(p.sources)),
                "provenance": "|".join(sorted(p.provenances)),
            }
            for p in problems
        ]
    )
    df_problems.to_csv(f_problems, sep="\t", index=False)

    df_treatments = pd.DataFrame(
        [
            {
                "treatment_key": t.key,
                "label": t.label,
                "type": t.ttype,
                "primary_domain": t.primary_domain,
                "secondary_domain": t.secondary_domain,
                "sources": "|".join(sorted(t.sources)),
                "evidence": "|".join(sorted(t.evidence)),
                "urls": "|".join(sorted(t.urls)),
                "mho_match_label": t.mho_match_label,
                "mho_match_iri": t.mho_match_iri,
                "mho_match_score": float(t.mho_match_score),
                "mho_ancestor_labels": t.mho_ancestor_labels,
            }
            for t in treatments
        ]
    )
    df_treatments.to_csv(f_treatments, sep="\t", index=False)

    pd.DataFrame(dx_trt_edges).to_csv(f_dx_trt, sep="\t", index=False)
    pd.DataFrame(dx_sym_edges).to_csv(f_dx_sym, sep="\t", index=False)
    pd.DataFrame(inferred_sym_trt).to_csv(f_sym_trt, sep="\t", index=False)

    log_kv("Wrote raw/lookup tables", {
        "problems_lookup.tsv": f_problems,
        "treatments_lookup.tsv": f_treatments,
        "raw_pairs_dx_treatment.tsv": f_dx_trt,
        "raw_pairs_dx_symptom.tsv": f_dx_sym,
        "inferred_pairs_symptom_treatment.tsv": f_sym_trt,
    }, verbose=verbose)

    # ---------------------------
    # 7) Build and save matrix
    # ---------------------------
    log("Step 7: Building final matrix CSV...", verbose=verbose)
    df_matrix = build_binary_matrix(
        treatments=treatments,
        problems=problems,
        dx_trt_edges=dx_trt_edges,
        sym_trt_edges=sym_trt_edges,
    )
    df_matrix.to_csv(f_matrix, index=False)
    log(f"Matrix saved: {f_matrix}", verbose=verbose)

    # ---------------------------
    # 8) Provenance + run log
    # ---------------------------
    problem_cols = [p.key for p in problems]
    density = float(df_matrix[problem_cols].to_numpy().mean()) if problem_cols else 0.0
    bps_counts = df_treatments["primary_domain"].value_counts(dropna=False).to_dict()

    prov = {
        "run_time": now_iso(),
        "out_dir": out_dir,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "HAS_RDFLIB": HAS_RDFLIB,
        },
        "sources": {
            "WHO_mhGAP_pdf_url_used": mhgap_url_used,
            "WHO_mhGAP_pdf_local": mhgap_pdf_path,
            "WHO_mhGAP_pdf_sha256": sha256_file(mhgap_pdf_path) if os.path.exists(mhgap_pdf_path) else "",
            "SCP_listing_base": SCP_LISTING_BASE,
            "SCP_treatment_prefix": SCP_TREATMENT_PREFIX,
            "SCP_snapshots_dir": snapshots_dir,
            "MHO_owl_url": MHO_OWL_URL,
            "MHO_owl_local": mho_path,
            "MHO_owl_sha256": sha256_file(mho_path) if os.path.exists(mho_path) else "",
        },
        "parameters": {
            "include_symptoms": int(args.include_symptoms),
            "min_symptom_support": int(args.min_symptom_support),
            "mhgap_prefer_v2": int(args.mhgap_prefer_v2),
            "enable_mhgap_subconditions": int(args.enable_mhgap_subconditions),
            "mhgap_subcondition_max_per_module": int(args.mhgap_subcondition_max_per_module),
            "mhgap_subcondition_min_lines": int(args.mhgap_subcondition_min_lines),
            "mho_validate_subconditions": int(args.mho_validate_subconditions),
            "mhgap_subcondition_mho_min_score": float(args.mhgap_subcondition_mho_min_score),
            "max_scp_pages": int(args.max_scp_pages),
            "max_scp_post_caps": int(args.max_scp_posts),
            "sleep_s": float(args.sleep_s),
            "mho_min_score": float(args.mho_min_score),
            "mho_max_candidates": int(args.mho_max_candidates),
        },
        "counts": {
            "n_problems_total": int(len(problems)),
            "n_dx": int(sum(1 for p in problems if p.kind == "DX")),
            "n_sym": int(sum(1 for p in problems if p.kind == "SYM")),
            "n_treatments": int(len(treatments)),
            "n_dx_trt_edges": int(len(dx_trt_edges)),
            "n_dx_sym_edges": int(len(dx_sym_edges)),
            "n_inferred_sym_trt_edges": int(len(sym_trt_edges)),
            "scp_treatment_urls_processed": int(len(scp_treatment_urls)),
            "scp_parse_failures": int(scp_fail),
            "bps_primary_counts": bps_counts,
        },
        "matrix": {
            "rows_treatments": int(df_matrix.shape[0]),
            "cols_problems": int(len(problem_cols)),
            "density_mean_cell": density,
        },
        "warnings": [
            "This is an annotation/research matrix derived from guideline text + curated evidence summaries. Not clinical advice.",
            "mhGAP extraction is heuristic text-mining; validate important edges against the downloaded PDF.",
            "SYM->TRT edges are inferred via shared diagnoses; treat as weaker evidence than direct DX->TRT links.",
            "MHO alignment is lexical; it improves metadata and classification robustness but does not create clinical evidence.",
            "mhGAP sub-condition extraction depends on PDF structure; enable MHO validation for fewer false positives.",
        ],
    }
    write_json(f_prov, prov)

    top_treat = (
        df_matrix.assign(_sum=df_matrix[problem_cols].sum(axis=1))
        .sort_values("_sum", ascending=False)
        .head(20)[["treatment_label", "primary_domain", "secondary_domain", "treatment_type", "_sum"]]
    )

    top_prob = (
        pd.DataFrame({"problem_key": problem_cols, "n_treatments": df_matrix[problem_cols].sum(axis=0).astype(int).values})
        .merge(df_problems[["problem_key", "kind", "label"]], on="problem_key", how="left")
        .sort_values("n_treatments", ascending=False)
        .head(30)[["kind", "label", "n_treatments"]]
    )

    log_lines = []
    log_lines.append("ANNOTATED PROBLEM → SOLUTION MAP (APA_ARCHIVE) [DOM SCP dx + mhGAP subconditions + improved symptoms]")
    log_lines.append("-" * 120)
    log_lines.append(f"run_time: {prov['run_time']}")
    log_lines.append(f"out_dir:  {out_dir}")
    log_lines.append("")
    log_lines.append("Counts:")
    for k, v in prov["counts"].items():
        log_lines.append(f"  {k}: {v}")
    log_lines.append("")
    log_lines.append("Matrix:")
    for k, v in prov["matrix"].items():
        log_lines.append(f"  {k}: {v}")
    log_lines.append("")
    log_lines.append("Top treatments by number of linked problems:")
    log_lines.append(top_treat.to_csv(sep="\t", index=False).rstrip())
    log_lines.append("")
    log_lines.append("Top problems by number of linked treatments:")
    log_lines.append(top_prob.to_csv(sep="\t", index=False).rstrip())
    log_lines.append("")
    log_lines.append("Files written:")
    for fp in [f_matrix, f_problems, f_treatments, f_dx_trt, f_dx_sym, f_sym_trt, f_prov, f_log]:
        log_lines.append(f"  {fp}")
    log_lines.append("")
    log_lines.append("Warnings:")
    for w in prov["warnings"]:
        log_lines.append(f"  - {w}")

    write_text(f_log, "\n".join(log_lines))

    log("=" * 110, verbose=verbose)
    log("RUN COMPLETE ✅", verbose=verbose)
    log(f"Final matrix: {f_matrix}", verbose=verbose)
    log(f"Run log:      {f_log}", verbose=verbose)
    log("=" * 110, verbose=verbose)


if __name__ == "__main__":
    main()
