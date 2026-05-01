"""
Pseudodata generators that let the full pipeline run without real PHOENIX
outputs or real LLM-judge calls.

Three artefacts (all under ``data/pseudodata/``):

- ``hcp_outputs.json``       — pseudo HCP outputs for every case.
- ``phoenix_outputs.json``   — pseudo PHOENIX outputs for every case.
- ``judgments_long.csv``     — long-format ratings (also copied to
  ``data/04_judgments/judgments_long.csv`` for the analysis stage).
"""

from .generate_hcp_outputs import generate_hcp_outputs
from .generate_phoenix_outputs import generate_phoenix_outputs
from .generate_judgments import generate_pseudo_judgments

CASE_IDS: tuple[str, ...] = tuple(f"C{i:02d}" for i in range(1, 11))

__all__ = [
    "CASE_IDS",
    "generate_hcp_outputs",
    "generate_phoenix_outputs",
    "generate_pseudo_judgments",
]
