"""
Parsing of HCP outputs (Qualtrics CSV) and PHOENIX outputs (system JSON)
into a single canonical per-part shape that the LLM judge can consume.
"""

from .canonical_schemas import (
    CANONICAL_FIELD_COUNTS,
    PART_KEYS,
    Part1Item,
    Part1Output,
    Part2Item,
    Part2Output,
    Part3Item,
    Part3Output,
    Part4Output,
    Part5Output,
    canonical_for_judge,
    coerce_part1,
    coerce_part2,
    coerce_part3,
    coerce_part4,
    coerce_part5,
)
from .qualtrics_parser import (
    HCP_RESPONSE,
    parse_qualtrics_csv,
    save_parsed_outputs,
)
from .system_output_loader import (
    SystemOutputBundle,
    load_system_outputs,
    save_system_outputs,
)

__all__ = [
    "CANONICAL_FIELD_COUNTS",
    "PART_KEYS",
    "Part1Item",
    "Part1Output",
    "Part2Item",
    "Part2Output",
    "Part3Item",
    "Part3Output",
    "Part4Output",
    "Part5Output",
    "canonical_for_judge",
    "coerce_part1",
    "coerce_part2",
    "coerce_part3",
    "coerce_part4",
    "coerce_part5",
    "HCP_RESPONSE",
    "parse_qualtrics_csv",
    "save_parsed_outputs",
    "SystemOutputBundle",
    "load_system_outputs",
    "save_system_outputs",
]
