"""
Per-part LMM analyses + cross-part synthesis for the LLM-as-judge design.

Each ``part{N}_*`` module exposes a ``run()`` entry point that consumes
``data/04_judgments/judgments_long.csv`` and writes figures plus a textual
report to ``results/part{N}_<slug>/``.

The cross-part synthesis lives in :mod:`analysis.synthesis`.
"""
