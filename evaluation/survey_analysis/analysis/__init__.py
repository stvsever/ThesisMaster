"""
Per-part LMM analyses + cross-part synthesis for the LLM-as-judge design.

Each ``part{N}_prompt`` module exposes a ``run()`` entry point that consumes
``data/04_judgments/judgments_long.csv`` and writes figures plus a textual
report to ``results/part{N}_prompt/``.

The cross-part synthesis lives in :mod:`analysis.synthesis`; supplementary
stability and sensitivity diagnostics live in :mod:`analysis.supplementary`.
"""
