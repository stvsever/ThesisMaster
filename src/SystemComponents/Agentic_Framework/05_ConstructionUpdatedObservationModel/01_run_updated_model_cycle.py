#!/usr/bin/env python3
"""
Compatibility wrapper.

Canonical Step-04 updated-model cycle runner now lives at:
src/SystemComponents/Agentic_Framework/04_ConstructionUpdatedObservationModel/01_run_updated_model_cycle.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType


def _load_canonical_module() -> ModuleType:
    current = Path(__file__).resolve()
    target = current.parents[1] / "04_ConstructionUpdatedObservationModel" / "01_run_updated_model_cycle.py"
    if str(target.parent) not in sys.path:
        sys.path.insert(0, str(target.parent))
    spec = importlib.util.spec_from_file_location("phoenix_step04_cycle_canonical", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load canonical Step-04 cycle module from {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CANONICAL = _load_canonical_module()
for _name in dir(_CANONICAL):
    if _name in {"__name__", "__loader__", "__package__", "__spec__"}:
        continue
    globals()[_name] = getattr(_CANONICAL, _name)


if __name__ == "__main__":
    raise SystemExit(_CANONICAL.main())
