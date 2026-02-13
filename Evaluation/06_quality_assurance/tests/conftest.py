from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import pytest


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "Evaluation").exists() and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError("Could not locate repository root from tests/conftest.py")


REPO_ROOT = _find_repo_root()


@lru_cache(maxsize=64)
def load_module_from_file(file_path: str, module_name: str) -> ModuleType:
    path = Path(file_path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def repo_file(rel_path: str) -> Path:
    return REPO_ROOT / rel_path


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def repo_file_fn():
    return repo_file


@pytest.fixture(scope="session")
def module_loader():
    return load_module_from_file
