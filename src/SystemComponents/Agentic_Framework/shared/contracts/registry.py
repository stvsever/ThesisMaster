from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ContractValidationResult:
    success: bool
    contract_name: str
    contract_version: str
    path: str
    message: str
    errors: Optional[list[str]] = None


class ContractValidator:
    DEFAULT_CONTRACT_VERSION = "1.0.0"

    def __init__(self, contracts_root: Optional[Path] = None) -> None:
        if contracts_root is None:
            contracts_root = Path(__file__).resolve().parent
        self.contracts_root = contracts_root
        self.schemas_root = self.contracts_root / "schemas"

    def _schema_path(self, contract_name: str) -> Path:
        return self.schemas_root / f"{contract_name}.schema.json"

    def _load_schema(self, contract_name: str) -> Dict[str, Any]:
        schema_path = self._schema_path(contract_name)
        if not schema_path.exists():
            raise FileNotFoundError(f"Contract schema not found: {schema_path}")
        return json.loads(schema_path.read_text(encoding="utf-8"))

    def _normalize_contract_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(payload)
        if not str(out.get("contract_version") or "").strip():
            out["contract_version"] = self.DEFAULT_CONTRACT_VERSION
        return out

    def validate_payload(self, *, contract_name: str, payload: Dict[str, Any], path: str = "<memory>") -> ContractValidationResult:
        schema = self._load_schema(contract_name)
        normalized = self._normalize_contract_payload(payload)
        try:
            import jsonschema  # type: ignore
        except Exception as exc:
            return ContractValidationResult(
                success=False,
                contract_name=contract_name,
                contract_version=str(normalized.get("contract_version") or self.DEFAULT_CONTRACT_VERSION),
                path=path,
                message=f"jsonschema dependency unavailable: {repr(exc)}",
                errors=[repr(exc)],
            )

        validator = jsonschema.Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(normalized), key=lambda e: list(e.path))
        if errors:
            msg = [f"{'.'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors]
            return ContractValidationResult(
                success=False,
                contract_name=contract_name,
                contract_version=str(normalized.get("contract_version") or self.DEFAULT_CONTRACT_VERSION),
                path=path,
                message="Contract validation failed.",
                errors=msg,
            )
        return ContractValidationResult(
            success=True,
            contract_name=contract_name,
            contract_version=str(normalized.get("contract_version") or self.DEFAULT_CONTRACT_VERSION),
            path=path,
            message="Contract validation passed.",
            errors=[],
        )

    def validate_file(self, *, contract_name: str, payload_path: Path) -> ContractValidationResult:
        if not payload_path.exists():
            return ContractValidationResult(
                success=False,
                contract_name=contract_name,
                contract_version=self.DEFAULT_CONTRACT_VERSION,
                path=str(payload_path),
                message="Payload file does not exist.",
                errors=["missing_file"],
            )
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        return self.validate_payload(contract_name=contract_name, payload=payload, path=str(payload_path))
