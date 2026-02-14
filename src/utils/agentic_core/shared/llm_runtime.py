from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, ValidationError


@dataclass
class StructuredLLMResult:
    success: bool
    provider: str
    model: str
    parsed: Optional[Dict[str, Any]]
    raw_text: str
    validation_error: Optional[str]
    used_repair: bool
    usage: Dict[str, Any]
    failure_reason: Optional[str] = None


class StructuredLLMClient:
    def __init__(
        self,
        *,
        model: str = "gpt-5-nano",
        timeout_seconds: float = 90.0,
        max_attempts: int = 2,
        repair_attempts: int = 1,
    ) -> None:
        self.model = str(model)
        self.timeout_seconds = float(timeout_seconds)
        self.max_attempts = int(max(1, max_attempts))
        self.repair_attempts = int(max(0, repair_attempts))
        self.base_backoff_seconds = 0.7

    def _get_client(self):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"openai package is not available: {repr(exc)}")
        return OpenAI(timeout=self.timeout_seconds)

    def _extract_response_text(self, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        output = getattr(response, "output", None)
        if isinstance(output, list):
            chunks: list[str] = []
            for item in output:
                content = getattr(item, "content", None)
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict):
                            if c.get("type") in {"output_text", "text"} and c.get("text"):
                                chunks.append(str(c["text"]))
                        else:
                            text = getattr(c, "text", None)
                            if text:
                                chunks.append(str(text))
            if chunks:
                return "\n".join(chunks).strip()

        if hasattr(response, "model_dump_json"):
            return response.model_dump_json(indent=2)
        return str(response)

    def _extract_usage(self, response: Any) -> Dict[str, Any]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        if isinstance(usage, dict):
            return usage
        if hasattr(usage, "model_dump"):
            try:
                return usage.model_dump()
            except Exception:
                return {"usage_repr": repr(usage)}
        return {"usage_repr": repr(usage)}

    def _responses_api_call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: Dict[str, Any],
    ) -> StructuredLLMResult:
        client = self._get_client()
        response = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "structured_output",
                    "schema": schema,
                    "strict": True,
                }
            },
        )
        raw_text = self._extract_response_text(response)
        return StructuredLLMResult(
            success=True,
            provider="openai_responses_api",
            model=self.model,
            parsed=None,
            raw_text=raw_text,
            validation_error=None,
            used_repair=False,
            usage=self._extract_usage(response),
        )

    def _validate_json(self, raw_text: str, schema_model: Type[BaseModel]) -> StructuredLLMResult:
        try:
            data = json.loads(raw_text)
        except Exception as exc:
            return StructuredLLMResult(
                success=False,
                provider="json_parser",
                model=self.model,
                parsed=None,
                raw_text=raw_text,
                validation_error=f"JSON parse error: {repr(exc)}",
                used_repair=False,
                usage={},
                failure_reason="schema_validation_failed",
            )
        try:
            validated = schema_model.model_validate(data)
            return StructuredLLMResult(
                success=True,
                provider="pydantic",
                model=self.model,
                parsed=validated.model_dump(mode="json"),
                raw_text=raw_text,
                validation_error=None,
                used_repair=False,
                usage={},
            )
        except ValidationError as exc:
            return StructuredLLMResult(
                success=False,
                provider="pydantic",
                model=self.model,
                parsed=None,
                raw_text=raw_text,
                validation_error=str(exc),
                used_repair=False,
                usage={},
                failure_reason="schema_validation_failed",
            )

    def _repair_json_once(
        self,
        *,
        schema: Dict[str, Any],
        schema_model: Type[BaseModel],
        raw_text: str,
        validation_error: str,
    ) -> StructuredLLMResult:
        repair_system = (
            "You are a strict JSON repair assistant. Return ONLY valid JSON that conforms exactly to the provided schema."
        )
        repair_user = (
            "Repair the following output so it is valid JSON and schema-compliant.\n\n"
            f"Schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
            f"Validation error:\n{validation_error}\n\n"
            f"Output to repair:\n{raw_text}"
        )
        call_result = self._responses_api_call(
            system_prompt=repair_system,
            user_prompt=repair_user,
            schema=schema,
        )
        validation_result = self._validate_json(call_result.raw_text, schema_model=schema_model)
        validation_result.provider = call_result.provider
        validation_result.model = call_result.model
        validation_result.usage = call_result.usage
        validation_result.used_repair = True
        return validation_result

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_model: Type[BaseModel],
    ) -> StructuredLLMResult:
        schema = schema_model.model_json_schema()
        last_error: Optional[StructuredLLMResult] = None

        for _ in range(self.max_attempts):
            try:
                call_result = self._responses_api_call(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    schema=schema,
                )
            except Exception as exc:
                last_error = StructuredLLMResult(
                    success=False,
                    provider="openai_responses_api",
                    model=self.model,
                    parsed=None,
                    raw_text="",
                    validation_error=repr(exc),
                    used_repair=False,
                    usage={},
                    failure_reason="provider_unavailable",
                )
                sleep_s = min(6.0, self.base_backoff_seconds * (2**_))
                time.sleep(sleep_s)
                continue

            validation_result = self._validate_json(call_result.raw_text, schema_model=schema_model)
            validation_result.provider = call_result.provider
            validation_result.model = call_result.model
            validation_result.usage = call_result.usage
            if validation_result.success:
                return validation_result

            last_error = validation_result
            if self.repair_attempts > 0:
                repair_result = self._repair_json_once(
                    schema=schema,
                    schema_model=schema_model,
                    raw_text=validation_result.raw_text,
                    validation_error=validation_result.validation_error or "validation_error",
                )
                if repair_result.success:
                    return repair_result
                repair_result.failure_reason = "repair_exhausted"
                last_error = repair_result
            sleep_s = min(6.0, self.base_backoff_seconds * (2**_))
            time.sleep(sleep_s)

        assert last_error is not None
        if last_error.failure_reason in {None, ""}:
            last_error.failure_reason = "budget_exceeded"
        return last_error
