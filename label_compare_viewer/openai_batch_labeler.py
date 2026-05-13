from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path
from typing import Any


DEFAULT_COST_RATES_PER_MILLION = {
    "gpt-5.5-pro": {"input": 30.00, "output": 180.00},
    "gpt-5.5": {"input": 5.00, "output": 30.00},
}


def estimate_openai_cost(
    model: str,
    usage: dict[str, Any] | None,
    rates_per_million: dict[str, dict[str, float]] | None = None,
) -> float | None:
    usage = usage or {}
    rates = rates_per_million or DEFAULT_COST_RATES_PER_MILLION
    model_rates = rates.get(model)
    if not model_rates:
        return None
    input_tokens = _as_int_or_none(usage.get("input_tokens"))
    output_tokens = _as_int_or_none(usage.get("output_tokens"))
    if output_tokens is None:
        output_tokens = _as_int_or_none(usage.get("reasoning_tokens"))
    if input_tokens is None and output_tokens is None:
        return None
    input_cost = (input_tokens or 0) * float(model_rates.get("input", 0.0)) / 1_000_000.0
    output_cost = (output_tokens or 0) * float(model_rates.get("output", 0.0)) / 1_000_000.0
    return input_cost + output_cost


def _as_int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class OpenAIBatchLabeler:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.5",
        reasoning_effort: str = "medium",
        image_detail: str = "original",
        max_output_tokens: int = 3000,
        use_background: bool = True,
    ):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.image_detail = image_detail
        self.max_output_tokens = int(max_output_tokens)
        self.use_background = bool(use_background)

    def _image_data_url(self, image_path: Path) -> str:
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{payload}"

    def _api_image_detail(self) -> tuple[str, str | None]:
        if self.image_detail == "original":
            return "high", "OpenAI image detail does not support 'original'; sent 'high' for full-detail analysis."
        if self.image_detail not in {"auto", "high", "low"}:
            return "auto", f"Unsupported image detail '{self.image_detail}' was replaced with 'auto'."
        return self.image_detail, None

    def _create_kwargs(self, image_path: Path, prompt: str, background: bool) -> dict[str, Any]:
        api_detail, _warning = self._api_image_detail()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "store": False,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": self._image_data_url(image_path),
                            "detail": api_detail,
                        },
                    ],
                }
            ],
            "max_output_tokens": self.max_output_tokens,
        }
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if background:
            kwargs["background"] = True
        return kwargs

    def label_image(self, image_path: Path, prompt: str) -> dict[str, Any]:
        started = time.monotonic()
        response = None
        warnings: list[str] = []
        _api_detail, detail_warning = self._api_image_detail()
        if detail_warning:
            warnings.append(detail_warning)

        if self.use_background:
            warnings.append("Background mode requires stored responses; used synchronous mode to preserve store=False.")

        if response is None:
            kwargs = self._create_kwargs(image_path, prompt, background=False)
            try:
                response = self.client.responses.create(**kwargs)
            except TypeError:
                kwargs.pop("reasoning", None)
                response = self.client.responses.create(**kwargs)
                warnings.append("reasoning effort was not accepted by this SDK/model and was omitted")
            except Exception as exc:
                if "reasoning" not in kwargs:
                    raise
                kwargs.pop("reasoning", None)
                try:
                    response = self.client.responses.create(**kwargs)
                    warnings.append(f"reasoning effort was rejected and was omitted: {exc}")
                except Exception:
                    raise exc

        status = getattr(response, "status", "completed")
        if status not in {"completed", None, ""}:
            error = getattr(response, "error", None)
            raise RuntimeError(f"OpenAI response ended with status {status}: {error}")

        usage = self._usage_dict(getattr(response, "usage", None))
        return {
            "output_text": getattr(response, "output_text", "") or "",
            "response_id": getattr(response, "id", ""),
            "status": status or "completed",
            "usage": usage,
            "warnings": warnings,
            "elapsed_seconds": time.monotonic() - started,
        }

    def _usage_dict(self, usage: Any) -> dict[str, Any]:
        if usage is None:
            return {}
        if isinstance(usage, dict):
            data = dict(usage)
        elif hasattr(usage, "model_dump"):
            data = usage.model_dump()
        else:
            data = {}
            for key in ("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens"):
                if hasattr(usage, key):
                    data[key] = getattr(usage, key)
        output_details = data.get("output_tokens_details") or {}
        if isinstance(output_details, dict) and "reasoning_tokens" in output_details:
            data["reasoning_tokens"] = output_details.get("reasoning_tokens")
        return data
