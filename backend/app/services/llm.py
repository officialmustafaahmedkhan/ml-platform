"""Pluggable LLM client for generating dataset Outcome labels.

Supports two providers without extra dependencies (stdlib ``urllib`` only):

* ``openai``  — any OpenAI-compatible ``/chat/completions`` endpoint.
* ``ollama``  — local Ollama ``/api/chat`` endpoint.

Both expose the same ``chat()`` interface so the labeling pipeline is
provider-agnostic. Set ``LLM_PROVIDER`` in the environment (or ``.env``).
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from ..config import settings

PROVIDERS = ("off", "openai", "ollama")


class LLMError(RuntimeError):
    pass


def _strip_code_fence(text: str) -> str:
    """Remove ```json ... ``` fences if the model wraps its answer in one."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return m.group(1) if m else text


def _parse_json(text: str) -> Any:
    """Robustly parse a JSON payload the model returned."""
    text = _strip_code_fence(text)
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    # Fall back to the first JSON array/object embedded in the text.
    for pattern in (r"\[.*\]", r"\{.*\}"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except (ValueError, TypeError):
                continue
    raise LLMError(f"LLM returned unparseable JSON: {text[:300]}")


class LLMClient:
    """Minimal chat client shared by the OpenAI and Ollama providers."""

    def __init__(self, provider: str):
        self.provider = provider.lower()
        if self.provider not in PROVIDERS:
            raise LLMError(f"Unknown LLM provider '{provider}' (expected {PROVIDERS})")
        if self.provider == "off":
            raise LLMError("LLM is disabled (set LLM_PROVIDER=openai or ollama)")

    def _post_json(self, url: str, headers: dict, payload: dict) -> dict:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=settings.LLM_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise LLMError(f"{self.provider} HTTP {exc.code}: {exc.read().decode('utf-8')[:400]}") from exc
        except Exception as exc:  # noqa: BLE001  (network / JSON errors)
            raise LLMError(f"{self.provider} request failed: {exc}") from exc

    def chat(self, messages: list[dict], *, temperature: float = 0.2, max_tokens: int = 2000) -> str:
        """Send a chat request; return the assistant message text."""
        if self.provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise LLMError("OPENAI_API_KEY is not configured")
            data = self._post_json(
                settings.OPENAI_BASE_URL.rstrip("/") + "/chat/completions",
                {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
                {
                    "model": settings.OPENAI_MODEL,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            return data["choices"][0]["message"]["content"].strip()

        if self.provider == "ollama":
            data = self._post_json(
                settings.OLLAMA_BASE_URL.rstrip("/") + "/api/chat",
                {},
                {
                    "model": settings.OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
            )
            return data["message"]["content"].strip()

        raise LLMError(f"Unsupported provider: {self.provider}")


def get_llm() -> Optional[LLMClient]:
    """Return a configured client, or ``None`` when the LLM is disabled."""
    if not settings.LLM_ENABLED:
        return None
    return LLMClient(settings.LLM_PROVIDER)


# --------------------------------------------------------------------------- #
# Labeling pipeline
# --------------------------------------------------------------------------- #
def propose_outcomes(df, num_categories: int = 3) -> list[dict]:
    """Ask the LLM to design outcome categories for a dataset.

    Returns a list of ``{"name": ..., "description": ...}`` dicts.
    """
    llm = get_llm()
    if llm is None:
        raise LLMError("LLM is disabled (set LLM_PROVIDER=openai or ollama)")

    columns = [str(c) for c in df.columns]
    preview = df.head(15).fillna("").astype(str).to_dict(orient="records")

    prompt = f"""You are a data-science assistant. A user uploaded a CSV with columns: {json.dumps(columns)}.
Here are sample rows:
{json.dumps(preview, indent=2, default=str)}

Design exactly {num_categories} meaningful "Outcome" categories that would make a good
classification target for this data (for example high/medium/low risk or value buckets).
Respond with ONLY valid JSON, an array of exactly {num_categories} objects,
each with keys "name" (short, e.g. "Low") and "description" (one sentence explaining what it means)."""
    raw = llm.chat([{"role": "user", "content": prompt}], temperature=0.4)
    outcomes = _parse_json(raw)
    if not isinstance(outcomes, list) or not outcomes:
        raise LLMError("LLM did not return a list of outcome categories")
    cleaned = []
    for item in outcomes[:num_categories]:
        if isinstance(item, dict) and item.get("name"):
            cleaned.append({"name": str(item["name"]).strip()[:40], "description": str(item.get("description", "")).strip()})
    if not cleaned:
        raise LLMError("LLM returned no usable outcome categories")
    return cleaned


def label_rows(df, outcome_names: list[str], batch_size: int = 25, max_rows: Optional[int] = None) -> list[str]:
    """Label each row by asking the LLM to classify batches of rows.

    Returns one label per row (aligned with ``df`` order).
    """
    llm = get_llm()
    if llm is None:
        raise LLMError("LLM is disabled (set LLM_PROVIDER=openai or ollama)")

    work = df if max_rows is None else df.head(max_rows)
    labels: list[str] = []
    batch_size = max(1, min(int(batch_size), 100))

    for start in range(0, len(work), batch_size):
        batch = work.iloc[start:start + batch_size]
        rows = batch.fillna("").astype(str).to_dict(orient="records")
        prompt = f"""You are labeling rows of a dataset. Classify EACH row into exactly one of these outcome categories:
{json.dumps(outcome_names)}

The rows (each is a JSON object) are:
{json.dumps(rows, default=str)}

Respond with ONLY a JSON array of strings, one label per row, in the SAME ORDER as the rows. Use only the exact category names given."""
        raw = llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=3000)
        parsed = _parse_json(raw)
        if not isinstance(parsed, list):
            raise LLMError("LLM did not return a label array")
        batch_labels = [str(v).strip() for v in parsed[: len(rows)]]
        if len(batch_labels) < len(rows):
            raise LLMError(f"LLM returned {len(batch_labels)} labels for {len(rows)} rows")
        labels.extend(batch_labels)

    if max_rows is not None and len(labels) < len(df):
        # Pad untagged rows with "Unknown" (keeps the column aligned).
        labels.extend(["Unknown"] * (len(df) - len(labels)))
    return labels


def llm_status() -> dict:
    """Describe the configured provider without making a network call."""
    return {
        "enabled": settings.LLM_ENABLED,
        "provider": settings.LLM_PROVIDER,
        "model": settings.OPENAI_MODEL if settings.LLM_PROVIDER.lower() == "openai"
                 else settings.OLLAMA_MODEL if settings.LLM_PROVIDER.lower() == "ollama"
                 else None,
        "api_key_configured": bool(settings.OPENAI_API_KEY),
        "endpoint": settings.OPENAI_BASE_URL if settings.LLM_PROVIDER.lower() == "openai"
                    else settings.OLLAMA_BASE_URL if settings.LLM_PROVIDER.lower() == "ollama"
                    else None,
    }
