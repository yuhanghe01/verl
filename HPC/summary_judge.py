from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TRAPI_ENDPOINT = "https://dev.hpc-lucia.com/model-proxy"
DEFAULT_TRAPI_API_VERSION = "2025-04-01-preview"
DEFAULT_TRAPI_SCOPE = "api://cd2a0884-fdc5-4c31-8a85-3c49067932d5/.default"
DEFAULT_GCR_HOST = "https://trapi.research.microsoft.com"
DEFAULT_GCR_BASE_URL = f"{DEFAULT_GCR_HOST}/redmond/interactive/openai/v1"
DEFAULT_GCR_SCOPE = "api://trapi/.default"
DEFAULT_JUDGE_MODEL = "gpt-5.2_2025-12-11"
DEFAULT_TRAPI_MAX_ATTEMPTS = 8
DEFAULT_TRAPI_MAX_RETRY_SLEEP_SECONDS = 300.0


@dataclass(frozen=True)
class SummaryJudgeConfig:
    mode: str = "none"
    model: str = DEFAULT_JUDGE_MODEL
    trials: int = 1
    max_input_chars: int = 12000


def judge_summary(
    *,
    prompt: str,
    reference_summary: str,
    candidate_summary: str,
    config: SummaryJudgeConfig,
) -> dict[str, Any]:
    if config.mode == "none":
        return offline_summary_proxy_score(reference_summary, candidate_summary)
    if config.mode == "trapi":
        trial_results = [
            _judge_summary_trapi(
                prompt=prompt,
                reference_summary=reference_summary,
                candidate_summary=candidate_summary,
                config=config,
            )
            for _ in range(max(1, config.trials))
        ]
        return aggregate_summary_trials(trial_results)
    raise ValueError(f"Unsupported summary judge mode: {config.mode}")


def offline_summary_proxy_score(reference_summary: str, candidate_summary: str) -> dict[str, Any]:
    ref_tokens = _token_set(reference_summary)
    pred_tokens = _token_set(candidate_summary)
    overlap = len(ref_tokens & pred_tokens)
    precision = overlap / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap / len(ref_tokens) if ref_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    length_ratio = len(candidate_summary.strip()) / max(1, len(reference_summary.strip()))
    length_score = max(0.0, min(1.0, 1.0 - abs(1.0 - min(length_ratio, 2.0)) * 0.5))
    score = 0.8 * f1 + 0.2 * length_score
    return {
        "judge_mode": "none",
        "primary_score": round(score, 6),
        "summary_judge": {
            "factual_correctness": None,
            "coverage": None,
            "no_hallucination": None,
            "concision": None,
            "rationale": "offline lexical proxy only; use --summary-judge trapi for GPT-5.2 judgment",
        },
        "lexical_proxy": {
            "token_precision": round(precision, 6),
            "token_recall": round(recall, 6),
            "token_f1": round(f1, 6),
            "length_ratio": round(length_ratio, 6),
        },
        "judge_trials": 0,
    }


def aggregate_summary_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        return offline_summary_proxy_score("", "")
    numeric_keys = ["factual_correctness", "coverage", "no_hallucination", "concision", "primary_score"]
    means = {key: _mean([float(t.get(key, 0.0)) for t in trials]) for key in numeric_keys}
    return {
        "judge_mode": "trapi",
        "primary_score": round(means["primary_score"], 6),
        "summary_judge": {
            "factual_correctness": round(means["factual_correctness"], 6),
            "coverage": round(means["coverage"], 6),
            "no_hallucination": round(means["no_hallucination"], 6),
            "concision": round(means["concision"], 6),
            "rationale": " | ".join(str(t.get("rationale", "")).strip() for t in trials if t.get("rationale")),
        },
        "trial_results": trials,
        "judge_trials": len(trials),
    }


def _judge_summary_trapi(
    *,
    prompt: str,
    reference_summary: str,
    candidate_summary: str,
    config: SummaryJudgeConfig,
) -> dict[str, Any]:
    raw = chat_json(
        system_prompt=SUMMARY_JUDGE_SYSTEM,
        user_prompt=build_summary_judge_prompt(
            prompt=prompt,
            reference_summary=reference_summary,
            candidate_summary=candidate_summary,
            max_input_chars=config.max_input_chars,
        ),
        model_name=config.model,
    )
    result = {
        "factual_correctness": _clamp_score(raw.get("factual_correctness")),
        "coverage": _clamp_score(raw.get("coverage")),
        "no_hallucination": _clamp_score(raw.get("no_hallucination")),
        "concision": _clamp_score(raw.get("concision")),
        "rationale": str(raw.get("rationale", "")),
    }
    result["primary_score"] = round(
        0.35 * result["factual_correctness"]
        + 0.30 * result["coverage"]
        + 0.25 * result["no_hallucination"]
        + 0.10 * result["concision"],
        6,
    )
    return result


SUMMARY_JUDGE_SYSTEM = """You are a strict evaluator for Azure HPC InfiniBand incident summaries.
Return one JSON object only. Judge whether the candidate summary is grounded in the provided incident thread and reference summary.
Use scores from 0.0 to 1.0:
- factual_correctness: candidate statements are correct and consistent with evidence.
- coverage: candidate captures trigger, investigation/root cause, and mitigation/resolution when present.
- no_hallucination: candidate avoids unsupported claims, names, dates, and actions.
- concision: candidate is concise and not filled with raw log/table noise.
Also return a short rationale."""


def build_summary_judge_prompt(
    *,
    prompt: str,
    reference_summary: str,
    candidate_summary: str,
    max_input_chars: int,
) -> str:
    trimmed_prompt = prompt[:max_input_chars]
    if len(prompt) > max_input_chars:
        trimmed_prompt += "\n[incident thread truncated for judge input]"
    return (
        "Incident thread:\n"
        f"{trimmed_prompt}\n\n"
        "Reference summary:\n"
        f"{reference_summary}\n\n"
        "Candidate summary:\n"
        f"{candidate_summary}\n\n"
        "Return JSON with keys: factual_correctness, coverage, no_hallucination, concision, rationale."
    )


def chat_json(system_prompt: str, user_prompt: str, model_name: str) -> dict[str, Any]:
    client = _get_azure_openai_client()
    max_attempts = int(os.environ.get("TRAPI_MAX_ATTEMPTS", str(DEFAULT_TRAPI_MAX_ATTEMPTS)))
    completion = None
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            break
        except Exception as exc:  # OpenAI error classes vary across package versions.
            last_exc = exc
            if attempt >= max_attempts or not _is_retryable_openai_error(exc):
                raise
            time.sleep(_retry_sleep_seconds(exc, attempt))
    if completion is None:
        raise RuntimeError("TRAPI chat completion failed") from last_exc
    content = completion.choices[0].message.content or ""
    try:
        return json.loads(_strip_code_fence(content))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Summary judge returned invalid JSON: {content[:500]}") from exc


def _get_azure_openai_client():
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install the openai package to use --summary-judge trapi") from exc

    token_provider = _load_token_provider()
    host = os.environ.get("TRAPI_HOST", DEFAULT_GCR_HOST)
    base_url = os.environ.get(
        "OPENAI_BASE_URL",
        os.environ.get(
            "TRAPI_OPENAI_BASE_URL",
            f"{host}/redmond/interactive/openai/v1",
        ),
    )
    return OpenAI(
        base_url=base_url,
        api_key=token_provider,
        default_headers={
            "traceparent": f"00-{uuid.uuid4().hex}-{uuid.uuid4().hex[:16]}-01",
            "baggage": f"LuciaSessionId={uuid.uuid4()}",
        },
        timeout=float(os.environ.get("TRAPI_TIMEOUT_SECONDS", "60")),
        max_retries=0,
    )


def _load_token_provider():
    static_token = (
        os.environ.get("TRAPI_ACCESS_TOKEN")
        or os.environ.get("AZURE_OPENAI_ACCESS_TOKEN")
        or os.environ.get("OPENAI_API_KEY")
    )
    if static_token:
        return static_token

    try:
        from azure.identity import ChainedTokenCredential, AzureCliCredential, ManagedIdentityCredential, get_bearer_token_provider
    except ModuleNotFoundError:
        return _build_az_cli_token_provider()

    client_id = (
        os.environ.get("TRAPI_AZURE_MANAGED_IDENTITY_CLIENT_ID")
        or os.environ.get("AZURE_MANAGED_IDENTITY_CLIENT_ID")
        or os.environ.get("AZURE_CLIENT_ID")
    )
    if client_id:
        credential = ChainedTokenCredential(
            AzureCliCredential(),
            ManagedIdentityCredential(client_id=client_id),
        )
    else:
        credential = ChainedTokenCredential(AzureCliCredential(), ManagedIdentityCredential())
    scope = os.environ.get(
        "OPENAI_SCOPE",
        os.environ.get(
            "TRAPI_SCOPE",
            os.environ.get("TRAPI_AZURE_OPENAI_SCOPE", os.environ.get("AZURE_OPENAI_SCOPE", DEFAULT_GCR_SCOPE)),
        ),
    )
    return get_bearer_token_provider(credential, scope)


def _build_az_cli_token_provider():
    scope = os.environ.get(
        "OPENAI_SCOPE",
        os.environ.get(
            "TRAPI_SCOPE",
            os.environ.get("TRAPI_AZURE_OPENAI_SCOPE", os.environ.get("AZURE_OPENAI_SCOPE", DEFAULT_GCR_SCOPE)),
        ),
    )

    def _provider() -> str:
        az = shutil.which("az") or shutil.which("az.cmd")
        if not az:
            raise RuntimeError("Azure CLI is required for TRAPI/model-proxy auth when azure-identity is unavailable")
        completed = subprocess.run(
            [az, "account", "get-access-token", "--scope", scope, "--query", "accessToken", "-o", "tsv"],
            check=True,
            capture_output=True,
            text=True,
        )
        token = completed.stdout.strip()
        if not token:
            raise RuntimeError("Azure CLI returned an empty access token")
        return token

    return _provider


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_/-]+", text.lower()))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return max(0.0, min(1.0, score))


def _strip_code_fence(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content)
    return content.strip()


def _is_retryable_openai_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate" in text or "timeout" in text or "temporarily" in text


def _retry_sleep_seconds(exc: Exception, attempt: int) -> float:
    max_sleep = float(os.environ.get("TRAPI_MAX_RETRY_SLEEP_SECONDS", str(DEFAULT_TRAPI_MAX_RETRY_SLEEP_SECONDS)))
    text = str(exc)
    match = re.search(r"try again in (\d+) seconds", text, flags=re.IGNORECASE)
    if match:
        return min(float(match.group(1)) + 2.0, max_sleep)
    return min(2.0**attempt, max_sleep)
 