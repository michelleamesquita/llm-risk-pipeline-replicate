#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adapter Replicate padronizado para o experimento.

Princípio metodológico:
- envia o MESMO prompt;
- aplica o MESMO limite máximo de saída (1800 por default);
- NÃO força temperature/top_p/top_k, porque esses parâmetros não estão
  disponíveis em todos os quatro endpoints do Replicate.

Isso evita afirmar uma padronização que a API não permite.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import replicate
import yaml

from diff_tools import repair_unified_diff


DIFF_SYSTEM_PROMPT = (
    "Return only a valid unified git diff that solves the task. "
    "Do not include markdown fences or explanations."
)
NON_RETRYABLE_ERRORS = (
    "cannot pickle 'async_generator' object",
    "exceed context window",
)


def load_profile(path: Path, name: str) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if name not in data:
        raise KeyError(f"Perfil {name!r} não encontrado em {path}")
    p = dict(data[name])
    p["profile_name"] = name
    return p


def extract_diff(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n")
    text = re.sub(r"```(?:diff|patch|[A-Za-z0-9_-]+)?\s*\n?", "", text)
    text = text.replace("```", "")

    i = text.find("diff --git ")
    if i >= 0:
        return text[i:].strip() + "\n"

    m = re.search(
        r"(?m)^---\s+(?:a/)?[^\n]+\n^\+\+\+\s+(?:b/)?[^\n]+",
        text,
    )
    return text[m.start():].strip() + "\n" if m else ""


def join_output(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="ignore")
    try:
        return "".join(str(x) for x in output)
    except TypeError:
        return str(output)


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(value, encoding="utf-8")
    temp.replace(path)


def make_input(profile: dict, prompt: str) -> tuple[dict, dict]:
    """
    Retorna:
      API input,
      metadata de parâmetros efetivos.

    Apenas o limite máximo de saída é deliberadamente padronizado.
    """
    schema = profile["schema"]
    max_out = int(profile.get("max_output_tokens", 1800))

    if schema == "gpt4o":
        api_input = {
            "prompt": prompt,
            "system_prompt": DIFF_SYSTEM_PROMPT,
            "max_completion_tokens": max_out,
        }

    elif schema == "claude4":
        api_input = {
            "prompt": prompt,
            "system_prompt": DIFF_SYSTEM_PROMPT,
            "max_tokens": max_out,
            "extended_thinking": False,
        }

    elif schema == "deepseekv3":
        api_input = {
            "prompt": prompt,
            "max_tokens": max_out,
        }

    elif schema == "codellama7":
        api_input = {
            "prompt": prompt,
            "system_prompt": (
                "Return only a valid unified git diff. "
                "No explanations and no markdown."
            ),
            "max_tokens": max_out,
        }

    else:
        raise ValueError(f"Schema não suportado: {schema}")

    # Controles opcionais são aplicados somente quando o perfil os declara.
    # Isso mantém os defaults dos demais endpoints e torna o tuning auditável.
    for control in ("temperature", "top_p", "top_k"):
        if control in profile:
            api_input[control] = profile[control]

    effective = {
        "max_output_tokens_standardized": max_out,
        "temperature_explicitly_set": "temperature" in api_input,
        "temperature": api_input.get("temperature"),
        "top_p_explicitly_set": "top_p" in api_input,
        "top_p": api_input.get("top_p"),
        "top_k_explicitly_set": "top_k" in api_input,
        "top_k": api_input.get("top_k"),
        "do_sample": api_input.get("do_sample"),
    }

    return api_input, effective


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--profile",
        required=True,
    )
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--prompt_file", required=True)
    ap.add_argument("--out_patch", required=True)
    ap.add_argument("--metadata_json")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("ERRO: REPLICATE_API_TOKEN não definido.", file=sys.stderr)
        return 2

    out = Path(args.out_patch)
    metadata_path = Path(args.metadata_json) if args.metadata_json else None
    out.unlink(missing_ok=True)

    profile = load_profile(Path(args.profiles), args.profile)
    prompt = Path(args.prompt_file).read_text(
        encoding="utf-8", errors="ignore"
    )

    api_input, effective = make_input(profile, prompt)

    metadata = {
        "profile": args.profile,
        "display_name": profile.get("display_name"),
        "provider": "replicate",
        "replicate_model": profile["replicate_model"],
        "schema": profile["schema"],
        "transport": (
            "stream"
            if profile.get("use_stream")
            else "run-with-polling"
        ),
        "protocol": (
            "endpoint_tuned"
            if profile.get("tuned")
            else "common_parameters_only"
        ),
        "same_prompt_protocol": True,
        "max_output_tokens_standardized": effective[
            "max_output_tokens_standardized"
        ],
        "temperature_used_as_experimental_factor": (
            effective["temperature_explicitly_set"]
        ),
        "top_p_used_as_experimental_factor": False,
        "top_k_used_as_experimental_factor": False,
        "effective_controls": effective,
        "requested_temperature": profile.get("requested_temperature"),
        "effective_temperature": effective.get("temperature"),
        "prompt_chars": len(prompt),
        "prompt_sha256": hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest(),
        "success": False,
        "error": None,
        "attempts": [],
    }

    raw = ""
    patch = ""
    last_error = None

    for attempt in range(1, max(1, args.retries) + 1):
        try:
            if profile.get("use_stream"):
                raw = "".join(
                    str(event)
                    for event in replicate.stream(
                        profile["replicate_model"],
                        input=api_input,
                    )
                )
            else:
                output = replicate.run(
                    profile["replicate_model"],
                    input=api_input,
                    wait=False,
                )
                raw = join_output(output)

            extracted = extract_diff(raw)
            patch, repair_report = repair_unified_diff(extracted)
            attempt_meta = {
                "attempt": attempt,
                "transport_error": None,
                "raw_output_chars": len(raw),
                "extracted_patch_chars": len(extracted),
                "validated_patch_chars": len(patch),
                "diff_validation": repair_report,
            }
            metadata["attempts"].append(attempt_meta)

            if patch:
                last_error = None
                break

            last_error = "MODEL_DID_NOT_RETURN_VALID_UNIFIED_DIFF"
            attempt_meta["validation_error"] = last_error
            print(
                f"[attempt {attempt}/{args.retries}] {last_error}",
                file=sys.stderr,
            )
        except Exception as exc:
            last_error = str(exc)
            metadata["attempts"].append({
                "attempt": attempt,
                "transport_error": last_error,
                "raw_output_chars": 0,
                "extracted_patch_chars": 0,
                "validated_patch_chars": 0,
            })
            print(
                f"[attempt {attempt}/{args.retries}] {last_error}",
                file=sys.stderr,
            )
            if any(fragment in last_error for fragment in NON_RETRYABLE_ERRORS):
                break
        if attempt < args.retries:
            time.sleep(min(2 ** attempt, 8))

    metadata["attempt_count"] = len(metadata["attempts"])
    metadata["raw_output_chars"] = len(raw)
    metadata["patch_chars"] = len(patch)

    if not patch:
        metadata["error"] = last_error
        if metadata_path:
            write_json_atomic(metadata_path, metadata)
        return 3 if last_error != "MODEL_DID_NOT_RETURN_VALID_UNIFIED_DIFF" else 4

    write_text_atomic(out, patch)

    metadata["success"] = True

    if metadata_path:
        write_json_atomic(metadata_path, metadata)

    print(f"OK: {args.profile} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())