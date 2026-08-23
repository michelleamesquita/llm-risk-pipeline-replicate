#!/usr/bin/env python3
"""
LangSmith LLM-as-a-judge — versão v2 da validação manual.

Entrada:
  manual_validation_review_blinded_v2.csv

O judge recebe SOMENTE:
  - arquivo/linha;
  - regra Bandit + CWE + claim;
  - BEFORE mapeado;
  - diff;
  - AFTER;
  - relação técnica de mapeamento da linha.

O judge NÃO recebe:
  - identidade do LLM;
  - severity;
  - Bandit confidence;
  - is_new_finding / existed_before;
  - LR/RF score;
  - resposta humana.

Saídas:
  - experimento LangSmith;
  - langsmith_bandit_judge_v2_results.csv

Requisitos:
  pip install -U langsmith openai pydantic

Variáveis:
  LANGSMITH_API_KEY
  OPENAI_API_KEY

Uso:
  python langsmith_bandit_judge_v2.py \
      --input manual_validation_v2/manual_validation_review_blinded_v2.csv \
      --judge-model "<modelo>" \
      --outdir langsmith_judge_v2
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
from langsmith import Client, traceable, wrappers
from openai import OpenAI, RateLimitError
from pydantic import BaseModel, Field


class JudgeResult(BaseModel):
    validity: Literal[
        "VALID_SECURITY_FINDING",
        "LIKELY_FALSE_POSITIVE",
        "UNCERTAIN",
    ]
    patch_attribution: Literal[
        "INTRODUCED_OR_AFFECTED_BY_PATCH",
        "PRE_EXISTING",
        "UNCERTAIN",
    ]
    confidence: int = Field(ge=1, le=5)
    reasoning: str


SYSTEM = """You are reviewing a static-analysis finding produced by Bandit.

Evaluate only the supplied evidence. Do not infer missing repository behavior.

VALIDITY
- VALID_SECURITY_FINDING:
  The shown code plausibly supports the specific security concern.
- LIKELY_FALSE_POSITIVE:
  The rule fires, but the shown context does not plausibly represent the
  security problem claimed.
- UNCERTAIN:
  The supplied evidence is insufficient.

PATCH ATTRIBUTION
- INTRODUCED_OR_AFFECTED_BY_PATCH:
  The diff created or materially changed the condition that causes the finding.
- PRE_EXISTING:
  The condition was already present before and was not materially introduced
  or changed by the patch.
- UNCERTAIN:
  Attribution cannot be established from the evidence.

Important:
1. Validity and patch attribution are separate judgments.
2. A Bandit alert is evidence to inspect, not proof of vulnerability.
3. Test code, hashing used for non-security purposes, controlled deserialization,
   assertions, RawSQL, mark_safe, etc. must be judged from the concrete context,
   not from the rule name alone.
4. If the mapping says ADDED_IN_PATCH, the BEFORE context is an insertion anchor.
5. Keep reasoning short and tied directly to code/diff evidence.
"""

JSON_INSTRUCTION = """
Return a JSON object with exactly these keys:
- validity: VALID_SECURITY_FINDING | LIKELY_FALSE_POSITIVE | UNCERTAIN
- patch_attribution: INTRODUCED_OR_AFFECTED_BY_PATCH | PRE_EXISTING | UNCERTAIN
- confidence: integer from 1 to 5
- reasoning: short string tied to the evidence
"""


def supports_structured_outputs(model: str) -> bool:
    name = model.strip().lower()
    if name in {"gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"} or name.startswith(
        ("gpt-4-0", "gpt-4-1", "gpt-4-turbo", "gpt-3.5-")
    ):
        return False
    return name.startswith(
        ("gpt-4o", "gpt-4.1", "gpt-4.5", "gpt-5", "o1", "o3", "o4")
    )


def parse_judge_result(content: str) -> JudgeResult | None:
    text = (content or "").strip()
    if not text:
        return None
    try:
        return JudgeResult.model_validate_json(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return JudgeResult.model_validate_json(text[start : end + 1])
        except Exception:
            return None


def user_message(i: dict) -> str:
    return f"""
REVIEW ID: {i.get('review_id_v2', '')}
FILE: {i.get('relative_filename', '')}
AFTER LINE: {i.get('after_line_number', '')}
MAPPED BEFORE LINE: {i.get('before_line_number_mapped', '')}
LINE MAPPING: {i.get('line_mapping_relation', '')}

BANDIT TEST: {i.get('test_id', '')} / {i.get('test_name', '')}
CWE: {i.get('cwe', '')}
BANDIT CLAIM:
{i.get('details', '')}

--- BEFORE ---
{i.get('code_context_before', '')}

--- PATCH / DIFF ---
{i.get('patch_diff_context', '')}

--- AFTER ---
{i.get('code_context_after', '')}
""".strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--outdir", default="langsmith_judge_v2")
    ap.add_argument("--dataset-name", default="")
    ap.add_argument("--experiment-prefix", default="bandit-judge-v2")
    ap.add_argument("--max-concurrency", type=int, default=2)
    args = ap.parse_args()

    if not os.environ.get("LANGSMITH_API_KEY"):
        raise SystemExit("LANGSMITH_API_KEY não definida.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY não definida.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input).fillna("")

    cols = [
        "review_id_v2",
        "relative_filename",
        "after_line_number",
        "before_line_number_mapped",
        "line_mapping_relation",
        "test_id",
        "test_name",
        "cwe",
        "details",
        "code_context_before",
        "patch_diff_context",
        "code_context_after",
        "context_status_v2",
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    dataset_name = args.dataset_name or f"bandit-validation-v2-{ts}"

    ls = Client()
    oai = wrappers.wrap_openai(OpenAI())

    dataset = ls.create_dataset(
        dataset_name=dataset_name,
        description=(
            "Blinded Bandit finding review v2: mapped BEFORE, patch diff and AFTER. "
            "Generator identity, Bandit severity/confidence and outcome metadata excluded."
        ),
    )

    ls.create_examples(
        dataset_id=dataset.id,
        examples=[
            {"inputs": {c: row[c] for c in cols}}
            for _, row in df.iterrows()
        ],
    )

    @traceable(name="bandit_review_item_v2")
    def target(inputs: dict) -> dict:
        # Target passthrough: o objeto sendo avaliado é o finding/contexto.
        return {"review_item": user_message(inputs)}

    use_structured = supports_structured_outputs(args.judge_model)
    if not use_structured:
        print(
            f"Aviso: {args.judge_model} não suporta Structured Outputs; "
            "usando JSON mode."
        )

    def call_judge_model(review_item: str) -> JudgeResult | None:
        last_error = None
        for attempt in range(6):
            try:
                if use_structured:
                    response = oai.beta.chat.completions.parse(
                        model=args.judge_model,
                        messages=[
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": review_item},
                        ],
                        response_format=JudgeResult,
                    )
                    return response.choices[0].message.parsed

                response = oai.chat.completions.create(
                    model=args.judge_model,
                    messages=[
                        {"role": "system", "content": SYSTEM + JSON_INSTRUCTION},
                        {"role": "user", "content": review_item},
                    ],
                    response_format={"type": "json_object"},
                )
                return parse_judge_result(response.choices[0].message.content or "")
            except RateLimitError as exc:
                last_error = exc
                time.sleep(min(2**attempt, 20))
        if last_error is not None:
            raise last_error
        return None

    def judge(inputs: dict, outputs: dict):
        parsed = call_judge_model(outputs["review_item"])

        if parsed is None:
            return [
                {"key": "bandit_validity", "value": "UNCERTAIN"},
                {"key": "patch_attribution", "value": "UNCERTAIN"},
                {"key": "judge_confidence", "score": 1},
            ]

        return [
            {
                "key": "bandit_validity",
                "value": parsed.validity,
                "comment": parsed.reasoning,
            },
            {
                "key": "patch_attribution",
                "value": parsed.patch_attribution,
                "comment": parsed.reasoning,
            },
            {
                "key": "judge_confidence",
                "score": parsed.confidence,
                "comment": parsed.reasoning,
            },
        ]

    results = ls.evaluate(
        target,
        data=dataset_name,
        evaluators=[judge],
        experiment_prefix=args.experiment_prefix,
        max_concurrency=args.max_concurrency,
        blocking=True,
        metadata={
            "judge_model": args.judge_model,
            "protocol": "manual-validation-v2",
            "blinded": True,
        },
    )

    def nested_inputs(inp):
        cur = inp or {}
        for _ in range(4):
            if not isinstance(cur, dict):
                return {}
            if cur.get("review_id_v2"):
                return cur
            nxt = cur.get("inputs")
            if not isinstance(nxt, dict):
                return cur
            cur = nxt
        return cur if isinstance(cur, dict) else {}

    rows = []
    for result in results:
        run = result["run"]
        inp = nested_inputs(run.inputs)
        rec = {
            "review_id_v2": inp.get("review_id_v2", ""),
            "langsmith_run_id": str(run.id),
            "judge_model": args.judge_model,
            "judge_validity": "",
            "judge_patch_attribution": "",
            "judge_confidence": "",
            "judge_reasoning": "",
        }

        for er in result["evaluation_results"]["results"]:
            key = getattr(er, "key", "")
            val = getattr(er, "value", None)
            score = getattr(er, "score", None)
            comment = getattr(er, "comment", None)

            if key == "bandit_validity":
                rec["judge_validity"] = val if val is not None else score
                rec["judge_reasoning"] = comment or ""
            elif key == "patch_attribution":
                rec["judge_patch_attribution"] = val if val is not None else score
            elif key == "judge_confidence":
                rec["judge_confidence"] = score if score is not None else val

        rows.append(rec)

    pd.DataFrame(rows).to_csv(
        outdir / "langsmith_bandit_judge_v2_results.csv",
        index=False,
    )

    protocol = {
        "dataset_name": dataset_name,
        "dataset_id": str(dataset.id),
        "judge_model": args.judge_model,
        "n_items": len(df),
        "blinded_fields": [
            "model",
            "severity",
            "Bandit confidence",
            "is_new_finding",
            "existed_before",
            "LR/RF score",
            "human labels",
        ],
        "primary_reference": "human review",
        "role_of_llm_judge": "complementary agreement/scalability analysis",
    }
    (outdir / "langsmith_judge_v2_protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(protocol, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
