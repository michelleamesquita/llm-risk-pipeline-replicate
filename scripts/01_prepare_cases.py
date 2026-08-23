#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_prepare_cases.py

Prepara casos do SWE-bench mantendo compatibilidade com o prompt original
do projeto e adicionando campos úteis para a análise robusta.

Compatibilidade com 02_prompt_builder.py:
- issue_title
- issue_body
- repo_name
- base_commit
- short_file_list
- test_query

Campos adicionais para análise:
- problem_statement
- gold_patch
- gold_changed_files
- FAIL_TO_PASS
- PASS_TO_PASS
- test_patch
- instance_id
- repo_full_name

IMPORTANTE
----------
O conteúdo de gold_patch NÃO é enviado ao LLM pelo 02_prompt_builder.py.
Somente os caminhos extraídos dele entram em short_file_list, preservando
o desenho experimental original usado no projeto.

Uso:
    python scripts/01_prepare_cases.py --max_cases 2

Todos os casos:
    python scripts/01_prepare_cases.py --max_cases 0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset


# ============================================================
# UTILITÁRIOS
# ============================================================

def first_line(text: str, limit: int = 120) -> str:
    """
    Usa a primeira linha do problem_statement como issue_title.
    """
    line = (text or "").strip().splitlines()[0] if text else ""
    if len(line) > limit:
        return line[:limit] + "…"
    return line


def normalize_path(path: str) -> str:
    """
    Normaliza caminho extraído de unified diff.
    """
    p = str(path or "").strip().replace("\\", "/")

    if p.startswith("a/") or p.startswith("b/"):
        p = p[2:]

    while p.startswith("./"):
        p = p[2:]

    return p.strip("/")


def extract_changed_files_from_patch(patch_text: str) -> List[str]:
    """
    Extrai os caminhos dos arquivos modificados no gold patch.

    Suporta principalmente:
        diff --git a/file.py b/file.py
        --- a/file.py
        +++ b/file.py

    Retorna lista única preservando ordem.
    """
    if not patch_text:
        return []

    found: List[str] = []
    seen = set()

    # Forma mais confiável: diff --git
    for match in re.finditer(
        r"(?m)^diff --git a/(.+?) b/(.+?)$",
        patch_text
    ):
        path = normalize_path(match.group(2))

        if path and path != "dev/null" and path not in seen:
            seen.add(path)
            found.append(path)

    # Fallback para unified diff sem "diff --git"
    if not found:
        for line in patch_text.splitlines():
            if not line.startswith("+++ "):
                continue

            raw = line[4:].strip()

            # Pode haver timestamp depois de TAB
            raw = raw.split("\t", 1)[0]

            if raw == "/dev/null":
                continue

            path = normalize_path(raw)

            if path and path != "dev/null" and path not in seen:
                seen.add(path)
                found.append(path)

    return found


def normalize_test_list(value: Any) -> Any:
    """
    Mantém o tipo original quando possível.
    SWE-bench pode fornecer lista ou string dependendo da versão/dataset.
    """
    if value is None:
        return ""
    return value


def test_query_text(fail_to_pass: Any, pass_to_pass: Any) -> str:
    """
    Campo compatível com o prompt antigo.

    Prioridade:
      FAIL_TO_PASS
      PASS_TO_PASS
    """
    value = fail_to_pass if fail_to_pass else pass_to_pass

    if isinstance(value, list):
        return "\n".join(str(x) for x in value)

    return str(value or "")


def repo_short_name(repo_full: str) -> str:
    """
    Nome curto opcional do repositório.

    Exemplo:
        django/django -> django
    """
    value = str(repo_full or "").strip().rstrip("/")
    return value.split("/")[-1] if value else ""


# ============================================================
# CONSTRUÇÃO DO CASE
# ============================================================

def build_case(row: Dict[str, Any]) -> Dict[str, Any]:
    instance_id = str(row.get("instance_id") or "").strip()
    repo_full = str(row.get("repo") or "").strip()
    base_commit = str(row.get("base_commit") or "").strip()

    problem_statement = str(
        row.get("problem_statement") or ""
    ).strip()

    gold_patch = str(
        row.get("patch") or ""
    )

    fail_to_pass = normalize_test_list(
        row.get("FAIL_TO_PASS")
    )

    pass_to_pass = normalize_test_list(
        row.get("PASS_TO_PASS")
    )

    if not instance_id:
        raise ValueError("instance_id ausente")

    if not repo_full:
        raise ValueError(f"{instance_id}: repo ausente")

    if not base_commit:
        raise ValueError(f"{instance_id}: base_commit ausente")

    if not problem_statement:
        raise ValueError(f"{instance_id}: problem_statement ausente")

    changed_files = extract_changed_files_from_patch(gold_patch)

    # Compatibilidade com o builder antigo:
    short_file_list = "\n".join(changed_files)

    case = {
        # ----------------------------------------------------
        # Campos que o pipeline usa como identificadores
        # ----------------------------------------------------
        "case_id": instance_id,
        "instance_id": instance_id,

        # Mantemos repo_name como no dataset SWE-bench original,
        # pois 02_prompt_builder.py usa esse valor no prompt.
        "repo_name": repo_full,

        # Campo extra para quando for útil trabalhar só com basename
        "repo_short_name": repo_short_name(repo_full),

        "repo_full_name": repo_full,
        "base_commit": base_commit,

        # ----------------------------------------------------
        # Compatibilidade EXATA com 02_prompt_builder.py antigo
        # ----------------------------------------------------
        "issue_title": first_line(problem_statement),
        "issue_body": problem_statement,
        "short_file_list": short_file_list,
        "test_query": test_query_text(
            fail_to_pass,
            pass_to_pass,
        ),

        # ----------------------------------------------------
        # Campos novos para rastreabilidade/análise
        # ----------------------------------------------------
        "problem_statement": problem_statement,

        # Gold patch é preservado para avaliação posterior.
        # NÃO é usado diretamente pelo prompt builder.
        "gold_patch": gold_patch,

        # Apenas caminhos do gold patch.
        # short_file_list é a representação textual usada no prompt.
        "gold_changed_files": changed_files,

        "gold_files_count": len(changed_files),

        "FAIL_TO_PASS": fail_to_pass,
        "PASS_TO_PASS": pass_to_pass,

        "test_patch": row.get("test_patch") or "",
        "hints_text": row.get("hints_text") or "",
        "created_at": row.get("created_at") or "",
        "version": row.get("version") or "",
        "environment_setup_commit": (
            row.get("environment_setup_commit") or ""
        ),
    }

    return case


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Prepara casos SWE-bench para o pipeline LLM/SAST."
    )

    ap.add_argument(
        "--hf_ds",
        default="princeton-nlp/SWE-bench",
        help="Dataset Hugging Face.",
    )

    ap.add_argument(
        "--split",
        default="test",
        help="Split do dataset.",
    )

    ap.add_argument(
        "--out_dir",
        default="runs/cases",
        help="Diretório de saída.",
    )

    ap.add_argument(
        "--max_cases",
        type=int,
        default=0,
        help="Máximo de casos. 0 = todos.",
    )

    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescreve casos já existentes.",
    )

    ap.add_argument(
        "--clean",
        action="store_true",
        help="Remove out_dir antes da geração.",
    )

    args = ap.parse_args()

    if args.max_cases < 0:
        raise SystemExit("--max_cases deve ser >= 0")

    out_dir = Path(args.out_dir)

    if args.clean and out_dir.exists():
        print(f"[clean] removendo {out_dir}")
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PREPARAÇÃO DOS CASOS SWE-BENCH")
    print("=" * 70)
    print(f"Dataset : {args.hf_ds}")
    print(f"Split   : {args.split}")
    print(f"Saída   : {out_dir}")
    print(
        f"Limite  : {args.max_cases}"
        if args.max_cases
        else "Limite  : todos"
    )
    print()

    ds = load_dataset(
        args.hf_ds,
        split=args.split,
    )

    processed = 0
    created = 0
    existing = 0
    invalid = 0

    for row in ds:
        if args.max_cases and processed >= args.max_cases:
            break

        try:
            case = build_case(dict(row))
        except Exception as exc:
            invalid += 1
            print(f"[INVALID] {exc}")
            continue

        processed += 1

        out_path = out_dir / f"{case['case_id']}.json"

        if out_path.exists() and not args.overwrite:
            existing += 1
            print(
                f"[EXISTS] {case['case_id']} "
                f"repo={case['repo_name']}"
            )
            continue

        out_path.write_text(
            json.dumps(
                case,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        created += 1

        print(
            f"[OK] {processed:04d} "
            f"{case['case_id']} | "
            f"repo={case['repo_name']} | "
            f"gold_files={case['gold_files_count']}"
        )

    print()
    print("=" * 70)
    print("RESUMO")
    print("=" * 70)
    print(f"Processados : {processed}")
    print(f"Criados    : {created}")
    print(f"Existentes : {existing}")
    print(f"Inválidos  : {invalid}")
    print(f"Saída      : {out_dir}")

    if processed == 0:
        print("ERRO: nenhum caso válido foi preparado.")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
