#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
10_build_flat_findings_robust.py

Gera um CSV final compatível com o antigo all_findings_flat.csv, mas:
- contém apenas findings AFTER em arquivos realmente tocados pelo patch;
- marca se o finding já existia no BEFORE;
- marca findings NOVOS;
- calcula is_risky_after e is_risky_new por arquivo;
- preserva as features de patch/prompt/temperature;
- gera também um CSV resumo por case x model.

Matching BEFORE/AFTER
---------------------
Não usa line_number na identidade principal porque o patch pode deslocar linhas.
Fingerprint = arquivo relativo + test_id + CWE + texto normalizado.

Como pode haver findings repetidos com mesmo fingerprint, o matching usa
contagem multiset (Counter): somente ocorrências excedentes no AFTER são novas.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

try:
    import yaml
except Exception:
    yaml = None

try:
    import tiktoken
except Exception:
    tiktoken = None


MODEL_NAME_BY_DIR = {
    "claud-sonnet_backup": "claude",
    "codellama_tuned_backup": "codellama-tuned",
    "deepseek_backup": "deepseek",
    "gpt-4o_backup": "gpt-4o",
    "claude": "claude",
    "codellama-tuned": "codellama-tuned",
    "deepseek": "deepseek",
    "gpt-4o": "gpt-4o",
}

EXPECTED_MODELS = ("gpt-4o", "claude", "deepseek", "codellama-tuned")
SEV_SCORE = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}


def safe_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def norm_relpath(p: str, case: str = "") -> str:
    s = str(p or "").replace("\\", "/")
    while "//" in s:
        s = s.replace("//", "/")
    while s.startswith("./"):
        s = s[2:]

    # O BEFORE é analisado em uma worktree temporária criada por
    # 05_sast_touched_before_after.py:
    #   .../swe_before_<id>/repo/<caminho relativo>
    # Remova essa raiz para que BEFORE e AFTER usem a mesma identidade.
    before_match = re.search(r"/swe_before_[^/]+/repo/(.+)$", s)
    if before_match:
        s = before_match.group(1)

    # Bandit pode devolver caminho absoluto contendo /repos_patched/<case>/
    if case:
        for marker in (f"/repos_patched/{case}/", f"/{case}/"):
            if marker in s:
                s = s.split(marker, 1)[1]

    if s.startswith("a/") or s.startswith("b/"):
        s = s[2:]
    return s.strip("/")


def normalize_text(text: str) -> str:
    x = re.sub(r"\s+", " ", str(text or "").strip().lower())
    # Evita diferenças irrelevantes de números/endereços na descrição.
    x = re.sub(r"\b0x[0-9a-f]+\b", "<hex>", x)
    return x


def get_cwe(it: dict) -> str:
    c = it.get("issue_cwe")
    if isinstance(c, dict):
        cid = c.get("id", "Unknown")
    else:
        cid = "Unknown"
    return f"CWE-{cid}"


def fingerprint(it: dict, case: str) -> str:
    return "|".join([
        norm_relpath(it.get("filename", ""), case),
        str(it.get("test_id", "") or ""),
        get_cwe(it),
        normalize_text(it.get("issue_text", "")),
    ])


def patch_metrics(text: str) -> dict:
    lines = text.splitlines() if text else []
    added = sum(1 for x in lines if x.startswith("+") and not x.startswith("+++"))
    removed = sum(1 for x in lines if x.startswith("-") and not x.startswith("---"))
    files = []
    seen = set()
    for m in re.finditer(r"(?m)^diff --git a/(.+?) b/(.+?)$", text or ""):
        p = norm_relpath(m.group(2))
        if p and p not in seen:
            seen.add(p); files.append(p)
    if not files:
        for x in lines:
            if x.startswith("+++ "):
                p = norm_relpath(x[4:].strip().split("\t", 1)[0])
                if p and p != "dev/null" and p not in seen:
                    seen.add(p); files.append(p)

    hunks = sum(1 for x in lines if x.startswith("@@"))
    churn = added + removed
    return {
        "patch_lines": len(lines),
        "patch_added": added,
        "patch_removed": removed,
        "patch_files_touched": len(files),
        "patch_hunks": hunks,
        "patch_churn": churn,
        "patch_net": added - removed,
        "_touched_files": files,
    }


def prompt_metrics(text: str, model: str) -> dict:
    chars = len(text or "")
    lines = len((text or "").splitlines())
    if not text:
        tokens = 0
    elif tiktoken is not None:
        try:
            if model == "gpt-4o":
                enc = tiktoken.encoding_for_model("gpt-4o")
            else:
                enc = tiktoken.get_encoding("cl100k_base")
            tokens = len(enc.encode(text))
        except Exception:
            tokens = max(1, chars // 4)
    else:
        tokens = max(1, chars // 4) if chars else 0

    lower = (text or "").lower()
    sec_keys = [
        "[security-guidelines]", "owasp", "secure", "deny-list",
        "allow-list", "parameterized", "no eval", "shell=true",
    ]
    return {
        "prompt_chars": chars,
        "prompt_lines": lines,
        "prompt_tokens": tokens,
        "prompt_has_security_guidelines": int(any(k in lower for k in sec_keys)),
    }


def load_profiles(path: Path) -> dict:
    if yaml is None or not path.is_file():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def model_temperature(model: str, profiles: dict):
    """
    Retorna temperature SOMENTE quando ela foi explicitamente configurada
    para o endpoint. Não inventa fallback 0.2.

    No protocolo padronizado atual, temperature não é uma feature experimental
    comparável entre os quatro modelos; portanto normalmente retorna None.
    """
    p = profiles.get(model)
    if not isinstance(p, dict):
        return None

    if "effective_temperature" in p:
        value = p.get("effective_temperature")
        return float(value) if value is not None else None

    if "temperature" in p:
        value = p.get("temperature")
        return float(value) if value is not None else None

    return None


def apply_strategy_flags(meta: dict) -> tuple[int, int]:
    strategy = str(meta.get("apply_strategy") or "")
    strict = bool(meta.get("strict_patch_apply"))
    fuzzy = bool(meta.get("fuzzy_patch_apply"))
    if not strict and not fuzzy:
        strict = strategy.startswith("git apply")
        fuzzy = strategy.startswith("patch ")
    return int(strict), int(fuzzy)


def find_artifact(directory: Path, case: str, suffix: str) -> Optional[Path]:
    candidates = [
        directory / f"{case}{suffix}",
    ]
    for p in candidates:
        if p.is_file():
            return p
    matches = sorted(directory.glob(f"*{case}*{suffix}"))
    return matches[0] if matches else None


def execution_status(
    pipeline_valid: bool,
    has_402: bool,
    has_429: bool,
    generation: Optional[dict],
    meta: Optional[dict],
    generation_success: bool,
    apply_success: bool,
) -> str:
    if pipeline_valid:
        return "pipeline_valid"
    if has_402:
        return "insufficient_credit"
    if has_429:
        return "rate_limited"
    if generation is None and meta is None:
        return "not_attempted"
    if generation_success and not apply_success:
        return "apply_failed"
    if generation and generation.get("error"):
        return "generation_failed"
    return "pipeline_incomplete"


def classify_execution(run_dir: Optional[Path], case: str) -> dict:
    meta = safe_json(run_dir / "metadata" / f"{case}.json") if run_dir else None
    generation = (
        safe_json(run_dir / "generation_metadata" / f"{case}.json")
        if run_dir else None
    )
    generation_text = json.dumps(generation or {}, ensure_ascii=False)
    pipeline_valid = bool(
        meta
        and meta.get("patch_apply_success")
        and meta.get("experiment_valid") is True
    )
    generation_success = bool(generation and generation.get("success"))
    apply_success = bool(
        pipeline_valid
        or (generation and generation.get("apply_success"))
        or (meta and meta.get("patch_apply_success"))
    )
    has_429 = "status: 429" in generation_text
    has_402 = "status: 402" in generation_text

    return {
        "case": case,
        "status": execution_status(
            pipeline_valid,
            has_402,
            has_429,
            generation,
            meta,
            generation_success,
            apply_success,
        ),
        "pipeline_valid": int(pipeline_valid),
        "generation_success": int(generation_success),
        "apply_success": int(apply_success),
        "artifact_http_429": int(has_429),
        "artifact_http_402": int(has_402),
        "end_to_end_attempt_count": (
            generation.get("end_to_end_attempt_count") if generation else None
        ),
    }


def recalculate_cwe_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df
    counts = df["cwe"].value_counts(dropna=False)
    prevalence = (counts / len(df)).to_dict()
    df["cwe_prevalence_overall"] = df["cwe"].map(
        lambda value: float(prevalence.get(value, 0.0))
    )
    df["cwe_severity_score"] = df["severity"].map(
        lambda value: SEV_SCORE.get(str(value).upper(), 0)
    )
    df["cwe_weighted_severity"] = (
        df["cwe_prevalence_overall"] * df["cwe_severity_score"]
    )
    return df


def build(
    root: Path,
    out_csv: Path,
    summary_csv: Path,
    profiles_path: Path,
    common_out_csv: Path,
    common_summary_csv: Path,
    status_csv: Path,
    before_out_csv: Path,
):
    profiles = load_profiles(profiles_path)
    rows = []
    before_rows = []
    summaries = []
    run_dirs_by_model = {}

    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        model = MODEL_NAME_BY_DIR.get(run_dir.name)
        if not model:
            continue
        run_dirs_by_model[model] = run_dir

        reports = run_dir / "reports"
        patches = run_dir / "patches"
        metadata_dir = run_dir / "metadata"
        if not reports.is_dir():
            continue

        for after_path in sorted(reports.glob("*_bandit_after.json")):
            case = after_path.name.replace("_bandit_after.json", "")
            before_path = reports / f"{case}_bandit_before.json"
            meta_path = metadata_dir / f"{case}.json"

            if not before_path.is_file() or not meta_path.is_file():
                print(f"[skip] sem BEFORE/metadata: {model} {case}")
                continue

            meta = safe_json(meta_path) or {}
            if not meta.get("patch_apply_success"):
                print(f"[skip] patch inválido: {model} {case}")
                continue
            if meta.get("experiment_valid") is not True:
                print(f"[skip] pipeline incompleto: {model} {case}")
                continue

            before = safe_json(before_path) or {}
            after = safe_json(after_path) or {}
            before_issues = before.get("results") or before.get("issues") or []
            after_issues = after.get("results") or after.get("issues") or []

            patch_path = find_artifact(patches, case, ".patch")
            prompt_path = find_artifact(patches, case, ".prompt.txt")
            patch_text = patch_path.read_text(encoding="utf-8", errors="ignore") if patch_path else ""
            prompt_text = prompt_path.read_text(encoding="utf-8", errors="ignore") if prompt_path else ""

            pm = patch_metrics(patch_text)
            prm = prompt_metrics(prompt_text, model)
            touched = set(meta.get("actual_changed_files") or pm["_touched_files"])
            touched = {norm_relpath(x, case) for x in touched}

            def only_touched(issues):
                out = []
                for it in issues:
                    rel = norm_relpath(it.get("filename", ""), case)
                    if rel in touched:
                        out.append(it)
                return out

            before_issues = only_touched(before_issues)
            after_issues = only_touched(after_issues)

            before_counter = Counter(fingerprint(x, case) for x in before_issues)
            consumed = Counter()

            new_flags = []
            for it in after_issues:
                fp = fingerprint(it, case)
                if consumed[fp] < before_counter[fp]:
                    is_new = 0
                    consumed[fp] += 1
                else:
                    is_new = 1
                new_flags.append(is_new)

            high_after_by_file = defaultdict(bool)
            high_new_by_file = defaultdict(bool)

            for it, is_new in zip(after_issues, new_flags):
                rel = norm_relpath(it.get("filename", ""), case)
                sev = str(it.get("issue_severity") or "").upper()
                if sev == "HIGH":
                    high_after_by_file[rel] = True
                    if is_new:
                        high_new_by_file[rel] = True

            temp = model_temperature(model, profiles)
            strict_apply, fuzzy_apply = apply_strategy_flags(meta)
            n_before = len(before_issues)
            n_after = len(after_issues)
            n_new = int(sum(new_flags))
            n_high_before = sum(
                str(x.get("issue_severity") or "").upper() == "HIGH"
                for x in before_issues
            )
            n_high_after = sum(
                str(x.get("issue_severity") or "").upper() == "HIGH"
                for x in after_issues
            )
            n_high_new = sum(
                1 for x, flag in zip(after_issues, new_flags)
                if flag and str(x.get("issue_severity") or "").upper() == "HIGH"
            )

            # Risk score simples e auditável: LOW=1, MEDIUM=2, HIGH=3.
            # Mantemos separadamente o risco total antes/depois e o risco
            # introduzido apenas por findings que não existiam no BEFORE.
            risk_before = sum(
                SEV_SCORE.get(str(x.get("issue_severity") or "").upper(), 0)
                for x in before_issues
            )
            risk_after = sum(
                SEV_SCORE.get(str(x.get("issue_severity") or "").upper(), 0)
                for x in after_issues
            )
            risk_introduced = sum(
                SEV_SCORE.get(str(x.get("issue_severity") or "").upper(), 0)
                for x, flag in zip(after_issues, new_flags) if flag
            )

            # Flat BEFORE: uma linha por finding existente no commit base,
            # restrito aos mesmos arquivos efetivamente tocados pelo patch.
            for it in before_issues:
                rel = norm_relpath(it.get("filename", ""), case)
                sev = str(it.get("issue_severity") or "").upper()
                before_rows.append({
                    "model": model,
                    "backup_dir": run_dir.name,
                    "repo": meta.get("repo_name", meta.get("repo", "")),
                    "case": case,
                    "report_file": before_path.name,
                    "filename": it.get("filename", ""),
                    "relative_filename": rel,
                    "line_number": it.get("line_number"),
                    "test_id": it.get("test_id"),
                    "test_name": it.get("test_name"),
                    "cwe": get_cwe(it),
                    "severity": sev,
                    "confidence": str(it.get("issue_confidence") or "").upper(),
                    "details": it.get("issue_text"),
                    "finding_fingerprint": fingerprint(it, case),
                    "severity_score": SEV_SCORE.get(sev, 0),
                    "file_touched_by_patch": 1,
                    **{k: v for k, v in pm.items() if not k.startswith("_")},
                    **prm,
                })

            summaries.append({
                "model": model,
                "backup_dir": run_dir.name,
                "case": case,
                "patch_apply_success": 1,
                "files_touched": len(touched),
                "findings_before": n_before,
                "findings_after": n_after,
                "findings_new": n_new,
                "findings_resolved_est": max(0, n_before - sum(consumed.values())),
                "high_before": n_high_before,
                "high_after": n_high_after,
                "high_new": n_high_new,
                "delta_high": n_high_after - n_high_before,
                "risk_before": risk_before,
                "risk_after": risk_after,
                "delta_risk": risk_after - risk_before,
                "risk_introduced": risk_introduced,
                "has_high_after": int(n_high_after > 0),
                "has_new_high": int(n_high_new > 0),
                **{k: v for k, v in pm.items() if not k.startswith("_")},
                **prm,
                "temperature": temp,
                "decoding_protocol": "common_parameters_only",
                "temperature_comparable": 0,
                "top_p_comparable": 0,
                "top_k_comparable": 0,
                "pipeline_valid": 1,
                "functional_tests_run": int(bool(
                    meta.get("functional_tests_run")
                )),
                "functional_correctness_claimed": int(bool(
                    meta.get("functional_correctness_claimed")
                )),
                "strict_patch_apply": strict_apply,
                "fuzzy_patch_apply": fuzzy_apply,
            })

            for it, is_new in zip(after_issues, new_flags):
                rel = norm_relpath(it.get("filename", ""), case)
                cwe = get_cwe(it)
                sev = str(it.get("issue_severity") or "").upper()
                conf = str(it.get("issue_confidence") or "").upper()
                fp = fingerprint(it, case)

                rows.append({
                    # Mesmas colunas-base do CSV antigo
                    "model": model,
                    "backup_dir": run_dir.name,
                    "repo": meta.get("repo_name", meta.get("repo", "")),
                    "case": case,
                    "report_file": after_path.name,
                    "filename": it.get("filename", ""),
                    "line_number": it.get("line_number"),
                    "test_id": it.get("test_id"),
                    "test_name": it.get("test_name"),
                    "cwe": cwe,
                    "severity": sev,
                    "confidence": conf,
                    "details": it.get("issue_text"),
                    **{k: v for k, v in pm.items() if not k.startswith("_")},
                    **prm,
                    "temperature": temp,
                    "decoding_protocol": "common_parameters_only",
                    "temperature_comparable": 0,
                    "top_p_comparable": 0,
                    "top_k_comparable": 0,
                    "pipeline_valid": 1,
                    "functional_tests_run": int(bool(
                        meta.get("functional_tests_run")
                    )),
                    "functional_correctness_claimed": int(bool(
                        meta.get("functional_correctness_claimed")
                    )),
                    "strict_patch_apply": strict_apply,
                    "fuzzy_patch_apply": fuzzy_apply,

                    # Compatibilidade: risco presente no arquivo após patch
                    "is_risky": int(high_after_by_file[rel]),

                    # NOVOS CAMPOS ROBUSTOS
                    "relative_filename": rel,
                    "file_touched_by_patch": 1,
                    "patch_apply_success": 1,
                    "finding_fingerprint": fp,
                    "existed_before": int(not is_new),
                    "is_new_finding": int(is_new),
                    "is_risky_after": int(high_after_by_file[rel]),
                    "is_risky_new": int(high_new_by_file[rel]),
                    "severity_score": SEV_SCORE.get(sev, 0),
                })

    df = pd.DataFrame(rows)

    old_order = [
        "model", "backup_dir", "repo", "case", "report_file",
        "filename", "line_number", "test_id", "test_name", "cwe",
        "severity", "confidence", "details",
        "patch_lines", "patch_added", "patch_removed",
        "patch_files_touched", "patch_hunks", "patch_churn", "patch_net",
        "prompt_chars", "prompt_lines", "prompt_tokens",
        "prompt_has_security_guidelines", "temperature",
        "is_risky",
        "cwe_prevalence_overall", "cwe_severity_score",
        "cwe_weighted_severity",
    ]
    empty_extras = [
        "decoding_protocol", "temperature_comparable",
        "top_p_comparable", "top_k_comparable", "pipeline_valid",
        "functional_tests_run", "functional_correctness_claimed",
        "strict_patch_apply", "fuzzy_patch_apply", "relative_filename",
        "file_touched_by_patch", "patch_apply_success",
        "finding_fingerprint", "existed_before", "is_new_finding",
        "is_risky_after", "is_risky_new", "severity_score",
    ]

    if not df.empty:
        # Mantém as métricas antigas de CWE, mas calculadas no NOVO dataset filtrado.
        df = recalculate_cwe_metrics(df)

        # Ordem: primeiro o schema antigo; depois campos novos.
        extras = [c for c in df.columns if c not in old_order]
        df = df[[c for c in old_order if c in df.columns] + extras]
    else:
        df = pd.DataFrame(columns=old_order + empty_extras)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    before_df = pd.DataFrame(before_rows)
    before_out_csv.parent.mkdir(parents=True, exist_ok=True)
    before_df.to_csv(before_out_csv, index=False)

    sdf = pd.DataFrame(summaries)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    sdf.to_csv(summary_csv, index=False)

    valid_models_by_case = (
        sdf.groupby("case")["model"].agg(set).to_dict()
        if not sdf.empty
        else {}
    )
    required_models = set(EXPECTED_MODELS)
    common_cases = {
        case
        for case, models in valid_models_by_case.items()
        if required_models.issubset(models)
    }

    common_df = df[df["case"].isin(common_cases)].copy()
    common_df = recalculate_cwe_metrics(common_df)
    common_out_csv.parent.mkdir(parents=True, exist_ok=True)
    common_df.to_csv(common_out_csv, index=False)

    common_sdf = (
        sdf[sdf["case"].isin(common_cases)].copy()
        if not sdf.empty
        else sdf.copy()
    )
    common_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    common_sdf.to_csv(common_summary_csv, index=False)

    attempted_cases = set()
    for run_dir in run_dirs_by_model.values():
        for directory_name in ("generation_metadata", "metadata"):
            directory = run_dir / directory_name
            if directory.is_dir():
                attempted_cases.update(path.stem for path in directory.glob("*.json"))
        patches_dir = run_dir / "patches"
        if patches_dir.is_dir():
            attempted_cases.update(
                path.name.removesuffix(".prompt.txt")
                for path in patches_dir.glob("*.prompt.txt")
            )

    status_rows = []
    for case in sorted(attempted_cases):
        for model in EXPECTED_MODELS:
            run_dir = run_dirs_by_model.get(model)
            status_rows.append({
                "model": model,
                "backup_dir": run_dir.name if run_dir else "",
                **classify_execution(run_dir, case),
            })
    status_df = pd.DataFrame(status_rows)
    status_csv.parent.mkdir(parents=True, exist_ok=True)
    status_df.to_csv(status_csv, index=False)

    valid_counts = (
        sdf.groupby("model")["case"].nunique().to_dict()
        if not sdf.empty
        else {}
    )
    minimum_valid = min(
        (int(valid_counts.get(model, 0)) for model in EXPECTED_MODELS),
        default=0,
    )

    print(f"CSV final: {out_csv}")
    print(f"Linhas AFTER em arquivos tocados: {len(df):,}")
    print(f"Flat BEFORE: {before_out_csv} ({len(before_df):,} linhas)")
    print(f"Resumo case x model: {summary_csv} ({len(sdf):,} linhas)")
    print(f"Menor total válido individual: {minimum_valid:,} casos")
    print(f"Interseção válida nos 4 modelos: {len(common_cases):,} casos")
    print(f"CSV comum: {common_out_csv} ({len(common_df):,} linhas)")
    print(
        f"Resumo comum: {common_summary_csv} "
        f"({len(common_sdf):,} linhas)"
    )
    print(f"Status case x model: {status_csv} ({len(status_df):,} linhas)")
    if len(df):
        print(f"Findings novos: {int(df['is_new_finding'].sum()):,}")
        print(f"Findings HIGH novos: {int(((df['is_new_finding']==1) & (df['severity']=='HIGH')).sum()):,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out-csv", default="all_findings_flat_robust.csv")
    ap.add_argument("--summary-csv", default="case_model_before_after_summary.csv")
    ap.add_argument("--before-out-csv", default="all_findings_before_flat_robust.csv")
    ap.add_argument(
        "--common-out-csv",
        default="all_findings_flat_robust_common.csv",
    )
    ap.add_argument(
        "--common-summary-csv",
        default="case_model_before_after_summary_common.csv",
    )
    ap.add_argument(
        "--status-csv",
        default="case_model_execution_status.csv",
    )
    ap.add_argument("--profiles", default="configs/model_profiles.yaml")
    args = ap.parse_args()

    build(
        Path(args.runs_root),
        Path(args.out_csv),
        Path(args.summary_csv),
        Path(args.profiles),
        Path(args.common_out_csv),
        Path(args.common_summary_csv),
        Path(args.status_csv),
        Path(args.before_out_csv),
    )


if __name__ == "__main__":
    main()