#!/usr/bin/env python3
"""
Benchmark do overhead operacional pré-SAST.

Mede o caminho crítico APÓS o LLM gerar o patch:

    patch já gerado
       -> ler patch (opcionalmente incluído)
       -> extrair patch structure
       -> montar vetor final de 33 features
          (prompt/problem-statement/model são tratados como pré-computados,
           pois já estão disponíveis antes/de durante a geração)
       -> imputação + StandardScaler + LogisticRegression.predict_proba

Produz duas medidas principais:

1. online_pre_sast_compute_ms
   Patch já está em memória. Mede parse do patch + montagem das 33 features + LR.

2. online_pre_sast_with_patch_read_ms
   Inclui também leitura do arquivo .patch do disco.

Também valida se o parser usado no benchmark reproduz as features de patch
congeladas no CSV. Isso é importante antes de publicar os tempos.

O treinamento da LR NÃO entra no cronômetro. Ele ocorre uma vez antes do benchmark.

Uso:
  python benchmark_pre_sast_feature_extraction.py \
    --features case_model_problem_statement_features.csv \
    --runs-root /Users/mac/Downloads/llm_risk_pipeline_replicate/runs_backup \
    --repeats 3000 \
    --outdir runtime_pre_sast_final

Saídas:
  patch_feature_validation.csv
  patch_feature_validation_summary.csv
  pre_sast_runtime_raw.csv
  pre_sast_runtime_summary.csv
  pre_sast_runtime_config.json

Interpretação:
- Os 10 problem-statement features, 6 prompt-envelope features e identidade
  do modelo podem ser computados/armazenados antes do patch existir.
- Portanto, o overhead pós-geração relevante é dominado por:
      patch parsing + feature-vector assembly + LR scoring.
- `online_pre_sast_with_patch_read_ms` é uma estimativa mais conservadora
  quando o patch precisa ser lido do filesystem.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


TARGET = "has_finding_after"

PS_FEATURES = [
    "ps_domain_explicit_security",
    "ps_security_relevant_domain_count",
    "ps_domain_database",
    "ps_domain_command_exec",
    "ps_domain_auth",
    "ps_domain_permissions",
    "ps_identifier_density",
    "ps_numeric_density",
    "ps_bullet_count",
    "ps_constraint_count",
]

PROMPT_BASE = ["prompt_chars", "prompt_lines", "prompt_tokens"]
PROMPT_DERIVED = [
    "prompt_density",
    "prompt_token_density",
    "prompt_size_category",
]

PATCH_BASE = [
    "patch_lines",
    "patch_added",
    "patch_removed",
    "patch_files_touched",
    "patch_hunks",
    "patch_churn",
    "patch_net",
]

PATCH_DERIVED = [
    "patch_density",
    "add_remove_ratio",
    "net_per_line",
    "hunks_per_file",
    "patch_complexity",
    "change_intensity",
]

MODEL_FEATURES = [
    "model_claude",
    "model_codellama-tuned",
    "model_deepseek",
    "model_gpt-4o",
]

ALL_FEATURES = (
    PROMPT_BASE
    + PROMPT_DERIVED
    + PATCH_BASE
    + PATCH_DERIVED
    + PS_FEATURES
    + MODEL_FEATURES
)


def prompt_size_category(chars: float) -> float:
    if chars <= 500:
        return 0.0
    if chars <= 1000:
        return 1.0
    if chars <= 2000:
        return 2.0
    return 3.0


def add_derived_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["prompt_density"] = out["prompt_chars"] / (out["prompt_lines"] + 1.0)
    out["prompt_token_density"] = (
        out["prompt_tokens"] / (out["prompt_chars"] + 1.0)
    )
    out["prompt_size_category"] = out["prompt_chars"].map(prompt_size_category)

    out["patch_density"] = out["patch_churn"] / (out["patch_lines"] + 1.0)
    out["add_remove_ratio"] = out["patch_added"] / (out["patch_removed"] + 1.0)
    out["net_per_line"] = out["patch_net"] / (out["patch_lines"] + 1.0)
    out["hunks_per_file"] = (
        out["patch_hunks"] / (out["patch_files_touched"] + 1.0)
    )
    out["patch_complexity"] = (
        out["patch_hunks"] * out["patch_files_touched"]
    )
    out["change_intensity"] = (
        out["patch_churn"] / (out["patch_files_touched"] + 1.0)
    )
    return out.replace([np.inf, -np.inf], np.nan)


def full_training_matrix(df: pd.DataFrame):
    work = add_derived_dataframe(df)

    for model in ["claude", "codellama-tuned", "deepseek", "gpt-4o"]:
        work[f"model_{model}"] = (
            work["model"].astype(str).eq(model).astype(float)
        )

    missing = [c for c in ALL_FEATURES + [TARGET] if c not in work.columns]
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")

    X = work[ALL_FEATURES].astype(float)
    y = work[TARGET].astype(int).to_numpy()
    return X, y


def fit_timing_model(df: pd.DataFrame):
    """
    Ajuste apenas para obter um pipeline real já carregado.
    O tempo de treino não é incluído.
    """
    X, y = full_training_matrix(df)

    imputer = SimpleImputer(strategy="median")
    Xi = imputer.fit_transform(X.to_numpy(dtype=float))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xi)

    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        solver="lbfgs",
        C=1.0,
        random_state=42,
    )
    lr.fit(Xs, y)
    return imputer, scaler, lr


def find_patch_file(backup: Path, case: str) -> Path | None:
    root = backup / "patches"
    if not root.exists():
        return None

    # Primeiro: nome/caminho contendo case.
    candidates = [
        p for p in root.rglob("*")
        if p.is_file() and case in str(p)
    ]
    if candidates:
        candidates.sort(
            key=lambda p: (
                0 if p.suffix.lower() in {".patch", ".diff"} else 1,
                len(p.parts),
                len(p.name),
            )
        )
        return candidates[0]

    # Fallback: conteúdo.
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if case in txt and ("diff --git " in txt or "@@ -" in txt):
            return p

    return None


def extract_patch_features(text: str) -> dict[str, float]:
    """
    Parser barato de unified diff.

    Observação:
    `patch_lines` é candidato = número total de linhas do patch.
    O script valida essa definição contra o CSV congelado antes de usarmos
    o benchmark no artigo.
    """
    lines = text.splitlines()

    added = 0
    removed = 0
    hunks = 0
    files = set()

    for line in lines:
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                b = parts[3]
                if b.startswith("b/"):
                    b = b[2:]
                files.add(b)
        elif line.startswith("+++ b/"):
            files.add(line[6:].strip())

        if line.startswith("@@"):
            hunks += 1
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1

    churn = added + removed
    net = added - removed

    return {
        "patch_lines": float(len(lines)),
        "patch_added": float(added),
        "patch_removed": float(removed),
        "patch_files_touched": float(len(files)),
        "patch_hunks": float(hunks),
        "patch_churn": float(churn),
        "patch_net": float(net),
    }


def assemble_33_feature_vector(row: pd.Series, patch: dict[str, float]) -> np.ndarray:
    prompt_chars = float(row["prompt_chars"])
    prompt_lines = float(row["prompt_lines"])
    prompt_tokens = float(row["prompt_tokens"])

    patch_lines = patch["patch_lines"]
    patch_added = patch["patch_added"]
    patch_removed = patch["patch_removed"]
    patch_files = patch["patch_files_touched"]
    patch_hunks = patch["patch_hunks"]
    patch_churn = patch["patch_churn"]
    patch_net = patch["patch_net"]

    data = {
        # prompt base
        "prompt_chars": prompt_chars,
        "prompt_lines": prompt_lines,
        "prompt_tokens": prompt_tokens,

        # prompt derived
        "prompt_density": prompt_chars / (prompt_lines + 1.0),
        "prompt_token_density": prompt_tokens / (prompt_chars + 1.0),
        "prompt_size_category": prompt_size_category(prompt_chars),

        # patch base
        **patch,

        # patch derived
        "patch_density": patch_churn / (patch_lines + 1.0),
        "add_remove_ratio": patch_added / (patch_removed + 1.0),
        "net_per_line": patch_net / (patch_lines + 1.0),
        "hunks_per_file": patch_hunks / (patch_files + 1.0),
        "patch_complexity": patch_hunks * patch_files,
        "change_intensity": patch_churn / (patch_files + 1.0),
    }

    for c in PS_FEATURES:
        data[c] = float(row[c])

    model = str(row["model"])
    for c in MODEL_FEATURES:
        expected = c.removeprefix("model_")
        data[c] = 1.0 if model == expected else 0.0

    return np.asarray([data[c] for c in ALL_FEATURES], dtype=float).reshape(1, -1)


def summary(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            "n": 0, "mean_ms": np.nan, "median_ms": np.nan,
            "q1_ms": np.nan, "q3_ms": np.nan, "iqr_ms": np.nan,
            "p95_ms": np.nan, "min_ms": np.nan, "max_ms": np.nan,
        }
    q1, q3 = np.quantile(x, [.25, .75])
    return {
        "n": int(len(x)),
        "mean_ms": float(np.mean(x)),
        "median_ms": float(np.median(x)),
        "q1_ms": float(q1),
        "q3_ms": float(q3),
        "iqr_ms": float(q3 - q1),
        "p95_ms": float(np.quantile(x, .95)),
        "min_ms": float(np.min(x)),
        "max_ms": float(np.max(x)),
    }


def validate_patch_features(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, r in manifest.iterrows():
        text = Path(r["patch_path"]).read_text(
            encoding="utf-8", errors="replace"
        )
        got = extract_patch_features(text)
        rec = {
            "case": r["case"],
            "model": r["model"],
            "patch_path": r["patch_path"],
        }
        for feature in PATCH_BASE:
            frozen = float(r[feature])
            candidate = float(got[feature])
            rec[f"{feature}_frozen"] = frozen
            rec[f"{feature}_candidate"] = candidate
            rec[f"{feature}_match"] = bool(
                math.isclose(frozen, candidate, rel_tol=0, abs_tol=1e-9)
            )
        rows.append(rec)

    detail = pd.DataFrame(rows)
    sum_rows = []
    for feature in PATCH_BASE:
        col = f"{feature}_match"
        sum_rows.append({
            "feature": feature,
            "n": int(len(detail)),
            "n_exact_match": int(detail[col].sum()),
            "exact_match_rate": float(detail[col].mean()) if len(detail) else np.nan,
        })
    return detail, pd.DataFrame(sum_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--repeats", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="runtime_pre_sast_final")
    ap.add_argument(
        "--max-artifacts",
        type=int,
        default=0,
        help="0 = usa todos os patches encontrados; >0 limita para debug.",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    required = (
        ["case", "model", "backup_dir", TARGET]
        + PROMPT_BASE + PATCH_BASE + PS_FEATURES
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")

    runs_root = Path(args.runs_root).expanduser().resolve()

    # Resolve caminhos UMA VEZ, fora do benchmark.
    manifest_rows = []
    for _, r in (
        df[required]
        .drop_duplicates(subset=["case", "model"])
        .iterrows()
    ):
        backup = runs_root / str(r["backup_dir"])
        patch = find_patch_file(backup, str(r["case"]))
        if patch is None:
            continue
        rec = r.to_dict()
        rec["patch_path"] = str(patch)
        rec["patch_size_bytes"] = patch.stat().st_size
        manifest_rows.append(rec)

    manifest = pd.DataFrame(manifest_rows)

    if manifest.empty:
        raise SystemExit(
            "Nenhum patch encontrado. Confira --runs-root e a estrutura <backup>/patches/."
        )

    if args.max_artifacts and len(manifest) > args.max_artifacts:
        manifest = manifest.sample(
            n=args.max_artifacts,
            random_state=args.seed,
        ).reset_index(drop=True)

    manifest.to_csv(
        outdir / "pre_sast_runtime_manifest.csv",
        index=False,
    )

    # Validação das definições.
    validation, validation_summary = validate_patch_features(manifest)
    validation.to_csv(
        outdir / "patch_feature_validation.csv",
        index=False,
    )
    validation_summary.to_csv(
        outdir / "patch_feature_validation_summary.csv",
        index=False,
    )

    print("\n" + "=" * 100)
    print("VALIDAÇÃO DO PARSER DE PATCH")
    print("=" * 100)
    print(validation_summary.to_string(index=False))

    # Carrega modelo real, fora do tempo.
    imputer, scaler, lr = fit_timing_model(df)

    # Pré-carrega textos para a variante compute-only.
    items = []
    for _, r in manifest.iterrows():
        p = Path(r["patch_path"])
        items.append((r, p, p.read_text(encoding="utf-8", errors="replace")))

    rng = np.random.default_rng(args.seed)

    # Warm-up.
    wr, wp, wt = items[int(rng.integers(0, len(items)))]
    wf = extract_patch_features(wt)
    wx = assemble_33_feature_vector(wr, wf)
    wi = imputer.transform(wx)
    ws = scaler.transform(wi)
    _ = lr.predict_proba(ws)[0, 1]

    raw_rows = []

    for rep in range(1, args.repeats + 1):
        row, path, cached_text = items[int(rng.integers(0, len(items)))]

        # -------------------------
        # COMPUTE-ONLY: patch já em memória
        # -------------------------
        t0 = time.perf_counter_ns()

        t_parse0 = time.perf_counter_ns()
        patch_features = extract_patch_features(cached_text)
        parse_ms = (time.perf_counter_ns() - t_parse0) / 1_000_000.0

        t_vec0 = time.perf_counter_ns()
        x = assemble_33_feature_vector(row, patch_features)
        vector_ms = (time.perf_counter_ns() - t_vec0) / 1_000_000.0

        t_lr0 = time.perf_counter_ns()
        xi = imputer.transform(x)
        xs = scaler.transform(xi)
        prob = lr.predict_proba(xs)[0, 1]
        lr_ms = (time.perf_counter_ns() - t_lr0) / 1_000_000.0

        compute_total_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

        # -------------------------
        # FILE-BASED: inclui leitura do patch
        # -------------------------
        td0 = time.perf_counter_ns()

        tr0 = time.perf_counter_ns()
        disk_text = path.read_text(encoding="utf-8", errors="replace")
        patch_read_ms = (time.perf_counter_ns() - tr0) / 1_000_000.0

        tp20 = time.perf_counter_ns()
        disk_patch_features = extract_patch_features(disk_text)
        disk_parse_ms = (time.perf_counter_ns() - tp20) / 1_000_000.0

        tv20 = time.perf_counter_ns()
        disk_x = assemble_33_feature_vector(row, disk_patch_features)
        disk_vector_ms = (time.perf_counter_ns() - tv20) / 1_000_000.0

        tl20 = time.perf_counter_ns()
        disk_xi = imputer.transform(disk_x)
        disk_xs = scaler.transform(disk_xi)
        disk_prob = lr.predict_proba(disk_xs)[0, 1]
        disk_lr_ms = (time.perf_counter_ns() - tl20) / 1_000_000.0

        with_read_total_ms = (
            time.perf_counter_ns() - td0
        ) / 1_000_000.0

        raw_rows.append({
            "repeat": rep,
            "case": row["case"],
            "model": row["model"],
            "patch_size_bytes": int(path.stat().st_size),

            "patch_parse_compute_ms": parse_ms,
            "vector_assembly_ms": vector_ms,
            "lr_transform_plus_score_ms": lr_ms,
            "online_pre_sast_compute_ms": compute_total_ms,

            "patch_read_ms": patch_read_ms,
            "patch_parse_after_read_ms": disk_parse_ms,
            "vector_assembly_after_read_ms": disk_vector_ms,
            "lr_after_read_ms": disk_lr_ms,
            "online_pre_sast_with_patch_read_ms": with_read_total_ms,

            "score_compute_only": float(prob),
            "score_with_read": float(disk_prob),
        })

    raw = pd.DataFrame(raw_rows)
    raw.to_csv(
        outdir / "pre_sast_runtime_raw.csv",
        index=False,
    )

    metric_cols = [
        "patch_parse_compute_ms",
        "vector_assembly_ms",
        "lr_transform_plus_score_ms",
        "online_pre_sast_compute_ms",
        "patch_read_ms",
        "online_pre_sast_with_patch_read_ms",
    ]

    summary_rows = []
    for metric in metric_cols:
        rec = {"component": metric}
        rec.update(summary(raw[metric]))
        summary_rows.append(rec)

    out_summary = pd.DataFrame(summary_rows)
    out_summary.to_csv(
        outdir / "pre_sast_runtime_summary.csv",
        index=False,
    )

    config = {
        "n_artifacts_found": int(len(manifest)),
        "benchmark_repeats": int(args.repeats),
        "seed": int(args.seed),
        "training_time_included": False,
        "git_time_included": False,
        "bandit_time_included": False,
        "compute_only_definition": (
            "patch text already in memory -> patch parse -> assemble 33-feature "
            "vector -> trained imputer/scaler/LR score"
        ),
        "with_patch_read_definition": (
            "local patch file read -> patch parse -> assemble 33-feature vector "
            "-> trained imputer/scaler/LR score"
        ),
        "pre_generation_cached_features": (
            "problem-statement features, prompt-envelope base information, "
            "and model identity"
        ),
        "note": (
            "Before publication, inspect patch_feature_validation_summary.csv. "
            "If frozen patch features are not reproduced closely/exactly, align "
            "the parser definition with the original extractor and rerun."
        ),
    }
    (outdir / "pre_sast_runtime_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("OVERHEAD PRÉ-SAST")
    print("=" * 100)
    print(out_summary.to_string(index=False))

    print("\n" + "=" * 100)
    print("CONFIGURAÇÃO")
    print("=" * 100)
    print(json.dumps(config, indent=2, ensure_ascii=False))

    print("\nArquivos salvos em:", outdir.resolve())


if __name__ == "__main__":
    main()
