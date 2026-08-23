#!/usr/bin/env python3
"""
Benchmark operacional: LR pré-SAST vs Bandit.

Objetivo
--------
Medir, NA MESMA MÁQUINA:

1) latência de score da Logistic Regression com modelo/scaler já carregados;
2) tempo real do Bandit sobre repositórios patchados/reconstruídos;
3) opcionalmente, combinar com um tempo externo de extração de features.

IMPORTANTE
----------
- O ajuste da LR neste script é SOMENTE para carregar um modelo real para
  benchmark de tempo. Não use os resultados desse ajuste para desempenho.
- O tempo de treinamento NÃO entra na comparação operacional.
- `lr_score_ms` NÃO inclui a extração original das features a partir de
  prompt/problem statement/patch.
- Se você medir essa extração separadamente, passe:
      --feature-extraction-ms <mediana_ms>
  e o script calculará `pre_sast_total_ms = extraction + LR score`.
- O Bandit é executado como no pipeline original:
      bandit -r <repo> -x tests -f json -o <arquivo>
- O primeiro run por repo é warm-up e não entra nas estatísticas.

Descoberta de repos
-------------------
A partir de case_model_problem_statement_features.csv:
  runs_root/<backup_dir>/repos_patched/<case>

Exemplo:
  /Users/mac/Downloads/llm_risk_pipeline_replicate/runs_backup/
      claud-sonnet_backup/repos_patched/django__django-12553/
      deepseek_backup/repos_patched/...
      ...

Uso recomendado:
---------------
python benchmark_operational_pre_sast_vs_bandit.py \
  --features case_model_problem_statement_features.csv \
  --runs-root /Users/mac/Downloads/llm_risk_pipeline_replicate/runs_backup \
  --n-per-model 10 \
  --bandit-repeats 5 \
  --lr-repeats 2000 \
  --outdir runtime_benchmark

Se você já tiver medido a extração completa das features:
python benchmark_operational_pre_sast_vs_bandit.py ... \
  --feature-extraction-ms 3.7
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


TARGET = "has_finding_after"
MODEL_COL = "model"
CASE_COL = "case"

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


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["prompt_density"] = out["prompt_chars"] / (out["prompt_lines"] + 1.0)
    out["prompt_token_density"] = (
        out["prompt_tokens"] / (out["prompt_chars"] + 1.0)
    )
    out["prompt_size_category"] = pd.cut(
        out["prompt_chars"],
        bins=[-np.inf, 500, 1000, 2000, np.inf],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).astype(float)

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


def build_feature_matrix(df: pd.DataFrame):
    work = add_derived(df)

    numeric = (
        PROMPT_BASE
        + PROMPT_DERIVED
        + PATCH_BASE
        + PATCH_DERIVED
        + PS_FEATURES
    )

    missing = [c for c in numeric + [MODEL_COL, TARGET] if c not in work.columns]
    if missing:
        raise ValueError(f"Colunas necessárias ausentes: {missing}")

    dummies = pd.get_dummies(
        work[MODEL_COL].astype(str),
        prefix="model",
        dtype=float,
    )

    # Garante exatamente os quatro modelos do experimento.
    wanted_dummies = [
        "model_claude",
        "model_codellama-tuned",
        "model_deepseek",
        "model_gpt-4o",
    ]
    for col in wanted_dummies:
        if col not in dummies:
            dummies[col] = 0.0
    dummies = dummies[wanted_dummies]

    X = pd.concat(
        [work[numeric].astype(float).reset_index(drop=True),
         dummies.reset_index(drop=True)],
        axis=1,
    )
    y = work[TARGET].astype(int).to_numpy()

    if X.shape[1] != 33:
        raise RuntimeError(f"Esperava 33 features, obtive {X.shape[1]}.")

    return work.reset_index(drop=True), X, y


def percentile_summary(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            "n": 0,
            "mean_ms": np.nan,
            "median_ms": np.nan,
            "q1_ms": np.nan,
            "q3_ms": np.nan,
            "iqr_ms": np.nan,
            "p95_ms": np.nan,
            "min_ms": np.nan,
            "max_ms": np.nan,
        }
    q1 = float(np.quantile(x, .25))
    q3 = float(np.quantile(x, .75))
    return {
        "n": int(len(x)),
        "mean_ms": float(np.mean(x)),
        "median_ms": float(np.median(x)),
        "q1_ms": q1,
        "q3_ms": q3,
        "iqr_ms": q3 - q1,
        "p95_ms": float(np.quantile(x, .95)),
        "min_ms": float(np.min(x)),
        "max_ms": float(np.max(x)),
    }


def fit_loaded_lr(X: pd.DataFrame, y: np.ndarray):
    """
    Treina uma única LR real para benchmark de latência.
    O treinamento fica FORA da região cronometrada.
    """
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


def benchmark_lr_single_patch(
    X: pd.DataFrame,
    imputer,
    scaler,
    lr,
    repeats: int,
    seed: int,
):
    """
    Mede transformações já aprendidas + predict_proba para UMA linha.

    O DataFrame/feature matrix já existe; portanto NÃO inclui extração de
    prompt/patch/problem statement.
    """
    rng = np.random.default_rng(seed)
    X_np = X.to_numpy(dtype=float)

    # Warm-up.
    warm_idx = int(rng.integers(0, len(X_np)))
    row = X_np[warm_idx:warm_idx + 1]
    row_i = imputer.transform(row)
    row_s = scaler.transform(row_i)
    _ = lr.predict_proba(row_s)

    rows = []
    for i in range(repeats):
        idx = int(rng.integers(0, len(X_np)))
        row = X_np[idx:idx + 1]

        t0 = time.perf_counter_ns()
        row_i = imputer.transform(row)
        row_s = scaler.transform(row_i)
        prob = lr.predict_proba(row_s)[0, 1]
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

        rows.append({
            "repeat": i + 1,
            "row_index": idx,
            "lr_score_ms": elapsed_ms,
            "probability": float(prob),
        })

    return pd.DataFrame(rows)


def repo_path_from_row(runs_root: Path, row: pd.Series):
    backup_dir = str(row["backup_dir"])
    case = str(row[CASE_COL])
    return runs_root / backup_dir / "repos_patched" / case


def discover_available_repos(
    work: pd.DataFrame,
    runs_root: Path,
):
    manifest = (
        work[[CASE_COL, MODEL_COL, "backup_dir"]]
        .drop_duplicates()
        .copy()
    )
    manifest["repo_path"] = manifest.apply(
        lambda r: str(repo_path_from_row(runs_root, r)),
        axis=1,
    )
    manifest["repo_exists"] = manifest["repo_path"].map(
        lambda p: Path(p).is_dir()
    )
    return manifest


def stratified_sample(manifest: pd.DataFrame, n_per_model: int, seed: int):
    available = manifest[manifest["repo_exists"]].copy()
    if available.empty:
        return available

    sampled = []
    for i, (model, sub) in enumerate(available.groupby(MODEL_COL)):
        n = min(n_per_model, len(sub))
        sampled.append(
            sub.sample(n=n, random_state=seed + i)
        )
    return pd.concat(sampled, ignore_index=True)


def bandit_cmd(repo_path: str, out_json: str, exclude: str):
    cmd = [
        "bandit",
        "-r",
        repo_path,
    ]
    if exclude:
        cmd += ["-x", exclude]
    cmd += ["-f", "json", "-o", out_json]
    return cmd


def run_bandit_once(repo_path: str, out_json: str, exclude: str, timeout: int):
    cmd = bandit_cmd(repo_path, out_json, exclude)
    t0 = time.perf_counter_ns()
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=False,
            timeout=timeout,
            check=False,
        )
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        ok = p.returncode in (0, 1)
        return elapsed_ms, int(p.returncode), ok, ""
    except subprocess.TimeoutExpired:
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        return elapsed_ms, -999, False, "TIMEOUT"
    except Exception as e:
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        return elapsed_ms, -998, False, type(e).__name__


def benchmark_bandit(
    sampled: pd.DataFrame,
    repeats: int,
    exclude: str,
    timeout: int,
    tmpdir: Path,
):
    rows = []

    for i, row in sampled.iterrows():
        repo = str(row["repo_path"])
        case = str(row[CASE_COL])
        model = str(row[MODEL_COL])

        # Warm-up descartado.
        warm_json = tmpdir / f"warm_{i}.json"
        warm_ms, warm_rc, warm_ok, warm_err = run_bandit_once(
            repo, str(warm_json), exclude, timeout
        )
        try:
            warm_json.unlink(missing_ok=True)
        except Exception:
            pass

        print(
            f"[{i+1:02d}/{len(sampled):02d}] {model} | {case} | "
            f"warm-up={warm_ms:.1f} ms | rc={warm_rc}"
        )

        for rep in range(1, repeats + 1):
            out_json = tmpdir / f"bandit_{i}_{rep}.json"
            elapsed_ms, rc, ok, err = run_bandit_once(
                repo, str(out_json), exclude, timeout
            )

            finding_count = np.nan
            if ok and out_json.exists():
                try:
                    data = json.loads(
                        out_json.read_text(
                            encoding="utf-8",
                            errors="replace",
                        )
                    )
                    finding_count = len(data.get("results", []))
                except Exception:
                    pass

            try:
                out_json.unlink(missing_ok=True)
            except Exception:
                pass

            rows.append({
                "case": case,
                "model": model,
                "backup_dir": row["backup_dir"],
                "repo_path": repo,
                "repeat": rep,
                "bandit_ms": elapsed_ms,
                "bandit_returncode": rc,
                "bandit_ok": bool(ok),
                "bandit_error": err,
                "finding_count": finding_count,
            })

            print(
                f"    run {rep}/{repeats}: "
                f"{elapsed_ms:.1f} ms | rc={rc} | findings={finding_count}"
            )

    return pd.DataFrame(rows)


def summarize_bandit(raw: pd.DataFrame):
    ok = raw[raw["bandit_ok"]].copy()
    if ok.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Primeiro resume repetições por case×model usando a mediana.
    per_repo = (
        ok.groupby(["case", "model", "repo_path"], as_index=False)
        .agg(
            bandit_median_ms=("bandit_ms", "median"),
            bandit_mean_ms=("bandit_ms", "mean"),
            bandit_min_ms=("bandit_ms", "min"),
            bandit_max_ms=("bandit_ms", "max"),
            repeats=("bandit_ms", "size"),
            findings=("finding_count", "median"),
        )
    )

    rows = []
    for scope, sub in [("OVERALL", per_repo)] + [
        (f"MODEL:{m}", g)
        for m, g in per_repo.groupby("model")
    ]:
        stats = percentile_summary(sub["bandit_median_ms"])
        stats["scope"] = scope
        stats["n_repos"] = int(len(sub))
        rows.append(stats)

    summary = pd.DataFrame(rows)
    return per_repo, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--runs-root", default="")
    ap.add_argument("--outdir", default="runtime_benchmark")
    ap.add_argument("--n-per-model", type=int, default=10)
    ap.add_argument("--bandit-repeats", type=int, default=5)
    ap.add_argument("--lr-repeats", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude", default="tests")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument(
        "--feature-extraction-ms",
        type=float,
        default=np.nan,
        help=(
            "Mediana externa, em ms, da extração completa das features "
            "a partir de prompt/problem statement/patch."
        ),
    )
    ap.add_argument(
        "--skip-bandit",
        action="store_true",
        help="Executa apenas benchmark da LR.",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    work, X, y = build_feature_matrix(df)

    # ----------------------------
    # LR single-patch latency
    # ----------------------------
    imputer, scaler, lr = fit_loaded_lr(X, y)

    lr_raw = benchmark_lr_single_patch(
        X,
        imputer,
        scaler,
        lr,
        repeats=args.lr_repeats,
        seed=args.seed,
    )
    lr_raw.to_csv(outdir / "lr_single_patch_latency_raw.csv", index=False)

    lr_stats = percentile_summary(lr_raw["lr_score_ms"])
    lr_summary = pd.DataFrame([{
        "component": "lr_loaded_single_patch_score",
        **lr_stats,
    }])
    lr_summary.to_csv(
        outdir / "lr_single_patch_latency_summary.csv",
        index=False,
    )

    print("\n" + "=" * 92)
    print("LR — LATÊNCIA DE UM PATCH, MODELO JÁ CARREGADO")
    print("=" * 92)
    print(lr_summary.to_string(index=False))

    bandit_summary = pd.DataFrame()
    per_repo = pd.DataFrame()
    manifest = pd.DataFrame()
    sampled = pd.DataFrame()
    bandit_raw = pd.DataFrame()

    # ----------------------------
    # Bandit
    # ----------------------------
    if not args.skip_bandit:
        if not args.runs_root:
            raise SystemExit(
                "--runs-root é obrigatório quando Bandit não é pulado."
            )

        if shutil.which("bandit") is None:
            raise SystemExit(
                "Bandit não encontrado no PATH. Ative o ambiente onde "
                "o pipeline original executava `bandit`."
            )

        runs_root = Path(args.runs_root).expanduser().resolve()

        manifest = discover_available_repos(work, runs_root)
        manifest.to_csv(
            outdir / "runtime_repo_manifest_discovered.csv",
            index=False,
        )

        sampled = stratified_sample(
            manifest,
            n_per_model=args.n_per_model,
            seed=args.seed,
        )
        sampled.to_csv(
            outdir / "runtime_repo_sample.csv",
            index=False,
        )

        print("\nRepos encontrados por modelo:")
        print(
            manifest.groupby(MODEL_COL)["repo_exists"]
            .agg(["sum", "count"])
            .to_string()
        )

        if sampled.empty:
            raise SystemExit(
                "Nenhum repos_patched encontrado. Veja "
                "runtime_repo_manifest_discovered.csv e confira --runs-root."
            )

        with tempfile.TemporaryDirectory(prefix="bandit_runtime_") as td:
            bandit_raw = benchmark_bandit(
                sampled,
                repeats=args.bandit_repeats,
                exclude=args.exclude,
                timeout=args.timeout,
                tmpdir=Path(td),
            )

        bandit_raw.to_csv(
            outdir / "bandit_runtime_raw.csv",
            index=False,
        )

        per_repo, bandit_summary = summarize_bandit(bandit_raw)
        per_repo.to_csv(
            outdir / "bandit_runtime_per_repo.csv",
            index=False,
        )
        bandit_summary.to_csv(
            outdir / "bandit_runtime_summary.csv",
            index=False,
        )

        print("\n" + "=" * 92)
        print("BANDIT — MEDIANA POR REPO, WARM-UP DESCARTADO")
        print("=" * 92)
        print(bandit_summary.to_string(index=False))

    # ----------------------------
    # Comparação
    # ----------------------------
    lr_median = float(lr_summary.loc[0, "median_ms"])
    feature_ms = args.feature_extraction_ms
    pre_sast_total = (
        feature_ms + lr_median
        if np.isfinite(feature_ms)
        else np.nan
    )

    comparison = {
        "lr_loaded_single_patch_median_ms": lr_median,
        "feature_extraction_median_ms_external": (
            float(feature_ms) if np.isfinite(feature_ms) else None
        ),
        "pre_sast_total_median_ms": (
            float(pre_sast_total) if np.isfinite(pre_sast_total) else None
        ),
        "bandit_overall_median_ms": None,
        "bandit_over_lr_score_ratio": None,
        "bandit_over_full_pre_sast_ratio": None,
        "important_interpretation": (
            "bandit_over_lr_score_ratio compara Bandit apenas com o score "
            "da LR já carregada; não é a razão completa do pipeline pré-SAST. "
            "Para a razão operacional completa, forneça o tempo de extração "
            "das features com --feature-extraction-ms."
        ),
    }

    if not bandit_summary.empty:
        overall = bandit_summary[
            bandit_summary["scope"].eq("OVERALL")
        ].iloc[0]
        bmed = float(overall["median_ms"])
        comparison["bandit_overall_median_ms"] = bmed
        comparison["bandit_over_lr_score_ratio"] = (
            bmed / lr_median if lr_median > 0 else None
        )
        if np.isfinite(pre_sast_total) and pre_sast_total > 0:
            comparison["bandit_over_full_pre_sast_ratio"] = (
                bmed / pre_sast_total
            )

    (
        outdir / "runtime_comparison.json"
    ).write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 92)
    print("COMPARAÇÃO")
    print("=" * 92)
    print(json.dumps(comparison, indent=2, ensure_ascii=False))

    print("\nArquivos salvos em:", outdir.resolve())


if __name__ == "__main__":
    main()
