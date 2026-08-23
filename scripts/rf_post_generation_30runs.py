#!/usr/bin/env python3
"""
Random Forest pós-geração / pré-SAST com 30 splits agrupados por case.

Objetivo
--------
Reaproveitar a estratégia antiga de feature engineering do patch/prompt, mas:
  * unidade de análise = case x model (1 linha por par);
  * split = GroupShuffleSplit por case (evita o mesmo case em treino e teste);
  * 30 execuções determinísticas (seeds 42..71 por padrão);
  * SEM temperature/top_p/top_k;
  * SEM CWE, severity, findings_after, risk_after ou qualquer saída do SAST como feature;
  * importância agregada em 30 runs;
  * permutation importance no TESTE;
  * ablação por grupos de features;
  * comparação de conjuntos: model-only, prompt-only, patch-only, PS-only etc.

Entrada recomendada
-------------------
problem_statement_ablation_results/case_model_problem_statement_features.csv

Saídas
------
rf_post_generation_metrics_30runs.csv
rf_post_generation_predictions.csv
rf_post_generation_feature_importance_all_runs.csv
rf_post_generation_feature_importance.csv
rf_post_generation_permutation_all_runs.csv
rf_post_generation_permutation_importance.csv
rf_post_generation_group_ablation_all_runs.csv
rf_post_generation_group_ablation.csv
rf_post_generation_feature_sets_all_runs.csv
rf_post_generation_feature_sets_summary.csv
rf_post_generation_summary.json
fig_rf_post_generation_feature_sets.png
fig_rf_post_generation_permutation_importance.png
fig_rf_post_generation_group_ablation.png
fig_rf_post_generation_confusion_matrix.png

Exemplo
-------
python scripts/rf_post_generation_30runs.py \
    --input problem_statement_ablation_results/case_model_problem_statement_features.csv \
    --outdir results_rf_post_generation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit


# -----------------------------------------------------------------------------
# Protocolo congelado
# -----------------------------------------------------------------------------
BASE_SEED = 42
N_RUNS = 30
TEST_SIZE = 0.20
TARGET = "has_finding_after"
GROUP_COL = "case"
MODEL_COL = "model"

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "n_jobs": -1,
}

# 10 features do problem statement já usadas/congeladas nos experimentos finais.
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

# Envelope do prompt. São observáveis antes de executar o SAST.
PROMPT_BASE_FEATURES = [
    "prompt_chars",
    "prompt_lines",
    "prompt_tokens",
]

# Estrutura do patch gerado. São observáveis depois da geração, antes do SAST.
PATCH_BASE_FEATURES = [
    "patch_lines",
    "patch_added",
    "patch_removed",
    "patch_files_touched",
    "patch_hunks",
    "patch_churn",
    "patch_net",
]

# Nomes das features derivadas reaproveitadas da estratégia antiga.
PROMPT_DERIVED_FEATURES = [
    "prompt_density",
    "prompt_token_density",
    "prompt_size_category",
]

PATCH_DERIVED_FEATURES = [
    "patch_density",
    "add_remove_ratio",
    "net_per_line",
    "hunks_per_file",
    "patch_complexity",
    "change_intensity",
]

# Nunca permitir essas colunas como preditores.
FORBIDDEN_FEATURE_NAMES = {
    "temperature",
    "top_p",
    "top_k",
    "is_risky",
    "findings_before",
    "findings_after",
    "findings_new",
    "findings_resolved_est",
    "high_before",
    "high_after",
    "high_new",
    "delta_high",
    "risk_before",
    "risk_after",
    "delta_risk",
    "risk_introduced",
    "has_high_after",
    "has_new_high",
    "has_new_finding",
    "has_new_high",
    "cwe",
    "severity",
    "confidence",
    "cwe_prevalence_overall",
    "cwe_severity_score",
    "cwe_weighted_severity",
}


def prompt_size_category(chars: pd.Series) -> pd.Series:
    """Replica a categorização antiga: 0<=500, 1<=1000, 2<=2000, 3>2000."""
    return pd.cut(
        chars,
        bins=[-np.inf, 500, 1000, 2000, np.inf],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).astype(float)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria apenas features disponíveis antes do SAST."""
    out = df.copy()

    # Prompt
    out["prompt_density"] = out["prompt_chars"] / (out["prompt_lines"] + 1.0)
    out["prompt_token_density"] = out["prompt_tokens"] / (out["prompt_chars"] + 1.0)
    out["prompt_size_category"] = prompt_size_category(out["prompt_chars"])

    # Patch
    out["patch_density"] = out["patch_churn"] / (out["patch_lines"] + 1.0)
    out["add_remove_ratio"] = out["patch_added"] / (out["patch_removed"] + 1.0)
    out["net_per_line"] = out["patch_net"] / (out["patch_lines"] + 1.0)
    out["hunks_per_file"] = out["patch_hunks"] / (out["patch_files_touched"] + 1.0)
    out["patch_complexity"] = out["patch_hunks"] * out["patch_files_touched"]
    out["change_intensity"] = out["patch_churn"] / (out["patch_files_touched"] + 1.0)

    # Limpa infinitos ocasionais sem olhar o target.
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def check_required_columns(df: pd.DataFrame) -> None:
    required = (
        [GROUP_COL, MODEL_COL, TARGET]
        + PS_FEATURES
        + PROMPT_BASE_FEATURES
        + PATCH_BASE_FEATURES
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")


def build_design_matrix(df: pd.DataFrame):
    """Monta X e os grupos de features, sem qualquer feature pós-SAST."""
    work = add_derived_features(df)

    numeric_features = (
        PROMPT_BASE_FEATURES
        + PROMPT_DERIVED_FEATURES
        + PATCH_BASE_FEATURES
        + PATCH_DERIVED_FEATURES
        + PS_FEATURES
    )

    illegal = sorted(set(numeric_features) & FORBIDDEN_FEATURE_NAMES)
    if illegal:
        raise RuntimeError(f"Feature proibida entrou no modelo: {illegal}")

    # O dataset comum tem todos os 4 modelos em cada case. One-hot explícito simplifica
    # a leitura das importâncias e não usa o target.
    model_dummies = pd.get_dummies(
        work[MODEL_COL].astype(str),
        prefix="model",
        dtype=float,
    )

    X_num = work[numeric_features].astype(float).copy()
    X = pd.concat([X_num, model_dummies], axis=1)

    # Imputação MEDIANA será calculada dentro de cada split, não aqui.
    groups = {
        "prompt_envelope": PROMPT_BASE_FEATURES + PROMPT_DERIVED_FEATURES,
        "patch_structure": PATCH_BASE_FEATURES + PATCH_DERIVED_FEATURES,
        "problem_statement": PS_FEATURES,
        "model_identity": model_dummies.columns.tolist(),
    }

    # Conjuntos úteis para reproduzir a tabela comparativa.
    feature_sets = {
        "model_only": groups["model_identity"],
        "prompt_only": groups["prompt_envelope"],
        "patch_only": groups["patch_structure"],
        "problem_statement_only": groups["problem_statement"],
        "model_plus_problem_statement": groups["model_identity"] + groups["problem_statement"],
        "patch_plus_problem_statement": groups["patch_structure"] + groups["problem_statement"],
        "full": X.columns.tolist(),
    }

    return work, X, groups, feature_sets


def fit_train_medians(X_train: pd.DataFrame) -> pd.Series:
    med = X_train.median(numeric_only=True)
    # Se alguma coluna for totalmente vazia no treino, usa zero como fallback explícito.
    return med.fillna(0.0)


def apply_medians(X: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    return X.fillna(medians).fillna(0.0)


def make_rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(random_state=seed, **RF_PARAMS)


def binary_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_1": precision_score(y_true, y_pred, zero_division=0),
        "recall_1": recall_score(y_true, y_pred, zero_division=0),
        "f1_1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }


def summarize_mean_std(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in value_cols:
        rows.append({
            "metric": c,
            "mean": df[c].mean(),
            "std": df[c].std(ddof=1),
            "min": df[c].min(),
            "max": df[c].max(),
        })
    return pd.DataFrame(rows)


def run_experiment(
    df: pd.DataFrame,
    X: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    feature_sets: dict[str, list[str]],
    n_runs: int,
    base_seed: int,
    test_size: float,
    perm_repeats: int,
):
    y = df[TARGET].astype(int).to_numpy()
    case_groups = df[GROUP_COL].astype(str).to_numpy()

    metrics_rows = []
    pred_rows = []
    fi_rows = []
    perm_rows = []
    ablation_rows = []
    feature_set_rows = []

    cm_total = np.zeros((2, 2), dtype=int)

    for run_idx in range(n_runs):
        seed = base_seed + run_idx
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(X, y, groups=case_groups))

        train_cases = set(case_groups[train_idx])
        test_cases = set(case_groups[test_idx])
        overlap = train_cases & test_cases
        if overlap:
            raise RuntimeError(f"Vazamento de case no run {run_idx + 1}: {sorted(overlap)[:5]}")

        X_train_raw = X.iloc[train_idx].copy()
        X_test_raw = X.iloc[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Imputação ajustada SOMENTE no treino.
        medians = fit_train_medians(X_train_raw)
        X_train = apply_medians(X_train_raw, medians)
        X_test = apply_medians(X_test_raw, medians)

        # ------------------------------------------------------------------
        # Modelo completo
        # ------------------------------------------------------------------
        rf = make_rf(seed)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_score = rf.predict_proba(X_test)[:, 1]

        m = binary_metrics(y_test, y_pred, y_score)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cm_total += cm

        metrics_rows.append({
            "run": run_idx + 1,
            "seed": seed,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "train_cases": len(train_cases),
            "test_cases": len(test_cases),
            **m,
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        })

        for pos, idx in enumerate(test_idx):
            pred_rows.append({
                "run": run_idx + 1,
                "seed": seed,
                "row_index": int(idx),
                "case": df.iloc[idx][GROUP_COL],
                "model": df.iloc[idx][MODEL_COL],
                "y_true": int(y_test[pos]),
                "y_pred": int(y_pred[pos]),
                "y_score": float(y_score[pos]),
            })

        # ------------------------------------------------------------------
        # Gini / impurity importance, agregada posteriormente nos 30 runs
        # ------------------------------------------------------------------
        for feat, imp in zip(X.columns, rf.feature_importances_):
            fi_rows.append({
                "run": run_idx + 1,
                "seed": seed,
                "feature": feat,
                "importance": float(imp),
            })

        # ------------------------------------------------------------------
        # Permutation importance no TEST SET
        # ------------------------------------------------------------------
        perm = permutation_importance(
            rf,
            X_test,
            y_test,
            scoring="roc_auc",
            n_repeats=perm_repeats,
            random_state=seed,
            n_jobs=1,  # dataset pequeno: evita overhead de processos do joblib
        )
        for j, feat in enumerate(X.columns):
            perm_rows.append({
                "run": run_idx + 1,
                "seed": seed,
                "feature": feat,
                "importance_mean": float(perm.importances_mean[j]),
                "importance_std_within_run": float(perm.importances_std[j]),
            })

        full_auc = m["roc_auc"]

        # ------------------------------------------------------------------
        # Ablação: remover um GRUPO do modelo completo, mantendo o mesmo split
        # ------------------------------------------------------------------
        for group_name, cols_remove in feature_groups.items():
            keep_cols = [c for c in X.columns if c not in set(cols_remove)]
            rf_reduced = make_rf(seed)
            rf_reduced.fit(X_train[keep_cols], y_train)
            score_reduced = rf_reduced.predict_proba(X_test[keep_cols])[:, 1]
            auc_reduced = roc_auc_score(y_test, score_reduced)
            ablation_rows.append({
                "run": run_idx + 1,
                "seed": seed,
                "group_removed": group_name,
                "auc_full": full_auc,
                "auc_without_group": auc_reduced,
                # positivo = desempenho piorou ao remover => grupo útil
                "auc_loss_when_removed": full_auc - auc_reduced,
            })

        # ------------------------------------------------------------------
        # Modelos com conjuntos de features isolados/combinados
        # ------------------------------------------------------------------
        for set_name, cols in feature_sets.items():
            if set_name == "full":
                # Reaproveita o modelo completo já treinado acima.
                sm = m
            else:
                rf_set = make_rf(seed)
                rf_set.fit(X_train[cols], y_train)
                pred_set = rf_set.predict(X_test[cols])
                score_set = rf_set.predict_proba(X_test[cols])[:, 1]
                sm = binary_metrics(y_test, pred_set, score_set)
            feature_set_rows.append({
                "run": run_idx + 1,
                "seed": seed,
                "feature_set": set_name,
                "n_features": len(cols),
                **sm,
            })

        print(
            f"Run {run_idx + 1:02d}/{n_runs} | seed={seed} | "
            f"cases train/test={len(train_cases)}/{len(test_cases)} | "
            f"AUC(full)={full_auc:.3f}"
        )

    return {
        "metrics": pd.DataFrame(metrics_rows),
        "predictions": pd.DataFrame(pred_rows),
        "feature_importance_all": pd.DataFrame(fi_rows),
        "permutation_all": pd.DataFrame(perm_rows),
        "ablation_all": pd.DataFrame(ablation_rows),
        "feature_sets_all": pd.DataFrame(feature_set_rows),
        "confusion_matrix_total": cm_total,
    }


def aggregate_results(results: dict):
    fi_all = results["feature_importance_all"]
    fi_summary = (
        fi_all.groupby("feature", as_index=False)
        .agg(
            importance_mean=("importance", "mean"),
            importance_std=("importance", "std"),
            importance_min=("importance", "min"),
            importance_max=("importance", "max"),
        )
        .sort_values("importance_mean", ascending=False)
    )

    perm_all = results["permutation_all"]
    perm_summary = (
        perm_all.groupby("feature", as_index=False)
        .agg(
            permutation_mean=("importance_mean", "mean"),
            permutation_std_across_runs=("importance_mean", "std"),
            positive_runs=("importance_mean", lambda s: int((s > 0).sum())),
            negative_runs=("importance_mean", lambda s: int((s < 0).sum())),
        )
        .sort_values("permutation_mean", ascending=False)
    )

    abl = results["ablation_all"]
    abl_summary = (
        abl.groupby("group_removed", as_index=False)
        .agg(
            auc_full_mean=("auc_full", "mean"),
            auc_without_group_mean=("auc_without_group", "mean"),
            auc_loss_mean=("auc_loss_when_removed", "mean"),
            auc_loss_std=("auc_loss_when_removed", "std"),
            positive_runs=("auc_loss_when_removed", lambda s: int((s > 0).sum())),
        )
        .sort_values("auc_loss_mean", ascending=False)
    )

    fs = results["feature_sets_all"]
    fs_summary = (
        fs.groupby("feature_set", as_index=False)
        .agg(
            n_features=("n_features", "first"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            precision_1_mean=("precision_1", "mean"),
            recall_1_mean=("recall_1", "mean"),
            f1_1_mean=("f1_1", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
        )
        .sort_values("roc_auc_mean", ascending=False)
    )

    return fi_summary, perm_summary, abl_summary, fs_summary


def plot_feature_sets(fs_summary: pd.DataFrame, path: Path):
    plot_df = fs_summary.sort_values("roc_auc_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(plot_df))
    ax.barh(
        y,
        plot_df["roc_auc_mean"],
        xerr=plot_df["roc_auc_std"],
        capsize=4,
    )
    ax.axvline(0.5, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["feature_set"])
    ax.set_xlabel("ROC-AUC médio ± DP (30 splits agrupados por case)")
    ax.set_title("Random Forest — comparação de conjuntos de features")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_permutation(perm_summary: pd.DataFrame, path: Path, top_k: int = 15):
    plot_df = perm_summary.head(top_k).sort_values("permutation_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(plot_df))
    ax.barh(
        y,
        plot_df["permutation_mean"],
        xerr=plot_df["permutation_std_across_runs"],
        capsize=4,
    )
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["feature"])
    ax.set_xlabel("Queda média de ROC-AUC ao permutar a feature")
    ax.set_title("Permutation importance no conjunto de teste")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_group_ablation(abl_summary: pd.DataFrame, path: Path):
    plot_df = abl_summary.sort_values("auc_loss_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(plot_df))
    ax.barh(
        y,
        plot_df["auc_loss_mean"],
        xerr=plot_df["auc_loss_std"],
        capsize=4,
    )
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["group_removed"])
    ax.set_xlabel("Perda de ROC-AUC ao remover o grupo (full − sem grupo)")
    ax.set_title("Ablação de grupos — Random Forest")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm: np.ndarray, path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Sem risco", "Risco"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Sem risco", "Risco"])
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusão agregada — 30 test splits")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="problem_statement_ablation_results/case_model_problem_statement_features.csv",
        help="CSV case x model com features do problem statement.",
    )
    ap.add_argument("--outdir", default="results_rf_post_generation")
    ap.add_argument("--n-runs", type=int, default=N_RUNS)
    ap.add_argument("--base-seed", type=int, default=BASE_SEED)
    ap.add_argument("--test-size", type=float, default=TEST_SIZE)
    ap.add_argument(
        "--perm-repeats",
        type=int,
        default=5,
        help="Repetições da permutation importance em cada test split (5 é suficiente e mais rápido).",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    check_required_columns(df)

    # Não filtramos por SAST. Apenas validamos o desenho experimental.
    n_rows = len(df)
    n_cases = df[GROUP_COL].nunique()
    n_models = df[MODEL_COL].nunique()
    positives = int(df[TARGET].sum())
    negatives = int(len(df) - positives)

    print("=" * 80)
    print("RF pós-geração / pré-SAST — 30 splits agrupados por case")
    print("=" * 80)
    print(f"Input: {in_path}")
    print(f"Rows: {n_rows} | Cases: {n_cases} | Models: {n_models}")
    print(f"Target {TARGET}: positive={positives}, negative={negatives}")

    # Verificação forte para o dataset comum esperado.
    counts = df.groupby(GROUP_COL)[MODEL_COL].nunique()
    if counts.nunique() != 1:
        print("[WARN] Nem todos os cases possuem o mesmo número de modelos.")
    else:
        print(f"Cada case possui {int(counts.iloc[0])} modelos.")

    work, X, feature_groups, feature_sets = build_design_matrix(df)
    print(f"Total de features no modelo completo: {X.shape[1]}")
    print("Grupos:")
    for name, cols in feature_groups.items():
        print(f"  - {name}: {len(cols)} features")

    results = run_experiment(
        work,
        X,
        feature_groups,
        feature_sets,
        n_runs=args.n_runs,
        base_seed=args.base_seed,
        test_size=args.test_size,
        perm_repeats=args.perm_repeats,
    )

    fi_summary, perm_summary, abl_summary, fs_summary = aggregate_results(results)

    # ------------------------------------------------------------------
    # Salvar CSVs
    # ------------------------------------------------------------------
    results["metrics"].to_csv(outdir / "rf_post_generation_metrics_30runs.csv", index=False)
    results["predictions"].to_csv(outdir / "rf_post_generation_predictions.csv", index=False)
    results["feature_importance_all"].to_csv(
        outdir / "rf_post_generation_feature_importance_all_runs.csv", index=False
    )
    fi_summary.to_csv(outdir / "rf_post_generation_feature_importance.csv", index=False)
    results["permutation_all"].to_csv(
        outdir / "rf_post_generation_permutation_all_runs.csv", index=False
    )
    perm_summary.to_csv(outdir / "rf_post_generation_permutation_importance.csv", index=False)
    results["ablation_all"].to_csv(
        outdir / "rf_post_generation_group_ablation_all_runs.csv", index=False
    )
    abl_summary.to_csv(outdir / "rf_post_generation_group_ablation.csv", index=False)
    results["feature_sets_all"].to_csv(
        outdir / "rf_post_generation_feature_sets_all_runs.csv", index=False
    )
    fs_summary.to_csv(outdir / "rf_post_generation_feature_sets_summary.csv", index=False)

    # ------------------------------------------------------------------
    # Figuras
    # ------------------------------------------------------------------
    plot_feature_sets(fs_summary, outdir / "fig_rf_post_generation_feature_sets.png")
    plot_permutation(perm_summary, outdir / "fig_rf_post_generation_permutation_importance.png")
    plot_group_ablation(abl_summary, outdir / "fig_rf_post_generation_group_ablation.png")
    plot_confusion(results["confusion_matrix_total"], outdir / "fig_rf_post_generation_confusion_matrix.png")

    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "precision_1",
        "recall_1",
        "f1_1",
        "roc_auc",
        "pr_auc",
    ]
    metric_summary = summarize_mean_std(results["metrics"], metric_cols)
    metric_summary.to_csv(outdir / "rf_post_generation_metrics_summary.csv", index=False)

    summary = {
        "input": str(in_path),
        "rows": n_rows,
        "unique_cases": n_cases,
        "models": sorted(df[MODEL_COL].astype(str).unique().tolist()),
        "positive": positives,
        "negative": negatives,
        "target": TARGET,
        "runs": args.n_runs,
        "base_seed": args.base_seed,
        "seeds": [args.base_seed + i for i in range(args.n_runs)],
        "split": f"{int((1-args.test_size)*100)}/{int(args.test_size*100)} GroupShuffleSplit grouped by case",
        "rf_params": RF_PARAMS,
        "temperature_features_removed": True,
        "sast_output_features_removed": True,
        "feature_groups": feature_groups,
        "full_feature_count": int(X.shape[1]),
        "full_model_metrics": {
            row["metric"]: {"mean": float(row["mean"]), "std": float(row["std"])}
            for _, row in metric_summary.iterrows()
        },
    }
    with open(outdir / "rf_post_generation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("RESULTADOS PRINCIPAIS")
    print("=" * 80)
    print("\nMétricas do modelo completo:")
    print(metric_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nConjuntos de features (ordenados por ROC-AUC):")
    print(
        fs_summary[["feature_set", "n_features", "roc_auc_mean", "roc_auc_std", "pr_auc_mean"]]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nTop 15 — permutation importance (TESTE):")
    print(
        perm_summary.head(15).to_string(index=False, float_format=lambda x: f"{x:.5f}")
    )

    print("\nAblação por grupo:")
    print(abl_summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print(f"\nArquivos salvos em: {outdir.resolve()}")


if __name__ == "__main__":
    main()
