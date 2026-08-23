#!/usr/bin/env python3
"""
Comparação FINAL: Logistic Regression vs Random Forest
pós-geração / pré-SAST, usando EXATAMENTE as mesmas 33 features
e os mesmos 30 GroupShuffleSplit agrupados por case.

Objetivo
--------
Responder de forma justa:

    "Com todas as informações baratas disponíveis DEPOIS que o LLM gera
     o patch, mas ANTES de executar o SAST, qual classificador funciona
     melhor: Logistic Regression ou Random Forest?"

Importante
----------
* Unidade de análise: case x model (1 linha por par).
* Target: has_finding_after.
* O Bandit/SAST fornece SOMENTE o rótulo usado para avaliação.
* Nenhuma saída do SAST entra como feature.
* LR e RF recebem exatamente as mesmas 33 features.
* LR e RF usam exatamente os mesmos 30 splits.
* Todas as linhas do mesmo case ficam juntas em treino OU teste.
* Imputação e scaling são ajustados SOMENTE no treino.
* A LR é escalada com StandardScaler; o RF usa os valores imputados sem scaling.
* temperature/top_p/top_k continuam fora.

Entrada recomendada
-------------------
problem_statement_ablation_results/case_model_problem_statement_features.csv

Saídas principais
-----------------
lr_rf_full_metrics_30runs.csv
lr_rf_full_metrics_summary.csv
lr_rf_full_predictions.csv
lr_rf_full_paired_deltas.csv
lr_rf_full_paired_tests.csv
lr_full_coefficients_all_runs.csv
lr_full_coefficients_summary.csv
lr_rf_full_summary.json

Figuras
-------
fig_lr_rf_full_metrics.png
fig_lr_rf_full_roc_auc_paired.png
fig_lr_full_confusion_matrix.png
fig_rf_full_confusion_matrix.png
fig_lr_full_coefficients.png

Exemplo
-------
python scripts/lr_rf_post_generation_30runs.py \
    --input problem_statement_ablation_results/case_model_problem_statement_features.csv \
    --outdir results_lr_rf_post_generation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import StandardScaler


# =============================================================================
# PROTOCOLO CONGELADO
# =============================================================================

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

LR_PARAMS = {
    "class_weight": "balanced",
    "max_iter": 5000,
    "solver": "lbfgs",
    "C": 1.0,
}

# 10 features congeladas do problem statement.
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

# Envelope do prompt.
PROMPT_BASE_FEATURES = [
    "prompt_chars",
    "prompt_lines",
    "prompt_tokens",
]

PROMPT_DERIVED_FEATURES = [
    "prompt_density",
    "prompt_token_density",
    "prompt_size_category",
]

# Estrutura do patch gerado.
PATCH_BASE_FEATURES = [
    "patch_lines",
    "patch_added",
    "patch_removed",
    "patch_files_touched",
    "patch_hunks",
    "patch_churn",
    "patch_net",
]

PATCH_DERIVED_FEATURES = [
    "patch_density",
    "add_remove_ratio",
    "net_per_line",
    "hunks_per_file",
    "patch_complexity",
    "change_intensity",
]

# Saídas do SAST / parâmetros não comparáveis que jamais podem entrar como features.
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
    "cwe",
    "severity",
    "confidence",
    "cwe_prevalence_overall",
    "cwe_severity_score",
    "cwe_weighted_severity",
}

METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "precision_1",
    "recall_1",
    "f1_1",
    "roc_auc",
    "pr_auc",
]


# =============================================================================
# FEATURES
# =============================================================================

def prompt_size_category(chars: pd.Series) -> pd.Series:
    """0 <=500; 1 <=1000; 2 <=2000; 3 >2000."""
    return pd.cut(
        chars,
        bins=[-np.inf, 500, 1000, 2000, np.inf],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).astype(float)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria somente features disponíveis antes do SAST."""
    out = df.copy()

    # Prompt
    out["prompt_density"] = out["prompt_chars"] / (out["prompt_lines"] + 1.0)
    out["prompt_token_density"] = (
        out["prompt_tokens"] / (out["prompt_chars"] + 1.0)
    )
    out["prompt_size_category"] = prompt_size_category(out["prompt_chars"])

    # Patch
    out["patch_density"] = out["patch_churn"] / (out["patch_lines"] + 1.0)
    out["add_remove_ratio"] = (
        out["patch_added"] / (out["patch_removed"] + 1.0)
    )
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

    # Mantém 4 dummies para reproduzir exatamente o conjunto FULL de 33 features.
    model_dummies = pd.get_dummies(
        work[MODEL_COL].astype(str),
        prefix="model",
        dtype=float,
    )

    X_num = work[numeric_features].astype(float).copy()
    X = pd.concat([X_num, model_dummies], axis=1)

    expected = len(numeric_features) + len(model_dummies.columns)
    if X.shape[1] != expected:
        raise RuntimeError("Contagem inesperada de features.")

    return work, X


# =============================================================================
# PREPROCESSAMENTO DENTRO DE CADA SPLIT
# =============================================================================

def fit_train_medians(X_train: pd.DataFrame) -> pd.Series:
    """Medianas calculadas exclusivamente no treino."""
    med = X_train.median(numeric_only=True)
    return med.fillna(0.0)


def apply_medians(X: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    return X.fillna(medians).fillna(0.0)


def make_rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        random_state=seed,
        **RF_PARAMS,
    )


def make_lr(seed: int) -> LogisticRegression:
    return LogisticRegression(
        random_state=seed,
        **LR_PARAMS,
    )


# =============================================================================
# MÉTRICAS
# =============================================================================

def binary_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_1": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall_1": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }


# =============================================================================
# ESTATÍSTICA PAREADA
# =============================================================================

def bootstrap_ci_mean(
    values: np.ndarray,
    n_boot: int = 20000,
    seed: int = 2026,
    alpha: float = 0.05,
) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    n = len(x)
    stats = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sample = rng.choice(x, size=n, replace=True)
        stats[i] = np.mean(sample)

    return (
        float(np.quantile(stats, alpha / 2)),
        float(np.quantile(stats, 1 - alpha / 2)),
    )


def rank_biserial_paired(deltas: np.ndarray) -> float:
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]

    if len(d) == 0:
        return 0.0

    ranks = rankdata(np.abs(d), method="average")
    w_plus = ranks[d > 0].sum()
    w_minus = ranks[d < 0].sum()
    denom = w_plus + w_minus

    if denom == 0:
        return 0.0

    return float((w_plus - w_minus) / denom)


def wilcoxon_safe(
    deltas: np.ndarray,
    alternative: str = "two-sided",
) -> tuple[float, float]:
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]

    if len(d) == 0 or np.allclose(d, 0):
        return 0.0, 1.0

    result = wilcoxon(
        d,
        zero_method="wilcox",
        correction=False,
        alternative=alternative,
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def benjamini_hochberg(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    q = np.full(len(p), np.nan, dtype=float)

    valid = np.isfinite(p)
    pv = p[valid]
    if len(pv) == 0:
        return q

    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)

    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    restored = np.empty(m, dtype=float)
    restored[order] = adjusted
    q[valid] = restored
    return q


def paired_model_statistics(
    metrics_long: pd.DataFrame,
    n_boot: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compara LR e RF nos MESMOS runs.
    delta = LR - RF.
      delta > 0 => LR melhor
      delta < 0 => RF melhor
    """
    rows = []
    delta_rows = []

    for metric_idx, metric in enumerate(METRIC_COLUMNS):
        pivot = metrics_long.pivot(
            index=["run", "seed"],
            columns="algorithm",
            values=metric,
        )

        if "logistic_regression" not in pivot.columns:
            raise ValueError("Resultados da Logistic Regression ausentes.")
        if "random_forest" not in pivot.columns:
            raise ValueError("Resultados do Random Forest ausentes.")

        pair = pivot[
            ["logistic_regression", "random_forest"]
        ].dropna()

        delta = (
            pair["logistic_regression"]
            - pair["random_forest"]
        ).to_numpy(dtype=float)

        ci_low, ci_high = bootstrap_ci_mean(
            delta,
            n_boot=n_boot,
            seed=bootstrap_seed + metric_idx,
        )

        stat_two, p_two = wilcoxon_safe(
            delta, alternative="two-sided"
        )
        stat_lr_gt, p_lr_gt = wilcoxon_safe(
            delta, alternative="greater"
        )
        stat_rf_gt, p_rf_gt = wilcoxon_safe(
            delta, alternative="less"
        )

        mean_lr = float(pair["logistic_regression"].mean())
        mean_rf = float(pair["random_forest"].mean())
        mean_delta = float(np.mean(delta))

        if mean_delta > 0:
            winner = "logistic_regression"
        elif mean_delta < 0:
            winner = "random_forest"
        else:
            winner = "tie"

        rows.append({
            "metric": metric,
            "n_runs": len(delta),
            "lr_mean": mean_lr,
            "rf_mean": mean_rf,
            "mean_delta_lr_minus_rf": mean_delta,
            "std_delta": float(np.std(delta, ddof=1)),
            "median_delta": float(np.median(delta)),
            "ci95_mean_low": ci_low,
            "ci95_mean_high": ci_high,
            "lr_wins": int((delta > 0).sum()),
            "rf_wins": int((delta < 0).sum()),
            "ties": int((delta == 0).sum()),
            "winner_by_mean": winner,
            "wilcoxon_stat_two_sided": stat_two,
            "p_two_sided": p_two,
            "p_lr_greater": p_lr_gt,
            "p_rf_greater": p_rf_gt,
            "rank_biserial_lr_minus_rf": rank_biserial_paired(delta),
        })

        for (run, seed), d in zip(pair.index, delta):
            delta_rows.append({
                "run": int(run),
                "seed": int(seed),
                "metric": metric,
                "lr_value": float(
                    pair.loc[(run, seed), "logistic_regression"]
                ),
                "rf_value": float(
                    pair.loc[(run, seed), "random_forest"]
                ),
                "delta_lr_minus_rf": float(d),
            })

    stats = pd.DataFrame(rows)
    stats["q_bh_two_sided"] = benjamini_hochberg(
        stats["p_two_sided"]
    )
    stats["q_bh_lr_greater"] = benjamini_hochberg(
        stats["p_lr_greater"]
    )
    stats["q_bh_rf_greater"] = benjamini_hochberg(
        stats["p_rf_greater"]
    )

    return stats, pd.DataFrame(delta_rows)


# =============================================================================
# EXPERIMENTO
# =============================================================================

def run_experiment(
    df: pd.DataFrame,
    X: pd.DataFrame,
    n_runs: int,
    base_seed: int,
    test_size: float,
):
    y = df[TARGET].astype(int).to_numpy()
    case_groups = df[GROUP_COL].astype(str).to_numpy()

    metrics_rows = []
    prediction_rows = []
    lr_coef_rows = []

    cm_lr_total = np.zeros((2, 2), dtype=int)
    cm_rf_total = np.zeros((2, 2), dtype=int)

    for run_idx in range(n_runs):
        run_number = run_idx + 1
        seed = base_seed + run_idx

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=seed,
        )
        train_idx, test_idx = next(
            splitter.split(X, y, groups=case_groups)
        )

        train_cases = set(case_groups[train_idx])
        test_cases = set(case_groups[test_idx])

        overlap = train_cases & test_cases
        if overlap:
            raise RuntimeError(
                f"Vazamento no run {run_number}: "
                f"{sorted(overlap)[:5]}"
            )

        X_train_raw = X.iloc[train_idx].copy()
        X_test_raw = X.iloc[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]

        # -------------------------------------------------------------
        # Imputação fitada APENAS no treino — compartilhada por LR e RF.
        # -------------------------------------------------------------
        medians = fit_train_medians(X_train_raw)
        X_train = apply_medians(X_train_raw, medians)
        X_test = apply_medians(X_test_raw, medians)

        # =============================================================
        # RANDOM FOREST — mesmas 33 features
        # =============================================================
        rf = make_rf(seed)
        rf.fit(X_train, y_train)

        rf_pred = rf.predict(X_test)
        rf_score = rf.predict_proba(X_test)[:, 1]
        rf_metrics = binary_metrics(
            y_test, rf_pred, rf_score
        )
        cm_rf = confusion_matrix(
            y_test, rf_pred, labels=[0, 1]
        )
        cm_rf_total += cm_rf

        metrics_rows.append({
            "run": run_number,
            "seed": seed,
            "algorithm": "random_forest",
            "n_features": X.shape[1],
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "train_cases": len(train_cases),
            "test_cases": len(test_cases),
            **rf_metrics,
            "tn": int(cm_rf[0, 0]),
            "fp": int(cm_rf[0, 1]),
            "fn": int(cm_rf[1, 0]),
            "tp": int(cm_rf[1, 1]),
        })

        # =============================================================
        # LOGISTIC REGRESSION — EXATAMENTE as mesmas 33 features
        # =============================================================
        # Scaling fitado APENAS no treino.
        scaler = StandardScaler()
        X_train_lr = scaler.fit_transform(X_train)
        X_test_lr = scaler.transform(X_test)

        lr = make_lr(seed)
        lr.fit(X_train_lr, y_train)

        lr_pred = lr.predict(X_test_lr)
        lr_score = lr.predict_proba(X_test_lr)[:, 1]
        lr_metrics = binary_metrics(
            y_test, lr_pred, lr_score
        )
        cm_lr = confusion_matrix(
            y_test, lr_pred, labels=[0, 1]
        )
        cm_lr_total += cm_lr

        metrics_rows.append({
            "run": run_number,
            "seed": seed,
            "algorithm": "logistic_regression",
            "n_features": X.shape[1],
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "train_cases": len(train_cases),
            "test_cases": len(test_cases),
            **lr_metrics,
            "tn": int(cm_lr[0, 0]),
            "fp": int(cm_lr[0, 1]),
            "fn": int(cm_lr[1, 0]),
            "tp": int(cm_lr[1, 1]),
        })

        # Coeficientes LR sobre features padronizadas.
        for feature, coef in zip(
            X.columns, lr.coef_[0]
        ):
            lr_coef_rows.append({
                "run": run_number,
                "seed": seed,
                "feature": feature,
                "coefficient": float(coef),
                "abs_coefficient": float(abs(coef)),
            })

        # Predições individuais dos dois algoritmos.
        for pos, idx in enumerate(test_idx):
            base = {
                "run": run_number,
                "seed": seed,
                "row_index": int(idx),
                "case": df.iloc[idx][GROUP_COL],
                "model": df.iloc[idx][MODEL_COL],
                "y_true": int(y_test[pos]),
            }

            prediction_rows.append({
                **base,
                "algorithm": "random_forest",
                "y_pred": int(rf_pred[pos]),
                "y_score": float(rf_score[pos]),
            })
            prediction_rows.append({
                **base,
                "algorithm": "logistic_regression",
                "y_pred": int(lr_pred[pos]),
                "y_score": float(lr_score[pos]),
            })

        print(
            f"Run {run_number:02d}/{n_runs} | seed={seed} | "
            f"cases train/test={len(train_cases)}/{len(test_cases)} | "
            f"AUC LR={lr_metrics['roc_auc']:.3f} | "
            f"AUC RF={rf_metrics['roc_auc']:.3f} | "
            f"Δ(LR-RF)={lr_metrics['roc_auc'] - rf_metrics['roc_auc']:+.3f}"
        )

    return {
        "metrics": pd.DataFrame(metrics_rows),
        "predictions": pd.DataFrame(prediction_rows),
        "lr_coefficients_all": pd.DataFrame(lr_coef_rows),
        "cm_lr_total": cm_lr_total,
        "cm_rf_total": cm_rf_total,
    }


# =============================================================================
# AGREGAÇÕES
# =============================================================================

def summarize_metrics(metrics_long: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for algorithm, sub in metrics_long.groupby("algorithm"):
        for metric in METRIC_COLUMNS:
            rows.append({
                "algorithm": algorithm,
                "metric": metric,
                "mean": float(sub[metric].mean()),
                "std": float(sub[metric].std(ddof=1)),
                "min": float(sub[metric].min()),
                "max": float(sub[metric].max()),
            })

    return pd.DataFrame(rows)


def summarize_lr_coefficients(
    coef_all: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        coef_all.groupby("feature", as_index=False)
        .agg(
            coefficient_mean=("coefficient", "mean"),
            coefficient_std=("coefficient", "std"),
            abs_coefficient_mean=("abs_coefficient", "mean"),
            positive_runs=("coefficient", lambda s: int((s > 0).sum())),
            negative_runs=("coefficient", lambda s: int((s < 0).sum())),
        )
        .sort_values("abs_coefficient_mean", ascending=False)
    )
    return summary


# =============================================================================
# FIGURAS
# =============================================================================

def plot_metric_comparison(
    summary: pd.DataFrame,
    path: Path,
) -> None:
    metric_order = METRIC_COLUMNS

    lr = (
        summary[summary["algorithm"] == "logistic_regression"]
        .set_index("metric")
        .loc[metric_order]
    )
    rf = (
        summary[summary["algorithm"] == "random_forest"]
        .set_index("metric")
        .loc[metric_order]
    )

    x = np.arange(len(metric_order))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x - width / 2,
        lr["mean"],
        width,
        yerr=lr["std"],
        capsize=4,
        label="Logistic Regression",
    )
    ax.bar(
        x + width / 2,
        rf["mean"],
        width,
        yerr=rf["std"],
        capsize=4,
        label="Random Forest",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        metric_order,
        rotation=25,
        ha="right",
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Média ± DP em 30 splits")
    ax.set_title(
        "LR vs RF — mesmas 33 features, mesmos splits agrupados por case"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_paired_roc_auc(
    metrics_long: pd.DataFrame,
    path: Path,
) -> None:
    pivot = metrics_long.pivot(
        index=["run", "seed"],
        columns="algorithm",
        values="roc_auc",
    )

    fig, ax = plt.subplots(figsize=(8, 7))

    for i, (_, row) in enumerate(pivot.iterrows()):
        ax.plot(
            [0, 1],
            [
                row["logistic_regression"],
                row["random_forest"],
            ],
            marker="o",
            linewidth=0.8,
            alpha=0.55,
        )

    means = [
        pivot["logistic_regression"].mean(),
        pivot["random_forest"].mean(),
    ]
    ax.scatter(
        [0, 1],
        means,
        marker="D",
        s=90,
        label="Média",
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["Logistic Regression", "Random Forest"]
    )
    ax.set_ylabel("ROC-AUC")
    ax.set_title(
        "ROC-AUC pareado — mesmos 30 test splits"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(
    cm: np.ndarray,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=12,
            )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Sem finding", "Com finding"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Sem finding", "Com finding"])
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lr_coefficients(
    summary: pd.DataFrame,
    path: Path,
    top_k: int = 15,
) -> None:
    plot_df = (
        summary.head(top_k)
        .sort_values("abs_coefficient_mean", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(plot_df))

    ax.barh(
        y,
        plot_df["abs_coefficient_mean"],
        xerr=plot_df["coefficient_std"],
        capsize=4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["feature"])
    ax.set_xlabel(
        "|coeficiente padronizado| médio nas 30 execuções"
    )
    ax.set_title(
        "Logistic Regression — magnitude dos coeficientes"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        default="problem_statement_ablation_results/case_model_problem_statement_features.csv",
    )
    ap.add_argument(
        "--outdir",
        default="results_lr_rf_post_generation",
    )
    ap.add_argument(
        "--n-runs",
        type=int,
        default=N_RUNS,
    )
    ap.add_argument(
        "--base-seed",
        type=int,
        default=BASE_SEED,
    )
    ap.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
    )
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=20000,
    )
    ap.add_argument(
        "--bootstrap-seed",
        type=int,
        default=2026,
    )

    args = ap.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    check_required_columns(df)

    n_rows = len(df)
    n_cases = df[GROUP_COL].nunique()
    n_models = df[MODEL_COL].nunique()
    positives = int(df[TARGET].sum())
    negatives = int(n_rows - positives)

    print("=" * 88)
    print("LR vs RF — PÓS-GERAÇÃO / PRÉ-SAST")
    print("=" * 88)
    print(f"Input: {input_path}")
    print(
        f"Rows={n_rows} | cases={n_cases} | "
        f"models={n_models}"
    )
    print(
        f"Target={TARGET} | positive={positives} | "
        f"negative={negatives}"
    )

    counts = df.groupby(GROUP_COL)[MODEL_COL].nunique()
    if counts.nunique() == 1:
        print(
            f"Cada case possui {int(counts.iloc[0])} modelos."
        )
    else:
        print(
            "[WARN] Nem todos os cases possuem o mesmo número de modelos."
        )

    work, X = build_design_matrix(df)

    print(f"Features FULL: {X.shape[1]}")
    print("Lista de features:")
    for feature in X.columns:
        print(f"  - {feature}")

    if X.shape[1] != 33:
        print(
            f"[WARN] Esperávamos 33 features, mas foram encontradas "
            f"{X.shape[1]}. Isso pode ocorrer se o dataset não tiver "
            f"exatamente 4 modelos."
        )

    results = run_experiment(
        work,
        X,
        n_runs=args.n_runs,
        base_seed=args.base_seed,
        test_size=args.test_size,
    )

    metrics_summary = summarize_metrics(
        results["metrics"]
    )

    paired_stats, paired_deltas = paired_model_statistics(
        results["metrics"],
        n_boot=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )

    lr_coef_summary = summarize_lr_coefficients(
        results["lr_coefficients_all"]
    )

    # -----------------------------------------------------------------
    # CSVs
    # -----------------------------------------------------------------
    results["metrics"].to_csv(
        outdir / "lr_rf_full_metrics_30runs.csv",
        index=False,
    )

    metrics_summary.to_csv(
        outdir / "lr_rf_full_metrics_summary.csv",
        index=False,
    )

    results["predictions"].to_csv(
        outdir / "lr_rf_full_predictions.csv",
        index=False,
    )

    paired_deltas.to_csv(
        outdir / "lr_rf_full_paired_deltas.csv",
        index=False,
    )

    paired_stats.to_csv(
        outdir / "lr_rf_full_paired_tests.csv",
        index=False,
    )

    results["lr_coefficients_all"].to_csv(
        outdir / "lr_full_coefficients_all_runs.csv",
        index=False,
    )

    lr_coef_summary.to_csv(
        outdir / "lr_full_coefficients_summary.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # FIGURAS
    # -----------------------------------------------------------------
    plot_metric_comparison(
        metrics_summary,
        outdir / "fig_lr_rf_full_metrics.png",
    )

    plot_paired_roc_auc(
        results["metrics"],
        outdir / "fig_lr_rf_full_roc_auc_paired.png",
    )

    plot_confusion(
        results["cm_lr_total"],
        "Logistic Regression — matriz agregada dos 30 test splits",
        outdir / "fig_lr_full_confusion_matrix.png",
    )

    plot_confusion(
        results["cm_rf_total"],
        "Random Forest — matriz agregada dos 30 test splits",
        outdir / "fig_rf_full_confusion_matrix.png",
    )

    plot_lr_coefficients(
        lr_coef_summary,
        outdir / "fig_lr_full_coefficients.png",
    )

    # -----------------------------------------------------------------
    # JSON
    # -----------------------------------------------------------------
    summary_json = {
        "input": str(input_path),
        "rows": n_rows,
        "unique_cases": n_cases,
        "models": sorted(
            df[MODEL_COL].astype(str).unique().tolist()
        ),
        "positive": positives,
        "negative": negatives,
        "target": TARGET,
        "task": (
            "post-generation / pre-SAST prediction of "
            "has_finding_after"
        ),
        "runs": args.n_runs,
        "base_seed": args.base_seed,
        "seeds": [
            args.base_seed + i
            for i in range(args.n_runs)
        ],
        "split": (
            f"{int((1-args.test_size)*100)}/"
            f"{int(args.test_size*100)} "
            "GroupShuffleSplit grouped by case"
        ),
        "full_feature_count": int(X.shape[1]),
        "full_features": X.columns.tolist(),
        "same_features_for_lr_and_rf": True,
        "same_splits_for_lr_and_rf": True,
        "lr_scaling": (
            "StandardScaler fit only on training data"
        ),
        "imputation": (
            "median fit only on training data, shared by LR and RF"
        ),
        "rf_params": RF_PARAMS,
        "lr_params": LR_PARAMS,
        "temperature_top_p_top_k_removed": True,
        "sast_output_features_removed": True,
        "paired_statistics": (
            "Wilcoxon signed-rank; bootstrap 95% CI of mean "
            "paired delta; rank-biserial; BH-FDR across metrics"
        ),
    }

    with open(
        outdir / "lr_rf_full_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary_json,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # -----------------------------------------------------------------
    # CONSOLE
    # -----------------------------------------------------------------
    print("\n" + "=" * 88)
    print("MÉTRICAS — MÉDIA ± DP")
    print("=" * 88)

    display_summary = metrics_summary.pivot(
        index="metric",
        columns="algorithm",
        values=["mean", "std"],
    )
    print(display_summary.round(4).to_string())

    print("\n" + "=" * 88)
    print("TESTE PAREADO — delta = LR − RF")
    print("=" * 88)

    display_cols = [
        "metric",
        "lr_mean",
        "rf_mean",
        "mean_delta_lr_minus_rf",
        "ci95_mean_low",
        "ci95_mean_high",
        "lr_wins",
        "rf_wins",
        "p_two_sided",
        "q_bh_two_sided",
        "rank_biserial_lr_minus_rf",
        "winner_by_mean",
    ]

    print(
        paired_stats[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    # Resultado principal em ROC-AUC
    auc_row = paired_stats[
        paired_stats["metric"] == "roc_auc"
    ].iloc[0]

    print("\n" + "-" * 88)
    print("RESULTADO PRINCIPAL — ROC-AUC")
    print("-" * 88)
    print(
        f"LR mean = {auc_row['lr_mean']:.4f}"
    )
    print(
        f"RF mean = {auc_row['rf_mean']:.4f}"
    )
    print(
        f"Δ(LR-RF) = "
        f"{auc_row['mean_delta_lr_minus_rf']:+.4f}"
    )
    print(
        f"Bootstrap IC95% = "
        f"[{auc_row['ci95_mean_low']:+.4f}, "
        f"{auc_row['ci95_mean_high']:+.4f}]"
    )
    print(
        f"Wilcoxon two-sided p = "
        f"{auc_row['p_two_sided']:.6g}"
    )
    print(
        f"BH-FDR q = "
        f"{auc_row['q_bh_two_sided']:.6g}"
    )
    print(
        f"LR wins = {int(auc_row['lr_wins'])}/"
        f"{int(auc_row['n_runs'])}"
    )
    print(
        f"RF wins = {int(auc_row['rf_wins'])}/"
        f"{int(auc_row['n_runs'])}"
    )
    print(
        f"Rank-biserial = "
        f"{auc_row['rank_biserial_lr_minus_rf']:+.3f}"
    )
    print(
        f"Winner by mean ROC-AUC: "
        f"{auc_row['winner_by_mean']}"
    )

    print("\nArquivos salvos em:")
    print(outdir.resolve())


if __name__ == "__main__":
    main()
