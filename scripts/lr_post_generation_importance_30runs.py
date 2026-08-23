#!/usr/bin/env python3
"""
Interpretabilidade da Logistic Regression principal (30 runs)
=============================================================

Tarefa:
    previsão pós-geração / pré-SAST de has_finding_after.

Este script usa EXATAMENTE a mesma família FULL de 33 features do experimento
LR vs RF e os mesmos 30 GroupShuffleSplit agrupados por case.

Objetivo:
    responder "quais features/grupos carregam sinal na LR principal?"
sem depender apenas dos coeficientes, que podem ser instáveis quando existem
features correlacionadas.

Produz quatro tipos de evidência:

1) Coeficientes padronizados da LR nas 30 execuções
   - média, DP, |coef| médio, consistência de sinal.

2) Permutation importance no TEST SET
   - queda de ROC-AUC ao embaralhar cada feature no conjunto de teste.

3) Ablação de grupos
   - FULL vs FULL sem:
       * problem_statement
       * prompt_envelope
       * model_identity
       * patch_structure

4) Comparação de conjuntos de features
   - full
   - patch_only
   - patch_plus_problem_statement
   - model_plus_problem_statement
   - problem_statement_only
   - model_only
   - prompt_only

Estatística:
    - Wilcoxon signed-rank pareado
    - bootstrap IC95% do delta médio
    - rank-biserial correlation
    - Benjamini-Hochberg FDR

IMPORTANTE:
    * Coeficientes da LR são DESCRITIVOS do modelo regularizado.
    * Como existem features correlacionadas (ex.: prompt_chars/tokens/lines e
      algumas métricas derivadas do patch), NÃO use o ranking de coeficientes
      isoladamente como prova causal ou inferencial.
    * Para o artigo, priorize:
          permutation importance + group ablation + desempenho preditivo.
    * SAST/Bandit fornece somente o rótulo de avaliação.
    * Nenhuma saída do SAST entra como feature.
    * temperature/top_p/top_k ficam fora.

Exemplo:
    python scripts/lr_post_generation_importance_30runs.py \
        --input problem_statement_ablation_results/case_model_problem_statement_features.csv \
        --outdir results_lr_importance \
        --n-runs 30 \
        --permutation-repeats 10 \
        --bootstrap 20000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


# =============================================================================
# PROTOCOLO
# =============================================================================

BASE_SEED = 42
N_RUNS = 30
TEST_SIZE = 0.20

DEFAULT_INPUT = (
    "problem_statement_ablation_results/"
    "case_model_problem_statement_features.csv"
)
INPUT_FALLBACK_DIR = Path("problem_statement_ablation_results")

TARGET = "has_finding_after"
GROUP_COL = "case"
MODEL_COL = "model"

LR_PARAMS = {
    "class_weight": "balanced",
    "max_iter": 5000,
    "solver": "lbfgs",
    "C": 1.0,
}

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

FORBIDDEN = {
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


# =============================================================================
# CONSTRUÇÃO DAS FEATURES
# =============================================================================

def prompt_size_category(chars: pd.Series) -> pd.Series:
    return pd.cut(
        chars,
        bins=[-np.inf, 500, 1000, 2000, np.inf],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).astype(float)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["prompt_density"] = out["prompt_chars"] / (out["prompt_lines"] + 1.0)
    out["prompt_token_density"] = (
        out["prompt_tokens"] / (out["prompt_chars"] + 1.0)
    )
    out["prompt_size_category"] = prompt_size_category(out["prompt_chars"])

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


def validate_input(df: pd.DataFrame) -> None:
    needed = (
        [GROUP_COL, MODEL_COL, TARGET]
        + PS_FEATURES
        + PROMPT_BASE_FEATURES
        + PATCH_BASE_FEATURES
    )
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")


def build_matrix(df: pd.DataFrame):
    work = add_derived_features(df)

    numeric_features = (
        PROMPT_BASE_FEATURES
        + PROMPT_DERIVED_FEATURES
        + PATCH_BASE_FEATURES
        + PATCH_DERIVED_FEATURES
        + PS_FEATURES
    )

    bad = sorted(set(numeric_features) & FORBIDDEN)
    if bad:
        raise RuntimeError(f"Features proibidas detectadas: {bad}")

    dummies = pd.get_dummies(
        work[MODEL_COL].astype(str),
        prefix="model",
        dtype=float,
    )

    X = pd.concat(
        [work[numeric_features].astype(float), dummies],
        axis=1,
    )

    model_features = dummies.columns.tolist()
    prompt_features = PROMPT_BASE_FEATURES + PROMPT_DERIVED_FEATURES
    patch_features = PATCH_BASE_FEATURES + PATCH_DERIVED_FEATURES

    groups = {
        "problem_statement": PS_FEATURES,
        "prompt_envelope": prompt_features,
        "patch_structure": patch_features,
        "model_identity": model_features,
    }

    feature_sets = {
        "full": X.columns.tolist(),
        "patch_only": patch_features,
        "patch_plus_problem_statement": patch_features + PS_FEATURES,
        "model_plus_problem_statement": model_features + PS_FEATURES,
        "problem_statement_only": PS_FEATURES,
        "model_only": model_features,
        "prompt_only": prompt_features,
    }

    return work, X, groups, feature_sets


# =============================================================================
# PREPROCESSAMENTO
# =============================================================================

def train_medians(X_train: pd.DataFrame) -> pd.Series:
    return X_train.median(numeric_only=True).fillna(0.0)


def apply_medians(X: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    return X.fillna(medians).fillna(0.0)


def fit_lr_on_split(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
):
    med = train_medians(X_train_raw)
    X_train = apply_medians(X_train_raw, med)
    X_test = apply_medians(X_test_raw, med)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(
        random_state=seed,
        **LR_PARAMS,
    )
    lr.fit(X_train_s, y_train)

    return lr, X_train_s, X_test_s, med, scaler


# =============================================================================
# ESTATÍSTICA
# =============================================================================

def bootstrap_ci_mean(
    values: np.ndarray,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    n = len(x)
    out = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        out[i] = np.mean(rng.choice(x, size=n, replace=True))

    return (
        float(np.quantile(out, alpha / 2)),
        float(np.quantile(out, 1 - alpha / 2)),
    )


def rank_biserial(deltas: np.ndarray) -> float:
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]

    if len(d) == 0:
        return 0.0

    ranks = rankdata(np.abs(d), method="average")
    wp = ranks[d > 0].sum()
    wm = ranks[d < 0].sum()

    if wp + wm == 0:
        return 0.0

    return float((wp - wm) / (wp + wm))


def wilcoxon_safe(deltas: np.ndarray):
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]

    if len(d) == 0 or np.allclose(d, 0):
        return 0.0, 1.0

    r = wilcoxon(
        d,
        alternative="two-sided",
        zero_method="wilcox",
        correction=False,
        method="auto",
    )
    return float(r.statistic), float(r.pvalue)


def bh_fdr(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    result = np.full(len(p), np.nan)

    valid = np.isfinite(p)
    pv = p[valid]
    if len(pv) == 0:
        return result

    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)

    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    restored = np.empty(m)
    restored[order] = q
    result[valid] = restored
    return result


def paired_summary(
    deltas_by_name: dict[str, list[float]],
    family: str,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows = []

    for i, (name, values) in enumerate(deltas_by_name.items()):
        d = np.asarray(values, dtype=float)
        lo, hi = bootstrap_ci_mean(
            d,
            n_boot=n_boot,
            seed=seed + i,
        )
        stat, p = wilcoxon_safe(d)

        rows.append({
            "family": family,
            "comparison": name,
            "n_runs": len(d),
            "mean_delta_full_minus_comparator": float(np.mean(d)),
            "std_delta": float(np.std(d, ddof=1)),
            "median_delta": float(np.median(d)),
            "ci95_mean_low": lo,
            "ci95_mean_high": hi,
            "positive_runs": int((d > 0).sum()),
            "negative_runs": int((d < 0).sum()),
            "zero_runs": int((d == 0).sum()),
            "wilcoxon_stat": stat,
            "p_two_sided": p,
            "rank_biserial": rank_biserial(d),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_bh_family"] = bh_fdr(out["p_two_sided"])
    return out


# =============================================================================
# DIAGNÓSTICO DE CORRELAÇÃO
# =============================================================================

def high_correlation_pairs(
    X: pd.DataFrame,
    threshold: float = 0.80,
) -> pd.DataFrame:
    """
    Diagnóstico descritivo. Ignora dummies de modelo.
    Spearman ajuda a mostrar por que coeficientes individuais devem ser interpretados
    com cautela.
    """
    cols = [c for c in X.columns if not c.startswith("model_")]
    data = X[cols].copy()
    data = data.fillna(data.median(numeric_only=True)).fillna(0.0)

    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            rho, p = spearmanr(data[a], data[b])
            if np.isfinite(rho) and abs(rho) >= threshold:
                rows.append({
                    "feature_a": a,
                    "feature_b": b,
                    "spearman_rho": float(rho),
                    "abs_rho": float(abs(rho)),
                    "p_value": float(p),
                })

    return pd.DataFrame(rows).sort_values(
        "abs_rho", ascending=False
    ) if rows else pd.DataFrame(
        columns=["feature_a", "feature_b", "spearman_rho", "abs_rho", "p_value"]
    )


# =============================================================================
# EXPERIMENTO
# =============================================================================

def run_experiment(
    work: pd.DataFrame,
    X: pd.DataFrame,
    groups: dict[str, list[str]],
    feature_sets: dict[str, list[str]],
    n_runs: int,
    base_seed: int,
    test_size: float,
    permutation_repeats: int,
):
    y = work[TARGET].astype(int).to_numpy()
    case_groups = work[GROUP_COL].astype(str).to_numpy()

    coef_rows = []
    perm_rows = []
    feature_set_rows = []
    ablation_rows = []

    fs_deltas = {
        f"full_vs_{name}": []
        for name in feature_sets
        if name != "full"
    }
    abl_deltas = {
        f"full_vs_without_{name}": []
        for name in groups
    }

    for run_idx in range(n_runs):
        run = run_idx + 1
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

        if train_cases & test_cases:
            raise RuntimeError(f"Case leakage no run {run}")

        y_train = y[train_idx]
        y_test = y[test_idx]

        # ---------------------------------------------------------------------
        # FULL
        # ---------------------------------------------------------------------
        X_train_full = X.iloc[train_idx].copy()
        X_test_full = X.iloc[test_idx].copy()

        lr_full, _, X_test_full_s, _, _ = fit_lr_on_split(
            X_train_full,
            X_test_full,
            y_train,
            seed,
        )

        score_full = lr_full.predict_proba(X_test_full_s)[:, 1]
        auc_full = roc_auc_score(y_test, score_full)
        pr_full = average_precision_score(y_test, score_full)

        # Coeficientes padronizados.
        for feature, coef in zip(X.columns, lr_full.coef_[0]):
            coef_rows.append({
                "run": run,
                "seed": seed,
                "feature": feature,
                "coefficient": float(coef),
                "abs_coefficient": float(abs(coef)),
                "direction": (
                    "positive" if coef > 0
                    else "negative" if coef < 0
                    else "zero"
                ),
            })

        # Permutation importance no TEST SET.
        pi = permutation_importance(
            lr_full,
            X_test_full_s,
            y_test,
            scoring="roc_auc",
            n_repeats=permutation_repeats,
            random_state=seed,
            n_jobs=-1,
        )

        for feature, imp_mean, imp_std in zip(
            X.columns,
            pi.importances_mean,
            pi.importances_std,
        ):
            perm_rows.append({
                "run": run,
                "seed": seed,
                "feature": feature,
                "permutation_importance": float(imp_mean),
                "permutation_repeat_std": float(imp_std),
            })

        # ---------------------------------------------------------------------
        # FEATURE SETS
        # ---------------------------------------------------------------------
        current_scores = {
            "full": {
                "roc_auc": auc_full,
                "pr_auc": pr_full,
            }
        }

        feature_set_rows.append({
            "run": run,
            "seed": seed,
            "feature_set": "full",
            "n_features": len(feature_sets["full"]),
            "roc_auc": auc_full,
            "pr_auc": pr_full,
        })

        for name, cols in feature_sets.items():
            if name == "full":
                continue

            Xtr = X.iloc[train_idx][cols].copy()
            Xte = X.iloc[test_idx][cols].copy()

            lr, _, Xte_s, _, _ = fit_lr_on_split(
                Xtr, Xte, y_train, seed
            )
            score = lr.predict_proba(Xte_s)[:, 1]

            auc = roc_auc_score(y_test, score)
            pr = average_precision_score(y_test, score)

            current_scores[name] = {
                "roc_auc": auc,
                "pr_auc": pr,
            }

            feature_set_rows.append({
                "run": run,
                "seed": seed,
                "feature_set": name,
                "n_features": len(cols),
                "roc_auc": auc,
                "pr_auc": pr,
            })

            fs_deltas[f"full_vs_{name}"].append(
                auc_full - auc
            )

        # ---------------------------------------------------------------------
        # GROUP ABLATION
        # ---------------------------------------------------------------------
        for group_name, removed_cols in groups.items():
            remaining = [
                c for c in X.columns
                if c not in set(removed_cols)
            ]

            Xtr = X.iloc[train_idx][remaining].copy()
            Xte = X.iloc[test_idx][remaining].copy()

            lr, _, Xte_s, _, _ = fit_lr_on_split(
                Xtr, Xte, y_train, seed
            )
            score = lr.predict_proba(Xte_s)[:, 1]
            auc_without = roc_auc_score(y_test, score)

            delta = auc_full - auc_without
            abl_deltas[f"full_vs_without_{group_name}"].append(delta)

            ablation_rows.append({
                "run": run,
                "seed": seed,
                "group_removed": group_name,
                "n_removed": len(removed_cols),
                "auc_full": auc_full,
                "auc_without_group": auc_without,
                "auc_loss_when_removed": delta,
            })

        print(
            f"Run {run:02d}/{n_runs} | "
            f"seed={seed} | "
            f"FULL AUC={auc_full:.3f} | PR-AUC={pr_full:.3f}"
        )

    return {
        "coefficients_all": pd.DataFrame(coef_rows),
        "permutation_all": pd.DataFrame(perm_rows),
        "feature_sets_all": pd.DataFrame(feature_set_rows),
        "ablation_all": pd.DataFrame(ablation_rows),
        "feature_set_deltas": fs_deltas,
        "ablation_deltas": abl_deltas,
    }


# =============================================================================
# RESUMOS
# =============================================================================

def summarize_coefficients(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("feature", as_index=False)
        .agg(
            coefficient_mean=("coefficient", "mean"),
            coefficient_std=("coefficient", "std"),
            abs_coefficient_mean=("abs_coefficient", "mean"),
            abs_coefficient_std=("abs_coefficient", "std"),
            positive_runs=("coefficient", lambda s: int((s > 0).sum())),
            negative_runs=("coefficient", lambda s: int((s < 0).sum())),
            zero_runs=("coefficient", lambda s: int((s == 0).sum())),
        )
        .assign(
            sign_consistency=lambda d: np.maximum(
                d["positive_runs"], d["negative_runs"]
            ) / (
                d["positive_runs"]
                + d["negative_runs"]
                + d["zero_runs"]
            )
        )
        .sort_values("abs_coefficient_mean", ascending=False)
    )


def summarize_permutation(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("feature", as_index=False)
        .agg(
            permutation_mean=("permutation_importance", "mean"),
            permutation_std_across_runs=("permutation_importance", "std"),
            positive_runs=("permutation_importance", lambda s: int((s > 0).sum())),
            negative_runs=("permutation_importance", lambda s: int((s < 0).sum())),
            zero_runs=("permutation_importance", lambda s: int((s == 0).sum())),
        )
        .sort_values("permutation_mean", ascending=False)
    )


def summarize_feature_sets(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["feature_set", "n_features"], as_index=False)
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
        )
        .sort_values("roc_auc_mean", ascending=False)
    )


def summarize_ablation(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["group_removed", "n_removed"], as_index=False)
        .agg(
            auc_full_mean=("auc_full", "mean"),
            auc_without_group_mean=("auc_without_group", "mean"),
            auc_loss_mean=("auc_loss_when_removed", "mean"),
            auc_loss_std=("auc_loss_when_removed", "std"),
            positive_runs=("auc_loss_when_removed", lambda s: int((s > 0).sum())),
            negative_runs=("auc_loss_when_removed", lambda s: int((s < 0).sum())),
        )
        .sort_values("auc_loss_mean", ascending=False)
    )


# =============================================================================
# FIGURAS
# =============================================================================

def plot_top_permutation(summary: pd.DataFrame, path: Path, top_k: int = 15):
    d = summary.head(top_k).sort_values("permutation_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(d))
    ax.barh(
        y,
        d["permutation_mean"],
        xerr=d["permutation_std_across_runs"],
        capsize=3,
    )
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["feature"])
    ax.set_xlabel("Queda média de ROC-AUC após permutação no teste")
    ax.set_title("Logistic Regression — permutation importance (30 runs)")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_top_coefficients(summary: pd.DataFrame, path: Path, top_k: int = 15):
    d = summary.head(top_k).sort_values("abs_coefficient_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(d))
    ax.barh(
        y,
        d["abs_coefficient_mean"],
        xerr=d["abs_coefficient_std"],
        capsize=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(d["feature"])
    ax.set_xlabel("|coeficiente padronizado| médio")
    ax.set_title("Logistic Regression — magnitude dos coeficientes (30 runs)")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(summary: pd.DataFrame, path: Path):
    d = summary.sort_values("auc_loss_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(d))
    ax.barh(
        y,
        d["auc_loss_mean"],
        xerr=d["auc_loss_std"],
        capsize=3,
    )
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["group_removed"])
    ax.set_xlabel("Δ ROC-AUC = FULL − sem grupo")
    ax.set_title("Logistic Regression — ablação de grupos")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_sets(summary: pd.DataFrame, path: Path):
    d = summary.sort_values("roc_auc_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(d))
    ax.barh(
        y,
        d["roc_auc_mean"],
        xerr=d["roc_auc_std"],
        capsize=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(d["feature_set"])
    ax.set_xlabel("ROC-AUC médio ± DP")
    ax.set_title("Logistic Regression — conjuntos de features")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=DEFAULT_INPUT,
    )
    ap.add_argument(
        "--outdir",
        default="results_lr_importance",
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
        "--permutation-repeats",
        type=int,
        default=10,
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
    ap.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.80,
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if not input_path.exists():
        fallback = INPUT_FALLBACK_DIR / input_path.name
        if fallback.exists():
            input_path = fallback
        else:
            raise FileNotFoundError(
                f"CSV não encontrado: {args.input}\n"
                f"Use: --input {DEFAULT_INPUT}"
            )

    df = pd.read_csv(input_path)
    validate_input(df)

    work, X, groups, feature_sets = build_matrix(df)

    print("=" * 90)
    print("LOGISTIC REGRESSION — IMPORTÂNCIA / ABLAÇÃO — 30 RUNS")
    print("=" * 90)
    print(f"Rows: {len(work)}")
    print(f"Cases: {work[GROUP_COL].nunique()}")
    print(f"Models: {work[MODEL_COL].nunique()}")
    print(f"Features FULL: {X.shape[1]}")
    print(f"Positive: {int(work[TARGET].sum())}")
    print(f"Negative: {int(len(work) - work[TARGET].sum())}")

    results = run_experiment(
        work=work,
        X=X,
        groups=groups,
        feature_sets=feature_sets,
        n_runs=args.n_runs,
        base_seed=args.base_seed,
        test_size=args.test_size,
        permutation_repeats=args.permutation_repeats,
    )

    coef_summary = summarize_coefficients(results["coefficients_all"])
    perm_summary = summarize_permutation(results["permutation_all"])
    fs_summary = summarize_feature_sets(results["feature_sets_all"])
    abl_summary = summarize_ablation(results["ablation_all"])

    fs_tests = paired_summary(
        results["feature_set_deltas"],
        family="feature_set",
        n_boot=args.bootstrap,
        seed=args.bootstrap_seed,
    )
    abl_tests = paired_summary(
        results["ablation_deltas"],
        family="ablation",
        n_boot=args.bootstrap,
        seed=args.bootstrap_seed + 500,
    )

    all_tests = pd.concat([fs_tests, abl_tests], ignore_index=True)
    if not all_tests.empty:
        all_tests["q_bh_all"] = bh_fdr(all_tests["p_two_sided"])

    corr_pairs = high_correlation_pairs(
        X,
        threshold=args.correlation_threshold,
    )

    # CSVs
    results["coefficients_all"].to_csv(
        outdir / "lr_coefficients_all_runs.csv", index=False
    )
    coef_summary.to_csv(
        outdir / "lr_coefficients_summary.csv", index=False
    )

    results["permutation_all"].to_csv(
        outdir / "lr_permutation_importance_all_runs.csv", index=False
    )
    perm_summary.to_csv(
        outdir / "lr_permutation_importance_summary.csv", index=False
    )

    results["feature_sets_all"].to_csv(
        outdir / "lr_feature_sets_all_runs.csv", index=False
    )
    fs_summary.to_csv(
        outdir / "lr_feature_sets_summary.csv", index=False
    )

    results["ablation_all"].to_csv(
        outdir / "lr_group_ablation_all_runs.csv", index=False
    )
    abl_summary.to_csv(
        outdir / "lr_group_ablation_summary.csv", index=False
    )

    all_tests.to_csv(
        outdir / "lr_paired_tests.csv", index=False
    )

    corr_pairs.to_csv(
        outdir / "lr_high_correlation_pairs.csv", index=False
    )

    # Figures
    plot_top_permutation(
        perm_summary,
        outdir / "fig_lr_permutation_importance.png",
    )
    plot_top_coefficients(
        coef_summary,
        outdir / "fig_lr_coefficients.png",
    )
    plot_ablation(
        abl_summary,
        outdir / "fig_lr_group_ablation.png",
    )
    plot_feature_sets(
        fs_summary,
        outdir / "fig_lr_feature_sets.png",
    )

    summary = {
        "input": str(args.input),
        "rows": len(work),
        "unique_cases": int(work[GROUP_COL].nunique()),
        "models": sorted(work[MODEL_COL].astype(str).unique().tolist()),
        "target": TARGET,
        "task": "post-generation / pre-SAST",
        "runs": args.n_runs,
        "seeds": [args.base_seed + i for i in range(args.n_runs)],
        "split": "GroupShuffleSplit grouped by case",
        "full_feature_count": int(X.shape[1]),
        "full_features": X.columns.tolist(),
        "lr_params": LR_PARAMS,
        "permutation_repeats": args.permutation_repeats,
        "interpretation_guardrail": (
            "Coefficients are descriptive for the L2-regularized predictive LR. "
            "Prioritize test-set permutation importance and group ablation because "
            "correlated predictors can redistribute coefficient magnitude."
        ),
    }

    with open(
        outdir / "lr_importance_run_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Console
    print("\n" + "=" * 90)
    print("TOP 15 — PERMUTATION IMPORTANCE (TESTE)")
    print("=" * 90)
    print(
        perm_summary.head(15).to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    print("\n" + "=" * 90)
    print("TOP 15 — |COEFICIENTE PADRONIZADO|")
    print("=" * 90)
    print(
        coef_summary.head(15).to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    print("\n" + "=" * 90)
    print("CONJUNTOS DE FEATURES")
    print("=" * 90)
    print(
        fs_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    print("\n" + "=" * 90)
    print("ABLAÇÃO DE GRUPOS")
    print("=" * 90)
    print(
        abl_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    print("\n" + "=" * 90)
    print("TESTES PAREADOS")
    print("=" * 90)
    print(
        all_tests.to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}",
        )
    )

    print("\n" + "=" * 90)
    print("MULTICOLINEARIDADE — pares com |Spearman rho| >= "
          f"{args.correlation_threshold:.2f}")
    print("=" * 90)
    if corr_pairs.empty:
        print("Nenhum par acima do limiar.")
    else:
        print(
            corr_pairs.head(30).to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    print("\nArquivos salvos em:", outdir.resolve())


if __name__ == "__main__":
    main()
