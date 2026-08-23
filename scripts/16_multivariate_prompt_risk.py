#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
16_multivariate_prompt_risk.py

Objetivo
--------
Responder:

"Quais características do prompt permanecem associadas ao risco quando
analisadas conjuntamente e controladas por outras variáveis?"

O script usa o CSV limpo do script 14:
    case_model_problem_statement_features.csv

Partes:
1. Regressão logística multivariada interpretável
   - OR ajustado
   - IC95%
   - p-value
   - FDR Benjamini-Hochberg

2. Avaliação preditiva com split agrupado por case
   - Logistic Regression
   - Random Forest
   - 30 execuções 80/20 GroupShuffleSplit
   - ROC-AUC, PR-AUC, Recall, F1, Balanced Accuracy

3. Comparação pareada entre:
   - controles apenas
   - controles + prompt risk features

Controles:
- tamanho/estrutura básica do prompt
- modelo LLM (one-hot)

Features conceituais principais:
- explicit security
- security-relevant domain count
- database
- command execution
- auth
- permissions
- identifier density
- numeric density
- bullet count
- constraints

Observação:
- Associações são ajustadas, mas ainda não causais.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import norm, wilcoxon

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIGURAÇÃO DE FEATURES
# ============================================================

MAIN_PROMPT_FEATURES = [
    # principais sinais conceituais (sem duplicar *_count e binário quando possível)
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

CONTROL_NUMERIC = [
    "ps_chars",
    "ps_lines",
    "ps_words",
]

CONTROL_CATEGORICAL = [
    "model",
]

TARGET = "has_finding_after"


# ============================================================
# FDR
# ============================================================

def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)

    n = len(p)
    out = np.full(n, np.nan)

    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0:
        return pd.Series(out, index=p_values.index)

    order = valid[np.argsort(p[valid])]
    m = len(order)

    raw = np.empty(m, dtype=float)

    for rank, idx in enumerate(order, start=1):
        raw[rank - 1] = p[idx] * m / rank

    raw = np.minimum.accumulate(raw[::-1])[::-1]
    raw = np.minimum(raw, 1.0)

    for adj, idx in zip(raw, order):
        out[idx] = adj

    return pd.Series(out, index=p_values.index)


# ============================================================
# DESIGN MATRIX PARA OR AJUSTADO
# ============================================================

def build_design_matrix(df: pd.DataFrame):
    """
    Cria matriz explícita para regressão logística interpretável.
    Padroniza apenas features numéricas contínuas.
    One-hot para model.
    """
    work = df.copy()

    required = set(MAIN_PROMPT_FEATURES + CONTROL_NUMERIC + CONTROL_CATEGORICAL + [TARGET])

    missing = sorted(required - set(work.columns))
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")

    for c in MAIN_PROMPT_FEATURES + CONTROL_NUMERIC:
        work[c] = pd.to_numeric(
            work[c],
            errors="coerce",
        ).fillna(0)

    # Mantemos binários como estão; escalamos contínuas.
    continuous = [
        "ps_security_relevant_domain_count",
        "ps_identifier_density",
        "ps_numeric_density",
        "ps_bullet_count",
        "ps_constraint_count",
        "ps_chars",
        "ps_lines",
        "ps_words",
    ]

    binary = [
        "ps_domain_explicit_security",
        "ps_domain_database",
        "ps_domain_command_exec",
        "ps_domain_auth",
        "ps_domain_permissions",
    ]

    X_num = pd.DataFrame(index=work.index)

    scaler = StandardScaler()

    if continuous:
        X_num[continuous] = scaler.fit_transform(work[continuous])

    for c in binary:
        X_num[c] = work[c].astype(float)

    model_dummies = pd.get_dummies(
        work["model"].astype(str),
        prefix="model",
        drop_first=True,
        dtype=float,
    )

    X = pd.concat(
        [
            pd.Series(1.0, index=work.index, name="intercept"),
            X_num,
            model_dummies,
        ],
        axis=1,
    )

    y = work[TARGET].astype(int).to_numpy()

    return X, y


# ============================================================
# LOGISTIC MLE COM NEWTON-RAPHSON
# ============================================================

def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic_mle(X: pd.DataFrame, y: np.ndarray, max_iter=200, tol=1e-8):
    """
    Logistic regression MLE sem regularização, com Hessiana.
    Retorna beta, covariance, converged.
    """
    Xv = X.to_numpy(dtype=float)
    yv = np.asarray(y, dtype=float)

    p = Xv.shape[1]
    beta = np.zeros(p, dtype=float)

    converged = False

    for _ in range(max_iter):
        eta = Xv @ beta
        mu = sigmoid(eta)

        w = mu * (1.0 - mu)
        w = np.clip(w, 1e-8, None)

        grad = Xv.T @ (yv - mu)

        H = -(Xv.T * w) @ Xv

        try:
            step = np.linalg.solve(-H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(-H) @ grad

        beta_new = beta + step

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            converged = True
            break

        beta = beta_new

    eta = Xv @ beta
    mu = sigmoid(eta)
    w = np.clip(mu * (1.0 - mu), 1e-8, None)

    fisher = (Xv.T * w) @ Xv

    try:
        cov = np.linalg.inv(fisher)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(fisher)

    return beta, cov, converged


def adjusted_or_table(X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, bool]:
    beta, cov, converged = fit_logistic_mle(X, y)

    se = np.sqrt(np.diag(cov))
    z_scores = beta / se
    p_values = 2.0 * (1.0 - norm.cdf(np.abs(z_scores)))

    z = norm.ppf(0.975)

    rows = []

    for i, feature in enumerate(X.columns):
        coef = beta[i]
        std = se[i]

        or_value = math.exp(coef)
        low = math.exp(coef - z * std)
        high = math.exp(coef + z * std)

        rows.append({
            "feature": feature,
            "coef": coef,
            "std_error": std,
            "adjusted_odds_ratio": or_value,
            "or_ci95_low": low,
            "or_ci95_high": high,
            "z_value": z_scores[i],
            "p_value": p_values[i],
        })

    result = pd.DataFrame(rows)

    result["fdr_bh"] = benjamini_hochberg(
        result["p_value"]
    )

    result["direction"] = np.select(
        [
            (
                (result["fdr_bh"] < 0.05)
                & (result["or_ci95_low"] > 1)
            ),
            (
                (result["fdr_bh"] < 0.05)
                & (result["or_ci95_high"] < 1)
            ),
        ],
        [
            "higher adjusted risk",
            "lower adjusted risk",
        ],
        default="inconclusive",
    )

    return result, converged


# ============================================================
# PREDICTIVE EVALUATION
# ============================================================

def make_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
):
    transformers = []

    if numeric_features:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            numeric_features,
        ))

    if categorical_features:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ))

    return ColumnTransformer(transformers)


def make_predictive_models(
    numeric_features: list[str],
    categorical_features: list[str],
    seed: int,
):
    return {
        "logistic_regression": Pipeline([
            ("prep", make_preprocessor(
                numeric_features,
                categorical_features,
            )),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=seed,
            )),
        ]),
        "random_forest": Pipeline([
            ("prep", make_preprocessor(
                numeric_features,
                categorical_features,
            )),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )),
        ]),
    }


def metric_dict(y_true, pred, prob):
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision_1": precision_score(y_true, pred, zero_division=0),
        "recall_1": recall_score(y_true, pred, zero_division=0),
        "f1_1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob),
        "pr_auc": average_precision_score(y_true, prob),
    }


def evaluate_predictive(df: pd.DataFrame, runs: int, base_seed: int):
    y = df[TARGET].astype(int).to_numpy()
    groups = df["case"].astype(str).to_numpy()

    feature_sets = {
        "Controls only": (
            CONTROL_NUMERIC,
            CONTROL_CATEGORICAL,
        ),
        "Controls + Prompt Risk Features": (
            CONTROL_NUMERIC + MAIN_PROMPT_FEATURES,
            CONTROL_CATEGORICAL,
        ),
        "Prompt Risk Features only": (
            MAIN_PROMPT_FEATURES,
            [],
        ),
    }

    rows = []

    for run_idx in range(runs):
        seed = base_seed + run_idx

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=seed,
        )

        train_idx, test_idx = next(
            splitter.split(df, y, groups=groups)
        )

        if (
            len(np.unique(y[train_idx])) < 2
            or len(np.unique(y[test_idx])) < 2
        ):
            continue

        for feature_set, (nums, cats) in feature_sets.items():
            cols = nums + cats

            for classifier_name, model in make_predictive_models(
                nums,
                cats,
                seed,
            ).items():

                model.fit(
                    df.iloc[train_idx][cols],
                    y[train_idx],
                )

                prob = model.predict_proba(
                    df.iloc[test_idx][cols]
                )[:, 1]

                pred = (prob >= 0.5).astype(int)

                m = metric_dict(
                    y[test_idx],
                    pred,
                    prob,
                )

                m.update({
                    "run": run_idx + 1,
                    "seed": seed,
                    "classifier": classifier_name,
                    "feature_set": feature_set,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "test_cases": len(np.unique(groups[test_idx])),
                })

                rows.append(m)

    return pd.DataFrame(rows)


# ============================================================
# PAIRED COMPARISON
# ============================================================

def rank_biserial(delta: np.ndarray) -> float:
    d = np.asarray(delta, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]

    if len(d) == 0:
        return 0.0

    ranks = pd.Series(
        np.abs(d)
    ).rank(
        method="average"
    ).to_numpy()

    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()

    denom = w_pos + w_neg

    return (
        float((w_pos - w_neg) / denom)
        if denom
        else 0.0
    )


def bootstrap_ci(values, n_boot=10000, seed=42):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)

    stats = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sample = rng.choice(
            x,
            size=len(x),
            replace=True,
        )
        stats[i] = np.mean(sample)

    return (
        float(np.quantile(stats, 0.025)),
        float(np.quantile(stats, 0.975)),
    )


def paired_auc_comparison(
    runs_df: pd.DataFrame,
    classifier: str,
    baseline: str,
    enriched: str,
    seed: int,
    n_boot: int,
):
    sub = runs_df[
        runs_df["classifier"].eq(classifier)
    ]

    pivot = sub.pivot(
        index="run",
        columns="feature_set",
        values="roc_auc",
    )

    paired = pd.DataFrame({
        "run": pivot.index,
        "auc_baseline": pivot[baseline],
        "auc_enriched": pivot[enriched],
    })

    paired["delta_auc"] = (
        paired["auc_enriched"]
        - paired["auc_baseline"]
    )

    delta = paired["delta_auc"].to_numpy()

    w = wilcoxon(
        delta,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )

    ci_low, ci_high = bootstrap_ci(
        delta,
        n_boot=n_boot,
        seed=seed,
    )

    summary = {
        "classifier": classifier,
        "baseline": baseline,
        "enriched": enriched,
        "mean_auc_baseline": float(
            paired["auc_baseline"].mean()
        ),
        "mean_auc_enriched": float(
            paired["auc_enriched"].mean()
        ),
        "mean_delta_auc": float(
            paired["delta_auc"].mean()
        ),
        "median_delta_auc": float(
            paired["delta_auc"].median()
        ),
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
        "wilcoxon_statistic": float(w.statistic),
        "wilcoxon_p_value": float(w.pvalue),
        "rank_biserial": rank_biserial(delta),
        "positive_runs": int(
            (paired["delta_auc"] > 0).sum()
        ),
        "negative_runs": int(
            (paired["delta_auc"] < 0).sum()
        ),
        "n_pairs": int(len(paired)),
    }

    return paired, summary


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=True,
        help=(
            "CSV do script 14, por exemplo "
            "case_model_problem_statement_features.csv"
        ),
    )

    ap.add_argument(
        "--runs",
        type=int,
        default=30,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    ap.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10000,
    )

    ap.add_argument(
        "--out-dir",
        default="multivariate_prompt_risk_results",
    )

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(args.input)

    if TARGET not in df.columns:
        if "findings_after" not in df.columns:
            raise SystemExit(
                f"Ausente target {TARGET} e findings_after."
            )

        df[TARGET] = (
            pd.to_numeric(
                df["findings_after"],
                errors="coerce",
            )
            .fillna(0)
            .gt(0)
            .astype(int)
        )

    required = set(
        ["case", TARGET]
        + MAIN_PROMPT_FEATURES
        + CONTROL_NUMERIC
        + CONTROL_CATEGORICAL
    )

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise SystemExit(
            "Colunas ausentes:\n- "
            + "\n- ".join(missing)
        )

    # --------------------------------------------------------
    # Multivariate adjusted OR
    # --------------------------------------------------------
    X, y = build_design_matrix(df)

    adjusted, converged = adjusted_or_table(
        X,
        y,
    )

    adjusted.to_csv(
        out_dir / "adjusted_odds_ratios.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Predictive evaluation
    # --------------------------------------------------------
    runs_df = evaluate_predictive(
        df,
        runs=args.runs,
        base_seed=args.seed,
    )

    runs_df.to_csv(
        out_dir / "predictive_all_runs.csv",
        index=False,
    )

    summary = (
        runs_df
        .groupby(
            ["classifier", "feature_set"]
        )
        .agg({
            "accuracy": ["mean", "std"],
            "balanced_accuracy": ["mean", "std"],
            "precision_1": ["mean", "std"],
            "recall_1": ["mean", "std"],
            "f1_1": ["mean", "std"],
            "roc_auc": ["mean", "std"],
            "pr_auc": ["mean", "std"],
        })
    )

    summary.columns = [
        "_".join(c)
        for c in summary.columns
    ]

    summary = summary.reset_index()

    summary.to_csv(
        out_dir / "predictive_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Paired tests
    # --------------------------------------------------------
    paired_rows = []
    paired_summaries = []

    for i, classifier in enumerate(
        ["logistic_regression", "random_forest"]
    ):
        paired, result = paired_auc_comparison(
            runs_df,
            classifier=classifier,
            baseline="Controls only",
            enriched="Controls + Prompt Risk Features",
            seed=args.seed + i,
            n_boot=args.bootstrap_resamples,
        )

        paired.to_csv(
            out_dir
            / f"paired_auc_{classifier}.csv",
            index=False,
        )

        paired_summaries.append(
            result
        )

    paired_summary_df = pd.DataFrame(
        paired_summaries
    )

    paired_summary_df.to_csv(
        out_dir / "paired_auc_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Run summary
    # --------------------------------------------------------
    report = {
        "rows": int(len(df)),
        "unique_cases": int(df["case"].nunique()),
        "positive": int(df[TARGET].sum()),
        "negative": int(len(df) - df[TARGET].sum()),
        "main_prompt_features": MAIN_PROMPT_FEATURES,
        "control_numeric": CONTROL_NUMERIC,
        "control_categorical": CONTROL_CATEGORICAL,
        "logistic_mle_converged": bool(converged),
        "runs": int(args.runs),
        "base_seed": int(args.seed),
        "split": "80/20 GroupShuffleSplit grouped by case",
        "note": (
            "Adjusted ORs are associative, not causal. "
            "Predictive evaluation keeps all rows from the same case "
            "in the same split."
        ),
    }

    (
        out_dir / "run_summary.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------
    print()
    print("=" * 92)
    print("MULTIVARIATE PROMPT RISK ANALYSIS")
    print("=" * 92)
    print(f"Rows       : {len(df)}")
    print(f"Cases      : {df['case'].nunique()}")
    print(f"Positivos  : {int(df[TARGET].sum())}")
    print(f"Convergiu  : {converged}")
    print()

    print("ADJUSTED ODDS RATIOS")
    display_adj = adjusted[
        adjusted["feature"] != "intercept"
    ][
        [
            "feature",
            "adjusted_odds_ratio",
            "or_ci95_low",
            "or_ci95_high",
            "p_value",
            "fdr_bh",
            "direction",
        ]
    ].sort_values(
        "fdr_bh",
        na_position="last",
    )

    print(
        display_adj.to_string(index=False)
    )

    print()
    print("PREDICTIVE SUMMARY")
    print(
        summary[
            [
                "classifier",
                "feature_set",
                "roc_auc_mean",
                "roc_auc_std",
                "recall_1_mean",
                "f1_1_mean",
                "pr_auc_mean",
            ]
        ].to_string(index=False)
    )

    print()
    print("PAIRED AUC TEST")
    print(
        paired_summary_df[
            [
                "classifier",
                "mean_auc_baseline",
                "mean_auc_enriched",
                "mean_delta_auc",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
                "wilcoxon_p_value",
                "rank_biserial",
                "positive_runs",
                "n_pairs",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Saída: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
