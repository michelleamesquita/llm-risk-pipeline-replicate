#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
17_final_prompt_risk_classifier.py

Experimento final do classificador pré-SAST.

Objetivos:
1) Congelar grupos conceituais de features do problem_statement.
2) Evitar redundância óbvia entre representações binárias/count.
3) Diagnosticar correlação e VIF dos controles de tamanho.
4) Comparar:
      Baseline
      Sensitive Operation
      Technical Specificity
      Structural / Constraint Signals
      All Prompt Risk Features
      Baseline + All Prompt Risk Features
5) Usar split 80/20 agrupado por case em 30 execuções.
6) Avaliar Logistic Regression e Random Forest.
7) Comparar cada conjunto contra Baseline com:
      ROC-AUC
      PR-AUC
      Recall
      F1
      IC95% bootstrap do delta
      Wilcoxon pareado
      rank-biserial
8) Produzir tabelas e figuras prontas para análise/artigo.

Uso:
python scripts/17_final_prompt_risk_classifier.py \
  --input problem_statement_ablation_results/case_model_problem_statement_features.csv \
  --runs 30 \
  --bootstrap-resamples 10000 \
  --out-dir final_prompt_risk_classifier_results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
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


TARGET = "has_finding_after"

# ------------------------------------------------------------
# FEATURE GROUPS CONGELADOS
# ------------------------------------------------------------

SENSITIVE_OPERATION = [
    "ps_domain_explicit_security",
    "ps_security_relevant_domain_count",
    "ps_domain_database",
    "ps_domain_command_exec",
    "ps_domain_auth",
    "ps_domain_permissions",
]

TECHNICAL_SPECIFICITY = [
    "ps_identifier_density",
    "ps_numeric_density",
]

STRUCTURAL_CONSTRAINT = [
    "ps_bullet_count",
    "ps_constraint_count",
]

ALL_PROMPT_RISK = (
    SENSITIVE_OPERATION
    + TECHNICAL_SPECIFICITY
    + STRUCTURAL_CONSTRAINT
)

# Baseline propositalmente simples:
# modelo + tamanho do problem statement.
# ps_words é mantido como controle primário de tamanho.
BASELINE_NUMERIC = [
    "ps_words",
]

BASELINE_CATEGORICAL = [
    "model",
]

# Apenas para diagnóstico de multicolinearidade.
SIZE_DIAGNOSTIC = [
    "ps_chars",
    "ps_lines",
    "ps_words",
]


def make_preprocessor(numeric, categorical):
    transformers = []

    if numeric:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ))

    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ))

    return ColumnTransformer(transformers)


def make_models(numeric, categorical, seed):
    return {
        "logistic_regression": Pipeline([
            ("prep", make_preprocessor(numeric, categorical)),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=seed,
            )),
        ]),
        "random_forest": Pipeline([
            ("prep", make_preprocessor(numeric, categorical)),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )),
        ]),
    }


def compute_metrics(y_true, pred, prob):
    return {
        "roc_auc": roc_auc_score(y_true, prob),
        "pr_auc": average_precision_score(y_true, prob),
        "precision_1": precision_score(y_true, pred, zero_division=0),
        "recall_1": recall_score(y_true, pred, zero_division=0),
        "f1_1": f1_score(y_true, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
    }


def bootstrap_ci(values, n_boot=10000, seed=42):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)

    for i in range(n_boot):
        means[i] = rng.choice(
            x, size=len(x), replace=True
        ).mean()

    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def rank_biserial(delta):
    d = np.asarray(delta, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]

    if len(d) == 0:
        return 0.0

    ranks = pd.Series(np.abs(d)).rank(
        method="average"
    ).to_numpy()

    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()

    denom = w_pos + w_neg
    return float((w_pos - w_neg) / denom) if denom else 0.0


def vif_table(df, features):
    X = df[features].apply(
        pd.to_numeric, errors="coerce"
    ).copy()

    X = X.fillna(X.median())

    rows = []

    for feature in features:
        y = X[feature].to_numpy()
        others = [c for c in features if c != feature]

        if not others:
            vif = 1.0
        else:
            model = LinearRegression()
            model.fit(X[others], y)
            r2 = model.score(X[others], y)
            vif = np.inf if r2 >= 0.999999 else 1.0 / (1.0 - r2)

        rows.append({
            "feature": feature,
            "vif": vif,
        })

    return pd.DataFrame(rows)


def feature_sets():
    return {
        "Baseline": (
            BASELINE_NUMERIC,
            BASELINE_CATEGORICAL,
        ),

        "Sensitive Operation": (
            SENSITIVE_OPERATION,
            [],
        ),

        "Technical Specificity": (
            TECHNICAL_SPECIFICITY,
            [],
        ),

        "Structural / Constraint Signals": (
            STRUCTURAL_CONSTRAINT,
            [],
        ),

        "All Prompt Risk Features": (
            ALL_PROMPT_RISK,
            [],
        ),

        "Baseline + All Prompt Risk Features": (
            BASELINE_NUMERIC + ALL_PROMPT_RISK,
            BASELINE_CATEGORICAL,
        ),
    }


def evaluate(df, runs, seed):
    y = df[TARGET].astype(int).to_numpy()
    groups = df["case"].astype(str).to_numpy()

    rows = []

    for run_idx in range(runs):
        run_seed = seed + run_idx

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=run_seed,
        )

        train_idx, test_idx = next(
            splitter.split(df, y, groups)
        )

        if (
            len(np.unique(y[train_idx])) < 2
            or len(np.unique(y[test_idx])) < 2
        ):
            continue

        for fs_name, (numeric, categorical) in feature_sets().items():
            cols = numeric + categorical

            for classifier, model in make_models(
                numeric, categorical, run_seed
            ).items():

                model.fit(
                    df.iloc[train_idx][cols],
                    y[train_idx],
                )

                prob = model.predict_proba(
                    df.iloc[test_idx][cols]
                )[:, 1]

                pred = (prob >= 0.5).astype(int)

                metrics = compute_metrics(
                    y[test_idx], pred, prob
                )

                metrics.update({
                    "run": run_idx + 1,
                    "seed": run_seed,
                    "classifier": classifier,
                    "feature_set": fs_name,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_test_cases": len(
                        np.unique(groups[test_idx])
                    ),
                })

                rows.append(metrics)

    return pd.DataFrame(rows)


def summarize_runs(runs_df):
    metrics = [
        "roc_auc",
        "pr_auc",
        "recall_1",
        "f1_1",
        "balanced_accuracy",
    ]

    agg = {}

    for metric in metrics:
        agg[metric] = ["mean", "std"]

    out = (
        runs_df
        .groupby(["classifier", "feature_set"])
        .agg(agg)
    )

    out.columns = [
        "_".join(col) for col in out.columns
    ]

    return out.reset_index()


def paired_comparisons(
    runs_df,
    metric,
    baseline,
    n_boot,
    seed,
):
    rows = []
    paired_rows = []

    for classifier in sorted(
        runs_df["classifier"].unique()
    ):
        sub = runs_df[
            runs_df["classifier"].eq(classifier)
        ]

        pivot = sub.pivot(
            index="run",
            columns="feature_set",
            values=metric,
        )

        if baseline not in pivot.columns:
            continue

        for fs in feature_sets():
            if fs == baseline:
                continue

            paired = pivot[
                [baseline, fs]
            ].dropna().copy()

            delta = (
                paired[fs] - paired[baseline]
            ).to_numpy()

            if len(delta) == 0:
                continue

            ci_low, ci_high = bootstrap_ci(
                delta,
                n_boot=n_boot,
                seed=seed,
            )

            try:
                w = wilcoxon(
                    delta,
                    zero_method="wilcox",
                    alternative="two-sided",
                    method="auto",
                )
                p_value = float(w.pvalue)
                statistic = float(w.statistic)
            except ValueError:
                p_value = 1.0
                statistic = 0.0

            rows.append({
                "classifier": classifier,
                "metric": metric,
                "baseline": baseline,
                "feature_set": fs,
                "baseline_mean": float(
                    paired[baseline].mean()
                ),
                "feature_set_mean": float(
                    paired[fs].mean()
                ),
                "mean_delta": float(
                    np.mean(delta)
                ),
                "median_delta": float(
                    np.median(delta)
                ),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_value": p_value,
                "rank_biserial": rank_biserial(delta),
                "positive_runs": int(
                    np.sum(delta > 0)
                ),
                "negative_runs": int(
                    np.sum(delta < 0)
                ),
                "n_pairs": int(len(delta)),
            })

            temp = pd.DataFrame({
                "run": paired.index,
                "classifier": classifier,
                "metric": metric,
                "baseline": baseline,
                "feature_set": fs,
                "baseline_value": paired[baseline].values,
                "feature_set_value": paired[fs].values,
                "delta": delta,
            })

            paired_rows.append(temp)

    return (
        pd.DataFrame(rows),
        pd.concat(
            paired_rows,
            ignore_index=True,
        ) if paired_rows else pd.DataFrame(),
    )


def plot_auc(summary, out_path):
    plot_df = summary[
        summary["classifier"].eq(
            "logistic_regression"
        )
    ].copy()

    order = [
        "Baseline",
        "Sensitive Operation",
        "Technical Specificity",
        "Structural / Constraint Signals",
        "All Prompt Risk Features",
        "Baseline + All Prompt Risk Features",
    ]

    plot_df["feature_set"] = pd.Categorical(
        plot_df["feature_set"],
        categories=order,
        ordered=True,
    )

    plot_df = plot_df.sort_values(
        "feature_set"
    )

    fig, ax = plt.subplots(
        figsize=(11, 6.5)
    )

    x = np.arange(len(plot_df))

    ax.bar(
        x,
        plot_df["roc_auc_mean"],
        yerr=plot_df["roc_auc_std"],
        capsize=4,
    )

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        plot_df["feature_set"],
        rotation=25,
        ha="right",
    )

    ax.set_ylabel("ROC-AUC médio ± DP")
    ax.set_title(
        "Ablation of prompt-derived risk dimensions — Logistic Regression"
    )

    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_delta(comparison_df, out_path):
    d = comparison_df[
        (comparison_df["classifier"] == "logistic_regression")
        & (comparison_df["metric"] == "roc_auc")
    ].copy()

    d = d.sort_values(
        "mean_delta"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    xerr = np.vstack([
        d["mean_delta"]
        - d["bootstrap_ci95_low"],
        d["bootstrap_ci95_high"]
        - d["mean_delta"],
    ])

    y = np.arange(len(d))

    ax.errorbar(
        d["mean_delta"],
        y,
        xerr=xerr,
        fmt="o",
        capsize=4,
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        d["feature_set"]
    )

    ax.set_xlabel(
        "Δ ROC-AUC vs baseline (IC95% bootstrap)"
    )

    ax.set_title(
        "Incremental predictive value of prompt-derived risk dimensions"
    )

    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=True,
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
        default="final_prompt_risk_classifier_results",
    )

    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(args.input)

    if TARGET not in df.columns:
        if "findings_after" not in df.columns:
            raise SystemExit(
                f"Target {TARGET} não encontrado."
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
        + ALL_PROMPT_RISK
        + BASELINE_NUMERIC
        + BASELINE_CATEGORICAL
        + SIZE_DIAGNOSTIC
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
    # Correlação/VIF dos controles de tamanho
    # --------------------------------------------------------
    corr = (
        df[SIZE_DIAGNOSTIC]
        .apply(pd.to_numeric, errors="coerce")
        .corr()
    )

    corr.to_csv(
        out / "size_feature_correlation.csv"
    )

    vif = vif_table(
        df,
        SIZE_DIAGNOSTIC,
    )

    vif.to_csv(
        out / "size_feature_vif.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Avaliação final
    # --------------------------------------------------------
    runs_df = evaluate(
        df,
        runs=args.runs,
        seed=args.seed,
    )

    runs_df.to_csv(
        out / "all_runs.csv",
        index=False,
    )

    summary = summarize_runs(
        runs_df
    )

    summary.to_csv(
        out / "classifier_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Comparações pareadas
    # --------------------------------------------------------
    all_comparisons = []
    all_paired = []

    for i, metric in enumerate(
        ["roc_auc", "pr_auc", "recall_1", "f1_1"]
    ):
        comp, paired = paired_comparisons(
            runs_df,
            metric=metric,
            baseline="Baseline",
            n_boot=args.bootstrap_resamples,
            seed=args.seed + i,
        )

        all_comparisons.append(comp)
        all_paired.append(paired)

    comparison_df = pd.concat(
        all_comparisons,
        ignore_index=True,
    )

    paired_df = pd.concat(
        all_paired,
        ignore_index=True,
    )

    comparison_df.to_csv(
        out / "paired_comparisons_vs_baseline.csv",
        index=False,
    )

    paired_df.to_csv(
        out / "paired_run_deltas.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Figuras
    # --------------------------------------------------------
    plot_auc(
        summary,
        out / "fig_ablation_auc.png",
    )

    plot_delta(
        comparison_df,
        out / "fig_delta_auc_vs_baseline.png",
    )

    # --------------------------------------------------------
    # Resumo
    # --------------------------------------------------------
    report = {
        "rows": int(len(df)),
        "unique_cases": int(
            df["case"].nunique()
        ),
        "positive": int(
            df[TARGET].sum()
        ),
        "negative": int(
            len(df) - df[TARGET].sum()
        ),
        "runs": args.runs,
        "split": (
            "80/20 GroupShuffleSplit; "
            "grouped by case"
        ),
        "baseline": {
            "numeric": BASELINE_NUMERIC,
            "categorical": BASELINE_CATEGORICAL,
        },
        "feature_groups": {
            "Sensitive Operation": SENSITIVE_OPERATION,
            "Technical Specificity": TECHNICAL_SPECIFICITY,
            "Structural / Constraint Signals": STRUCTURAL_CONSTRAINT,
            "All Prompt Risk Features": ALL_PROMPT_RISK,
        },
        "interpretation": (
            "This experiment evaluates incremental predictive value "
            "of prompt-derived features. It does not establish causality."
        ),
    }

    (
        out / "run_summary.json"
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
    print("=" * 100)
    print("FINAL PROMPT RISK CLASSIFIER EXPERIMENT")
    print("=" * 100)

    print(f"Rows      : {len(df)}")
    print(f"Cases     : {df['case'].nunique()}")
    print(f"Positivos : {int(df[TARGET].sum())}")
    print()

    print("VIF — SIZE CONTROLS")
    print(
        vif.to_string(index=False)
    )

    print()
    print("CLASSIFIER SUMMARY")
    print(
        summary[
            [
                "classifier",
                "feature_set",
                "roc_auc_mean",
                "roc_auc_std",
                "pr_auc_mean",
                "recall_1_mean",
                "f1_1_mean",
            ]
        ].sort_values(
            ["classifier", "roc_auc_mean"],
            ascending=[True, False],
        ).to_string(index=False)
    )

    print()
    print("ROC-AUC — PAIRED COMPARISONS VS BASELINE")

    auc_comp = comparison_df[
        comparison_df["metric"].eq(
            "roc_auc"
        )
    ]

    print(
        auc_comp[
            [
                "classifier",
                "feature_set",
                "baseline_mean",
                "feature_set_mean",
                "mean_delta",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
                "wilcoxon_p_value",
                "rank_biserial",
                "positive_runs",
                "n_pairs",
            ]
        ].sort_values(
            ["classifier", "mean_delta"],
            ascending=[True, False],
        ).to_string(index=False)
    )

    print()
    print(f"Saída: {out}")


if __name__ == "__main__":
    main()
