#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
11_classifier_article_protocol.py

Classificador alinhado ao artigo:
- Regressão Logística (baseline)
- Random Forest (principal)
- engenharia de atributos do artigo
- 30 repetições
- split 80/20 AGRUPADO POR CASE para evitar leakage
- métricas por classe
- matriz de confusão
- importância de features
- SHAP beeswarm (se shap estiver instalado)

Entrada:
  case_model_before_after_summary_common.csv
ou
  risk_analysis_case_model.csv

Alvo recomendado agora:
  has_finding_after = findings_after > 0

Quando houver 300 cases e delta_risk robusto:
  risk_increased = delta_risk > 0
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

warnings.filterwarnings("ignore")

BASE_NUMERIC = [
    "patch_lines",
    "patch_added",
    "patch_removed",
    "patch_files_touched",
    "patch_hunks",
    "patch_churn",
    "patch_net",
    "prompt_chars",
    "prompt_lines",
    "prompt_tokens",
    "prompt_has_security_guidelines",
]

DERIVED = [
    "patch_density",
    "add_remove_ratio",
    "net_per_line",
    "hunks_per_file",
    "prompt_density",
    "prompt_token_density",
    "prompt_size_category",
    "patch_complexity",
    "change_intensity",
]

CAT_FEATURES = ["model"]


def div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)


def add_features(df):
    df = df.copy()

    for c in BASE_NUMERIC:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["patch_density"] = div(df["patch_churn"], np.maximum(df["patch_lines"], 1))
    df["add_remove_ratio"] = div(df["patch_added"], np.maximum(df["patch_removed"], 1))
    df["net_per_line"] = div(df["patch_net"], np.maximum(df["patch_lines"], 1))
    df["hunks_per_file"] = div(df["patch_hunks"], np.maximum(df["patch_files_touched"], 1))
    df["prompt_density"] = div(df["prompt_chars"], np.maximum(df["prompt_lines"], 1))
    df["prompt_token_density"] = div(df["prompt_chars"], np.maximum(df["prompt_tokens"], 1))

    # Categoria ordinal simples, inspirada no artigo.
    chars = df["prompt_chars"]
    q1, q2 = chars.quantile([0.33, 0.66]).tolist()
    df["prompt_size_category"] = np.select(
        [chars <= q1, chars <= q2],
        [0, 1],
        default=2,
    )

    df["patch_complexity"] = df["patch_hunks"] * df["patch_files_touched"]
    df["change_intensity"] = div(
        df["patch_churn"],
        np.maximum(df["patch_files_touched"], 1)
    )

    return df


def make_target(df, target):
    if target == "has_finding_after":
        return (df["findings_after"] > 0).astype(int)

    if target == "has_high_after":
        return (df["high_after"] > 0).astype(int)

    if target == "has_new_finding":
        return (df["findings_new"] > 0).astype(int)

    if target == "risk_increased":
        if "delta_risk" not in df.columns:
            raise ValueError("delta_risk não existe no CSV.")
        return (df["delta_risk"] > 0).astype(int)

    raise ValueError(f"Target desconhecido: {target}")


def preprocessing():
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", MinMaxScaler()),
    ])

    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", num, BASE_NUMERIC + DERIVED),
        ("cat", cat, CAT_FEATURES),
    ])


def factories(seed):
    return {
        "logistic_regression": Pipeline([
            ("prep", preprocessing()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=seed,
            )),
        ]),
        "random_forest": Pipeline([
            ("prep", preprocessing()),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )),
        ]),
    }


def calc_metrics(y, pred, prob):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision_1": precision_score(y, pred, zero_division=0),
        "recall_1": recall_score(y, pred, zero_division=0),
        "f1_1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def feature_names(pipe):
    prep = pipe.named_steps["prep"]

    names = list(BASE_NUMERIC + DERIVED)

    ohe = prep.named_transformers_["cat"].named_steps["onehot"]
    names += ohe.get_feature_names_out(CAT_FEATURES).tolist()

    return names


def plot_metrics(metrics_summary, path):
    labels = metrics_summary["model"].tolist()
    precision = metrics_summary["precision_1_mean"].tolist()
    recall = metrics_summary["recall_1_mean"].tolist()
    f1 = metrics_summary["f1_1_mean"].tolist()

    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Métricas da classe de risco")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Sem risco", "Risco"])
    ax.set_yticklabels(["Sem risco", "Risco"])
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão — Random Forest")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_importance(importance, path, top=15):
    d = importance.head(top).sort_values("importance")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(d["feature"], d["importance"])
    ax.set_xlabel("Importância")
    ax.set_title("Top Features — Random Forest")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def try_shap(pipe, X, out_path, max_rows=300):
    try:
        import shap
    except Exception:
        return False, "shap não instalado"

    prep = pipe.named_steps["prep"]
    rf = pipe.named_steps["clf"]

    Xt = prep.transform(X)

    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()

    names = feature_names(pipe)

    if len(Xt) > max_rows:
        idx = np.random.RandomState(42).choice(
            len(Xt), size=max_rows, replace=False
        )
        Xt = Xt[idx]

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(Xt)

    # Compatibilidade entre versões do SHAP.
    if isinstance(shap_values, list):
        values = shap_values[1]
    elif getattr(shap_values, "ndim", 0) == 3:
        values = shap_values[:, :, 1]
    else:
        values = shap_values

    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        values,
        Xt,
        feature_names=names,
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return True, ""


def run_experiment(df, target, runs, seed, out_dir):
    df = add_features(df)
    y = make_target(df, target).to_numpy()
    groups = df["case"].astype(str).to_numpy()

    feature_cols = BASE_NUMERIC + DERIVED + CAT_FEATURES
    X = df[feature_cols]

    if y.sum() < 10:
        raise RuntimeError(
            f"Target muito esparso: somente {int(y.sum())} positivos."
        )

    rows = []
    pred_rows = []

    # Guarda a matriz acumulada da RF para figura.
    rf_cm = np.zeros((2, 2), dtype=int)

    for run_idx in range(runs):
        run_seed = seed + run_idx

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=run_seed,
        )
        train_idx, test_idx = next(
            splitter.split(X, y, groups=groups)
        )

        # Proteção: exige duas classes no train/test.
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue

        for model_name, pipe in factories(run_seed).items():
            pipe.fit(X.iloc[train_idx], y[train_idx])

            prob = pipe.predict_proba(X.iloc[test_idx])[:, 1]
            pred = (prob >= 0.5).astype(int)

            m = calc_metrics(y[test_idx], pred, prob)
            m.update({
                "run": run_idx + 1,
                "seed": run_seed,
                "model": model_name,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "test_cases": len(np.unique(groups[test_idx])),
            })
            rows.append(m)

            p = pd.DataFrame({
                "case": groups[test_idx],
                "llm_model": df.iloc[test_idx]["model"].values,
                "classifier": model_name,
                "run": run_idx + 1,
                "y_true": y[test_idx],
                "y_pred": pred,
                "probability": prob,
            })
            pred_rows.append(p)

            if model_name == "random_forest":
                rf_cm += confusion_matrix(
                    y[test_idx], pred, labels=[0, 1]
                )

    results = pd.DataFrame(rows)
    preds = pd.concat(pred_rows, ignore_index=True)

    results.to_csv(out_dir / "all_runs_metrics.csv", index=False)
    preds.to_csv(out_dir / "all_runs_predictions.csv", index=False)

    agg = (
        results.groupby("model")
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

    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(out_dir / "metrics_mean_std.csv", index=False)

    plot_metrics(agg, out_dir / "fig_metrics_class1.png")
    plot_confusion(rf_cm, out_dir / "fig_confusion_matrix_rf.png")

    # Modelo de referência RF com dataset completo só para interpretabilidade.
    rf = factories(seed)["random_forest"]
    rf.fit(X, y)

    names = feature_names(rf)
    imps = rf.named_steps["clf"].feature_importances_

    imp = pd.DataFrame({
        "feature": names,
        "importance": imps,
    }).sort_values("importance", ascending=False)

    imp.to_csv(out_dir / "rf_feature_importance.csv", index=False)
    plot_importance(imp, out_dir / "fig_rf_feature_importance.png")

    shap_ok, shap_error = try_shap(
        rf,
        X,
        out_dir / "fig_shap_beeswarm.png",
    )

    summary = {
        "rows": len(df),
        "unique_cases": int(df["case"].nunique()),
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
        "target": target,
        "runs_requested": runs,
        "runs_completed": int(results["run"].nunique()),
        "split": "80/20 GroupShuffleSplit by case",
        "base_seed": seed,
        "models": {
            "logistic_regression": {
                "role": "baseline",
            },
            "random_forest": {
                "role": "main",
                "n_estimators": 100,
                "max_depth": 15,
                "class_weight": "balanced",
            },
        },
        "temperature_features_removed": True,
        "reason_temperature_removed": (
            "temperature/top_p/top_k are not uniformly exposed by all "
            "Replicate endpoints in the current experiment"
        ),
        "shap_generated": shap_ok,
        "shap_error": shap_error or None,
    }

    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return agg, imp, summary


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=True,
        help="CSV case x model comum.",
    )

    ap.add_argument(
        "--target",
        default="has_finding_after",
        choices=[
            "has_finding_after",
            "has_high_after",
            "has_new_finding",
            "risk_increased",
        ],
    )

    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--out-dir",
        default="classifier_article_results",
    )

    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    agg, imp, summary = run_experiment(
        df,
        target=args.target,
        runs=args.runs,
        seed=args.seed,
        out_dir=out,
    )

    print("\n=== DATASET ===")
    print(f"Rows: {summary['rows']}")
    print(f"Cases: {summary['unique_cases']}")
    print(f"Positivos: {summary['positive']}")
    print(f"Negativos: {summary['negative']}")

    print("\n=== MÉTRICAS MÉDIAS ± DP ===")
    print(agg.to_string(index=False))

    print("\n=== TOP FEATURES RF ===")
    print(imp.head(15).to_string(index=False))

    print("\nArquivos gerados:")
    for p in sorted(out.iterdir()):
        print(" -", p)


if __name__ == "__main__":
    main()
