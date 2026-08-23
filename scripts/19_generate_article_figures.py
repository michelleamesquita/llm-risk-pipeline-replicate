#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
19_generate_article_figures.py

Gera as figuras do artigo no cenário robusto/new pipeline, preservando ao máximo
os nomes esperados pelo main.tex antigo.

Saídas padrão em ./imgs:
- distribuicao_variaveis_por_risco.png
- corr.jpg
- RF.png
- rf_lr_repeated_runs_matriz_confusao.png
- rf_feature_importance_article.png
- previsao.png

Observação:
- O SHAP beeswarm continua vindo do script 11_classifier_article_protocol.py
  (ou pode ser copiado de classifier_article_results/fig_shap_beeswarm.png quando existir).
- A figura de pipeline/modelagem é melhor tratada manualmente a partir do diagrama.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


FEATURES_FOR_DISTRIBUTION = [
    "patch_lines",
    "patch_added",
    "patch_removed",
    "patch_files_touched",
    "patch_hunks",
    "patch_churn",
    "patch_net",
]

FEATURES_FOR_CORR = [
    "findings_before",
    "findings_after",
    "findings_new",
    "high_before",
    "high_after",
    "high_new",
    "risk_before",
    "risk_after",
    "delta_risk",
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
]

METRIC_LABELS = {
    "precision_1": "Precision",
    "recall_1": "Recall",
    "f1_1": "F1",
}

ALGO_LABELS = {
    "logistic_regression": "Regressão Logística",
    "random_forest": "Random Forest",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def robust_target(df: pd.DataFrame, target: str) -> pd.Series:
    if target == "has_finding_after":
        return (df["findings_after"] > 0).astype(int)
    if target == "has_high_after":
        return (df["high_after"] > 0).astype(int)
    if target == "has_new_finding":
        return (df["findings_new"] > 0).astype(int)
    if target == "risk_increased":
        return (df["delta_risk"] > 0).astype(int)
    raise ValueError(f"Target desconhecido: {target}")



def load_metrics(metrics_summary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_summary_path)
    expected = {"algorithm", "metric", "mean", "std"}
    if not expected.issubset(df.columns):
        raise ValueError(
            f"Arquivo de métricas não possui colunas esperadas {sorted(expected)}: {metrics_summary_path}"
        )
    return df



def load_predictions(predictions_path: Path) -> pd.DataFrame:
    df = pd.read_csv(predictions_path)
    expected = {"algorithm", "y_true", "y_pred", "y_score"}
    if not expected.issubset(df.columns):
        raise ValueError(
            f"Arquivo de predições não possui colunas esperadas {sorted(expected)}: {predictions_path}"
        )
    return df



def load_importance(importance_path: Path) -> pd.DataFrame:
    df = pd.read_csv(importance_path)
    expected = {"feature", "importance"}
    if not expected.issubset(df.columns):
        raise ValueError(
            f"Arquivo de importância não possui colunas esperadas {sorted(expected)}: {importance_path}"
        )
    return df.sort_values("importance", ascending=False).reset_index(drop=True)



def distribution_figure(df: pd.DataFrame, y: pd.Series, out_path: Path) -> None:
    available = [c for c in FEATURES_FOR_DISTRIBUTION if c in df.columns]

    if not available:
        raise ValueError(
            "Nenhuma feature de distribuição encontrada no CSV."
        )

    n = len(available)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(18, 4.8 * nrows)
    )

    axes = np.array(axes).reshape(-1)

    color0 = "#2ca02c"  # Sem risco
    color1 = "#ff6b6b"  # Com risco

    for ax, col in zip(axes, available):

        vals0 = (
            pd.to_numeric(
                df.loc[y == 0, col],
                errors="coerce"
            )
            .dropna()
            .to_numpy()
        )

        vals1 = (
            pd.to_numeric(
                df.loc[y == 1, col],
                errors="coerce"
            )
            .dropna()
            .to_numpy()
        )

        if len(vals0) and len(vals1):
            combined = np.concatenate([vals0, vals1])
        elif len(vals0):
            combined = vals0
        else:
            combined = vals1

        if len(combined) == 0:
            ax.set_visible(False)
            continue

        # Limita apenas caudas muito extremas para
        # evitar que poucos outliers comprimam o gráfico.
        upper = np.nanpercentile(combined, 99)

        if np.nanmin(combined) < 0:
            lower = np.nanpercentile(combined, 1)
        else:
            lower = max(0, np.nanmin(combined))

        if upper <= lower:
            lower = np.nanmin(combined)
            upper = np.nanmax(combined)

        bins = np.linspace(lower, upper, 30)

        # Histogramas sobrepostos
        ax.hist(
            vals0,
            bins=bins,
            alpha=0.35,
            color=color0,
            edgecolor="black",
            linewidth=0.7,
            label="Sem risco",
        )

        ax.hist(
            vals1,
            bins=bins,
            alpha=0.35,
            color=color1,
            edgecolor="black",
            linewidth=0.7,
            label="Com risco",
        )

        # Linha sobre o histograma da classe 0
        if len(vals0) > 1:
            counts0, edges0 = np.histogram(
                vals0,
                bins=bins
            )

            mids0 = (
                edges0[:-1] + edges0[1:]
            ) / 2

            ax.plot(
                mids0,
                counts0,
                color=color0,
                linewidth=1.5,
            )

        # Linha sobre o histograma da classe 1
        if len(vals1) > 1:
            counts1, edges1 = np.histogram(
                vals1,
                bins=bins
            )

            mids1 = (
                edges1[:-1] + edges1[1:]
            ) / 2

            ax.plot(
                mids1,
                counts1,
                color="red",
                linewidth=1.5,
            )

        ax.set_title(
            f"Distribuicao de {col}"
        )

        ax.set_xlabel(col)
        ax.set_ylabel("Count")

        ax.grid(
            axis="y",
            alpha=0.2
        )

        ax.legend(
            loc="upper right",
            fontsize=8
        )

    # Remove painéis vazios
    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=220,
        bbox_inches="tight"
    )

    plt.close(fig)



def correlation_figure(df: pd.DataFrame, y: pd.Series, out_path: Path) -> None:
    cols = [c for c in FEATURES_FOR_CORR if c in df.columns]
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    numeric["target_class"] = y
    corr = numeric.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    ax.set_title("Matriz de correlação — cenário robusto")

    # valores só nas correlações mais fortes, para não poluir
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            val = corr.iloc[i, j]
            if abs(val) >= 0.65 or i == j:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)



def metrics_figure(metrics_df: pd.DataFrame, out_path: Path) -> None:
    wanted = ["precision_1", "recall_1", "f1_1"]
    subset = metrics_df[metrics_df["metric"].isin(wanted)].copy()

    algos = [a for a in ["logistic_regression", "random_forest"] if a in subset["algorithm"].unique()]
    x = np.arange(len(algos))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for idx, metric in enumerate(wanted):
        vals = []
        errs = []
        for algo in algos:
            row = subset[(subset["algorithm"] == algo) & (subset["metric"] == metric)]
            vals.append(float(row["mean"].iloc[0]))
            errs.append(float(row["std"].iloc[0]))
        ax.bar(x + (idx - 1) * width, vals, width, yerr=errs, capsize=3, label=METRIC_LABELS[metric])

    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABELS.get(a, a) for a in algos], rotation=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Desempenho preditivo da classe positiva")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)



def _confusion_from_df(df: pd.DataFrame) -> np.ndarray:
    y_true = df["y_true"].astype(int).to_numpy()
    y_pred = df["y_pred"].astype(int).to_numpy()
    cm = np.zeros((2, 2), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[yt, yp] += 1
    return cm



def confusion_figure(pred_df: pd.DataFrame, out_path: Path) -> None:
    algos = [a for a in ["logistic_regression", "random_forest"] if a in pred_df["algorithm"].unique()]
    fig, axes = plt.subplots(1, len(algos), figsize=(5.6 * len(algos), 4.8))
    if len(algos) == 1:
        axes = [axes]

    for ax, algo in zip(axes, algos):
        cm = _confusion_from_df(pred_df[pred_df["algorithm"] == algo])
        im = ax.imshow(cm)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Classe 0", "Classe 1"])
        ax.set_yticklabels(["Classe 0", "Classe 1"])
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_title(ALGO_LABELS.get(algo, algo))
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]}", ha="center", va="center", fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Matrizes de confusão agregadas nas 30 execuções", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)



def importance_figure(importance_df: pd.DataFrame, out_path: Path, top_n: int = 15) -> None:
    top = importance_df.head(top_n).sort_values("importance")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Importância")
    ax.set_title("Top features — Random Forest")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)



def prediction_score_figure(pred_df: pd.DataFrame, out_path: Path) -> None:
    algos = [a for a in ["logistic_regression", "random_forest"] if a in pred_df["algorithm"].unique()]
    fig, axes = plt.subplots(1, len(algos), figsize=(5.8 * len(algos), 4.8))
    if len(algos) == 1:
        axes = [axes]

    bins = np.linspace(0, 1, 16)
    for ax, algo in zip(axes, algos):
        d = pred_df[pred_df["algorithm"] == algo].copy()
        score = d["y_score"].astype(float)
        y = d["y_true"].astype(int)
        ax.hist(score[y == 0], bins=bins, alpha=0.55, label="Classe 0")
        ax.hist(score[y == 1], bins=bins, alpha=0.55, label="Classe 1")
        ax.set_title(ALGO_LABELS.get(algo, algo))
        ax.set_xlabel("Probabilidade prevista")
        ax.set_ylabel("Frequência")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    fig.suptitle("Distribuição dos scores previstos", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", required=True, help="case_model_before_after_summary_common.csv")
    ap.add_argument("--metrics-summary", required=True, help="lr_rf_full_metrics_summary.csv")
    ap.add_argument("--predictions", required=True, help="lr_rf_full_predictions.csv")
    ap.add_argument("--importance-csv", required=True, help="rf_feature_importance.csv ou permutation importance")
    ap.add_argument("--out-dir", default="imgs")
    ap.add_argument(
        "--target",
        default="has_finding_after",
        choices=["has_finding_after", "has_high_after", "has_new_finding", "risk_increased"],
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    summary_df = pd.read_csv(args.summary_csv)
    y = robust_target(summary_df, args.target)

    metrics_df = load_metrics(Path(args.metrics_summary))
    pred_df = load_predictions(Path(args.predictions))
    imp_df = load_importance(Path(args.importance_csv))

    distribution_figure(summary_df, y, out_dir / "distribuicao_variaveis_por_risco.png")
    correlation_figure(summary_df, y, out_dir / "corr.jpg")
    metrics_figure(metrics_df, out_dir / "RF.png")
    confusion_figure(pred_df, out_dir / "rf_lr_repeated_runs_matriz_confusao.png")
    importance_figure(imp_df, out_dir / "rf_feature_importance_article.png")
    prediction_score_figure(pred_df, out_dir / "previsao.png")

    print("Figuras geradas em:", out_dir)
    for name in [
        "distribuicao_variaveis_por_risco.png",
        "corr.jpg",
        "RF.png",
        "rf_lr_repeated_runs_matriz_confusao.png",
        "rf_feature_importance_article.png",
        "previsao.png",
    ]:
        print(" -", out_dir / name)


if __name__ == "__main__":
    main()