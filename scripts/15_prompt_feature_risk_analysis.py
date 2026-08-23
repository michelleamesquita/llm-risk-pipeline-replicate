#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
15_prompt_feature_risk_analysis.py

Objetivo
--------
Responder diretamente:

"Quais características do prompt estão associadas a maior ou menor risco?"

O script usa o CSV limpo gerado pelo script 14, com features extraídas
EXCLUSIVAMENTE do problem_statement original do SWE-bench.

Para cada feature binária/indicadora calcula:
- prevalência da feature;
- risco observado quando a feature está presente;
- risco observado quando a feature está ausente;
- diferença absoluta de risco (Risk Difference);
- Risk Ratio (RR);
- Odds Ratio (OR);
- IC95% de RR;
- IC95% de OR;
- Fisher exact test (2x2);
- correção de múltiplas comparações por Benjamini-Hochberg (FDR);
- direção do efeito (aumenta / reduz / inconclusivo).

Para features numéricas/de contagem:
- também cria uma versão binária presence = feature > 0;
- calcula Spearman rho com o target;
- opcionalmente compara quartil superior vs quartil inferior.

Target padrão:
    has_finding_after = findings_after > 0

Entrada recomendada:
    problem_statement_ablation_results/
      case_model_problem_statement_features.csv

Uso:
python scripts/15_prompt_feature_risk_analysis.py \
  --input problem_statement_ablation_results/case_model_problem_statement_features.csv \
  --out-dir prompt_feature_risk_results

Saídas:
- prompt_feature_risk_table.csv
- prompt_feature_spearman.csv
- prompt_feature_top_vs_bottom_quartile.csv
- fig_prompt_feature_risk_ratio.png
- fig_prompt_feature_risk_difference.png
- run_summary.json

Nota metodológica
-----------------
Essas análises são ASSOCIATIVAS, não causais.
Um RR > 1 não prova que a feature causa vulnerabilidade.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import fisher_exact, spearmanr, norm


# ============================================================
# CONFIGURAÇÃO
# ============================================================

EXCLUDE_COLUMNS = {
    "case",
    "model",
    "findings_after",
    "has_finding_after",
    "problem_statement_chars",
    "findings_before",
    "findings_new",
    "findings_resolved_est",
    "high_before",
    "high_after",
    "high_new",
    "risk_before",
    "risk_after",
    "delta_risk",
    "risk_introduced",
}

FEATURE_PREFIX = "ps_"


# ============================================================
# UTILITÁRIOS
# ============================================================

def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """
    Ajuste FDR Benjamini-Hochberg.
    """
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)

    n = len(p)
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    ranked = np.empty(n, dtype=float)
    ranked[:] = np.nan

    valid_order = [idx for idx in order if np.isfinite(p[idx])]
    m = len(valid_order)

    if m == 0:
        return pd.Series(ranked, index=p_values.index)

    adjusted = np.empty(m, dtype=float)

    for rank, idx in enumerate(valid_order, start=1):
        adjusted[rank - 1] = p[idx] * m / rank

    # monotonicidade reversa
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)

    for adj, idx in zip(adjusted, valid_order):
        ranked[idx] = adj

    return pd.Series(ranked, index=p_values.index)


def contingency_counts(x: np.ndarray, y: np.ndarray):
    """
    2x2:
              y=1   y=0
      x=1      a     b
      x=0      c     d
    """
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)

    a = int(np.sum((x == 1) & (y == 1)))
    b = int(np.sum((x == 1) & (y == 0)))
    c = int(np.sum((x == 0) & (y == 1)))
    d = int(np.sum((x == 0) & (y == 0)))

    return a, b, c, d


def haldane_anscombe(a, b, c, d):
    """
    Correção +0.5 se houver célula zero.
    """
    if min(a, b, c, d) == 0:
        return a + 0.5, b + 0.5, c + 0.5, d + 0.5
    return float(a), float(b), float(c), float(d)


def risk_ratio_and_ci(a, b, c, d, alpha=0.05):
    """
    RR = risk(feature present) / risk(feature absent)
    CI por aproximação log.
    """
    aa, bb, cc, dd = haldane_anscombe(a, b, c, d)

    risk1 = aa / (aa + bb)
    risk0 = cc / (cc + dd)

    rr = risk1 / risk0 if risk0 > 0 else math.nan

    if not np.isfinite(rr) or rr <= 0:
        return rr, math.nan, math.nan

    se_log_rr = math.sqrt(
        (1 / aa) - (1 / (aa + bb))
        + (1 / cc) - (1 / (cc + dd))
    )

    z = norm.ppf(1 - alpha / 2)

    low = math.exp(math.log(rr) - z * se_log_rr)
    high = math.exp(math.log(rr) + z * se_log_rr)

    return rr, low, high


def odds_ratio_and_ci(a, b, c, d, alpha=0.05):
    """
    OR e IC95% por log(OR).
    """
    aa, bb, cc, dd = haldane_anscombe(a, b, c, d)

    or_value = (aa * dd) / (bb * cc)

    if or_value <= 0:
        return or_value, math.nan, math.nan

    se_log_or = math.sqrt(
        1 / aa + 1 / bb + 1 / cc + 1 / dd
    )

    z = norm.ppf(1 - alpha / 2)

    low = math.exp(math.log(or_value) - z * se_log_or)
    high = math.exp(math.log(or_value) + z * se_log_or)

    return or_value, low, high


def effect_direction(rr, ci_low, ci_high, fdr_p):
    if not np.isfinite(rr):
        return "inconclusive"

    if (
        np.isfinite(ci_low)
        and np.isfinite(ci_high)
        and np.isfinite(fdr_p)
        and fdr_p < 0.05
    ):
        if ci_low > 1:
            return "higher risk"
        if ci_high < 1:
            return "lower risk"

    return "inconclusive"


# ============================================================
# ANÁLISE BINÁRIA
# ============================================================

def analyze_binary_feature(
    df: pd.DataFrame,
    feature: str,
    target: str,
) -> dict:

    values = pd.to_numeric(
        df[feature],
        errors="coerce",
    ).fillna(0)

    x = (values > 0).astype(int).to_numpy()
    y = df[target].astype(int).to_numpy()

    a, b, c, d = contingency_counts(x, y)

    n_present = a + b
    n_absent = c + d

    risk_present = (
        a / n_present
        if n_present > 0
        else math.nan
    )

    risk_absent = (
        c / n_absent
        if n_absent > 0
        else math.nan
    )

    risk_difference = (
        risk_present - risk_absent
        if np.isfinite(risk_present) and np.isfinite(risk_absent)
        else math.nan
    )

    rr, rr_low, rr_high = risk_ratio_and_ci(
        a, b, c, d
    )

    or_value, or_low, or_high = odds_ratio_and_ci(
        a, b, c, d
    )

    try:
        _, fisher_p = fisher_exact(
            [[a, b], [c, d]],
            alternative="two-sided",
        )
    except Exception:
        fisher_p = math.nan

    return {
        "feature": feature,
        "n_total": int(len(df)),
        "n_present": int(n_present),
        "n_absent": int(n_absent),
        "prevalence": (
            n_present / len(df)
            if len(df)
            else math.nan
        ),
        "risk_present": risk_present,
        "risk_absent": risk_absent,
        "risk_difference": risk_difference,
        "risk_ratio": rr,
        "rr_ci95_low": rr_low,
        "rr_ci95_high": rr_high,
        "odds_ratio": or_value,
        "or_ci95_low": or_low,
        "or_ci95_high": or_high,
        "fisher_p_value": fisher_p,
        "a_present_risk": a,
        "b_present_no_risk": b,
        "c_absent_risk": c,
        "d_absent_no_risk": d,
    }


# ============================================================
# SPEARMAN / QUARTIS
# ============================================================

def spearman_analysis(
    df: pd.DataFrame,
    features: list[str],
    target: str,
) -> pd.DataFrame:

    rows = []

    y = df[target].astype(int)

    for feature in features:
        x = pd.to_numeric(
            df[feature],
            errors="coerce",
        ).fillna(0)

        if x.nunique() <= 1:
            rho = math.nan
            p = math.nan
        else:
            rho, p = spearmanr(x, y)

        rows.append({
            "feature": feature,
            "spearman_rho": rho,
            "spearman_p_value": p,
            "n_unique_values": int(x.nunique()),
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        result["spearman_fdr_bh"] = benjamini_hochberg(
            result["spearman_p_value"]
        )

    return result


def top_bottom_quartile_analysis(
    df: pd.DataFrame,
    features: list[str],
    target: str,
) -> pd.DataFrame:

    rows = []

    y = df[target].astype(int)

    for feature in features:
        x = pd.to_numeric(
            df[feature],
            errors="coerce",
        ).fillna(0)

        if x.nunique() < 4:
            continue

        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)

        low_mask = x <= q1
        high_mask = x >= q3

        low_n = int(low_mask.sum())
        high_n = int(high_mask.sum())

        if low_n == 0 or high_n == 0:
            continue

        low_risk = float(
            y[low_mask].mean()
        )

        high_risk = float(
            y[high_mask].mean()
        )

        rr = (
            high_risk / low_risk
            if low_risk > 0
            else math.nan
        )

        rows.append({
            "feature": feature,
            "q1": q1,
            "q3": q3,
            "low_n": low_n,
            "high_n": high_n,
            "risk_bottom_quartile": low_risk,
            "risk_top_quartile": high_risk,
            "risk_difference_top_minus_bottom": (
                high_risk - low_risk
            ),
            "risk_ratio_top_vs_bottom": rr,
        })

    return pd.DataFrame(rows)


# ============================================================
# FIGURAS
# ============================================================

def plot_risk_ratio(
    table: pd.DataFrame,
    out_path: Path,
    top_n: int = 20,
):
    data = table.copy()

    data = data[
        np.isfinite(data["risk_ratio"])
        & np.isfinite(data["rr_ci95_low"])
        & np.isfinite(data["rr_ci95_high"])
    ].copy()

    # Evita features quase constantes.
    data = data[
        (data["prevalence"] >= 0.05)
        & (data["prevalence"] <= 0.95)
    ]

    if data.empty:
        return

    data["distance_from_null"] = np.abs(
        np.log(data["risk_ratio"])
    )

    data = (
        data
        .sort_values(
            "distance_from_null",
            ascending=False,
        )
        .head(top_n)
        .sort_values(
            "risk_ratio",
            ascending=True,
        )
    )

    y = np.arange(len(data))

    x = data["risk_ratio"].to_numpy()

    xerr = np.vstack([
        x - data["rr_ci95_low"].to_numpy(),
        data["rr_ci95_high"].to_numpy() - x,
    ])

    fig, ax = plt.subplots(
        figsize=(10, max(5, 0.45 * len(data)))
    )

    ax.errorbar(
        x,
        y,
        xerr=xerr,
        fmt="o",
        capsize=3,
    )

    ax.axvline(
        1.0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        data["feature"]
    )

    ax.set_xlabel(
        "Risk Ratio (95% CI)"
    )

    ax.set_title(
        "Prompt features associated with observed risk"
    )

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_risk_difference(
    table: pd.DataFrame,
    out_path: Path,
    top_n: int = 20,
):
    data = table.copy()

    data = data[
        np.isfinite(
            data["risk_difference"]
        )
    ].copy()

    data = data[
        (data["prevalence"] >= 0.05)
        & (data["prevalence"] <= 0.95)
    ]

    if data.empty:
        return

    data = (
        data
        .assign(
            abs_rd=lambda d: np.abs(
                d["risk_difference"]
            )
        )
        .sort_values(
            "abs_rd",
            ascending=False,
        )
        .head(top_n)
        .sort_values(
            "risk_difference",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, max(5, 0.45 * len(data)))
    )

    ax.barh(
        data["feature"],
        data["risk_difference"],
    )

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Absolute risk difference"
    )

    ax.set_title(
        "Risk difference when prompt feature is present"
    )

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


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
        "--out-dir",
        default="prompt_feature_risk_results",
    )

    ap.add_argument(
        "--target",
        default="has_finding_after",
    )

    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        input_path
    )

    # Target
    if args.target not in df.columns:
        if (
            args.target == "has_finding_after"
            and "findings_after" in df.columns
        ):
            df["has_finding_after"] = (
                pd.to_numeric(
                    df["findings_after"],
                    errors="coerce",
                )
                .fillna(0)
                .gt(0)
                .astype(int)
            )
        else:
            raise SystemExit(
                f"Target ausente: {args.target}"
            )

    # Features do problem_statement
    feature_cols = [
        c
        for c in df.columns
        if c.startswith(FEATURE_PREFIX)
        and c not in EXCLUDE_COLUMNS
    ]

    if not feature_cols:
        raise SystemExit(
            "Nenhuma feature ps_* encontrada."
        )

    # --------------------------------------------------------
    # Binary presence analysis
    # --------------------------------------------------------
    binary_rows = [
        analyze_binary_feature(
            df,
            feature,
            args.target,
        )
        for feature in feature_cols
    ]

    binary_table = pd.DataFrame(
        binary_rows
    )

    binary_table[
        "fisher_fdr_bh"
    ] = benjamini_hochberg(
        binary_table[
            "fisher_p_value"
        ]
    )

    binary_table[
        "effect_direction"
    ] = [
        effect_direction(
            rr,
            low,
            high,
            fdr,
        )
        for rr, low, high, fdr in zip(
            binary_table["risk_ratio"],
            binary_table["rr_ci95_low"],
            binary_table["rr_ci95_high"],
            binary_table["fisher_fdr_bh"],
        )
    ]

    binary_table = binary_table.sort_values(
        [
            "fisher_fdr_bh",
            "risk_ratio",
        ],
        ascending=[
            True,
            False,
        ],
        na_position="last",
    )

    binary_table.to_csv(
        out_dir
        / "prompt_feature_risk_table.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Spearman
    # --------------------------------------------------------
    spearman_table = spearman_analysis(
        df,
        feature_cols,
        args.target,
    )

    spearman_table.to_csv(
        out_dir
        / "prompt_feature_spearman.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Quartis
    # --------------------------------------------------------
    quartile_table = top_bottom_quartile_analysis(
        df,
        feature_cols,
        args.target,
    )

    quartile_table.to_csv(
        out_dir
        / "prompt_feature_top_vs_bottom_quartile.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------
    plot_risk_ratio(
        binary_table,
        out_dir
        / "fig_prompt_feature_risk_ratio.png",
    )

    plot_risk_difference(
        binary_table,
        out_dir
        / "fig_prompt_feature_risk_difference.png",
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    significant = binary_table[
        binary_table["fisher_fdr_bh"] < 0.05
    ]

    summary = {
        "rows": int(len(df)),
        "unique_cases": (
            int(df["case"].nunique())
            if "case" in df.columns
            else None
        ),
        "target": args.target,
        "positive": int(
            df[args.target].sum()
        ),
        "negative": int(
            len(df)
            - df[args.target].sum()
        ),
        "n_features_analyzed": int(
            len(feature_cols)
        ),
        "n_features_fdr_lt_0_05": int(
            len(significant)
        ),
        "feature_prefix": FEATURE_PREFIX,
        "interpretation_note": (
            "Associations are not causal. RR/OR indicate association "
            "between prompt-derived features and observed SAST findings."
        ),
    }

    (
        out_dir
        / "run_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------
    print()
    print("=" * 90)
    print("PROMPT FEATURE RISK ANALYSIS")
    print("=" * 90)
    print(f"Rows      : {len(df)}")
    print(
        f"Cases     : "
        f"{df['case'].nunique() if 'case' in df.columns else 'N/A'}"
    )
    print(
        f"Positivos : "
        f"{int(df[args.target].sum())}"
    )
    print(
        f"Features  : "
        f"{len(feature_cols)}"
    )
    print()

    show = binary_table[
        [
            "feature",
            "prevalence",
            "risk_present",
            "risk_absent",
            "risk_difference",
            "risk_ratio",
            "rr_ci95_low",
            "rr_ci95_high",
            "fisher_p_value",
            "fisher_fdr_bh",
            "effect_direction",
        ]
    ].head(25)

    print(show.to_string(index=False))

    print()
    print(
        f"Features FDR<0.05: "
        f"{len(significant)}"
    )
    print()
    print(f"Saída: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
