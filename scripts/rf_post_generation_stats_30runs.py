#!/usr/bin/env python3
"""
Estatística pareada para o experimento RF pós-geração / pré-SAST (30 runs).

Usa os CSVs produzidos por rf_post_generation_30runs.py e compara os mesmos
splits/run seeds de forma pareada.

Testes:
  1) Wilcoxon signed-rank pareado (duas caudas)
  2) Wilcoxon direcional: H1 = delta > 0
  3) Bootstrap percentile 95% CI da diferença média entre runs
  4) Bootstrap 95% CI da mediana
  5) Rank-biserial correlation como tamanho de efeito
  6) Benjamini-Hochberg FDR para múltiplas comparações

Convenção:
  delta > 0 = o modelo FULL foi melhor.

Entradas esperadas no --outdir:
  rf_post_generation_feature_sets_all_runs.csv
  rf_post_generation_group_ablation_all_runs.csv

Saídas:
  rf_post_generation_paired_feature_sets.csv
  rf_post_generation_paired_ablation.csv
  rf_post_generation_paired_tests_all.csv
  rf_post_generation_paired_deltas_all_runs.csv
  fig_rf_post_generation_paired_roc_auc.png
  rf_post_generation_stats_summary.txt

Exemplo:
  python rf_post_generation_stats_30runs.py \
      --outdir results_rf_post_generation \
      --bootstrap 20000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


EXPECTED_RUNS = 30
BASELINE_SET = "full"

# Comparações que interessam para a narrativa principal.
FEATURE_SET_COMPARISONS = [
    "patch_only",
    "patch_plus_problem_statement",
    "model_plus_problem_statement",
    "problem_statement_only",
    "model_only",
    "prompt_only",
]

ABLATION_GROUPS = [
    "problem_statement",
    "prompt_envelope",
    "model_identity",
    "patch_structure",
]


def bootstrap_ci(
    values: np.ndarray,
    statistic: str = "mean",
    n_boot: int = 20000,
    seed: int = 2026,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap percentile CI sobre os deltas pareados entre runs."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    n = len(x)

    if statistic == "mean":
        stat_fn = np.mean
    elif statistic == "median":
        stat_fn = np.median
    else:
        raise ValueError("statistic deve ser 'mean' ou 'median'")

    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sample = rng.choice(x, size=n, replace=True)
        stats[b] = stat_fn(sample)

    low = np.quantile(stats, alpha / 2)
    high = np.quantile(stats, 1 - alpha / 2)
    return float(low), float(high)


def rank_biserial_paired(deltas: np.ndarray) -> float:
    """
    Rank-biserial correlation para diferenças pareadas.
    +1: todos os ranks favorecem FULL
    -1: todos os ranks favorecem comparador
     0: equilíbrio.
    """
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


def wilcoxon_safe(deltas: np.ndarray, alternative: str) -> tuple[float, float]:
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]

    if len(d) == 0 or np.allclose(d, 0):
        return 0.0, 1.0

    # zero_method='wilcox' descarta diferenças exatamente zero.
    # method='auto' deixa scipy escolher aproximação/exato quando aplicável.
    res = wilcoxon(
        d,
        zero_method="wilcox",
        correction=False,
        alternative=alternative,
        method="auto",
    )
    return float(res.statistic), float(res.pvalue)


def benjamini_hochberg(pvalues: pd.Series) -> np.ndarray:
    """Benjamini-Hochberg FDR, sem dependência de statsmodels."""
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

    # Garante monotonicidade de trás para frente.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    out_valid = np.empty(m, dtype=float)
    out_valid[order] = adjusted
    q[valid] = out_valid
    return q


def describe_delta(
    deltas: np.ndarray,
    comparison: str,
    family: str,
    metric: str,
    n_boot: int,
    bootstrap_seed: int,
) -> dict:
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]

    mean_delta = float(np.mean(d))
    median_delta = float(np.median(d))
    std_delta = float(np.std(d, ddof=1)) if len(d) > 1 else np.nan

    mean_low, mean_high = bootstrap_ci(
        d, "mean", n_boot=n_boot, seed=bootstrap_seed
    )
    med_low, med_high = bootstrap_ci(
        d, "median", n_boot=n_boot, seed=bootstrap_seed + 1
    )

    w2_stat, p_two = wilcoxon_safe(d, alternative="two-sided")
    wg_stat, p_greater = wilcoxon_safe(d, alternative="greater")

    return {
        "family": family,
        "metric": metric,
        "comparison": comparison,
        "n_runs": int(len(d)),
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "median_delta": median_delta,
        "ci95_mean_low": mean_low,
        "ci95_mean_high": mean_high,
        "ci95_median_low": med_low,
        "ci95_median_high": med_high,
        "positive_runs": int((d > 0).sum()),
        "negative_runs": int((d < 0).sum()),
        "zero_runs": int((d == 0).sum()),
        "wilcoxon_stat_two_sided": w2_stat,
        "p_two_sided": p_two,
        "wilcoxon_stat_greater": wg_stat,
        "p_greater": p_greater,
        "rank_biserial": rank_biserial_paired(d),
    }


def feature_set_statistics(
    fs: pd.DataFrame,
    n_boot: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"run", "seed", "feature_set", "roc_auc", "pr_auc"}
    missing = required - set(fs.columns)
    if missing:
        raise ValueError(f"feature_sets_all_runs sem colunas: {sorted(missing)}")

    rows = []
    delta_rows = []

    # ROC-AUC é a análise primária; PR-AUC entra como análise secundária.
    for metric in ["roc_auc", "pr_auc"]:
        pivot = fs.pivot(index=["run", "seed"], columns="feature_set", values=metric)

        if BASELINE_SET not in pivot.columns:
            raise ValueError("Feature set 'full' não encontrado.")

        for i, comparator in enumerate(FEATURE_SET_COMPARISONS):
            if comparator not in pivot.columns:
                print(f"[WARN] feature_set ausente: {comparator}")
                continue

            pair = pivot[[BASELINE_SET, comparator]].dropna()
            deltas = pair[BASELINE_SET] - pair[comparator]

            comparison_name = f"full_vs_{comparator}"
            rows.append(
                describe_delta(
                    deltas.to_numpy(),
                    comparison=comparison_name,
                    family="feature_set",
                    metric=metric,
                    n_boot=n_boot,
                    bootstrap_seed=bootstrap_seed + i + (0 if metric == "roc_auc" else 100),
                )
            )

            for (run, seed), delta in deltas.items():
                delta_rows.append({
                    "family": "feature_set",
                    "metric": metric,
                    "comparison": comparison_name,
                    "run": run,
                    "seed": seed,
                    "delta_full_minus_comparator": float(delta),
                })

    return pd.DataFrame(rows), pd.DataFrame(delta_rows)


def ablation_statistics(
    abl: pd.DataFrame,
    n_boot: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "run",
        "seed",
        "group_removed",
        "auc_full",
        "auc_without_group",
        "auc_loss_when_removed",
    }
    missing = required - set(abl.columns)
    if missing:
        raise ValueError(f"group_ablation_all_runs sem colunas: {sorted(missing)}")

    rows = []
    delta_rows = []

    for i, group in enumerate(ABLATION_GROUPS):
        sub = abl[abl["group_removed"] == group].copy()
        if sub.empty:
            print(f"[WARN] grupo de ablação ausente: {group}")
            continue

        # Já está definido como full - sem grupo.
        deltas = sub["auc_loss_when_removed"].to_numpy(dtype=float)

        comparison_name = f"full_vs_without_{group}"
        rows.append(
            describe_delta(
                deltas,
                comparison=comparison_name,
                family="ablation",
                metric="roc_auc",
                n_boot=n_boot,
                bootstrap_seed=bootstrap_seed + 500 + i,
            )
        )

        for _, r in sub.iterrows():
            delta_rows.append({
                "family": "ablation",
                "metric": "roc_auc",
                "comparison": comparison_name,
                "run": int(r["run"]),
                "seed": int(r["seed"]),
                "delta_full_minus_comparator": float(r["auc_loss_when_removed"]),
            })

    return pd.DataFrame(rows), pd.DataFrame(delta_rows)


def add_fdr_columns(stats: pd.DataFrame) -> pd.DataFrame:
    out = stats.copy()

    # Correção global em todas as comparações relatadas.
    out["q_bh_all_two_sided"] = benjamini_hochberg(out["p_two_sided"])
    out["q_bh_all_greater"] = benjamini_hochberg(out["p_greater"])

    # Correção também dentro de cada família/métrica.
    out["q_bh_family_two_sided"] = np.nan
    out["q_bh_family_greater"] = np.nan

    for _, idx in out.groupby(["family", "metric"]).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_family_two_sided"] = benjamini_hochberg(
            out.loc[idx, "p_two_sided"]
        )
        out.loc[idx, "q_bh_family_greater"] = benjamini_hochberg(
            out.loc[idx, "p_greater"]
        )

    return out


def plot_roc_auc_deltas(stats: pd.DataFrame, output: Path) -> None:
    plot_df = stats[stats["metric"] == "roc_auc"].copy()
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values("mean_delta", ascending=True)
    y = np.arange(len(plot_df))

    means = plot_df["mean_delta"].to_numpy()
    lower = means - plot_df["ci95_mean_low"].to_numpy()
    upper = plot_df["ci95_mean_high"].to_numpy() - means

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.errorbar(
        means,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=4,
    )
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["comparison"])
    ax.set_xlabel("Δ ROC-AUC médio (FULL − comparador), bootstrap IC95%")
    ax.set_title("Comparações pareadas nos mesmos 30 splits por case")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def format_result_line(row: pd.Series) -> str:
    sig = "SIM" if row["q_bh_family_two_sided"] < 0.05 else "NÃO"
    return (
        f"{row['comparison']}: "
        f"Δ={row['mean_delta']:+.4f} "
        f"[IC95% {row['ci95_mean_low']:+.4f}, {row['ci95_mean_high']:+.4f}], "
        f"Wilcoxon p={row['p_two_sided']:.6g}, "
        f"q_FDR={row['q_bh_family_two_sided']:.6g}, "
        f"wins={int(row['positive_runs'])}/{int(row['n_runs'])}, "
        f"r_rb={row['rank_biserial']:+.3f}, "
        f"FDR<0.05={sig}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outdir",
        default="results_rf_post_generation",
        help="Diretório produzido por rf_post_generation_30runs.py",
    )
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=20000,
        help="Número de reamostragens bootstrap (default: 20000)",
    )
    ap.add_argument(
        "--bootstrap-seed",
        type=int,
        default=2026,
    )
    ap.add_argument(
        "--expected-runs",
        type=int,
        default=EXPECTED_RUNS,
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    fs_path = outdir / "rf_post_generation_feature_sets_all_runs.csv"
    abl_path = outdir / "rf_post_generation_group_ablation_all_runs.csv"

    if not fs_path.exists():
        raise FileNotFoundError(fs_path)
    if not abl_path.exists():
        raise FileNotFoundError(abl_path)

    fs = pd.read_csv(fs_path)
    abl = pd.read_csv(abl_path)

    n_runs_fs = fs["run"].nunique()
    n_runs_abl = abl["run"].nunique()

    print("=" * 88)
    print("ESTATÍSTICA PAREADA — RF PÓS-GERAÇÃO / PRÉ-SAST")
    print("=" * 88)
    print(f"Feature sets: {n_runs_fs} runs")
    print(f"Ablações:     {n_runs_abl} runs")

    if n_runs_fs != args.expected_runs or n_runs_abl != args.expected_runs:
        print(
            f"[WARN] Esperados {args.expected_runs} runs, "
            f"mas encontrei feature_sets={n_runs_fs}, ablation={n_runs_abl}."
        )

    fs_stats, fs_deltas = feature_set_statistics(
        fs, n_boot=args.bootstrap, bootstrap_seed=args.bootstrap_seed
    )
    abl_stats, abl_deltas = ablation_statistics(
        abl, n_boot=args.bootstrap, bootstrap_seed=args.bootstrap_seed
    )

    all_stats = pd.concat([fs_stats, abl_stats], ignore_index=True)
    all_stats = add_fdr_columns(all_stats)

    # Propaga q-values de volta para as duas tabelas.
    fs_stats_final = all_stats[all_stats["family"] == "feature_set"].copy()
    abl_stats_final = all_stats[all_stats["family"] == "ablation"].copy()

    all_deltas = pd.concat([fs_deltas, abl_deltas], ignore_index=True)

    fs_stats_final.to_csv(
        outdir / "rf_post_generation_paired_feature_sets.csv", index=False
    )
    abl_stats_final.to_csv(
        outdir / "rf_post_generation_paired_ablation.csv", index=False
    )
    all_stats.to_csv(
        outdir / "rf_post_generation_paired_tests_all.csv", index=False
    )
    all_deltas.to_csv(
        outdir / "rf_post_generation_paired_deltas_all_runs.csv", index=False
    )

    plot_roc_auc_deltas(
        all_stats,
        outdir / "fig_rf_post_generation_paired_roc_auc.png",
    )

    # Resumo textual pronto para inspeção.
    summary_path = outdir / "rf_post_generation_stats_summary.txt"
    lines = []
    lines.append("ESTATÍSTICA PAREADA — 30 RUNS\n")
    lines.append(
        "Convenção: delta > 0 significa que FULL teve ROC-AUC/PR-AUC maior.\n"
    )
    lines.append(
        "Teste principal reportado abaixo: Wilcoxon duas caudas + "
        "Benjamini-Hochberg FDR dentro de cada família/métrica; "
        "IC95% por bootstrap dos deltas pareados.\n"
    )

    for family in ["feature_set", "ablation"]:
        lines.append(f"\n[{family.upper()}]\n")
        sub = all_stats[
            (all_stats["family"] == family)
            & (all_stats["metric"] == "roc_auc")
        ]
        for _, row in sub.iterrows():
            lines.append(format_result_line(row) + "\n")

    summary_path.write_text("".join(lines), encoding="utf-8")

    # Console
    print("\n" + "-" * 88)
    print("ROC-AUC — FEATURE SETS")
    print("-" * 88)
    for _, row in fs_stats_final[
        fs_stats_final["metric"] == "roc_auc"
    ].iterrows():
        print(format_result_line(row))

    print("\n" + "-" * 88)
    print("ROC-AUC — ABLAÇÕES")
    print("-" * 88)
    for _, row in abl_stats_final.iterrows():
        print(format_result_line(row))

    print("\n" + "-" * 88)
    print("PR-AUC — FEATURE SETS (secundário)")
    print("-" * 88)
    for _, row in fs_stats_final[
        fs_stats_final["metric"] == "pr_auc"
    ].iterrows():
        print(format_result_line(row))

    print("\nArquivos gerados:")
    for name in [
        "rf_post_generation_paired_feature_sets.csv",
        "rf_post_generation_paired_ablation.csv",
        "rf_post_generation_paired_tests_all.csv",
        "rf_post_generation_paired_deltas_all_runs.csv",
        "fig_rf_post_generation_paired_roc_auc.png",
        "rf_post_generation_stats_summary.txt",
    ]:
        print(" -", outdir / name)


if __name__ == "__main__":
    main()
