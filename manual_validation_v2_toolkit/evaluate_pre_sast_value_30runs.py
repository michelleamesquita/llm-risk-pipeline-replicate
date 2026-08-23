#!/usr/bin/env python3
"""
Avaliação de utilidade prática do classificador LR antes do SAST.

Pergunta:
  O score pré-SAST consegue PRIORIZAR patches com findings do Bandit?

Desenho:
  - mesmas 33 features do modelo LR principal;
  - 30 GroupShuffleSplit agrupados por case;
  - target has_finding_after;
  - políticas de triagem avaliadas apenas no TESTE externo;
  - thresholds para recall-alvo são escolhidos SOMENTE no treino externo
    por previsões OOF internas (StratifiedGroupKFold quando disponível).

Políticas:
  1) always_sast:
       baseline: 100% dos patches passam imediatamente pelo SAST.
  2) fixed_threshold_0.50:
       referência simples, sem tuning.
  3) learned_recall_90:
       threshold aprendido no treino para ~90% recall.
  4) learned_recall_95:
       threshold aprendido no treino para ~95% recall.
  5) top_20_percent:
       cenário de priorização: os 20% patches com maior score são analisados primeiro.
  6) top_50_percent:
       cenário de priorização: os 50% patches com maior score são analisados primeiro.

Métricas:
  - coverage: fração de patches priorizados/selecionados;
  - workload_reduction = 1 - coverage;
  - finding_recall: fração dos patches Bandit-positive capturada;
  - miss_rate = 1 - finding_recall;
  - precision_selected;
  - lift: prevalência de findings entre selecionados / prevalência global;
  - TP/FP/FN/TN.

Interpretação:
  * Para PRIORITIZATION, todos podem passar pelo SAST depois; workload_reduction
    representa redução da fila IMEDIATA, não economia total.
  * Para SELECTIVE ANALYSIS, workload_reduction representa SAST evitado, mas
    miss_rate quantifica o custo em findings perdidos.
  * O experimento NÃO demonstra ganho de tempo sem medir runtime real.

Runtime opcional:
  --runtime-input runtime_measurements.csv

Colunas aceitas:
  feature_extraction_ms
  lr_inference_ms
  bandit_ms

O script sumariza mediana, IQR, p95 e razões quando essas colunas existirem.

Uso:
  python evaluate_pre_sast_value_30runs.py \
      --input case_model_problem_statement_features.csv \
      --outdir results_pre_sast_value \
      --n-runs 30 \
      --bootstrap 20000

Opcional:
  python evaluate_pre_sast_value_30runs.py ... \
      --runtime-input runtime_measurements.csv
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except ImportError:
    HAS_STRATIFIED_GROUP_KFOLD = False


TARGET = "has_finding_after"
GROUP_COL = "case"
MODEL_COL = "model"

BASE_SEED = 42
N_RUNS = 30
TEST_SIZE = 0.20

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

PROMPT_BASE = ["prompt_chars", "prompt_lines", "prompt_tokens"]
PROMPT_DERIVED = [
    "prompt_density", "prompt_token_density", "prompt_size_category"
]
PATCH_BASE = [
    "patch_lines", "patch_added", "patch_removed",
    "patch_files_touched", "patch_hunks", "patch_churn", "patch_net"
]
PATCH_DERIVED = [
    "patch_density", "add_remove_ratio", "net_per_line",
    "hunks_per_file", "patch_complexity", "change_intensity"
]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["prompt_density"] = out["prompt_chars"] / (out["prompt_lines"] + 1.0)
    out["prompt_token_density"] = out["prompt_tokens"] / (out["prompt_chars"] + 1.0)
    out["prompt_size_category"] = pd.cut(
        out["prompt_chars"],
        bins=[-np.inf, 500, 1000, 2000, np.inf],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).astype(float)

    out["patch_density"] = out["patch_churn"] / (out["patch_lines"] + 1.0)
    out["add_remove_ratio"] = out["patch_added"] / (out["patch_removed"] + 1.0)
    out["net_per_line"] = out["patch_net"] / (out["patch_lines"] + 1.0)
    out["hunks_per_file"] = out["patch_hunks"] / (out["patch_files_touched"] + 1.0)
    out["patch_complexity"] = out["patch_hunks"] * out["patch_files_touched"]
    out["change_intensity"] = out["patch_churn"] / (out["patch_files_touched"] + 1.0)

    return out.replace([np.inf, -np.inf], np.nan)


def build_X(df: pd.DataFrame):
    t0 = time.perf_counter()
    work = add_derived(df)

    numeric = PROMPT_BASE + PROMPT_DERIVED + PATCH_BASE + PATCH_DERIVED + PS_FEATURES
    dummies = pd.get_dummies(
        work[MODEL_COL].astype(str),
        prefix="model",
        dtype=float,
    )
    X = pd.concat([work[numeric].astype(float), dummies], axis=1)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return work, X, elapsed_ms


def fit_transform_lr(X_train_raw, X_val_raw, y_train, seed):
    med = X_train_raw.median(numeric_only=True).fillna(0.0)
    Xtr = X_train_raw.fillna(med).fillna(0.0)
    Xva = X_val_raw.fillna(med).fillna(0.0)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)

    model = LogisticRegression(random_state=seed, **LR_PARAMS)
    model.fit(Xtr_s, y_train)
    return model, Xva_s


def outer_fit_with_inference_time(X_train_raw, X_test_raw, y_train, seed):
    med = X_train_raw.median(numeric_only=True).fillna(0.0)
    Xtr = X_train_raw.fillna(med).fillna(0.0)
    Xte = X_test_raw.fillna(med).fillna(0.0)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    model = LogisticRegression(random_state=seed, **LR_PARAMS)
    model.fit(Xtr_s, y_train)

    t0 = time.perf_counter()
    Xte_s = scaler.transform(Xte)
    scores = model.predict_proba(Xte_s)[:, 1]
    inference_ms = (time.perf_counter() - t0) * 1000.0
    return scores, inference_ms


def make_inner_splitter(seed: int, n_splits: int = 5):
    if HAS_STRATIFIED_GROUP_KFOLD:
        return StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
    return GroupKFold(n_splits=n_splits)


def inner_oof_scores(X_train: pd.DataFrame, y_train: np.ndarray, groups_train: np.ndarray, seed: int):
    oof = np.full(len(y_train), np.nan, dtype=float)
    splitter = make_inner_splitter(seed, n_splits=5)

    for fold_idx, (itr, iva) in enumerate(
        splitter.split(X_train, y_train, groups=groups_train)
    ):
        model, Xva_s = fit_transform_lr(
            X_train.iloc[itr],
            X_train.iloc[iva],
            y_train[itr],
            seed + 1000 + fold_idx,
        )
        oof[iva] = model.predict_proba(Xva_s)[:, 1]

    if np.isnan(oof).any():
        raise RuntimeError("OOF interno incompleto.")
    return oof


def choose_threshold_for_recall(y_true, scores, target_recall: float) -> float:
    """
    Escolhe o MAIOR threshold cujo recall observado no treino OOF
    é >= target_recall. Isso minimiza coverage no treino sob a restrição.
    """
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)

    candidates = np.unique(s)
    candidates = np.sort(candidates)[::-1]

    feasible = []
    for t in candidates:
        pred = s >= t
        r = recall_score(y, pred, zero_division=0)
        if r >= target_recall:
            feasible.append(t)

    if not feasible:
        return float(np.min(s) - 1e-12)

    return float(max(feasible))


def policy_metrics(y_true, selected: np.ndarray, policy: str, threshold=np.nan) -> dict:
    y = np.asarray(y_true, dtype=int)
    selected = np.asarray(selected, dtype=bool)

    tp = int(((y == 1) & selected).sum())
    fp = int(((y == 0) & selected).sum())
    fn = int(((y == 1) & (~selected)).sum())
    tn = int(((y == 0) & (~selected)).sum())

    n = len(y)
    n_sel = int(selected.sum())
    positives = int((y == 1).sum())

    coverage = n_sel / n if n else np.nan
    workload_reduction = 1.0 - coverage if np.isfinite(coverage) else np.nan
    finding_recall = tp / positives if positives else np.nan
    miss_rate = fn / positives if positives else np.nan
    precision = tp / n_sel if n_sel else np.nan
    baseline_prev = positives / n if n else np.nan
    selected_prev = precision
    lift = (
        selected_prev / baseline_prev
        if np.isfinite(selected_prev) and baseline_prev > 0
        else np.nan
    )

    return {
        "policy": policy,
        "threshold": threshold,
        "n_test": n,
        "n_selected": n_sel,
        "positives": positives,
        "coverage": coverage,
        "workload_reduction": workload_reduction,
        "finding_recall": finding_recall,
        "miss_rate": miss_rate,
        "precision_selected": precision,
        "baseline_prevalence": baseline_prev,
        "selected_prevalence": selected_prev,
        "lift": lift,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def top_fraction_mask(scores: np.ndarray, fraction: float) -> np.ndarray:
    n = len(scores)
    k = max(1, int(math.ceil(n * fraction)))
    order = np.argsort(scores)[::-1]
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def bootstrap_ci(values, n_boot=20000, seed=2026):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        stats[i] = np.mean(rng.choice(x, size=len(x), replace=True))
    return float(np.quantile(stats, .025)), float(np.quantile(stats, .975))


def summarize_policies(all_runs: pd.DataFrame, n_boot: int, seed: int):
    metrics = [
        "coverage", "workload_reduction", "finding_recall",
        "miss_rate", "precision_selected", "lift"
    ]
    rows = []
    for p_idx, (policy, sub) in enumerate(all_runs.groupby("policy")):
        row = {"policy": policy, "n_runs": sub["run"].nunique()}
        for m_idx, metric in enumerate(metrics):
            vals = sub[metric].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=seed + p_idx*20 + m_idx)
            row[f"{metric}_mean"] = float(np.nanmean(vals))
            row[f"{metric}_std"] = float(np.nanstd(vals, ddof=1))
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_runtime(runtime_path: str, outdir: Path):
    runtime = pd.read_csv(runtime_path)
    accepted = [
        c for c in ["feature_extraction_ms", "lr_inference_ms", "bandit_ms"]
        if c in runtime.columns
    ]
    if not accepted:
        raise ValueError(
            "runtime-input precisa ter pelo menos uma de: "
            "feature_extraction_ms, lr_inference_ms, bandit_ms"
        )

    rows = []
    for col in accepted:
        s = pd.to_numeric(runtime[col], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append({
            "component": col,
            "n": len(s),
            "mean_ms": float(s.mean()),
            "median_ms": float(s.median()),
            "q1_ms": float(s.quantile(.25)),
            "q3_ms": float(s.quantile(.75)),
            "p95_ms": float(s.quantile(.95)),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "pre_sast_runtime_summary.csv", index=False)

    # Razões apenas quando as colunas necessárias existem.
    ratio_rows = []
    if "bandit_ms" in runtime.columns and "lr_inference_ms" in runtime.columns:
        b = pd.to_numeric(runtime["bandit_ms"], errors="coerce")
        l = pd.to_numeric(runtime["lr_inference_ms"], errors="coerce")
        mask = b.notna() & l.notna() & l.gt(0)
        if mask.any():
            ratio = b[mask] / l[mask]
            ratio_rows.append({
                "ratio": "bandit_ms / lr_inference_ms",
                "n": int(mask.sum()),
                "median": float(ratio.median()),
                "q1": float(ratio.quantile(.25)),
                "q3": float(ratio.quantile(.75)),
            })

    if (
        "bandit_ms" in runtime.columns
        and "lr_inference_ms" in runtime.columns
        and "feature_extraction_ms" in runtime.columns
    ):
        b = pd.to_numeric(runtime["bandit_ms"], errors="coerce")
        l = pd.to_numeric(runtime["lr_inference_ms"], errors="coerce")
        f = pd.to_numeric(runtime["feature_extraction_ms"], errors="coerce")
        pre = l + f
        mask = b.notna() & pre.notna() & pre.gt(0)
        if mask.any():
            ratio = b[mask] / pre[mask]
            ratio_rows.append({
                "ratio": "bandit_ms / (feature_extraction_ms + lr_inference_ms)",
                "n": int(mask.sum()),
                "median": float(ratio.median()),
                "q1": float(ratio.quantile(.25)),
                "q3": float(ratio.quantile(.75)),
            })

    pd.DataFrame(ratio_rows).to_csv(
        outdir / "pre_sast_runtime_ratios.csv", index=False
    )
    return summary, pd.DataFrame(ratio_rows)


def plot_tradeoff(summary: pd.DataFrame, path: Path):
    d = summary[summary["policy"] != "always_sast"].copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(d["workload_reduction_mean"], d["finding_recall_mean"])
    for _, r in d.iterrows():
        ax.annotate(
            r["policy"],
            (r["workload_reduction_mean"], r["finding_recall_mean"]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_xlabel("Redução da fila imediata / workload")
    ax.set_ylabel("Recall de patches com finding")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Trade-off de triagem pré-SAST — 30 splits")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="results_pre_sast_value")
    ap.add_argument("--n-runs", type=int, default=N_RUNS)
    ap.add_argument("--base-seed", type=int, default=BASE_SEED)
    ap.add_argument("--test-size", type=float, default=TEST_SIZE)
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--bootstrap-seed", type=int, default=2026)
    ap.add_argument("--runtime-input", default="")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    required = {
        GROUP_COL, MODEL_COL, TARGET,
        *PS_FEATURES, *PROMPT_BASE, *PATCH_BASE,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing)}")

    work, X, build_ms = build_X(df)
    y = work[TARGET].astype(int).to_numpy()
    groups = work[GROUP_COL].astype(str).to_numpy()

    rows = []
    prediction_rows = []
    inference_rows = []

    for run_idx in range(args.n_runs):
        run = run_idx + 1
        seed = args.base_seed + run_idx

        outer = GroupShuffleSplit(
            n_splits=1,
            test_size=args.test_size,
            random_state=seed,
        )
        train_idx, test_idx = next(outer.split(X, y, groups=groups))

        train_cases = set(groups[train_idx])
        test_cases = set(groups[test_idx])
        if train_cases & test_cases:
            raise RuntimeError("Case leakage detectado.")

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]
        g_train = groups[train_idx]

        # Thresholds escolhidos apenas via OOF dentro do treino.
        oof_scores = inner_oof_scores(
            X_train, y_train, g_train, seed
        )
        t90 = choose_threshold_for_recall(y_train, oof_scores, 0.90)
        t95 = choose_threshold_for_recall(y_train, oof_scores, 0.95)

        test_scores, inference_ms = outer_fit_with_inference_time(
            X_train, X_test, y_train, seed
        )

        inference_rows.append({
            "run": run,
            "seed": seed,
            "n_test": len(test_idx),
            "lr_transform_plus_inference_ms": inference_ms,
            "lr_ms_per_patch": inference_ms / len(test_idx),
        })

        policies = [
            policy_metrics(
                y_test, np.ones(len(y_test), dtype=bool),
                "always_sast", threshold=np.nan
            ),
            policy_metrics(
                y_test, test_scores >= 0.50,
                "fixed_threshold_0.50", threshold=0.50
            ),
            policy_metrics(
                y_test, test_scores >= t90,
                "learned_recall_90", threshold=t90
            ),
            policy_metrics(
                y_test, test_scores >= t95,
                "learned_recall_95", threshold=t95
            ),
            policy_metrics(
                y_test, top_fraction_mask(test_scores, 0.20),
                "top_20_percent", threshold=np.nan
            ),
            policy_metrics(
                y_test, top_fraction_mask(test_scores, 0.50),
                "top_50_percent", threshold=np.nan
            ),
        ]

        for p in policies:
            rows.append({
                "run": run,
                "seed": seed,
                "train_cases": len(train_cases),
                "test_cases": len(test_cases),
                "threshold_train_oof_recall_90": t90,
                "threshold_train_oof_recall_95": t95,
                **p,
            })

        for pos, idx in enumerate(test_idx):
            prediction_rows.append({
                "run": run,
                "seed": seed,
                "row_index": int(idx),
                "case": work.iloc[idx][GROUP_COL],
                "model": work.iloc[idx][MODEL_COL],
                "y_true": int(y_test[pos]),
                "lr_score": float(test_scores[pos]),
            })

        print(
            f"Run {run:02d}/{args.n_runs} | "
            f"t90={t90:.3f} t95={t95:.3f} | "
            f"top20 recall="
            f"{[p for p in policies if p['policy']=='top_20_percent'][0]['finding_recall']:.3f} | "
            f"top50 recall="
            f"{[p for p in policies if p['policy']=='top_50_percent'][0]['finding_recall']:.3f}"
        )

    all_runs = pd.DataFrame(rows)
    predictions = pd.DataFrame(prediction_rows)
    inference_df = pd.DataFrame(inference_rows)

    summary = summarize_policies(
        all_runs,
        n_boot=args.bootstrap,
        seed=args.bootstrap_seed,
    )

    all_runs.to_csv(outdir / "pre_sast_triage_all_runs.csv", index=False)
    summary.to_csv(outdir / "pre_sast_triage_summary.csv", index=False)
    predictions.to_csv(outdir / "pre_sast_triage_predictions.csv", index=False)
    inference_df.to_csv(outdir / "pre_sast_lr_inference_timing.csv", index=False)

    plot_tradeoff(
        summary,
        outdir / "fig_pre_sast_triage_tradeoff.png",
    )

    runtime_summary = None
    runtime_ratios = None
    if args.runtime_input:
        runtime_summary, runtime_ratios = summarize_runtime(
            args.runtime_input, outdir
        )
    else:
        # Template para instrumentação externa do pipeline real.
        template = (
            work[[GROUP_COL, MODEL_COL]]
            .drop_duplicates()
            .copy()
        )
        template["feature_extraction_ms"] = ""
        template["lr_inference_ms"] = ""
        template["bandit_ms"] = ""
        template.to_csv(
            outdir / "runtime_measurements_template.csv",
            index=False,
        )

    protocol = {
        "task": "post-generation / pre-SAST triage",
        "rows": int(len(work)),
        "cases": int(work[GROUP_COL].nunique()),
        "features": int(X.shape[1]),
        "n_runs": int(args.n_runs),
        "outer_split": "GroupShuffleSplit grouped by case",
        "inner_threshold_selection": (
            "5-fold StratifiedGroupKFold on outer-train"
            if HAS_STRATIFIED_GROUP_KFOLD
            else "5-fold GroupKFold on outer-train"
        ),
        "target": TARGET,
        "thresholds_learned_without_outer_test": True,
        "feature_matrix_build_ms_total": build_ms,
        "feature_matrix_build_ms_per_row_in_memory": build_ms / len(work),
        "runtime_input_used": bool(args.runtime_input),
        "interpretation": {
            "prioritization": (
                "top-k policies measure how many Bandit-positive patches appear "
                "in the immediate review/SAST queue. This does not claim total SAST cost reduction."
            ),
            "selective_analysis": (
                "threshold policies can simulate SAST avoidance, but misses must "
                "be reported explicitly."
            ),
        },
    }
    (outdir / "pre_sast_value_protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("UTILIDADE PRÉ-SAST — RESUMO")
    print("=" * 100)
    cols = [
        "policy",
        "coverage_mean",
        "workload_reduction_mean",
        "finding_recall_mean",
        "miss_rate_mean",
        "precision_selected_mean",
        "lift_mean",
    ]
    print(
        summary[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nTempo LR medido no próprio experimento:")
    print(
        inference_df[
            ["lr_transform_plus_inference_ms", "lr_ms_per_patch"]
        ].describe().round(4).to_string()
    )

    if runtime_summary is not None:
        print("\nRuntime externo:")
        print(runtime_summary.to_string(index=False))
        if runtime_ratios is not None and not runtime_ratios.empty:
            print("\nRazões:")
            print(runtime_ratios.to_string(index=False))

    print("\nArquivos salvos em:", outdir.resolve())


if __name__ == "__main__":
    main()
