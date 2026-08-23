#!/usr/bin/env python3
import os
import json
import math
import glob
import argparse
from collections import defaultdict, Counter

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


BACKUPS = [
    ("claud-sonnet_backup", "claude"),
    ("codellama_tuned_backup", "codellama-tuned"),
    ("deepseek_backup", "deepseek"),
    ("gpt-4o_backup", "gpt-4o"),
]

OWASP_TOP10_CWES = {
    # This is a compact subset aligned with configs/bandit_cwe_mapping.yaml comments
    "CWE-20", "CWE-78", "CWE-79", "CWE-89", "CWE-200", "CWE-287",
    "CWE-295", "CWE-319", "CWE-326", "CWE-327", "CWE-502", "CWE-614"
}

HUB_CWES = {"CWE-703", "CWE-327"}

CONF_MAP = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
SEV_LEVELS = ["LOW", "MEDIUM", "HIGH"]


def _safe_read_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def load_bandit_findings(reports_dir: str) -> pd.DataFrame:
    rows = []
    for fp in glob.glob(os.path.join(reports_dir, "*_bandit_after.json")):
        data = _safe_read_json(fp)
        if not data:
            continue
        results = data.get("results") or data.get("issues") or []
        for it in results:
            cwe_id = f"CWE-{it.get('issue_cwe', {}).get('id', 'Unknown')}"
            rows.append({
                "filename": it.get("filename", ""),
                "line": it.get("line_number"),
                "test_id": it.get("test_id"),
                "severity": (it.get("issue_severity") or "").upper(),
                "confidence": (it.get("issue_confidence") or "").upper(),
                "cwe": cwe_id,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # De-duplicate by filename|line|test to avoid double counting
    df["key"] = df.apply(lambda r: f"{r['filename']}|{r['line']}|{r['test_id']}", axis=1)
    df = df.drop_duplicates("key")
    return df


def shannon_entropy(labels: list) -> float:
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = sum(counts.values())
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            ent -= p * math.log(p, 2)
    return ent


def compute_cooccurrence_avg(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0
    # Build per-file set of CWEs then average pairwise cooccurrence counts across files
    by_file = defaultdict(set)
    for _, r in df.iterrows():
        by_file[r["filename"]].add(r["cwe"])
    # count pairs
    pair_counts = Counter()
    for cwes in by_file.values():
        cwe_list = sorted(list(cwes))
        for i in range(len(cwe_list)):
            for j in range(i + 1, len(cwe_list)):
                pair_counts[(cwe_list[i], cwe_list[j])] += 1
    if not pair_counts:
        return 0.0
    return float(np.mean(list(pair_counts.values())))


def compute_cluster_density(df: pd.DataFrame) -> float:
    # Lightweight approximation: density = 2E / (N*(N-1)) for CWE cooccurrence graph
    if df is None or df.empty:
        return 0.0
    by_file = defaultdict(set)
    for _, r in df.iterrows():
        by_file[r["filename"]].add(r["cwe"])
    # Nodes
    all_cwes = set()
    for s in by_file.values():
        all_cwes.update(s)
    nodes = sorted(list(all_cwes))
    idx = {c: i for i, c in enumerate(nodes)}
    n = len(nodes)
    if n <= 1:
        return 0.0
    # Undirected simple graph edges via cooccurrence
    adj = [[0]*n for _ in range(n)]
    for s in by_file.values():
        l = list(s)
        for i in range(len(l)):
            for j in range(i+1, len(l)):
                a = idx[l[i]]; b = idx[l[j]]
                if a != b:
                    adj[min(a,b)][max(a,b)] = 1
    E = 0
    for i in range(n):
        for j in range(i+1, n):
            E += adj[i][j]
    density = (2.0 * E) / (n * (n - 1))
    return float(density)


def compute_features_for_model(model_name: str, reports_dir: str) -> dict:
    df = load_bandit_findings(reports_dir)
    total = 0 if df is None or df.empty else len(df)

    # counts per CWE and severity
    cwe_counts = Counter([] if df is None or df.empty else list(df["cwe"]))
    unique_cwe = len(cwe_counts)

    sev_counts = {s: 0 for s in SEV_LEVELS}
    if df is not None and not df.empty:
        for s, cnt in Counter(df["severity"]).items():
            if s in sev_counts:
                sev_counts[s] = cnt

    p_high = (sev_counts["HIGH"] / total) if total else 0.0
    p_medium = (sev_counts["MEDIUM"] / total) if total else 0.0

    ent = shannon_entropy(list(cwe_counts.elements()))

    hub_presence = 1 if any(c in HUB_CWES for c in cwe_counts.keys()) else 0

    if df is not None and not df.empty:
        avg_conf = float(np.mean([CONF_MAP.get(c, 1) for c in df["confidence"]]))
    else:
        avg_conf = 0.0

    top10_ratio = 0.0
    if total:
        top10 = sum(cwe_counts[c] for c in cwe_counts.keys() if c in OWASP_TOP10_CWES)
        top10_ratio = top10 / total

    cooc_avg = compute_cooccurrence_avg(df)
    cluster_density = compute_cluster_density(df)

    return {
        "model": model_name,
        "n_findings": total,
        "n_cwe_unique": unique_cwe,
        "p_high": p_high,
        "p_medium": p_medium,
        "entropy_cwe": ent,
        "hub_presence": hub_presence,
        "avg_confidence": avg_conf,
        "top10_owasp_ratio": top10_ratio,
        "cooc_avg": cooc_avg,
        "cluster_density": cluster_density,
    }


def build_dataset(runs_backup_root: str) -> pd.DataFrame:
    rows = []
    for folder, name in BACKUPS:
        reports_dir = os.path.join(runs_backup_root, folder, "reports")
        if not os.path.isdir(reports_dir):
            continue
        feats = compute_features_for_model(name, reports_dir)
        rows.append(feats)
    df = pd.DataFrame(rows)
    return df


def derive_labels(df: pd.DataFrame, method: str = "median_split") -> pd.Series:
    # Risk label proxy: binary classification. Several options; default median on n_findings weighted by severity mix
    if df is None or df.empty:
        return pd.Series(dtype=int)
    # Score: weighted severity count proxy using p_high and totals
    score = df["n_findings"] * (1 + df["p_high"] + 0.5 * df["p_medium"]) * (1 + df["top10_owasp_ratio"]) \
            * (1 + 0.3 * df["hub_presence"]) * (1 + 0.1 * df["entropy_cwe"]) \
            * (1 + 0.05 * df["cluster_density"]) * (1 + 0.05 * df["cooc_avg"]) \
            * (1 + 0.05 * (df["avg_confidence"] / 3.0))
    if method == "median_split":
        thr = float(np.median(score))
        label = (score >= thr).astype(int)
    else:
        # fallback: top-50% as high risk
        thr = float(np.median(score))
        label = (score >= thr).astype(int)
    return label


def fit_logreg(df: pd.DataFrame, y: pd.Series):
    feature_cols = [
        "n_findings",
        "n_cwe_unique",
        "p_high",
        "p_medium",
        "entropy_cwe",
        "hub_presence",
        "avg_confidence",
        "top10_owasp_ratio",
        "cooc_avg",
        "cluster_density",
    ]
    X = df[feature_cols].copy()
    # Pipeline with standardization is helpful; use liblinear for small N, balanced to handle tiny dataset
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced"))
    ])
    pipe.fit(X, y)
    # Extract coefficients
    clf = pipe.named_steps["clf"]
    scaler = pipe.named_steps["scaler"]
    # Compute coefficients on original feature scale: beta_scaled / std
    coefs = clf.coef_[0] / (scaler.scale_ + 1e-12)
    intercept = clf.intercept_[0] - np.sum((scaler.mean_ / (scaler.scale_ + 1e-12)) * clf.coef_[0])
    coef_df = pd.DataFrame({"feature": feature_cols, "coef": coefs}).sort_values("coef", ascending=False)
    return pipe, coef_df, float(intercept)


def evaluate_model(pipe: Pipeline, df: pd.DataFrame, y: pd.Series) -> dict:
    feature_cols = [
        "n_findings",
        "n_cwe_unique",
        "p_high",
        "p_medium",
        "entropy_cwe",
        "hub_presence",
        "avg_confidence",
        "top10_owasp_ratio",
        "cooc_avg",
        "cluster_density",
    ]
    X = df[feature_cols].copy()
    prob = pipe.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y, prob))
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y, pred)
    rep = classification_report(y, pred, output_dict=True, zero_division=0)
    return {"auc": auc, "confusion_matrix": cm.tolist(), "report": rep}


def main():
    ap = argparse.ArgumentParser(description="Risk classifier baseline (Logistic Regression) from Bandit reports")
    ap.add_argument("--runs_backup", default="/Documents/swe-sec/runs_backup", help="Root runs_backup path")
    ap.add_argument("--out_dir", default="/Documents/swe-sec/runs_backup/compare/all4", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Build dataset
    df = build_dataset(args.runs_backup)
    if df is None or df.empty:
        print("[error] No data found to build dataset")
        return

    # Labels
    y = derive_labels(df, method="median_split")
    df_out = df.copy()
    df_out["risk_label"] = y

    # Fit
    pipe, coef_df, intercept = fit_logreg(df_out, y)
    metrics = evaluate_model(pipe, df_out, y)

    # Save artifacts
    dataset_csv = os.path.join(args.out_dir, "risk_dataset.csv")
    coefs_csv = os.path.join(args.out_dir, "logreg_coefs.csv")
    metrics_json = os.path.join(args.out_dir, "logreg_metrics.json")

    df_out.to_csv(dataset_csv, index=False)
    coef_df.to_csv(coefs_csv, index=False)
    with open(metrics_json, 'w') as f:
        json.dump({"intercept": intercept, **metrics}, f, indent=2)

    # Console summary
    print("-> Dataset:", dataset_csv)
    print("-> Coefficients:", coefs_csv)
    print("-> Metrics:", metrics_json)


if __name__ == "__main__":
    main()
