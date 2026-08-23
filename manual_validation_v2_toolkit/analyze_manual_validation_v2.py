#!/usr/bin/env python3
"""
Analisa a validação manual v2 e, opcionalmente, compara com LangSmith judge.

Entradas:
  --reviewer1 manual_review_R1.csv
  --key manual_validation_key_v2.csv

Opcional:
  --reviewer2 manual_review_R2.csv
  --adjudication adjudication_completed.csv
  --judge langsmith_bandit_judge_v2_results.csv

Fluxo:
  1) se houver R2, calcula agreement + Cohen's kappa R1 x R2;
  2) gera disagreements_for_adjudication.csv (vazio se só houver R1);
  3) se adjudicação for fornecida, usa rótulo adjudicado como final;
  4) sem adjudicação: consenso R1=R2, ou só R1 se não houver segundo revisor;
  5) resume validade e atribuição, incluindo ALL_NEW_FINDINGS;
  6) se judge for fornecido, compara judge x rótulo humano final.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


VALID = "VALID_SECURITY_FINDING"
FALSE = "LIKELY_FALSE_POSITIVE"
UNC = "UNCERTAIN"
INTRO = "INTRODUCED_OR_AFFECTED_BY_PATCH"
PRE = "PRE_EXISTING"


def norm(x):
    return x.fillna("").astype(str).str.strip().str.upper()


def wilson(k, n, z=1.959963984540054):
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return max(0, center-half), min(1, center+half)


def load_reviewer(path, suffix):
    d = pd.read_csv(path)
    required = {
        "review_id_v2", "validity", "patch_attribution",
        "reviewer_confidence_1_to_5", "reviewer_notes",
    }
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"{path}: colunas ausentes {sorted(missing)}")
    keep = list(required)
    d = d[keep].copy()
    return d.rename(columns={
        "validity": f"validity_{suffix}",
        "patch_attribution": f"attribution_{suffix}",
        "reviewer_confidence_1_to_5": f"confidence_{suffix}",
        "reviewer_notes": f"notes_{suffix}",
    })


def kappa_summary(df, c1, c2, dimension):
    a = norm(df[c1])
    b = norm(df[c2])
    mask = a.ne("") & b.ne("")
    if mask.sum() < 2:
        return {
            "dimension": dimension,
            "n_double_rated": int(mask.sum()),
            "raw_agreement": np.nan,
            "cohen_kappa": np.nan,
        }
    return {
        "dimension": dimension,
        "n_double_rated": int(mask.sum()),
        "raw_agreement": float((a[mask] == b[mask]).mean()),
        "cohen_kappa": float(cohen_kappa_score(a[mask], b[mask])),
    }


def validity_summary(sub, label="all"):
    v = norm(sub["final_validity"])
    n_valid = int(v.eq(VALID).sum())
    n_false = int(v.eq(FALSE).sum())
    n_unc = int(v.eq(UNC).sum())
    decidable = n_valid + n_false
    lo, hi = wilson(n_valid, decidable)
    return {
        "scope": label,
        "n_rows": len(sub),
        "n_final_labeled": int(v.ne("").sum()),
        "valid": n_valid,
        "likely_false_positive": n_false,
        "uncertain": n_unc,
        "valid_rate_among_decidable": (
            n_valid / decidable if decidable else np.nan
        ),
        "valid_rate_ci95_low": lo,
        "valid_rate_ci95_high": hi,
    }


def attribution_summary(sub, label="all"):
    a = norm(sub["final_patch_attribution"])
    n_intro = int(a.eq(INTRO).sum())
    n_pre = int(a.eq(PRE).sum())
    n_unc = int(a.eq(UNC).sum())
    decidable = n_intro + n_pre
    lo, hi = wilson(n_intro, decidable)
    return {
        "scope": label,
        "introduced_or_affected": n_intro,
        "pre_existing": n_pre,
        "attribution_uncertain": n_unc,
        "introduced_rate_among_decidable": (
            n_intro / decidable if decidable else np.nan
        ),
        "introduced_rate_ci95_low": lo,
        "introduced_rate_ci95_high": hi,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviewer1", required=True)
    ap.add_argument(
        "--reviewer2",
        default="",
        help="Opcional. Sem R2, o rótulo final é o do revisor 1.",
    )
    ap.add_argument("--key", required=True)
    ap.add_argument("--adjudication", default="")
    ap.add_argument("--judge", default="")
    ap.add_argument("--outdir", default="manual_validation_v2_analysis")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    r1 = load_reviewer(args.reviewer1, "r1")
    key = pd.read_csv(args.key)

    df = key.merge(r1, on="review_id_v2", how="left", validate="one_to_one")
    single_reviewer = not args.reviewer2
    if single_reviewer:
        df["validity_r2"] = ""
        df["attribution_r2"] = ""
        df["confidence_r2"] = ""
        df["notes_r2"] = ""
        print("Aviso: sem --reviewer2; rótulo final = revisor 1 (sem kappa).")
    else:
        r2 = load_reviewer(args.reviewer2, "r2")
        df = df.merge(r2, on="review_id_v2", how="left", validate="one_to_one")

    interrater = pd.DataFrame([
        kappa_summary(df, "validity_r1", "validity_r2", "validity"),
        kappa_summary(
            df, "attribution_r1", "attribution_r2", "patch_attribution"
        ),
    ])
    interrater.to_csv(outdir / "interrater_v2.csv", index=False)

    v1, v2 = norm(df["validity_r1"]), norm(df["validity_r2"])
    a1, a2 = norm(df["attribution_r1"]), norm(df["attribution_r2"])

    if single_reviewer:
        disagree = pd.Series(False, index=df.index)
    else:
        disagree = (
            (v1.ne("") & v2.ne("") & v1.ne(v2))
            | (a1.ne("") & a2.ne("") & a1.ne(a2))
        )

    adjud_cols = [
        "review_id_v2",
        "model",
        "case",
        "relative_filename",
        "test_id",
        "cwe",
        "sample_group_v2",
        "validity_r1",
        "validity_r2",
        "attribution_r1",
        "attribution_r2",
    ]
    adjud = df.loc[disagree, [c for c in adjud_cols if c in df.columns]].copy()
    adjud["adjudicated_validity"] = ""
    adjud["adjudicated_patch_attribution"] = ""
    adjud["adjudication_notes"] = ""
    adjud.to_csv(
        outdir / "disagreements_for_adjudication.csv",
        index=False,
    )

    # Consenso inicial: R1 sozinho, ou R1=R2.
    if single_reviewer:
        df["final_validity"] = v1
        df["final_patch_attribution"] = a1
    else:
        df["final_validity"] = np.where(
            v1.eq(v2) & v1.ne(""), v1, ""
        )
        df["final_patch_attribution"] = np.where(
            a1.eq(a2) & a1.ne(""), a1, ""
        )

    # Adjudicação opcional.
    if args.adjudication:
        adj = pd.read_csv(args.adjudication)
        cols = [
            "review_id_v2",
            "adjudicated_validity",
            "adjudicated_patch_attribution",
            "adjudication_notes",
        ]
        missing = [c for c in cols if c not in adj.columns]
        if missing:
            raise ValueError(f"Adjudicação sem colunas: {missing}")
        df = df.merge(adj[cols], on="review_id_v2", how="left")
        av = norm(df["adjudicated_validity"])
        aa = norm(df["adjudicated_patch_attribution"])
        df.loc[av.ne(""), "final_validity"] = av[av.ne("")]
        df.loc[aa.ne(""), "final_patch_attribution"] = aa[aa.ne("")]

    scopes = [("ALL", df)]
    if "sample_group_v2" in df.columns:
        for name, sub in df.groupby("sample_group_v2"):
            scopes.append((str(name), sub))

    summary_rows = []
    for name, sub in scopes:
        row = validity_summary(sub, name)
        row.update(attribution_summary(sub, name))
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "manual_validation_v2_summary.csv", index=False)

    # Especial: novos findings.
    if "sample_group_v2" in df.columns:
        new = df[df["sample_group_v2"].eq("ALL_NEW_FINDINGS")].copy()
        new.to_csv(outdir / "all_new_findings_human_review.csv", index=False)

    # Judge opcional.
    judge_summary = pd.DataFrame()
    if args.judge:
        judge = pd.read_csv(args.judge)
        if "review_id_v2" not in judge.columns:
            raise ValueError("CSV do judge sem coluna review_id_v2")
        judge["review_id_v2"] = judge["review_id_v2"].astype(str).str.strip()
        judge = judge[judge["review_id_v2"].ne("") & judge["review_id_v2"].ne("nan")]
        if judge.empty:
            raise ValueError(
                "CSV do judge sem review_id_v2 preenchido. "
                "Regenere langsmith_bandit_judge_v2_results.csv."
            )
        joined = df.merge(
            judge,
            on="review_id_v2",
            how="inner",
            validate="one_to_one",
        )
        rows = []
        for dim, hcol, jcol in [
            ("validity", "final_validity", "judge_validity"),
            (
                "patch_attribution",
                "final_patch_attribution",
                "judge_patch_attribution",
            ),
        ]:
            h = norm(joined[hcol])
            j = norm(joined[jcol])
            mask = h.ne("") & j.ne("")
            if mask.sum() >= 2:
                rows.append({
                    "dimension": dim,
                    "scope": "all_labeled",
                    "n": int(mask.sum()),
                    "agreement": float((h[mask] == j[mask]).mean()),
                    "cohen_kappa": float(
                        cohen_kappa_score(h[mask], j[mask])
                    ),
                })
                dec = mask & h.ne(UNC) & j.ne(UNC)
                if dec.sum() >= 2:
                    rows.append({
                        "dimension": dim,
                        "scope": "decidable_only",
                        "n": int(dec.sum()),
                        "agreement": float((h[dec] == j[dec]).mean()),
                        "cohen_kappa": float(
                            cohen_kappa_score(h[dec], j[dec])
                        ),
                    })
        judge_summary = pd.DataFrame(rows)
        judge_summary.to_csv(
            outdir / "human_vs_langsmith_v2.csv",
            index=False,
        )
        joined.to_csv(
            outdir / "human_and_langsmith_joined_v2.csv",
            index=False,
        )

    df.to_csv(outdir / "manual_validation_v2_full_results.csv", index=False)

    print("=" * 90)
    print("INTER-RATER")
    print("=" * 90)
    print(interrater.to_string(index=False))
    print("\n" + "=" * 90)
    print("RESULTADOS HUMANOS")
    print("=" * 90)
    print(summary.to_string(index=False))
    print(f"\nDivergências para adjudicação: {int(disagree.sum())}")
    if not judge_summary.empty:
        print("\n" + "=" * 90)
        print("HUMANO x LANGSMITH")
        print("=" * 90)
        print(judge_summary.to_string(index=False))


if __name__ == "__main__":
    main()
