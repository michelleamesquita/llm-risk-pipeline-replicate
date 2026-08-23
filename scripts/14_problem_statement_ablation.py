#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
14_problem_statement_ablation.py

Objetivo
--------
Recalcular as features do prompt EXCLUSIVAMENTE a partir do problem_statement
original do SWE-bench, evitando contaminação pelo prompt template.

Depois executa uma ablação limpa entre:

A) Density / Complexity
B) Technical Evidence
C) Security-Relevant Task Semantics
D) combinações A+B, A+C, B+C e A+B+C

Modelos:
- Logistic Regression
- Random Forest

Avaliação:
- 30 repetições
- 80/20 GroupShuffleSplit por case
- Wilcoxon pareado
- bootstrap 95% CI
- rank-biserial correlation

Target padrão:
    has_finding_after = findings_after > 0

Entrada:
    case_model_before_after_summary_common.csv
    runs/cases/<case>.json

Uso:
python scripts/14_problem_statement_ablation.py \
  --input case_model_before_after_summary_common.csv \
  --cases-dir runs/cases \
  --runs 30 \
  --seed 42 \
  --bootstrap-resamples 10000 \
  --out-dir problem_statement_ablation_results
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# REGEX BÁSICAS
# ============================================================

FILE_PATH_RE = re.compile(
    r"(?<![\w./-])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\."
    r"(?:py|js|ts|java|go|rb|php|cs|cpp|c|h|json|yaml|yml|toml|ini|cfg|md)"
)
TRACEBACK_RE = re.compile(
    r"\bTraceback\b|File\s+[\"'][^\"']+[\"'],\s+line\s+\d+",
    re.I,
)
TEST_RE = re.compile(
    r"\btest(?:s|ing)?\b|pytest|unittest|FAIL_TO_PASS|PASS_TO_PASS",
    re.I,
)
URL_RE = re.compile(r"https?://[^\s)\]>]+", re.I)
IDENTIFIER_RE = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_.]*\b"
)
NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


# ============================================================
# DOMÍNIOS SECURITY-RELEVANT
# ============================================================

SECURITY_RELEVANT_PATTERNS = {
    "explicit_security": [
        r"\bsecurity\b",
        r"\bsecure\b",
        r"\bvulnerab\w*\b",
        r"\bcwe-\d+\b",
        r"\bowasp\b",
        r"\binjection\b",
        r"\bxss\b",
        r"\bsqli?\b",
        r"\bexploit\w*\b",
        r"\bunsafe\b",
    ],
    "auth": [
        r"\bauthentication\b",
        r"\bauthorization\b",
        r"\bauth\b",
        r"\blogin\b",
        r"\bsession\b",
        r"\btoken\b",
        r"\bjwt\b",
        r"\bcredential\w*\b",
        r"\bpassword\b",
    ],
    "input_validation": [
        r"\bvalidate\b",
        r"\bvalidation\b",
        r"\bsanitize\b",
        r"\bescape\b",
        r"\buser input\b",
        r"\buntrusted\b",
        r"\bparameter\b",
        r"\bpayload\b",
    ],
    "database": [
        r"\bdatabase\b",
        r"\bsql\b",
        r"\bquery\b",
        r"\borm\b",
        r"\bpostgres\w*\b",
        r"\bmysql\b",
        r"\bsqlite\b",
        r"\btransaction\b",
    ],
    "network_api": [
        r"\bapi\b",
        r"\bhttp\b",
        r"\bhttps\b",
        r"\brequest\b",
        r"\bresponse\b",
        r"\bendpoint\b",
        r"\bsocket\b",
        r"\bnetwork\b",
        r"\burl\b",
        r"\bwebhook\b",
    ],
    "filesystem": [
        r"\bfile\b",
        r"\bfilesystem\b",
        r"\bpath\b",
        r"\bdirectory\b",
        r"\bfolder\b",
        r"\btempfile\b",
    ],
    "command_exec": [
        r"\bshell\b",
        r"\bcommand\b",
        r"\bsubprocess\b",
        r"\bos\.system\b",
        r"\bexec\(",
        r"\beval\(",
        r"\bspawn\b",
        r"\bpopen\b",
    ],
    "crypto_secret": [
        r"\bcrypto\w*\b",
        r"\bencrypt\w*\b",
        r"\bdecrypt\w*\b",
        r"\bhash\b",
        r"\bsecret\b",
        r"\bprivate key\b",
        r"\bcertificate\b",
        r"\bssl\b",
        r"\btls\b",
    ],
    "dependency": [
        r"\bdependency\b",
        r"\bdependencies\b",
        r"\bpackage\b",
        r"\bimport\b",
        r"\brequirement\w*\b",
        r"\bversion\b",
    ],
    "serialization": [
        r"\bserializ\w*\b",
        r"\bdeserializ\w*\b",
        r"\bpickle\b",
        r"\byaml\b",
        r"\bjson\b",
        r"\bmarshal\b",
    ],
    "web_output": [
        r"\bhtml\b",
        r"\btemplate\b",
        r"\brender\b",
        r"\bjavascript\b",
        r"\bdom\b",
        r"\boutput\b",
    ],
    "permissions": [
        r"\bpermission\b",
        r"\bprivilege\b",
        r"\baccess control\b",
        r"\brole\b",
        r"\bowner\b",
        r"\badmin\b",
    ],
    "concurrency": [
        r"\bthread\b",
        r"\basync\b",
        r"\bawait\b",
        r"\brace condition\b",
        r"\block\b",
        r"\bconcurrent\b",
    ],
    "error_handling": [
        r"\bexception\b",
        r"\btraceback\b",
        r"\berror handling\b",
        r"\btry\b",
        r"\bexcept\b",
        r"\braise\b",
        r"\bfailure\b",
    ],
}

CONSTRAINT_PATTERNS = [
    r"\bmust\b",
    r"\bshould\b",
    r"\brequired\b",
    r"\brequirement\b",
    r"\bpreserve\b",
    r"\bmaintain\b",
    r"\bcompatible\b",
    r"\bdo not\b",
    r"\bdon't\b",
    r"\bwithout\b",
    r"\bonly\b",
    r"\bavoid\b",
    r"\bnever\b",
]

MODAL_PATTERNS = [
    r"\bmust\b",
    r"\bshould\b",
    r"\bshall\b",
    r"\bneed(?:s|ed)? to\b",
    r"\brequired to\b",
    r"\bexpected to\b",
]


def count_patterns(text: str, patterns: list[str]) -> int:
    return sum(
        len(re.findall(p, text, flags=re.I))
        for p in patterns
    )


def has_patterns(text: str, patterns: list[str]) -> int:
    return int(
        any(re.search(p, text, flags=re.I) for p in patterns)
    )


def safe_div(a, b):
    if b == 0:
        return 0.0
    return float(a) / float(b)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def load_problem_statement(
    cases_dir: Path,
    case: str,
) -> str:
    path = cases_dir / f"{case}.json"

    if not path.is_file():
        return ""

    try:
        obj = json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        return ""

    return str(
        obj.get("problem_statement")
        or obj.get("issue_body")
        or ""
    )


def extract_problem_features(text: str) -> dict:
    raw = str(text or "")
    lower = raw.lower()

    lines = raw.splitlines()
    words = WORD_RE.findall(raw)

    file_paths = FILE_PATH_RE.findall(raw)
    identifiers = IDENTIFIER_RE.findall(raw)
    numeric_literals = NUMERIC_RE.findall(raw)
    urls = URL_RE.findall(raw)

    bullet_count = sum(
        1
        for line in lines
        if re.match(
            r"^\s*(?:[-*+]|\d+[.)])\s+",
            line,
        )
    )

    # --------------------------------------------------------
    # A) Density / Complexity
    # --------------------------------------------------------
    chars = len(raw)
    line_count = len(lines)
    word_count = len(words)

    density_features = {
        "ps_chars": chars,
        "ps_lines": line_count,
        "ps_words": word_count,
        "ps_questions": raw.count("?"),
        "ps_numeric_literal_count": len(numeric_literals),
        "ps_identifier_count": len(identifiers),
        "ps_bullet_count": bullet_count,
        "ps_constraint_count": count_patterns(
            lower,
            CONSTRAINT_PATTERNS,
        ),
        "ps_modal_count": count_patterns(
            lower,
            MODAL_PATTERNS,
        ),
        "ps_chars_per_line": safe_div(
            chars,
            max(line_count, 1),
        ),
        "ps_words_per_line": safe_div(
            word_count,
            max(line_count, 1),
        ),
        "ps_identifier_density": safe_div(
            len(identifiers),
            max(word_count, 1),
        ),
        "ps_numeric_density": safe_div(
            len(numeric_literals),
            max(word_count, 1),
        ),
    }

    # --------------------------------------------------------
    # B) Technical Evidence
    # --------------------------------------------------------
    technical_features = {
        "ps_file_path_count": len(file_paths),
        "ps_unique_file_path_count": len(set(file_paths)),
        "ps_test_reference_count": len(
            TEST_RE.findall(raw)
        ),
        "ps_traceback_count": len(
            TRACEBACK_RE.findall(raw)
        ),
        "ps_url_count": len(urls),
        "ps_code_fence_count": raw.count("```") // 2,
    }

    # --------------------------------------------------------
    # C) Security-Relevant Task Semantics
    # --------------------------------------------------------
    domain_flags = {}
    domain_counts = {}

    for domain, patterns in SECURITY_RELEVANT_PATTERNS.items():
        domain_flags[f"ps_domain_{domain}"] = has_patterns(
            lower,
            patterns,
        )
        domain_counts[f"ps_domain_{domain}_count"] = (
            count_patterns(
                lower,
                patterns,
            )
        )

    non_explicit_domains = [
        k
        for k in SECURITY_RELEVANT_PATTERNS
        if k != "explicit_security"
    ]

    security_features = {
        **domain_flags,
        **domain_counts,
        "ps_security_relevant_domain_count": sum(
            domain_flags[f"ps_domain_{d}"]
            for d in non_explicit_domains
        ),
        "ps_explicit_security_term_count": (
            domain_counts[
                "ps_domain_explicit_security_count"
            ]
        ),
    }

    return {
        **density_features,
        **technical_features,
        **security_features,
    }


# ============================================================
# FEATURE GROUPS
# ============================================================

DENSITY_COMPLEXITY = [
    "ps_chars",
    "ps_lines",
    "ps_words",
    "ps_questions",
    "ps_numeric_literal_count",
    "ps_identifier_count",
    "ps_bullet_count",
    "ps_constraint_count",
    "ps_modal_count",
    "ps_chars_per_line",
    "ps_words_per_line",
    "ps_identifier_density",
    "ps_numeric_density",
]

TECHNICAL_EVIDENCE = [
    "ps_file_path_count",
    "ps_unique_file_path_count",
    "ps_test_reference_count",
    "ps_traceback_count",
    "ps_url_count",
    "ps_code_fence_count",
]

SECURITY_RELEVANT = [
    c
    for c in (
        [f"ps_domain_{d}" for d in SECURITY_RELEVANT_PATTERNS]
        + [
            f"ps_domain_{d}_count"
            for d in SECURITY_RELEVANT_PATTERNS
        ]
        + [
            "ps_security_relevant_domain_count",
            "ps_explicit_security_term_count",
        ]
    )
]

FEATURE_SETS = {
    "Density / Complexity": DENSITY_COMPLEXITY,
    "Technical Evidence": TECHNICAL_EVIDENCE,
    "Security-Relevant Semantics": SECURITY_RELEVANT,

    "Density + Technical": (
        DENSITY_COMPLEXITY
        + TECHNICAL_EVIDENCE
    ),

    "Density + Security-Relevant": (
        DENSITY_COMPLEXITY
        + SECURITY_RELEVANT
    ),

    "Technical + Security-Relevant": (
        TECHNICAL_EVIDENCE
        + SECURITY_RELEVANT
    ),

    "Density + Technical + Security-Relevant": (
        DENSITY_COMPLEXITY
        + TECHNICAL_EVIDENCE
        + SECURITY_RELEVANT
    ),
}


# ============================================================
# ML
# ============================================================

def make_preprocessor(
    features: list[str],
) -> ColumnTransformer:
    return ColumnTransformer([
        (
            "num",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scale",
                    MinMaxScaler()
                ),
            ]),
            features,
        )
    ])


def make_models(
    features: list[str],
    seed: int,
) -> dict:

    return {
        "logistic_regression": Pipeline([
            (
                "prep",
                make_preprocessor(features),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=seed,
                ),
            ),
        ]),

        "random_forest": Pipeline([
            (
                "prep",
                make_preprocessor(features),
            ),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]),
    }


def metrics(
    y_true,
    y_pred,
    y_prob,
) -> dict:

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "balanced_accuracy": (
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision_1": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall_1": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1_1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            y_prob,
        ),
        "pr_auc": average_precision_score(
            y_true,
            y_prob,
        ),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def evaluate(
    df: pd.DataFrame,
    runs: int,
    base_seed: int,
) -> pd.DataFrame:

    y = df[
        "has_finding_after"
    ].astype(int).to_numpy()

    groups = df[
        "case"
    ].astype(str).to_numpy()

    rows = []

    for run_index in range(runs):
        seed = base_seed + run_index

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=seed,
        )

        train_idx, test_idx = next(
            splitter.split(
                df,
                y,
                groups=groups,
            )
        )

        if (
            len(np.unique(y[train_idx])) < 2
            or len(np.unique(y[test_idx])) < 2
        ):
            continue

        for feature_set, feature_cols in FEATURE_SETS.items():

            for classifier, model in make_models(
                feature_cols,
                seed,
            ).items():

                model.fit(
                    df.iloc[
                        train_idx
                    ][feature_cols],
                    y[train_idx],
                )

                prob = model.predict_proba(
                    df.iloc[
                        test_idx
                    ][feature_cols]
                )[:, 1]

                pred = (
                    prob >= 0.5
                ).astype(int)

                row = metrics(
                    y[test_idx],
                    pred,
                    prob,
                )

                row.update({
                    "run": run_index + 1,
                    "seed": seed,
                    "classifier": classifier,
                    "feature_set": feature_set,
                    "n_features": len(
                        feature_cols
                    ),
                })

                rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# ESTATÍSTICA
# ============================================================

def rank_biserial(
    delta: np.ndarray,
) -> float:

    d = np.asarray(
        delta,
        dtype=float,
    )

    d = d[
        np.isfinite(d)
    ]
    d = d[
        d != 0
    ]

    if len(d) == 0:
        return 0.0

    abs_d = np.abs(d)

    # ranks médios
    ranks = pd.Series(
        abs_d
    ).rank(
        method="average"
    ).to_numpy()

    w_pos = ranks[
        d > 0
    ].sum()

    w_neg = ranks[
        d < 0
    ].sum()

    denom = (
        w_pos + w_neg
    )

    return (
        float(
            (w_pos - w_neg)
            / denom
        )
        if denom
        else 0.0
    )


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:

    x = np.asarray(
        values,
        dtype=float,
    )

    rng = np.random.default_rng(
        seed
    )

    stats = []

    for _ in range(
        n_boot
    ):
        sample = rng.choice(
            x,
            size=len(x),
            replace=True,
        )
        stats.append(
            np.mean(sample)
        )

    return (
        float(
            np.quantile(
                stats,
                0.025,
            )
        ),
        float(
            np.quantile(
                stats,
                0.975,
            )
        ),
    )


def paired_test(
    runs_df: pd.DataFrame,
    baseline: str,
    enriched: str,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:

    rf = runs_df[
        runs_df["classifier"]
        .eq("random_forest")
    ]

    pivot = rf.pivot(
        index="run",
        columns="feature_set",
        values="roc_auc",
    )

    paired = pd.DataFrame({
        "run": pivot.index,
        "auc_baseline": (
            pivot[baseline].values
        ),
        "auc_enriched": (
            pivot[enriched].values
        ),
    })

    paired["delta_auc"] = (
        paired[
            "auc_enriched"
        ]
        - paired[
            "auc_baseline"
        ]
    )

    delta = paired[
        "delta_auc"
    ].to_numpy()

    w = wilcoxon(
        delta,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )

    ci_low, ci_high = (
        bootstrap_ci(
            delta,
            n_boot=n_boot,
            seed=seed,
        )
    )

    result = {
        "baseline": baseline,
        "enriched": enriched,
        "n_pairs": len(delta),
        "mean_auc_baseline": float(
            paired[
                "auc_baseline"
            ].mean()
        ),
        "mean_auc_enriched": float(
            paired[
                "auc_enriched"
            ].mean()
        ),
        "mean_delta_auc": float(
            paired[
                "delta_auc"
            ].mean()
        ),
        "median_delta_auc": float(
            paired[
                "delta_auc"
            ].median()
        ),
        "positive_runs": int(
            (
                paired[
                    "delta_auc"
                ] > 0
            ).sum()
        ),
        "wilcoxon_statistic": float(
            w.statistic
        ),
        "wilcoxon_p_value": float(
            w.pvalue
        ),
        "rank_biserial": (
            rank_biserial(
                delta
            )
        ),
        "bootstrap_ci95_low": (
            ci_low
        ),
        "bootstrap_ci95_high": (
            ci_high
        ),
    }

    return paired, result


# ============================================================
# FIGURA
# ============================================================

def plot_auc(
    summary: pd.DataFrame,
    out_path: Path,
) -> None:

    rf = summary[
        summary["classifier"]
        .eq("random_forest")
    ].copy()

    rf = rf.sort_values(
        "roc_auc_mean",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        rf["feature_set"],
        rf["roc_auc_mean"],
        xerr=rf[
            "roc_auc_std"
        ],
        capsize=3,
    )

    ax.axvline(
        0.5,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "ROC-AUC médio ± DP"
    )

    ax.set_title(
        "Ablação limpa do problem_statement — Random Forest"
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
    )

    ap.add_argument(
        "--cases-dir",
        default="runs/cases",
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
        default=(
            "problem_statement_ablation_results"
        ),
    )

    args = ap.parse_args()

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases_dir = Path(
        args.cases_dir
    )

    df = pd.read_csv(
        args.input
    )

    required = {
        "case",
        "model",
        "findings_after",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:
        raise SystemExit(
            f"Colunas ausentes: "
            f"{sorted(missing)}"
        )

    df[
        "has_finding_after"
    ] = (
        pd.to_numeric(
            df[
                "findings_after"
            ],
            errors="coerce",
        )
        .fillna(0)
        .gt(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # Extrai problem_statement por case
    # --------------------------------------------------------
    feature_rows = []

    missing_cases = 0

    for case in (
        df["case"]
        .astype(str)
        .drop_duplicates()
    ):

        text = (
            load_problem_statement(
                cases_dir,
                case,
            )
        )

        if not text:
            missing_cases += 1

        feature_rows.append({
            "case": case,
            "problem_statement_chars": (
                len(text)
            ),
            **extract_problem_features(
                text
            ),
        })

    feature_df = pd.DataFrame(
        feature_rows
    )

    enriched = df.merge(
        feature_df,
        on="case",
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Salva features limpas
    # --------------------------------------------------------
    feature_df.to_csv(
        out_dir
        / "problem_statement_features.csv",
        index=False,
    )

    enriched.to_csv(
        out_dir
        / "case_model_problem_statement_features.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Auditoria de prevalência
    # --------------------------------------------------------
    audit_rows = []

    for feature in (
        DENSITY_COMPLEXITY
        + TECHNICAL_EVIDENCE
        + SECURITY_RELEVANT
    ):

        values = (
            pd.to_numeric(
                enriched[
                    feature
                ],
                errors="coerce",
            )
            .fillna(0)
        )

        audit_rows.append({
            "feature": feature,
            "nonzero_rows": int(
                (
                    values != 0
                ).sum()
            ),
            "fraction_nonzero": float(
                (
                    values != 0
                ).mean()
            ),
            "mean": float(
                values.mean()
            ),
            "max": float(
                values.max()
            ),
        })

    audit = pd.DataFrame(
        audit_rows
    )

    audit.to_csv(
        out_dir
        / "problem_statement_feature_prevalence.csv",
        index=False,
    )

    # --------------------------------------------------------
    # ML
    # --------------------------------------------------------
    runs_df = evaluate(
        enriched,
        runs=args.runs,
        base_seed=args.seed,
    )

    runs_df.to_csv(
        out_dir
        / "ablation_all_runs.csv",
        index=False,
    )

    summary = (
        runs_df
        .groupby(
            [
                "classifier",
                "feature_set",
            ]
        )
        .agg({
            "accuracy": [
                "mean",
                "std",
            ],
            "balanced_accuracy": [
                "mean",
                "std",
            ],
            "precision_1": [
                "mean",
                "std",
            ],
            "recall_1": [
                "mean",
                "std",
            ],
            "f1_1": [
                "mean",
                "std",
            ],
            "roc_auc": [
                "mean",
                "std",
            ],
            "pr_auc": [
                "mean",
                "std",
            ],
        })
    )

    summary.columns = [
        "_".join(c)
        for c in summary.columns
    ]

    summary = (
        summary
        .reset_index()
    )

    summary.to_csv(
        out_dir
        / "ablation_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Testes principais
    # --------------------------------------------------------
    comparisons = [
        (
            "Technical Evidence",
            "Technical + Security-Relevant",
        ),
        (
            "Density / Complexity",
            "Density + Security-Relevant",
        ),
        (
            "Density + Technical",
            "Density + Technical + Security-Relevant",
        ),
    ]

    test_results = []

    for i, (
        baseline,
        enriched_name,
    ) in enumerate(
        comparisons
    ):

        paired, result = (
            paired_test(
                runs_df,
                baseline=baseline,
                enriched=enriched_name,
                n_boot=(
                    args.bootstrap_resamples
                ),
                seed=(
                    args.seed
                    + i
                ),
            )
        )

        paired.to_csv(
            out_dir
            / (
                "paired_"
                + baseline
                .lower()
                .replace(
                    " ",
                    "_",
                )
                .replace(
                    "/",
                    "_",
                )
                .replace(
                    "+",
                    "plus",
                )
                + "_vs_"
                + enriched_name
                .lower()
                .replace(
                    " ",
                    "_",
                )
                .replace(
                    "/",
                    "_",
                )
                .replace(
                    "+",
                    "plus",
                )
                + ".csv"
            ),
            index=False,
        )

        test_results.append(
            result
        )

    pd.DataFrame(
        test_results
    ).to_csv(
        out_dir
        / "paired_statistical_tests.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Figura
    # --------------------------------------------------------
    plot_auc(
        summary,
        out_dir
        / "fig_problem_statement_ablation_auc.png",
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------
    print()
    print("=" * 90)
    print("PROBLEM STATEMENT ABLATION")
    print("=" * 90)

    print(
        f"Rows: {len(enriched)}"
    )
    print(
        f"Cases: "
        f"{enriched['case'].nunique()}"
    )
    print(
        f"Missing problem_statement: "
        f"{missing_cases}"
    )
    print(
        f"Positivos: "
        f"{int(enriched['has_finding_after'].sum())}"
    )
    print()

    display = (
        summary[
            summary[
                "classifier"
            ].eq(
                "random_forest"
            )
        ][
            [
                "feature_set",
                "roc_auc_mean",
                "roc_auc_std",
                "recall_1_mean",
                "f1_1_mean",
            ]
        ]
        .sort_values(
            "roc_auc_mean",
            ascending=False,
        )
    )

    print(
        display.to_string(
            index=False
        )
    )

    print()
    print("TESTES PAREADOS")
    print(
        pd.DataFrame(
            test_results
        )[
            [
                "baseline",
                "enriched",
                "mean_delta_auc",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
                "wilcoxon_p_value",
                "rank_biserial",
                "positive_runs",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"Saída: {out_dir}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
