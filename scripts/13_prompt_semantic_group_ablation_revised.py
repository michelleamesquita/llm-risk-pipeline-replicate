#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
13_prompt_semantic_group_ablation.py — versão revisada

Objetivo
--------
Investigar QUAL grupo de características semânticas do prompt carrega sinal
preditivo sobre risco observado posteriormente pelo SAST.

Ajustes desta versão
--------------------
1. "Security semantics" foi renomeado para:
      Security-Relevant Task Semantics
   porque o prompt não precisa conter uma instrução explícita de segurança.
   O grupo representa DOMÍNIOS FUNCIONAIS potencialmente relevantes à segurança
   (autenticação, API, filesystem, execução de comandos, secrets etc.).

2. Auditoria das features:
   - quantas observações ativam cada feature semântica;
   - quantas têm instrução explícita de segurança;
   - opcionalmente, quais termos/padrões concretos do prompt acionaram cada
     domínio relevante à segurança.

3. Comparações pareadas:
   - Wilcoxon signed-rank test sobre o ROC-AUC das mesmas 30 execuções;
   - bootstrap pareado com IC 95% do ΔAUC;
   - rank-biserial correlation como tamanho de efeito.

4. Continua SEM utilizar CWE, severity, confidence ou qualquer saída pós-SAST
   como feature de entrada do classificador.

Entrada principal
-----------------
case_model_with_prompt_semantic_features.csv

Uso básico
----------
python scripts/13_prompt_semantic_group_ablation.py \
  --input semantic_prompt_results/case_model_with_prompt_semantic_features.csv \
  --runs 30 \
  --seed 42 \
  --out-dir semantic_group_ablation_results

Para auditoria lexical exata dos prompts, informe também:
  --runs-root runs_backup
  --cases-dir runs/cases

O script tenta ler o prompt real em:
  runs_backup/<modelo>/patches/<case>.prompt.txt

e usa runs/cases/<case>.json apenas como fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

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
# NOMENCLATURA E GRUPOS SEMÂNTICOS
# ============================================================

SECURITY_RELEVANT_TASK_SEMANTICS = [
    "sem_mentions_security",
    "sem_mentions_auth",
    "sem_mentions_input_validation",
    "sem_mentions_database",
    "sem_mentions_network_api",
    "sem_mentions_filesystem",
    "sem_mentions_command_exec",
    "sem_mentions_crypto_secret",
    "sem_mentions_dependency",
    "sem_mentions_serialization",
    "sem_mentions_web_output",
    "sem_mentions_permissions",
    "sem_mentions_concurrency",
    "sem_mentions_error_handling",
    "sem_security_term_count",
    "sem_domain_count",
]

CONSTRAINTS_REQUIREMENTS = [
    "sem_requirement_count",
    "sem_bullet_count",
    "sem_constraint_count",
    "sem_negation_constraint_count",
    "sem_modal_count",
    "sem_imperative_count",
]

FILES_TESTS_EVIDENCE = [
    "sem_file_path_count",
    "sem_unique_file_path_count",
    "sem_test_reference_count",
    "sem_traceback_count",
    "sem_identifier_count",
    "sem_url_count",
]

TASK_STRUCTURE_COMPLEXITY = [
    "sem_question_count",
    "sem_code_block_count",
    "sem_numeric_literal_count",
]

PROMPT_TEMPLATE_CONTEXT = [
    "sem_has_explicit_security_guidance",
    "sem_has_test_objective",
    "sem_has_relevant_files_section",
    "sem_has_repository_context",
]

ALL_SEMANTIC = (
    SECURITY_RELEVANT_TASK_SEMANTICS
    + CONSTRAINTS_REQUIREMENTS
    + FILES_TESTS_EVIDENCE
    + TASK_STRUCTURE_COMPLEXITY
    + PROMPT_TEMPLATE_CONTEXT
)

FEATURE_SETS = {
    "Security-Relevant Task Semantics": SECURITY_RELEVANT_TASK_SEMANTICS,
    "Constraints / requirements": CONSTRAINTS_REQUIREMENTS,
    "Files / tests / evidence": FILES_TESTS_EVIDENCE,
    "Task structure / complexity": TASK_STRUCTURE_COMPLEXITY,
    "Prompt template / context": PROMPT_TEMPLATE_CONTEXT,

    "Security-Relevant + Constraints": (
        SECURITY_RELEVANT_TASK_SEMANTICS
        + CONSTRAINTS_REQUIREMENTS
    ),

    "Security-Relevant + Files/Tests": (
        SECURITY_RELEVANT_TASK_SEMANTICS
        + FILES_TESTS_EVIDENCE
    ),

    "Security-Relevant + Constraints + Files/Tests": (
        SECURITY_RELEVANT_TASK_SEMANTICS
        + CONSTRAINTS_REQUIREMENTS
        + FILES_TESTS_EVIDENCE
    ),

    "All semantic": ALL_SEMANTIC,
}


# ============================================================
# AUDITORIA LEXICAL — DOMÍNIOS FUNCIONAIS
# ============================================================

# A intenção aqui NÃO é declarar que o prompt é "de segurança".
# Esses padrões apenas marcam domínios funcionais que podem ter
# implicações de segurança.

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


MODEL_DIR_CANDIDATES = {
    "gpt-4o": ["gpt-4o_backup", "gpt-4o"],
    "claude": ["claud-sonnet_backup", "claude_backup", "claude"],
    "deepseek": ["deepseek_backup", "deepseek"],
    "codellama-tuned": [
        "codellama_tuned_backup",
        "codellama-tuned_backup",
        "codellama_backup",
        "codellama-tuned",
    ],
    "codellama": [
        "codellama_backup",
        "codellama_tuned_backup",
        "codellama-tuned",
    ],
}


def unique_regex_matches(text: str, patterns: Iterable[str]) -> list[str]:
    """
    Retorna os termos concretos encontrados, não apenas o nome do regex.
    """
    found: list[str] = []
    seen = set()

    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            term = match.group(0)
            key = term.lower()
            if key not in seen:
                seen.add(key)
                found.append(term)

    return found


def find_prompt_file(
    runs_root: Path,
    model: str,
    case: str,
) -> Path | None:

    for run_name in MODEL_DIR_CANDIDATES.get(model, [model]):
        p = runs_root / run_name / "patches" / f"{case}.prompt.txt"
        if p.is_file():
            return p

    for p in runs_root.glob(f"*/patches/{case}.prompt.txt"):
        if p.is_file():
            return p

    return None


def prompt_from_case_json(cases_dir: Path, case: str) -> str:
    path = cases_dir / f"{case}.json"
    if not path.is_file():
        return ""

    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    return "\n".join(
        str(x)
        for x in [
            obj.get("issue_title") or "",
            obj.get("issue_body") or obj.get("problem_statement") or "",
            obj.get("short_file_list") or "",
            obj.get("test_query") or "",
        ]
        if x
    )


def load_prompt_for_audit(
    runs_root: Path,
    cases_dir: Path,
    model: str,
    case: str,
) -> tuple[str, str]:

    p = find_prompt_file(runs_root, model, case)

    if p:
        return (
            p.read_text(encoding="utf-8", errors="ignore"),
            str(p),
        )

    fallback = prompt_from_case_json(cases_dir, case)

    if fallback:
        return fallback, f"case_json:{case}"

    return "", ""


def build_lexical_audit(
    df: pd.DataFrame,
    runs_root: Path,
    cases_dir: Path,
) -> pd.DataFrame:

    rows = []

    for _, row in df[["case", "model"]].drop_duplicates().iterrows():
        case = str(row["case"])
        model = str(row["model"])

        text, source = load_prompt_for_audit(
            runs_root,
            cases_dir,
            model,
            case,
        )

        result = {
            "case": case,
            "model": model,
            "prompt_source": source,
            "prompt_chars_for_audit": len(text),
        }

        any_relevant = False

        for domain, patterns in SECURITY_RELEVANT_PATTERNS.items():
            matches = unique_regex_matches(text, patterns)

            result[f"audit_{domain}_count"] = len(matches)
            result[f"audit_{domain}_terms"] = " | ".join(matches)

            if domain != "explicit_security" and matches:
                any_relevant = True

        result["audit_has_explicit_security_terms"] = int(
            result["audit_explicit_security_count"] > 0
        )
        result["audit_has_security_relevant_task_terms"] = int(any_relevant)

        rows.append(result)

    return pd.DataFrame(rows)


# ============================================================
# FEATURE PREVALENCE AUDIT
# ============================================================

def build_feature_prevalence(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for feature in ALL_SEMANTIC:
        values = pd.to_numeric(
            df[feature],
            errors="coerce",
        ).fillna(0)

        rows.append({
            "feature": feature,
            "nonzero_rows": int((values != 0).sum()),
            "nonzero_fraction": float((values != 0).mean()),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "max": float(values.max()),
            "unique_values": int(values.nunique(dropna=True)),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["nonzero_fraction", "feature"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


# ============================================================
# ML
# ============================================================

def make_preprocessor(features: list[str]) -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", MinMaxScaler()),
    ])

    return ColumnTransformer([
        ("num", numeric, features),
    ])


def make_models(features: list[str], seed: int) -> dict:
    return {
        "logistic_regression": Pipeline([
            ("prep", make_preprocessor(features)),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=seed,
            )),
        ]),

        "random_forest": Pipeline([
            ("prep", make_preprocessor(features)),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )),
        ]),
    }


def classification_metrics(
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
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "precision_1": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall_1": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, y_prob)
        ),
        "pr_auc": float(
            average_precision_score(y_true, y_prob)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate(
    df: pd.DataFrame,
    runs: int,
    seed: int,
) -> pd.DataFrame:

    y = df["has_finding_after"].astype(int).to_numpy()
    groups = df["case"].astype(str).to_numpy()

    rows = []

    for run_index in range(runs):
        run_seed = seed + run_index

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=run_seed,
        )

        train_idx, test_idx = next(
            splitter.split(df, y, groups=groups)
        )

        if (
            len(np.unique(y[train_idx])) < 2
            or len(np.unique(y[test_idx])) < 2
        ):
            continue

        for feature_set_name, feature_cols in FEATURE_SETS.items():

            for classifier_name, model in make_models(
                feature_cols,
                run_seed,
            ).items():

                model.fit(
                    df.iloc[train_idx][feature_cols],
                    y[train_idx],
                )

                prob = model.predict_proba(
                    df.iloc[test_idx][feature_cols]
                )[:, 1]

                pred = (prob >= 0.5).astype(int)

                result = classification_metrics(
                    y[test_idx],
                    pred,
                    prob,
                )

                result.update({
                    "run": run_index + 1,
                    "seed": run_seed,
                    "classifier": classifier_name,
                    "feature_set": feature_set_name,
                    "n_features": len(feature_cols),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "train_cases": len(np.unique(groups[train_idx])),
                    "test_cases": len(np.unique(groups[test_idx])),
                })

                rows.append(result)

    return pd.DataFrame(rows)


# ============================================================
# IMPORTÂNCIA
# ============================================================

def fit_rf_importance(
    df: pd.DataFrame,
    features: list[str],
    seed: int,
) -> pd.DataFrame:

    y = df["has_finding_after"].astype(int).to_numpy()

    model = Pipeline([
        ("prep", make_preprocessor(features)),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )),
    ])

    model.fit(df[features], y)

    values = model.named_steps["clf"].feature_importances_

    return (
        pd.DataFrame({
            "feature": features,
            "importance": values,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ============================================================
# ESTATÍSTICA PAREADA
# ============================================================

def paired_rank_biserial(delta: np.ndarray) -> float:
    """
    Rank-biserial correlation para dados pareados.

    RBC = (W+ - W-) / (W+ + W-)
    Ignora diferenças exatamente iguais a zero.
    """
    d = np.asarray(delta, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]

    if len(d) == 0:
        return 0.0

    abs_d = np.abs(d)
    order = np.argsort(abs_d)

    ranks = np.empty(len(d), dtype=float)

    # rank médio para empates
    sorted_abs = abs_d[order]
    i = 0
    current_rank = 1

    while i < len(sorted_abs):
        j = i + 1
        while j < len(sorted_abs) and sorted_abs[j] == sorted_abs[i]:
            j += 1

        rank_values = np.arange(
            current_rank,
            current_rank + (j - i),
            dtype=float,
        )

        mean_rank = rank_values.mean()

        for k in range(i, j):
            ranks[order[k]] = mean_rank

        current_rank += (j - i)
        i = j

    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()

    denom = w_pos + w_neg

    if denom == 0:
        return 0.0

    return float((w_pos - w_neg) / denom)


def bootstrap_ci(
    values: np.ndarray,
    statistic: str = "mean",
    n_boot: int = 10000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float]:

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return (math.nan, math.nan)

    rng = np.random.default_rng(seed)

    stats = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sample = rng.choice(
            x,
            size=len(x),
            replace=True,
        )

        if statistic == "median":
            stats[i] = np.median(sample)
        else:
            stats[i] = np.mean(sample)

    alpha = 1 - confidence

    low = np.quantile(stats, alpha / 2)
    high = np.quantile(stats, 1 - alpha / 2)

    return float(low), float(high)


def paired_auc_test(
    all_runs: pd.DataFrame,
    baseline: str,
    enriched: str,
    seed: int,
    n_boot: int,
) -> tuple[pd.DataFrame, dict]:

    rf = all_runs[
        all_runs["classifier"].eq("random_forest")
    ]

    pivot = rf.pivot(
        index="run",
        columns="feature_set",
        values="roc_auc",
    )

    if baseline not in pivot.columns:
        raise ValueError(
            f"Baseline ausente na tabela: {baseline}"
        )

    if enriched not in pivot.columns:
        raise ValueError(
            f"Modelo enriquecido ausente: {enriched}"
        )

    paired = pd.DataFrame({
        "run": pivot.index,
        "auc_baseline": pivot[baseline].values,
        "auc_enriched": pivot[enriched].values,
    })

    paired["delta_auc"] = (
        paired["auc_enriched"]
        - paired["auc_baseline"]
    )

    delta = paired["delta_auc"].to_numpy(dtype=float)

    # Wilcoxon two-sided; zero_method='wilcox' remove zeros.
    try:
        w = wilcoxon(
            delta,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )
        wilcoxon_stat = float(w.statistic)
        wilcoxon_p = float(w.pvalue)
    except ValueError:
        wilcoxon_stat = math.nan
        wilcoxon_p = math.nan

    mean_ci = bootstrap_ci(
        delta,
        statistic="mean",
        n_boot=n_boot,
        seed=seed,
    )

    median_ci = bootstrap_ci(
        delta,
        statistic="median",
        n_boot=n_boot,
        seed=seed + 1,
    )

    result = {
        "baseline": baseline,
        "enriched": enriched,
        "n_pairs": int(len(delta)),
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
        "positive_delta_runs": int(
            (paired["delta_auc"] > 0).sum()
        ),
        "negative_delta_runs": int(
            (paired["delta_auc"] < 0).sum()
        ),
        "zero_delta_runs": int(
            (paired["delta_auc"] == 0).sum()
        ),
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_p_value": wilcoxon_p,
        "rank_biserial_correlation": (
            paired_rank_biserial(delta)
        ),
        "bootstrap_mean_delta_ci95_low": mean_ci[0],
        "bootstrap_mean_delta_ci95_high": mean_ci[1],
        "bootstrap_median_delta_ci95_low": median_ci[0],
        "bootstrap_median_delta_ci95_high": median_ci[1],
        "bootstrap_resamples": int(n_boot),
    }

    return paired, result


# ============================================================
# FIGURAS
# ============================================================

def plot_auc(summary: pd.DataFrame, out_path: Path) -> None:
    rf = summary[
        summary["classifier"].eq("random_forest")
    ].copy()

    rf = rf.sort_values(
        "roc_auc_mean",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.barh(
        rf["feature_set"],
        rf["roc_auc_mean"],
        xerr=rf["roc_auc_std"],
        capsize=3,
    )

    ax.axvline(
        0.5,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel("ROC-AUC médio ± DP")
    ax.set_title(
        "Ablação de grupos semânticos do prompt — Random Forest"
    )

    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_recall(summary: pd.DataFrame, out_path: Path) -> None:
    rf = summary[
        summary["classifier"].eq("random_forest")
    ].copy()

    rf = rf.sort_values(
        "recall_1_mean",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.barh(
        rf["feature_set"],
        rf["recall_1_mean"],
        xerr=rf["recall_1_std"],
        capsize=3,
    )

    ax.set_xlim(0, 1)

    ax.set_xlabel(
        "Recall médio da classe de risco ± DP"
    )

    ax.set_title(
        "Recall por grupo semântico — Random Forest"
    )

    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_feature_importance(
    importance: pd.DataFrame,
    out_path: Path,
    top_n: int = 20,
) -> None:

    data = (
        importance
        .head(top_n)
        .sort_values(
            "importance",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.barh(
        data["feature"],
        data["importance"],
    )

    ax.set_xlabel("Importância")

    ax.set_title(
        "Importância das features semânticas — Random Forest"
    )

    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_paired_delta(
    paired: pd.DataFrame,
    title: str,
    out_path: Path,
) -> None:

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        paired["run"].astype(str),
        paired["delta_auc"],
    )

    ax.axhline(0, linewidth=1)

    ax.set_xlabel("Execução")
    ax.set_ylabel("Δ ROC-AUC")
    ax.set_title(title)

    ax.tick_params(
        axis="x",
        labelsize=7,
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
            "CSV enriquecido do script 12, por exemplo "
            "case_model_with_prompt_semantic_features.csv"
        ),
    )

    ap.add_argument(
        "--runs-root",
        default="runs_backup",
        help="Usado para auditoria lexical dos prompts.",
    )

    ap.add_argument(
        "--cases-dir",
        default="runs/cases",
        help="Fallback da auditoria lexical.",
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
        default="semantic_group_ablation_results",
    )

    args = ap.parse_args()

    if args.runs < 1:
        raise SystemExit("--runs deve ser >= 1")

    if args.bootstrap_resamples < 1000:
        raise SystemExit(
            "--bootstrap-resamples deve ser >= 1000"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(args.input)

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------
    if "has_finding_after" not in df.columns:
        if "findings_after" not in df.columns:
            raise SystemExit(
                "CSV precisa de has_finding_after ou findings_after."
            )

        df["has_finding_after"] = (
            pd.to_numeric(
                df["findings_after"],
                errors="coerce",
            )
            .fillna(0)
            .gt(0)
            .astype(int)
        )

    # --------------------------------------------------------
    # Schema
    # --------------------------------------------------------
    required = {
        "case",
        "model",
        "has_finding_after",
        *ALL_SEMANTIC,
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise SystemExit(
            "Features ausentes no CSV:\n- "
            + "\n- ".join(missing)
        )

    for col in ALL_SEMANTIC:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        ).fillna(0)

    # --------------------------------------------------------
    # Auditoria das features já calculadas
    # --------------------------------------------------------
    prevalence = build_feature_prevalence(df)

    prevalence.to_csv(
        out_dir / "semantic_feature_prevalence.csv",
        index=False,
    )

    explicit_security_summary = {
        "sem_mentions_security_nonzero": int(
            (df["sem_mentions_security"] != 0).sum()
        ),
        "sem_mentions_security_fraction": float(
            (df["sem_mentions_security"] != 0).mean()
        ),
        "sem_has_explicit_security_guidance_nonzero": int(
            (df["sem_has_explicit_security_guidance"] != 0).sum()
        ),
        "sem_has_explicit_security_guidance_fraction": float(
            (
                df["sem_has_explicit_security_guidance"] != 0
            ).mean()
        ),
    }

    # --------------------------------------------------------
    # Auditoria lexical exata dos prompts
    # --------------------------------------------------------
    runs_root = Path(args.runs_root)
    cases_dir = Path(args.cases_dir)

    lexical_audit = build_lexical_audit(
        df,
        runs_root,
        cases_dir,
    )

    lexical_audit.to_csv(
        out_dir / "security_relevant_term_audit.csv",
        index=False,
    )

    lexical_summary_rows = []

    for domain in SECURITY_RELEVANT_PATTERNS:
        count_col = f"audit_{domain}_count"

        lexical_summary_rows.append({
            "domain": domain,
            "rows_with_match": int(
                (lexical_audit[count_col] > 0).sum()
            ),
            "fraction_with_match": float(
                (lexical_audit[count_col] > 0).mean()
            ),
            "total_distinct_term_hits": int(
                lexical_audit[count_col].sum()
            ),
        })

    lexical_summary = pd.DataFrame(
        lexical_summary_rows
    ).sort_values(
        "fraction_with_match",
        ascending=False,
    )

    lexical_summary.to_csv(
        out_dir / "security_relevant_domain_prevalence.csv",
        index=False,
    )

    # --------------------------------------------------------
    # ML
    # --------------------------------------------------------
    all_runs = evaluate(
        df=df,
        runs=args.runs,
        seed=args.seed,
    )

    all_runs.to_csv(
        out_dir / "semantic_group_all_runs.csv",
        index=False,
    )

    summary = (
        all_runs
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
        "_".join(col)
        for col in summary.columns
    ]

    summary = (
        summary
        .reset_index()
        .sort_values(
            ["classifier", "roc_auc_mean"],
            ascending=[True, False],
        )
    )

    summary.to_csv(
        out_dir / "semantic_group_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------
    importance = fit_rf_importance(
        df,
        ALL_SEMANTIC,
        args.seed,
    )

    importance.to_csv(
        out_dir / "semantic_feature_importance_rf.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Estatística pareada principal
    # --------------------------------------------------------
    paired_security, test_security = paired_auc_test(
        all_runs,
        baseline="Files / tests / evidence",
        enriched="Security-Relevant + Files/Tests",
        seed=args.seed,
        n_boot=args.bootstrap_resamples,
    )

    paired_security.to_csv(
        out_dir
        / "paired_delta_security_relevant_over_files_tests.csv",
        index=False,
    )

    pd.DataFrame([test_security]).to_csv(
        out_dir
        / "paired_test_security_relevant_over_files_tests.csv",
        index=False,
    )

    # Segunda comparação: Security-Relevant sozinho + constraints
    paired_constraints, test_constraints = paired_auc_test(
        all_runs,
        baseline="Security-Relevant Task Semantics",
        enriched="Security-Relevant + Constraints",
        seed=args.seed + 100,
        n_boot=args.bootstrap_resamples,
    )

    paired_constraints.to_csv(
        out_dir
        / "paired_delta_constraints_over_security_relevant.csv",
        index=False,
    )

    pd.DataFrame([test_constraints]).to_csv(
        out_dir
        / "paired_test_constraints_over_security_relevant.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Figuras
    # --------------------------------------------------------
    plot_auc(
        summary,
        out_dir / "fig_semantic_group_auc.png",
    )

    plot_recall(
        summary,
        out_dir / "fig_semantic_group_recall.png",
    )

    plot_feature_importance(
        importance,
        out_dir / "fig_semantic_feature_importance.png",
    )

    plot_paired_delta(
        paired_security,
        title=(
            "Ganho ao adicionar Security-Relevant Task Semantics "
            "a Files/Tests"
        ),
        out_path=(
            out_dir
            / "fig_delta_security_relevant_over_files_tests.png"
        ),
    )

    plot_paired_delta(
        paired_constraints,
        title=(
            "Ganho ao adicionar Constraints "
            "a Security-Relevant Task Semantics"
        ),
        out_path=(
            out_dir
            / "fig_delta_constraints_over_security_relevant.png"
        ),
    )

    # --------------------------------------------------------
    # JSON auditável
    # --------------------------------------------------------
    report = {
        "rows": int(len(df)),
        "unique_cases": int(
            df["case"].nunique()
        ),
        "positives": int(
            df["has_finding_after"].sum()
        ),
        "negatives": int(
            len(df)
            - df["has_finding_after"].sum()
        ),
        "runs_requested": int(args.runs),
        "base_seed": int(args.seed),
        "split": (
            "80/20 GroupShuffleSplit grouped by case"
        ),
        "target": (
            "has_finding_after = findings_after > 0"
        ),
        "feature_group_name": (
            "Security-Relevant Task Semantics"
        ),
        "feature_group_definition": (
            "Functional prompt domains potentially relevant to security; "
            "does not imply that the prompt explicitly requests security."
        ),
        "explicit_security_audit": (
            explicit_security_summary
        ),
        "feature_groups": FEATURE_SETS,
        "post_sast_features_used": False,
        "classifiers": [
            "logistic_regression",
            "random_forest",
        ],
        "primary_paired_test": test_security,
        "secondary_paired_test": test_constraints,
    }

    (
        out_dir
        / "semantic_group_run_summary.json"
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
    print("PROMPT SEMANTIC GROUP ABLATION — REVISED")
    print("=" * 92)

    print(f"Rows       : {len(df)}")
    print(
        f"Cases      : "
        f"{df['case'].nunique()}"
    )
    print(
        f"Positivos  : "
        f"{int(df['has_finding_after'].sum())}"
    )
    print(
        f"Negativos  : "
        f"{int(len(df) - df['has_finding_after'].sum())}"
    )

    print()
    print("AUDITORIA DE SEGURANÇA EXPLÍCITA")
    print(
        "sem_mentions_security != 0: "
        f"{explicit_security_summary['sem_mentions_security_nonzero']}"
        f"/{len(df)}"
    )
    print(
        "sem_has_explicit_security_guidance != 0: "
        f"{explicit_security_summary['sem_has_explicit_security_guidance_nonzero']}"
        f"/{len(df)}"
    )

    print()
    display_cols = [
        "classifier",
        "feature_set",
        "roc_auc_mean",
        "roc_auc_std",
        "recall_1_mean",
        "recall_1_std",
        "f1_1_mean",
    ]

    print(
        summary[display_cols]
        .to_string(index=False)
    )

    print()
    print("TOP 15 SEMANTIC FEATURES — RF")
    print(
        importance
        .head(15)
        .to_string(index=False)
    )

    print()
    print("TESTE PAREADO PRINCIPAL")
    print(
        f"Baseline : {test_security['baseline']}"
    )
    print(
        f"Enriched : {test_security['enriched']}"
    )
    print(
        f"ΔAUC médio   : "
        f"{test_security['mean_delta_auc']:.4f}"
    )
    print(
        f"IC95% bootstrap da média: "
        f"[{test_security['bootstrap_mean_delta_ci95_low']:.4f}, "
        f"{test_security['bootstrap_mean_delta_ci95_high']:.4f}]"
    )
    print(
        f"ΔAUC mediano : "
        f"{test_security['median_delta_auc']:.4f}"
    )
    print(
        f"Wilcoxon p   : "
        f"{test_security['wilcoxon_p_value']:.6f}"
    )
    print(
        f"Rank-biserial: "
        f"{test_security['rank_biserial_correlation']:.4f}"
    )
    print(
        f"Melhorou em  : "
        f"{test_security['positive_delta_runs']}/"
        f"{test_security['n_pairs']} runs"
    )

    print()
    print(f"Saída: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
