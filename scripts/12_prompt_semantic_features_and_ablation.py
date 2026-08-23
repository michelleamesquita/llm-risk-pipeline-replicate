#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
12_prompt_semantic_features_and_ablation.py

Objetivo
--------
Enriquecer o dataset case x model com características SEMÂNTICAS do prompt
disponíveis ANTES do SAST e testar se elas acrescentam poder preditivo.

O script:
1. lê case_model_before_after_summary_common.csv;
2. recupera o prompt real usado por cada case/model em runs_backup;
3. se não encontrar, reconstrói texto-base a partir de runs/cases/<case>.json;
4. extrai features semânticas determinísticas (regex/contagens);
5. NÃO usa CWE, severity, confidence ou qualquer saída do Bandit como feature;
6. treina:
   - Regressão Logística (baseline)
   - Random Forest (principal, protocolo do artigo)
7. repete N=30 splits 80/20 agrupados por case;
8. executa ablação:
   - Model only
   - Prompt size only
   - Prompt semantic only
   - Patch only
   - Patch + Model
   - Patch + Prompt semantic
   - Prompt size + Prompt semantic
   - Prompt semantic + Patch + Model
   - All pre-SAST features
9. gera CSVs e figuras.

Uso
---
python scripts/12_prompt_semantic_features_and_ablation.py \
  --input case_model_before_after_summary_common.csv \
  --runs-root runs_backup \
  --cases-dir runs/cases \
  --runs 30 \
  --seed 42 \
  --out-dir semantic_prompt_results

Observação metodológica
-----------------------
As features semânticas são lexicais/determinísticas e auditáveis.
Nenhuma delas depende da resposta do Bandit.
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
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# ---------------------------------------------------------------------
# Configuração de modelos/diretórios
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Grupos de features
# ---------------------------------------------------------------------

PATCH_FEATURES = [
    "patch_lines",
    "patch_added",
    "patch_removed",
    "patch_files_touched",
    "patch_hunks",
    "patch_churn",
    "patch_net",
    "patch_density",
    "add_remove_ratio",
    "net_per_line",
    "hunks_per_file",
    "patch_complexity",
    "change_intensity",
]

PROMPT_SIZE_FEATURES = [
    "prompt_chars",
    "prompt_lines",
    "prompt_tokens",
    "prompt_density",
    "prompt_token_density",
    "prompt_size_category",
]

PROMPT_SEMANTIC_FEATURES = [
    # estrutura/requisitos
    "sem_requirement_count",
    "sem_bullet_count",
    "sem_question_count",
    "sem_code_block_count",
    "sem_traceback_count",
    "sem_file_path_count",
    "sem_unique_file_path_count",
    "sem_test_reference_count",
    "sem_constraint_count",
    "sem_negation_constraint_count",

    # domínios de segurança/engenharia
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

    # escopo/complexidade textual
    "sem_security_term_count",
    "sem_domain_count",
    "sem_modal_count",
    "sem_imperative_count",
    "sem_identifier_count",
    "sem_numeric_literal_count",
    "sem_url_count",
    "sem_has_explicit_security_guidance",
    "sem_has_test_objective",
    "sem_has_relevant_files_section",
    "sem_has_repository_context",
]


# ---------------------------------------------------------------------
# Vocabulários determinísticos
# ---------------------------------------------------------------------

DOMAIN_PATTERNS = {
    "security": [
        r"\bsecurity\b", r"\bsecure\b", r"\bvulnerab", r"\bcwe-\d+\b",
        r"\bowasp\b", r"\binjection\b", r"\bxss\b", r"\bsqli?\b",
        r"\bexploit", r"\bunsafe\b",
    ],
    "auth": [
        r"\bauthentication\b", r"\bauthorization\b", r"\bauth\b",
        r"\blogin\b", r"\bsession\b", r"\btoken\b", r"\bjwt\b",
        r"\bcredential", r"\bpassword\b", r"\bpermission",
    ],
    "input_validation": [
        r"\bvalidate\b", r"\bvalidation\b", r"\bsanitize\b", r"\bescape\b",
        r"\buser input\b", r"\buntrusted\b", r"\bparameter\b",
        r"\brequest data\b", r"\bpayload\b",
    ],
    "database": [
        r"\bdatabase\b", r"\bsql\b", r"\bquery\b", r"\borm\b",
        r"\bpostgres", r"\bmysql\b", r"\bsqlite\b", r"\btransaction\b",
    ],
    "network_api": [
        r"\bapi\b", r"\bhttp\b", r"\bhttps\b", r"\brequest\b",
        r"\bresponse\b", r"\bendpoint\b", r"\bsocket\b", r"\bnetwork\b",
        r"\burl\b", r"\bwebhook\b",
    ],
    "filesystem": [
        r"\bfile\b", r"\bfilesystem\b", r"\bpath\b", r"\bdirectory\b",
        r"\bfolder\b", r"\bread\b", r"\bwrite\b", r"\bopen\(",
        r"\btempfile\b",
    ],
    "command_exec": [
        r"\bshell\b", r"\bcommand\b", r"\bsubprocess\b", r"\bos\.system\b",
        r"\bexec\(", r"\beval\(", r"\bspawn\b", r"\bpopen\b",
    ],
    "crypto_secret": [
        r"\bcrypto", r"\bencrypt", r"\bdecrypt", r"\bhash\b",
        r"\bsecret\b", r"\bprivate key\b", r"\bcertificate\b",
        r"\bssl\b", r"\btls\b", r"\bkey\b",
    ],
    "dependency": [
        r"\bdependency\b", r"\bdependencies\b", r"\bpackage\b",
        r"\bimport\b", r"\brequirement", r"\bversion\b",
    ],
    "serialization": [
        r"\bserializ", r"\bdeserializ", r"\bpickle\b", r"\byaml\b",
        r"\bjson\b", r"\bmarshal\b",
    ],
    "web_output": [
        r"\bhtml\b", r"\btemplate\b", r"\brender\b", r"\bresponse\b",
        r"\bjavascript\b", r"\bdom\b", r"\boutput\b",
    ],
    "permissions": [
        r"\bpermission\b", r"\bprivilege\b", r"\baccess control\b",
        r"\brole\b", r"\bowner\b", r"\badmin\b",
    ],
    "concurrency": [
        r"\bthread\b", r"\basync\b", r"\bawait\b", r"\brace condition\b",
        r"\block\b", r"\bconcurrent\b",
    ],
    "error_handling": [
        r"\bexception\b", r"\btraceback\b", r"\berror handling\b",
        r"\btry\b", r"\bexcept\b", r"\braise\b", r"\bfailure\b",
    ],
}

CONSTRAINT_PATTERNS = [
    r"\bmust\b", r"\bshould\b", r"\brequired\b", r"\brequirement\b",
    r"\bpreserve\b", r"\bmaintain\b", r"\bcompatible\b",
    r"\bdo not\b", r"\bdon't\b", r"\bwithout\b", r"\bonly\b",
    r"\bavoid\b", r"\bnever\b", r"\bno new\b",
]

NEGATION_PATTERNS = [
    r"\bdo not\b", r"\bdon't\b", r"\bmust not\b", r"\bshould not\b",
    r"\bnever\b", r"\bwithout\b", r"\bavoid\b", r"\bno new\b",
]

MODAL_PATTERNS = [
    r"\bmust\b", r"\bshould\b", r"\bshall\b", r"\bneed(?:s|ed)? to\b",
    r"\brequired to\b", r"\bexpected to\b",
]

IMPERATIVE_WORDS = [
    "fix", "generate", "preserve", "maintain", "return", "implement",
    "update", "change", "remove", "add", "ensure", "avoid", "use",
    "handle", "support", "validate", "prevent", "make",
]


# ---------------------------------------------------------------------
# Texto e regex
# ---------------------------------------------------------------------

FILE_PATH_RE = re.compile(
    r"(?<![\w./-])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.(?:py|js|ts|java|go|rb|php|cs|cpp|c|h|json|yaml|yml|toml|ini|cfg|md)"
)
TRACEBACK_RE = re.compile(r"\bTraceback\b|File\s+[\"'][^\"']+[\"'],\s+line\s+\d+", re.I)
TEST_RE = re.compile(r"\btest(?:s|ing)?\b|pytest|unittest|FAIL_TO_PASS|PASS_TO_PASS", re.I)
URL_RE = re.compile(r"https?://[^\s)\]>]+", re.I)
IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_.]*\b")
NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def count_patterns(text: str, patterns: Iterable[str]) -> int:
    return sum(len(re.findall(p, text, flags=re.I)) for p in patterns)


def has_patterns(text: str, patterns: Iterable[str]) -> int:
    return int(any(re.search(p, text, flags=re.I) for p in patterns))


def semantic_prompt_features(text: str) -> dict:
    raw = str(text or "")
    lower = raw.lower()
    lines = raw.splitlines()

    file_paths = FILE_PATH_RE.findall(raw)
    bullets = sum(
        1 for line in lines
        if re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", line)
    )

    # requisitos: bullets + modais/regras explícitas
    modal_count = count_patterns(lower, MODAL_PATTERNS)
    constraint_count = count_patterns(lower, CONSTRAINT_PATTERNS)

    code_fences = raw.count("```") // 2
    tracebacks = len(TRACEBACK_RE.findall(raw))
    tests = len(TEST_RE.findall(raw))

    domain_hits = {}
    for domain, pats in DOMAIN_PATTERNS.items():
        domain_hits[domain] = has_patterns(lower, pats)

    security_term_count = count_patterns(
        lower,
        DOMAIN_PATTERNS["security"]
        + DOMAIN_PATTERNS["auth"]
        + DOMAIN_PATTERNS["input_validation"]
        + DOMAIN_PATTERNS["command_exec"]
        + DOMAIN_PATTERNS["crypto_secret"]
    )

    imperative_count = sum(
        len(re.findall(rf"(?im)^\s*(?:[-*]\s*)?{re.escape(word)}\b", raw))
        for word in IMPERATIVE_WORDS
    )

    requirement_count = (
        bullets
        + modal_count
        + len(re.findall(r"(?im)^\s*\[(?:RULES|OBJECTIVE|REQUIREMENTS?)\]\s*$", raw))
    )

    return {
        "sem_requirement_count": requirement_count,
        "sem_bullet_count": bullets,
        "sem_question_count": raw.count("?"),
        "sem_code_block_count": code_fences,
        "sem_traceback_count": tracebacks,
        "sem_file_path_count": len(file_paths),
        "sem_unique_file_path_count": len(set(file_paths)),
        "sem_test_reference_count": tests,
        "sem_constraint_count": constraint_count,
        "sem_negation_constraint_count": count_patterns(lower, NEGATION_PATTERNS),

        "sem_mentions_security": domain_hits["security"],
        "sem_mentions_auth": domain_hits["auth"],
        "sem_mentions_input_validation": domain_hits["input_validation"],
        "sem_mentions_database": domain_hits["database"],
        "sem_mentions_network_api": domain_hits["network_api"],
        "sem_mentions_filesystem": domain_hits["filesystem"],
        "sem_mentions_command_exec": domain_hits["command_exec"],
        "sem_mentions_crypto_secret": domain_hits["crypto_secret"],
        "sem_mentions_dependency": domain_hits["dependency"],
        "sem_mentions_serialization": domain_hits["serialization"],
        "sem_mentions_web_output": domain_hits["web_output"],
        "sem_mentions_permissions": domain_hits["permissions"],
        "sem_mentions_concurrency": domain_hits["concurrency"],
        "sem_mentions_error_handling": domain_hits["error_handling"],

        "sem_security_term_count": security_term_count,
        "sem_domain_count": sum(domain_hits.values()),
        "sem_modal_count": modal_count,
        "sem_imperative_count": imperative_count,
        "sem_identifier_count": len(IDENTIFIER_RE.findall(raw)),
        "sem_numeric_literal_count": len(NUMERIC_RE.findall(raw)),
        "sem_url_count": len(URL_RE.findall(raw)),
        "sem_has_explicit_security_guidance": int(
            "[security-guidelines]" in lower
            or "security guidelines" in lower
            or "secure coding" in lower
        ),
        "sem_has_test_objective": int(
            "[objective]" in lower
            or "tests pass" in lower
            or "make these tests pass" in lower
        ),
        "sem_has_relevant_files_section": int("[relevant files]" in lower),
        "sem_has_repository_context": int(
            "[repository]" in lower
            or "commit_base:" in lower
            or "repo:" in lower
        ),
    }


# ---------------------------------------------------------------------
# Recuperação do prompt
# ---------------------------------------------------------------------

def find_prompt_file(
    runs_root: Path,
    model: str,
    case: str,
) -> Path | None:
    candidates = MODEL_DIR_CANDIDATES.get(model, [model])

    for run_name in candidates:
        p = runs_root / run_name / "patches" / f"{case}.prompt.txt"
        if p.is_file():
            return p

    # fallback controlado por glob somente em runs_root.
    for p in runs_root.glob(f"*/patches/{case}.prompt.txt"):
        if p.is_file():
            return p

    return None


def text_from_case_json(cases_dir: Path, case: str) -> str:
    path = cases_dir / f"{case}.json"
    if not path.is_file():
        return ""

    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    fields = [
        obj.get("issue_title") or "",
        obj.get("issue_body") or obj.get("problem_statement") or "",
        obj.get("short_file_list") or "",
        obj.get("test_query") or "",
    ]

    return "\n".join(str(x) for x in fields if x)


def load_prompt_text(
    runs_root: Path,
    cases_dir: Path,
    model: str,
    case: str,
) -> tuple[str, str]:
    prompt_path = find_prompt_file(runs_root, model, case)
    if prompt_path:
        return (
            prompt_path.read_text(encoding="utf-8", errors="ignore"),
            str(prompt_path),
        )

    fallback = text_from_case_json(cases_dir, case)
    if fallback:
        return fallback, f"case_json:{case}"

    return "", ""


# ---------------------------------------------------------------------
# Features derivadas antigas do artigo
# ---------------------------------------------------------------------

def safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.divide(
        a, b,
        out=np.zeros_like(a, dtype=float),
        where=b != 0,
    )


def add_article_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    base_numeric = [
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

    for col in base_numeric:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["patch_density"] = safe_div(
        df["patch_churn"], np.maximum(df["patch_lines"], 1)
    )
    df["add_remove_ratio"] = safe_div(
        df["patch_added"], np.maximum(df["patch_removed"], 1)
    )
    df["net_per_line"] = safe_div(
        df["patch_net"], np.maximum(df["patch_lines"], 1)
    )
    df["hunks_per_file"] = safe_div(
        df["patch_hunks"], np.maximum(df["patch_files_touched"], 1)
    )
    df["prompt_density"] = safe_div(
        df["prompt_chars"], np.maximum(df["prompt_lines"], 1)
    )
    df["prompt_token_density"] = safe_div(
        df["prompt_chars"], np.maximum(df["prompt_tokens"], 1)
    )

    q1, q2 = df["prompt_chars"].quantile([0.33, 0.66]).tolist()
    df["prompt_size_category"] = np.select(
        [df["prompt_chars"] <= q1, df["prompt_chars"] <= q2],
        [0, 1],
        default=2,
    )

    df["patch_complexity"] = (
        df["patch_hunks"] * df["patch_files_touched"]
    )
    df["change_intensity"] = safe_div(
        df["patch_churn"], np.maximum(df["patch_files_touched"], 1)
    )

    return df


# ---------------------------------------------------------------------
# ML
# ---------------------------------------------------------------------

def make_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
):
    transformers = []

    if numeric_features:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", MinMaxScaler()),
            ]),
            numeric_features,
        ))

    if categorical_features:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ))

    return ColumnTransformer(transformers)


def make_models(
    numeric_features: list[str],
    categorical_features: list[str],
    seed: int,
):
    return {
        "logistic_regression": Pipeline([
            ("prep", make_preprocessor(
                numeric_features,
                categorical_features,
            )),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=seed,
            )),
        ]),
        "random_forest": Pipeline([
            ("prep", make_preprocessor(
                numeric_features,
                categorical_features,
            )),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )),
        ]),
    }


def calc_metrics(y_true, pred, prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision_1": precision_score(
            y_true, pred, zero_division=0
        ),
        "recall_1": recall_score(
            y_true, pred, zero_division=0
        ),
        "f1_1": f1_score(
            y_true, pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, prob),
        "pr_auc": average_precision_score(y_true, prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_feature_sets(
    df: pd.DataFrame,
    target_col: str,
    runs: int,
    base_seed: int,
) -> pd.DataFrame:

    feature_sets = {
        "Model only": ([], ["model"]),

        "Prompt size only": (
            PROMPT_SIZE_FEATURES,
            [],
        ),

        "Prompt semantic only": (
            PROMPT_SEMANTIC_FEATURES,
            [],
        ),

        "Patch only": (
            PATCH_FEATURES,
            [],
        ),

        "Patch + Model": (
            PATCH_FEATURES,
            ["model"],
        ),

        "Patch + Prompt semantic": (
            PATCH_FEATURES + PROMPT_SEMANTIC_FEATURES,
            [],
        ),

        "Prompt size + Prompt semantic": (
            PROMPT_SIZE_FEATURES + PROMPT_SEMANTIC_FEATURES,
            [],
        ),

        "Prompt semantic + Patch + Model": (
            PATCH_FEATURES + PROMPT_SEMANTIC_FEATURES,
            ["model"],
        ),

        "All pre-SAST features": (
            PATCH_FEATURES
            + PROMPT_SIZE_FEATURES
            + PROMPT_SEMANTIC_FEATURES,
            ["model"],
        ),
    }

    y = df[target_col].astype(int).to_numpy()
    groups = df["case"].astype(str).to_numpy()

    rows = []

    for run_index in range(runs):
        seed = base_seed + run_index

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=seed,
        )

        train_idx, test_idx = next(
            splitter.split(df, y, groups=groups)
        )

        if (
            len(np.unique(y[train_idx])) < 2
            or len(np.unique(y[test_idx])) < 2
        ):
            continue

        for feature_set, (nums, cats) in feature_sets.items():
            cols = nums + cats

            for classifier_name, model in make_models(
                nums, cats, seed
            ).items():

                model.fit(
                    df.iloc[train_idx][cols],
                    y[train_idx],
                )

                prob = model.predict_proba(
                    df.iloc[test_idx][cols]
                )[:, 1]
                pred = (prob >= 0.5).astype(int)

                m = calc_metrics(
                    y[test_idx],
                    pred,
                    prob,
                )

                m.update({
                    "run": run_index + 1,
                    "seed": seed,
                    "classifier": classifier_name,
                    "feature_set": feature_set,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "train_cases": len(set(groups[train_idx])),
                    "test_cases": len(set(groups[test_idx])),
                })

                rows.append(m)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------

def plot_ablation(summary: pd.DataFrame, out_path: Path) -> None:
    rf = summary[
        summary["classifier"].eq("random_forest")
    ].copy()

    rf = rf.sort_values("roc_auc_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        rf["feature_set"],
        rf["roc_auc_mean"],
        xerr=rf["roc_auc_std"],
        capsize=3,
    )
    ax.axvline(0.5, linestyle="--", linewidth=1)
    ax.set_xlabel("ROC-AUC médio ± DP")
    ax.set_title(
        "Ablação de atributos pré-SAST — Random Forest"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_semantic_gain(
    metrics: pd.DataFrame,
    out_path: Path,
) -> None:
    rf = metrics[
        metrics["classifier"].eq("random_forest")
    ]

    pivot = rf.pivot(
        index="run",
        columns="feature_set",
        values="roc_auc",
    )

    baseline = "Patch + Model"
    enriched = "Prompt semantic + Patch + Model"

    if baseline not in pivot or enriched not in pivot:
        return

    delta = pivot[enriched] - pivot[baseline]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(delta.index.astype(str), delta.values)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Execução")
    ax.set_ylabel("Δ ROC-AUC")
    ax.set_title(
        "Ganho ao adicionar atributos semânticos do prompt"
    )
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=True,
        help="CSV comum case x model.",
    )

    ap.add_argument(
        "--runs-root",
        default="runs_backup",
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
        "--out-dir",
        default="semantic_prompt_results",
    )

    args = ap.parse_args()

    if args.runs < 1:
        raise SystemExit("--runs deve ser >= 1")

    input_path = Path(args.input)
    runs_root = Path(args.runs_root)
    cases_dir = Path(args.cases_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    required = {"case", "model", "findings_after"}
    missing = required - set(df.columns)

    if missing:
        raise SystemExit(
            f"CSV sem colunas obrigatórias: {sorted(missing)}"
        )

    df = add_article_features(df)

    semantic_rows = []
    missing_prompts = 0

    for _, row in df.iterrows():
        case = str(row["case"])
        model = str(row["model"])

        prompt_text, source = load_prompt_text(
            runs_root,
            cases_dir,
            model,
            case,
        )

        if not prompt_text:
            missing_prompts += 1

        sem = semantic_prompt_features(prompt_text)

        semantic_rows.append({
            "case": case,
            "model": model,
            "prompt_semantic_source": source,
            "prompt_semantic_text_chars": len(prompt_text),
            **sem,
        })

    semantic_df = pd.DataFrame(semantic_rows)

    # Evita colisão case/model em datasets duplicados.
    semantic_df = semantic_df.drop_duplicates(
        ["case", "model"],
        keep="first",
    )

    # Remove eventuais colunas semânticas pré-existentes antes do merge.
    drop_existing = [
        c for c in semantic_df.columns
        if c in df.columns and c not in {"case", "model"}
    ]
    if drop_existing:
        df = df.drop(columns=drop_existing)

    enriched = df.merge(
        semantic_df,
        on=["case", "model"],
        how="left",
        validate="many_to_one",
    )

    for col in PROMPT_SEMANTIC_FEATURES:
        enriched[col] = pd.to_numeric(
            enriched[col],
            errors="coerce",
        ).fillna(0)

    enriched["has_finding_after"] = (
        enriched["findings_after"] > 0
    ).astype(int)

    enriched.to_csv(
        out_dir / "case_model_with_prompt_semantic_features.csv",
        index=False,
    )

    semantic_df.to_csv(
        out_dir / "prompt_semantic_features.csv",
        index=False,
    )

    metrics = evaluate_feature_sets(
        enriched,
        target_col="has_finding_after",
        runs=args.runs,
        base_seed=args.seed,
    )

    metrics.to_csv(
        out_dir / "ablation_all_runs.csv",
        index=False,
    )

    summary = (
        metrics
        .groupby(["classifier", "feature_set"])
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
    summary = summary.reset_index()

    summary.to_csv(
        out_dir / "ablation_summary.csv",
        index=False,
    )

    # Comparação pareada mais importante para a hipótese.
    rf = metrics[
        metrics["classifier"].eq("random_forest")
    ]

    pivot = rf.pivot(
        index="run",
        columns="feature_set",
        values="roc_auc",
    )

    paired = pd.DataFrame(index=pivot.index)

    if (
        "Patch + Model" in pivot
        and "Prompt semantic + Patch + Model" in pivot
    ):
        paired["auc_patch_model"] = pivot["Patch + Model"]
        paired["auc_patch_semantic_model"] = (
            pivot["Prompt semantic + Patch + Model"]
        )
        paired["delta_auc_semantic_prompt"] = (
            paired["auc_patch_semantic_model"]
            - paired["auc_patch_model"]
        )

    if (
        "Patch only" in pivot
        and "Patch + Prompt semantic" in pivot
    ):
        paired["auc_patch_only"] = pivot["Patch only"]
        paired["auc_patch_plus_semantic"] = (
            pivot["Patch + Prompt semantic"]
        )
        paired["delta_auc_semantic_over_patch"] = (
            paired["auc_patch_plus_semantic"]
            - paired["auc_patch_only"]
        )

    paired.reset_index().to_csv(
        out_dir / "paired_prompt_semantic_gain.csv",
        index=False,
    )

    plot_ablation(
        summary,
        out_dir / "fig_ablation_semantic_prompt.png",
    )

    plot_semantic_gain(
        metrics,
        out_dir / "fig_semantic_prompt_delta_auc.png",
    )

    report = {
        "rows": int(len(enriched)),
        "unique_cases": int(enriched["case"].nunique()),
        "models": sorted(
            enriched["model"].astype(str).unique().tolist()
        ),
        "positive_has_finding_after": int(
            enriched["has_finding_after"].sum()
        ),
        "negative_has_finding_after": int(
            len(enriched)
            - enriched["has_finding_after"].sum()
        ),
        "missing_prompt_text_rows": int(missing_prompts),
        "runs": args.runs,
        "base_seed": args.seed,
        "split": "80/20 GroupShuffleSplit by case",
        "semantic_features": PROMPT_SEMANTIC_FEATURES,
        "post_sast_features_used": False,
        "primary_comparison": (
            "Prompt semantic + Patch + Model "
            "vs Patch + Model"
        ),
    }

    (out_dir / "run_summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("SEMANTIC PROMPT EXPERIMENT")
    print("=" * 80)
    print(f"Rows               : {len(enriched)}")
    print(f"Cases              : {enriched['case'].nunique()}")
    print(f"Prompts ausentes   : {missing_prompts}")
    print(
        f"Target positivos   : "
        f"{int(enriched['has_finding_after'].sum())}"
    )
    print()

    cols = [
        "classifier",
        "feature_set",
        "roc_auc_mean",
        "roc_auc_std",
        "recall_1_mean",
        "f1_1_mean",
    ]

    print(summary[cols].to_string(index=False))

    if (
        "delta_auc_semantic_prompt" in paired.columns
        and len(paired)
    ):
        delta = paired["delta_auc_semantic_prompt"]
        print()
        print("Prompt semantic adicionado a Patch + Model")
        print(f"ΔAUC médio   : {delta.mean():.4f}")
        print(f"ΔAUC mediano : {delta.median():.4f}")
        print(
            f"Melhorou em  : "
            f"{int((delta > 0).sum())}/{len(delta)} execuções"
        )

    print()
    print(f"Saída: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
