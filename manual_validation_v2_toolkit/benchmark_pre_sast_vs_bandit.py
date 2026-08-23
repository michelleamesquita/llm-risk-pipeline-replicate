#!/usr/bin/env python3
"""
Benchmark genérico de tempo: pré-SAST vs SAST.

Este script NÃO sabe como o seu extrator de features é implementado.
Você fornece comandos-template para:
  - feature extraction / pre-SAST preparation (opcional)
  - SAST/Bandit

O script mede wall-clock com time.perf_counter() e salva um CSV que pode ser
passado a evaluate_pre_sast_value_30runs.py via --runtime-input.

Manifesto CSV:
  precisa conter pelo menos:
    case
    model

  e quaisquer campos usados nos templates, por exemplo:
    repo_path
    prompt_path
    patch_path

Placeholders:
  {case}, {model}, {repo_path}, etc. são substituídos por valores da linha.

Exemplo:
  python benchmark_pre_sast_vs_bandit.py \
      --manifest runtime_manifest.csv \
      --feature-command 'python extract_features_one.py --case {case} --model {model}' \
      --bandit-command 'bandit -r {repo_path} -q' \
      --repeats 3 \
      --output runtime_measurements.csv

Segurança/reprodutibilidade:
  - shell=False;
  - o comando é tokenizado com shlex.split;
  - stdout/stderr são descartados por padrão;
  - registra exit code;
  - mede cada repetição separadamente e também produz resumo por case/model.

Para LR inference, prefira o tempo já medido por evaluate_pre_sast_value_30runs.py.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd


def format_command(template: str, row: pd.Series):
    values = {c: str(row[c]) for c in row.index}
    formatted = template.format(**values)
    return shlex.split(formatted)


def run_timed(cmd, timeout: int):
    t0 = time.perf_counter()
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
            shell=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return elapsed_ms, int(p.returncode), ""
    except subprocess.TimeoutExpired:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return elapsed_ms, -999, "TIMEOUT"
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return elapsed_ms, -998, type(e).__name__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--feature-command", default="")
    ap.add_argument("--bandit-command", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--output", default="runtime_measurements.csv")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0=todos; útil para smoke test.",
    )
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    for col in ["case", "model"]:
        if col not in manifest.columns:
            raise ValueError(f"Manifesto sem coluna obrigatória: {col}")

    if args.limit > 0:
        manifest = manifest.head(args.limit).copy()

    rows = []

    for idx, row in manifest.iterrows():
        for repeat in range(1, args.repeats + 1):
            rec = {
                "case": row["case"],
                "model": row["model"],
                "repeat": repeat,
            }

            if args.feature_command:
                feature_cmd = format_command(args.feature_command, row)
                ms, rc, err = run_timed(feature_cmd, args.timeout)
                rec["feature_extraction_ms"] = ms
                rec["feature_exit_code"] = rc
                rec["feature_error"] = err
            else:
                rec["feature_extraction_ms"] = np.nan
                rec["feature_exit_code"] = np.nan
                rec["feature_error"] = ""

            bandit_cmd = format_command(args.bandit_command, row)
            ms, rc, err = run_timed(bandit_cmd, args.timeout)
            rec["bandit_ms"] = ms
            rec["bandit_exit_code"] = rc
            rec["bandit_error"] = err

            rows.append(rec)
            print(
                f"{idx+1}/{len(manifest)} repeat={repeat} "
                f"case={row['case']} model={row['model']} "
                f"feature_ms={rec['feature_extraction_ms']} "
                f"bandit_ms={rec['bandit_ms']:.2f}"
            )

    raw = pd.DataFrame(rows)
    raw_path = Path(args.output)
    raw.to_csv(raw_path, index=False)

    agg_dict = {
        "bandit_ms": "median",
    }
    if raw["feature_extraction_ms"].notna().any():
        agg_dict["feature_extraction_ms"] = "median"

    summary = (
        raw.groupby(["case", "model"], as_index=False)
        .agg(agg_dict)
    )
    summary_path = raw_path.with_name(raw_path.stem + "_median_by_case_model.csv")
    summary.to_csv(summary_path, index=False)

    print("\nRaw:", raw_path.resolve())
    print("Median by case/model:", summary_path.resolve())


if __name__ == "__main__":
    main()
