#!/usr/bin/env python3
"""
Benchmark Bandit v2 — repositórios reconstruídos.

Diferenças da versão anterior:
1. por padrão NÃO usa `-x tests`;
2. busca `--n-per-model` REPOSITÓRIOS RECONSTRUÍDOS COM SUCESSO por modelo,
   tentando casos adicionais quando há falha;
3. registra versão/configuração do Bandit;
4. resume falhas por estágio e por modelo;
5. mantém clone/checkout/apply fora do cronômetro; cronometra só Bandit.

IMPORTANTE: antes do benchmark final, confirme que os argumentos do Bandit
reproduzem a execução que gerou `has_finding_after`.

Uso:
  python benchmark_reconstructed_bandit_v2.py \
    --features case_model_problem_statement_features.csv \
    --runs-root /Users/mac/Downloads/llm_risk_pipeline_replicate/runs_backup \
    --n-per-model 10 \
    --repeats 5 \
    --outdir runtime_bandit_final_v2

Se o pipeline original excluía diretórios específicos:
  --exclude 'caminho1,caminho2'
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Reutiliza funções já testadas da versão anterior.
from benchmark_reconstructed_bandit import (
    find_metadata_files,
    load_json,
    first_value_for_keys,
    REPO_KEYS,
    COMMIT_KEYS,
    extract_patch_from_metadata,
    find_patch_file,
    normalize_repo_url,
    ensure_cache,
    ensure_commit,
    reconstruct_repo,
    bandit_once,
    summarize_ms,
)


def bandit_version():
    try:
        p = subprocess.run(
            ["bandit", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return (p.stdout or p.stderr).strip()
    except Exception as e:
        return f"UNKNOWN ({type(e).__name__})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--outdir", default="runtime_bandit_reconstructed_v2")
    ap.add_argument("--cache-root", default="~/.cache/llm-risk-runtime-repos")
    ap.add_argument(
        "--n-per-model",
        type=int,
        default=10,
        help="Número ALVO de repos reconstruídos com sucesso por modelo.",
    )
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--exclude",
        default="",
        help="Valor passado a `bandit -x`. Vazio = nenhum exclude.",
    )
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    if shutil.which("git") is None:
        raise SystemExit("git não encontrado no PATH.")
    if shutil.which("bandit") is None:
        raise SystemExit("bandit não encontrado no PATH.")

    features = pd.read_csv(args.features)
    required = {"case", "model", "backup_dir"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")

    rows = (
        features[["case", "model", "backup_dir"]]
        .drop_duplicates()
        .copy()
    )

    # Candidatos em ordem aleatória reprodutível.
    candidate_pools = {}
    for i, (model, sub) in enumerate(rows.groupby("model")):
        candidate_pools[str(model)] = (
            sub.sample(frac=1.0, random_state=args.seed + i)
            .reset_index(drop=True)
        )

    runs_root = Path(args.runs_root).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config = {
        "bandit_version": bandit_version(),
        "bandit_scan": "recursive_repo",
        "bandit_exclude": args.exclude,
        "target_successful_repos_per_model": args.n_per_model,
        "repeats_per_repo": args.repeats,
        "warmup_per_repo": 1,
        "warmup_included_in_stats": False,
        "clone_checkout_patch_apply_included_in_bandit_time": False,
        "seed": args.seed,
    }
    (outdir / "benchmark_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest_rows = []
    failures = []
    runtime_rows = []
    success_by_model = {m: 0 for m in candidate_pools}
    attempts_by_model = {m: 0 for m in candidate_pools}

    with tempfile.TemporaryDirectory(prefix="runtime_reconstruct_v2_") as td:
        td = Path(td)
        global_idx = 0

        for model, pool in candidate_pools.items():
            for _, r in pool.iterrows():
                if success_by_model[model] >= args.n_per_model:
                    break

                global_idx += 1
                attempts_by_model[model] += 1
                case = str(r["case"])
                backup = runs_root / str(r["backup_dir"])

                meta_files = find_metadata_files(backup, case)
                meta_path = None
                repo_value = ""
                base_commit = ""
                patch_text = ""

                for mp in meta_files:
                    obj = load_json(mp)
                    if obj is None:
                        continue
                    rv = first_value_for_keys(obj, REPO_KEYS)
                    bc = first_value_for_keys(obj, COMMIT_KEYS)
                    if not patch_text:
                        patch_text = extract_patch_from_metadata(obj)
                    if rv or bc:
                        meta_path = mp
                        repo_value = rv
                        base_commit = bc
                        if rv and bc:
                            break

                patch_path = find_patch_file(backup, case)
                if patch_path is not None:
                    try:
                        patch_text = patch_path.read_text(
                            encoding="utf-8", errors="replace"
                        )
                    except Exception:
                        pass

                repo_url = normalize_repo_url(repo_value)

                base_rec = {
                    "case": case,
                    "model": model,
                    "backup_dir": str(r["backup_dir"]),
                    "metadata_file": str(meta_path or ""),
                    "patch_file": str(patch_path or ""),
                    "repo_value": repo_value,
                    "repo_url": repo_url,
                    "base_commit": base_commit,
                }

                if not repo_url or not base_commit or not patch_text:
                    reason = []
                    if not repo_url:
                        reason.append("MISSING_REPO")
                    if not base_commit:
                        reason.append("MISSING_BASE_COMMIT")
                    if not patch_text:
                        reason.append("MISSING_PATCH")
                    failures.append({
                        **base_rec,
                        "stage": "DISCOVERY",
                        "error": "|".join(reason),
                    })
                    continue

                cache, ok, err = ensure_cache(repo_url, cache_root)
                if not ok:
                    failures.append({
                        **base_rec, "stage": "CACHE", "error": err
                    })
                    continue

                ok, err = ensure_commit(cache, base_commit)
                if not ok:
                    failures.append({
                        **base_rec, "stage": "COMMIT", "error": err
                    })
                    continue

                dest = td / f"repo_{global_idx}"
                ok, apply_status, err = reconstruct_repo(
                    cache, base_commit, patch_text, dest
                )
                if not ok:
                    failures.append({
                        **base_rec,
                        "stage": apply_status,
                        "error": err,
                    })
                    continue

                success_by_model[model] += 1
                manifest_rows.append({
                    **base_rec,
                    "apply_status": apply_status,
                    "temp_repo": str(dest),
                    "success_index_within_model": success_by_model[model],
                })

                # Warm-up descartado.
                warm_out = td / f"warm_{global_idx}.json"
                warm_ms, warm_rc, warm_ok = bandit_once(
                    dest, warm_out, args.exclude, args.timeout
                )
                warm_out.unlink(missing_ok=True)

                print(
                    f"{model}: sucesso {success_by_model[model]}/{args.n_per_model} "
                    f"| tentativa {attempts_by_model[model]} "
                    f"| {case} | warm-up {warm_ms:.1f} ms"
                )

                for rep in range(1, args.repeats + 1):
                    out_json = td / f"bandit_{global_idx}_{rep}.json"
                    ms, rc, ok = bandit_once(
                        dest, out_json, args.exclude, args.timeout
                    )

                    finding_count = np.nan
                    if ok and out_json.exists():
                        try:
                            obj = json.loads(
                                out_json.read_text(
                                    encoding="utf-8",
                                    errors="replace",
                                )
                            )
                            finding_count = len(obj.get("results", []))
                        except Exception:
                            pass

                    runtime_rows.append({
                        "case": case,
                        "model": model,
                        "repeat": rep,
                        "bandit_ms": ms,
                        "bandit_returncode": rc,
                        "bandit_ok": ok,
                        "finding_count": finding_count,
                        "apply_status": apply_status,
                    })
                    out_json.unlink(missing_ok=True)

    manifest = pd.DataFrame(manifest_rows)
    fail_df = pd.DataFrame(failures)
    raw = pd.DataFrame(runtime_rows)

    manifest.to_csv(outdir / "reconstruction_manifest.csv", index=False)
    fail_df.to_csv(outdir / "reconstruction_failures.csv", index=False)
    raw.to_csv(outdir / "bandit_runtime_raw.csv", index=False)

    # Falhas resumidas.
    if not fail_df.empty:
        failure_summary = (
            fail_df.groupby(["model", "stage"], dropna=False)
            .size()
            .reset_index(name="n_failures")
        )
    else:
        failure_summary = pd.DataFrame(
            columns=["model", "stage", "n_failures"]
        )
    failure_summary.to_csv(
        outdir / "reconstruction_failure_summary.csv",
        index=False,
    )

    success_summary = pd.DataFrame([
        {
            "model": m,
            "attempts": attempts_by_model[m],
            "successful_repos": success_by_model[m],
            "target": args.n_per_model,
            "success_rate": (
                success_by_model[m] / attempts_by_model[m]
                if attempts_by_model[m] else np.nan
            ),
        }
        for m in candidate_pools
    ])
    success_summary.to_csv(
        outdir / "reconstruction_success_by_model.csv",
        index=False,
    )

    if raw.empty:
        print("\nNenhum benchmark concluído.")
        print(failure_summary.to_string(index=False))
        return

    ok = raw[raw["bandit_ok"]].copy()
    per_repo = (
        ok.groupby(["case", "model"], as_index=False)
        .agg(
            bandit_median_ms=("bandit_ms", "median"),
            bandit_mean_ms=("bandit_ms", "mean"),
            repeats=("bandit_ms", "size"),
            findings=("finding_count", "median"),
        )
    )
    per_repo.to_csv(outdir / "bandit_runtime_per_repo.csv", index=False)

    summary_rows = []
    scopes = [("OVERALL", per_repo)] + [
        (f"MODEL:{m}", g) for m, g in per_repo.groupby("model")
    ]
    for scope, sub in scopes:
        rec = {"scope": scope, "n_repos": len(sub)}
        rec.update(summarize_ms(sub["bandit_median_ms"]))
        summary_rows.append(rec)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "bandit_runtime_summary.csv", index=False)

    print("\n" + "=" * 100)
    print("CONFIGURAÇÃO")
    print("=" * 100)
    print(json.dumps(config, indent=2, ensure_ascii=False))

    print("\n" + "=" * 100)
    print("SUCESSO DE RECONSTRUÇÃO")
    print("=" * 100)
    print(success_summary.to_string(index=False))

    print("\n" + "=" * 100)
    print("FALHAS POR ESTÁGIO")
    print("=" * 100)
    print(
        failure_summary.to_string(index=False)
        if not failure_summary.empty
        else "Nenhuma"
    )

    print("\n" + "=" * 100)
    print("BANDIT — REPOS RECONSTRUÍDOS")
    print("=" * 100)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
