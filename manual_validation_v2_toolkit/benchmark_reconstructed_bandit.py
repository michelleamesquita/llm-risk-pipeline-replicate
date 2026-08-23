#!/usr/bin/env python3
"""
Reconstrói repos temporários a partir de metadata/ + patches/ e mede Bandit.

Princípio metodológico:
- clone/fetch/checkout/apply patch = PREPARAÇÃO, NÃO entra no tempo do SAST;
- cronometra somente o comando Bandit, com repo já reconstruído;
- warm-up por repo é descartado;
- mede vários repos por modelo e várias repetições por repo.

Esperado em runs_root:
  <backup>/
    metadata/ ou generation_metadata/   JSONs
    patches/                            diffs
    reports/
    repos_patched/                      pode estar vazio

Uso:
  python benchmark_reconstructed_bandit.py \
    --features case_model_problem_statement_features.csv \
    --runs-root /Users/mac/Downloads/llm_risk_pipeline_replicate/runs_backup \
    --n-per-model 10 \
    --repeats 5 \
    --outdir runtime_bandit_reconstructed

Opcional:
  --cache-root ~/.cache/llm-risk-repos

Saídas:
  reconstruction_manifest.csv
  reconstruction_failures.csv
  bandit_runtime_raw.csv
  bandit_runtime_per_repo.csv
  bandit_runtime_summary.csv
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


CASE_COL = "case"
MODEL_COL = "model"

# Chaves comuns em metadata SWE-bench / pipelines derivados.
CASE_KEYS = [
    "instance_id", "case", "case_id", "task_id", "id"
]
REPO_KEYS = [
    "repo", "repository", "repo_name", "repository_name",
    "repo_url", "repository_url", "git_url", "clone_url"
]
COMMIT_KEYS = [
    "base_commit", "base_sha", "commit", "commit_sha",
    "base_commit_sha", "repo_commit"
]
PATCH_KEYS = [
    "patch", "model_patch", "generated_patch", "diff"
]


def walk_json(obj, path=""):
    """Yield (key, value, dotted_path) recursively."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            yield k, v, p
            yield from walk_json(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{path}[{i}]"
            yield from walk_json(v, p)


def first_value_for_keys(obj, keys):
    wanted = {k.lower() for k in keys}
    for k, v, _ in walk_json(obj):
        if str(k).lower() in wanted and isinstance(v, (str, int, float)):
            text = str(v).strip()
            if text:
                return text
    return ""


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def contains_case(path: Path, case: str):
    return case in path.name or case in str(path)


def find_metadata_files(backup: Path, case: str):
    roots = [backup / "metadata", backup / "generation_metadata"]
    out = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.json"):
            if contains_case(p, case):
                out.append(p)
    # Fallback: inspeciona JSONs e busca instance_id/case internamente.
    if not out:
        for root in roots:
            if not root.exists():
                continue
            for p in root.rglob("*.json"):
                obj = load_json(p)
                if obj is None:
                    continue
                v = first_value_for_keys(obj, CASE_KEYS)
                if v == case:
                    out.append(p)
    return out


def find_patch_file(backup: Path, case: str):
    root = backup / "patches"
    if not root.exists():
        return None

    # Prioridade para nomes contendo o case.
    candidates = [
        p for p in root.rglob("*")
        if p.is_file() and contains_case(p, case)
    ]
    if candidates:
        # Prefere .patch/.diff e depois menor profundidade.
        candidates.sort(
            key=lambda p: (
                0 if p.suffix.lower() in {".patch", ".diff"} else 1,
                len(p.parts),
                len(p.name),
            )
        )
        return candidates[0]

    # Fallback: procura case no conteúdo.
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if case in txt and ("diff --git " in txt or "@@ -" in txt):
            return p
    return None


def extract_patch_from_metadata(obj):
    text = first_value_for_keys(obj, PATCH_KEYS)
    if "diff --git " in text or "@@ -" in text:
        return text
    return ""


def normalize_repo_url(repo_value: str):
    repo_value = repo_value.strip()
    if not repo_value:
        return ""

    if repo_value.startswith(("http://", "https://", "git@", "ssh://")):
        return repo_value

    # SWE-bench normalmente usa owner/repo.
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_value):
        return f"https://github.com/{repo_value}.git"

    return repo_value


def repo_slug(repo_url: str):
    x = repo_url.rstrip("/").split("/")[-2:]
    slug = "__".join(x)
    slug = slug.removesuffix(".git")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", slug)
    return slug or "repo"


def run(cmd, cwd=None, timeout=600):
    return subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )


def ensure_cache(repo_url: str, cache_root: Path):
    cache_root.mkdir(parents=True, exist_ok=True)
    cache = cache_root / repo_slug(repo_url)

    if (cache / ".git").exists():
        p = run(["git", "fetch", "--all", "--tags", "--prune"], cwd=cache, timeout=900)
        return cache, p.returncode == 0, p.stderr[-2000:]

    p = run(["git", "clone", "--no-checkout", repo_url, str(cache)], timeout=1800)
    return cache, p.returncode == 0, p.stderr[-2000:]


def ensure_commit(cache: Path, commit: str):
    p = run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=cache)
    if p.returncode == 0:
        return True, ""
    p = run(["git", "fetch", "origin", commit], cwd=cache, timeout=900)
    if p.returncode != 0:
        return False, p.stderr[-2000:]
    p = run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=cache)
    return p.returncode == 0, p.stderr[-2000:]


def reconstruct_repo(cache: Path, commit: str, patch_text: str, dest: Path):
    # Clone local compartilhado: preparação fora do cronômetro Bandit.
    p = run(["git", "clone", "--shared", str(cache), str(dest)], timeout=900)
    if p.returncode != 0:
        return False, "CLONE_LOCAL_FAILED", p.stderr[-2000:]

    p = run(["git", "checkout", "--detach", commit], cwd=dest, timeout=300)
    if p.returncode != 0:
        return False, "CHECKOUT_FAILED", p.stderr[-2000:]

    patch_file = dest.parent / f"{dest.name}.patch"
    patch_file.write_text(patch_text, encoding="utf-8")

    p = run(["git", "apply", "--check", str(patch_file)], cwd=dest, timeout=120)
    if p.returncode != 0:
        return False, "STRICT_APPLY_CHECK_FAILED", p.stderr[-2000:]

    p = run(["git", "apply", str(patch_file)], cwd=dest, timeout=120)
    if p.returncode != 0:
        return False, "STRICT_APPLY_FAILED", p.stderr[-2000:]

    return True, "STRICT_APPLY_OK", ""


def bandit_once(repo: Path, out_json: Path, exclude: str, timeout: int):
    cmd = ["bandit", "-r", str(repo)]
    if exclude:
        cmd += ["-x", exclude]
    cmd += ["-f", "json", "-o", str(out_json)]

    t0 = time.perf_counter_ns()
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
        ms = (time.perf_counter_ns() - t0) / 1_000_000
        ok = p.returncode in (0, 1)
        return ms, p.returncode, ok
    except subprocess.TimeoutExpired:
        ms = (time.perf_counter_ns() - t0) / 1_000_000
        return ms, -999, False


def summarize_ms(values):
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    if len(x) == 0:
        return {}
    q1, q3 = np.quantile(x, [.25, .75])
    return {
        "n": len(x),
        "mean_ms": float(np.mean(x)),
        "median_ms": float(np.median(x)),
        "q1_ms": float(q1),
        "q3_ms": float(q3),
        "iqr_ms": float(q3 - q1),
        "p95_ms": float(np.quantile(x, .95)),
        "min_ms": float(np.min(x)),
        "max_ms": float(np.max(x)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--outdir", default="runtime_bandit_reconstructed")
    ap.add_argument("--cache-root", default="~/.cache/llm-risk-runtime-repos")
    ap.add_argument("--n-per-model", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude", default="tests")
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
        raise ValueError(f"Colunas ausentes no features CSV: {sorted(missing)}")

    rows = (
        features[["case", "model", "backup_dir"]]
        .drop_duplicates()
        .copy()
    )

    # Amostra estratificada por modelo.
    sampled = []
    for i, (model, sub) in enumerate(rows.groupby("model")):
        n = min(args.n_per_model, len(sub))
        sampled.append(sub.sample(n=n, random_state=args.seed + i))
    sampled = pd.concat(sampled, ignore_index=True)

    runs_root = Path(args.runs_root).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    failures = []
    runtime_rows = []

    with tempfile.TemporaryDirectory(prefix="runtime_reconstruct_") as td:
        td = Path(td)

        for i, r in sampled.iterrows():
            case = str(r["case"])
            model = str(r["model"])
            backup = runs_root / str(r["backup_dir"])

            meta_files = find_metadata_files(backup, case)
            metadata = None
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
                    metadata = obj
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
                failures.append({**base_rec, "stage": "DISCOVERY", "error": "|".join(reason)})
                print(f"[{i+1}/{len(sampled)}] {model} {case}: discovery falhou {reason}")
                continue

            cache, ok, err = ensure_cache(repo_url, cache_root)
            if not ok:
                failures.append({**base_rec, "stage": "CACHE", "error": err})
                print(f"[{i+1}/{len(sampled)}] {model} {case}: clone/fetch falhou")
                continue

            ok, err = ensure_commit(cache, base_commit)
            if not ok:
                failures.append({**base_rec, "stage": "COMMIT", "error": err})
                print(f"[{i+1}/{len(sampled)}] {model} {case}: commit não encontrado")
                continue

            dest = td / f"repo_{i}"
            ok, apply_status, err = reconstruct_repo(
                cache, base_commit, patch_text, dest
            )
            if not ok:
                failures.append({
                    **base_rec,
                    "stage": apply_status,
                    "error": err,
                })
                print(f"[{i+1}/{len(sampled)}] {model} {case}: {apply_status}")
                continue

            manifest_rows.append({
                **base_rec,
                "apply_status": apply_status,
                "temp_repo": str(dest),
            })

            # Warm-up descartado.
            warm_out = td / f"warm_{i}.json"
            warm_ms, warm_rc, warm_ok = bandit_once(
                dest, warm_out, args.exclude, args.timeout
            )
            try:
                warm_out.unlink(missing_ok=True)
            except Exception:
                pass

            print(
                f"[{i+1}/{len(sampled)}] {model} {case}: "
                f"reconstruído | warm-up {warm_ms:.1f} ms"
            )

            for rep in range(1, args.repeats + 1):
                out_json = td / f"bandit_{i}_{rep}.json"
                ms, rc, ok = bandit_once(
                    dest, out_json, args.exclude, args.timeout
                )

                finding_count = np.nan
                if ok and out_json.exists():
                    try:
                        obj = json.loads(
                            out_json.read_text(
                                encoding="utf-8", errors="replace"
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
                try:
                    out_json.unlink(missing_ok=True)
                except Exception:
                    pass

    manifest = pd.DataFrame(manifest_rows)
    fail_df = pd.DataFrame(failures)
    raw = pd.DataFrame(runtime_rows)

    manifest.to_csv(outdir / "reconstruction_manifest.csv", index=False)
    fail_df.to_csv(outdir / "reconstruction_failures.csv", index=False)
    raw.to_csv(outdir / "bandit_runtime_raw.csv", index=False)

    if raw.empty:
        print("\nNenhum benchmark concluído. Veja reconstruction_failures.csv")
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
        (f"MODEL:{m}", g)
        for m, g in per_repo.groupby("model")
    ]
    for scope, sub in scopes:
        rec = {"scope": scope, "n_repos": len(sub)}
        rec.update(summarize_ms(sub["bandit_median_ms"]))
        summary_rows.append(rec)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "bandit_runtime_summary.csv", index=False)

    print("\n" + "=" * 100)
    print("BANDIT — REPOS RECONSTRUÍDOS")
    print("=" * 100)
    print(summary.to_string(index=False))

    print("\nReconstruídos:", len(manifest))
    print("Falhas:", len(fail_df))
    print("Saídas:", outdir.resolve())


if __name__ == "__main__":
    main()
