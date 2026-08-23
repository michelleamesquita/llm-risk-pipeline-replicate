#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Clona o repositório GitHub do caso SWE-bench no commit base."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from repo_paths import (  # noqa: E402
    GITHUB_SLUG_RE,
    find_local_repo,
    github_slug,
    is_git_repo,
    local_clone_dest,
)


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)


def has_commit(repo: Path, commit: str) -> bool:
    if not commit:
        return False
    result = run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=str(repo),
    )
    return result.returncode == 0


def normalized_remote(value: str) -> str:
    return str(value or "").strip().removesuffix(".git").rstrip("/")


def ensure_clone(case: dict, repos_root: Path) -> Path:
    slug = github_slug(case)
    if not GITHUB_SLUG_RE.fullmatch(slug):
        raise ValueError(f"esperado owner/repo, obtido {slug!r}")

    dest = local_clone_dest(case, repos_root)
    existing = find_local_repo(case, repos_root)
    if existing is not None:
        dest = existing

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not is_git_repo(dest):
        shutil.rmtree(dest)

    url = f"https://github.com/{slug}.git"
    if not dest.exists():
        result = run(["git", "clone", "--filter=blob:none", url, str(dest)])
        if result.returncode != 0:
            raise RuntimeError(f"CLONE_FAIL {result.stderr}")
    else:
        remote = run(["git", "remote", "get-url", "origin"], cwd=str(dest))
        if remote.returncode != 0:
            raise RuntimeError(f"REMOTE_FAIL {remote.stderr}")
        expected = normalized_remote(url)
        actual = normalized_remote(remote.stdout)
        if actual != expected:
            raise RuntimeError(
                f"REMOTE_MISMATCH {dest}: esperado {expected}, obtido {actual}"
            )
    return dest


def ensure_repo_cases(cases: list[dict], repos_root: Path) -> int:
    if not cases:
        return 0

    try:
        dest = ensure_clone(cases[0], repos_root)
    except (ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 2

    missing = [
        str(case.get("base_commit") or "")
        for case in cases
        if not has_commit(dest, str(case.get("base_commit") or ""))
    ]
    if missing:
        result = run(["git", "fetch", "--all", "--tags"], cwd=str(dest))
        if result.returncode != 0:
            print(f"FETCH_FAIL {result.stderr}", file=sys.stderr)
            return 3

    still_missing = [commit for commit in missing if not has_commit(dest, commit)]
    for commit in list(still_missing):
        result = run(["git", "fetch", "origin", commit], cwd=str(dest))
        if result.returncode == 0 and has_commit(dest, commit):
            still_missing.remove(commit)

    if still_missing:
        print(
            f"COMMIT_MISSING {dest}: {', '.join(still_missing[:5])}",
            file=sys.stderr,
        )
        return 3

    print(
        f"Repo pronto: {dest} "
        f"({len(cases)} caso(s), {len(missing)} commit(s) buscado(s))"
    )
    return 0


def load_cases(
    case_json: str | None,
    cases_dir: str | None,
    max_cases: int,
) -> list[dict]:
    if case_json:
        return [json.loads(Path(case_json).read_text(encoding="utf-8"))]
    paths = sorted(Path(cases_dir or "").glob("*.json"))
    if max_cases:
        paths = paths[:max_cases]
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in paths
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--case_json")
    source.add_argument("--cases_dir")
    ap.add_argument("--out_root", default="data/swe-bench/repos")
    ap.add_argument("--max_cases", type=int, default=0)
    args = ap.parse_args()

    repos_root = Path(args.out_root)
    cases = load_cases(args.case_json, args.cases_dir, args.max_cases)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        slug = github_slug(case)
        if not slug:
            print(f"INVALID_REPO no caso {case.get('case_id')}", file=sys.stderr)
            return 2
        grouped[slug].append(case)

    for slug in sorted(grouped):
        rc = ensure_repo_cases(grouped[slug], repos_root)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
