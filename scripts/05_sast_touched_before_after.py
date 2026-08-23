#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_sast_touched_before_after.py

Executa Bandit APENAS nos arquivos tocados pelo patch:

BEFORE: conteúdo no base_commit
AFTER : working tree com patch aplicado

Produz dois JSONs no formato Bandit:
  <case>_bandit_before.json
  <case>_bandit_after.json

Para o BEFORE, cria uma worktree temporária no base_commit. Assim os caminhos,
imports relativos e estrutura do projeto permanecem próximos do ambiente real.
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)


def existing_files(root: Path, relpaths: list[str]) -> list[str]:
    out = []
    for rel in relpaths:
        p = root / rel
        if p.is_file() and p.suffix == ".py":
            out.append(str(p))
    return out


def run_bandit(files: list[str], out_json: Path, cwd: Path) -> int:
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # Sem arquivos Python: produz um relatório vazio válido.
    if not files:
        empty = {
            "errors": [],
            "generated_at": None,
            "metrics": {},
            "results": [],
        }
        out_json.write_text(json.dumps(empty, indent=2), encoding="utf-8")
        return 0

    cmd = ["bandit", "-f", "json", "-o", str(out_json)] + files
    r = run(cmd, cwd=str(cwd))
    # Bandit retorna 1 quando encontra issues.
    if r.returncode not in (0, 1):
        print(r.stderr)
        return r.returncode
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_src", required=True,
                    help="clone/base repo com o base_commit disponível")
    ap.add_argument("--patched_repo", required=True)
    ap.add_argument("--base_commit", required=True)
    ap.add_argument("--metadata_json", required=True)
    ap.add_argument("--before_json", required=True)
    ap.add_argument("--after_json", required=True)
    args = ap.parse_args()

    meta = json.loads(Path(args.metadata_json).read_text(encoding="utf-8"))
    if not meta.get("patch_apply_success"):
        raise SystemExit("Patch não aplicado com sucesso; SAST não deve ser executado.")

    touched = meta.get("actual_changed_files") or meta.get("touched_files") or []
    touched = [str(x).replace("\\", "/") for x in touched]

    repo_src = Path(args.repo_src).resolve()
    patched = Path(args.patched_repo).resolve()
    before_json = Path(args.before_json).resolve()
    after_json = Path(args.after_json).resolve()

    temp_parent = Path(tempfile.mkdtemp(prefix="swe_before_"))
    before_repo = temp_parent / "repo"

    try:
        # Worktree independente no commit base.
        r = run(
            ["git", "worktree", "add", "--detach", str(before_repo), args.base_commit],
            cwd=str(repo_src),
        )
        if r.returncode != 0:
            raise RuntimeError(f"git worktree add falhou: {r.stderr}")

        before_files = existing_files(before_repo, touched)
        after_files = existing_files(patched, touched)

        print(f"Arquivos Python tocados BEFORE: {len(before_files)}")
        print(f"Arquivos Python tocados AFTER : {len(after_files)}")

        rc1 = run_bandit(before_files, before_json, before_repo)
        rc2 = run_bandit(after_files, after_json, patched)

        meta["bandit_scope"] = "touched_python_files_only"
        meta["before_scanned_files"] = [
            os.path.relpath(x, before_repo).replace("\\", "/") for x in before_files
        ]
        meta["after_scanned_files"] = [
            os.path.relpath(x, patched).replace("\\", "/") for x in after_files
        ]
        Path(args.metadata_json).write_text(json.dumps(meta, indent=2), encoding="utf-8")

        if rc1 != 0 or rc2 != 0:
            return 5
        return 0
    finally:
        try:
            run(["git", "worktree", "remove", "--force", str(before_repo)], cwd=str(repo_src))
        except Exception:
            pass
        shutil.rmtree(temp_parent, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
