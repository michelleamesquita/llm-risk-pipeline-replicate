#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_apply_patch_robust.py

Clona o commit base, extrai os arquivos realmente tocados pelo patch,
aplica SOMENTE o patch produzido pelo LLM e grava metadata JSON.

Diferença essencial para a versão antiga:
- se o patch falhar, NÃO insere comentário/placeholder artificial;
- o caso termina com código != 0 e deve ser excluído da análise principal.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import stat
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from diff_tools import repair_unified_diff, relocate_hunks
from repo_paths import strip_duplicated_path_prefix


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)


def rm_tree(path: Path) -> None:
    if not path.exists():
        return

    def handle_remove_error(func, target, _exc):
        os.chmod(target, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
        func(target)

    shutil.rmtree(path, onerror=handle_remove_error)


def clean_patch_content(content: str, repo_dir: str = "") -> str:
    """Normaliza diffs de LLM sem inventar hunks."""
    content = str(content or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    in_diff = False
    for raw in content.splitlines():
        # Linha de contexto vazia no unified diff é um único espaço.
        if raw == " ":
            line = " "
        else:
            line = raw.rstrip()

        if not line and not in_diff:
            continue
        if line.startswith("diff --git") or line.startswith("--- "):
            in_diff = True
        if not in_diff:
            continue

        # Hashes placeholder comuns em diffs gerados por LLM.
        if line.startswith("index ") and re.search(
            r"1234567|89abcde|deadbeef|0000000", line, re.I
        ):
            continue

        line = strip_duplicated_path_prefix(line, [repo_dir])
        lines.append(line)

    while lines and lines[-1] in {"", "+", "-"}:
        lines.pop()

    return "\n".join(lines) + "\n"


def normalize_relpath(p: str) -> str:
    p = (p or "").strip().replace("\\", "/")
    if p.startswith("a/") or p.startswith("b/"):
        p = p[2:]
    while p.startswith("./"):
        p = p[2:]
    return p.strip("/")


def extract_touched_files(patch_text: str) -> list[str]:
    files = []
    seen = set()

    # Prefer "diff --git a/x b/x"
    for m in re.finditer(r"(?m)^diff --git a/(.+?) b/(.+?)$", patch_text):
        path = normalize_relpath(m.group(2))
        if path and path != "/dev/null" and path not in seen:
            seen.add(path)
            files.append(path)

    # Fallback unified diff headers
    if not files:
        for line in patch_text.splitlines():
            if line.startswith("+++ "):
                value = line[4:].strip().split("\t", 1)[0]
                if value == "/dev/null":
                    continue
                path = normalize_relpath(value)
                if path and path not in seen:
                    seen.add(path)
                    files.append(path)

    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_src", required=True)
    ap.add_argument("--base_commit", required=True)
    ap.add_argument("--patch_file", required=True)
    ap.add_argument("--out_repo", required=True)
    ap.add_argument("--metadata_json", required=True)
    args = ap.parse_args()

    if not re.fullmatch(r"[0-9a-fA-F]{7,64}", args.base_commit):
        ap.error("--base_commit deve ser um hash Git hexadecimal")

    patch_path = Path(args.patch_file)
    out_repo = Path(args.out_repo)
    metadata_path = Path(args.metadata_json)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    repo_dir = Path(args.repo_src).name
    content = patch_path.read_text(encoding="utf-8", errors="ignore")
    cleaned = clean_patch_content(content, repo_dir)
    cleaned, normalization_report = repair_unified_diff(
        cleaned,
        [repo_dir],
    )
    touched_files = extract_touched_files(cleaned)

    metadata = {
        "repo_src": os.path.abspath(args.repo_src),
        "base_commit": args.base_commit,
        "patch_file": os.path.abspath(args.patch_file),
        "out_repo": os.path.abspath(args.out_repo),
        "touched_files": touched_files,
        "patch_files_touched": len(touched_files),
        "patch_apply_success": False,
        "apply_strategy": None,
        "error": None,
        "diff_repair": {
            "normalization": normalization_report,
        },
    }

    if not touched_files:
        metadata["error"] = "NO_TOUCHED_FILES_PARSED"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print("PATCH_INVALID: nenhum arquivo tocado foi identificado")
        return 4

    if out_repo.exists():
        rm_tree(out_repo)

    out_repo.parent.mkdir(parents=True, exist_ok=True)
    r = run(["git", "clone", "--shared", "--quiet", "--no-checkout", args.repo_src, str(out_repo)])
    if r.returncode != 0:
        metadata["error"] = "CLONE_FAIL"
        metadata["stderr"] = (r.stderr or "")[-4000:]
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return 2

    # Em clones fonte parciais (`--filter=blob:none`), `--shared` aponta o
    # remote do clone novo para o caminho local. Restaura o remote GitHub e as
    # flags promisor para que o checkout baixe blobs ausentes sob demanda.
    source_remote = run(
        ["git", "remote", "get-url", "origin"],
        cwd=str(args.repo_src),
    )
    if source_remote.returncode == 0 and source_remote.stdout.strip():
        run(
            ["git", "remote", "set-url", "origin", source_remote.stdout.strip()],
            cwd=str(out_repo),
        )
        source_promisor = run(
            ["git", "config", "--get", "remote.origin.promisor"],
            cwd=str(args.repo_src),
        )
        if source_promisor.stdout.strip().lower() == "true":
            run(
                ["git", "config", "remote.origin.promisor", "true"],
                cwd=str(out_repo),
            )
            source_filter = run(
                ["git", "config", "--get", "remote.origin.partialclonefilter"],
                cwd=str(args.repo_src),
            )
            if source_filter.stdout.strip():
                run(
                    [
                        "git", "config",
                        "remote.origin.partialclonefilter",
                        source_filter.stdout.strip(),
                    ],
                    cwd=str(out_repo),
                )

    r = run(["git", "checkout", "-f", args.base_commit], cwd=str(out_repo))
    if r.returncode != 0:
        metadata["error"] = "CHECKOUT_FAIL"
        metadata["stderr"] = (r.stderr or "")[-4000:]
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return 2

    cleaned, relocation_report = relocate_hunks(
        cleaned,
        out_repo,
        [repo_dir],
    )
    metadata["diff_repair"]["relocation"] = relocation_report

    fixed_patch = patch_path.with_suffix(patch_path.suffix + ".fixed")
    fixed_patch.write_text(cleaned, encoding="utf-8")

    patch_abs = str(fixed_patch.resolve())
    strategies = [
        ["git", "apply", "--ignore-whitespace", "--recount", "-p1"],
        ["git", "apply", "--ignore-whitespace", "--recount", "-p0"],
        ["git", "apply", "--ignore-whitespace", "-p1"],
        ["git", "apply", "--ignore-whitespace", "-p0"],
        ["git", "apply", "-p1"],
        ["git", "apply", "-p0"],
        # Mesma fallback do harness SWE-bench: contexto aproximado, sem placeholder.
        ["patch", "--batch", "--forward", "--fuzz=5", "-p1", "-i"],
        ["patch", "--batch", "--forward", "--fuzz=5", "-p0", "-i"],
    ]

    attempts = []
    try:
        for cmd in strategies:
            # `patch` pode aplicar apenas alguns hunks e retornar erro. Cada
            # estratégia deve partir exatamente do mesmo commit base.
            run(
                ["git", "reset", "--hard", args.base_commit],
                cwd=str(out_repo),
            )
            run(["git", "clean", "-fd"], cwd=str(out_repo))

            r = run(cmd + [patch_abs], cwd=str(out_repo))
            err = (r.stderr or r.stdout or "").strip()
            attempt = {
                "cmd": " ".join(cmd),
                "returncode": r.returncode,
                "stderr": err[-2000:],
            }
            attempts.append(attempt)
            if r.returncode == 0:
                diff = run(["git", "diff", "--name-only"], cwd=str(out_repo))
                actual = [
                    normalize_relpath(x)
                    for x in diff.stdout.splitlines()
                    if normalize_relpath(x)
                ]
                if not actual:
                    attempt["returncode"] = 4
                    attempt["stderr"] = "aplicação não alterou arquivos"
                    continue

                strategy = " ".join(cmd)
                metadata["patch_apply_success"] = True
                metadata["apply_strategy"] = strategy
                metadata["strict_patch_apply"] = strategy.startswith("git apply")
                metadata["fuzzy_patch_apply"] = strategy.startswith("patch ")
                metadata["error"] = None
                metadata["apply_attempts"] = attempts
                metadata["actual_changed_files"] = actual
                metadata["actual_files_count"] = len(actual)

                metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                print(f"PATCH_OK: {len(actual)} arquivo(s) efetivamente alterado(s) via {metadata['apply_strategy']}")
                return 0

        last_err = attempts[-1]["stderr"] if attempts else ""
        metadata["error"] = "PATCH_APPLY_FAIL"
        metadata["stderr"] = last_err
        metadata["apply_attempts"] = attempts
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print("PATCH_APPLY_FAIL: caso excluído; nenhuma alteração artificial foi aplicada")
        if last_err:
            print(last_err)
        return 3
    finally:
        try:
            fixed_patch.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
