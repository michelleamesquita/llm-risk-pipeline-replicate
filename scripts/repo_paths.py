#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Caminhos dos clones SWE-bench.

Convenção única, usada por fetch, orquestrador e apply:

- Slug GitHub (URL): owner/repo          ex. django/django
- Pasta local:       repos/<basename>    ex. data/swe-bench/repos/django

Nunca passe case['repo_name'] cru para Path: se for owner/repo vira
data/swe-bench/repos/django/django, que não existe.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

GITHUB_SLUG_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
LOCAL_DIR_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
DEFAULT_REPOS_ROOT = Path("data") / "swe-bench" / "repos"


def _clean(value: object) -> str:
    return str(value or "").strip().rstrip("/")


def github_slug(case: dict) -> str:
    for key in ("repo_full_name", "repo", "repo_name"):
        value = _clean(case.get(key))
        if "/" in value and GITHUB_SLUG_RE.fullmatch(value):
            return value
    return ""


def local_dir_name(case: dict, slug: Optional[str] = None) -> str:
    """Nome da pasta local: sempre basename, sem barra."""
    short = _clean(case.get("repo_short_name"))
    if short and "/" not in short:
        return short

    slug = slug or github_slug(case)
    if slug:
        return slug.rsplit("/", 1)[-1]

    name = _clean(case.get("repo_name"))
    return name.split("/")[-1] if name else ""


def is_git_repo(path: Path) -> bool:
    git = path / ".git"
    return git.is_dir() or git.is_file()


def candidate_local_dirs(case: dict, repos_root: Path) -> list[Path]:
    """Ordem: canônico (basename), depois layout legado owner/repo."""
    slug = github_slug(case)
    short = local_dir_name(case, slug)
    names: list[str] = []
    if short:
        names.append(short)
    if slug:
        names.append(slug)

    seen = set()
    out: list[Path] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out.append(repos_root / name)
    return out


def find_local_repo(case: dict, repos_root: Path) -> Optional[Path]:
    for path in candidate_local_dirs(case, repos_root):
        if is_git_repo(path):
            return path
    return None


def local_clone_dest(case: dict, repos_root: Path) -> Path:
    """Destino canônico do clone. Reusa clone legado se já existir."""
    existing = find_local_repo(case, repos_root)
    if existing is not None:
        return existing
    name = local_dir_name(case)
    if not LOCAL_DIR_RE.fullmatch(name):
        raise ValueError(f"nome local inválido: {name!r}")
    return repos_root / name


def strip_duplicated_path_prefix(line: str, repo_dirs: Iterable[str]) -> str:
    """LLM às vezes gera a/django/django/file.py em vez de a/django/file.py."""
    for repo_dir in repo_dirs:
        if not repo_dir or "/" in repo_dir:
            continue
        dup = f"{repo_dir}/{repo_dir}/"
        for prefix in ("a/", "b/", ""):
            line = line.replace(f"{prefix}{dup}", f"{prefix}{repo_dir}/")
    return line
