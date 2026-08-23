#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
diff_tools.py

Reparo determinístico de diffs unificados produzidos por LLM.

Modelos menores emitem diffs sintaticamente inválidos mesmo quando a
intenção da edição está correta:

- contagens erradas no cabeçalho ``@@ -a,b +c,d @@``;
- hunks truncados no meio do corpo;
- linhas de contexto vazias sem o espaço inicial obrigatório;
- caminhos sem prefixo ``a/``/``b/`` e com timestamp de ``diff -u``;
- hunks cujo ``-`` e ``+`` são idênticos (não alteram nada);
- números de linha inventados.

Nada aqui inventa conteúdo: o módulo apenas descarta hunks inúteis,
recalcula contagens a partir do corpo realmente presente e reposiciona
hunks procurando o lado antigo no arquivo real do commit base. Toda
transformação é reportada para poder ser registrada no metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from repo_paths import strip_duplicated_path_prefix

HUNK_RE = re.compile(
    r"^@@+\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s*@@+(.*)$"
)
DIFF_GIT_RE = re.compile(r"^diff --git\s+(\S+)\s+(\S+)\s*$")
PLACEHOLDER_INDEX_RE = re.compile(r"1234567|89abcde|deadbeef|0000000", re.I)
TIMESTAMP_SUFFIX_RE = re.compile(
    r"\s+\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?"
    r"(?:\s*[+-]\d{4})?\s*$"
)
BODY_PREFIXES = (" ", "+", "-", "\\")
DEV_NULL = "/dev/null"


def _split_lines(text: str) -> list[str]:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")


def normalize_relpath(value: str) -> str:
    value = (value or "").strip().replace("\\", "/")
    if value == DEV_NULL:
        return value
    if value.startswith("a/") or value.startswith("b/"):
        value = value[2:]
    while value.startswith("./"):
        value = value[2:]
    return value.strip("/")


def _header_path(raw: str) -> str:
    """Extrai o caminho de uma linha ``---``/``+++``.

    ``diff -u`` anexa o mtime depois de um TAB, mas LLMs costumam usar
    espaços, então o timestamp também é removido por regex.
    """
    value = raw.split("\t", 1)[0].strip()
    return TIMESTAMP_SUFFIX_RE.sub("", value).strip()


def _is_body(line: str) -> bool:
    if line.startswith("--- ") or line.startswith("+++ "):
        return False
    return line[:1] in BODY_PREFIXES


def _empty_line_is_context(lines: list[str], index: int) -> bool:
    """Linha vazia dentro de um hunk é contexto se o hunk continua depois."""
    for candidate in lines[index + 1:]:
        if candidate == "":
            continue
        if candidate.startswith("diff --git") or HUNK_RE.match(candidate):
            return False
        return _is_body(candidate)
    return False


@dataclass
class Hunk:
    old_start: int
    new_start: int
    heading: str = ""
    body: list[str] = field(default_factory=list)
    declared_old_count: int | None = None
    declared_new_count: int | None = None
    relocated_from: int | None = None

    def old_lines(self) -> list[str]:
        return [ln[1:] for ln in self.body if ln[:1] in (" ", "-")]

    def old_count(self) -> int:
        return sum(1 for ln in self.body if ln[:1] in (" ", "-"))

    def new_count(self) -> int:
        return sum(1 for ln in self.body if ln[:1] in (" ", "+"))

    def is_useful(self) -> bool:
        minus = [ln[1:] for ln in self.body if ln[:1] == "-"]
        plus = [ln[1:] for ln in self.body if ln[:1] == "+"]
        if not minus and not plus:
            return False
        return minus != plus


@dataclass
class FileDiff:
    old_path: str = ""
    new_path: str = ""
    hunks: list[Hunk] = field(default_factory=list)

    @property
    def is_new(self) -> bool:
        return self.old_path == DEV_NULL

    @property
    def is_delete(self) -> bool:
        return self.new_path == DEV_NULL

    def target(self) -> str:
        return self.new_path if not self.is_delete else self.old_path


def parse_diff(text: str) -> list[FileDiff]:
    lines = _split_lines(text)
    files: list[FileDiff] = []
    current: FileDiff | None = None
    hunk: Hunk | None = None
    index = 0
    total = len(lines)

    while index < total:
        line = lines[index]

        git_header = DIFF_GIT_RE.match(line)
        if git_header:
            current = FileDiff(
                old_path=normalize_relpath(git_header.group(1)),
                new_path=normalize_relpath(git_header.group(2)),
            )
            files.append(current)
            hunk = None
            index += 1
            continue

        if (
            line.startswith("--- ")
            and index + 1 < total
            and lines[index + 1].startswith("+++ ")
        ):
            old_path = normalize_relpath(_header_path(line[4:]))
            new_path = normalize_relpath(_header_path(lines[index + 1][4:]))
            # Um `diff --git` imediatamente anterior já abriu a seção.
            if current is None or current.hunks:
                current = FileDiff()
                files.append(current)
            current.old_path = old_path
            current.new_path = new_path
            hunk = None
            index += 2
            continue

        hunk_header = HUNK_RE.match(line)
        if hunk_header and current is not None:
            hunk = Hunk(
                old_start=int(hunk_header.group(1)),
                new_start=int(hunk_header.group(3)),
                heading=hunk_header.group(5).rstrip(),
                declared_old_count=(
                    int(hunk_header.group(2))
                    if hunk_header.group(2) is not None
                    else 1
                ),
                declared_new_count=(
                    int(hunk_header.group(4))
                    if hunk_header.group(4) is not None
                    else 1
                ),
            )
            current.hunks.append(hunk)
            index += 1
            continue

        if hunk is not None:
            if line == "":
                if _empty_line_is_context(lines, index):
                    hunk.body.append(" ")
                else:
                    hunk = None
                index += 1
                continue
            if _is_body(line):
                hunk.body.append(line)
                index += 1
                continue
            hunk = None

        index += 1

    return files


def _resolve_paths(item: FileDiff, repo_dirs) -> tuple[str, str]:
    old_path = item.old_path or item.new_path
    new_path = item.new_path or item.old_path
    for name in repo_dirs:
        old_path = strip_duplicated_path_prefix(old_path, [name])
        new_path = strip_duplicated_path_prefix(new_path, [name])
    return old_path, new_path


def render_diff(files: list[FileDiff], repo_dirs=()) -> tuple[str, dict]:
    out: list[str] = []
    report = {
        "files_kept": 0,
        "files_dropped": 0,
        "hunks_kept": 0,
        "hunks_dropped_noop": 0,
        "hunks_dropped_empty": 0,
        "hunks_recounted": 0,
        "hunks_relocated": 0,
    }

    for item in files:
        kept: list[Hunk] = []
        for hunk in item.hunks:
            if not hunk.body:
                report["hunks_dropped_empty"] += 1
                continue
            if not hunk.is_useful():
                report["hunks_dropped_noop"] += 1
                continue
            kept.append(hunk)

        old_path, new_path = _resolve_paths(item, repo_dirs)
        if not kept or not (old_path or new_path):
            report["files_dropped"] += 1
            continue

        report["files_kept"] += 1
        out.append(f"diff --git a/{old_path} b/{new_path}")
        out.append(f"--- {DEV_NULL}" if item.is_new else f"--- a/{old_path}")
        out.append(f"+++ {DEV_NULL}" if item.is_delete else f"+++ b/{new_path}")

        offset = 0
        for hunk in kept:
            old_count = hunk.old_count()
            new_count = hunk.new_count()
            old_start = 0 if item.is_new else max(1, hunk.old_start)
            new_start = 0 if item.is_delete else max(1, old_start + offset)
            offset += new_count - old_count

            if (hunk.declared_old_count, hunk.declared_new_count) != (
                old_count,
                new_count,
            ):
                report["hunks_recounted"] += 1
            if hunk.relocated_from is not None:
                report["hunks_relocated"] += 1
            report["hunks_kept"] += 1

            out.append(
                f"@@ -{old_start},{old_count} "
                f"+{new_start},{new_count} @@{hunk.heading}"
            )
            out.extend(hunk.body)

    return ("\n".join(out) + "\n") if out else "", report


def repair_unified_diff(text: str, repo_dirs=()) -> tuple[str, dict]:
    """Normaliza um diff de LLM sem inventar hunks.

    Retorna ``(patch, report)``; ``patch`` é vazio se nada de útil sobrou.
    """
    return render_diff(parse_diff(text), repo_dirs)


def _match_at(source: list[str], block: list[str], start: int, mode: str) -> bool:
    window = source[start:start + len(block)]
    if len(window) != len(block):
        return False
    if mode == "exact":
        return window == block
    if mode == "rstrip":
        return [x.rstrip() for x in window] == [x.rstrip() for x in block]
    return [x.strip() for x in window] == [x.strip() for x in block]


def _find_block(
    source: list[str],
    block: list[str],
    hint: int,
    floor: int,
) -> int | None:
    if not block:
        return None
    limit = len(source) - len(block)
    if limit < 0:
        return None

    for mode in ("exact", "rstrip", "strip"):
        matches = [
            i
            for i in range(max(0, floor), limit + 1)
            if _match_at(source, block, i, mode)
        ]
        if matches:
            return min(matches, key=lambda i: (abs(i - hint), i))
    return None


def relocate_hunks(
    text: str,
    repo_root: Path,
    repo_dirs=(),
) -> tuple[str, dict]:
    """Corrige os números de linha procurando o lado antigo no arquivo real.

    LLMs acertam o conteúdo com muito mais frequência do que a posição.
    Se o bloco antigo (contexto + linhas removidas) existir no arquivo do
    commit base, o cabeçalho passa a apontar para a posição verdadeira.
    """
    files = parse_diff(text)
    stats = {"relocated": 0, "already_correct": 0, "not_found": 0, "no_file": 0}

    for item in files:
        if item.is_new:
            continue
        target = Path(repo_root) / normalize_relpath(item.old_path)
        if not target.is_file():
            stats["no_file"] += len(item.hunks)
            continue

        source = target.read_text(
            encoding="utf-8", errors="ignore"
        ).replace("\r\n", "\n").split("\n")

        floor = 0
        for hunk in item.hunks:
            block = hunk.old_lines()
            found = _find_block(source, block, hunk.old_start - 1, floor)
            if found is None:
                stats["not_found"] += 1
                continue
            floor = found + 1
            if found + 1 == hunk.old_start:
                stats["already_correct"] += 1
                continue
            hunk.relocated_from = hunk.old_start
            hunk.old_start = found + 1
            stats["relocated"] += 1

    patch, render_report = render_diff(files, repo_dirs)
    render_report.update(
        {f"relocate_{k}": v for k, v in stats.items()}
    )
    return patch, render_report
