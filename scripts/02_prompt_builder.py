#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Monta o prompt a partir do case JSON (schema atual ou swe-sec)."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath


STOP_WORDS = {
    "about", "after", "before", "being", "below", "cannot", "could",
    "description", "does", "field", "fields", "from", "have", "into",
    "issue", "model", "should", "some", "that", "their", "there", "these",
    "this", "value", "values", "when", "where", "which", "with", "would",
}
TRACEBACK_RE = re.compile(
    r'File\s+["\']([^"\']+\.py)["\'],\s+line\s+(\d+)'
)


def render_template(tpl: str, mapping: dict) -> str:
    for key, value in mapping.items():
        tpl = tpl.replace("{" + key + "}", str(value or ""))
    return tpl


def first_present(case: dict, *keys: str, default: str = "") -> str:
    for key in keys:
        value = case.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return str(value)
    return default


def short_title(text: str) -> str:
    line1 = (text or "").strip().splitlines()[0] if text else ""
    return (line1[:120] + "…") if len(line1) > 120 else line1


def compact_issue(text: str, title: str, limit: int) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # Alguns casos legados concatenam o mesmo issue duas vezes.
    if title:
        repeated_at = text.find(title, max(len(title), len(text) // 3))
        if repeated_at > 0:
            text = text[:repeated_at]
        if text.startswith(title):
            text = text[len(title):].lstrip()

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if limit <= 0 or len(text) <= limit:
        return text

    marker = "\n\n[issue truncated for shared context budget]"
    available = max(0, limit - len(marker))
    return text[:available].rstrip() + marker


def short_file_list_from_patch(patch_text: str, limit: int = 12) -> str:
    if not patch_text:
        return ""
    seen = set()
    out = []
    for line in patch_text.splitlines():
        if line.startswith(("--- a/", "+++ b/")):
            path = line[6:]
            if path and path != "/dev/null" and path not in seen:
                seen.add(path)
                out.append(path)
        if len(out) >= limit:
            break
    return "\n".join(out)


def normalize_candidate_path(value: str) -> str | None:
    path = (
        str(value or "")
        .strip()
        .strip("`\"'()[]{}<>,:")
        .replace("\\", "/")
    )
    pure = PurePosixPath(path)
    if (
        not path
        or path.startswith(("~/", "site-packages/"))
        or "/site-packages/" in path
        or pure.is_absolute()
        or ".." in pure.parts
        or path == "/dev/null"
    ):
        return None
    return str(pure)


def referenced_python_paths(text: str) -> list[str]:
    """Extrai caminhos .py sem regex ambígua/backtracking."""
    paths = []
    for token in str(text or "").split():
        marker = token.find(".py")
        if marker < 0:
            continue
        candidate = token[:marker + 3].lstrip("`\"'([{")
        if candidate:
            paths.append(candidate)
    return paths


def referenced_locations(case: dict) -> dict[str, list[int]]:
    """Extrai arquivos/linhas citados em tracebacks e consultas de teste."""
    title = first_present(case, "issue_title")
    problem = compact_issue(
        first_present(case, "issue_body", "problem_statement"),
        title,
        20000,
    )
    tests = first_present(case, "test_query", "FAIL_TO_PASS", "PASS_TO_PASS")
    locations: dict[str, list[int]] = {}

    for raw_path, raw_line in TRACEBACK_RE.findall(problem):
        path = normalize_candidate_path(raw_path)
        if path:
            locations.setdefault(path, []).append(int(raw_line))

    for raw_path in referenced_python_paths(f"{problem}\n{tests}"):
        path = normalize_candidate_path(raw_path)
        if path:
            locations.setdefault(path, [])

    return locations


def relevant_paths(case: dict) -> list[str]:
    value = first_present(case, "short_file_list")
    if not value:
        value = short_file_list_from_patch(
            first_present(case, "gold_patch", "patch")
        )

    explicit = []
    for raw in value.splitlines():
        normalized = normalize_candidate_path(raw)
        if normalized and normalized not in explicit:
            explicit.append(normalized)

    locations = referenced_locations(case)
    referenced = list(locations)
    matched_references = [path for path in referenced if path in explicit]
    trace_references = [
        path
        for path, lines in locations.items()
        if lines and path not in matched_references
    ]
    extra_references = [
        path
        for path in referenced
        if path not in explicit
        and path not in trace_references
        and "/" in path
    ]

    out = []
    seen = set()
    candidates = (
        matched_references
        + trace_references
        + explicit
        + extra_references
    )
    for raw in candidates:
        normalized = normalize_candidate_path(raw)
        if normalized is None:
            continue
        if normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out[:12]


def issue_terms(case: dict) -> dict[str, int]:
    problem = first_present(case, "issue_body", "problem_statement")
    title = first_present(case, "issue_title")
    problem = compact_issue(problem, title, 10000)
    tests = first_present(case, "test_query", "FAIL_TO_PASS", "PASS_TO_PASS")
    text = f"{title}\n{problem}\n{tests}"
    weighted: dict[str, int] = {}

    for symbol in re.findall(r"\bin\s+([A-Za-z_]\w*)", problem):
        weighted[symbol.lower()] = 10

    for symbol in re.findall(r"::([A-Za-z_]\w*)", tests):
        weighted[symbol.lower()] = 10

    for quoted in re.findall(r"`([^`\n]{2,100})`", text):
        for term in re.findall(r"[A-Za-z_][A-Za-z0-9_.]{2,}", quoted):
            weighted[term.lower()] = 8

    for term in re.findall(r"[A-Za-z_][A-Za-z0-9_.]{3,}", text):
        lowered = term.lower()
        if lowered not in STOP_WORDS:
            weighted[lowered] = max(weighted.get(lowered, 0), 1)

    return dict(sorted(
        weighted.items(),
        key=lambda item: (-item[1], -len(item[0]), item[0]),
    )[:100])


def git_file_at_commit(repo: Path, commit: str, relpath: str) -> str | None:
    if not re.fullmatch(r"[0-9a-fA-F]{7,64}", commit):
        return None
    result = subprocess.run(
        ["git", "show", f"{commit}:{relpath}"],
        cwd=str(repo),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or b"\x00" in result.stdout:
        return None
    return result.stdout.decode("utf-8", errors="replace")


def merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def source_excerpt(
    relpath: str,
    content: str,
    terms: dict[str, int],
    char_budget: int,
    preferred_lines: list[int] | None = None,
) -> str:
    header = f"--- BEGIN FILE: {relpath} @ base commit ---"
    footer = f"--- END FILE: {relpath} ---"
    if len(content) <= char_budget:
        return f"{header}\n{content.rstrip()}\n{footer}"

    lines = content.splitlines()
    scored = []
    for index, line in enumerate(lines):
        lowered = line.lower()
        stripped = lowered.lstrip()
        score = 0
        for term, weight in terms.items():
            if term not in lowered:
                continue
            score += weight
            if stripped.startswith((f"class {term}", f"def {term}")):
                score += 20
        if score:
            scored.append((score, index))

    radius = 12
    max_windows = 1 if char_budget < 2600 else 3
    windows: list[tuple[int, int]] = []

    def add_window(start: int, end: int) -> None:
        for position, (current_start, current_end) in enumerate(windows):
            if start <= current_end + 1 and end >= current_start - 1:
                windows[position] = (
                    min(start, current_start),
                    max(end, current_end),
                )
                return
        windows.append((start, end))

    # Linhas explícitas do traceback têm prioridade sobre similaridade lexical.
    for line_number in preferred_lines or []:
        index = max(0, min(len(lines) - 1, line_number - 1))
        add_window(
            max(0, index - radius),
            min(len(lines) - 1, index + radius),
        )
        if len(windows) >= max_windows:
            break

    for _score, index in sorted(scored, key=lambda item: (-item[0], item[1])):
        if any(start <= index <= end for start, end in windows):
            continue
        add_window(
            max(0, index - radius),
            min(len(lines) - 1, index + radius),
        )
        if len(windows) >= max_windows:
            break

    if not windows:
        half = max(20, min(60, len(lines) // 8))
        windows = [(0, half - 1), (max(0, len(lines) - half), len(lines) - 1)]

    sections = []
    used = len(header) + len(footer) + 2
    for start, end in windows:
        label = f"--- {relpath}: lines {start + 1}-{end + 1} ---"
        body = "\n".join(lines[start:end + 1])
        remaining = char_budget - used - len(label) - 2
        if remaining <= 0:
            break
        body = body[:remaining]
        sections.append(f"{label}\n{body}")
        used += len(label) + len(body) + 2

    return f"{header}\n" + "\n".join(sections) + f"\n{footer}"


def build_source_context(
    case: dict,
    repo_src: str | None,
    max_chars: int,
) -> str:
    paths = relevant_paths(case)
    if not repo_src:
        return "[source context unavailable: --repo_src was not provided]"
    if not paths:
        return "[source context unavailable: no relevant files identified]"

    repo = Path(repo_src).resolve()
    if not (repo / ".git").exists():
        return f"[source context unavailable: invalid repository {repo}]"

    commit = first_present(case, "base_commit")
    terms = issue_terms(case)
    locations = referenced_locations(case)
    available = []
    for relpath in paths:
        content = git_file_at_commit(repo, commit, relpath)
        if content is not None:
            available.append((relpath, content))
    available.sort(key=lambda item: "/tests/" in item[0])

    if not available:
        return "[source context unavailable: relevant files missing at base commit]"

    sections = []
    remaining = max_chars

    for index, (relpath, content) in enumerate(available):
        remaining_items = available[index:]
        weights = [
            1 if "/tests/" in candidate_path else 3
            for candidate_path, _content in remaining_items
        ]
        current_weight = weights[0]
        budget = max(
            600,
            remaining * current_weight // max(1, sum(weights)),
        )
        section = source_excerpt(
            relpath,
            content,
            terms,
            budget,
            locations.get(relpath),
        )
        if len(section) > remaining:
            section = section[:remaining]
        sections.append(section)
        remaining -= len(section)
        if remaining <= 0:
            break

    return "\n\n".join(sections)


def prompt_fields(
    case: dict,
    source_context: str = "",
    max_issue_chars: int = 1800,
) -> dict:
    problem = first_present(case, "issue_body", "problem_statement")
    title = first_present(case, "issue_title") or short_title(problem)
    problem = compact_issue(problem, title, max_issue_chars)
    files = "\n".join(relevant_paths(case))
    tests = first_present(case, "test_query", "FAIL_TO_PASS", "PASS_TO_PASS")
    repo = first_present(case, "repo_full_name", "repo_name")

    return {
        "issue_title": title,
        "issue_body": problem,
        "repo_name": repo,
        "base_commit": first_present(case, "base_commit"),
        "short_file_list": files,
        "test_query": tests,
        "source_context": source_context,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_json", required=True)
    ap.add_argument("--template_md", default="configs/prompt_template.md")
    ap.add_argument("--out_prompt", required=True)
    ap.add_argument("--repo_src")
    ap.add_argument("--max_source_chars", type=int, default=3500)
    ap.add_argument("--max_issue_chars", type=int, default=1800)
    ap.add_argument("--max_prompt_chars", type=int, default=7600)
    args = ap.parse_args()

    case = json.load(open(args.case_json, "r", encoding="utf-8"))
    tpl = open(args.template_md, "r", encoding="utf-8").read()
    context = build_source_context(
        case,
        args.repo_src,
        max(0, args.max_source_chars),
    )
    prompt = render_template(
        tpl,
        prompt_fields(case, context, max(0, args.max_issue_chars)),
    )
    if args.max_prompt_chars > 0 and len(prompt) > args.max_prompt_chars:
        raise SystemExit(
            "PROMPT_TOO_LONG_AFTER_COMPACTION: "
            f"{len(prompt)} > {args.max_prompt_chars}"
        )

    out_dir = os.path.dirname(args.out_prompt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    open(args.out_prompt, "w", encoding="utf-8").write(prompt)
    print("Prompt salvo em", args.out_prompt)
