#!/usr/bin/env python3
"""Planeja somente os pares caso/modelo ausentes da interseção válida."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODEL_DIRS = {
    "gpt-4o": "gpt-4o_backup",
    "claude": "claud-sonnet_backup",
    "deepseek": "deepseek_backup",
    "codellama-tuned": "codellama_tuned_backup",
}


def safe_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def valid_cases(runs_root: Path, run_dir: str) -> set[str]:
    metadata_dir = runs_root / run_dir / "metadata"
    if not metadata_dir.is_dir():
        return set()

    valid = set()
    for path in metadata_dir.glob("*.json"):
        metadata = safe_json(path)
        if (
            metadata.get("experiment_valid") is True
            and metadata.get("patch_apply_success") is True
        ):
            valid.add(path.stem)
    return valid


def load_attempted_pairs(state_path: Path) -> set[str]:
    state = safe_json(state_path)
    values = state.get("attempted_pairs", [])
    if not isinstance(values, list):
        return set()
    return {str(value) for value in values}


def write_lines(path: Path, values: list[str]) -> None:
    text = "".join(f"{value}\n" for value in values)
    path.write_text(text, encoding="utf-8")


def build_plan(
    runs_root: Path,
    cases_dir: Path,
    out_dir: Path,
    state_path: Path,
    target_common: int,
    batch_cases: int,
    commit_state: bool,
) -> dict[str, Any]:
    case_ids = sorted(path.stem for path in cases_dir.glob("*.json"))
    if not case_ids:
        raise ValueError(f"Nenhum caso encontrado em {cases_dir}")

    valid_by_model = {
        model: valid_cases(runs_root, run_dir)
        for model, run_dir in MODEL_DIRS.items()
    }
    common = set.intersection(*valid_by_model.values())
    attempted_pairs = load_attempted_pairs(state_path)

    candidates = []
    deferred = 0
    for case_id in case_ids:
        missing = tuple(
            model
            for model in MODEL_DIRS
            if case_id not in valid_by_model[model]
        )
        if not missing:
            continue

        pair_keys = {f"{model}|{case_id}" for model in missing}
        if pair_keys & attempted_pairs:
            deferred += 1
            continue

        candidates.append((
            len(missing),
            case_id,
            missing,
        ))

    candidates.sort(key=lambda item: (item[0], item[1]))
    needed = max(0, target_common - len(common))
    selected = candidates[:min(batch_cases, needed)]

    selected_by_model = {model: [] for model in MODEL_DIRS}
    for _, case_id, missing in selected:
        for model in missing:
            selected_by_model[model].append(case_id)

    out_dir.mkdir(parents=True, exist_ok=True)
    for model, case_list in selected_by_model.items():
        write_lines(out_dir / f"{model}.txt", case_list)

    manifest_path = out_dir / "manifest.tsv"
    manifest_lines = [
        f"{model}\t{len(case_list)}\t{out_dir / f'{model}.txt'}"
        for model, case_list in selected_by_model.items()
    ]
    write_lines(manifest_path, manifest_lines)

    summary = {
        "target_common": target_common,
        "common_before": len(common),
        "needed": needed,
        "selected_cases": len(selected),
        "selected_pairs": sum(map(len, selected_by_model.values())),
        "eligible_candidates": len(candidates),
        "deferred_cases": deferred,
        "valid_by_model": {
            model: len(values) for model, values in valid_by_model.items()
        },
        "selected_by_model": {
            model: len(values) for model, values in selected_by_model.items()
        },
        "manifest": str(manifest_path),
    }
    (out_dir / "plan.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    if commit_state and selected:
        for _, case_id, missing in selected:
            attempted_pairs.update(
                f"{model}|{case_id}" for model in missing
            )
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = state_path.with_suffix(f"{state_path.suffix}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "attempted_pairs": sorted(attempted_pairs),
                    "last_plan": summary,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(state_path)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs_backup")
    parser.add_argument("--cases-dir", default="runs/cases")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--state",
        default="runs_backup/target_common_state.json",
    )
    parser.add_argument("--target-common", type=int, default=300)
    parser.add_argument("--batch-cases", type=int, default=100)
    parser.add_argument("--commit-state", action="store_true")
    args = parser.parse_args()

    if args.target_common < 1:
        parser.error("--target-common deve ser >= 1")
    if args.batch_cases < 1:
        parser.error("--batch-cases deve ser >= 1")

    try:
        summary = build_plan(
            Path(args.runs_root),
            Path(args.cases_dir),
            Path(args.out_dir),
            Path(args.state),
            args.target_common,
            args.batch_cases,
            args.commit_state,
        )
    except ValueError as exc:
        parser.error(str(exc))

    for model, count in summary["valid_by_model"].items():
        print(f"{model}: {count} valid")
    print(
        f"Interseção: {summary['common_before']} / "
        f"{summary['target_common']}"
    )
    print(
        f"Plano: {summary['selected_cases']} casos, "
        f"{summary['selected_pairs']} chamadas de pipeline"
    )
    for model, count in summary["selected_by_model"].items():
        print(f"  {model}: {count} casos ausentes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
