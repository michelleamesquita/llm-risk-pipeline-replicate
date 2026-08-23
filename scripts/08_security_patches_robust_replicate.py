#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
08_security_patches_robust_replicate.py

Pipeline robusto:
case
-> prompt
-> Replicate
-> patch
-> aplica patch real
-> Bandit BEFORE/AFTER somente nos arquivos tocados
-> salva relatórios/metadados
-> remove repos_patched/<case> por padrão
-> próximo caso

Use --keep-patched-repos para preservar os repositórios modificados
quando precisar rodar análises posteriores de rede/dependência.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from repo_paths import DEFAULT_REPOS_ROOT, find_local_repo, local_dir_name

PY = ROOT / ".venv" / "bin" / "python"
if not PY.exists():
    PY = Path(sys.executable)

FUZZ_APPLY_PREFIX = "patch "
PROMPT_PROTOCOL_VERSION = "source-context-v6-focused"
SOURCE_CONTEXT_STRATEGY = "verified-traceback-first-weighted-base-excerpt"
SOURCE_CONTEXT_MAX_CHARS = 3500
ISSUE_MAX_CHARS = 1800
PROMPT_MAX_CHARS = 7600


def run(cmd):
    cmd = [str(x) for x in cmd]
    print(">", " ".join(cmd))
    return subprocess.run(cmd).returncode


def safe_remove_tree(path: Path) -> None:
    """
    Remove somente diretórios dentro de runs_backup/*/repos_patched/.
    Proteção adicional contra path incorreto.
    """
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path

    parts = resolved.parts
    allowed_root = (ROOT / "runs_backup").resolve()
    if (
        "repos_patched" not in parts
        or not resolved.is_relative_to(allowed_root)
    ):
        print(f"[WARN] limpeza recusada fora de repos_patched: {resolved}")
        return

    def handle_remove_error(func, target, _exc):
        os.chmod(target, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
        func(target)

    if resolved.exists():
        shutil.rmtree(resolved, onerror=handle_remove_error)
        print(f"[CLEAN] removido: {resolved}")


def safe_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def remove_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception as exc:
        print(f"[WARN] não foi possível remover {path}: {exc}")


def completed_case(
    apply_meta: Path,
    prompt: Path,
    patch: Path,
    gen_meta: Path,
    before: Path,
    after: Path,
) -> bool:
    meta = safe_json(apply_meta)
    return (
        meta.get("experiment_valid") is True
        and meta.get("prompt_protocol_version") == PROMPT_PROTOCOL_VERSION
        and all(p.is_file() for p in (
            prompt, patch, gen_meta, apply_meta, before, after
        ))
    )


def reusable_generation(
    patch: Path,
    gen_meta: Path,
    prompt: Path,
) -> bool:
    meta = safe_json(gen_meta)
    if not prompt.is_file():
        return False
    prompt_text = prompt.read_text(encoding="utf-8", errors="ignore")
    expected_hash = hashlib.sha256(
        prompt_text.encode("utf-8")
    ).hexdigest()
    recorded_hash = meta.get("prompt_sha256")
    same_prompt = (
        recorded_hash == expected_hash
        if recorded_hash
        else meta.get("prompt_chars") == len(prompt_text)
    )
    return (
        patch.is_file()
        and patch.stat().st_size > 0
        and meta.get("success") is True
        and meta.get("prompt_protocol_version") == PROMPT_PROTOCOL_VERSION
        and same_prompt
    )


def load_profile(profiles_path: Path, profile_name: str) -> dict:
    data = yaml.safe_load(profiles_path.read_text(encoding="utf-8")) or {}
    if profile_name not in data:
        raise KeyError(f"Perfil {profile_name!r} não encontrado em {profiles_path}")
    return dict(data[profile_name])


def repos_root() -> Path:
    return ROOT / DEFAULT_REPOS_ROOT


def resolve_repo_src(case: dict, case_path: Path) -> Path | None:
    root = repos_root()
    found = find_local_repo(case, root)
    base_commit = str(case.get("base_commit") or "")
    if found is not None and base_commit:
        check = subprocess.run(
            ["git", "cat-file", "-e", f"{base_commit}^{{commit}}"],
            cwd=str(found),
            capture_output=True,
            text=True,
            check=False,
        )
        if check.returncode == 0:
            return found

    if found is not None:
        print(f"[FETCH missing commit] {base_commit}")

    rc = run([
        PY,
        ROOT / "scripts" / "00_fetch_repo.py",
        "--case_json",
        case_path,
        "--out_root",
        root,
    ])
    if rc:
        return None

    found = find_local_repo(case, root)
    if found is None:
        return None

    check = subprocess.run(
        ["git", "cat-file", "-e", f"{base_commit}^{{commit}}"],
        cwd=str(found),
        capture_output=True,
        text=True,
        check=False,
    )
    if check.returncode == 0:
        return found
    return None


def select_case_paths(cases_dir: Path, case_list_path: Path | None) -> list[Path]:
    cases = sorted(cases_dir.glob("*.json"))
    if not cases:
        raise ValueError(f"Nenhum case encontrado em {cases_dir}")
    if case_list_path is None:
        return cases
    if not case_list_path.is_file():
        raise ValueError(f"--case-list não encontrado: {case_list_path}")

    requested = []
    seen = set()
    for raw_line in case_list_path.read_text(encoding="utf-8").splitlines():
        case_id = raw_line.strip()
        if not case_id or case_id.startswith("#"):
            continue
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", case_id):
            raise ValueError(f"case_id inválido em --case-list: {case_id!r}")
        if case_id not in seen:
            seen.add(case_id)
            requested.append(case_id)

    available = {path.stem: path for path in cases}
    missing = [case_id for case_id in requested if case_id not in available]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} case_id(s) de --case-list não encontrados: "
            f"{preview}"
        )
    selected = [available[case_id] for case_id in requested]
    if not selected:
        raise ValueError("--case-list não contém casos processáveis")
    return selected


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--profile",
        required=True,
    )

    ap.add_argument(
        "--profiles",
        default=str(ROOT / "configs" / "model_profiles.yaml"),
    )

    ap.add_argument(
        "--cases-dir",
        default=str(ROOT / "runs" / "cases"),
    )

    ap.add_argument(
        "--case-list",
        help=(
            "Arquivo texto opcional com um case_id por linha. Quando definido, "
            "somente esses casos são processados."
        ),
    )

    ap.add_argument(
        "--template-md",
        default=str(ROOT / "configs" / "prompt_template.md"),
    )

    ap.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Máximo de casos INCOMPLETOS tentados nesta execução; casos completos retomados não contam. 0 = todos.",
    )

    ap.add_argument(
        "--force-generate",
        action="store_true",
        help="Ignora geração existente e chama o Replicate novamente.",
    )

    ap.add_argument(
        "--max-generation-attempts",
        type=int,
        default=3,
        help=(
            "Máximo uniforme de gerações por caso, incluindo retries após "
            "diff inválido ou patch inaplicável."
        ),
    )

    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="Retorna sucesso mesmo se parte dos casos falhar.",
    )

    ap.add_argument(
        "--keep-patched-repos",
        action="store_true",
        help="Não remove repos_patched/<case> após SAST bem-sucedido.",
    )

    ap.add_argument(
        "--cleanup-on-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Remove repos_patched/<case> em falhas (padrão). Use "
            "--no-cleanup-on-failure para preservar material de debugging."
        ),
    )

    args = ap.parse_args()

    if args.max_cases < 0:
        raise SystemExit("--max-cases deve ser >= 0")
    if args.max_generation_attempts < 1:
        raise SystemExit("--max-generation-attempts deve ser >= 1")

    profiles_path = Path(args.profiles)
    profile = load_profile(profiles_path, args.profile)

    run_dir = profile["run_dir"]

    base = ROOT / "runs_backup" / run_dir
    patches = base / "patches"
    repos = base / "repos_patched"
    reports = base / "reports"
    metadata = base / "metadata"
    generation_meta = base / "generation_metadata"

    for d in [patches, repos, reports, metadata, generation_meta]:
        d.mkdir(parents=True, exist_ok=True)

    valid = 0
    attempted = 0
    resumed = 0
    generation_reused = 0
    generation_failed_cases = 0
    apply_failed_cases = 0
    cleaned = 0
    failed = 0

    try:
        cases = select_case_paths(
            Path(args.cases_dir),
            Path(args.case_list) if args.case_list else None,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    for case_path in cases:
        case = json.loads(case_path.read_text(encoding="utf-8"))

        cid = case["case_id"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(cid)):
            print(f"[FAIL invalid case_id] {cid!r}")
            failed += 1
            continue
        repo_name = case["repo_name"]
        base_commit = case["base_commit"]

        prompt = patches / f"{cid}.prompt.txt"
        patch = patches / f"{cid}.patch"

        gen_meta = generation_meta / f"{cid}.json"
        apply_meta = metadata / f"{cid}.json"

        patched_repo = repos / cid

        before = reports / f"{cid}_bandit_before.json"
        after = reports / f"{cid}_bandit_after.json"

        print()
        print("=" * 80)
        print(f"PROFILE={args.profile} | CASE={cid}")
        print("=" * 80)

        if not args.force_generate and completed_case(
            apply_meta, prompt, patch, gen_meta, before, after
        ):
            meta = safe_json(apply_meta)
            strategy = str(meta.get("apply_strategy") or "")
            meta.update({
                "pipeline_valid": True,
                "functional_tests_run": False,
                "functional_correctness_claimed": False,
                "strict_patch_apply": strategy.startswith("git apply"),
                "fuzzy_patch_apply": strategy.startswith(FUZZ_APPLY_PREFIX),
            })
            apply_meta.write_text(
                json.dumps(meta, indent=2),
                encoding="utf-8",
            )
            valid += 1
            resumed += 1
            print(f"[RESUME complete] {args.profile}: {cid}")
            continue

        # MAX_CASES limita apenas trabalho novo/incompleto. Isso evita que
        # casos já válidos no começo da lista impeçam o pipeline de avançar.
        if args.max_cases and attempted >= args.max_cases:
            break
        attempted += 1

        # Um caso incompleto não pode herdar relatórios/metadata de outra
        # tentativa. A geração é preservada separadamente para poder retomá-la.
        remove_file(apply_meta)
        remove_file(before)
        remove_file(after)

        # ----------------------------------------------------
        # 0) Repo fonte/base — antes da geração, para não gastar API à toa
        # ----------------------------------------------------
        repo_src = resolve_repo_src(case, case_path)
        if repo_src is None:
            expected = repos_root() / local_dir_name(case)
            print(f"[FAIL repo missing] {expected}")
            failed += 1
            continue

        # ----------------------------------------------------
        # 1) Prompt
        # ----------------------------------------------------
        rc = run([
            PY,
            ROOT / "scripts" / "02_prompt_builder.py",
            "--case_json",
            case_path,
            "--template_md",
            args.template_md,
            "--out_prompt",
            prompt,
            "--repo_src",
            repo_src,
            "--max_source_chars",
            SOURCE_CONTEXT_MAX_CHARS,
            "--max_issue_chars",
            ISSUE_MAX_CHARS,
            "--max_prompt_chars",
            PROMPT_MAX_CHARS,
        ])

        if rc:
            print(f"[FAIL prompt] {cid}")
            failed += 1
            if args.cleanup_on_failure:
                safe_remove_tree(patched_repo)
            continue

        # ----------------------------------------------------
        # 2-3) Geração e aplicação com política uniforme de tentativas
        # ----------------------------------------------------
        can_reuse_generation = (
            not args.force_generate
            and reusable_generation(patch, gen_meta, prompt)
        )
        end_to_end_attempts = []
        apply_succeeded = False
        last_stage = "generation"

        for generation_attempt in range(1, args.max_generation_attempts + 1):
            reused_this_attempt = (
                can_reuse_generation and generation_attempt == 1
            )

            if reused_this_attempt:
                generation_reused += 1
                print(f"[REUSE generation] {cid}")
            else:
                # Uma nova tentativa nunca pode herdar saída/metadata antigos.
                last_stage = "generation"
                remove_file(patch)
                remove_file(gen_meta)
                rc = run([
                    PY,
                    ROOT / "scripts" / "03_model_adapter_replicate.py",
                    "--profile",
                    args.profile,
                    "--profiles",
                    profiles_path,
                    "--prompt_file",
                    prompt,
                    "--out_patch",
                    patch,
                    "--metadata_json",
                    gen_meta,
                    "--retries",
                    "1",
                ])

                if rc:
                    end_to_end_attempts.append({
                        "attempt": generation_attempt,
                        "generation_reused": False,
                        "stage": "generation",
                        "returncode": rc,
                        "generation": safe_json(gen_meta),
                    })
                    print(
                        f"[RETRY generation {generation_attempt}/"
                        f"{args.max_generation_attempts}] {cid}"
                    )
                    continue

                generated = safe_json(gen_meta)
                generated.update({
                    "prompt_protocol_version": PROMPT_PROTOCOL_VERSION,
                    "source_context_strategy": SOURCE_CONTEXT_STRATEGY,
                    "source_context_max_chars": SOURCE_CONTEXT_MAX_CHARS,
                    "issue_max_chars": ISSUE_MAX_CHARS,
                    "prompt_max_chars": PROMPT_MAX_CHARS,
                })
                gen_meta.write_text(
                    json.dumps(generated, indent=2),
                    encoding="utf-8",
                )

            remove_file(apply_meta)
            rc = run([
                PY,
                ROOT / "scripts" / "04_apply_patch_robust.py",
                "--repo_src",
                repo_src,
                "--base_commit",
                base_commit,
                "--patch_file",
                patch,
                "--out_repo",
                patched_repo,
                "--metadata_json",
                apply_meta,
            ])
            apply_result = safe_json(apply_meta)
            end_to_end_attempts.append({
                "attempt": generation_attempt,
                "generation_reused": reused_this_attempt,
                "stage": "apply",
                "returncode": rc,
                "apply_error": apply_result.get("error"),
                "apply_strategy": apply_result.get("apply_strategy"),
                "diff_repair": apply_result.get("diff_repair"),
            })

            if rc == 0:
                apply_succeeded = True
                break

            last_stage = "apply"
            print(
                f"[RETRY apply {generation_attempt}/"
                f"{args.max_generation_attempts}] {cid}"
            )
            if args.cleanup_on_failure:
                safe_remove_tree(patched_repo)

        generated = safe_json(gen_meta)
        generated["end_to_end_attempts"] = end_to_end_attempts
        generated["end_to_end_attempt_count"] = len(end_to_end_attempts)
        generated["apply_success"] = apply_succeeded
        gen_meta.write_text(
            json.dumps(generated, indent=2),
            encoding="utf-8",
        )

        if not apply_succeeded:
            print(f"[FAIL {last_stage}] {cid}")
            failed += 1
            if last_stage == "generation":
                generation_failed_cases += 1
            else:
                apply_failed_cases += 1
            if args.cleanup_on_failure:
                safe_remove_tree(patched_repo)
            continue

        # ----------------------------------------------------
        # 5) Enriquece metadata
        # ----------------------------------------------------
        meta = json.loads(apply_meta.read_text(encoding="utf-8"))

        meta.update({
            "repo_name": repo_name,
            "case_id": cid,
            "model_profile": args.profile,
            "provider": "replicate",
            "replicate_model": profile["replicate_model"],
            "requested_temperature": profile.get("requested_temperature"),
            "effective_temperature": profile.get("effective_temperature"),
            "generation_metadata_json": str(gen_meta),
            "generation_attempt_count": len(end_to_end_attempts),
            "max_generation_attempts": args.max_generation_attempts,
            "patched_repo_cleanup_policy": (
                "keep"
                if args.keep_patched_repos
                else "remove_after_success"
            ),
        })

        apply_meta.write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )

        # ----------------------------------------------------
        # 6) Bandit BEFORE/AFTER nos arquivos tocados
        # ----------------------------------------------------
        rc = run([
            PY,
            ROOT / "scripts" / "05_sast_touched_before_after.py",
            "--repo_src",
            repo_src,
            "--patched_repo",
            patched_repo,
            "--base_commit",
            base_commit,
            "--metadata_json",
            apply_meta,
            "--before_json",
            before,
            "--after_json",
            after,
        ])

        if rc:
            print(f"[FAIL SAST] {cid}")
            failed += 1

            if args.cleanup_on_failure:
                safe_remove_tree(patched_repo)

            continue

        # ----------------------------------------------------
        # 7) Confirma artefatos persistentes antes da limpeza
        # ----------------------------------------------------
        required_outputs = [
            prompt,
            patch,
            gen_meta,
            apply_meta,
            before,
            after,
        ]

        missing = [p for p in required_outputs if not p.exists()]

        if missing:
            print("[FAIL outputs] faltando:")
            for p in missing:
                print(f"  - {p}")

            failed += 1

            if args.cleanup_on_failure:
                safe_remove_tree(patched_repo)

            continue

        # ----------------------------------------------------
        # 8) Marca sucesso no metadata
        # ----------------------------------------------------
        meta = json.loads(apply_meta.read_text(encoding="utf-8"))

        meta.update({
            "experiment_valid": True,
            "pipeline_valid": True,
            "prompt_protocol_version": PROMPT_PROTOCOL_VERSION,
            "source_context_strategy": SOURCE_CONTEXT_STRATEGY,
            "source_context_max_chars": SOURCE_CONTEXT_MAX_CHARS,
            "issue_max_chars": ISSUE_MAX_CHARS,
            "prompt_max_chars": PROMPT_MAX_CHARS,
            "functional_tests_run": False,
            "functional_correctness_claimed": False,
            "strict_patch_apply": str(
                meta.get("apply_strategy") or ""
            ).startswith("git apply"),
            "fuzzy_patch_apply": str(
                meta.get("apply_strategy") or ""
            ).startswith(FUZZ_APPLY_PREFIX),
            "bandit_before_json": str(before),
            "bandit_after_json": str(after),
            "prompt_file": str(prompt),
            "patch_file": str(patch),
        })

        apply_meta.write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )

        valid += 1
        print(f"[PIPELINE_VALID] {args.profile}: {cid}")

        # ----------------------------------------------------
        # 9) LIMPEZA DE DISCO
        # ----------------------------------------------------
        if not args.keep_patched_repos:
            try:
                safe_remove_tree(patched_repo)
                cleaned += 1

                # Atualiza metadata após limpeza.
                meta = json.loads(
                    apply_meta.read_text(encoding="utf-8")
                )

                meta.update({
                    "patched_repo_removed_after_analysis": True,
                    "patched_repo_path": str(patched_repo),
                })

                apply_meta.write_text(
                    json.dumps(meta, indent=2),
                    encoding="utf-8",
                )

            except Exception as exc:
                print(
                    f"[WARN] não foi possível remover {patched_repo}: {exc}"
                )

    print()
    print("=" * 80)
    print("EXECUÇÃO FINALIZADA")
    print("=" * 80)
    print(f"Profile            : {args.profile}")
    print(f"Tentados           : {attempted}")
    print(f"Pipeline válidos   : {valid}")
    print(f"Retomados completos: {resumed}")
    print(f"Gerações reutilizadas: {generation_reused}")
    print(f"Falhas de geração  : {generation_failed_cases}")
    print(f"Falhas de aplicação: {apply_failed_cases}")
    print(f"Falhas             : {failed}")
    print(f"Repos válidos removidos: {cleaned}")
    print(
        f"Repos preservados  : "
        f"{'SIM' if args.keep_patched_repos else 'NÃO'}"
    )
    print()

    if failed and not args.allow_partial:
        return 2
    return 0 if valid > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())