#!/usr/bin/env bash

set -uo pipefail

if [[ -z "${REPLICATE_API_TOKEN:-}" ]]; then
    echo "ERRO: exporte REPLICATE_API_TOKEN antes de executar." >&2
    exit 2
fi

MAX_CASES="${MAX_CASES:-}"
GENERATION_ATTEMPTS="${GENERATION_ATTEMPTS:-3}"
CONFIRM_ALL="${CONFIRM_ALL:-0}"
BUILD_FLAT="${BUILD_FLAT:-1}"

if [[ ! "$GENERATION_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERRO: GENERATION_ATTEMPTS deve ser um inteiro >= 1." >&2
    exit 2
fi

if [[ -n "$MAX_CASES" && ! "$MAX_CASES" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERRO: MAX_CASES deve ser vazio ou um inteiro >= 1." >&2
    exit 2
fi

if [[ -z "$MAX_CASES" && "$CONFIRM_ALL" != "1" ]]; then
    echo "ERRO: defina MAX_CASES=N ou CONFIRM_ALL=1." >&2
    exit 2
fi

if [[ "$BUILD_FLAT" != "0" && "$BUILD_FLAT" != "1" ]]; then
    echo "ERRO: BUILD_FLAT deve ser 0 ou 1." >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

profiles=(
    "gpt-4o"
    "claude"
    "deepseek"
    "codellama-tuned"
)

timestamp="$(date +%Y%m%d_%H%M%S)"
log_dir="$ROOT/runs_backup/parallel_logs/$timestamp"
mkdir -p "$log_dir"

pids=()
names=()

stop_children() {
    trap - INT TERM
    echo
    echo "Interrompendo execuções paralelas..."
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait
    exit 130
}

trap stop_children INT TERM

for profile in "${profiles[@]}"; do
    log_file="$log_dir/$profile.log"
    command=(
        make run
        "PROFILE=$profile"
        "GENERATION_ATTEMPTS=$GENERATION_ATTEMPTS"
        "ALLOW_PARTIAL=1"
    )

    if [[ -n "$MAX_CASES" ]]; then
        command+=("MAX_CASES=$MAX_CASES")
    else
        command+=("CONFIRM_ALL=1")
    fi

    echo "Iniciando $profile -> $log_file"
    "${command[@]}" >"$log_file" 2>&1 &
    pids+=("$!")
    names+=("$profile")
done

status=0
for index in "${!pids[@]}"; do
    pid="${pids[$index]}"
    profile="${names[$index]}"
    if wait "$pid"; then
        echo "OK: $profile"
    else
        returncode=$?
        echo "FALHA: $profile (código $returncode)"
        status=1
    fi
done

trap - INT TERM

if [[ "$BUILD_FLAT" == "1" ]]; then
    echo "Gerando CSVs consolidados..."
    if ! make build-flat; then
        echo "FALHA: não foi possível gerar os CSVs." >&2
        status=1
    fi
fi

echo "Logs: $log_dir"
exit "$status"
