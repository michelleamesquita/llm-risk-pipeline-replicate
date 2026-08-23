#!/usr/bin/env bash
set -uo pipefail

if [[ -z "${REPLICATE_API_TOKEN:-}" ]]; then
  echo "ERRO: exporte REPLICATE_API_TOKEN antes de executar." >&2
  exit 2
fi

TARGET_COMMON="${TARGET_COMMON:-300}"
BATCH_CASES="${BATCH_CASES:-100}"
GENERATION_ATTEMPTS="${GENERATION_ATTEMPTS:-3}"
MAX_ROUNDS="${MAX_ROUNDS:-30}"
BUILD_FLAT="${BUILD_FLAT:-1}"
RESET_TARGET_STATE="${RESET_TARGET_STATE:-0}"

for spec in "TARGET_COMMON:$TARGET_COMMON" "BATCH_CASES:$BATCH_CASES" "GENERATION_ATTEMPTS:$GENERATION_ATTEMPTS" "MAX_ROUNDS:$MAX_ROUNDS"; do
  name="${spec%%:*}"; value="${spec#*:}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERRO: $name deve ser inteiro >= 1." >&2; exit 2
  fi
done
if [[ "$BUILD_FLAT" != "0" && "$BUILD_FLAT" != "1" ]]; then
  echo "ERRO: BUILD_FLAT deve ser 0 ou 1." >&2; exit 2
fi
if [[ "$RESET_TARGET_STATE" != "0" && "$RESET_TARGET_STATE" != "1" ]]; then
  echo "ERRO: RESET_TARGET_STATE deve ser 0 ou 1." >&2; exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STATE_FILE="$ROOT/runs_backup/target_common_state.json"

if [[ "$RESET_TARGET_STATE" == "1" ]]; then
  rm -f "$STATE_FILE"
  echo "Estado de seleção reiniciado: $STATE_FILE"
fi

common_count() {
python - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1]) / "runs_backup"
model_dirs={
 "gpt-4o":"gpt-4o_backup",
 "claude":"claud-sonnet_backup",
 "deepseek":"deepseek_backup",
 "codellama-tuned":"codellama_tuned_backup",
}
sets=[]
for model,d in model_dirs.items():
    valid=set()
    md=root/d/"metadata"
    if md.is_dir():
        for p in md.glob("*.json"):
            try: m=json.loads(p.read_text())
            except Exception: continue
            if m.get("experiment_valid") is True and m.get("patch_apply_success"):
                valid.add(p.stem)
    print(f"{model}: {len(valid)} valid", file=sys.stderr)
    sets.append(valid)
print(len(set.intersection(*sets)) if sets else 0)
PY
}

start="$(common_count)"
echo "Interseção inicial: $start / $TARGET_COMMON"
if (( start >= TARGET_COMMON )); then
  echo "Meta já atingida; nenhuma chamada de API necessária."
  exit 0
fi

for ((round=1; round<=MAX_ROUNDS; round++)); do
  before="$(common_count)"
  echo "===== RODADA $round | common=$before/$TARGET_COMMON ====="

  timestamp="$(date +%Y%m%d_%H%M%S)_r${round}"
  log_dir="$ROOT/runs_backup/parallel_logs/$timestamp"
  plan_dir="$ROOT/runs/target_common_plans/$timestamp"
  mkdir -p "$log_dir"

  python scripts/target_common_planner.py \
    --runs-root runs_backup \
    --cases-dir runs/cases \
    --out-dir "$plan_dir" \
    --state "$STATE_FILE" \
    --target-common "$TARGET_COMMON" \
    --batch-cases "$BATCH_CASES" \
    --commit-state

  selected_pairs=0
  while IFS=$'\t' read -r profile count case_list; do
    if [[ -z "$profile" || "$count" == "0" ]]; then
      continue
    fi
    selected_pairs=$((selected_pairs + count))
    log_file="$log_dir/$profile.log"
    echo "Executando somente ausentes: $profile ($count casos)"
    if make run \
      "PROFILE=$profile" \
      "CASE_LIST=$case_list" \
      "GENERATION_ATTEMPTS=$GENERATION_ATTEMPTS" \
      "MAX_CASES=$count" \
      "ALLOW_PARTIAL=1" >"$log_file" 2>&1; then
      echo "OK: $profile"
    else
      echo "AVISO: $profile terminou com falha; veja $log_file" >&2
    fi
  done <"$plan_dir/manifest.tsv"

  if (( selected_pairs == 0 )); then
    echo "Sem novos pares caso/modelo elegíveis. Revise $STATE_FILE." >&2
    if [[ "$BUILD_FLAT" == "1" ]]; then make build-flat || true; fi
    exit 3
  fi

  after="$(common_count)"
  echo "Interseção após rodada $round: $after / $TARGET_COMMON"

  if (( after >= TARGET_COMMON )); then
    echo "META ATINGIDA: $after casos comuns válidos."
    if [[ "$BUILD_FLAT" == "1" ]]; then make build-flat; fi
    exit 0
  fi

  if (( after <= before )); then
    echo "AVISO: esta rodada não aumentou a interseção; a próxima usará outros pares." >&2
  fi
done

echo "Meta não atingida após $MAX_ROUNDS rodadas. Rode novamente ou revise os logs/status." >&2
if [[ "$BUILD_FLAT" == "1" ]]; then make build-flat || true; fi
exit 3