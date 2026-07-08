#!/usr/bin/env bash
# One PyBullet process per episode (avoids long GUI session bugs).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

N="${1:-100}"
SEED="${2:-42}"
OUT="${3:-qsfp_insert/il_flow/dataset/raw}"

for ((i = 0; i < N; i++)); do
  seed=$((SEED + i))
  extra=()
  [[ "$i" -eq 0 && "${OVERWRITE:-0}" == 1 ]] && extra+=(--overwrite)
  echo "=== run $((i + 1))/$N seed=$seed ==="
  python qsfp_insert/il_flow/collect_il.py --episodes 1 --seed "$seed" --out-dir "$OUT" "${extra[@]}"
done

echo "done: $N single-episode runs → $OUT"
