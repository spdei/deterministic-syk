#!/usr/bin/env bash
# Runs the full reproduction chain from the repository root, logging each step to logs/.
# Usage:   bash run_all.sh            (everything)
#          bash run_all.sh supplement (only the supplement, needs the ED outputs)
# In the background:   mkdir -p logs && nohup bash run_all.sh > logs/run_all.log 2>&1 &
# Interpreter: python3 by default (stock Debian/Ubuntu have no "python"); override with
#              PYTHON=/path/to/python bash run_all.sh
# Resuming:    each finished step leaves logs/<step>.done; if the run is interrupted, run the
#              same command again and finished steps are skipped. rm -rf logs to start over.
set -euo pipefail

case "${1:-all}" in
  all|supplement) ;;
  *) echo "usage: bash run_all.sh [supplement]" >&2; exit 2 ;;
esac

PY="${PYTHON:-$(command -v python3 || command -v python || true)}"
if [ -z "$PY" ]; then
  echo "error: no python3 found; install it or set PYTHON=/path/to/python" >&2
  exit 1
fi
case "$PY" in
  /*) ;;
  */*) PY="$PWD/$PY" ;;            # a relative path: resolve it before changing directory
esac

cd "$(dirname "$0")/code"
mkdir -p ../logs

step () {
  local log
  log="../logs/$(echo "$*" | tr ' ' '_').log"
  if [ -e "${log%.log}.done" ]; then
    echo "=== $*: already done (skipped; rm -rf logs to rerun everything)"
    return 0
  fi
  echo "=== $(date '+%H:%M:%S')  $*"
  if ! "$PY" "$@" > "$log" 2>&1; then
    echo "    FAILED ($(date '+%H:%M:%S')); last lines of logs/${log#../logs/}:"
    tail -n 20 "$log" | sed 's/^/      /'
    exit 1
  fi
  touch "${log%.log}.done"
  echo "    done ($(date '+%H:%M:%S'))"
}

step hsyk_selfcheck.py
if [ "${1:-all}" != "supplement" ]; then
  step hsyk_numerics.py full
  step hsyk_numerics.py postprocess
  step hsyk_numerics.py otoc_reanalysis
  step dense_sweep.py
  step make_fig_deltaG_vs_N.py
  step hsyk_constructability.py
fi
step hsyk_supplement.py all
"$PY" hsyk_supplement.py check_data | tee ../logs/check_data.log
