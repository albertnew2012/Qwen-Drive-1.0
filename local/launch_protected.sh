#!/bin/bash
# Launch a CPU inference job that cannot disturb a GPU training job sharing the box.
#   bash local/launch_protected.sh <logfile> <cmd...>
# Applies, ~40 s after start (once torch's thread pool exists):
#   * renice 19 on the process AND every one of its threads
#   * taskset confining it to physical cores RESERVE..N-1, leaving 0..RESERVE-1 free
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=$1; shift
RESERVE=${RESERVE_CORES:-4}

export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=src

"$@" > "$LOG" 2>&1 &
PID=$!
echo "launched pid $PID -> $LOG"

(
  sleep 40
  # the python process, not the shell wrapper
  renice -n 19 -p "$PID" >/dev/null 2>&1
  n=0
  for t in $(ls /proc/"$PID"/task 2>/dev/null); do
    renice -n 19 -p "$t" >/dev/null 2>&1 && n=$((n+1))
  done
  PHYS=$(( $(lscpu -p=CPU | grep -vc '^#') / 2 ))
  if [ "$PHYS" -gt "$RESERVE" ]; then
    taskset -apc "$RESERVE-$((PHYS-1)),$((PHYS+RESERVE))-$((2*PHYS-1))" "$PID" >/dev/null 2>&1
  fi
  echo "protected: nice 19 on $n threads, pinned off cores 0-$((RESERVE-1))" >> "$LOG"
) &
echo "$PID"
