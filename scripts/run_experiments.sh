#!/bin/bash
# Re-run the Deep RAPM experiment suite under both data presets
# (see experiments/config.py): "regular" reproduces the March 2026 design on
# the repaired data; "all" adds playoffs and preseason and every season.
#
#     caffeinate -ims bash scripts/run_experiments.sh            # everything
#     STAGES="phase0_baselines phase3_m10" bash scripts/run_experiments.sh
#     PRESETS="all" bash scripts/run_experiments.sh
#
# Stages run in order of importance so partial results are useful early;
# each (stage, preset) pair logs to outputs/logs/experiments/ and appends a
# line to outputs/logs/experiments/chain.log. Results land in
# experiments/results/<preset>/.
cd "$(dirname "$0")/.." || exit 1
export PYTHONWARNINGS=ignore
LOG=outputs/logs/experiments
mkdir -p "$LOG"
STAGES=${STAGES:-"phase0_baselines phase3_m10 phase3_m10b phase3_m10c phase2_m9 phase2_neural phase2_m8r phase2_m11 phase2_m11b phase2_m9_kfold"}
PRESETS=${PRESETS:-"regular all"}

# M10's player features read data/player_anthropometrics.parquet; wait for a
# running fetch to finish rather than read a half-written file.
for _ in $(seq 1 180); do
  grep -q "^DONE" outputs/logs/fetch_anthropometrics.log 2>/dev/null && break
  pgrep -f fetch_anthropometrics.py >/dev/null || break
  sleep 10
done

run() {
  local stage=$1 preset=$2 log=$LOG/${stage}_${preset}.log
  echo "$(date '+%Y-%m-%d %H:%M') START $stage [$preset]" >> "$LOG/chain.log"
  DEEP_RAPM_DATA=$preset conda run -n supercluster --cwd "$PWD" --no-capture-output \
    python -u -m "experiments.$stage" > "$log" 2>&1
  local rc=$?
  # conda run returns 0 even when the script dies; trust the log instead.
  grep -q "^Traceback\|ERROR conda" "$log" && rc=FAILED
  echo "$(date '+%Y-%m-%d %H:%M') END   $stage [$preset] rc=$rc" >> "$LOG/chain.log"
}

for stage in $STAGES; do
  for preset in $PRESETS; do
    run "$stage" "$preset"
  done
done
echo "$(date '+%Y-%m-%d %H:%M') CHAIN DONE" >> "$LOG/chain.log"
