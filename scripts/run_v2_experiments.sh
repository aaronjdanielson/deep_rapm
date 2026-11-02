#!/usr/bin/env bash
# Rerun the RAPM v2 experiment plan (docs/rapm_v2_design.md §6) in two lanes.
#
#   caffeinate -ims bash scripts/run_v2_experiments.sh            # both lanes
#   LANES=A bash scripts/run_v2_experiments.sh                    # one lane
#
# Lane A: forward-objective fits (v2, v2.1, ablations A/B/C, κ sweep).
# Lane B: v1 / v1_home scoring with the predictive density, the evidence fits,
#         then the ridge-family variants.
# Every step runs whatever happened to the previous one; outcomes go to
# outputs/logs/v2_experiments/status.tsv (step, exit code, seconds, last line).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=outputs/logs/v2_experiments
mkdir -p "$LOG"
STATUS="$LOG/status.tsv"
THREADS=${THREADS:-5}
export OMP_NUM_THREADS=$THREADS VECLIB_MAXIMUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS

step() {   # step <name> <command...>
    local name=$1; shift
    local t0=$(date +%s)
    "$@" > "$LOG/$name.log" 2>&1
    local rc=$?
    local last=$(grep -v -i warning "$LOG/$name.log" | grep -v '^\s*$' | tail -1 | cut -c1-200)
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$rc" "$(( $(date +%s) - t0 ))" "$last" >> "$STATUS"
}

lane_a() {
    step fit_v2          $PY scripts/fit_v2.py --name v2        --out outputs/v2
    step fit_v2.1        $PY scripts/fit_v2.py --name v2.1      --out outputs/v2_1      --shrink 500
    step fit_v2.1-A      $PY scripts/fit_v2.py --name v2.1-A    --out outputs/v2_1A     --shrink 500 --no-learn-precision --no-learn-alpha
    step fit_v2.1-B      $PY scripts/fit_v2.py --name v2.1-B    --out outputs/v2_1B     --shrink 500 --no-learn-mean --no-learn-alpha
    step fit_v2.1-C      $PY scripts/fit_v2.py --name v2.1-C    --out outputs/v2_1C     --shrink 500 --no-learn-alpha
    step fit_v2.1-k200   $PY scripts/fit_v2.py --name v2.1-k200 --out outputs/v2_1k200  --shrink 200
    step fit_v2.1-k1000  $PY scripts/fit_v2.py --name v2.1-k1000 --out outputs/v2_1k1000 --shrink 1000
}

lane_b() {
    step score_v1        $PY scripts/score_forward.py --variants v1
    step score_v1_home   $PY scripts/score_forward.py --variants v1_home
    step fit_v2eb        $PY scripts/fit_v2.py --name v2eb       --out outputs/v2eb      --objective evidence
    step fit_v2.1eb      $PY scripts/fit_v2.py --name v2.1eb     --out outputs/v2_1eb    --objective evidence --shrink 500
    step fit_v2.1eb-ref  $PY scripts/fit_v2.py --name v2.1eb-ref --out outputs/v2_1eb_ref --objective evidence --shrink 500 --evidence-noise reference
    step score_variants  $PY scripts/score_forward.py --variants comp_off comp_p90 hl730 hl180 offseason_0.5 no_preseason
}

printf '%s\tSTART lanes=%s threads=%s\n' "$(date '+%F %T')" "${LANES:-AB}" "$THREADS" >> "$STATUS"
case "${LANES:-AB}" in
    A)  lane_a ;;
    B)  lane_b ;;
    *)  lane_a & lane_b & wait ;;
esac
printf '%s\tALL_DONE\n' "$(date '+%F %T')" >> "$STATUS"
