#!/bin/bash
# Backfill whole seasons through the nba_api extractor, repair flagged games
# in the seasons we already have, extend the player tables, and validate
# everything against official box scores.
#
#     caffeinate -ims bash scripts/backfill_seasons.sh 2017-18 2016-17 2015-16 2014-15 2013-14
#     REPAIR_SEASONS="" bash scripts/backfill_seasons.sh 2012-13      # skip the repair pass
#
# Stages (each logs to outputs/logs/backfill/<stage>.log, one line per stage
# in chain.log; a stage whose log contains a traceback is marked FAILED but
# the chain continues):
#   repair_{fetch,parse,swap}  games flagged severity=error in
#                              data/quality/game_quality.parquet (run
#                              scripts/validate_data.py first)
#   fetch / parse / swap       the new seasons, from the official schedule
#   extend_players             rosters + vocab (additive) for the new seasons
#   anthropometrics            height/weight for the new vocab players
#   validate                   scripts/validate_data.py --reconcile, all seasons
cd "$(dirname "$0")/.." || exit 1
export PYTHONWARNINGS=ignore
LOG=outputs/logs/backfill
mkdir -p "$LOG"
NEW="$*"
if [ -z "${REPAIR_SEASONS+x}" ]; then
  REPAIR_SEASONS=$(python3.11 -c "
import pandas as pd
q = pd.read_parquet('data/quality/game_quality.parquet')
print(' '.join(sorted(q[q.severity == 'error'].season.unique())))")
fi

stage() {
  local name=$1; shift
  echo "$(date '+%Y-%m-%d %H:%M') START $name" >> "$LOG/chain.log"
  "$@" > "$LOG/$name.log" 2>&1
  local rc=$?
  grep -q "^Traceback" "$LOG/$name.log" && rc=FAILED
  echo "$(date '+%Y-%m-%d %H:%M') END   $name rc=$rc" >> "$LOG/chain.log"
}

if [ -n "$REPAIR_SEASONS" ]; then
  stage repair_fetch python3.11 -u scripts/recollect_nba_api.py fetch --mode repair --seasons $REPAIR_SEASONS
  stage repair_parse python3.11 -u scripts/recollect_nba_api.py parse --mode repair --seasons $REPAIR_SEASONS
  stage repair_swap  python3.11 -u scripts/recollect_nba_api.py swap  --seasons $REPAIR_SEASONS
fi
if [ -n "$NEW" ]; then
  stage fetch           python3.11 -u scripts/recollect_nba_api.py fetch --seasons $NEW
  stage parse           python3.11 -u scripts/recollect_nba_api.py parse --seasons $NEW
  stage swap            python3.11 -u scripts/recollect_nba_api.py swap  --seasons $NEW
  stage extend_players  python3.11 -u scripts/extend_players.py $NEW
  stage anthropometrics python3.11 -u scripts/fetch_anthropometrics.py
fi
stage validate python3.11 -u scripts/validate_data.py --reconcile
echo "$(date '+%Y-%m-%d %H:%M') BACKFILL DONE" >> "$LOG/chain.log"
