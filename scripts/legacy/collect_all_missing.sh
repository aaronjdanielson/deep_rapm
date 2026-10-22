#!/bin/bash
# Collect missing regular-season and playoff games across all seasons.
# Skips games already saved to data/games/.
# Total: 1456 games

OUTDIR="/Users/aarondanielson/Dropbox/deep_rapm/data"

# ── 2018-19 regular (383 games) ────────────────────────────
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800001.parquet" ]; then
  echo "[1/1456]  skip 0021800001 (already collected)"
else
  echo "[1/1456]  2018-19 regular  0021800001"
  collect-possessions --game 0021800001 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800002.parquet" ]; then
  echo "[2/1456]  skip 0021800002 (already collected)"
else
  echo "[2/1456]  2018-19 regular  0021800002"
  collect-possessions --game 0021800002 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800003.parquet" ]; then
  echo "[3/1456]  skip 0021800003 (already collected)"
else
  echo "[3/1456]  2018-19 regular  0021800003"
  collect-possessions --game 0021800003 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800004.parquet" ]; then
  echo "[4/1456]  skip 0021800004 (already collected)"
else
  echo "[4/1456]  2018-19 regular  0021800004"
  collect-possessions --game 0021800004 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800005.parquet" ]; then
  echo "[5/1456]  skip 0021800005 (already collected)"
else
  echo "[5/1456]  2018-19 regular  0021800005"
  collect-possessions --game 0021800005 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800006.parquet" ]; then
  echo "[6/1456]  skip 0021800006 (already collected)"
else
  echo "[6/1456]  2018-19 regular  0021800006"
  collect-possessions --game 0021800006 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800007.parquet" ]; then
  echo "[7/1456]  skip 0021800007 (already collected)"
else
  echo "[7/1456]  2018-19 regular  0021800007"
  collect-possessions --game 0021800007 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800008.parquet" ]; then
  echo "[8/1456]  skip 0021800008 (already collected)"
else
  echo "[8/1456]  2018-19 regular  0021800008"
  collect-possessions --game 0021800008 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800009.parquet" ]; then
  echo "[9/1456]  skip 0021800009 (already collected)"
else
  echo "[9/1456]  2018-19 regular  0021800009"
  collect-possessions --game 0021800009 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800010.parquet" ]; then
  echo "[10/1456]  skip 0021800010 (already collected)"
else
  echo "[10/1456]  2018-19 regular  0021800010"
  collect-possessions --game 0021800010 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800011.parquet" ]; then
  echo "[11/1456]  skip 0021800011 (already collected)"
else
  echo "[11/1456]  2018-19 regular  0021800011"
  collect-possessions --game 0021800011 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800012.parquet" ]; then
  echo "[12/1456]  skip 0021800012 (already collected)"
else
  echo "[12/1456]  2018-19 regular  0021800012"
  collect-possessions --game 0021800012 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800013.parquet" ]; then
  echo "[13/1456]  skip 0021800013 (already collected)"
else
  echo "[13/1456]  2018-19 regular  0021800013"
  collect-possessions --game 0021800013 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800014.parquet" ]; then
  echo "[14/1456]  skip 0021800014 (already collected)"
else
  echo "[14/1456]  2018-19 regular  0021800014"
  collect-possessions --game 0021800014 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800015.parquet" ]; then
  echo "[15/1456]  skip 0021800015 (already collected)"
else
  echo "[15/1456]  2018-19 regular  0021800015"
  collect-possessions --game 0021800015 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800016.parquet" ]; then
  echo "[16/1456]  skip 0021800016 (already collected)"
else
  echo "[16/1456]  2018-19 regular  0021800016"
  collect-possessions --game 0021800016 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800017.parquet" ]; then
  echo "[17/1456]  skip 0021800017 (already collected)"
else
  echo "[17/1456]  2018-19 regular  0021800017"
  collect-possessions --game 0021800017 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800018.parquet" ]; then
  echo "[18/1456]  skip 0021800018 (already collected)"
else
  echo "[18/1456]  2018-19 regular  0021800018"
  collect-possessions --game 0021800018 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800019.parquet" ]; then
  echo "[19/1456]  skip 0021800019 (already collected)"
else
  echo "[19/1456]  2018-19 regular  0021800019"
  collect-possessions --game 0021800019 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800020.parquet" ]; then
  echo "[20/1456]  skip 0021800020 (already collected)"
else
  echo "[20/1456]  2018-19 regular  0021800020"
  collect-possessions --game 0021800020 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800021.parquet" ]; then
  echo "[21/1456]  skip 0021800021 (already collected)"
else
  echo "[21/1456]  2018-19 regular  0021800021"
  collect-possessions --game 0021800021 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800022.parquet" ]; then
  echo "[22/1456]  skip 0021800022 (already collected)"
else
  echo "[22/1456]  2018-19 regular  0021800022"
  collect-possessions --game 0021800022 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800023.parquet" ]; then
  echo "[23/1456]  skip 0021800023 (already collected)"
else
  echo "[23/1456]  2018-19 regular  0021800023"
  collect-possessions --game 0021800023 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800024.parquet" ]; then
  echo "[24/1456]  skip 0021800024 (already collected)"
else
  echo "[24/1456]  2018-19 regular  0021800024"
  collect-possessions --game 0021800024 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800025.parquet" ]; then
  echo "[25/1456]  skip 0021800025 (already collected)"
else
  echo "[25/1456]  2018-19 regular  0021800025"
  collect-possessions --game 0021800025 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800026.parquet" ]; then
  echo "[26/1456]  skip 0021800026 (already collected)"
else
  echo "[26/1456]  2018-19 regular  0021800026"
  collect-possessions --game 0021800026 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800027.parquet" ]; then
  echo "[27/1456]  skip 0021800027 (already collected)"
else
  echo "[27/1456]  2018-19 regular  0021800027"
  collect-possessions --game 0021800027 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800028.parquet" ]; then
  echo "[28/1456]  skip 0021800028 (already collected)"
else
  echo "[28/1456]  2018-19 regular  0021800028"
  collect-possessions --game 0021800028 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800029.parquet" ]; then
  echo "[29/1456]  skip 0021800029 (already collected)"
else
  echo "[29/1456]  2018-19 regular  0021800029"
  collect-possessions --game 0021800029 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800030.parquet" ]; then
  echo "[30/1456]  skip 0021800030 (already collected)"
else
  echo "[30/1456]  2018-19 regular  0021800030"
  collect-possessions --game 0021800030 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800031.parquet" ]; then
  echo "[31/1456]  skip 0021800031 (already collected)"
else
  echo "[31/1456]  2018-19 regular  0021800031"
  collect-possessions --game 0021800031 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800032.parquet" ]; then
  echo "[32/1456]  skip 0021800032 (already collected)"
else
  echo "[32/1456]  2018-19 regular  0021800032"
  collect-possessions --game 0021800032 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800033.parquet" ]; then
  echo "[33/1456]  skip 0021800033 (already collected)"
else
  echo "[33/1456]  2018-19 regular  0021800033"
  collect-possessions --game 0021800033 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800034.parquet" ]; then
  echo "[34/1456]  skip 0021800034 (already collected)"
else
  echo "[34/1456]  2018-19 regular  0021800034"
  collect-possessions --game 0021800034 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800035.parquet" ]; then
  echo "[35/1456]  skip 0021800035 (already collected)"
else
  echo "[35/1456]  2018-19 regular  0021800035"
  collect-possessions --game 0021800035 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800036.parquet" ]; then
  echo "[36/1456]  skip 0021800036 (already collected)"
else
  echo "[36/1456]  2018-19 regular  0021800036"
  collect-possessions --game 0021800036 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800037.parquet" ]; then
  echo "[37/1456]  skip 0021800037 (already collected)"
else
  echo "[37/1456]  2018-19 regular  0021800037"
  collect-possessions --game 0021800037 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800038.parquet" ]; then
  echo "[38/1456]  skip 0021800038 (already collected)"
else
  echo "[38/1456]  2018-19 regular  0021800038"
  collect-possessions --game 0021800038 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800039.parquet" ]; then
  echo "[39/1456]  skip 0021800039 (already collected)"
else
  echo "[39/1456]  2018-19 regular  0021800039"
  collect-possessions --game 0021800039 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800040.parquet" ]; then
  echo "[40/1456]  skip 0021800040 (already collected)"
else
  echo "[40/1456]  2018-19 regular  0021800040"
  collect-possessions --game 0021800040 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800041.parquet" ]; then
  echo "[41/1456]  skip 0021800041 (already collected)"
else
  echo "[41/1456]  2018-19 regular  0021800041"
  collect-possessions --game 0021800041 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800042.parquet" ]; then
  echo "[42/1456]  skip 0021800042 (already collected)"
else
  echo "[42/1456]  2018-19 regular  0021800042"
  collect-possessions --game 0021800042 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800043.parquet" ]; then
  echo "[43/1456]  skip 0021800043 (already collected)"
else
  echo "[43/1456]  2018-19 regular  0021800043"
  collect-possessions --game 0021800043 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800044.parquet" ]; then
  echo "[44/1456]  skip 0021800044 (already collected)"
else
  echo "[44/1456]  2018-19 regular  0021800044"
  collect-possessions --game 0021800044 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800045.parquet" ]; then
  echo "[45/1456]  skip 0021800045 (already collected)"
else
  echo "[45/1456]  2018-19 regular  0021800045"
  collect-possessions --game 0021800045 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800046.parquet" ]; then
  echo "[46/1456]  skip 0021800046 (already collected)"
else
  echo "[46/1456]  2018-19 regular  0021800046"
  collect-possessions --game 0021800046 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800047.parquet" ]; then
  echo "[47/1456]  skip 0021800047 (already collected)"
else
  echo "[47/1456]  2018-19 regular  0021800047"
  collect-possessions --game 0021800047 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800048.parquet" ]; then
  echo "[48/1456]  skip 0021800048 (already collected)"
else
  echo "[48/1456]  2018-19 regular  0021800048"
  collect-possessions --game 0021800048 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800049.parquet" ]; then
  echo "[49/1456]  skip 0021800049 (already collected)"
else
  echo "[49/1456]  2018-19 regular  0021800049"
  collect-possessions --game 0021800049 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800050.parquet" ]; then
  echo "[50/1456]  skip 0021800050 (already collected)"
else
  echo "[50/1456]  2018-19 regular  0021800050"
  collect-possessions --game 0021800050 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800051.parquet" ]; then
  echo "[51/1456]  skip 0021800051 (already collected)"
else
  echo "[51/1456]  2018-19 regular  0021800051"
  collect-possessions --game 0021800051 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800052.parquet" ]; then
  echo "[52/1456]  skip 0021800052 (already collected)"
else
  echo "[52/1456]  2018-19 regular  0021800052"
  collect-possessions --game 0021800052 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800053.parquet" ]; then
  echo "[53/1456]  skip 0021800053 (already collected)"
else
  echo "[53/1456]  2018-19 regular  0021800053"
  collect-possessions --game 0021800053 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800054.parquet" ]; then
  echo "[54/1456]  skip 0021800054 (already collected)"
else
  echo "[54/1456]  2018-19 regular  0021800054"
  collect-possessions --game 0021800054 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800055.parquet" ]; then
  echo "[55/1456]  skip 0021800055 (already collected)"
else
  echo "[55/1456]  2018-19 regular  0021800055"
  collect-possessions --game 0021800055 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800056.parquet" ]; then
  echo "[56/1456]  skip 0021800056 (already collected)"
else
  echo "[56/1456]  2018-19 regular  0021800056"
  collect-possessions --game 0021800056 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800057.parquet" ]; then
  echo "[57/1456]  skip 0021800057 (already collected)"
else
  echo "[57/1456]  2018-19 regular  0021800057"
  collect-possessions --game 0021800057 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800058.parquet" ]; then
  echo "[58/1456]  skip 0021800058 (already collected)"
else
  echo "[58/1456]  2018-19 regular  0021800058"
  collect-possessions --game 0021800058 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800059.parquet" ]; then
  echo "[59/1456]  skip 0021800059 (already collected)"
else
  echo "[59/1456]  2018-19 regular  0021800059"
  collect-possessions --game 0021800059 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800060.parquet" ]; then
  echo "[60/1456]  skip 0021800060 (already collected)"
else
  echo "[60/1456]  2018-19 regular  0021800060"
  collect-possessions --game 0021800060 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800061.parquet" ]; then
  echo "[61/1456]  skip 0021800061 (already collected)"
else
  echo "[61/1456]  2018-19 regular  0021800061"
  collect-possessions --game 0021800061 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800062.parquet" ]; then
  echo "[62/1456]  skip 0021800062 (already collected)"
else
  echo "[62/1456]  2018-19 regular  0021800062"
  collect-possessions --game 0021800062 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800063.parquet" ]; then
  echo "[63/1456]  skip 0021800063 (already collected)"
else
  echo "[63/1456]  2018-19 regular  0021800063"
  collect-possessions --game 0021800063 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800064.parquet" ]; then
  echo "[64/1456]  skip 0021800064 (already collected)"
else
  echo "[64/1456]  2018-19 regular  0021800064"
  collect-possessions --game 0021800064 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800065.parquet" ]; then
  echo "[65/1456]  skip 0021800065 (already collected)"
else
  echo "[65/1456]  2018-19 regular  0021800065"
  collect-possessions --game 0021800065 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800066.parquet" ]; then
  echo "[66/1456]  skip 0021800066 (already collected)"
else
  echo "[66/1456]  2018-19 regular  0021800066"
  collect-possessions --game 0021800066 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800067.parquet" ]; then
  echo "[67/1456]  skip 0021800067 (already collected)"
else
  echo "[67/1456]  2018-19 regular  0021800067"
  collect-possessions --game 0021800067 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800068.parquet" ]; then
  echo "[68/1456]  skip 0021800068 (already collected)"
else
  echo "[68/1456]  2018-19 regular  0021800068"
  collect-possessions --game 0021800068 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800069.parquet" ]; then
  echo "[69/1456]  skip 0021800069 (already collected)"
else
  echo "[69/1456]  2018-19 regular  0021800069"
  collect-possessions --game 0021800069 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800070.parquet" ]; then
  echo "[70/1456]  skip 0021800070 (already collected)"
else
  echo "[70/1456]  2018-19 regular  0021800070"
  collect-possessions --game 0021800070 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800071.parquet" ]; then
  echo "[71/1456]  skip 0021800071 (already collected)"
else
  echo "[71/1456]  2018-19 regular  0021800071"
  collect-possessions --game 0021800071 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800072.parquet" ]; then
  echo "[72/1456]  skip 0021800072 (already collected)"
else
  echo "[72/1456]  2018-19 regular  0021800072"
  collect-possessions --game 0021800072 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800073.parquet" ]; then
  echo "[73/1456]  skip 0021800073 (already collected)"
else
  echo "[73/1456]  2018-19 regular  0021800073"
  collect-possessions --game 0021800073 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800074.parquet" ]; then
  echo "[74/1456]  skip 0021800074 (already collected)"
else
  echo "[74/1456]  2018-19 regular  0021800074"
  collect-possessions --game 0021800074 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800075.parquet" ]; then
  echo "[75/1456]  skip 0021800075 (already collected)"
else
  echo "[75/1456]  2018-19 regular  0021800075"
  collect-possessions --game 0021800075 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800076.parquet" ]; then
  echo "[76/1456]  skip 0021800076 (already collected)"
else
  echo "[76/1456]  2018-19 regular  0021800076"
  collect-possessions --game 0021800076 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800077.parquet" ]; then
  echo "[77/1456]  skip 0021800077 (already collected)"
else
  echo "[77/1456]  2018-19 regular  0021800077"
  collect-possessions --game 0021800077 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800078.parquet" ]; then
  echo "[78/1456]  skip 0021800078 (already collected)"
else
  echo "[78/1456]  2018-19 regular  0021800078"
  collect-possessions --game 0021800078 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800079.parquet" ]; then
  echo "[79/1456]  skip 0021800079 (already collected)"
else
  echo "[79/1456]  2018-19 regular  0021800079"
  collect-possessions --game 0021800079 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800080.parquet" ]; then
  echo "[80/1456]  skip 0021800080 (already collected)"
else
  echo "[80/1456]  2018-19 regular  0021800080"
  collect-possessions --game 0021800080 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800081.parquet" ]; then
  echo "[81/1456]  skip 0021800081 (already collected)"
else
  echo "[81/1456]  2018-19 regular  0021800081"
  collect-possessions --game 0021800081 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800082.parquet" ]; then
  echo "[82/1456]  skip 0021800082 (already collected)"
else
  echo "[82/1456]  2018-19 regular  0021800082"
  collect-possessions --game 0021800082 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800083.parquet" ]; then
  echo "[83/1456]  skip 0021800083 (already collected)"
else
  echo "[83/1456]  2018-19 regular  0021800083"
  collect-possessions --game 0021800083 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800084.parquet" ]; then
  echo "[84/1456]  skip 0021800084 (already collected)"
else
  echo "[84/1456]  2018-19 regular  0021800084"
  collect-possessions --game 0021800084 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800085.parquet" ]; then
  echo "[85/1456]  skip 0021800085 (already collected)"
else
  echo "[85/1456]  2018-19 regular  0021800085"
  collect-possessions --game 0021800085 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800086.parquet" ]; then
  echo "[86/1456]  skip 0021800086 (already collected)"
else
  echo "[86/1456]  2018-19 regular  0021800086"
  collect-possessions --game 0021800086 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800087.parquet" ]; then
  echo "[87/1456]  skip 0021800087 (already collected)"
else
  echo "[87/1456]  2018-19 regular  0021800087"
  collect-possessions --game 0021800087 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800088.parquet" ]; then
  echo "[88/1456]  skip 0021800088 (already collected)"
else
  echo "[88/1456]  2018-19 regular  0021800088"
  collect-possessions --game 0021800088 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800089.parquet" ]; then
  echo "[89/1456]  skip 0021800089 (already collected)"
else
  echo "[89/1456]  2018-19 regular  0021800089"
  collect-possessions --game 0021800089 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800090.parquet" ]; then
  echo "[90/1456]  skip 0021800090 (already collected)"
else
  echo "[90/1456]  2018-19 regular  0021800090"
  collect-possessions --game 0021800090 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800091.parquet" ]; then
  echo "[91/1456]  skip 0021800091 (already collected)"
else
  echo "[91/1456]  2018-19 regular  0021800091"
  collect-possessions --game 0021800091 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800092.parquet" ]; then
  echo "[92/1456]  skip 0021800092 (already collected)"
else
  echo "[92/1456]  2018-19 regular  0021800092"
  collect-possessions --game 0021800092 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800093.parquet" ]; then
  echo "[93/1456]  skip 0021800093 (already collected)"
else
  echo "[93/1456]  2018-19 regular  0021800093"
  collect-possessions --game 0021800093 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800094.parquet" ]; then
  echo "[94/1456]  skip 0021800094 (already collected)"
else
  echo "[94/1456]  2018-19 regular  0021800094"
  collect-possessions --game 0021800094 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800095.parquet" ]; then
  echo "[95/1456]  skip 0021800095 (already collected)"
else
  echo "[95/1456]  2018-19 regular  0021800095"
  collect-possessions --game 0021800095 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800096.parquet" ]; then
  echo "[96/1456]  skip 0021800096 (already collected)"
else
  echo "[96/1456]  2018-19 regular  0021800096"
  collect-possessions --game 0021800096 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800097.parquet" ]; then
  echo "[97/1456]  skip 0021800097 (already collected)"
else
  echo "[97/1456]  2018-19 regular  0021800097"
  collect-possessions --game 0021800097 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800098.parquet" ]; then
  echo "[98/1456]  skip 0021800098 (already collected)"
else
  echo "[98/1456]  2018-19 regular  0021800098"
  collect-possessions --game 0021800098 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800099.parquet" ]; then
  echo "[99/1456]  skip 0021800099 (already collected)"
else
  echo "[99/1456]  2018-19 regular  0021800099"
  collect-possessions --game 0021800099 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800100.parquet" ]; then
  echo "[100/1456]  skip 0021800100 (already collected)"
else
  echo "[100/1456]  2018-19 regular  0021800100"
  collect-possessions --game 0021800100 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800101.parquet" ]; then
  echo "[101/1456]  skip 0021800101 (already collected)"
else
  echo "[101/1456]  2018-19 regular  0021800101"
  collect-possessions --game 0021800101 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800102.parquet" ]; then
  echo "[102/1456]  skip 0021800102 (already collected)"
else
  echo "[102/1456]  2018-19 regular  0021800102"
  collect-possessions --game 0021800102 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800103.parquet" ]; then
  echo "[103/1456]  skip 0021800103 (already collected)"
else
  echo "[103/1456]  2018-19 regular  0021800103"
  collect-possessions --game 0021800103 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800104.parquet" ]; then
  echo "[104/1456]  skip 0021800104 (already collected)"
else
  echo "[104/1456]  2018-19 regular  0021800104"
  collect-possessions --game 0021800104 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800105.parquet" ]; then
  echo "[105/1456]  skip 0021800105 (already collected)"
else
  echo "[105/1456]  2018-19 regular  0021800105"
  collect-possessions --game 0021800105 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800106.parquet" ]; then
  echo "[106/1456]  skip 0021800106 (already collected)"
else
  echo "[106/1456]  2018-19 regular  0021800106"
  collect-possessions --game 0021800106 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800107.parquet" ]; then
  echo "[107/1456]  skip 0021800107 (already collected)"
else
  echo "[107/1456]  2018-19 regular  0021800107"
  collect-possessions --game 0021800107 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800108.parquet" ]; then
  echo "[108/1456]  skip 0021800108 (already collected)"
else
  echo "[108/1456]  2018-19 regular  0021800108"
  collect-possessions --game 0021800108 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800109.parquet" ]; then
  echo "[109/1456]  skip 0021800109 (already collected)"
else
  echo "[109/1456]  2018-19 regular  0021800109"
  collect-possessions --game 0021800109 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800110.parquet" ]; then
  echo "[110/1456]  skip 0021800110 (already collected)"
else
  echo "[110/1456]  2018-19 regular  0021800110"
  collect-possessions --game 0021800110 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800111.parquet" ]; then
  echo "[111/1456]  skip 0021800111 (already collected)"
else
  echo "[111/1456]  2018-19 regular  0021800111"
  collect-possessions --game 0021800111 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800112.parquet" ]; then
  echo "[112/1456]  skip 0021800112 (already collected)"
else
  echo "[112/1456]  2018-19 regular  0021800112"
  collect-possessions --game 0021800112 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800113.parquet" ]; then
  echo "[113/1456]  skip 0021800113 (already collected)"
else
  echo "[113/1456]  2018-19 regular  0021800113"
  collect-possessions --game 0021800113 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800114.parquet" ]; then
  echo "[114/1456]  skip 0021800114 (already collected)"
else
  echo "[114/1456]  2018-19 regular  0021800114"
  collect-possessions --game 0021800114 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800115.parquet" ]; then
  echo "[115/1456]  skip 0021800115 (already collected)"
else
  echo "[115/1456]  2018-19 regular  0021800115"
  collect-possessions --game 0021800115 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800116.parquet" ]; then
  echo "[116/1456]  skip 0021800116 (already collected)"
else
  echo "[116/1456]  2018-19 regular  0021800116"
  collect-possessions --game 0021800116 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800117.parquet" ]; then
  echo "[117/1456]  skip 0021800117 (already collected)"
else
  echo "[117/1456]  2018-19 regular  0021800117"
  collect-possessions --game 0021800117 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800118.parquet" ]; then
  echo "[118/1456]  skip 0021800118 (already collected)"
else
  echo "[118/1456]  2018-19 regular  0021800118"
  collect-possessions --game 0021800118 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800119.parquet" ]; then
  echo "[119/1456]  skip 0021800119 (already collected)"
else
  echo "[119/1456]  2018-19 regular  0021800119"
  collect-possessions --game 0021800119 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800120.parquet" ]; then
  echo "[120/1456]  skip 0021800120 (already collected)"
else
  echo "[120/1456]  2018-19 regular  0021800120"
  collect-possessions --game 0021800120 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800121.parquet" ]; then
  echo "[121/1456]  skip 0021800121 (already collected)"
else
  echo "[121/1456]  2018-19 regular  0021800121"
  collect-possessions --game 0021800121 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800122.parquet" ]; then
  echo "[122/1456]  skip 0021800122 (already collected)"
else
  echo "[122/1456]  2018-19 regular  0021800122"
  collect-possessions --game 0021800122 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800123.parquet" ]; then
  echo "[123/1456]  skip 0021800123 (already collected)"
else
  echo "[123/1456]  2018-19 regular  0021800123"
  collect-possessions --game 0021800123 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800124.parquet" ]; then
  echo "[124/1456]  skip 0021800124 (already collected)"
else
  echo "[124/1456]  2018-19 regular  0021800124"
  collect-possessions --game 0021800124 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800125.parquet" ]; then
  echo "[125/1456]  skip 0021800125 (already collected)"
else
  echo "[125/1456]  2018-19 regular  0021800125"
  collect-possessions --game 0021800125 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800126.parquet" ]; then
  echo "[126/1456]  skip 0021800126 (already collected)"
else
  echo "[126/1456]  2018-19 regular  0021800126"
  collect-possessions --game 0021800126 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800127.parquet" ]; then
  echo "[127/1456]  skip 0021800127 (already collected)"
else
  echo "[127/1456]  2018-19 regular  0021800127"
  collect-possessions --game 0021800127 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800128.parquet" ]; then
  echo "[128/1456]  skip 0021800128 (already collected)"
else
  echo "[128/1456]  2018-19 regular  0021800128"
  collect-possessions --game 0021800128 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800129.parquet" ]; then
  echo "[129/1456]  skip 0021800129 (already collected)"
else
  echo "[129/1456]  2018-19 regular  0021800129"
  collect-possessions --game 0021800129 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800130.parquet" ]; then
  echo "[130/1456]  skip 0021800130 (already collected)"
else
  echo "[130/1456]  2018-19 regular  0021800130"
  collect-possessions --game 0021800130 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800131.parquet" ]; then
  echo "[131/1456]  skip 0021800131 (already collected)"
else
  echo "[131/1456]  2018-19 regular  0021800131"
  collect-possessions --game 0021800131 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800132.parquet" ]; then
  echo "[132/1456]  skip 0021800132 (already collected)"
else
  echo "[132/1456]  2018-19 regular  0021800132"
  collect-possessions --game 0021800132 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800133.parquet" ]; then
  echo "[133/1456]  skip 0021800133 (already collected)"
else
  echo "[133/1456]  2018-19 regular  0021800133"
  collect-possessions --game 0021800133 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800134.parquet" ]; then
  echo "[134/1456]  skip 0021800134 (already collected)"
else
  echo "[134/1456]  2018-19 regular  0021800134"
  collect-possessions --game 0021800134 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800135.parquet" ]; then
  echo "[135/1456]  skip 0021800135 (already collected)"
else
  echo "[135/1456]  2018-19 regular  0021800135"
  collect-possessions --game 0021800135 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800136.parquet" ]; then
  echo "[136/1456]  skip 0021800136 (already collected)"
else
  echo "[136/1456]  2018-19 regular  0021800136"
  collect-possessions --game 0021800136 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800137.parquet" ]; then
  echo "[137/1456]  skip 0021800137 (already collected)"
else
  echo "[137/1456]  2018-19 regular  0021800137"
  collect-possessions --game 0021800137 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800138.parquet" ]; then
  echo "[138/1456]  skip 0021800138 (already collected)"
else
  echo "[138/1456]  2018-19 regular  0021800138"
  collect-possessions --game 0021800138 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800139.parquet" ]; then
  echo "[139/1456]  skip 0021800139 (already collected)"
else
  echo "[139/1456]  2018-19 regular  0021800139"
  collect-possessions --game 0021800139 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800140.parquet" ]; then
  echo "[140/1456]  skip 0021800140 (already collected)"
else
  echo "[140/1456]  2018-19 regular  0021800140"
  collect-possessions --game 0021800140 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800141.parquet" ]; then
  echo "[141/1456]  skip 0021800141 (already collected)"
else
  echo "[141/1456]  2018-19 regular  0021800141"
  collect-possessions --game 0021800141 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800142.parquet" ]; then
  echo "[142/1456]  skip 0021800142 (already collected)"
else
  echo "[142/1456]  2018-19 regular  0021800142"
  collect-possessions --game 0021800142 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800143.parquet" ]; then
  echo "[143/1456]  skip 0021800143 (already collected)"
else
  echo "[143/1456]  2018-19 regular  0021800143"
  collect-possessions --game 0021800143 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800144.parquet" ]; then
  echo "[144/1456]  skip 0021800144 (already collected)"
else
  echo "[144/1456]  2018-19 regular  0021800144"
  collect-possessions --game 0021800144 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800145.parquet" ]; then
  echo "[145/1456]  skip 0021800145 (already collected)"
else
  echo "[145/1456]  2018-19 regular  0021800145"
  collect-possessions --game 0021800145 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800146.parquet" ]; then
  echo "[146/1456]  skip 0021800146 (already collected)"
else
  echo "[146/1456]  2018-19 regular  0021800146"
  collect-possessions --game 0021800146 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800147.parquet" ]; then
  echo "[147/1456]  skip 0021800147 (already collected)"
else
  echo "[147/1456]  2018-19 regular  0021800147"
  collect-possessions --game 0021800147 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800148.parquet" ]; then
  echo "[148/1456]  skip 0021800148 (already collected)"
else
  echo "[148/1456]  2018-19 regular  0021800148"
  collect-possessions --game 0021800148 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800149.parquet" ]; then
  echo "[149/1456]  skip 0021800149 (already collected)"
else
  echo "[149/1456]  2018-19 regular  0021800149"
  collect-possessions --game 0021800149 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800150.parquet" ]; then
  echo "[150/1456]  skip 0021800150 (already collected)"
else
  echo "[150/1456]  2018-19 regular  0021800150"
  collect-possessions --game 0021800150 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800151.parquet" ]; then
  echo "[151/1456]  skip 0021800151 (already collected)"
else
  echo "[151/1456]  2018-19 regular  0021800151"
  collect-possessions --game 0021800151 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800152.parquet" ]; then
  echo "[152/1456]  skip 0021800152 (already collected)"
else
  echo "[152/1456]  2018-19 regular  0021800152"
  collect-possessions --game 0021800152 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800153.parquet" ]; then
  echo "[153/1456]  skip 0021800153 (already collected)"
else
  echo "[153/1456]  2018-19 regular  0021800153"
  collect-possessions --game 0021800153 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800154.parquet" ]; then
  echo "[154/1456]  skip 0021800154 (already collected)"
else
  echo "[154/1456]  2018-19 regular  0021800154"
  collect-possessions --game 0021800154 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800155.parquet" ]; then
  echo "[155/1456]  skip 0021800155 (already collected)"
else
  echo "[155/1456]  2018-19 regular  0021800155"
  collect-possessions --game 0021800155 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800156.parquet" ]; then
  echo "[156/1456]  skip 0021800156 (already collected)"
else
  echo "[156/1456]  2018-19 regular  0021800156"
  collect-possessions --game 0021800156 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800157.parquet" ]; then
  echo "[157/1456]  skip 0021800157 (already collected)"
else
  echo "[157/1456]  2018-19 regular  0021800157"
  collect-possessions --game 0021800157 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800158.parquet" ]; then
  echo "[158/1456]  skip 0021800158 (already collected)"
else
  echo "[158/1456]  2018-19 regular  0021800158"
  collect-possessions --game 0021800158 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800159.parquet" ]; then
  echo "[159/1456]  skip 0021800159 (already collected)"
else
  echo "[159/1456]  2018-19 regular  0021800159"
  collect-possessions --game 0021800159 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800160.parquet" ]; then
  echo "[160/1456]  skip 0021800160 (already collected)"
else
  echo "[160/1456]  2018-19 regular  0021800160"
  collect-possessions --game 0021800160 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800161.parquet" ]; then
  echo "[161/1456]  skip 0021800161 (already collected)"
else
  echo "[161/1456]  2018-19 regular  0021800161"
  collect-possessions --game 0021800161 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800162.parquet" ]; then
  echo "[162/1456]  skip 0021800162 (already collected)"
else
  echo "[162/1456]  2018-19 regular  0021800162"
  collect-possessions --game 0021800162 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800163.parquet" ]; then
  echo "[163/1456]  skip 0021800163 (already collected)"
else
  echo "[163/1456]  2018-19 regular  0021800163"
  collect-possessions --game 0021800163 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800164.parquet" ]; then
  echo "[164/1456]  skip 0021800164 (already collected)"
else
  echo "[164/1456]  2018-19 regular  0021800164"
  collect-possessions --game 0021800164 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800165.parquet" ]; then
  echo "[165/1456]  skip 0021800165 (already collected)"
else
  echo "[165/1456]  2018-19 regular  0021800165"
  collect-possessions --game 0021800165 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800166.parquet" ]; then
  echo "[166/1456]  skip 0021800166 (already collected)"
else
  echo "[166/1456]  2018-19 regular  0021800166"
  collect-possessions --game 0021800166 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800167.parquet" ]; then
  echo "[167/1456]  skip 0021800167 (already collected)"
else
  echo "[167/1456]  2018-19 regular  0021800167"
  collect-possessions --game 0021800167 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800168.parquet" ]; then
  echo "[168/1456]  skip 0021800168 (already collected)"
else
  echo "[168/1456]  2018-19 regular  0021800168"
  collect-possessions --game 0021800168 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800169.parquet" ]; then
  echo "[169/1456]  skip 0021800169 (already collected)"
else
  echo "[169/1456]  2018-19 regular  0021800169"
  collect-possessions --game 0021800169 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800170.parquet" ]; then
  echo "[170/1456]  skip 0021800170 (already collected)"
else
  echo "[170/1456]  2018-19 regular  0021800170"
  collect-possessions --game 0021800170 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800171.parquet" ]; then
  echo "[171/1456]  skip 0021800171 (already collected)"
else
  echo "[171/1456]  2018-19 regular  0021800171"
  collect-possessions --game 0021800171 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800172.parquet" ]; then
  echo "[172/1456]  skip 0021800172 (already collected)"
else
  echo "[172/1456]  2018-19 regular  0021800172"
  collect-possessions --game 0021800172 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800173.parquet" ]; then
  echo "[173/1456]  skip 0021800173 (already collected)"
else
  echo "[173/1456]  2018-19 regular  0021800173"
  collect-possessions --game 0021800173 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800174.parquet" ]; then
  echo "[174/1456]  skip 0021800174 (already collected)"
else
  echo "[174/1456]  2018-19 regular  0021800174"
  collect-possessions --game 0021800174 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800175.parquet" ]; then
  echo "[175/1456]  skip 0021800175 (already collected)"
else
  echo "[175/1456]  2018-19 regular  0021800175"
  collect-possessions --game 0021800175 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800176.parquet" ]; then
  echo "[176/1456]  skip 0021800176 (already collected)"
else
  echo "[176/1456]  2018-19 regular  0021800176"
  collect-possessions --game 0021800176 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800177.parquet" ]; then
  echo "[177/1456]  skip 0021800177 (already collected)"
else
  echo "[177/1456]  2018-19 regular  0021800177"
  collect-possessions --game 0021800177 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800178.parquet" ]; then
  echo "[178/1456]  skip 0021800178 (already collected)"
else
  echo "[178/1456]  2018-19 regular  0021800178"
  collect-possessions --game 0021800178 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800179.parquet" ]; then
  echo "[179/1456]  skip 0021800179 (already collected)"
else
  echo "[179/1456]  2018-19 regular  0021800179"
  collect-possessions --game 0021800179 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800180.parquet" ]; then
  echo "[180/1456]  skip 0021800180 (already collected)"
else
  echo "[180/1456]  2018-19 regular  0021800180"
  collect-possessions --game 0021800180 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800181.parquet" ]; then
  echo "[181/1456]  skip 0021800181 (already collected)"
else
  echo "[181/1456]  2018-19 regular  0021800181"
  collect-possessions --game 0021800181 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800182.parquet" ]; then
  echo "[182/1456]  skip 0021800182 (already collected)"
else
  echo "[182/1456]  2018-19 regular  0021800182"
  collect-possessions --game 0021800182 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800183.parquet" ]; then
  echo "[183/1456]  skip 0021800183 (already collected)"
else
  echo "[183/1456]  2018-19 regular  0021800183"
  collect-possessions --game 0021800183 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800184.parquet" ]; then
  echo "[184/1456]  skip 0021800184 (already collected)"
else
  echo "[184/1456]  2018-19 regular  0021800184"
  collect-possessions --game 0021800184 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800185.parquet" ]; then
  echo "[185/1456]  skip 0021800185 (already collected)"
else
  echo "[185/1456]  2018-19 regular  0021800185"
  collect-possessions --game 0021800185 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800186.parquet" ]; then
  echo "[186/1456]  skip 0021800186 (already collected)"
else
  echo "[186/1456]  2018-19 regular  0021800186"
  collect-possessions --game 0021800186 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800187.parquet" ]; then
  echo "[187/1456]  skip 0021800187 (already collected)"
else
  echo "[187/1456]  2018-19 regular  0021800187"
  collect-possessions --game 0021800187 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800188.parquet" ]; then
  echo "[188/1456]  skip 0021800188 (already collected)"
else
  echo "[188/1456]  2018-19 regular  0021800188"
  collect-possessions --game 0021800188 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800189.parquet" ]; then
  echo "[189/1456]  skip 0021800189 (already collected)"
else
  echo "[189/1456]  2018-19 regular  0021800189"
  collect-possessions --game 0021800189 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800190.parquet" ]; then
  echo "[190/1456]  skip 0021800190 (already collected)"
else
  echo "[190/1456]  2018-19 regular  0021800190"
  collect-possessions --game 0021800190 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800191.parquet" ]; then
  echo "[191/1456]  skip 0021800191 (already collected)"
else
  echo "[191/1456]  2018-19 regular  0021800191"
  collect-possessions --game 0021800191 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800192.parquet" ]; then
  echo "[192/1456]  skip 0021800192 (already collected)"
else
  echo "[192/1456]  2018-19 regular  0021800192"
  collect-possessions --game 0021800192 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800193.parquet" ]; then
  echo "[193/1456]  skip 0021800193 (already collected)"
else
  echo "[193/1456]  2018-19 regular  0021800193"
  collect-possessions --game 0021800193 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800194.parquet" ]; then
  echo "[194/1456]  skip 0021800194 (already collected)"
else
  echo "[194/1456]  2018-19 regular  0021800194"
  collect-possessions --game 0021800194 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800195.parquet" ]; then
  echo "[195/1456]  skip 0021800195 (already collected)"
else
  echo "[195/1456]  2018-19 regular  0021800195"
  collect-possessions --game 0021800195 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800196.parquet" ]; then
  echo "[196/1456]  skip 0021800196 (already collected)"
else
  echo "[196/1456]  2018-19 regular  0021800196"
  collect-possessions --game 0021800196 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800197.parquet" ]; then
  echo "[197/1456]  skip 0021800197 (already collected)"
else
  echo "[197/1456]  2018-19 regular  0021800197"
  collect-possessions --game 0021800197 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800198.parquet" ]; then
  echo "[198/1456]  skip 0021800198 (already collected)"
else
  echo "[198/1456]  2018-19 regular  0021800198"
  collect-possessions --game 0021800198 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800199.parquet" ]; then
  echo "[199/1456]  skip 0021800199 (already collected)"
else
  echo "[199/1456]  2018-19 regular  0021800199"
  collect-possessions --game 0021800199 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800200.parquet" ]; then
  echo "[200/1456]  skip 0021800200 (already collected)"
else
  echo "[200/1456]  2018-19 regular  0021800200"
  collect-possessions --game 0021800200 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800201.parquet" ]; then
  echo "[201/1456]  skip 0021800201 (already collected)"
else
  echo "[201/1456]  2018-19 regular  0021800201"
  collect-possessions --game 0021800201 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800202.parquet" ]; then
  echo "[202/1456]  skip 0021800202 (already collected)"
else
  echo "[202/1456]  2018-19 regular  0021800202"
  collect-possessions --game 0021800202 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800203.parquet" ]; then
  echo "[203/1456]  skip 0021800203 (already collected)"
else
  echo "[203/1456]  2018-19 regular  0021800203"
  collect-possessions --game 0021800203 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800204.parquet" ]; then
  echo "[204/1456]  skip 0021800204 (already collected)"
else
  echo "[204/1456]  2018-19 regular  0021800204"
  collect-possessions --game 0021800204 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800205.parquet" ]; then
  echo "[205/1456]  skip 0021800205 (already collected)"
else
  echo "[205/1456]  2018-19 regular  0021800205"
  collect-possessions --game 0021800205 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800206.parquet" ]; then
  echo "[206/1456]  skip 0021800206 (already collected)"
else
  echo "[206/1456]  2018-19 regular  0021800206"
  collect-possessions --game 0021800206 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800207.parquet" ]; then
  echo "[207/1456]  skip 0021800207 (already collected)"
else
  echo "[207/1456]  2018-19 regular  0021800207"
  collect-possessions --game 0021800207 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800208.parquet" ]; then
  echo "[208/1456]  skip 0021800208 (already collected)"
else
  echo "[208/1456]  2018-19 regular  0021800208"
  collect-possessions --game 0021800208 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800209.parquet" ]; then
  echo "[209/1456]  skip 0021800209 (already collected)"
else
  echo "[209/1456]  2018-19 regular  0021800209"
  collect-possessions --game 0021800209 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800210.parquet" ]; then
  echo "[210/1456]  skip 0021800210 (already collected)"
else
  echo "[210/1456]  2018-19 regular  0021800210"
  collect-possessions --game 0021800210 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800211.parquet" ]; then
  echo "[211/1456]  skip 0021800211 (already collected)"
else
  echo "[211/1456]  2018-19 regular  0021800211"
  collect-possessions --game 0021800211 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800212.parquet" ]; then
  echo "[212/1456]  skip 0021800212 (already collected)"
else
  echo "[212/1456]  2018-19 regular  0021800212"
  collect-possessions --game 0021800212 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800213.parquet" ]; then
  echo "[213/1456]  skip 0021800213 (already collected)"
else
  echo "[213/1456]  2018-19 regular  0021800213"
  collect-possessions --game 0021800213 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800214.parquet" ]; then
  echo "[214/1456]  skip 0021800214 (already collected)"
else
  echo "[214/1456]  2018-19 regular  0021800214"
  collect-possessions --game 0021800214 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800215.parquet" ]; then
  echo "[215/1456]  skip 0021800215 (already collected)"
else
  echo "[215/1456]  2018-19 regular  0021800215"
  collect-possessions --game 0021800215 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800216.parquet" ]; then
  echo "[216/1456]  skip 0021800216 (already collected)"
else
  echo "[216/1456]  2018-19 regular  0021800216"
  collect-possessions --game 0021800216 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800217.parquet" ]; then
  echo "[217/1456]  skip 0021800217 (already collected)"
else
  echo "[217/1456]  2018-19 regular  0021800217"
  collect-possessions --game 0021800217 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800218.parquet" ]; then
  echo "[218/1456]  skip 0021800218 (already collected)"
else
  echo "[218/1456]  2018-19 regular  0021800218"
  collect-possessions --game 0021800218 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800219.parquet" ]; then
  echo "[219/1456]  skip 0021800219 (already collected)"
else
  echo "[219/1456]  2018-19 regular  0021800219"
  collect-possessions --game 0021800219 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800220.parquet" ]; then
  echo "[220/1456]  skip 0021800220 (already collected)"
else
  echo "[220/1456]  2018-19 regular  0021800220"
  collect-possessions --game 0021800220 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800221.parquet" ]; then
  echo "[221/1456]  skip 0021800221 (already collected)"
else
  echo "[221/1456]  2018-19 regular  0021800221"
  collect-possessions --game 0021800221 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800222.parquet" ]; then
  echo "[222/1456]  skip 0021800222 (already collected)"
else
  echo "[222/1456]  2018-19 regular  0021800222"
  collect-possessions --game 0021800222 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800223.parquet" ]; then
  echo "[223/1456]  skip 0021800223 (already collected)"
else
  echo "[223/1456]  2018-19 regular  0021800223"
  collect-possessions --game 0021800223 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800224.parquet" ]; then
  echo "[224/1456]  skip 0021800224 (already collected)"
else
  echo "[224/1456]  2018-19 regular  0021800224"
  collect-possessions --game 0021800224 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800225.parquet" ]; then
  echo "[225/1456]  skip 0021800225 (already collected)"
else
  echo "[225/1456]  2018-19 regular  0021800225"
  collect-possessions --game 0021800225 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800226.parquet" ]; then
  echo "[226/1456]  skip 0021800226 (already collected)"
else
  echo "[226/1456]  2018-19 regular  0021800226"
  collect-possessions --game 0021800226 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800227.parquet" ]; then
  echo "[227/1456]  skip 0021800227 (already collected)"
else
  echo "[227/1456]  2018-19 regular  0021800227"
  collect-possessions --game 0021800227 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800228.parquet" ]; then
  echo "[228/1456]  skip 0021800228 (already collected)"
else
  echo "[228/1456]  2018-19 regular  0021800228"
  collect-possessions --game 0021800228 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800229.parquet" ]; then
  echo "[229/1456]  skip 0021800229 (already collected)"
else
  echo "[229/1456]  2018-19 regular  0021800229"
  collect-possessions --game 0021800229 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800230.parquet" ]; then
  echo "[230/1456]  skip 0021800230 (already collected)"
else
  echo "[230/1456]  2018-19 regular  0021800230"
  collect-possessions --game 0021800230 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800231.parquet" ]; then
  echo "[231/1456]  skip 0021800231 (already collected)"
else
  echo "[231/1456]  2018-19 regular  0021800231"
  collect-possessions --game 0021800231 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800232.parquet" ]; then
  echo "[232/1456]  skip 0021800232 (already collected)"
else
  echo "[232/1456]  2018-19 regular  0021800232"
  collect-possessions --game 0021800232 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800233.parquet" ]; then
  echo "[233/1456]  skip 0021800233 (already collected)"
else
  echo "[233/1456]  2018-19 regular  0021800233"
  collect-possessions --game 0021800233 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800234.parquet" ]; then
  echo "[234/1456]  skip 0021800234 (already collected)"
else
  echo "[234/1456]  2018-19 regular  0021800234"
  collect-possessions --game 0021800234 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800235.parquet" ]; then
  echo "[235/1456]  skip 0021800235 (already collected)"
else
  echo "[235/1456]  2018-19 regular  0021800235"
  collect-possessions --game 0021800235 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800236.parquet" ]; then
  echo "[236/1456]  skip 0021800236 (already collected)"
else
  echo "[236/1456]  2018-19 regular  0021800236"
  collect-possessions --game 0021800236 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800237.parquet" ]; then
  echo "[237/1456]  skip 0021800237 (already collected)"
else
  echo "[237/1456]  2018-19 regular  0021800237"
  collect-possessions --game 0021800237 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800238.parquet" ]; then
  echo "[238/1456]  skip 0021800238 (already collected)"
else
  echo "[238/1456]  2018-19 regular  0021800238"
  collect-possessions --game 0021800238 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800239.parquet" ]; then
  echo "[239/1456]  skip 0021800239 (already collected)"
else
  echo "[239/1456]  2018-19 regular  0021800239"
  collect-possessions --game 0021800239 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800240.parquet" ]; then
  echo "[240/1456]  skip 0021800240 (already collected)"
else
  echo "[240/1456]  2018-19 regular  0021800240"
  collect-possessions --game 0021800240 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800241.parquet" ]; then
  echo "[241/1456]  skip 0021800241 (already collected)"
else
  echo "[241/1456]  2018-19 regular  0021800241"
  collect-possessions --game 0021800241 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800242.parquet" ]; then
  echo "[242/1456]  skip 0021800242 (already collected)"
else
  echo "[242/1456]  2018-19 regular  0021800242"
  collect-possessions --game 0021800242 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800243.parquet" ]; then
  echo "[243/1456]  skip 0021800243 (already collected)"
else
  echo "[243/1456]  2018-19 regular  0021800243"
  collect-possessions --game 0021800243 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800244.parquet" ]; then
  echo "[244/1456]  skip 0021800244 (already collected)"
else
  echo "[244/1456]  2018-19 regular  0021800244"
  collect-possessions --game 0021800244 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800245.parquet" ]; then
  echo "[245/1456]  skip 0021800245 (already collected)"
else
  echo "[245/1456]  2018-19 regular  0021800245"
  collect-possessions --game 0021800245 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800246.parquet" ]; then
  echo "[246/1456]  skip 0021800246 (already collected)"
else
  echo "[246/1456]  2018-19 regular  0021800246"
  collect-possessions --game 0021800246 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800247.parquet" ]; then
  echo "[247/1456]  skip 0021800247 (already collected)"
else
  echo "[247/1456]  2018-19 regular  0021800247"
  collect-possessions --game 0021800247 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800248.parquet" ]; then
  echo "[248/1456]  skip 0021800248 (already collected)"
else
  echo "[248/1456]  2018-19 regular  0021800248"
  collect-possessions --game 0021800248 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800249.parquet" ]; then
  echo "[249/1456]  skip 0021800249 (already collected)"
else
  echo "[249/1456]  2018-19 regular  0021800249"
  collect-possessions --game 0021800249 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800250.parquet" ]; then
  echo "[250/1456]  skip 0021800250 (already collected)"
else
  echo "[250/1456]  2018-19 regular  0021800250"
  collect-possessions --game 0021800250 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800251.parquet" ]; then
  echo "[251/1456]  skip 0021800251 (already collected)"
else
  echo "[251/1456]  2018-19 regular  0021800251"
  collect-possessions --game 0021800251 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800252.parquet" ]; then
  echo "[252/1456]  skip 0021800252 (already collected)"
else
  echo "[252/1456]  2018-19 regular  0021800252"
  collect-possessions --game 0021800252 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800253.parquet" ]; then
  echo "[253/1456]  skip 0021800253 (already collected)"
else
  echo "[253/1456]  2018-19 regular  0021800253"
  collect-possessions --game 0021800253 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800254.parquet" ]; then
  echo "[254/1456]  skip 0021800254 (already collected)"
else
  echo "[254/1456]  2018-19 regular  0021800254"
  collect-possessions --game 0021800254 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800255.parquet" ]; then
  echo "[255/1456]  skip 0021800255 (already collected)"
else
  echo "[255/1456]  2018-19 regular  0021800255"
  collect-possessions --game 0021800255 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800256.parquet" ]; then
  echo "[256/1456]  skip 0021800256 (already collected)"
else
  echo "[256/1456]  2018-19 regular  0021800256"
  collect-possessions --game 0021800256 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800257.parquet" ]; then
  echo "[257/1456]  skip 0021800257 (already collected)"
else
  echo "[257/1456]  2018-19 regular  0021800257"
  collect-possessions --game 0021800257 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800258.parquet" ]; then
  echo "[258/1456]  skip 0021800258 (already collected)"
else
  echo "[258/1456]  2018-19 regular  0021800258"
  collect-possessions --game 0021800258 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800259.parquet" ]; then
  echo "[259/1456]  skip 0021800259 (already collected)"
else
  echo "[259/1456]  2018-19 regular  0021800259"
  collect-possessions --game 0021800259 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800260.parquet" ]; then
  echo "[260/1456]  skip 0021800260 (already collected)"
else
  echo "[260/1456]  2018-19 regular  0021800260"
  collect-possessions --game 0021800260 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800261.parquet" ]; then
  echo "[261/1456]  skip 0021800261 (already collected)"
else
  echo "[261/1456]  2018-19 regular  0021800261"
  collect-possessions --game 0021800261 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800262.parquet" ]; then
  echo "[262/1456]  skip 0021800262 (already collected)"
else
  echo "[262/1456]  2018-19 regular  0021800262"
  collect-possessions --game 0021800262 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800263.parquet" ]; then
  echo "[263/1456]  skip 0021800263 (already collected)"
else
  echo "[263/1456]  2018-19 regular  0021800263"
  collect-possessions --game 0021800263 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800264.parquet" ]; then
  echo "[264/1456]  skip 0021800264 (already collected)"
else
  echo "[264/1456]  2018-19 regular  0021800264"
  collect-possessions --game 0021800264 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800265.parquet" ]; then
  echo "[265/1456]  skip 0021800265 (already collected)"
else
  echo "[265/1456]  2018-19 regular  0021800265"
  collect-possessions --game 0021800265 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800266.parquet" ]; then
  echo "[266/1456]  skip 0021800266 (already collected)"
else
  echo "[266/1456]  2018-19 regular  0021800266"
  collect-possessions --game 0021800266 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800267.parquet" ]; then
  echo "[267/1456]  skip 0021800267 (already collected)"
else
  echo "[267/1456]  2018-19 regular  0021800267"
  collect-possessions --game 0021800267 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800268.parquet" ]; then
  echo "[268/1456]  skip 0021800268 (already collected)"
else
  echo "[268/1456]  2018-19 regular  0021800268"
  collect-possessions --game 0021800268 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800269.parquet" ]; then
  echo "[269/1456]  skip 0021800269 (already collected)"
else
  echo "[269/1456]  2018-19 regular  0021800269"
  collect-possessions --game 0021800269 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800270.parquet" ]; then
  echo "[270/1456]  skip 0021800270 (already collected)"
else
  echo "[270/1456]  2018-19 regular  0021800270"
  collect-possessions --game 0021800270 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800271.parquet" ]; then
  echo "[271/1456]  skip 0021800271 (already collected)"
else
  echo "[271/1456]  2018-19 regular  0021800271"
  collect-possessions --game 0021800271 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800272.parquet" ]; then
  echo "[272/1456]  skip 0021800272 (already collected)"
else
  echo "[272/1456]  2018-19 regular  0021800272"
  collect-possessions --game 0021800272 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800273.parquet" ]; then
  echo "[273/1456]  skip 0021800273 (already collected)"
else
  echo "[273/1456]  2018-19 regular  0021800273"
  collect-possessions --game 0021800273 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800274.parquet" ]; then
  echo "[274/1456]  skip 0021800274 (already collected)"
else
  echo "[274/1456]  2018-19 regular  0021800274"
  collect-possessions --game 0021800274 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800275.parquet" ]; then
  echo "[275/1456]  skip 0021800275 (already collected)"
else
  echo "[275/1456]  2018-19 regular  0021800275"
  collect-possessions --game 0021800275 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800276.parquet" ]; then
  echo "[276/1456]  skip 0021800276 (already collected)"
else
  echo "[276/1456]  2018-19 regular  0021800276"
  collect-possessions --game 0021800276 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800277.parquet" ]; then
  echo "[277/1456]  skip 0021800277 (already collected)"
else
  echo "[277/1456]  2018-19 regular  0021800277"
  collect-possessions --game 0021800277 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800278.parquet" ]; then
  echo "[278/1456]  skip 0021800278 (already collected)"
else
  echo "[278/1456]  2018-19 regular  0021800278"
  collect-possessions --game 0021800278 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800279.parquet" ]; then
  echo "[279/1456]  skip 0021800279 (already collected)"
else
  echo "[279/1456]  2018-19 regular  0021800279"
  collect-possessions --game 0021800279 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800280.parquet" ]; then
  echo "[280/1456]  skip 0021800280 (already collected)"
else
  echo "[280/1456]  2018-19 regular  0021800280"
  collect-possessions --game 0021800280 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800281.parquet" ]; then
  echo "[281/1456]  skip 0021800281 (already collected)"
else
  echo "[281/1456]  2018-19 regular  0021800281"
  collect-possessions --game 0021800281 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800282.parquet" ]; then
  echo "[282/1456]  skip 0021800282 (already collected)"
else
  echo "[282/1456]  2018-19 regular  0021800282"
  collect-possessions --game 0021800282 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800283.parquet" ]; then
  echo "[283/1456]  skip 0021800283 (already collected)"
else
  echo "[283/1456]  2018-19 regular  0021800283"
  collect-possessions --game 0021800283 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800284.parquet" ]; then
  echo "[284/1456]  skip 0021800284 (already collected)"
else
  echo "[284/1456]  2018-19 regular  0021800284"
  collect-possessions --game 0021800284 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800285.parquet" ]; then
  echo "[285/1456]  skip 0021800285 (already collected)"
else
  echo "[285/1456]  2018-19 regular  0021800285"
  collect-possessions --game 0021800285 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800286.parquet" ]; then
  echo "[286/1456]  skip 0021800286 (already collected)"
else
  echo "[286/1456]  2018-19 regular  0021800286"
  collect-possessions --game 0021800286 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800287.parquet" ]; then
  echo "[287/1456]  skip 0021800287 (already collected)"
else
  echo "[287/1456]  2018-19 regular  0021800287"
  collect-possessions --game 0021800287 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800288.parquet" ]; then
  echo "[288/1456]  skip 0021800288 (already collected)"
else
  echo "[288/1456]  2018-19 regular  0021800288"
  collect-possessions --game 0021800288 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800289.parquet" ]; then
  echo "[289/1456]  skip 0021800289 (already collected)"
else
  echo "[289/1456]  2018-19 regular  0021800289"
  collect-possessions --game 0021800289 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800290.parquet" ]; then
  echo "[290/1456]  skip 0021800290 (already collected)"
else
  echo "[290/1456]  2018-19 regular  0021800290"
  collect-possessions --game 0021800290 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800291.parquet" ]; then
  echo "[291/1456]  skip 0021800291 (already collected)"
else
  echo "[291/1456]  2018-19 regular  0021800291"
  collect-possessions --game 0021800291 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800292.parquet" ]; then
  echo "[292/1456]  skip 0021800292 (already collected)"
else
  echo "[292/1456]  2018-19 regular  0021800292"
  collect-possessions --game 0021800292 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800293.parquet" ]; then
  echo "[293/1456]  skip 0021800293 (already collected)"
else
  echo "[293/1456]  2018-19 regular  0021800293"
  collect-possessions --game 0021800293 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800294.parquet" ]; then
  echo "[294/1456]  skip 0021800294 (already collected)"
else
  echo "[294/1456]  2018-19 regular  0021800294"
  collect-possessions --game 0021800294 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800295.parquet" ]; then
  echo "[295/1456]  skip 0021800295 (already collected)"
else
  echo "[295/1456]  2018-19 regular  0021800295"
  collect-possessions --game 0021800295 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800296.parquet" ]; then
  echo "[296/1456]  skip 0021800296 (already collected)"
else
  echo "[296/1456]  2018-19 regular  0021800296"
  collect-possessions --game 0021800296 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800297.parquet" ]; then
  echo "[297/1456]  skip 0021800297 (already collected)"
else
  echo "[297/1456]  2018-19 regular  0021800297"
  collect-possessions --game 0021800297 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800298.parquet" ]; then
  echo "[298/1456]  skip 0021800298 (already collected)"
else
  echo "[298/1456]  2018-19 regular  0021800298"
  collect-possessions --game 0021800298 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800299.parquet" ]; then
  echo "[299/1456]  skip 0021800299 (already collected)"
else
  echo "[299/1456]  2018-19 regular  0021800299"
  collect-possessions --game 0021800299 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800300.parquet" ]; then
  echo "[300/1456]  skip 0021800300 (already collected)"
else
  echo "[300/1456]  2018-19 regular  0021800300"
  collect-possessions --game 0021800300 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800301.parquet" ]; then
  echo "[301/1456]  skip 0021800301 (already collected)"
else
  echo "[301/1456]  2018-19 regular  0021800301"
  collect-possessions --game 0021800301 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800302.parquet" ]; then
  echo "[302/1456]  skip 0021800302 (already collected)"
else
  echo "[302/1456]  2018-19 regular  0021800302"
  collect-possessions --game 0021800302 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800303.parquet" ]; then
  echo "[303/1456]  skip 0021800303 (already collected)"
else
  echo "[303/1456]  2018-19 regular  0021800303"
  collect-possessions --game 0021800303 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800304.parquet" ]; then
  echo "[304/1456]  skip 0021800304 (already collected)"
else
  echo "[304/1456]  2018-19 regular  0021800304"
  collect-possessions --game 0021800304 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800305.parquet" ]; then
  echo "[305/1456]  skip 0021800305 (already collected)"
else
  echo "[305/1456]  2018-19 regular  0021800305"
  collect-possessions --game 0021800305 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800306.parquet" ]; then
  echo "[306/1456]  skip 0021800306 (already collected)"
else
  echo "[306/1456]  2018-19 regular  0021800306"
  collect-possessions --game 0021800306 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800307.parquet" ]; then
  echo "[307/1456]  skip 0021800307 (already collected)"
else
  echo "[307/1456]  2018-19 regular  0021800307"
  collect-possessions --game 0021800307 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800308.parquet" ]; then
  echo "[308/1456]  skip 0021800308 (already collected)"
else
  echo "[308/1456]  2018-19 regular  0021800308"
  collect-possessions --game 0021800308 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800309.parquet" ]; then
  echo "[309/1456]  skip 0021800309 (already collected)"
else
  echo "[309/1456]  2018-19 regular  0021800309"
  collect-possessions --game 0021800309 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800310.parquet" ]; then
  echo "[310/1456]  skip 0021800310 (already collected)"
else
  echo "[310/1456]  2018-19 regular  0021800310"
  collect-possessions --game 0021800310 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800311.parquet" ]; then
  echo "[311/1456]  skip 0021800311 (already collected)"
else
  echo "[311/1456]  2018-19 regular  0021800311"
  collect-possessions --game 0021800311 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800312.parquet" ]; then
  echo "[312/1456]  skip 0021800312 (already collected)"
else
  echo "[312/1456]  2018-19 regular  0021800312"
  collect-possessions --game 0021800312 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800313.parquet" ]; then
  echo "[313/1456]  skip 0021800313 (already collected)"
else
  echo "[313/1456]  2018-19 regular  0021800313"
  collect-possessions --game 0021800313 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800314.parquet" ]; then
  echo "[314/1456]  skip 0021800314 (already collected)"
else
  echo "[314/1456]  2018-19 regular  0021800314"
  collect-possessions --game 0021800314 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800315.parquet" ]; then
  echo "[315/1456]  skip 0021800315 (already collected)"
else
  echo "[315/1456]  2018-19 regular  0021800315"
  collect-possessions --game 0021800315 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800316.parquet" ]; then
  echo "[316/1456]  skip 0021800316 (already collected)"
else
  echo "[316/1456]  2018-19 regular  0021800316"
  collect-possessions --game 0021800316 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800317.parquet" ]; then
  echo "[317/1456]  skip 0021800317 (already collected)"
else
  echo "[317/1456]  2018-19 regular  0021800317"
  collect-possessions --game 0021800317 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800318.parquet" ]; then
  echo "[318/1456]  skip 0021800318 (already collected)"
else
  echo "[318/1456]  2018-19 regular  0021800318"
  collect-possessions --game 0021800318 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800319.parquet" ]; then
  echo "[319/1456]  skip 0021800319 (already collected)"
else
  echo "[319/1456]  2018-19 regular  0021800319"
  collect-possessions --game 0021800319 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800320.parquet" ]; then
  echo "[320/1456]  skip 0021800320 (already collected)"
else
  echo "[320/1456]  2018-19 regular  0021800320"
  collect-possessions --game 0021800320 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800321.parquet" ]; then
  echo "[321/1456]  skip 0021800321 (already collected)"
else
  echo "[321/1456]  2018-19 regular  0021800321"
  collect-possessions --game 0021800321 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800322.parquet" ]; then
  echo "[322/1456]  skip 0021800322 (already collected)"
else
  echo "[322/1456]  2018-19 regular  0021800322"
  collect-possessions --game 0021800322 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800323.parquet" ]; then
  echo "[323/1456]  skip 0021800323 (already collected)"
else
  echo "[323/1456]  2018-19 regular  0021800323"
  collect-possessions --game 0021800323 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800324.parquet" ]; then
  echo "[324/1456]  skip 0021800324 (already collected)"
else
  echo "[324/1456]  2018-19 regular  0021800324"
  collect-possessions --game 0021800324 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800325.parquet" ]; then
  echo "[325/1456]  skip 0021800325 (already collected)"
else
  echo "[325/1456]  2018-19 regular  0021800325"
  collect-possessions --game 0021800325 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800326.parquet" ]; then
  echo "[326/1456]  skip 0021800326 (already collected)"
else
  echo "[326/1456]  2018-19 regular  0021800326"
  collect-possessions --game 0021800326 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800327.parquet" ]; then
  echo "[327/1456]  skip 0021800327 (already collected)"
else
  echo "[327/1456]  2018-19 regular  0021800327"
  collect-possessions --game 0021800327 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800328.parquet" ]; then
  echo "[328/1456]  skip 0021800328 (already collected)"
else
  echo "[328/1456]  2018-19 regular  0021800328"
  collect-possessions --game 0021800328 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800329.parquet" ]; then
  echo "[329/1456]  skip 0021800329 (already collected)"
else
  echo "[329/1456]  2018-19 regular  0021800329"
  collect-possessions --game 0021800329 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800330.parquet" ]; then
  echo "[330/1456]  skip 0021800330 (already collected)"
else
  echo "[330/1456]  2018-19 regular  0021800330"
  collect-possessions --game 0021800330 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800331.parquet" ]; then
  echo "[331/1456]  skip 0021800331 (already collected)"
else
  echo "[331/1456]  2018-19 regular  0021800331"
  collect-possessions --game 0021800331 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800332.parquet" ]; then
  echo "[332/1456]  skip 0021800332 (already collected)"
else
  echo "[332/1456]  2018-19 regular  0021800332"
  collect-possessions --game 0021800332 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800333.parquet" ]; then
  echo "[333/1456]  skip 0021800333 (already collected)"
else
  echo "[333/1456]  2018-19 regular  0021800333"
  collect-possessions --game 0021800333 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800334.parquet" ]; then
  echo "[334/1456]  skip 0021800334 (already collected)"
else
  echo "[334/1456]  2018-19 regular  0021800334"
  collect-possessions --game 0021800334 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800335.parquet" ]; then
  echo "[335/1456]  skip 0021800335 (already collected)"
else
  echo "[335/1456]  2018-19 regular  0021800335"
  collect-possessions --game 0021800335 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800336.parquet" ]; then
  echo "[336/1456]  skip 0021800336 (already collected)"
else
  echo "[336/1456]  2018-19 regular  0021800336"
  collect-possessions --game 0021800336 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800337.parquet" ]; then
  echo "[337/1456]  skip 0021800337 (already collected)"
else
  echo "[337/1456]  2018-19 regular  0021800337"
  collect-possessions --game 0021800337 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800338.parquet" ]; then
  echo "[338/1456]  skip 0021800338 (already collected)"
else
  echo "[338/1456]  2018-19 regular  0021800338"
  collect-possessions --game 0021800338 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800339.parquet" ]; then
  echo "[339/1456]  skip 0021800339 (already collected)"
else
  echo "[339/1456]  2018-19 regular  0021800339"
  collect-possessions --game 0021800339 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800340.parquet" ]; then
  echo "[340/1456]  skip 0021800340 (already collected)"
else
  echo "[340/1456]  2018-19 regular  0021800340"
  collect-possessions --game 0021800340 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800341.parquet" ]; then
  echo "[341/1456]  skip 0021800341 (already collected)"
else
  echo "[341/1456]  2018-19 regular  0021800341"
  collect-possessions --game 0021800341 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800342.parquet" ]; then
  echo "[342/1456]  skip 0021800342 (already collected)"
else
  echo "[342/1456]  2018-19 regular  0021800342"
  collect-possessions --game 0021800342 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800343.parquet" ]; then
  echo "[343/1456]  skip 0021800343 (already collected)"
else
  echo "[343/1456]  2018-19 regular  0021800343"
  collect-possessions --game 0021800343 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800344.parquet" ]; then
  echo "[344/1456]  skip 0021800344 (already collected)"
else
  echo "[344/1456]  2018-19 regular  0021800344"
  collect-possessions --game 0021800344 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800345.parquet" ]; then
  echo "[345/1456]  skip 0021800345 (already collected)"
else
  echo "[345/1456]  2018-19 regular  0021800345"
  collect-possessions --game 0021800345 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800346.parquet" ]; then
  echo "[346/1456]  skip 0021800346 (already collected)"
else
  echo "[346/1456]  2018-19 regular  0021800346"
  collect-possessions --game 0021800346 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800347.parquet" ]; then
  echo "[347/1456]  skip 0021800347 (already collected)"
else
  echo "[347/1456]  2018-19 regular  0021800347"
  collect-possessions --game 0021800347 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800348.parquet" ]; then
  echo "[348/1456]  skip 0021800348 (already collected)"
else
  echo "[348/1456]  2018-19 regular  0021800348"
  collect-possessions --game 0021800348 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800349.parquet" ]; then
  echo "[349/1456]  skip 0021800349 (already collected)"
else
  echo "[349/1456]  2018-19 regular  0021800349"
  collect-possessions --game 0021800349 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800350.parquet" ]; then
  echo "[350/1456]  skip 0021800350 (already collected)"
else
  echo "[350/1456]  2018-19 regular  0021800350"
  collect-possessions --game 0021800350 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800351.parquet" ]; then
  echo "[351/1456]  skip 0021800351 (already collected)"
else
  echo "[351/1456]  2018-19 regular  0021800351"
  collect-possessions --game 0021800351 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800352.parquet" ]; then
  echo "[352/1456]  skip 0021800352 (already collected)"
else
  echo "[352/1456]  2018-19 regular  0021800352"
  collect-possessions --game 0021800352 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800353.parquet" ]; then
  echo "[353/1456]  skip 0021800353 (already collected)"
else
  echo "[353/1456]  2018-19 regular  0021800353"
  collect-possessions --game 0021800353 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800354.parquet" ]; then
  echo "[354/1456]  skip 0021800354 (already collected)"
else
  echo "[354/1456]  2018-19 regular  0021800354"
  collect-possessions --game 0021800354 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800355.parquet" ]; then
  echo "[355/1456]  skip 0021800355 (already collected)"
else
  echo "[355/1456]  2018-19 regular  0021800355"
  collect-possessions --game 0021800355 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800356.parquet" ]; then
  echo "[356/1456]  skip 0021800356 (already collected)"
else
  echo "[356/1456]  2018-19 regular  0021800356"
  collect-possessions --game 0021800356 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800357.parquet" ]; then
  echo "[357/1456]  skip 0021800357 (already collected)"
else
  echo "[357/1456]  2018-19 regular  0021800357"
  collect-possessions --game 0021800357 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800358.parquet" ]; then
  echo "[358/1456]  skip 0021800358 (already collected)"
else
  echo "[358/1456]  2018-19 regular  0021800358"
  collect-possessions --game 0021800358 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800359.parquet" ]; then
  echo "[359/1456]  skip 0021800359 (already collected)"
else
  echo "[359/1456]  2018-19 regular  0021800359"
  collect-possessions --game 0021800359 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800360.parquet" ]; then
  echo "[360/1456]  skip 0021800360 (already collected)"
else
  echo "[360/1456]  2018-19 regular  0021800360"
  collect-possessions --game 0021800360 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800361.parquet" ]; then
  echo "[361/1456]  skip 0021800361 (already collected)"
else
  echo "[361/1456]  2018-19 regular  0021800361"
  collect-possessions --game 0021800361 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800362.parquet" ]; then
  echo "[362/1456]  skip 0021800362 (already collected)"
else
  echo "[362/1456]  2018-19 regular  0021800362"
  collect-possessions --game 0021800362 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800363.parquet" ]; then
  echo "[363/1456]  skip 0021800363 (already collected)"
else
  echo "[363/1456]  2018-19 regular  0021800363"
  collect-possessions --game 0021800363 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800364.parquet" ]; then
  echo "[364/1456]  skip 0021800364 (already collected)"
else
  echo "[364/1456]  2018-19 regular  0021800364"
  collect-possessions --game 0021800364 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800365.parquet" ]; then
  echo "[365/1456]  skip 0021800365 (already collected)"
else
  echo "[365/1456]  2018-19 regular  0021800365"
  collect-possessions --game 0021800365 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800366.parquet" ]; then
  echo "[366/1456]  skip 0021800366 (already collected)"
else
  echo "[366/1456]  2018-19 regular  0021800366"
  collect-possessions --game 0021800366 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800367.parquet" ]; then
  echo "[367/1456]  skip 0021800367 (already collected)"
else
  echo "[367/1456]  2018-19 regular  0021800367"
  collect-possessions --game 0021800367 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800368.parquet" ]; then
  echo "[368/1456]  skip 0021800368 (already collected)"
else
  echo "[368/1456]  2018-19 regular  0021800368"
  collect-possessions --game 0021800368 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800369.parquet" ]; then
  echo "[369/1456]  skip 0021800369 (already collected)"
else
  echo "[369/1456]  2018-19 regular  0021800369"
  collect-possessions --game 0021800369 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800370.parquet" ]; then
  echo "[370/1456]  skip 0021800370 (already collected)"
else
  echo "[370/1456]  2018-19 regular  0021800370"
  collect-possessions --game 0021800370 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800371.parquet" ]; then
  echo "[371/1456]  skip 0021800371 (already collected)"
else
  echo "[371/1456]  2018-19 regular  0021800371"
  collect-possessions --game 0021800371 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800372.parquet" ]; then
  echo "[372/1456]  skip 0021800372 (already collected)"
else
  echo "[372/1456]  2018-19 regular  0021800372"
  collect-possessions --game 0021800372 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800373.parquet" ]; then
  echo "[373/1456]  skip 0021800373 (already collected)"
else
  echo "[373/1456]  2018-19 regular  0021800373"
  collect-possessions --game 0021800373 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800374.parquet" ]; then
  echo "[374/1456]  skip 0021800374 (already collected)"
else
  echo "[374/1456]  2018-19 regular  0021800374"
  collect-possessions --game 0021800374 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800375.parquet" ]; then
  echo "[375/1456]  skip 0021800375 (already collected)"
else
  echo "[375/1456]  2018-19 regular  0021800375"
  collect-possessions --game 0021800375 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800376.parquet" ]; then
  echo "[376/1456]  skip 0021800376 (already collected)"
else
  echo "[376/1456]  2018-19 regular  0021800376"
  collect-possessions --game 0021800376 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800377.parquet" ]; then
  echo "[377/1456]  skip 0021800377 (already collected)"
else
  echo "[377/1456]  2018-19 regular  0021800377"
  collect-possessions --game 0021800377 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800379.parquet" ]; then
  echo "[378/1456]  skip 0021800379 (already collected)"
else
  echo "[378/1456]  2018-19 regular  0021800379"
  collect-possessions --game 0021800379 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800381.parquet" ]; then
  echo "[379/1456]  skip 0021800381 (already collected)"
else
  echo "[379/1456]  2018-19 regular  0021800381"
  collect-possessions --game 0021800381 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800382.parquet" ]; then
  echo "[380/1456]  skip 0021800382 (already collected)"
else
  echo "[380/1456]  2018-19 regular  0021800382"
  collect-possessions --game 0021800382 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800383.parquet" ]; then
  echo "[381/1456]  skip 0021800383 (already collected)"
else
  echo "[381/1456]  2018-19 regular  0021800383"
  collect-possessions --game 0021800383 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800384.parquet" ]; then
  echo "[382/1456]  skip 0021800384 (already collected)"
else
  echo "[382/1456]  2018-19 regular  0021800384"
  collect-possessions --game 0021800384 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021800386.parquet" ]; then
  echo "[383/1456]  skip 0021800386 (already collected)"
else
  echo "[383/1456]  2018-19 regular  0021800386"
  collect-possessions --game 0021800386 --output-dir "$OUTDIR"
  sleep 0.5
fi

# ── 2019-20 regular (5 games) ──────────────────────────────
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021900023.parquet" ]; then
  echo "[384/1456]  skip 0021900023 (already collected)"
else
  echo "[384/1456]  2019-20 regular  0021900023"
  collect-possessions --game 0021900023 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021900120.parquet" ]; then
  echo "[385/1456]  skip 0021900120 (already collected)"
else
  echo "[385/1456]  2019-20 regular  0021900120"
  collect-possessions --game 0021900120 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021900272.parquet" ]; then
  echo "[386/1456]  skip 0021900272 (already collected)"
else
  echo "[386/1456]  2019-20 regular  0021900272"
  collect-possessions --game 0021900272 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021900696.parquet" ]; then
  echo "[387/1456]  skip 0021900696 (already collected)"
else
  echo "[387/1456]  2019-20 regular  0021900696"
  collect-possessions --game 0021900696 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0021900787.parquet" ]; then
  echo "[388/1456]  skip 0021900787 (already collected)"
else
  echo "[388/1456]  2019-20 regular  0021900787"
  collect-possessions --game 0021900787 --output-dir "$OUTDIR"
  sleep 0.5
fi

# ── 2020-21 regular (1 games) ──────────────────────────────
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022000645.parquet" ]; then
  echo "[389/1456]  skip 0022000645 (already collected)"
else
  echo "[389/1456]  2020-21 regular  0022000645"
  collect-possessions --game 0022000645 --output-dir "$OUTDIR"
  sleep 0.5
fi

# ── 2021-22 regular (308 games) ────────────────────────────
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100291.parquet" ]; then
  echo "[390/1456]  skip 0022100291 (already collected)"
else
  echo "[390/1456]  2021-22 regular  0022100291"
  collect-possessions --game 0022100291 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100430.parquet" ]; then
  echo "[391/1456]  skip 0022100430 (already collected)"
else
  echo "[391/1456]  2021-22 regular  0022100430"
  collect-possessions --game 0022100430 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100458.parquet" ]; then
  echo "[392/1456]  skip 0022100458 (already collected)"
else
  echo "[392/1456]  2021-22 regular  0022100458"
  collect-possessions --game 0022100458 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100464.parquet" ]; then
  echo "[393/1456]  skip 0022100464 (already collected)"
else
  echo "[393/1456]  2021-22 regular  0022100464"
  collect-possessions --game 0022100464 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100521.parquet" ]; then
  echo "[394/1456]  skip 0022100521 (already collected)"
else
  echo "[394/1456]  2021-22 regular  0022100521"
  collect-possessions --game 0022100521 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100528.parquet" ]; then
  echo "[395/1456]  skip 0022100528 (already collected)"
else
  echo "[395/1456]  2021-22 regular  0022100528"
  collect-possessions --game 0022100528 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100710.parquet" ]; then
  echo "[396/1456]  skip 0022100710 (already collected)"
else
  echo "[396/1456]  2021-22 regular  0022100710"
  collect-possessions --game 0022100710 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100726.parquet" ]; then
  echo "[397/1456]  skip 0022100726 (already collected)"
else
  echo "[397/1456]  2021-22 regular  0022100726"
  collect-possessions --game 0022100726 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100773.parquet" ]; then
  echo "[398/1456]  skip 0022100773 (already collected)"
else
  echo "[398/1456]  2021-22 regular  0022100773"
  collect-possessions --game 0022100773 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100774.parquet" ]; then
  echo "[399/1456]  skip 0022100774 (already collected)"
else
  echo "[399/1456]  2021-22 regular  0022100774"
  collect-possessions --game 0022100774 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100775.parquet" ]; then
  echo "[400/1456]  skip 0022100775 (already collected)"
else
  echo "[400/1456]  2021-22 regular  0022100775"
  collect-possessions --game 0022100775 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100776.parquet" ]; then
  echo "[401/1456]  skip 0022100776 (already collected)"
else
  echo "[401/1456]  2021-22 regular  0022100776"
  collect-possessions --game 0022100776 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100777.parquet" ]; then
  echo "[402/1456]  skip 0022100777 (already collected)"
else
  echo "[402/1456]  2021-22 regular  0022100777"
  collect-possessions --game 0022100777 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100778.parquet" ]; then
  echo "[403/1456]  skip 0022100778 (already collected)"
else
  echo "[403/1456]  2021-22 regular  0022100778"
  collect-possessions --game 0022100778 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100779.parquet" ]; then
  echo "[404/1456]  skip 0022100779 (already collected)"
else
  echo "[404/1456]  2021-22 regular  0022100779"
  collect-possessions --game 0022100779 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100780.parquet" ]; then
  echo "[405/1456]  skip 0022100780 (already collected)"
else
  echo "[405/1456]  2021-22 regular  0022100780"
  collect-possessions --game 0022100780 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100781.parquet" ]; then
  echo "[406/1456]  skip 0022100781 (already collected)"
else
  echo "[406/1456]  2021-22 regular  0022100781"
  collect-possessions --game 0022100781 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100782.parquet" ]; then
  echo "[407/1456]  skip 0022100782 (already collected)"
else
  echo "[407/1456]  2021-22 regular  0022100782"
  collect-possessions --game 0022100782 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100783.parquet" ]; then
  echo "[408/1456]  skip 0022100783 (already collected)"
else
  echo "[408/1456]  2021-22 regular  0022100783"
  collect-possessions --game 0022100783 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100785.parquet" ]; then
  echo "[409/1456]  skip 0022100785 (already collected)"
else
  echo "[409/1456]  2021-22 regular  0022100785"
  collect-possessions --game 0022100785 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100786.parquet" ]; then
  echo "[410/1456]  skip 0022100786 (already collected)"
else
  echo "[410/1456]  2021-22 regular  0022100786"
  collect-possessions --game 0022100786 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100787.parquet" ]; then
  echo "[411/1456]  skip 0022100787 (already collected)"
else
  echo "[411/1456]  2021-22 regular  0022100787"
  collect-possessions --game 0022100787 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100788.parquet" ]; then
  echo "[412/1456]  skip 0022100788 (already collected)"
else
  echo "[412/1456]  2021-22 regular  0022100788"
  collect-possessions --game 0022100788 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100789.parquet" ]; then
  echo "[413/1456]  skip 0022100789 (already collected)"
else
  echo "[413/1456]  2021-22 regular  0022100789"
  collect-possessions --game 0022100789 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100790.parquet" ]; then
  echo "[414/1456]  skip 0022100790 (already collected)"
else
  echo "[414/1456]  2021-22 regular  0022100790"
  collect-possessions --game 0022100790 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100791.parquet" ]; then
  echo "[415/1456]  skip 0022100791 (already collected)"
else
  echo "[415/1456]  2021-22 regular  0022100791"
  collect-possessions --game 0022100791 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100792.parquet" ]; then
  echo "[416/1456]  skip 0022100792 (already collected)"
else
  echo "[416/1456]  2021-22 regular  0022100792"
  collect-possessions --game 0022100792 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100793.parquet" ]; then
  echo "[417/1456]  skip 0022100793 (already collected)"
else
  echo "[417/1456]  2021-22 regular  0022100793"
  collect-possessions --game 0022100793 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100794.parquet" ]; then
  echo "[418/1456]  skip 0022100794 (already collected)"
else
  echo "[418/1456]  2021-22 regular  0022100794"
  collect-possessions --game 0022100794 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100795.parquet" ]; then
  echo "[419/1456]  skip 0022100795 (already collected)"
else
  echo "[419/1456]  2021-22 regular  0022100795"
  collect-possessions --game 0022100795 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100796.parquet" ]; then
  echo "[420/1456]  skip 0022100796 (already collected)"
else
  echo "[420/1456]  2021-22 regular  0022100796"
  collect-possessions --game 0022100796 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100797.parquet" ]; then
  echo "[421/1456]  skip 0022100797 (already collected)"
else
  echo "[421/1456]  2021-22 regular  0022100797"
  collect-possessions --game 0022100797 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100798.parquet" ]; then
  echo "[422/1456]  skip 0022100798 (already collected)"
else
  echo "[422/1456]  2021-22 regular  0022100798"
  collect-possessions --game 0022100798 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100799.parquet" ]; then
  echo "[423/1456]  skip 0022100799 (already collected)"
else
  echo "[423/1456]  2021-22 regular  0022100799"
  collect-possessions --game 0022100799 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100800.parquet" ]; then
  echo "[424/1456]  skip 0022100800 (already collected)"
else
  echo "[424/1456]  2021-22 regular  0022100800"
  collect-possessions --game 0022100800 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100801.parquet" ]; then
  echo "[425/1456]  skip 0022100801 (already collected)"
else
  echo "[425/1456]  2021-22 regular  0022100801"
  collect-possessions --game 0022100801 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100802.parquet" ]; then
  echo "[426/1456]  skip 0022100802 (already collected)"
else
  echo "[426/1456]  2021-22 regular  0022100802"
  collect-possessions --game 0022100802 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100803.parquet" ]; then
  echo "[427/1456]  skip 0022100803 (already collected)"
else
  echo "[427/1456]  2021-22 regular  0022100803"
  collect-possessions --game 0022100803 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100804.parquet" ]; then
  echo "[428/1456]  skip 0022100804 (already collected)"
else
  echo "[428/1456]  2021-22 regular  0022100804"
  collect-possessions --game 0022100804 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100805.parquet" ]; then
  echo "[429/1456]  skip 0022100805 (already collected)"
else
  echo "[429/1456]  2021-22 regular  0022100805"
  collect-possessions --game 0022100805 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100806.parquet" ]; then
  echo "[430/1456]  skip 0022100806 (already collected)"
else
  echo "[430/1456]  2021-22 regular  0022100806"
  collect-possessions --game 0022100806 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100807.parquet" ]; then
  echo "[431/1456]  skip 0022100807 (already collected)"
else
  echo "[431/1456]  2021-22 regular  0022100807"
  collect-possessions --game 0022100807 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100808.parquet" ]; then
  echo "[432/1456]  skip 0022100808 (already collected)"
else
  echo "[432/1456]  2021-22 regular  0022100808"
  collect-possessions --game 0022100808 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100809.parquet" ]; then
  echo "[433/1456]  skip 0022100809 (already collected)"
else
  echo "[433/1456]  2021-22 regular  0022100809"
  collect-possessions --game 0022100809 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100810.parquet" ]; then
  echo "[434/1456]  skip 0022100810 (already collected)"
else
  echo "[434/1456]  2021-22 regular  0022100810"
  collect-possessions --game 0022100810 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100811.parquet" ]; then
  echo "[435/1456]  skip 0022100811 (already collected)"
else
  echo "[435/1456]  2021-22 regular  0022100811"
  collect-possessions --game 0022100811 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100812.parquet" ]; then
  echo "[436/1456]  skip 0022100812 (already collected)"
else
  echo "[436/1456]  2021-22 regular  0022100812"
  collect-possessions --game 0022100812 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100813.parquet" ]; then
  echo "[437/1456]  skip 0022100813 (already collected)"
else
  echo "[437/1456]  2021-22 regular  0022100813"
  collect-possessions --game 0022100813 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100814.parquet" ]; then
  echo "[438/1456]  skip 0022100814 (already collected)"
else
  echo "[438/1456]  2021-22 regular  0022100814"
  collect-possessions --game 0022100814 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100815.parquet" ]; then
  echo "[439/1456]  skip 0022100815 (already collected)"
else
  echo "[439/1456]  2021-22 regular  0022100815"
  collect-possessions --game 0022100815 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100816.parquet" ]; then
  echo "[440/1456]  skip 0022100816 (already collected)"
else
  echo "[440/1456]  2021-22 regular  0022100816"
  collect-possessions --game 0022100816 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100817.parquet" ]; then
  echo "[441/1456]  skip 0022100817 (already collected)"
else
  echo "[441/1456]  2021-22 regular  0022100817"
  collect-possessions --game 0022100817 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100818.parquet" ]; then
  echo "[442/1456]  skip 0022100818 (already collected)"
else
  echo "[442/1456]  2021-22 regular  0022100818"
  collect-possessions --game 0022100818 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100819.parquet" ]; then
  echo "[443/1456]  skip 0022100819 (already collected)"
else
  echo "[443/1456]  2021-22 regular  0022100819"
  collect-possessions --game 0022100819 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100820.parquet" ]; then
  echo "[444/1456]  skip 0022100820 (already collected)"
else
  echo "[444/1456]  2021-22 regular  0022100820"
  collect-possessions --game 0022100820 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100821.parquet" ]; then
  echo "[445/1456]  skip 0022100821 (already collected)"
else
  echo "[445/1456]  2021-22 regular  0022100821"
  collect-possessions --game 0022100821 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100822.parquet" ]; then
  echo "[446/1456]  skip 0022100822 (already collected)"
else
  echo "[446/1456]  2021-22 regular  0022100822"
  collect-possessions --game 0022100822 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100823.parquet" ]; then
  echo "[447/1456]  skip 0022100823 (already collected)"
else
  echo "[447/1456]  2021-22 regular  0022100823"
  collect-possessions --game 0022100823 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100824.parquet" ]; then
  echo "[448/1456]  skip 0022100824 (already collected)"
else
  echo "[448/1456]  2021-22 regular  0022100824"
  collect-possessions --game 0022100824 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100825.parquet" ]; then
  echo "[449/1456]  skip 0022100825 (already collected)"
else
  echo "[449/1456]  2021-22 regular  0022100825"
  collect-possessions --game 0022100825 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100826.parquet" ]; then
  echo "[450/1456]  skip 0022100826 (already collected)"
else
  echo "[450/1456]  2021-22 regular  0022100826"
  collect-possessions --game 0022100826 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100827.parquet" ]; then
  echo "[451/1456]  skip 0022100827 (already collected)"
else
  echo "[451/1456]  2021-22 regular  0022100827"
  collect-possessions --game 0022100827 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100828.parquet" ]; then
  echo "[452/1456]  skip 0022100828 (already collected)"
else
  echo "[452/1456]  2021-22 regular  0022100828"
  collect-possessions --game 0022100828 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100829.parquet" ]; then
  echo "[453/1456]  skip 0022100829 (already collected)"
else
  echo "[453/1456]  2021-22 regular  0022100829"
  collect-possessions --game 0022100829 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100830.parquet" ]; then
  echo "[454/1456]  skip 0022100830 (already collected)"
else
  echo "[454/1456]  2021-22 regular  0022100830"
  collect-possessions --game 0022100830 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100831.parquet" ]; then
  echo "[455/1456]  skip 0022100831 (already collected)"
else
  echo "[455/1456]  2021-22 regular  0022100831"
  collect-possessions --game 0022100831 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100832.parquet" ]; then
  echo "[456/1456]  skip 0022100832 (already collected)"
else
  echo "[456/1456]  2021-22 regular  0022100832"
  collect-possessions --game 0022100832 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100833.parquet" ]; then
  echo "[457/1456]  skip 0022100833 (already collected)"
else
  echo "[457/1456]  2021-22 regular  0022100833"
  collect-possessions --game 0022100833 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100834.parquet" ]; then
  echo "[458/1456]  skip 0022100834 (already collected)"
else
  echo "[458/1456]  2021-22 regular  0022100834"
  collect-possessions --game 0022100834 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100835.parquet" ]; then
  echo "[459/1456]  skip 0022100835 (already collected)"
else
  echo "[459/1456]  2021-22 regular  0022100835"
  collect-possessions --game 0022100835 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100836.parquet" ]; then
  echo "[460/1456]  skip 0022100836 (already collected)"
else
  echo "[460/1456]  2021-22 regular  0022100836"
  collect-possessions --game 0022100836 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100837.parquet" ]; then
  echo "[461/1456]  skip 0022100837 (already collected)"
else
  echo "[461/1456]  2021-22 regular  0022100837"
  collect-possessions --game 0022100837 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100838.parquet" ]; then
  echo "[462/1456]  skip 0022100838 (already collected)"
else
  echo "[462/1456]  2021-22 regular  0022100838"
  collect-possessions --game 0022100838 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100839.parquet" ]; then
  echo "[463/1456]  skip 0022100839 (already collected)"
else
  echo "[463/1456]  2021-22 regular  0022100839"
  collect-possessions --game 0022100839 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100840.parquet" ]; then
  echo "[464/1456]  skip 0022100840 (already collected)"
else
  echo "[464/1456]  2021-22 regular  0022100840"
  collect-possessions --game 0022100840 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100841.parquet" ]; then
  echo "[465/1456]  skip 0022100841 (already collected)"
else
  echo "[465/1456]  2021-22 regular  0022100841"
  collect-possessions --game 0022100841 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100842.parquet" ]; then
  echo "[466/1456]  skip 0022100842 (already collected)"
else
  echo "[466/1456]  2021-22 regular  0022100842"
  collect-possessions --game 0022100842 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100843.parquet" ]; then
  echo "[467/1456]  skip 0022100843 (already collected)"
else
  echo "[467/1456]  2021-22 regular  0022100843"
  collect-possessions --game 0022100843 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100844.parquet" ]; then
  echo "[468/1456]  skip 0022100844 (already collected)"
else
  echo "[468/1456]  2021-22 regular  0022100844"
  collect-possessions --game 0022100844 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100845.parquet" ]; then
  echo "[469/1456]  skip 0022100845 (already collected)"
else
  echo "[469/1456]  2021-22 regular  0022100845"
  collect-possessions --game 0022100845 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100846.parquet" ]; then
  echo "[470/1456]  skip 0022100846 (already collected)"
else
  echo "[470/1456]  2021-22 regular  0022100846"
  collect-possessions --game 0022100846 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100847.parquet" ]; then
  echo "[471/1456]  skip 0022100847 (already collected)"
else
  echo "[471/1456]  2021-22 regular  0022100847"
  collect-possessions --game 0022100847 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100848.parquet" ]; then
  echo "[472/1456]  skip 0022100848 (already collected)"
else
  echo "[472/1456]  2021-22 regular  0022100848"
  collect-possessions --game 0022100848 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100849.parquet" ]; then
  echo "[473/1456]  skip 0022100849 (already collected)"
else
  echo "[473/1456]  2021-22 regular  0022100849"
  collect-possessions --game 0022100849 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100850.parquet" ]; then
  echo "[474/1456]  skip 0022100850 (already collected)"
else
  echo "[474/1456]  2021-22 regular  0022100850"
  collect-possessions --game 0022100850 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100851.parquet" ]; then
  echo "[475/1456]  skip 0022100851 (already collected)"
else
  echo "[475/1456]  2021-22 regular  0022100851"
  collect-possessions --game 0022100851 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100852.parquet" ]; then
  echo "[476/1456]  skip 0022100852 (already collected)"
else
  echo "[476/1456]  2021-22 regular  0022100852"
  collect-possessions --game 0022100852 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100853.parquet" ]; then
  echo "[477/1456]  skip 0022100853 (already collected)"
else
  echo "[477/1456]  2021-22 regular  0022100853"
  collect-possessions --game 0022100853 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100854.parquet" ]; then
  echo "[478/1456]  skip 0022100854 (already collected)"
else
  echo "[478/1456]  2021-22 regular  0022100854"
  collect-possessions --game 0022100854 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100855.parquet" ]; then
  echo "[479/1456]  skip 0022100855 (already collected)"
else
  echo "[479/1456]  2021-22 regular  0022100855"
  collect-possessions --game 0022100855 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100856.parquet" ]; then
  echo "[480/1456]  skip 0022100856 (already collected)"
else
  echo "[480/1456]  2021-22 regular  0022100856"
  collect-possessions --game 0022100856 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100857.parquet" ]; then
  echo "[481/1456]  skip 0022100857 (already collected)"
else
  echo "[481/1456]  2021-22 regular  0022100857"
  collect-possessions --game 0022100857 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100858.parquet" ]; then
  echo "[482/1456]  skip 0022100858 (already collected)"
else
  echo "[482/1456]  2021-22 regular  0022100858"
  collect-possessions --game 0022100858 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100859.parquet" ]; then
  echo "[483/1456]  skip 0022100859 (already collected)"
else
  echo "[483/1456]  2021-22 regular  0022100859"
  collect-possessions --game 0022100859 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100860.parquet" ]; then
  echo "[484/1456]  skip 0022100860 (already collected)"
else
  echo "[484/1456]  2021-22 regular  0022100860"
  collect-possessions --game 0022100860 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100861.parquet" ]; then
  echo "[485/1456]  skip 0022100861 (already collected)"
else
  echo "[485/1456]  2021-22 regular  0022100861"
  collect-possessions --game 0022100861 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100862.parquet" ]; then
  echo "[486/1456]  skip 0022100862 (already collected)"
else
  echo "[486/1456]  2021-22 regular  0022100862"
  collect-possessions --game 0022100862 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100863.parquet" ]; then
  echo "[487/1456]  skip 0022100863 (already collected)"
else
  echo "[487/1456]  2021-22 regular  0022100863"
  collect-possessions --game 0022100863 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100864.parquet" ]; then
  echo "[488/1456]  skip 0022100864 (already collected)"
else
  echo "[488/1456]  2021-22 regular  0022100864"
  collect-possessions --game 0022100864 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100865.parquet" ]; then
  echo "[489/1456]  skip 0022100865 (already collected)"
else
  echo "[489/1456]  2021-22 regular  0022100865"
  collect-possessions --game 0022100865 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100866.parquet" ]; then
  echo "[490/1456]  skip 0022100866 (already collected)"
else
  echo "[490/1456]  2021-22 regular  0022100866"
  collect-possessions --game 0022100866 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100867.parquet" ]; then
  echo "[491/1456]  skip 0022100867 (already collected)"
else
  echo "[491/1456]  2021-22 regular  0022100867"
  collect-possessions --game 0022100867 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100868.parquet" ]; then
  echo "[492/1456]  skip 0022100868 (already collected)"
else
  echo "[492/1456]  2021-22 regular  0022100868"
  collect-possessions --game 0022100868 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100869.parquet" ]; then
  echo "[493/1456]  skip 0022100869 (already collected)"
else
  echo "[493/1456]  2021-22 regular  0022100869"
  collect-possessions --game 0022100869 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100870.parquet" ]; then
  echo "[494/1456]  skip 0022100870 (already collected)"
else
  echo "[494/1456]  2021-22 regular  0022100870"
  collect-possessions --game 0022100870 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100871.parquet" ]; then
  echo "[495/1456]  skip 0022100871 (already collected)"
else
  echo "[495/1456]  2021-22 regular  0022100871"
  collect-possessions --game 0022100871 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100872.parquet" ]; then
  echo "[496/1456]  skip 0022100872 (already collected)"
else
  echo "[496/1456]  2021-22 regular  0022100872"
  collect-possessions --game 0022100872 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100873.parquet" ]; then
  echo "[497/1456]  skip 0022100873 (already collected)"
else
  echo "[497/1456]  2021-22 regular  0022100873"
  collect-possessions --game 0022100873 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100874.parquet" ]; then
  echo "[498/1456]  skip 0022100874 (already collected)"
else
  echo "[498/1456]  2021-22 regular  0022100874"
  collect-possessions --game 0022100874 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100875.parquet" ]; then
  echo "[499/1456]  skip 0022100875 (already collected)"
else
  echo "[499/1456]  2021-22 regular  0022100875"
  collect-possessions --game 0022100875 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100876.parquet" ]; then
  echo "[500/1456]  skip 0022100876 (already collected)"
else
  echo "[500/1456]  2021-22 regular  0022100876"
  collect-possessions --game 0022100876 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100877.parquet" ]; then
  echo "[501/1456]  skip 0022100877 (already collected)"
else
  echo "[501/1456]  2021-22 regular  0022100877"
  collect-possessions --game 0022100877 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100878.parquet" ]; then
  echo "[502/1456]  skip 0022100878 (already collected)"
else
  echo "[502/1456]  2021-22 regular  0022100878"
  collect-possessions --game 0022100878 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100879.parquet" ]; then
  echo "[503/1456]  skip 0022100879 (already collected)"
else
  echo "[503/1456]  2021-22 regular  0022100879"
  collect-possessions --game 0022100879 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100880.parquet" ]; then
  echo "[504/1456]  skip 0022100880 (already collected)"
else
  echo "[504/1456]  2021-22 regular  0022100880"
  collect-possessions --game 0022100880 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100881.parquet" ]; then
  echo "[505/1456]  skip 0022100881 (already collected)"
else
  echo "[505/1456]  2021-22 regular  0022100881"
  collect-possessions --game 0022100881 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100882.parquet" ]; then
  echo "[506/1456]  skip 0022100882 (already collected)"
else
  echo "[506/1456]  2021-22 regular  0022100882"
  collect-possessions --game 0022100882 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100883.parquet" ]; then
  echo "[507/1456]  skip 0022100883 (already collected)"
else
  echo "[507/1456]  2021-22 regular  0022100883"
  collect-possessions --game 0022100883 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100884.parquet" ]; then
  echo "[508/1456]  skip 0022100884 (already collected)"
else
  echo "[508/1456]  2021-22 regular  0022100884"
  collect-possessions --game 0022100884 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100885.parquet" ]; then
  echo "[509/1456]  skip 0022100885 (already collected)"
else
  echo "[509/1456]  2021-22 regular  0022100885"
  collect-possessions --game 0022100885 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100886.parquet" ]; then
  echo "[510/1456]  skip 0022100886 (already collected)"
else
  echo "[510/1456]  2021-22 regular  0022100886"
  collect-possessions --game 0022100886 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100887.parquet" ]; then
  echo "[511/1456]  skip 0022100887 (already collected)"
else
  echo "[511/1456]  2021-22 regular  0022100887"
  collect-possessions --game 0022100887 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100888.parquet" ]; then
  echo "[512/1456]  skip 0022100888 (already collected)"
else
  echo "[512/1456]  2021-22 regular  0022100888"
  collect-possessions --game 0022100888 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100889.parquet" ]; then
  echo "[513/1456]  skip 0022100889 (already collected)"
else
  echo "[513/1456]  2021-22 regular  0022100889"
  collect-possessions --game 0022100889 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100890.parquet" ]; then
  echo "[514/1456]  skip 0022100890 (already collected)"
else
  echo "[514/1456]  2021-22 regular  0022100890"
  collect-possessions --game 0022100890 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100891.parquet" ]; then
  echo "[515/1456]  skip 0022100891 (already collected)"
else
  echo "[515/1456]  2021-22 regular  0022100891"
  collect-possessions --game 0022100891 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100892.parquet" ]; then
  echo "[516/1456]  skip 0022100892 (already collected)"
else
  echo "[516/1456]  2021-22 regular  0022100892"
  collect-possessions --game 0022100892 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100893.parquet" ]; then
  echo "[517/1456]  skip 0022100893 (already collected)"
else
  echo "[517/1456]  2021-22 regular  0022100893"
  collect-possessions --game 0022100893 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100894.parquet" ]; then
  echo "[518/1456]  skip 0022100894 (already collected)"
else
  echo "[518/1456]  2021-22 regular  0022100894"
  collect-possessions --game 0022100894 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100895.parquet" ]; then
  echo "[519/1456]  skip 0022100895 (already collected)"
else
  echo "[519/1456]  2021-22 regular  0022100895"
  collect-possessions --game 0022100895 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100896.parquet" ]; then
  echo "[520/1456]  skip 0022100896 (already collected)"
else
  echo "[520/1456]  2021-22 regular  0022100896"
  collect-possessions --game 0022100896 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100897.parquet" ]; then
  echo "[521/1456]  skip 0022100897 (already collected)"
else
  echo "[521/1456]  2021-22 regular  0022100897"
  collect-possessions --game 0022100897 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100898.parquet" ]; then
  echo "[522/1456]  skip 0022100898 (already collected)"
else
  echo "[522/1456]  2021-22 regular  0022100898"
  collect-possessions --game 0022100898 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100899.parquet" ]; then
  echo "[523/1456]  skip 0022100899 (already collected)"
else
  echo "[523/1456]  2021-22 regular  0022100899"
  collect-possessions --game 0022100899 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100900.parquet" ]; then
  echo "[524/1456]  skip 0022100900 (already collected)"
else
  echo "[524/1456]  2021-22 regular  0022100900"
  collect-possessions --game 0022100900 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100901.parquet" ]; then
  echo "[525/1456]  skip 0022100901 (already collected)"
else
  echo "[525/1456]  2021-22 regular  0022100901"
  collect-possessions --game 0022100901 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100902.parquet" ]; then
  echo "[526/1456]  skip 0022100902 (already collected)"
else
  echo "[526/1456]  2021-22 regular  0022100902"
  collect-possessions --game 0022100902 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100903.parquet" ]; then
  echo "[527/1456]  skip 0022100903 (already collected)"
else
  echo "[527/1456]  2021-22 regular  0022100903"
  collect-possessions --game 0022100903 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100904.parquet" ]; then
  echo "[528/1456]  skip 0022100904 (already collected)"
else
  echo "[528/1456]  2021-22 regular  0022100904"
  collect-possessions --game 0022100904 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100905.parquet" ]; then
  echo "[529/1456]  skip 0022100905 (already collected)"
else
  echo "[529/1456]  2021-22 regular  0022100905"
  collect-possessions --game 0022100905 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100906.parquet" ]; then
  echo "[530/1456]  skip 0022100906 (already collected)"
else
  echo "[530/1456]  2021-22 regular  0022100906"
  collect-possessions --game 0022100906 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100907.parquet" ]; then
  echo "[531/1456]  skip 0022100907 (already collected)"
else
  echo "[531/1456]  2021-22 regular  0022100907"
  collect-possessions --game 0022100907 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100908.parquet" ]; then
  echo "[532/1456]  skip 0022100908 (already collected)"
else
  echo "[532/1456]  2021-22 regular  0022100908"
  collect-possessions --game 0022100908 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100909.parquet" ]; then
  echo "[533/1456]  skip 0022100909 (already collected)"
else
  echo "[533/1456]  2021-22 regular  0022100909"
  collect-possessions --game 0022100909 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100910.parquet" ]; then
  echo "[534/1456]  skip 0022100910 (already collected)"
else
  echo "[534/1456]  2021-22 regular  0022100910"
  collect-possessions --game 0022100910 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100911.parquet" ]; then
  echo "[535/1456]  skip 0022100911 (already collected)"
else
  echo "[535/1456]  2021-22 regular  0022100911"
  collect-possessions --game 0022100911 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100912.parquet" ]; then
  echo "[536/1456]  skip 0022100912 (already collected)"
else
  echo "[536/1456]  2021-22 regular  0022100912"
  collect-possessions --game 0022100912 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100913.parquet" ]; then
  echo "[537/1456]  skip 0022100913 (already collected)"
else
  echo "[537/1456]  2021-22 regular  0022100913"
  collect-possessions --game 0022100913 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100914.parquet" ]; then
  echo "[538/1456]  skip 0022100914 (already collected)"
else
  echo "[538/1456]  2021-22 regular  0022100914"
  collect-possessions --game 0022100914 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100915.parquet" ]; then
  echo "[539/1456]  skip 0022100915 (already collected)"
else
  echo "[539/1456]  2021-22 regular  0022100915"
  collect-possessions --game 0022100915 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100916.parquet" ]; then
  echo "[540/1456]  skip 0022100916 (already collected)"
else
  echo "[540/1456]  2021-22 regular  0022100916"
  collect-possessions --game 0022100916 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100917.parquet" ]; then
  echo "[541/1456]  skip 0022100917 (already collected)"
else
  echo "[541/1456]  2021-22 regular  0022100917"
  collect-possessions --game 0022100917 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100918.parquet" ]; then
  echo "[542/1456]  skip 0022100918 (already collected)"
else
  echo "[542/1456]  2021-22 regular  0022100918"
  collect-possessions --game 0022100918 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100919.parquet" ]; then
  echo "[543/1456]  skip 0022100919 (already collected)"
else
  echo "[543/1456]  2021-22 regular  0022100919"
  collect-possessions --game 0022100919 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100920.parquet" ]; then
  echo "[544/1456]  skip 0022100920 (already collected)"
else
  echo "[544/1456]  2021-22 regular  0022100920"
  collect-possessions --game 0022100920 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100921.parquet" ]; then
  echo "[545/1456]  skip 0022100921 (already collected)"
else
  echo "[545/1456]  2021-22 regular  0022100921"
  collect-possessions --game 0022100921 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100922.parquet" ]; then
  echo "[546/1456]  skip 0022100922 (already collected)"
else
  echo "[546/1456]  2021-22 regular  0022100922"
  collect-possessions --game 0022100922 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100923.parquet" ]; then
  echo "[547/1456]  skip 0022100923 (already collected)"
else
  echo "[547/1456]  2021-22 regular  0022100923"
  collect-possessions --game 0022100923 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100924.parquet" ]; then
  echo "[548/1456]  skip 0022100924 (already collected)"
else
  echo "[548/1456]  2021-22 regular  0022100924"
  collect-possessions --game 0022100924 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100925.parquet" ]; then
  echo "[549/1456]  skip 0022100925 (already collected)"
else
  echo "[549/1456]  2021-22 regular  0022100925"
  collect-possessions --game 0022100925 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100926.parquet" ]; then
  echo "[550/1456]  skip 0022100926 (already collected)"
else
  echo "[550/1456]  2021-22 regular  0022100926"
  collect-possessions --game 0022100926 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100927.parquet" ]; then
  echo "[551/1456]  skip 0022100927 (already collected)"
else
  echo "[551/1456]  2021-22 regular  0022100927"
  collect-possessions --game 0022100927 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100928.parquet" ]; then
  echo "[552/1456]  skip 0022100928 (already collected)"
else
  echo "[552/1456]  2021-22 regular  0022100928"
  collect-possessions --game 0022100928 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100929.parquet" ]; then
  echo "[553/1456]  skip 0022100929 (already collected)"
else
  echo "[553/1456]  2021-22 regular  0022100929"
  collect-possessions --game 0022100929 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100930.parquet" ]; then
  echo "[554/1456]  skip 0022100930 (already collected)"
else
  echo "[554/1456]  2021-22 regular  0022100930"
  collect-possessions --game 0022100930 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100931.parquet" ]; then
  echo "[555/1456]  skip 0022100931 (already collected)"
else
  echo "[555/1456]  2021-22 regular  0022100931"
  collect-possessions --game 0022100931 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100932.parquet" ]; then
  echo "[556/1456]  skip 0022100932 (already collected)"
else
  echo "[556/1456]  2021-22 regular  0022100932"
  collect-possessions --game 0022100932 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100933.parquet" ]; then
  echo "[557/1456]  skip 0022100933 (already collected)"
else
  echo "[557/1456]  2021-22 regular  0022100933"
  collect-possessions --game 0022100933 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100934.parquet" ]; then
  echo "[558/1456]  skip 0022100934 (already collected)"
else
  echo "[558/1456]  2021-22 regular  0022100934"
  collect-possessions --game 0022100934 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100935.parquet" ]; then
  echo "[559/1456]  skip 0022100935 (already collected)"
else
  echo "[559/1456]  2021-22 regular  0022100935"
  collect-possessions --game 0022100935 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100936.parquet" ]; then
  echo "[560/1456]  skip 0022100936 (already collected)"
else
  echo "[560/1456]  2021-22 regular  0022100936"
  collect-possessions --game 0022100936 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100937.parquet" ]; then
  echo "[561/1456]  skip 0022100937 (already collected)"
else
  echo "[561/1456]  2021-22 regular  0022100937"
  collect-possessions --game 0022100937 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100938.parquet" ]; then
  echo "[562/1456]  skip 0022100938 (already collected)"
else
  echo "[562/1456]  2021-22 regular  0022100938"
  collect-possessions --game 0022100938 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100939.parquet" ]; then
  echo "[563/1456]  skip 0022100939 (already collected)"
else
  echo "[563/1456]  2021-22 regular  0022100939"
  collect-possessions --game 0022100939 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100940.parquet" ]; then
  echo "[564/1456]  skip 0022100940 (already collected)"
else
  echo "[564/1456]  2021-22 regular  0022100940"
  collect-possessions --game 0022100940 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100941.parquet" ]; then
  echo "[565/1456]  skip 0022100941 (already collected)"
else
  echo "[565/1456]  2021-22 regular  0022100941"
  collect-possessions --game 0022100941 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100942.parquet" ]; then
  echo "[566/1456]  skip 0022100942 (already collected)"
else
  echo "[566/1456]  2021-22 regular  0022100942"
  collect-possessions --game 0022100942 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100943.parquet" ]; then
  echo "[567/1456]  skip 0022100943 (already collected)"
else
  echo "[567/1456]  2021-22 regular  0022100943"
  collect-possessions --game 0022100943 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100944.parquet" ]; then
  echo "[568/1456]  skip 0022100944 (already collected)"
else
  echo "[568/1456]  2021-22 regular  0022100944"
  collect-possessions --game 0022100944 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100945.parquet" ]; then
  echo "[569/1456]  skip 0022100945 (already collected)"
else
  echo "[569/1456]  2021-22 regular  0022100945"
  collect-possessions --game 0022100945 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100946.parquet" ]; then
  echo "[570/1456]  skip 0022100946 (already collected)"
else
  echo "[570/1456]  2021-22 regular  0022100946"
  collect-possessions --game 0022100946 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100947.parquet" ]; then
  echo "[571/1456]  skip 0022100947 (already collected)"
else
  echo "[571/1456]  2021-22 regular  0022100947"
  collect-possessions --game 0022100947 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100948.parquet" ]; then
  echo "[572/1456]  skip 0022100948 (already collected)"
else
  echo "[572/1456]  2021-22 regular  0022100948"
  collect-possessions --game 0022100948 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100949.parquet" ]; then
  echo "[573/1456]  skip 0022100949 (already collected)"
else
  echo "[573/1456]  2021-22 regular  0022100949"
  collect-possessions --game 0022100949 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100950.parquet" ]; then
  echo "[574/1456]  skip 0022100950 (already collected)"
else
  echo "[574/1456]  2021-22 regular  0022100950"
  collect-possessions --game 0022100950 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100951.parquet" ]; then
  echo "[575/1456]  skip 0022100951 (already collected)"
else
  echo "[575/1456]  2021-22 regular  0022100951"
  collect-possessions --game 0022100951 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100952.parquet" ]; then
  echo "[576/1456]  skip 0022100952 (already collected)"
else
  echo "[576/1456]  2021-22 regular  0022100952"
  collect-possessions --game 0022100952 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100953.parquet" ]; then
  echo "[577/1456]  skip 0022100953 (already collected)"
else
  echo "[577/1456]  2021-22 regular  0022100953"
  collect-possessions --game 0022100953 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100954.parquet" ]; then
  echo "[578/1456]  skip 0022100954 (already collected)"
else
  echo "[578/1456]  2021-22 regular  0022100954"
  collect-possessions --game 0022100954 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100955.parquet" ]; then
  echo "[579/1456]  skip 0022100955 (already collected)"
else
  echo "[579/1456]  2021-22 regular  0022100955"
  collect-possessions --game 0022100955 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100956.parquet" ]; then
  echo "[580/1456]  skip 0022100956 (already collected)"
else
  echo "[580/1456]  2021-22 regular  0022100956"
  collect-possessions --game 0022100956 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100957.parquet" ]; then
  echo "[581/1456]  skip 0022100957 (already collected)"
else
  echo "[581/1456]  2021-22 regular  0022100957"
  collect-possessions --game 0022100957 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100958.parquet" ]; then
  echo "[582/1456]  skip 0022100958 (already collected)"
else
  echo "[582/1456]  2021-22 regular  0022100958"
  collect-possessions --game 0022100958 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100959.parquet" ]; then
  echo "[583/1456]  skip 0022100959 (already collected)"
else
  echo "[583/1456]  2021-22 regular  0022100959"
  collect-possessions --game 0022100959 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100960.parquet" ]; then
  echo "[584/1456]  skip 0022100960 (already collected)"
else
  echo "[584/1456]  2021-22 regular  0022100960"
  collect-possessions --game 0022100960 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100961.parquet" ]; then
  echo "[585/1456]  skip 0022100961 (already collected)"
else
  echo "[585/1456]  2021-22 regular  0022100961"
  collect-possessions --game 0022100961 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100962.parquet" ]; then
  echo "[586/1456]  skip 0022100962 (already collected)"
else
  echo "[586/1456]  2021-22 regular  0022100962"
  collect-possessions --game 0022100962 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100963.parquet" ]; then
  echo "[587/1456]  skip 0022100963 (already collected)"
else
  echo "[587/1456]  2021-22 regular  0022100963"
  collect-possessions --game 0022100963 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100964.parquet" ]; then
  echo "[588/1456]  skip 0022100964 (already collected)"
else
  echo "[588/1456]  2021-22 regular  0022100964"
  collect-possessions --game 0022100964 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100965.parquet" ]; then
  echo "[589/1456]  skip 0022100965 (already collected)"
else
  echo "[589/1456]  2021-22 regular  0022100965"
  collect-possessions --game 0022100965 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100966.parquet" ]; then
  echo "[590/1456]  skip 0022100966 (already collected)"
else
  echo "[590/1456]  2021-22 regular  0022100966"
  collect-possessions --game 0022100966 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100967.parquet" ]; then
  echo "[591/1456]  skip 0022100967 (already collected)"
else
  echo "[591/1456]  2021-22 regular  0022100967"
  collect-possessions --game 0022100967 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100968.parquet" ]; then
  echo "[592/1456]  skip 0022100968 (already collected)"
else
  echo "[592/1456]  2021-22 regular  0022100968"
  collect-possessions --game 0022100968 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100969.parquet" ]; then
  echo "[593/1456]  skip 0022100969 (already collected)"
else
  echo "[593/1456]  2021-22 regular  0022100969"
  collect-possessions --game 0022100969 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100970.parquet" ]; then
  echo "[594/1456]  skip 0022100970 (already collected)"
else
  echo "[594/1456]  2021-22 regular  0022100970"
  collect-possessions --game 0022100970 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100971.parquet" ]; then
  echo "[595/1456]  skip 0022100971 (already collected)"
else
  echo "[595/1456]  2021-22 regular  0022100971"
  collect-possessions --game 0022100971 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100972.parquet" ]; then
  echo "[596/1456]  skip 0022100972 (already collected)"
else
  echo "[596/1456]  2021-22 regular  0022100972"
  collect-possessions --game 0022100972 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100973.parquet" ]; then
  echo "[597/1456]  skip 0022100973 (already collected)"
else
  echo "[597/1456]  2021-22 regular  0022100973"
  collect-possessions --game 0022100973 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100974.parquet" ]; then
  echo "[598/1456]  skip 0022100974 (already collected)"
else
  echo "[598/1456]  2021-22 regular  0022100974"
  collect-possessions --game 0022100974 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100975.parquet" ]; then
  echo "[599/1456]  skip 0022100975 (already collected)"
else
  echo "[599/1456]  2021-22 regular  0022100975"
  collect-possessions --game 0022100975 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100976.parquet" ]; then
  echo "[600/1456]  skip 0022100976 (already collected)"
else
  echo "[600/1456]  2021-22 regular  0022100976"
  collect-possessions --game 0022100976 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100977.parquet" ]; then
  echo "[601/1456]  skip 0022100977 (already collected)"
else
  echo "[601/1456]  2021-22 regular  0022100977"
  collect-possessions --game 0022100977 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100978.parquet" ]; then
  echo "[602/1456]  skip 0022100978 (already collected)"
else
  echo "[602/1456]  2021-22 regular  0022100978"
  collect-possessions --game 0022100978 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100979.parquet" ]; then
  echo "[603/1456]  skip 0022100979 (already collected)"
else
  echo "[603/1456]  2021-22 regular  0022100979"
  collect-possessions --game 0022100979 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100980.parquet" ]; then
  echo "[604/1456]  skip 0022100980 (already collected)"
else
  echo "[604/1456]  2021-22 regular  0022100980"
  collect-possessions --game 0022100980 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100981.parquet" ]; then
  echo "[605/1456]  skip 0022100981 (already collected)"
else
  echo "[605/1456]  2021-22 regular  0022100981"
  collect-possessions --game 0022100981 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100982.parquet" ]; then
  echo "[606/1456]  skip 0022100982 (already collected)"
else
  echo "[606/1456]  2021-22 regular  0022100982"
  collect-possessions --game 0022100982 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100983.parquet" ]; then
  echo "[607/1456]  skip 0022100983 (already collected)"
else
  echo "[607/1456]  2021-22 regular  0022100983"
  collect-possessions --game 0022100983 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100984.parquet" ]; then
  echo "[608/1456]  skip 0022100984 (already collected)"
else
  echo "[608/1456]  2021-22 regular  0022100984"
  collect-possessions --game 0022100984 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100985.parquet" ]; then
  echo "[609/1456]  skip 0022100985 (already collected)"
else
  echo "[609/1456]  2021-22 regular  0022100985"
  collect-possessions --game 0022100985 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100986.parquet" ]; then
  echo "[610/1456]  skip 0022100986 (already collected)"
else
  echo "[610/1456]  2021-22 regular  0022100986"
  collect-possessions --game 0022100986 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100987.parquet" ]; then
  echo "[611/1456]  skip 0022100987 (already collected)"
else
  echo "[611/1456]  2021-22 regular  0022100987"
  collect-possessions --game 0022100987 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100988.parquet" ]; then
  echo "[612/1456]  skip 0022100988 (already collected)"
else
  echo "[612/1456]  2021-22 regular  0022100988"
  collect-possessions --game 0022100988 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100989.parquet" ]; then
  echo "[613/1456]  skip 0022100989 (already collected)"
else
  echo "[613/1456]  2021-22 regular  0022100989"
  collect-possessions --game 0022100989 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100990.parquet" ]; then
  echo "[614/1456]  skip 0022100990 (already collected)"
else
  echo "[614/1456]  2021-22 regular  0022100990"
  collect-possessions --game 0022100990 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100991.parquet" ]; then
  echo "[615/1456]  skip 0022100991 (already collected)"
else
  echo "[615/1456]  2021-22 regular  0022100991"
  collect-possessions --game 0022100991 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100992.parquet" ]; then
  echo "[616/1456]  skip 0022100992 (already collected)"
else
  echo "[616/1456]  2021-22 regular  0022100992"
  collect-possessions --game 0022100992 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100993.parquet" ]; then
  echo "[617/1456]  skip 0022100993 (already collected)"
else
  echo "[617/1456]  2021-22 regular  0022100993"
  collect-possessions --game 0022100993 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100994.parquet" ]; then
  echo "[618/1456]  skip 0022100994 (already collected)"
else
  echo "[618/1456]  2021-22 regular  0022100994"
  collect-possessions --game 0022100994 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100995.parquet" ]; then
  echo "[619/1456]  skip 0022100995 (already collected)"
else
  echo "[619/1456]  2021-22 regular  0022100995"
  collect-possessions --game 0022100995 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100996.parquet" ]; then
  echo "[620/1456]  skip 0022100996 (already collected)"
else
  echo "[620/1456]  2021-22 regular  0022100996"
  collect-possessions --game 0022100996 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100997.parquet" ]; then
  echo "[621/1456]  skip 0022100997 (already collected)"
else
  echo "[621/1456]  2021-22 regular  0022100997"
  collect-possessions --game 0022100997 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100998.parquet" ]; then
  echo "[622/1456]  skip 0022100998 (already collected)"
else
  echo "[622/1456]  2021-22 regular  0022100998"
  collect-possessions --game 0022100998 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022100999.parquet" ]; then
  echo "[623/1456]  skip 0022100999 (already collected)"
else
  echo "[623/1456]  2021-22 regular  0022100999"
  collect-possessions --game 0022100999 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101000.parquet" ]; then
  echo "[624/1456]  skip 0022101000 (already collected)"
else
  echo "[624/1456]  2021-22 regular  0022101000"
  collect-possessions --game 0022101000 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101001.parquet" ]; then
  echo "[625/1456]  skip 0022101001 (already collected)"
else
  echo "[625/1456]  2021-22 regular  0022101001"
  collect-possessions --game 0022101001 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101002.parquet" ]; then
  echo "[626/1456]  skip 0022101002 (already collected)"
else
  echo "[626/1456]  2021-22 regular  0022101002"
  collect-possessions --game 0022101002 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101003.parquet" ]; then
  echo "[627/1456]  skip 0022101003 (already collected)"
else
  echo "[627/1456]  2021-22 regular  0022101003"
  collect-possessions --game 0022101003 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101004.parquet" ]; then
  echo "[628/1456]  skip 0022101004 (already collected)"
else
  echo "[628/1456]  2021-22 regular  0022101004"
  collect-possessions --game 0022101004 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101005.parquet" ]; then
  echo "[629/1456]  skip 0022101005 (already collected)"
else
  echo "[629/1456]  2021-22 regular  0022101005"
  collect-possessions --game 0022101005 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101006.parquet" ]; then
  echo "[630/1456]  skip 0022101006 (already collected)"
else
  echo "[630/1456]  2021-22 regular  0022101006"
  collect-possessions --game 0022101006 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101007.parquet" ]; then
  echo "[631/1456]  skip 0022101007 (already collected)"
else
  echo "[631/1456]  2021-22 regular  0022101007"
  collect-possessions --game 0022101007 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101008.parquet" ]; then
  echo "[632/1456]  skip 0022101008 (already collected)"
else
  echo "[632/1456]  2021-22 regular  0022101008"
  collect-possessions --game 0022101008 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101009.parquet" ]; then
  echo "[633/1456]  skip 0022101009 (already collected)"
else
  echo "[633/1456]  2021-22 regular  0022101009"
  collect-possessions --game 0022101009 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101010.parquet" ]; then
  echo "[634/1456]  skip 0022101010 (already collected)"
else
  echo "[634/1456]  2021-22 regular  0022101010"
  collect-possessions --game 0022101010 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101011.parquet" ]; then
  echo "[635/1456]  skip 0022101011 (already collected)"
else
  echo "[635/1456]  2021-22 regular  0022101011"
  collect-possessions --game 0022101011 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101012.parquet" ]; then
  echo "[636/1456]  skip 0022101012 (already collected)"
else
  echo "[636/1456]  2021-22 regular  0022101012"
  collect-possessions --game 0022101012 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101013.parquet" ]; then
  echo "[637/1456]  skip 0022101013 (already collected)"
else
  echo "[637/1456]  2021-22 regular  0022101013"
  collect-possessions --game 0022101013 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101014.parquet" ]; then
  echo "[638/1456]  skip 0022101014 (already collected)"
else
  echo "[638/1456]  2021-22 regular  0022101014"
  collect-possessions --game 0022101014 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101015.parquet" ]; then
  echo "[639/1456]  skip 0022101015 (already collected)"
else
  echo "[639/1456]  2021-22 regular  0022101015"
  collect-possessions --game 0022101015 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101016.parquet" ]; then
  echo "[640/1456]  skip 0022101016 (already collected)"
else
  echo "[640/1456]  2021-22 regular  0022101016"
  collect-possessions --game 0022101016 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101017.parquet" ]; then
  echo "[641/1456]  skip 0022101017 (already collected)"
else
  echo "[641/1456]  2021-22 regular  0022101017"
  collect-possessions --game 0022101017 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101018.parquet" ]; then
  echo "[642/1456]  skip 0022101018 (already collected)"
else
  echo "[642/1456]  2021-22 regular  0022101018"
  collect-possessions --game 0022101018 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101019.parquet" ]; then
  echo "[643/1456]  skip 0022101019 (already collected)"
else
  echo "[643/1456]  2021-22 regular  0022101019"
  collect-possessions --game 0022101019 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101020.parquet" ]; then
  echo "[644/1456]  skip 0022101020 (already collected)"
else
  echo "[644/1456]  2021-22 regular  0022101020"
  collect-possessions --game 0022101020 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101021.parquet" ]; then
  echo "[645/1456]  skip 0022101021 (already collected)"
else
  echo "[645/1456]  2021-22 regular  0022101021"
  collect-possessions --game 0022101021 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101022.parquet" ]; then
  echo "[646/1456]  skip 0022101022 (already collected)"
else
  echo "[646/1456]  2021-22 regular  0022101022"
  collect-possessions --game 0022101022 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101023.parquet" ]; then
  echo "[647/1456]  skip 0022101023 (already collected)"
else
  echo "[647/1456]  2021-22 regular  0022101023"
  collect-possessions --game 0022101023 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101024.parquet" ]; then
  echo "[648/1456]  skip 0022101024 (already collected)"
else
  echo "[648/1456]  2021-22 regular  0022101024"
  collect-possessions --game 0022101024 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101025.parquet" ]; then
  echo "[649/1456]  skip 0022101025 (already collected)"
else
  echo "[649/1456]  2021-22 regular  0022101025"
  collect-possessions --game 0022101025 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101026.parquet" ]; then
  echo "[650/1456]  skip 0022101026 (already collected)"
else
  echo "[650/1456]  2021-22 regular  0022101026"
  collect-possessions --game 0022101026 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101027.parquet" ]; then
  echo "[651/1456]  skip 0022101027 (already collected)"
else
  echo "[651/1456]  2021-22 regular  0022101027"
  collect-possessions --game 0022101027 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101028.parquet" ]; then
  echo "[652/1456]  skip 0022101028 (already collected)"
else
  echo "[652/1456]  2021-22 regular  0022101028"
  collect-possessions --game 0022101028 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101029.parquet" ]; then
  echo "[653/1456]  skip 0022101029 (already collected)"
else
  echo "[653/1456]  2021-22 regular  0022101029"
  collect-possessions --game 0022101029 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101031.parquet" ]; then
  echo "[654/1456]  skip 0022101031 (already collected)"
else
  echo "[654/1456]  2021-22 regular  0022101031"
  collect-possessions --game 0022101031 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101032.parquet" ]; then
  echo "[655/1456]  skip 0022101032 (already collected)"
else
  echo "[655/1456]  2021-22 regular  0022101032"
  collect-possessions --game 0022101032 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101034.parquet" ]; then
  echo "[656/1456]  skip 0022101034 (already collected)"
else
  echo "[656/1456]  2021-22 regular  0022101034"
  collect-possessions --game 0022101034 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101035.parquet" ]; then
  echo "[657/1456]  skip 0022101035 (already collected)"
else
  echo "[657/1456]  2021-22 regular  0022101035"
  collect-possessions --game 0022101035 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101036.parquet" ]; then
  echo "[658/1456]  skip 0022101036 (already collected)"
else
  echo "[658/1456]  2021-22 regular  0022101036"
  collect-possessions --game 0022101036 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101037.parquet" ]; then
  echo "[659/1456]  skip 0022101037 (already collected)"
else
  echo "[659/1456]  2021-22 regular  0022101037"
  collect-possessions --game 0022101037 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101038.parquet" ]; then
  echo "[660/1456]  skip 0022101038 (already collected)"
else
  echo "[660/1456]  2021-22 regular  0022101038"
  collect-possessions --game 0022101038 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101040.parquet" ]; then
  echo "[661/1456]  skip 0022101040 (already collected)"
else
  echo "[661/1456]  2021-22 regular  0022101040"
  collect-possessions --game 0022101040 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101041.parquet" ]; then
  echo "[662/1456]  skip 0022101041 (already collected)"
else
  echo "[662/1456]  2021-22 regular  0022101041"
  collect-possessions --game 0022101041 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101042.parquet" ]; then
  echo "[663/1456]  skip 0022101042 (already collected)"
else
  echo "[663/1456]  2021-22 regular  0022101042"
  collect-possessions --game 0022101042 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101043.parquet" ]; then
  echo "[664/1456]  skip 0022101043 (already collected)"
else
  echo "[664/1456]  2021-22 regular  0022101043"
  collect-possessions --game 0022101043 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101044.parquet" ]; then
  echo "[665/1456]  skip 0022101044 (already collected)"
else
  echo "[665/1456]  2021-22 regular  0022101044"
  collect-possessions --game 0022101044 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101045.parquet" ]; then
  echo "[666/1456]  skip 0022101045 (already collected)"
else
  echo "[666/1456]  2021-22 regular  0022101045"
  collect-possessions --game 0022101045 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101046.parquet" ]; then
  echo "[667/1456]  skip 0022101046 (already collected)"
else
  echo "[667/1456]  2021-22 regular  0022101046"
  collect-possessions --game 0022101046 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101047.parquet" ]; then
  echo "[668/1456]  skip 0022101047 (already collected)"
else
  echo "[668/1456]  2021-22 regular  0022101047"
  collect-possessions --game 0022101047 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101048.parquet" ]; then
  echo "[669/1456]  skip 0022101048 (already collected)"
else
  echo "[669/1456]  2021-22 regular  0022101048"
  collect-possessions --game 0022101048 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101049.parquet" ]; then
  echo "[670/1456]  skip 0022101049 (already collected)"
else
  echo "[670/1456]  2021-22 regular  0022101049"
  collect-possessions --game 0022101049 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101050.parquet" ]; then
  echo "[671/1456]  skip 0022101050 (already collected)"
else
  echo "[671/1456]  2021-22 regular  0022101050"
  collect-possessions --game 0022101050 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101051.parquet" ]; then
  echo "[672/1456]  skip 0022101051 (already collected)"
else
  echo "[672/1456]  2021-22 regular  0022101051"
  collect-possessions --game 0022101051 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101052.parquet" ]; then
  echo "[673/1456]  skip 0022101052 (already collected)"
else
  echo "[673/1456]  2021-22 regular  0022101052"
  collect-possessions --game 0022101052 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101053.parquet" ]; then
  echo "[674/1456]  skip 0022101053 (already collected)"
else
  echo "[674/1456]  2021-22 regular  0022101053"
  collect-possessions --game 0022101053 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101054.parquet" ]; then
  echo "[675/1456]  skip 0022101054 (already collected)"
else
  echo "[675/1456]  2021-22 regular  0022101054"
  collect-possessions --game 0022101054 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101055.parquet" ]; then
  echo "[676/1456]  skip 0022101055 (already collected)"
else
  echo "[676/1456]  2021-22 regular  0022101055"
  collect-possessions --game 0022101055 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101056.parquet" ]; then
  echo "[677/1456]  skip 0022101056 (already collected)"
else
  echo "[677/1456]  2021-22 regular  0022101056"
  collect-possessions --game 0022101056 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101057.parquet" ]; then
  echo "[678/1456]  skip 0022101057 (already collected)"
else
  echo "[678/1456]  2021-22 regular  0022101057"
  collect-possessions --game 0022101057 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101058.parquet" ]; then
  echo "[679/1456]  skip 0022101058 (already collected)"
else
  echo "[679/1456]  2021-22 regular  0022101058"
  collect-possessions --game 0022101058 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101059.parquet" ]; then
  echo "[680/1456]  skip 0022101059 (already collected)"
else
  echo "[680/1456]  2021-22 regular  0022101059"
  collect-possessions --game 0022101059 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101060.parquet" ]; then
  echo "[681/1456]  skip 0022101060 (already collected)"
else
  echo "[681/1456]  2021-22 regular  0022101060"
  collect-possessions --game 0022101060 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101061.parquet" ]; then
  echo "[682/1456]  skip 0022101061 (already collected)"
else
  echo "[682/1456]  2021-22 regular  0022101061"
  collect-possessions --game 0022101061 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101063.parquet" ]; then
  echo "[683/1456]  skip 0022101063 (already collected)"
else
  echo "[683/1456]  2021-22 regular  0022101063"
  collect-possessions --game 0022101063 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101064.parquet" ]; then
  echo "[684/1456]  skip 0022101064 (already collected)"
else
  echo "[684/1456]  2021-22 regular  0022101064"
  collect-possessions --game 0022101064 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101065.parquet" ]; then
  echo "[685/1456]  skip 0022101065 (already collected)"
else
  echo "[685/1456]  2021-22 regular  0022101065"
  collect-possessions --game 0022101065 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101066.parquet" ]; then
  echo "[686/1456]  skip 0022101066 (already collected)"
else
  echo "[686/1456]  2021-22 regular  0022101066"
  collect-possessions --game 0022101066 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101067.parquet" ]; then
  echo "[687/1456]  skip 0022101067 (already collected)"
else
  echo "[687/1456]  2021-22 regular  0022101067"
  collect-possessions --game 0022101067 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101068.parquet" ]; then
  echo "[688/1456]  skip 0022101068 (already collected)"
else
  echo "[688/1456]  2021-22 regular  0022101068"
  collect-possessions --game 0022101068 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101069.parquet" ]; then
  echo "[689/1456]  skip 0022101069 (already collected)"
else
  echo "[689/1456]  2021-22 regular  0022101069"
  collect-possessions --game 0022101069 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101070.parquet" ]; then
  echo "[690/1456]  skip 0022101070 (already collected)"
else
  echo "[690/1456]  2021-22 regular  0022101070"
  collect-possessions --game 0022101070 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101071.parquet" ]; then
  echo "[691/1456]  skip 0022101071 (already collected)"
else
  echo "[691/1456]  2021-22 regular  0022101071"
  collect-possessions --game 0022101071 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101073.parquet" ]; then
  echo "[692/1456]  skip 0022101073 (already collected)"
else
  echo "[692/1456]  2021-22 regular  0022101073"
  collect-possessions --game 0022101073 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101074.parquet" ]; then
  echo "[693/1456]  skip 0022101074 (already collected)"
else
  echo "[693/1456]  2021-22 regular  0022101074"
  collect-possessions --game 0022101074 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101075.parquet" ]; then
  echo "[694/1456]  skip 0022101075 (already collected)"
else
  echo "[694/1456]  2021-22 regular  0022101075"
  collect-possessions --game 0022101075 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101076.parquet" ]; then
  echo "[695/1456]  skip 0022101076 (already collected)"
else
  echo "[695/1456]  2021-22 regular  0022101076"
  collect-possessions --game 0022101076 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101077.parquet" ]; then
  echo "[696/1456]  skip 0022101077 (already collected)"
else
  echo "[696/1456]  2021-22 regular  0022101077"
  collect-possessions --game 0022101077 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022101078.parquet" ]; then
  echo "[697/1456]  skip 0022101078 (already collected)"
else
  echo "[697/1456]  2021-22 regular  0022101078"
  collect-possessions --game 0022101078 --output-dir "$OUTDIR"
  sleep 0.5
fi

# ── 2022-23 regular (3 games) ──────────────────────────────
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022200025.parquet" ]; then
  echo "[698/1456]  skip 0022200025 (already collected)"
else
  echo "[698/1456]  2022-23 regular  0022200025"
  collect-possessions --game 0022200025 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022201194.parquet" ]; then
  echo "[699/1456]  skip 0022201194 (already collected)"
else
  echo "[699/1456]  2022-23 regular  0022201194"
  collect-possessions --game 0022201194 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022201205.parquet" ]; then
  echo "[700/1456]  skip 0022201205 (already collected)"
else
  echo "[700/1456]  2022-23 regular  0022201205"
  collect-possessions --game 0022201205 --output-dir "$OUTDIR"
  sleep 0.5
fi

# ── 2023-24 regular (5 games) ──────────────────────────────
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022300512.parquet" ]; then
  echo "[701/1456]  skip 0022300512 (already collected)"
else
  echo "[701/1456]  2023-24 regular  0022300512"
  collect-possessions --game 0022300512 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022300517.parquet" ]; then
  echo "[702/1456]  skip 0022300517 (already collected)"
else
  echo "[702/1456]  2023-24 regular  0022300517"
  collect-possessions --game 0022300517 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022300623.parquet" ]; then
  echo "[703/1456]  skip 0022300623 (already collected)"
else
  echo "[703/1456]  2023-24 regular  0022300623"
  collect-possessions --game 0022300623 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022300710.parquet" ]; then
  echo "[704/1456]  skip 0022300710 (already collected)"
else
  echo "[704/1456]  2023-24 regular  0022300710"
  collect-possessions --game 0022300710 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022300856.parquet" ]; then
  echo "[705/1456]  skip 0022300856 (already collected)"
else
  echo "[705/1456]  2023-24 regular  0022300856"
  collect-possessions --game 0022300856 --output-dir "$OUTDIR"
  sleep 0.5
fi

# ── 2024-25 regular (2 games) ──────────────────────────────
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022400390.parquet" ]; then
  echo "[706/1456]  skip 0022400390 (already collected)"
else
  echo "[706/1456]  2024-25 regular  0022400390"
  collect-possessions --game 0022400390 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022400762.parquet" ]; then
  echo "[707/1456]  skip 0022400762 (already collected)"
else
  echo "[707/1456]  2024-25 regular  0022400762"
  collect-possessions --game 0022400762 --output-dir "$OUTDIR"
  sleep 0.5
fi

# ── 2025-26 regular (162 games) ────────────────────────────
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500057.parquet" ]; then
  echo "[708/1456]  skip 0022500057 (already collected)"
else
  echo "[708/1456]  2025-26 regular  0022500057"
  collect-possessions --game 0022500057 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500058.parquet" ]; then
  echo "[709/1456]  skip 0022500058 (already collected)"
else
  echo "[709/1456]  2025-26 regular  0022500058"
  collect-possessions --game 0022500058 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500059.parquet" ]; then
  echo "[710/1456]  skip 0022500059 (already collected)"
else
  echo "[710/1456]  2025-26 regular  0022500059"
  collect-possessions --game 0022500059 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500060.parquet" ]; then
  echo "[711/1456]  skip 0022500060 (already collected)"
else
  echo "[711/1456]  2025-26 regular  0022500060"
  collect-possessions --game 0022500060 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500061.parquet" ]; then
  echo "[712/1456]  skip 0022500061 (already collected)"
else
  echo "[712/1456]  2025-26 regular  0022500061"
  collect-possessions --game 0022500061 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500062.parquet" ]; then
  echo "[713/1456]  skip 0022500062 (already collected)"
else
  echo "[713/1456]  2025-26 regular  0022500062"
  collect-possessions --game 0022500062 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500063.parquet" ]; then
  echo "[714/1456]  skip 0022500063 (already collected)"
else
  echo "[714/1456]  2025-26 regular  0022500063"
  collect-possessions --game 0022500063 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500064.parquet" ]; then
  echo "[715/1456]  skip 0022500064 (already collected)"
else
  echo "[715/1456]  2025-26 regular  0022500064"
  collect-possessions --game 0022500064 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500065.parquet" ]; then
  echo "[716/1456]  skip 0022500065 (already collected)"
else
  echo "[716/1456]  2025-26 regular  0022500065"
  collect-possessions --game 0022500065 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500066.parquet" ]; then
  echo "[717/1456]  skip 0022500066 (already collected)"
else
  echo "[717/1456]  2025-26 regular  0022500066"
  collect-possessions --game 0022500066 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500067.parquet" ]; then
  echo "[718/1456]  skip 0022500067 (already collected)"
else
  echo "[718/1456]  2025-26 regular  0022500067"
  collect-possessions --game 0022500067 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500068.parquet" ]; then
  echo "[719/1456]  skip 0022500068 (already collected)"
else
  echo "[719/1456]  2025-26 regular  0022500068"
  collect-possessions --game 0022500068 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500069.parquet" ]; then
  echo "[720/1456]  skip 0022500069 (already collected)"
else
  echo "[720/1456]  2025-26 regular  0022500069"
  collect-possessions --game 0022500069 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500070.parquet" ]; then
  echo "[721/1456]  skip 0022500070 (already collected)"
else
  echo "[721/1456]  2025-26 regular  0022500070"
  collect-possessions --game 0022500070 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500071.parquet" ]; then
  echo "[722/1456]  skip 0022500071 (already collected)"
else
  echo "[722/1456]  2025-26 regular  0022500071"
  collect-possessions --game 0022500071 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500072.parquet" ]; then
  echo "[723/1456]  skip 0022500072 (already collected)"
else
  echo "[723/1456]  2025-26 regular  0022500072"
  collect-possessions --game 0022500072 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500073.parquet" ]; then
  echo "[724/1456]  skip 0022500073 (already collected)"
else
  echo "[724/1456]  2025-26 regular  0022500073"
  collect-possessions --game 0022500073 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500074.parquet" ]; then
  echo "[725/1456]  skip 0022500074 (already collected)"
else
  echo "[725/1456]  2025-26 regular  0022500074"
  collect-possessions --game 0022500074 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500075.parquet" ]; then
  echo "[726/1456]  skip 0022500075 (already collected)"
else
  echo "[726/1456]  2025-26 regular  0022500075"
  collect-possessions --game 0022500075 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500076.parquet" ]; then
  echo "[727/1456]  skip 0022500076 (already collected)"
else
  echo "[727/1456]  2025-26 regular  0022500076"
  collect-possessions --game 0022500076 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500077.parquet" ]; then
  echo "[728/1456]  skip 0022500077 (already collected)"
else
  echo "[728/1456]  2025-26 regular  0022500077"
  collect-possessions --game 0022500077 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500078.parquet" ]; then
  echo "[729/1456]  skip 0022500078 (already collected)"
else
  echo "[729/1456]  2025-26 regular  0022500078"
  collect-possessions --game 0022500078 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500079.parquet" ]; then
  echo "[730/1456]  skip 0022500079 (already collected)"
else
  echo "[730/1456]  2025-26 regular  0022500079"
  collect-possessions --game 0022500079 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500242.parquet" ]; then
  echo "[731/1456]  skip 0022500242 (already collected)"
else
  echo "[731/1456]  2025-26 regular  0022500242"
  collect-possessions --game 0022500242 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500243.parquet" ]; then
  echo "[732/1456]  skip 0022500243 (already collected)"
else
  echo "[732/1456]  2025-26 regular  0022500243"
  collect-possessions --game 0022500243 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500244.parquet" ]; then
  echo "[733/1456]  skip 0022500244 (already collected)"
else
  echo "[733/1456]  2025-26 regular  0022500244"
  collect-possessions --game 0022500244 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500245.parquet" ]; then
  echo "[734/1456]  skip 0022500245 (already collected)"
else
  echo "[734/1456]  2025-26 regular  0022500245"
  collect-possessions --game 0022500245 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500246.parquet" ]; then
  echo "[735/1456]  skip 0022500246 (already collected)"
else
  echo "[735/1456]  2025-26 regular  0022500246"
  collect-possessions --game 0022500246 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500247.parquet" ]; then
  echo "[736/1456]  skip 0022500247 (already collected)"
else
  echo "[736/1456]  2025-26 regular  0022500247"
  collect-possessions --game 0022500247 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500248.parquet" ]; then
  echo "[737/1456]  skip 0022500248 (already collected)"
else
  echo "[737/1456]  2025-26 regular  0022500248"
  collect-possessions --game 0022500248 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500249.parquet" ]; then
  echo "[738/1456]  skip 0022500249 (already collected)"
else
  echo "[738/1456]  2025-26 regular  0022500249"
  collect-possessions --game 0022500249 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500250.parquet" ]; then
  echo "[739/1456]  skip 0022500250 (already collected)"
else
  echo "[739/1456]  2025-26 regular  0022500250"
  collect-possessions --game 0022500250 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500251.parquet" ]; then
  echo "[740/1456]  skip 0022500251 (already collected)"
else
  echo "[740/1456]  2025-26 regular  0022500251"
  collect-possessions --game 0022500251 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500252.parquet" ]; then
  echo "[741/1456]  skip 0022500252 (already collected)"
else
  echo "[741/1456]  2025-26 regular  0022500252"
  collect-possessions --game 0022500252 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500253.parquet" ]; then
  echo "[742/1456]  skip 0022500253 (already collected)"
else
  echo "[742/1456]  2025-26 regular  0022500253"
  collect-possessions --game 0022500253 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500254.parquet" ]; then
  echo "[743/1456]  skip 0022500254 (already collected)"
else
  echo "[743/1456]  2025-26 regular  0022500254"
  collect-possessions --game 0022500254 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500255.parquet" ]; then
  echo "[744/1456]  skip 0022500255 (already collected)"
else
  echo "[744/1456]  2025-26 regular  0022500255"
  collect-possessions --game 0022500255 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500256.parquet" ]; then
  echo "[745/1456]  skip 0022500256 (already collected)"
else
  echo "[745/1456]  2025-26 regular  0022500256"
  collect-possessions --game 0022500256 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500257.parquet" ]; then
  echo "[746/1456]  skip 0022500257 (already collected)"
else
  echo "[746/1456]  2025-26 regular  0022500257"
  collect-possessions --game 0022500257 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500259.parquet" ]; then
  echo "[747/1456]  skip 0022500259 (already collected)"
else
  echo "[747/1456]  2025-26 regular  0022500259"
  collect-possessions --game 0022500259 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500260.parquet" ]; then
  echo "[748/1456]  skip 0022500260 (already collected)"
else
  echo "[748/1456]  2025-26 regular  0022500260"
  collect-possessions --game 0022500260 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500261.parquet" ]; then
  echo "[749/1456]  skip 0022500261 (already collected)"
else
  echo "[749/1456]  2025-26 regular  0022500261"
  collect-possessions --game 0022500261 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500262.parquet" ]; then
  echo "[750/1456]  skip 0022500262 (already collected)"
else
  echo "[750/1456]  2025-26 regular  0022500262"
  collect-possessions --game 0022500262 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500263.parquet" ]; then
  echo "[751/1456]  skip 0022500263 (already collected)"
else
  echo "[751/1456]  2025-26 regular  0022500263"
  collect-possessions --game 0022500263 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500265.parquet" ]; then
  echo "[752/1456]  skip 0022500265 (already collected)"
else
  echo "[752/1456]  2025-26 regular  0022500265"
  collect-possessions --game 0022500265 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500293.parquet" ]; then
  echo "[753/1456]  skip 0022500293 (already collected)"
else
  echo "[753/1456]  2025-26 regular  0022500293"
  collect-possessions --game 0022500293 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500294.parquet" ]; then
  echo "[754/1456]  skip 0022500294 (already collected)"
else
  echo "[754/1456]  2025-26 regular  0022500294"
  collect-possessions --game 0022500294 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500295.parquet" ]; then
  echo "[755/1456]  skip 0022500295 (already collected)"
else
  echo "[755/1456]  2025-26 regular  0022500295"
  collect-possessions --game 0022500295 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500296.parquet" ]; then
  echo "[756/1456]  skip 0022500296 (already collected)"
else
  echo "[756/1456]  2025-26 regular  0022500296"
  collect-possessions --game 0022500296 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500297.parquet" ]; then
  echo "[757/1456]  skip 0022500297 (already collected)"
else
  echo "[757/1456]  2025-26 regular  0022500297"
  collect-possessions --game 0022500297 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500298.parquet" ]; then
  echo "[758/1456]  skip 0022500298 (already collected)"
else
  echo "[758/1456]  2025-26 regular  0022500298"
  collect-possessions --game 0022500298 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500299.parquet" ]; then
  echo "[759/1456]  skip 0022500299 (already collected)"
else
  echo "[759/1456]  2025-26 regular  0022500299"
  collect-possessions --game 0022500299 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500300.parquet" ]; then
  echo "[760/1456]  skip 0022500300 (already collected)"
else
  echo "[760/1456]  2025-26 regular  0022500300"
  collect-possessions --game 0022500300 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500302.parquet" ]; then
  echo "[761/1456]  skip 0022500302 (already collected)"
else
  echo "[761/1456]  2025-26 regular  0022500302"
  collect-possessions --game 0022500302 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500304.parquet" ]; then
  echo "[762/1456]  skip 0022500304 (already collected)"
else
  echo "[762/1456]  2025-26 regular  0022500304"
  collect-possessions --game 0022500304 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500467.parquet" ]; then
  echo "[763/1456]  skip 0022500467 (already collected)"
else
  echo "[763/1456]  2025-26 regular  0022500467"
  collect-possessions --game 0022500467 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500469.parquet" ]; then
  echo "[764/1456]  skip 0022500469 (already collected)"
else
  echo "[764/1456]  2025-26 regular  0022500469"
  collect-possessions --game 0022500469 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500470.parquet" ]; then
  echo "[765/1456]  skip 0022500470 (already collected)"
else
  echo "[765/1456]  2025-26 regular  0022500470"
  collect-possessions --game 0022500470 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500471.parquet" ]; then
  echo "[766/1456]  skip 0022500471 (already collected)"
else
  echo "[766/1456]  2025-26 regular  0022500471"
  collect-possessions --game 0022500471 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500472.parquet" ]; then
  echo "[767/1456]  skip 0022500472 (already collected)"
else
  echo "[767/1456]  2025-26 regular  0022500472"
  collect-possessions --game 0022500472 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500473.parquet" ]; then
  echo "[768/1456]  skip 0022500473 (already collected)"
else
  echo "[768/1456]  2025-26 regular  0022500473"
  collect-possessions --game 0022500473 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500474.parquet" ]; then
  echo "[769/1456]  skip 0022500474 (already collected)"
else
  echo "[769/1456]  2025-26 regular  0022500474"
  collect-possessions --game 0022500474 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500475.parquet" ]; then
  echo "[770/1456]  skip 0022500475 (already collected)"
else
  echo "[770/1456]  2025-26 regular  0022500475"
  collect-possessions --game 0022500475 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500516.parquet" ]; then
  echo "[771/1456]  skip 0022500516 (already collected)"
else
  echo "[771/1456]  2025-26 regular  0022500516"
  collect-possessions --game 0022500516 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500517.parquet" ]; then
  echo "[772/1456]  skip 0022500517 (already collected)"
else
  echo "[772/1456]  2025-26 regular  0022500517"
  collect-possessions --game 0022500517 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500518.parquet" ]; then
  echo "[773/1456]  skip 0022500518 (already collected)"
else
  echo "[773/1456]  2025-26 regular  0022500518"
  collect-possessions --game 0022500518 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500519.parquet" ]; then
  echo "[774/1456]  skip 0022500519 (already collected)"
else
  echo "[774/1456]  2025-26 regular  0022500519"
  collect-possessions --game 0022500519 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500520.parquet" ]; then
  echo "[775/1456]  skip 0022500520 (already collected)"
else
  echo "[775/1456]  2025-26 regular  0022500520"
  collect-possessions --game 0022500520 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500521.parquet" ]; then
  echo "[776/1456]  skip 0022500521 (already collected)"
else
  echo "[776/1456]  2025-26 regular  0022500521"
  collect-possessions --game 0022500521 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500522.parquet" ]; then
  echo "[777/1456]  skip 0022500522 (already collected)"
else
  echo "[777/1456]  2025-26 regular  0022500522"
  collect-possessions --game 0022500522 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500523.parquet" ]; then
  echo "[778/1456]  skip 0022500523 (already collected)"
else
  echo "[778/1456]  2025-26 regular  0022500523"
  collect-possessions --game 0022500523 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500524.parquet" ]; then
  echo "[779/1456]  skip 0022500524 (already collected)"
else
  echo "[779/1456]  2025-26 regular  0022500524"
  collect-possessions --game 0022500524 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500525.parquet" ]; then
  echo "[780/1456]  skip 0022500525 (already collected)"
else
  echo "[780/1456]  2025-26 regular  0022500525"
  collect-possessions --game 0022500525 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500526.parquet" ]; then
  echo "[781/1456]  skip 0022500526 (already collected)"
else
  echo "[781/1456]  2025-26 regular  0022500526"
  collect-possessions --game 0022500526 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500527.parquet" ]; then
  echo "[782/1456]  skip 0022500527 (already collected)"
else
  echo "[782/1456]  2025-26 regular  0022500527"
  collect-possessions --game 0022500527 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500528.parquet" ]; then
  echo "[783/1456]  skip 0022500528 (already collected)"
else
  echo "[783/1456]  2025-26 regular  0022500528"
  collect-possessions --game 0022500528 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500530.parquet" ]; then
  echo "[784/1456]  skip 0022500530 (already collected)"
else
  echo "[784/1456]  2025-26 regular  0022500530"
  collect-possessions --game 0022500530 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500531.parquet" ]; then
  echo "[785/1456]  skip 0022500531 (already collected)"
else
  echo "[785/1456]  2025-26 regular  0022500531"
  collect-possessions --game 0022500531 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500532.parquet" ]; then
  echo "[786/1456]  skip 0022500532 (already collected)"
else
  echo "[786/1456]  2025-26 regular  0022500532"
  collect-possessions --game 0022500532 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500533.parquet" ]; then
  echo "[787/1456]  skip 0022500533 (already collected)"
else
  echo "[787/1456]  2025-26 regular  0022500533"
  collect-possessions --game 0022500533 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500534.parquet" ]; then
  echo "[788/1456]  skip 0022500534 (already collected)"
else
  echo "[788/1456]  2025-26 regular  0022500534"
  collect-possessions --game 0022500534 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500535.parquet" ]; then
  echo "[789/1456]  skip 0022500535 (already collected)"
else
  echo "[789/1456]  2025-26 regular  0022500535"
  collect-possessions --game 0022500535 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500536.parquet" ]; then
  echo "[790/1456]  skip 0022500536 (already collected)"
else
  echo "[790/1456]  2025-26 regular  0022500536"
  collect-possessions --game 0022500536 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500537.parquet" ]; then
  echo "[791/1456]  skip 0022500537 (already collected)"
else
  echo "[791/1456]  2025-26 regular  0022500537"
  collect-possessions --game 0022500537 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500538.parquet" ]; then
  echo "[792/1456]  skip 0022500538 (already collected)"
else
  echo "[792/1456]  2025-26 regular  0022500538"
  collect-possessions --game 0022500538 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500539.parquet" ]; then
  echo "[793/1456]  skip 0022500539 (already collected)"
else
  echo "[793/1456]  2025-26 regular  0022500539"
  collect-possessions --game 0022500539 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500540.parquet" ]; then
  echo "[794/1456]  skip 0022500540 (already collected)"
else
  echo "[794/1456]  2025-26 regular  0022500540"
  collect-possessions --game 0022500540 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500541.parquet" ]; then
  echo "[795/1456]  skip 0022500541 (already collected)"
else
  echo "[795/1456]  2025-26 regular  0022500541"
  collect-possessions --game 0022500541 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500542.parquet" ]; then
  echo "[796/1456]  skip 0022500542 (already collected)"
else
  echo "[796/1456]  2025-26 regular  0022500542"
  collect-possessions --game 0022500542 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500543.parquet" ]; then
  echo "[797/1456]  skip 0022500543 (already collected)"
else
  echo "[797/1456]  2025-26 regular  0022500543"
  collect-possessions --game 0022500543 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500544.parquet" ]; then
  echo "[798/1456]  skip 0022500544 (already collected)"
else
  echo "[798/1456]  2025-26 regular  0022500544"
  collect-possessions --game 0022500544 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500545.parquet" ]; then
  echo "[799/1456]  skip 0022500545 (already collected)"
else
  echo "[799/1456]  2025-26 regular  0022500545"
  collect-possessions --game 0022500545 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500546.parquet" ]; then
  echo "[800/1456]  skip 0022500546 (already collected)"
else
  echo "[800/1456]  2025-26 regular  0022500546"
  collect-possessions --game 0022500546 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500547.parquet" ]; then
  echo "[801/1456]  skip 0022500547 (already collected)"
else
  echo "[801/1456]  2025-26 regular  0022500547"
  collect-possessions --game 0022500547 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500549.parquet" ]; then
  echo "[802/1456]  skip 0022500549 (already collected)"
else
  echo "[802/1456]  2025-26 regular  0022500549"
  collect-possessions --game 0022500549 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500551.parquet" ]; then
  echo "[803/1456]  skip 0022500551 (already collected)"
else
  echo "[803/1456]  2025-26 regular  0022500551"
  collect-possessions --game 0022500551 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500747.parquet" ]; then
  echo "[804/1456]  skip 0022500747 (already collected)"
else
  echo "[804/1456]  2025-26 regular  0022500747"
  collect-possessions --game 0022500747 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500748.parquet" ]; then
  echo "[805/1456]  skip 0022500748 (already collected)"
else
  echo "[805/1456]  2025-26 regular  0022500748"
  collect-possessions --game 0022500748 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500749.parquet" ]; then
  echo "[806/1456]  skip 0022500749 (already collected)"
else
  echo "[806/1456]  2025-26 regular  0022500749"
  collect-possessions --game 0022500749 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500750.parquet" ]; then
  echo "[807/1456]  skip 0022500750 (already collected)"
else
  echo "[807/1456]  2025-26 regular  0022500750"
  collect-possessions --game 0022500750 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500751.parquet" ]; then
  echo "[808/1456]  skip 0022500751 (already collected)"
else
  echo "[808/1456]  2025-26 regular  0022500751"
  collect-possessions --game 0022500751 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500752.parquet" ]; then
  echo "[809/1456]  skip 0022500752 (already collected)"
else
  echo "[809/1456]  2025-26 regular  0022500752"
  collect-possessions --game 0022500752 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500753.parquet" ]; then
  echo "[810/1456]  skip 0022500753 (already collected)"
else
  echo "[810/1456]  2025-26 regular  0022500753"
  collect-possessions --game 0022500753 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500754.parquet" ]; then
  echo "[811/1456]  skip 0022500754 (already collected)"
else
  echo "[811/1456]  2025-26 regular  0022500754"
  collect-possessions --game 0022500754 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500755.parquet" ]; then
  echo "[812/1456]  skip 0022500755 (already collected)"
else
  echo "[812/1456]  2025-26 regular  0022500755"
  collect-possessions --game 0022500755 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500756.parquet" ]; then
  echo "[813/1456]  skip 0022500756 (already collected)"
else
  echo "[813/1456]  2025-26 regular  0022500756"
  collect-possessions --game 0022500756 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500757.parquet" ]; then
  echo "[814/1456]  skip 0022500757 (already collected)"
else
  echo "[814/1456]  2025-26 regular  0022500757"
  collect-possessions --game 0022500757 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500758.parquet" ]; then
  echo "[815/1456]  skip 0022500758 (already collected)"
else
  echo "[815/1456]  2025-26 regular  0022500758"
  collect-possessions --game 0022500758 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500759.parquet" ]; then
  echo "[816/1456]  skip 0022500759 (already collected)"
else
  echo "[816/1456]  2025-26 regular  0022500759"
  collect-possessions --game 0022500759 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500760.parquet" ]; then
  echo "[817/1456]  skip 0022500760 (already collected)"
else
  echo "[817/1456]  2025-26 regular  0022500760"
  collect-possessions --game 0022500760 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500761.parquet" ]; then
  echo "[818/1456]  skip 0022500761 (already collected)"
else
  echo "[818/1456]  2025-26 regular  0022500761"
  collect-possessions --game 0022500761 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500762.parquet" ]; then
  echo "[819/1456]  skip 0022500762 (already collected)"
else
  echo "[819/1456]  2025-26 regular  0022500762"
  collect-possessions --game 0022500762 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500763.parquet" ]; then
  echo "[820/1456]  skip 0022500763 (already collected)"
else
  echo "[820/1456]  2025-26 regular  0022500763"
  collect-possessions --game 0022500763 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500764.parquet" ]; then
  echo "[821/1456]  skip 0022500764 (already collected)"
else
  echo "[821/1456]  2025-26 regular  0022500764"
  collect-possessions --game 0022500764 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500765.parquet" ]; then
  echo "[822/1456]  skip 0022500765 (already collected)"
else
  echo "[822/1456]  2025-26 regular  0022500765"
  collect-possessions --game 0022500765 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500766.parquet" ]; then
  echo "[823/1456]  skip 0022500766 (already collected)"
else
  echo "[823/1456]  2025-26 regular  0022500766"
  collect-possessions --game 0022500766 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500767.parquet" ]; then
  echo "[824/1456]  skip 0022500767 (already collected)"
else
  echo "[824/1456]  2025-26 regular  0022500767"
  collect-possessions --game 0022500767 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500768.parquet" ]; then
  echo "[825/1456]  skip 0022500768 (already collected)"
else
  echo "[825/1456]  2025-26 regular  0022500768"
  collect-possessions --game 0022500768 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500769.parquet" ]; then
  echo "[826/1456]  skip 0022500769 (already collected)"
else
  echo "[826/1456]  2025-26 regular  0022500769"
  collect-possessions --game 0022500769 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500770.parquet" ]; then
  echo "[827/1456]  skip 0022500770 (already collected)"
else
  echo "[827/1456]  2025-26 regular  0022500770"
  collect-possessions --game 0022500770 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500811.parquet" ]; then
  echo "[828/1456]  skip 0022500811 (already collected)"
else
  echo "[828/1456]  2025-26 regular  0022500811"
  collect-possessions --game 0022500811 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500812.parquet" ]; then
  echo "[829/1456]  skip 0022500812 (already collected)"
else
  echo "[829/1456]  2025-26 regular  0022500812"
  collect-possessions --game 0022500812 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500813.parquet" ]; then
  echo "[830/1456]  skip 0022500813 (already collected)"
else
  echo "[830/1456]  2025-26 regular  0022500813"
  collect-possessions --game 0022500813 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500814.parquet" ]; then
  echo "[831/1456]  skip 0022500814 (already collected)"
else
  echo "[831/1456]  2025-26 regular  0022500814"
  collect-possessions --game 0022500814 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500815.parquet" ]; then
  echo "[832/1456]  skip 0022500815 (already collected)"
else
  echo "[832/1456]  2025-26 regular  0022500815"
  collect-possessions --game 0022500815 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500816.parquet" ]; then
  echo "[833/1456]  skip 0022500816 (already collected)"
else
  echo "[833/1456]  2025-26 regular  0022500816"
  collect-possessions --game 0022500816 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500817.parquet" ]; then
  echo "[834/1456]  skip 0022500817 (already collected)"
else
  echo "[834/1456]  2025-26 regular  0022500817"
  collect-possessions --game 0022500817 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500818.parquet" ]; then
  echo "[835/1456]  skip 0022500818 (already collected)"
else
  echo "[835/1456]  2025-26 regular  0022500818"
  collect-possessions --game 0022500818 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500819.parquet" ]; then
  echo "[836/1456]  skip 0022500819 (already collected)"
else
  echo "[836/1456]  2025-26 regular  0022500819"
  collect-possessions --game 0022500819 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500820.parquet" ]; then
  echo "[837/1456]  skip 0022500820 (already collected)"
else
  echo "[837/1456]  2025-26 regular  0022500820"
  collect-possessions --game 0022500820 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500821.parquet" ]; then
  echo "[838/1456]  skip 0022500821 (already collected)"
else
  echo "[838/1456]  2025-26 regular  0022500821"
  collect-possessions --game 0022500821 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500822.parquet" ]; then
  echo "[839/1456]  skip 0022500822 (already collected)"
else
  echo "[839/1456]  2025-26 regular  0022500822"
  collect-possessions --game 0022500822 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500823.parquet" ]; then
  echo "[840/1456]  skip 0022500823 (already collected)"
else
  echo "[840/1456]  2025-26 regular  0022500823"
  collect-possessions --game 0022500823 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500824.parquet" ]; then
  echo "[841/1456]  skip 0022500824 (already collected)"
else
  echo "[841/1456]  2025-26 regular  0022500824"
  collect-possessions --game 0022500824 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500825.parquet" ]; then
  echo "[842/1456]  skip 0022500825 (already collected)"
else
  echo "[842/1456]  2025-26 regular  0022500825"
  collect-possessions --game 0022500825 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500826.parquet" ]; then
  echo "[843/1456]  skip 0022500826 (already collected)"
else
  echo "[843/1456]  2025-26 regular  0022500826"
  collect-possessions --game 0022500826 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500827.parquet" ]; then
  echo "[844/1456]  skip 0022500827 (already collected)"
else
  echo "[844/1456]  2025-26 regular  0022500827"
  collect-possessions --game 0022500827 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500828.parquet" ]; then
  echo "[845/1456]  skip 0022500828 (already collected)"
else
  echo "[845/1456]  2025-26 regular  0022500828"
  collect-possessions --game 0022500828 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500829.parquet" ]; then
  echo "[846/1456]  skip 0022500829 (already collected)"
else
  echo "[846/1456]  2025-26 regular  0022500829"
  collect-possessions --game 0022500829 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500830.parquet" ]; then
  echo "[847/1456]  skip 0022500830 (already collected)"
else
  echo "[847/1456]  2025-26 regular  0022500830"
  collect-possessions --game 0022500830 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500831.parquet" ]; then
  echo "[848/1456]  skip 0022500831 (already collected)"
else
  echo "[848/1456]  2025-26 regular  0022500831"
  collect-possessions --game 0022500831 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500832.parquet" ]; then
  echo "[849/1456]  skip 0022500832 (already collected)"
else
  echo "[849/1456]  2025-26 regular  0022500832"
  collect-possessions --game 0022500832 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500833.parquet" ]; then
  echo "[850/1456]  skip 0022500833 (already collected)"
else
  echo "[850/1456]  2025-26 regular  0022500833"
  collect-possessions --game 0022500833 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500834.parquet" ]; then
  echo "[851/1456]  skip 0022500834 (already collected)"
else
  echo "[851/1456]  2025-26 regular  0022500834"
  collect-possessions --game 0022500834 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500835.parquet" ]; then
  echo "[852/1456]  skip 0022500835 (already collected)"
else
  echo "[852/1456]  2025-26 regular  0022500835"
  collect-possessions --game 0022500835 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500836.parquet" ]; then
  echo "[853/1456]  skip 0022500836 (already collected)"
else
  echo "[853/1456]  2025-26 regular  0022500836"
  collect-possessions --game 0022500836 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500837.parquet" ]; then
  echo "[854/1456]  skip 0022500837 (already collected)"
else
  echo "[854/1456]  2025-26 regular  0022500837"
  collect-possessions --game 0022500837 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500838.parquet" ]; then
  echo "[855/1456]  skip 0022500838 (already collected)"
else
  echo "[855/1456]  2025-26 regular  0022500838"
  collect-possessions --game 0022500838 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500839.parquet" ]; then
  echo "[856/1456]  skip 0022500839 (already collected)"
else
  echo "[856/1456]  2025-26 regular  0022500839"
  collect-possessions --game 0022500839 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500840.parquet" ]; then
  echo "[857/1456]  skip 0022500840 (already collected)"
else
  echo "[857/1456]  2025-26 regular  0022500840"
  collect-possessions --game 0022500840 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500841.parquet" ]; then
  echo "[858/1456]  skip 0022500841 (already collected)"
else
  echo "[858/1456]  2025-26 regular  0022500841"
  collect-possessions --game 0022500841 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500842.parquet" ]; then
  echo "[859/1456]  skip 0022500842 (already collected)"
else
  echo "[859/1456]  2025-26 regular  0022500842"
  collect-possessions --game 0022500842 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500843.parquet" ]; then
  echo "[860/1456]  skip 0022500843 (already collected)"
else
  echo "[860/1456]  2025-26 regular  0022500843"
  collect-possessions --game 0022500843 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500844.parquet" ]; then
  echo "[861/1456]  skip 0022500844 (already collected)"
else
  echo "[861/1456]  2025-26 regular  0022500844"
  collect-possessions --game 0022500844 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500845.parquet" ]; then
  echo "[862/1456]  skip 0022500845 (already collected)"
else
  echo "[862/1456]  2025-26 regular  0022500845"
  collect-possessions --game 0022500845 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022500846.parquet" ]; then
  echo "[863/1456]  skip 0022500846 (already collected)"
else
  echo "[863/1456]  2025-26 regular  0022500846"
  collect-possessions --game 0022500846 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022501016.parquet" ]; then
  echo "[864/1456]  skip 0022501016 (already collected)"
else
  echo "[864/1456]  2025-26 regular  0022501016"
  collect-possessions --game 0022501016 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022501017.parquet" ]; then
  echo "[865/1456]  skip 0022501017 (already collected)"
else
  echo "[865/1456]  2025-26 regular  0022501017"
  collect-possessions --game 0022501017 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022501018.parquet" ]; then
  echo "[866/1456]  skip 0022501018 (already collected)"
else
  echo "[866/1456]  2025-26 regular  0022501018"
  collect-possessions --game 0022501018 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022501019.parquet" ]; then
  echo "[867/1456]  skip 0022501019 (already collected)"
else
  echo "[867/1456]  2025-26 regular  0022501019"
  collect-possessions --game 0022501019 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022501020.parquet" ]; then
  echo "[868/1456]  skip 0022501020 (already collected)"
else
  echo "[868/1456]  2025-26 regular  0022501020"
  collect-possessions --game 0022501020 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0022501021.parquet" ]; then
  echo "[869/1456]  skip 0022501021 (already collected)"
else
  echo "[869/1456]  2025-26 regular  0022501021"
  collect-possessions --game 0022501021 --output-dir "$OUTDIR"
  sleep 0.5
fi

# ── 2018-19 playoffs (82 games) ────────────────────────────
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800101.parquet" ]; then
  echo "[870/1456]  skip 0041800101 (already collected)"
else
  echo "[870/1456]  2018-19 playoffs  0041800101"
  collect-possessions --game 0041800101 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800102.parquet" ]; then
  echo "[871/1456]  skip 0041800102 (already collected)"
else
  echo "[871/1456]  2018-19 playoffs  0041800102"
  collect-possessions --game 0041800102 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800103.parquet" ]; then
  echo "[872/1456]  skip 0041800103 (already collected)"
else
  echo "[872/1456]  2018-19 playoffs  0041800103"
  collect-possessions --game 0041800103 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800104.parquet" ]; then
  echo "[873/1456]  skip 0041800104 (already collected)"
else
  echo "[873/1456]  2018-19 playoffs  0041800104"
  collect-possessions --game 0041800104 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800111.parquet" ]; then
  echo "[874/1456]  skip 0041800111 (already collected)"
else
  echo "[874/1456]  2018-19 playoffs  0041800111"
  collect-possessions --game 0041800111 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800112.parquet" ]; then
  echo "[875/1456]  skip 0041800112 (already collected)"
else
  echo "[875/1456]  2018-19 playoffs  0041800112"
  collect-possessions --game 0041800112 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800113.parquet" ]; then
  echo "[876/1456]  skip 0041800113 (already collected)"
else
  echo "[876/1456]  2018-19 playoffs  0041800113"
  collect-possessions --game 0041800113 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800114.parquet" ]; then
  echo "[877/1456]  skip 0041800114 (already collected)"
else
  echo "[877/1456]  2018-19 playoffs  0041800114"
  collect-possessions --game 0041800114 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800115.parquet" ]; then
  echo "[878/1456]  skip 0041800115 (already collected)"
else
  echo "[878/1456]  2018-19 playoffs  0041800115"
  collect-possessions --game 0041800115 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800121.parquet" ]; then
  echo "[879/1456]  skip 0041800121 (already collected)"
else
  echo "[879/1456]  2018-19 playoffs  0041800121"
  collect-possessions --game 0041800121 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800122.parquet" ]; then
  echo "[880/1456]  skip 0041800122 (already collected)"
else
  echo "[880/1456]  2018-19 playoffs  0041800122"
  collect-possessions --game 0041800122 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800123.parquet" ]; then
  echo "[881/1456]  skip 0041800123 (already collected)"
else
  echo "[881/1456]  2018-19 playoffs  0041800123"
  collect-possessions --game 0041800123 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800124.parquet" ]; then
  echo "[882/1456]  skip 0041800124 (already collected)"
else
  echo "[882/1456]  2018-19 playoffs  0041800124"
  collect-possessions --game 0041800124 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800125.parquet" ]; then
  echo "[883/1456]  skip 0041800125 (already collected)"
else
  echo "[883/1456]  2018-19 playoffs  0041800125"
  collect-possessions --game 0041800125 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800131.parquet" ]; then
  echo "[884/1456]  skip 0041800131 (already collected)"
else
  echo "[884/1456]  2018-19 playoffs  0041800131"
  collect-possessions --game 0041800131 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800132.parquet" ]; then
  echo "[885/1456]  skip 0041800132 (already collected)"
else
  echo "[885/1456]  2018-19 playoffs  0041800132"
  collect-possessions --game 0041800132 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800133.parquet" ]; then
  echo "[886/1456]  skip 0041800133 (already collected)"
else
  echo "[886/1456]  2018-19 playoffs  0041800133"
  collect-possessions --game 0041800133 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800134.parquet" ]; then
  echo "[887/1456]  skip 0041800134 (already collected)"
else
  echo "[887/1456]  2018-19 playoffs  0041800134"
  collect-possessions --game 0041800134 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800141.parquet" ]; then
  echo "[888/1456]  skip 0041800141 (already collected)"
else
  echo "[888/1456]  2018-19 playoffs  0041800141"
  collect-possessions --game 0041800141 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800142.parquet" ]; then
  echo "[889/1456]  skip 0041800142 (already collected)"
else
  echo "[889/1456]  2018-19 playoffs  0041800142"
  collect-possessions --game 0041800142 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800143.parquet" ]; then
  echo "[890/1456]  skip 0041800143 (already collected)"
else
  echo "[890/1456]  2018-19 playoffs  0041800143"
  collect-possessions --game 0041800143 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800144.parquet" ]; then
  echo "[891/1456]  skip 0041800144 (already collected)"
else
  echo "[891/1456]  2018-19 playoffs  0041800144"
  collect-possessions --game 0041800144 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800145.parquet" ]; then
  echo "[892/1456]  skip 0041800145 (already collected)"
else
  echo "[892/1456]  2018-19 playoffs  0041800145"
  collect-possessions --game 0041800145 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800146.parquet" ]; then
  echo "[893/1456]  skip 0041800146 (already collected)"
else
  echo "[893/1456]  2018-19 playoffs  0041800146"
  collect-possessions --game 0041800146 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800151.parquet" ]; then
  echo "[894/1456]  skip 0041800151 (already collected)"
else
  echo "[894/1456]  2018-19 playoffs  0041800151"
  collect-possessions --game 0041800151 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800152.parquet" ]; then
  echo "[895/1456]  skip 0041800152 (already collected)"
else
  echo "[895/1456]  2018-19 playoffs  0041800152"
  collect-possessions --game 0041800152 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800153.parquet" ]; then
  echo "[896/1456]  skip 0041800153 (already collected)"
else
  echo "[896/1456]  2018-19 playoffs  0041800153"
  collect-possessions --game 0041800153 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800154.parquet" ]; then
  echo "[897/1456]  skip 0041800154 (already collected)"
else
  echo "[897/1456]  2018-19 playoffs  0041800154"
  collect-possessions --game 0041800154 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800155.parquet" ]; then
  echo "[898/1456]  skip 0041800155 (already collected)"
else
  echo "[898/1456]  2018-19 playoffs  0041800155"
  collect-possessions --game 0041800155 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800156.parquet" ]; then
  echo "[899/1456]  skip 0041800156 (already collected)"
else
  echo "[899/1456]  2018-19 playoffs  0041800156"
  collect-possessions --game 0041800156 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800157.parquet" ]; then
  echo "[900/1456]  skip 0041800157 (already collected)"
else
  echo "[900/1456]  2018-19 playoffs  0041800157"
  collect-possessions --game 0041800157 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800161.parquet" ]; then
  echo "[901/1456]  skip 0041800161 (already collected)"
else
  echo "[901/1456]  2018-19 playoffs  0041800161"
  collect-possessions --game 0041800161 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800162.parquet" ]; then
  echo "[902/1456]  skip 0041800162 (already collected)"
else
  echo "[902/1456]  2018-19 playoffs  0041800162"
  collect-possessions --game 0041800162 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800163.parquet" ]; then
  echo "[903/1456]  skip 0041800163 (already collected)"
else
  echo "[903/1456]  2018-19 playoffs  0041800163"
  collect-possessions --game 0041800163 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800164.parquet" ]; then
  echo "[904/1456]  skip 0041800164 (already collected)"
else
  echo "[904/1456]  2018-19 playoffs  0041800164"
  collect-possessions --game 0041800164 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800165.parquet" ]; then
  echo "[905/1456]  skip 0041800165 (already collected)"
else
  echo "[905/1456]  2018-19 playoffs  0041800165"
  collect-possessions --game 0041800165 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800171.parquet" ]; then
  echo "[906/1456]  skip 0041800171 (already collected)"
else
  echo "[906/1456]  2018-19 playoffs  0041800171"
  collect-possessions --game 0041800171 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800172.parquet" ]; then
  echo "[907/1456]  skip 0041800172 (already collected)"
else
  echo "[907/1456]  2018-19 playoffs  0041800172"
  collect-possessions --game 0041800172 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800173.parquet" ]; then
  echo "[908/1456]  skip 0041800173 (already collected)"
else
  echo "[908/1456]  2018-19 playoffs  0041800173"
  collect-possessions --game 0041800173 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800174.parquet" ]; then
  echo "[909/1456]  skip 0041800174 (already collected)"
else
  echo "[909/1456]  2018-19 playoffs  0041800174"
  collect-possessions --game 0041800174 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800175.parquet" ]; then
  echo "[910/1456]  skip 0041800175 (already collected)"
else
  echo "[910/1456]  2018-19 playoffs  0041800175"
  collect-possessions --game 0041800175 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800201.parquet" ]; then
  echo "[911/1456]  skip 0041800201 (already collected)"
else
  echo "[911/1456]  2018-19 playoffs  0041800201"
  collect-possessions --game 0041800201 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800202.parquet" ]; then
  echo "[912/1456]  skip 0041800202 (already collected)"
else
  echo "[912/1456]  2018-19 playoffs  0041800202"
  collect-possessions --game 0041800202 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800203.parquet" ]; then
  echo "[913/1456]  skip 0041800203 (already collected)"
else
  echo "[913/1456]  2018-19 playoffs  0041800203"
  collect-possessions --game 0041800203 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800204.parquet" ]; then
  echo "[914/1456]  skip 0041800204 (already collected)"
else
  echo "[914/1456]  2018-19 playoffs  0041800204"
  collect-possessions --game 0041800204 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800205.parquet" ]; then
  echo "[915/1456]  skip 0041800205 (already collected)"
else
  echo "[915/1456]  2018-19 playoffs  0041800205"
  collect-possessions --game 0041800205 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800211.parquet" ]; then
  echo "[916/1456]  skip 0041800211 (already collected)"
else
  echo "[916/1456]  2018-19 playoffs  0041800211"
  collect-possessions --game 0041800211 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800212.parquet" ]; then
  echo "[917/1456]  skip 0041800212 (already collected)"
else
  echo "[917/1456]  2018-19 playoffs  0041800212"
  collect-possessions --game 0041800212 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800213.parquet" ]; then
  echo "[918/1456]  skip 0041800213 (already collected)"
else
  echo "[918/1456]  2018-19 playoffs  0041800213"
  collect-possessions --game 0041800213 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800214.parquet" ]; then
  echo "[919/1456]  skip 0041800214 (already collected)"
else
  echo "[919/1456]  2018-19 playoffs  0041800214"
  collect-possessions --game 0041800214 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800215.parquet" ]; then
  echo "[920/1456]  skip 0041800215 (already collected)"
else
  echo "[920/1456]  2018-19 playoffs  0041800215"
  collect-possessions --game 0041800215 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800216.parquet" ]; then
  echo "[921/1456]  skip 0041800216 (already collected)"
else
  echo "[921/1456]  2018-19 playoffs  0041800216"
  collect-possessions --game 0041800216 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800217.parquet" ]; then
  echo "[922/1456]  skip 0041800217 (already collected)"
else
  echo "[922/1456]  2018-19 playoffs  0041800217"
  collect-possessions --game 0041800217 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800221.parquet" ]; then
  echo "[923/1456]  skip 0041800221 (already collected)"
else
  echo "[923/1456]  2018-19 playoffs  0041800221"
  collect-possessions --game 0041800221 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800222.parquet" ]; then
  echo "[924/1456]  skip 0041800222 (already collected)"
else
  echo "[924/1456]  2018-19 playoffs  0041800222"
  collect-possessions --game 0041800222 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800223.parquet" ]; then
  echo "[925/1456]  skip 0041800223 (already collected)"
else
  echo "[925/1456]  2018-19 playoffs  0041800223"
  collect-possessions --game 0041800223 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800224.parquet" ]; then
  echo "[926/1456]  skip 0041800224 (already collected)"
else
  echo "[926/1456]  2018-19 playoffs  0041800224"
  collect-possessions --game 0041800224 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800225.parquet" ]; then
  echo "[927/1456]  skip 0041800225 (already collected)"
else
  echo "[927/1456]  2018-19 playoffs  0041800225"
  collect-possessions --game 0041800225 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800226.parquet" ]; then
  echo "[928/1456]  skip 0041800226 (already collected)"
else
  echo "[928/1456]  2018-19 playoffs  0041800226"
  collect-possessions --game 0041800226 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800231.parquet" ]; then
  echo "[929/1456]  skip 0041800231 (already collected)"
else
  echo "[929/1456]  2018-19 playoffs  0041800231"
  collect-possessions --game 0041800231 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800232.parquet" ]; then
  echo "[930/1456]  skip 0041800232 (already collected)"
else
  echo "[930/1456]  2018-19 playoffs  0041800232"
  collect-possessions --game 0041800232 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800233.parquet" ]; then
  echo "[931/1456]  skip 0041800233 (already collected)"
else
  echo "[931/1456]  2018-19 playoffs  0041800233"
  collect-possessions --game 0041800233 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800234.parquet" ]; then
  echo "[932/1456]  skip 0041800234 (already collected)"
else
  echo "[932/1456]  2018-19 playoffs  0041800234"
  collect-possessions --game 0041800234 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800235.parquet" ]; then
  echo "[933/1456]  skip 0041800235 (already collected)"
else
  echo "[933/1456]  2018-19 playoffs  0041800235"
  collect-possessions --game 0041800235 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800236.parquet" ]; then
  echo "[934/1456]  skip 0041800236 (already collected)"
else
  echo "[934/1456]  2018-19 playoffs  0041800236"
  collect-possessions --game 0041800236 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800237.parquet" ]; then
  echo "[935/1456]  skip 0041800237 (already collected)"
else
  echo "[935/1456]  2018-19 playoffs  0041800237"
  collect-possessions --game 0041800237 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800301.parquet" ]; then
  echo "[936/1456]  skip 0041800301 (already collected)"
else
  echo "[936/1456]  2018-19 playoffs  0041800301"
  collect-possessions --game 0041800301 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800302.parquet" ]; then
  echo "[937/1456]  skip 0041800302 (already collected)"
else
  echo "[937/1456]  2018-19 playoffs  0041800302"
  collect-possessions --game 0041800302 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800303.parquet" ]; then
  echo "[938/1456]  skip 0041800303 (already collected)"
else
  echo "[938/1456]  2018-19 playoffs  0041800303"
  collect-possessions --game 0041800303 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800304.parquet" ]; then
  echo "[939/1456]  skip 0041800304 (already collected)"
else
  echo "[939/1456]  2018-19 playoffs  0041800304"
  collect-possessions --game 0041800304 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800305.parquet" ]; then
  echo "[940/1456]  skip 0041800305 (already collected)"
else
  echo "[940/1456]  2018-19 playoffs  0041800305"
  collect-possessions --game 0041800305 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800306.parquet" ]; then
  echo "[941/1456]  skip 0041800306 (already collected)"
else
  echo "[941/1456]  2018-19 playoffs  0041800306"
  collect-possessions --game 0041800306 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800311.parquet" ]; then
  echo "[942/1456]  skip 0041800311 (already collected)"
else
  echo "[942/1456]  2018-19 playoffs  0041800311"
  collect-possessions --game 0041800311 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800312.parquet" ]; then
  echo "[943/1456]  skip 0041800312 (already collected)"
else
  echo "[943/1456]  2018-19 playoffs  0041800312"
  collect-possessions --game 0041800312 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800313.parquet" ]; then
  echo "[944/1456]  skip 0041800313 (already collected)"
else
  echo "[944/1456]  2018-19 playoffs  0041800313"
  collect-possessions --game 0041800313 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800314.parquet" ]; then
  echo "[945/1456]  skip 0041800314 (already collected)"
else
  echo "[945/1456]  2018-19 playoffs  0041800314"
  collect-possessions --game 0041800314 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800401.parquet" ]; then
  echo "[946/1456]  skip 0041800401 (already collected)"
else
  echo "[946/1456]  2018-19 playoffs  0041800401"
  collect-possessions --game 0041800401 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800402.parquet" ]; then
  echo "[947/1456]  skip 0041800402 (already collected)"
else
  echo "[947/1456]  2018-19 playoffs  0041800402"
  collect-possessions --game 0041800402 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800403.parquet" ]; then
  echo "[948/1456]  skip 0041800403 (already collected)"
else
  echo "[948/1456]  2018-19 playoffs  0041800403"
  collect-possessions --game 0041800403 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800404.parquet" ]; then
  echo "[949/1456]  skip 0041800404 (already collected)"
else
  echo "[949/1456]  2018-19 playoffs  0041800404"
  collect-possessions --game 0041800404 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800405.parquet" ]; then
  echo "[950/1456]  skip 0041800405 (already collected)"
else
  echo "[950/1456]  2018-19 playoffs  0041800405"
  collect-possessions --game 0041800405 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041800406.parquet" ]; then
  echo "[951/1456]  skip 0041800406 (already collected)"
else
  echo "[951/1456]  2018-19 playoffs  0041800406"
  collect-possessions --game 0041800406 --output-dir "$OUTDIR"
  sleep 0.5
fi

# ── 2019-20 playoffs (83 games) ────────────────────────────
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900101.parquet" ]; then
  echo "[952/1456]  skip 0041900101 (already collected)"
else
  echo "[952/1456]  2019-20 playoffs  0041900101"
  collect-possessions --game 0041900101 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900102.parquet" ]; then
  echo "[953/1456]  skip 0041900102 (already collected)"
else
  echo "[953/1456]  2019-20 playoffs  0041900102"
  collect-possessions --game 0041900102 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900103.parquet" ]; then
  echo "[954/1456]  skip 0041900103 (already collected)"
else
  echo "[954/1456]  2019-20 playoffs  0041900103"
  collect-possessions --game 0041900103 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900104.parquet" ]; then
  echo "[955/1456]  skip 0041900104 (already collected)"
else
  echo "[955/1456]  2019-20 playoffs  0041900104"
  collect-possessions --game 0041900104 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900105.parquet" ]; then
  echo "[956/1456]  skip 0041900105 (already collected)"
else
  echo "[956/1456]  2019-20 playoffs  0041900105"
  collect-possessions --game 0041900105 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900111.parquet" ]; then
  echo "[957/1456]  skip 0041900111 (already collected)"
else
  echo "[957/1456]  2019-20 playoffs  0041900111"
  collect-possessions --game 0041900111 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900112.parquet" ]; then
  echo "[958/1456]  skip 0041900112 (already collected)"
else
  echo "[958/1456]  2019-20 playoffs  0041900112"
  collect-possessions --game 0041900112 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900113.parquet" ]; then
  echo "[959/1456]  skip 0041900113 (already collected)"
else
  echo "[959/1456]  2019-20 playoffs  0041900113"
  collect-possessions --game 0041900113 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900114.parquet" ]; then
  echo "[960/1456]  skip 0041900114 (already collected)"
else
  echo "[960/1456]  2019-20 playoffs  0041900114"
  collect-possessions --game 0041900114 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900121.parquet" ]; then
  echo "[961/1456]  skip 0041900121 (already collected)"
else
  echo "[961/1456]  2019-20 playoffs  0041900121"
  collect-possessions --game 0041900121 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900122.parquet" ]; then
  echo "[962/1456]  skip 0041900122 (already collected)"
else
  echo "[962/1456]  2019-20 playoffs  0041900122"
  collect-possessions --game 0041900122 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900123.parquet" ]; then
  echo "[963/1456]  skip 0041900123 (already collected)"
else
  echo "[963/1456]  2019-20 playoffs  0041900123"
  collect-possessions --game 0041900123 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900124.parquet" ]; then
  echo "[964/1456]  skip 0041900124 (already collected)"
else
  echo "[964/1456]  2019-20 playoffs  0041900124"
  collect-possessions --game 0041900124 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900131.parquet" ]; then
  echo "[965/1456]  skip 0041900131 (already collected)"
else
  echo "[965/1456]  2019-20 playoffs  0041900131"
  collect-possessions --game 0041900131 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900132.parquet" ]; then
  echo "[966/1456]  skip 0041900132 (already collected)"
else
  echo "[966/1456]  2019-20 playoffs  0041900132"
  collect-possessions --game 0041900132 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900133.parquet" ]; then
  echo "[967/1456]  skip 0041900133 (already collected)"
else
  echo "[967/1456]  2019-20 playoffs  0041900133"
  collect-possessions --game 0041900133 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900134.parquet" ]; then
  echo "[968/1456]  skip 0041900134 (already collected)"
else
  echo "[968/1456]  2019-20 playoffs  0041900134"
  collect-possessions --game 0041900134 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900141.parquet" ]; then
  echo "[969/1456]  skip 0041900141 (already collected)"
else
  echo "[969/1456]  2019-20 playoffs  0041900141"
  collect-possessions --game 0041900141 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900142.parquet" ]; then
  echo "[970/1456]  skip 0041900142 (already collected)"
else
  echo "[970/1456]  2019-20 playoffs  0041900142"
  collect-possessions --game 0041900142 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900143.parquet" ]; then
  echo "[971/1456]  skip 0041900143 (already collected)"
else
  echo "[971/1456]  2019-20 playoffs  0041900143"
  collect-possessions --game 0041900143 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900144.parquet" ]; then
  echo "[972/1456]  skip 0041900144 (already collected)"
else
  echo "[972/1456]  2019-20 playoffs  0041900144"
  collect-possessions --game 0041900144 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900145.parquet" ]; then
  echo "[973/1456]  skip 0041900145 (already collected)"
else
  echo "[973/1456]  2019-20 playoffs  0041900145"
  collect-possessions --game 0041900145 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900151.parquet" ]; then
  echo "[974/1456]  skip 0041900151 (already collected)"
else
  echo "[974/1456]  2019-20 playoffs  0041900151"
  collect-possessions --game 0041900151 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900152.parquet" ]; then
  echo "[975/1456]  skip 0041900152 (already collected)"
else
  echo "[975/1456]  2019-20 playoffs  0041900152"
  collect-possessions --game 0041900152 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900153.parquet" ]; then
  echo "[976/1456]  skip 0041900153 (already collected)"
else
  echo "[976/1456]  2019-20 playoffs  0041900153"
  collect-possessions --game 0041900153 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900154.parquet" ]; then
  echo "[977/1456]  skip 0041900154 (already collected)"
else
  echo "[977/1456]  2019-20 playoffs  0041900154"
  collect-possessions --game 0041900154 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900155.parquet" ]; then
  echo "[978/1456]  skip 0041900155 (already collected)"
else
  echo "[978/1456]  2019-20 playoffs  0041900155"
  collect-possessions --game 0041900155 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900156.parquet" ]; then
  echo "[979/1456]  skip 0041900156 (already collected)"
else
  echo "[979/1456]  2019-20 playoffs  0041900156"
  collect-possessions --game 0041900156 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900161.parquet" ]; then
  echo "[980/1456]  skip 0041900161 (already collected)"
else
  echo "[980/1456]  2019-20 playoffs  0041900161"
  collect-possessions --game 0041900161 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900162.parquet" ]; then
  echo "[981/1456]  skip 0041900162 (already collected)"
else
  echo "[981/1456]  2019-20 playoffs  0041900162"
  collect-possessions --game 0041900162 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900163.parquet" ]; then
  echo "[982/1456]  skip 0041900163 (already collected)"
else
  echo "[982/1456]  2019-20 playoffs  0041900163"
  collect-possessions --game 0041900163 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900164.parquet" ]; then
  echo "[983/1456]  skip 0041900164 (already collected)"
else
  echo "[983/1456]  2019-20 playoffs  0041900164"
  collect-possessions --game 0041900164 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900165.parquet" ]; then
  echo "[984/1456]  skip 0041900165 (already collected)"
else
  echo "[984/1456]  2019-20 playoffs  0041900165"
  collect-possessions --game 0041900165 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900166.parquet" ]; then
  echo "[985/1456]  skip 0041900166 (already collected)"
else
  echo "[985/1456]  2019-20 playoffs  0041900166"
  collect-possessions --game 0041900166 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900167.parquet" ]; then
  echo "[986/1456]  skip 0041900167 (already collected)"
else
  echo "[986/1456]  2019-20 playoffs  0041900167"
  collect-possessions --game 0041900167 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900171.parquet" ]; then
  echo "[987/1456]  skip 0041900171 (already collected)"
else
  echo "[987/1456]  2019-20 playoffs  0041900171"
  collect-possessions --game 0041900171 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900172.parquet" ]; then
  echo "[988/1456]  skip 0041900172 (already collected)"
else
  echo "[988/1456]  2019-20 playoffs  0041900172"
  collect-possessions --game 0041900172 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900173.parquet" ]; then
  echo "[989/1456]  skip 0041900173 (already collected)"
else
  echo "[989/1456]  2019-20 playoffs  0041900173"
  collect-possessions --game 0041900173 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900174.parquet" ]; then
  echo "[990/1456]  skip 0041900174 (already collected)"
else
  echo "[990/1456]  2019-20 playoffs  0041900174"
  collect-possessions --game 0041900174 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900175.parquet" ]; then
  echo "[991/1456]  skip 0041900175 (already collected)"
else
  echo "[991/1456]  2019-20 playoffs  0041900175"
  collect-possessions --game 0041900175 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900176.parquet" ]; then
  echo "[992/1456]  skip 0041900176 (already collected)"
else
  echo "[992/1456]  2019-20 playoffs  0041900176"
  collect-possessions --game 0041900176 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900177.parquet" ]; then
  echo "[993/1456]  skip 0041900177 (already collected)"
else
  echo "[993/1456]  2019-20 playoffs  0041900177"
  collect-possessions --game 0041900177 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900201.parquet" ]; then
  echo "[994/1456]  skip 0041900201 (already collected)"
else
  echo "[994/1456]  2019-20 playoffs  0041900201"
  collect-possessions --game 0041900201 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900202.parquet" ]; then
  echo "[995/1456]  skip 0041900202 (already collected)"
else
  echo "[995/1456]  2019-20 playoffs  0041900202"
  collect-possessions --game 0041900202 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900203.parquet" ]; then
  echo "[996/1456]  skip 0041900203 (already collected)"
else
  echo "[996/1456]  2019-20 playoffs  0041900203"
  collect-possessions --game 0041900203 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900204.parquet" ]; then
  echo "[997/1456]  skip 0041900204 (already collected)"
else
  echo "[997/1456]  2019-20 playoffs  0041900204"
  collect-possessions --game 0041900204 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900205.parquet" ]; then
  echo "[998/1456]  skip 0041900205 (already collected)"
else
  echo "[998/1456]  2019-20 playoffs  0041900205"
  collect-possessions --game 0041900205 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900211.parquet" ]; then
  echo "[999/1456]  skip 0041900211 (already collected)"
else
  echo "[999/1456]  2019-20 playoffs  0041900211"
  collect-possessions --game 0041900211 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900212.parquet" ]; then
  echo "[1000/1456]  skip 0041900212 (already collected)"
else
  echo "[1000/1456]  2019-20 playoffs  0041900212"
  collect-possessions --game 0041900212 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900213.parquet" ]; then
  echo "[1001/1456]  skip 0041900213 (already collected)"
else
  echo "[1001/1456]  2019-20 playoffs  0041900213"
  collect-possessions --game 0041900213 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900214.parquet" ]; then
  echo "[1002/1456]  skip 0041900214 (already collected)"
else
  echo "[1002/1456]  2019-20 playoffs  0041900214"
  collect-possessions --game 0041900214 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900215.parquet" ]; then
  echo "[1003/1456]  skip 0041900215 (already collected)"
else
  echo "[1003/1456]  2019-20 playoffs  0041900215"
  collect-possessions --game 0041900215 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900216.parquet" ]; then
  echo "[1004/1456]  skip 0041900216 (already collected)"
else
  echo "[1004/1456]  2019-20 playoffs  0041900216"
  collect-possessions --game 0041900216 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900217.parquet" ]; then
  echo "[1005/1456]  skip 0041900217 (already collected)"
else
  echo "[1005/1456]  2019-20 playoffs  0041900217"
  collect-possessions --game 0041900217 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900221.parquet" ]; then
  echo "[1006/1456]  skip 0041900221 (already collected)"
else
  echo "[1006/1456]  2019-20 playoffs  0041900221"
  collect-possessions --game 0041900221 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900222.parquet" ]; then
  echo "[1007/1456]  skip 0041900222 (already collected)"
else
  echo "[1007/1456]  2019-20 playoffs  0041900222"
  collect-possessions --game 0041900222 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900223.parquet" ]; then
  echo "[1008/1456]  skip 0041900223 (already collected)"
else
  echo "[1008/1456]  2019-20 playoffs  0041900223"
  collect-possessions --game 0041900223 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900224.parquet" ]; then
  echo "[1009/1456]  skip 0041900224 (already collected)"
else
  echo "[1009/1456]  2019-20 playoffs  0041900224"
  collect-possessions --game 0041900224 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900225.parquet" ]; then
  echo "[1010/1456]  skip 0041900225 (already collected)"
else
  echo "[1010/1456]  2019-20 playoffs  0041900225"
  collect-possessions --game 0041900225 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900231.parquet" ]; then
  echo "[1011/1456]  skip 0041900231 (already collected)"
else
  echo "[1011/1456]  2019-20 playoffs  0041900231"
  collect-possessions --game 0041900231 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900232.parquet" ]; then
  echo "[1012/1456]  skip 0041900232 (already collected)"
else
  echo "[1012/1456]  2019-20 playoffs  0041900232"
  collect-possessions --game 0041900232 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900233.parquet" ]; then
  echo "[1013/1456]  skip 0041900233 (already collected)"
else
  echo "[1013/1456]  2019-20 playoffs  0041900233"
  collect-possessions --game 0041900233 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900234.parquet" ]; then
  echo "[1014/1456]  skip 0041900234 (already collected)"
else
  echo "[1014/1456]  2019-20 playoffs  0041900234"
  collect-possessions --game 0041900234 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900235.parquet" ]; then
  echo "[1015/1456]  skip 0041900235 (already collected)"
else
  echo "[1015/1456]  2019-20 playoffs  0041900235"
  collect-possessions --game 0041900235 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900236.parquet" ]; then
  echo "[1016/1456]  skip 0041900236 (already collected)"
else
  echo "[1016/1456]  2019-20 playoffs  0041900236"
  collect-possessions --game 0041900236 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900237.parquet" ]; then
  echo "[1017/1456]  skip 0041900237 (already collected)"
else
  echo "[1017/1456]  2019-20 playoffs  0041900237"
  collect-possessions --game 0041900237 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900301.parquet" ]; then
  echo "[1018/1456]  skip 0041900301 (already collected)"
else
  echo "[1018/1456]  2019-20 playoffs  0041900301"
  collect-possessions --game 0041900301 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900302.parquet" ]; then
  echo "[1019/1456]  skip 0041900302 (already collected)"
else
  echo "[1019/1456]  2019-20 playoffs  0041900302"
  collect-possessions --game 0041900302 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900303.parquet" ]; then
  echo "[1020/1456]  skip 0041900303 (already collected)"
else
  echo "[1020/1456]  2019-20 playoffs  0041900303"
  collect-possessions --game 0041900303 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900304.parquet" ]; then
  echo "[1021/1456]  skip 0041900304 (already collected)"
else
  echo "[1021/1456]  2019-20 playoffs  0041900304"
  collect-possessions --game 0041900304 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900305.parquet" ]; then
  echo "[1022/1456]  skip 0041900305 (already collected)"
else
  echo "[1022/1456]  2019-20 playoffs  0041900305"
  collect-possessions --game 0041900305 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900306.parquet" ]; then
  echo "[1023/1456]  skip 0041900306 (already collected)"
else
  echo "[1023/1456]  2019-20 playoffs  0041900306"
  collect-possessions --game 0041900306 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900311.parquet" ]; then
  echo "[1024/1456]  skip 0041900311 (already collected)"
else
  echo "[1024/1456]  2019-20 playoffs  0041900311"
  collect-possessions --game 0041900311 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900312.parquet" ]; then
  echo "[1025/1456]  skip 0041900312 (already collected)"
else
  echo "[1025/1456]  2019-20 playoffs  0041900312"
  collect-possessions --game 0041900312 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900313.parquet" ]; then
  echo "[1026/1456]  skip 0041900313 (already collected)"
else
  echo "[1026/1456]  2019-20 playoffs  0041900313"
  collect-possessions --game 0041900313 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900314.parquet" ]; then
  echo "[1027/1456]  skip 0041900314 (already collected)"
else
  echo "[1027/1456]  2019-20 playoffs  0041900314"
  collect-possessions --game 0041900314 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900315.parquet" ]; then
  echo "[1028/1456]  skip 0041900315 (already collected)"
else
  echo "[1028/1456]  2019-20 playoffs  0041900315"
  collect-possessions --game 0041900315 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900401.parquet" ]; then
  echo "[1029/1456]  skip 0041900401 (already collected)"
else
  echo "[1029/1456]  2019-20 playoffs  0041900401"
  collect-possessions --game 0041900401 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900402.parquet" ]; then
  echo "[1030/1456]  skip 0041900402 (already collected)"
else
  echo "[1030/1456]  2019-20 playoffs  0041900402"
  collect-possessions --game 0041900402 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900403.parquet" ]; then
  echo "[1031/1456]  skip 0041900403 (already collected)"
else
  echo "[1031/1456]  2019-20 playoffs  0041900403"
  collect-possessions --game 0041900403 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900404.parquet" ]; then
  echo "[1032/1456]  skip 0041900404 (already collected)"
else
  echo "[1032/1456]  2019-20 playoffs  0041900404"
  collect-possessions --game 0041900404 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900405.parquet" ]; then
  echo "[1033/1456]  skip 0041900405 (already collected)"
else
  echo "[1033/1456]  2019-20 playoffs  0041900405"
  collect-possessions --game 0041900405 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0041900406.parquet" ]; then
  echo "[1034/1456]  skip 0041900406 (already collected)"
else
  echo "[1034/1456]  2019-20 playoffs  0041900406"
  collect-possessions --game 0041900406 --output-dir "$OUTDIR"
  sleep 0.5
fi

# ── 2020-21 playoffs (85 games) ────────────────────────────
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000101.parquet" ]; then
  echo "[1035/1456]  skip 0042000101 (already collected)"
else
  echo "[1035/1456]  2020-21 playoffs  0042000101"
  collect-possessions --game 0042000101 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000102.parquet" ]; then
  echo "[1036/1456]  skip 0042000102 (already collected)"
else
  echo "[1036/1456]  2020-21 playoffs  0042000102"
  collect-possessions --game 0042000102 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000103.parquet" ]; then
  echo "[1037/1456]  skip 0042000103 (already collected)"
else
  echo "[1037/1456]  2020-21 playoffs  0042000103"
  collect-possessions --game 0042000103 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000104.parquet" ]; then
  echo "[1038/1456]  skip 0042000104 (already collected)"
else
  echo "[1038/1456]  2020-21 playoffs  0042000104"
  collect-possessions --game 0042000104 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000105.parquet" ]; then
  echo "[1039/1456]  skip 0042000105 (already collected)"
else
  echo "[1039/1456]  2020-21 playoffs  0042000105"
  collect-possessions --game 0042000105 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000111.parquet" ]; then
  echo "[1040/1456]  skip 0042000111 (already collected)"
else
  echo "[1040/1456]  2020-21 playoffs  0042000111"
  collect-possessions --game 0042000111 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000112.parquet" ]; then
  echo "[1041/1456]  skip 0042000112 (already collected)"
else
  echo "[1041/1456]  2020-21 playoffs  0042000112"
  collect-possessions --game 0042000112 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000113.parquet" ]; then
  echo "[1042/1456]  skip 0042000113 (already collected)"
else
  echo "[1042/1456]  2020-21 playoffs  0042000113"
  collect-possessions --game 0042000113 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000114.parquet" ]; then
  echo "[1043/1456]  skip 0042000114 (already collected)"
else
  echo "[1043/1456]  2020-21 playoffs  0042000114"
  collect-possessions --game 0042000114 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000115.parquet" ]; then
  echo "[1044/1456]  skip 0042000115 (already collected)"
else
  echo "[1044/1456]  2020-21 playoffs  0042000115"
  collect-possessions --game 0042000115 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000121.parquet" ]; then
  echo "[1045/1456]  skip 0042000121 (already collected)"
else
  echo "[1045/1456]  2020-21 playoffs  0042000121"
  collect-possessions --game 0042000121 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000122.parquet" ]; then
  echo "[1046/1456]  skip 0042000122 (already collected)"
else
  echo "[1046/1456]  2020-21 playoffs  0042000122"
  collect-possessions --game 0042000122 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000123.parquet" ]; then
  echo "[1047/1456]  skip 0042000123 (already collected)"
else
  echo "[1047/1456]  2020-21 playoffs  0042000123"
  collect-possessions --game 0042000123 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000124.parquet" ]; then
  echo "[1048/1456]  skip 0042000124 (already collected)"
else
  echo "[1048/1456]  2020-21 playoffs  0042000124"
  collect-possessions --game 0042000124 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000131.parquet" ]; then
  echo "[1049/1456]  skip 0042000131 (already collected)"
else
  echo "[1049/1456]  2020-21 playoffs  0042000131"
  collect-possessions --game 0042000131 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000132.parquet" ]; then
  echo "[1050/1456]  skip 0042000132 (already collected)"
else
  echo "[1050/1456]  2020-21 playoffs  0042000132"
  collect-possessions --game 0042000132 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000133.parquet" ]; then
  echo "[1051/1456]  skip 0042000133 (already collected)"
else
  echo "[1051/1456]  2020-21 playoffs  0042000133"
  collect-possessions --game 0042000133 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000134.parquet" ]; then
  echo "[1052/1456]  skip 0042000134 (already collected)"
else
  echo "[1052/1456]  2020-21 playoffs  0042000134"
  collect-possessions --game 0042000134 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000135.parquet" ]; then
  echo "[1053/1456]  skip 0042000135 (already collected)"
else
  echo "[1053/1456]  2020-21 playoffs  0042000135"
  collect-possessions --game 0042000135 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000141.parquet" ]; then
  echo "[1054/1456]  skip 0042000141 (already collected)"
else
  echo "[1054/1456]  2020-21 playoffs  0042000141"
  collect-possessions --game 0042000141 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000142.parquet" ]; then
  echo "[1055/1456]  skip 0042000142 (already collected)"
else
  echo "[1055/1456]  2020-21 playoffs  0042000142"
  collect-possessions --game 0042000142 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000143.parquet" ]; then
  echo "[1056/1456]  skip 0042000143 (already collected)"
else
  echo "[1056/1456]  2020-21 playoffs  0042000143"
  collect-possessions --game 0042000143 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000144.parquet" ]; then
  echo "[1057/1456]  skip 0042000144 (already collected)"
else
  echo "[1057/1456]  2020-21 playoffs  0042000144"
  collect-possessions --game 0042000144 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000145.parquet" ]; then
  echo "[1058/1456]  skip 0042000145 (already collected)"
else
  echo "[1058/1456]  2020-21 playoffs  0042000145"
  collect-possessions --game 0042000145 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000151.parquet" ]; then
  echo "[1059/1456]  skip 0042000151 (already collected)"
else
  echo "[1059/1456]  2020-21 playoffs  0042000151"
  collect-possessions --game 0042000151 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000152.parquet" ]; then
  echo "[1060/1456]  skip 0042000152 (already collected)"
else
  echo "[1060/1456]  2020-21 playoffs  0042000152"
  collect-possessions --game 0042000152 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000153.parquet" ]; then
  echo "[1061/1456]  skip 0042000153 (already collected)"
else
  echo "[1061/1456]  2020-21 playoffs  0042000153"
  collect-possessions --game 0042000153 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000154.parquet" ]; then
  echo "[1062/1456]  skip 0042000154 (already collected)"
else
  echo "[1062/1456]  2020-21 playoffs  0042000154"
  collect-possessions --game 0042000154 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000155.parquet" ]; then
  echo "[1063/1456]  skip 0042000155 (already collected)"
else
  echo "[1063/1456]  2020-21 playoffs  0042000155"
  collect-possessions --game 0042000155 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000156.parquet" ]; then
  echo "[1064/1456]  skip 0042000156 (already collected)"
else
  echo "[1064/1456]  2020-21 playoffs  0042000156"
  collect-possessions --game 0042000156 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000161.parquet" ]; then
  echo "[1065/1456]  skip 0042000161 (already collected)"
else
  echo "[1065/1456]  2020-21 playoffs  0042000161"
  collect-possessions --game 0042000161 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000162.parquet" ]; then
  echo "[1066/1456]  skip 0042000162 (already collected)"
else
  echo "[1066/1456]  2020-21 playoffs  0042000162"
  collect-possessions --game 0042000162 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000163.parquet" ]; then
  echo "[1067/1456]  skip 0042000163 (already collected)"
else
  echo "[1067/1456]  2020-21 playoffs  0042000163"
  collect-possessions --game 0042000163 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000164.parquet" ]; then
  echo "[1068/1456]  skip 0042000164 (already collected)"
else
  echo "[1068/1456]  2020-21 playoffs  0042000164"
  collect-possessions --game 0042000164 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000165.parquet" ]; then
  echo "[1069/1456]  skip 0042000165 (already collected)"
else
  echo "[1069/1456]  2020-21 playoffs  0042000165"
  collect-possessions --game 0042000165 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000166.parquet" ]; then
  echo "[1070/1456]  skip 0042000166 (already collected)"
else
  echo "[1070/1456]  2020-21 playoffs  0042000166"
  collect-possessions --game 0042000166 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000171.parquet" ]; then
  echo "[1071/1456]  skip 0042000171 (already collected)"
else
  echo "[1071/1456]  2020-21 playoffs  0042000171"
  collect-possessions --game 0042000171 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000172.parquet" ]; then
  echo "[1072/1456]  skip 0042000172 (already collected)"
else
  echo "[1072/1456]  2020-21 playoffs  0042000172"
  collect-possessions --game 0042000172 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000173.parquet" ]; then
  echo "[1073/1456]  skip 0042000173 (already collected)"
else
  echo "[1073/1456]  2020-21 playoffs  0042000173"
  collect-possessions --game 0042000173 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000174.parquet" ]; then
  echo "[1074/1456]  skip 0042000174 (already collected)"
else
  echo "[1074/1456]  2020-21 playoffs  0042000174"
  collect-possessions --game 0042000174 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000175.parquet" ]; then
  echo "[1075/1456]  skip 0042000175 (already collected)"
else
  echo "[1075/1456]  2020-21 playoffs  0042000175"
  collect-possessions --game 0042000175 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000176.parquet" ]; then
  echo "[1076/1456]  skip 0042000176 (already collected)"
else
  echo "[1076/1456]  2020-21 playoffs  0042000176"
  collect-possessions --game 0042000176 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000177.parquet" ]; then
  echo "[1077/1456]  skip 0042000177 (already collected)"
else
  echo "[1077/1456]  2020-21 playoffs  0042000177"
  collect-possessions --game 0042000177 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000201.parquet" ]; then
  echo "[1078/1456]  skip 0042000201 (already collected)"
else
  echo "[1078/1456]  2020-21 playoffs  0042000201"
  collect-possessions --game 0042000201 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000202.parquet" ]; then
  echo "[1079/1456]  skip 0042000202 (already collected)"
else
  echo "[1079/1456]  2020-21 playoffs  0042000202"
  collect-possessions --game 0042000202 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000203.parquet" ]; then
  echo "[1080/1456]  skip 0042000203 (already collected)"
else
  echo "[1080/1456]  2020-21 playoffs  0042000203"
  collect-possessions --game 0042000203 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000204.parquet" ]; then
  echo "[1081/1456]  skip 0042000204 (already collected)"
else
  echo "[1081/1456]  2020-21 playoffs  0042000204"
  collect-possessions --game 0042000204 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000205.parquet" ]; then
  echo "[1082/1456]  skip 0042000205 (already collected)"
else
  echo "[1082/1456]  2020-21 playoffs  0042000205"
  collect-possessions --game 0042000205 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000206.parquet" ]; then
  echo "[1083/1456]  skip 0042000206 (already collected)"
else
  echo "[1083/1456]  2020-21 playoffs  0042000206"
  collect-possessions --game 0042000206 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000207.parquet" ]; then
  echo "[1084/1456]  skip 0042000207 (already collected)"
else
  echo "[1084/1456]  2020-21 playoffs  0042000207"
  collect-possessions --game 0042000207 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000211.parquet" ]; then
  echo "[1085/1456]  skip 0042000211 (already collected)"
else
  echo "[1085/1456]  2020-21 playoffs  0042000211"
  collect-possessions --game 0042000211 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000212.parquet" ]; then
  echo "[1086/1456]  skip 0042000212 (already collected)"
else
  echo "[1086/1456]  2020-21 playoffs  0042000212"
  collect-possessions --game 0042000212 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000213.parquet" ]; then
  echo "[1087/1456]  skip 0042000213 (already collected)"
else
  echo "[1087/1456]  2020-21 playoffs  0042000213"
  collect-possessions --game 0042000213 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000214.parquet" ]; then
  echo "[1088/1456]  skip 0042000214 (already collected)"
else
  echo "[1088/1456]  2020-21 playoffs  0042000214"
  collect-possessions --game 0042000214 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000215.parquet" ]; then
  echo "[1089/1456]  skip 0042000215 (already collected)"
else
  echo "[1089/1456]  2020-21 playoffs  0042000215"
  collect-possessions --game 0042000215 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000216.parquet" ]; then
  echo "[1090/1456]  skip 0042000216 (already collected)"
else
  echo "[1090/1456]  2020-21 playoffs  0042000216"
  collect-possessions --game 0042000216 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000217.parquet" ]; then
  echo "[1091/1456]  skip 0042000217 (already collected)"
else
  echo "[1091/1456]  2020-21 playoffs  0042000217"
  collect-possessions --game 0042000217 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000221.parquet" ]; then
  echo "[1092/1456]  skip 0042000221 (already collected)"
else
  echo "[1092/1456]  2020-21 playoffs  0042000221"
  collect-possessions --game 0042000221 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000222.parquet" ]; then
  echo "[1093/1456]  skip 0042000222 (already collected)"
else
  echo "[1093/1456]  2020-21 playoffs  0042000222"
  collect-possessions --game 0042000222 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000223.parquet" ]; then
  echo "[1094/1456]  skip 0042000223 (already collected)"
else
  echo "[1094/1456]  2020-21 playoffs  0042000223"
  collect-possessions --game 0042000223 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000224.parquet" ]; then
  echo "[1095/1456]  skip 0042000224 (already collected)"
else
  echo "[1095/1456]  2020-21 playoffs  0042000224"
  collect-possessions --game 0042000224 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000225.parquet" ]; then
  echo "[1096/1456]  skip 0042000225 (already collected)"
else
  echo "[1096/1456]  2020-21 playoffs  0042000225"
  collect-possessions --game 0042000225 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000226.parquet" ]; then
  echo "[1097/1456]  skip 0042000226 (already collected)"
else
  echo "[1097/1456]  2020-21 playoffs  0042000226"
  collect-possessions --game 0042000226 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000231.parquet" ]; then
  echo "[1098/1456]  skip 0042000231 (already collected)"
else
  echo "[1098/1456]  2020-21 playoffs  0042000231"
  collect-possessions --game 0042000231 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000232.parquet" ]; then
  echo "[1099/1456]  skip 0042000232 (already collected)"
else
  echo "[1099/1456]  2020-21 playoffs  0042000232"
  collect-possessions --game 0042000232 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000233.parquet" ]; then
  echo "[1100/1456]  skip 0042000233 (already collected)"
else
  echo "[1100/1456]  2020-21 playoffs  0042000233"
  collect-possessions --game 0042000233 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000234.parquet" ]; then
  echo "[1101/1456]  skip 0042000234 (already collected)"
else
  echo "[1101/1456]  2020-21 playoffs  0042000234"
  collect-possessions --game 0042000234 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000301.parquet" ]; then
  echo "[1102/1456]  skip 0042000301 (already collected)"
else
  echo "[1102/1456]  2020-21 playoffs  0042000301"
  collect-possessions --game 0042000301 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000302.parquet" ]; then
  echo "[1103/1456]  skip 0042000302 (already collected)"
else
  echo "[1103/1456]  2020-21 playoffs  0042000302"
  collect-possessions --game 0042000302 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000303.parquet" ]; then
  echo "[1104/1456]  skip 0042000303 (already collected)"
else
  echo "[1104/1456]  2020-21 playoffs  0042000303"
  collect-possessions --game 0042000303 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000304.parquet" ]; then
  echo "[1105/1456]  skip 0042000304 (already collected)"
else
  echo "[1105/1456]  2020-21 playoffs  0042000304"
  collect-possessions --game 0042000304 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000305.parquet" ]; then
  echo "[1106/1456]  skip 0042000305 (already collected)"
else
  echo "[1106/1456]  2020-21 playoffs  0042000305"
  collect-possessions --game 0042000305 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000306.parquet" ]; then
  echo "[1107/1456]  skip 0042000306 (already collected)"
else
  echo "[1107/1456]  2020-21 playoffs  0042000306"
  collect-possessions --game 0042000306 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000311.parquet" ]; then
  echo "[1108/1456]  skip 0042000311 (already collected)"
else
  echo "[1108/1456]  2020-21 playoffs  0042000311"
  collect-possessions --game 0042000311 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000312.parquet" ]; then
  echo "[1109/1456]  skip 0042000312 (already collected)"
else
  echo "[1109/1456]  2020-21 playoffs  0042000312"
  collect-possessions --game 0042000312 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000313.parquet" ]; then
  echo "[1110/1456]  skip 0042000313 (already collected)"
else
  echo "[1110/1456]  2020-21 playoffs  0042000313"
  collect-possessions --game 0042000313 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000314.parquet" ]; then
  echo "[1111/1456]  skip 0042000314 (already collected)"
else
  echo "[1111/1456]  2020-21 playoffs  0042000314"
  collect-possessions --game 0042000314 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000315.parquet" ]; then
  echo "[1112/1456]  skip 0042000315 (already collected)"
else
  echo "[1112/1456]  2020-21 playoffs  0042000315"
  collect-possessions --game 0042000315 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000316.parquet" ]; then
  echo "[1113/1456]  skip 0042000316 (already collected)"
else
  echo "[1113/1456]  2020-21 playoffs  0042000316"
  collect-possessions --game 0042000316 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000401.parquet" ]; then
  echo "[1114/1456]  skip 0042000401 (already collected)"
else
  echo "[1114/1456]  2020-21 playoffs  0042000401"
  collect-possessions --game 0042000401 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000402.parquet" ]; then
  echo "[1115/1456]  skip 0042000402 (already collected)"
else
  echo "[1115/1456]  2020-21 playoffs  0042000402"
  collect-possessions --game 0042000402 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000403.parquet" ]; then
  echo "[1116/1456]  skip 0042000403 (already collected)"
else
  echo "[1116/1456]  2020-21 playoffs  0042000403"
  collect-possessions --game 0042000403 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000404.parquet" ]; then
  echo "[1117/1456]  skip 0042000404 (already collected)"
else
  echo "[1117/1456]  2020-21 playoffs  0042000404"
  collect-possessions --game 0042000404 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000405.parquet" ]; then
  echo "[1118/1456]  skip 0042000405 (already collected)"
else
  echo "[1118/1456]  2020-21 playoffs  0042000405"
  collect-possessions --game 0042000405 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042000406.parquet" ]; then
  echo "[1119/1456]  skip 0042000406 (already collected)"
else
  echo "[1119/1456]  2020-21 playoffs  0042000406"
  collect-possessions --game 0042000406 --output-dir "$OUTDIR"
  sleep 0.5
fi

# ── 2021-22 playoffs (87 games) ────────────────────────────
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100101.parquet" ]; then
  echo "[1120/1456]  skip 0042100101 (already collected)"
else
  echo "[1120/1456]  2021-22 playoffs  0042100101"
  collect-possessions --game 0042100101 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100102.parquet" ]; then
  echo "[1121/1456]  skip 0042100102 (already collected)"
else
  echo "[1121/1456]  2021-22 playoffs  0042100102"
  collect-possessions --game 0042100102 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100103.parquet" ]; then
  echo "[1122/1456]  skip 0042100103 (already collected)"
else
  echo "[1122/1456]  2021-22 playoffs  0042100103"
  collect-possessions --game 0042100103 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100104.parquet" ]; then
  echo "[1123/1456]  skip 0042100104 (already collected)"
else
  echo "[1123/1456]  2021-22 playoffs  0042100104"
  collect-possessions --game 0042100104 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100105.parquet" ]; then
  echo "[1124/1456]  skip 0042100105 (already collected)"
else
  echo "[1124/1456]  2021-22 playoffs  0042100105"
  collect-possessions --game 0042100105 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100111.parquet" ]; then
  echo "[1125/1456]  skip 0042100111 (already collected)"
else
  echo "[1125/1456]  2021-22 playoffs  0042100111"
  collect-possessions --game 0042100111 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100112.parquet" ]; then
  echo "[1126/1456]  skip 0042100112 (already collected)"
else
  echo "[1126/1456]  2021-22 playoffs  0042100112"
  collect-possessions --game 0042100112 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100113.parquet" ]; then
  echo "[1127/1456]  skip 0042100113 (already collected)"
else
  echo "[1127/1456]  2021-22 playoffs  0042100113"
  collect-possessions --game 0042100113 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100114.parquet" ]; then
  echo "[1128/1456]  skip 0042100114 (already collected)"
else
  echo "[1128/1456]  2021-22 playoffs  0042100114"
  collect-possessions --game 0042100114 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100121.parquet" ]; then
  echo "[1129/1456]  skip 0042100121 (already collected)"
else
  echo "[1129/1456]  2021-22 playoffs  0042100121"
  collect-possessions --game 0042100121 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100122.parquet" ]; then
  echo "[1130/1456]  skip 0042100122 (already collected)"
else
  echo "[1130/1456]  2021-22 playoffs  0042100122"
  collect-possessions --game 0042100122 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100123.parquet" ]; then
  echo "[1131/1456]  skip 0042100123 (already collected)"
else
  echo "[1131/1456]  2021-22 playoffs  0042100123"
  collect-possessions --game 0042100123 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100124.parquet" ]; then
  echo "[1132/1456]  skip 0042100124 (already collected)"
else
  echo "[1132/1456]  2021-22 playoffs  0042100124"
  collect-possessions --game 0042100124 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100125.parquet" ]; then
  echo "[1133/1456]  skip 0042100125 (already collected)"
else
  echo "[1133/1456]  2021-22 playoffs  0042100125"
  collect-possessions --game 0042100125 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100131.parquet" ]; then
  echo "[1134/1456]  skip 0042100131 (already collected)"
else
  echo "[1134/1456]  2021-22 playoffs  0042100131"
  collect-possessions --game 0042100131 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100132.parquet" ]; then
  echo "[1135/1456]  skip 0042100132 (already collected)"
else
  echo "[1135/1456]  2021-22 playoffs  0042100132"
  collect-possessions --game 0042100132 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100133.parquet" ]; then
  echo "[1136/1456]  skip 0042100133 (already collected)"
else
  echo "[1136/1456]  2021-22 playoffs  0042100133"
  collect-possessions --game 0042100133 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100134.parquet" ]; then
  echo "[1137/1456]  skip 0042100134 (already collected)"
else
  echo "[1137/1456]  2021-22 playoffs  0042100134"
  collect-possessions --game 0042100134 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100135.parquet" ]; then
  echo "[1138/1456]  skip 0042100135 (already collected)"
else
  echo "[1138/1456]  2021-22 playoffs  0042100135"
  collect-possessions --game 0042100135 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100136.parquet" ]; then
  echo "[1139/1456]  skip 0042100136 (already collected)"
else
  echo "[1139/1456]  2021-22 playoffs  0042100136"
  collect-possessions --game 0042100136 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100141.parquet" ]; then
  echo "[1140/1456]  skip 0042100141 (already collected)"
else
  echo "[1140/1456]  2021-22 playoffs  0042100141"
  collect-possessions --game 0042100141 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100142.parquet" ]; then
  echo "[1141/1456]  skip 0042100142 (already collected)"
else
  echo "[1141/1456]  2021-22 playoffs  0042100142"
  collect-possessions --game 0042100142 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100143.parquet" ]; then
  echo "[1142/1456]  skip 0042100143 (already collected)"
else
  echo "[1142/1456]  2021-22 playoffs  0042100143"
  collect-possessions --game 0042100143 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100144.parquet" ]; then
  echo "[1143/1456]  skip 0042100144 (already collected)"
else
  echo "[1143/1456]  2021-22 playoffs  0042100144"
  collect-possessions --game 0042100144 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100145.parquet" ]; then
  echo "[1144/1456]  skip 0042100145 (already collected)"
else
  echo "[1144/1456]  2021-22 playoffs  0042100145"
  collect-possessions --game 0042100145 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100146.parquet" ]; then
  echo "[1145/1456]  skip 0042100146 (already collected)"
else
  echo "[1145/1456]  2021-22 playoffs  0042100146"
  collect-possessions --game 0042100146 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100151.parquet" ]; then
  echo "[1146/1456]  skip 0042100151 (already collected)"
else
  echo "[1146/1456]  2021-22 playoffs  0042100151"
  collect-possessions --game 0042100151 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100152.parquet" ]; then
  echo "[1147/1456]  skip 0042100152 (already collected)"
else
  echo "[1147/1456]  2021-22 playoffs  0042100152"
  collect-possessions --game 0042100152 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100153.parquet" ]; then
  echo "[1148/1456]  skip 0042100153 (already collected)"
else
  echo "[1148/1456]  2021-22 playoffs  0042100153"
  collect-possessions --game 0042100153 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100154.parquet" ]; then
  echo "[1149/1456]  skip 0042100154 (already collected)"
else
  echo "[1149/1456]  2021-22 playoffs  0042100154"
  collect-possessions --game 0042100154 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100155.parquet" ]; then
  echo "[1150/1456]  skip 0042100155 (already collected)"
else
  echo "[1150/1456]  2021-22 playoffs  0042100155"
  collect-possessions --game 0042100155 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100156.parquet" ]; then
  echo "[1151/1456]  skip 0042100156 (already collected)"
else
  echo "[1151/1456]  2021-22 playoffs  0042100156"
  collect-possessions --game 0042100156 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100161.parquet" ]; then
  echo "[1152/1456]  skip 0042100161 (already collected)"
else
  echo "[1152/1456]  2021-22 playoffs  0042100161"
  collect-possessions --game 0042100161 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100162.parquet" ]; then
  echo "[1153/1456]  skip 0042100162 (already collected)"
else
  echo "[1153/1456]  2021-22 playoffs  0042100162"
  collect-possessions --game 0042100162 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100163.parquet" ]; then
  echo "[1154/1456]  skip 0042100163 (already collected)"
else
  echo "[1154/1456]  2021-22 playoffs  0042100163"
  collect-possessions --game 0042100163 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100164.parquet" ]; then
  echo "[1155/1456]  skip 0042100164 (already collected)"
else
  echo "[1155/1456]  2021-22 playoffs  0042100164"
  collect-possessions --game 0042100164 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100165.parquet" ]; then
  echo "[1156/1456]  skip 0042100165 (already collected)"
else
  echo "[1156/1456]  2021-22 playoffs  0042100165"
  collect-possessions --game 0042100165 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100171.parquet" ]; then
  echo "[1157/1456]  skip 0042100171 (already collected)"
else
  echo "[1157/1456]  2021-22 playoffs  0042100171"
  collect-possessions --game 0042100171 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100172.parquet" ]; then
  echo "[1158/1456]  skip 0042100172 (already collected)"
else
  echo "[1158/1456]  2021-22 playoffs  0042100172"
  collect-possessions --game 0042100172 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100173.parquet" ]; then
  echo "[1159/1456]  skip 0042100173 (already collected)"
else
  echo "[1159/1456]  2021-22 playoffs  0042100173"
  collect-possessions --game 0042100173 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100174.parquet" ]; then
  echo "[1160/1456]  skip 0042100174 (already collected)"
else
  echo "[1160/1456]  2021-22 playoffs  0042100174"
  collect-possessions --game 0042100174 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100175.parquet" ]; then
  echo "[1161/1456]  skip 0042100175 (already collected)"
else
  echo "[1161/1456]  2021-22 playoffs  0042100175"
  collect-possessions --game 0042100175 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100176.parquet" ]; then
  echo "[1162/1456]  skip 0042100176 (already collected)"
else
  echo "[1162/1456]  2021-22 playoffs  0042100176"
  collect-possessions --game 0042100176 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100201.parquet" ]; then
  echo "[1163/1456]  skip 0042100201 (already collected)"
else
  echo "[1163/1456]  2021-22 playoffs  0042100201"
  collect-possessions --game 0042100201 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100202.parquet" ]; then
  echo "[1164/1456]  skip 0042100202 (already collected)"
else
  echo "[1164/1456]  2021-22 playoffs  0042100202"
  collect-possessions --game 0042100202 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100203.parquet" ]; then
  echo "[1165/1456]  skip 0042100203 (already collected)"
else
  echo "[1165/1456]  2021-22 playoffs  0042100203"
  collect-possessions --game 0042100203 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100204.parquet" ]; then
  echo "[1166/1456]  skip 0042100204 (already collected)"
else
  echo "[1166/1456]  2021-22 playoffs  0042100204"
  collect-possessions --game 0042100204 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100205.parquet" ]; then
  echo "[1167/1456]  skip 0042100205 (already collected)"
else
  echo "[1167/1456]  2021-22 playoffs  0042100205"
  collect-possessions --game 0042100205 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100206.parquet" ]; then
  echo "[1168/1456]  skip 0042100206 (already collected)"
else
  echo "[1168/1456]  2021-22 playoffs  0042100206"
  collect-possessions --game 0042100206 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100211.parquet" ]; then
  echo "[1169/1456]  skip 0042100211 (already collected)"
else
  echo "[1169/1456]  2021-22 playoffs  0042100211"
  collect-possessions --game 0042100211 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100212.parquet" ]; then
  echo "[1170/1456]  skip 0042100212 (already collected)"
else
  echo "[1170/1456]  2021-22 playoffs  0042100212"
  collect-possessions --game 0042100212 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100213.parquet" ]; then
  echo "[1171/1456]  skip 0042100213 (already collected)"
else
  echo "[1171/1456]  2021-22 playoffs  0042100213"
  collect-possessions --game 0042100213 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100214.parquet" ]; then
  echo "[1172/1456]  skip 0042100214 (already collected)"
else
  echo "[1172/1456]  2021-22 playoffs  0042100214"
  collect-possessions --game 0042100214 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100215.parquet" ]; then
  echo "[1173/1456]  skip 0042100215 (already collected)"
else
  echo "[1173/1456]  2021-22 playoffs  0042100215"
  collect-possessions --game 0042100215 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100216.parquet" ]; then
  echo "[1174/1456]  skip 0042100216 (already collected)"
else
  echo "[1174/1456]  2021-22 playoffs  0042100216"
  collect-possessions --game 0042100216 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100217.parquet" ]; then
  echo "[1175/1456]  skip 0042100217 (already collected)"
else
  echo "[1175/1456]  2021-22 playoffs  0042100217"
  collect-possessions --game 0042100217 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100221.parquet" ]; then
  echo "[1176/1456]  skip 0042100221 (already collected)"
else
  echo "[1176/1456]  2021-22 playoffs  0042100221"
  collect-possessions --game 0042100221 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100222.parquet" ]; then
  echo "[1177/1456]  skip 0042100222 (already collected)"
else
  echo "[1177/1456]  2021-22 playoffs  0042100222"
  collect-possessions --game 0042100222 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100223.parquet" ]; then
  echo "[1178/1456]  skip 0042100223 (already collected)"
else
  echo "[1178/1456]  2021-22 playoffs  0042100223"
  collect-possessions --game 0042100223 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100224.parquet" ]; then
  echo "[1179/1456]  skip 0042100224 (already collected)"
else
  echo "[1179/1456]  2021-22 playoffs  0042100224"
  collect-possessions --game 0042100224 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100225.parquet" ]; then
  echo "[1180/1456]  skip 0042100225 (already collected)"
else
  echo "[1180/1456]  2021-22 playoffs  0042100225"
  collect-possessions --game 0042100225 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100226.parquet" ]; then
  echo "[1181/1456]  skip 0042100226 (already collected)"
else
  echo "[1181/1456]  2021-22 playoffs  0042100226"
  collect-possessions --game 0042100226 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100227.parquet" ]; then
  echo "[1182/1456]  skip 0042100227 (already collected)"
else
  echo "[1182/1456]  2021-22 playoffs  0042100227"
  collect-possessions --game 0042100227 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100231.parquet" ]; then
  echo "[1183/1456]  skip 0042100231 (already collected)"
else
  echo "[1183/1456]  2021-22 playoffs  0042100231"
  collect-possessions --game 0042100231 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100232.parquet" ]; then
  echo "[1184/1456]  skip 0042100232 (already collected)"
else
  echo "[1184/1456]  2021-22 playoffs  0042100232"
  collect-possessions --game 0042100232 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100233.parquet" ]; then
  echo "[1185/1456]  skip 0042100233 (already collected)"
else
  echo "[1185/1456]  2021-22 playoffs  0042100233"
  collect-possessions --game 0042100233 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100234.parquet" ]; then
  echo "[1186/1456]  skip 0042100234 (already collected)"
else
  echo "[1186/1456]  2021-22 playoffs  0042100234"
  collect-possessions --game 0042100234 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100235.parquet" ]; then
  echo "[1187/1456]  skip 0042100235 (already collected)"
else
  echo "[1187/1456]  2021-22 playoffs  0042100235"
  collect-possessions --game 0042100235 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100236.parquet" ]; then
  echo "[1188/1456]  skip 0042100236 (already collected)"
else
  echo "[1188/1456]  2021-22 playoffs  0042100236"
  collect-possessions --game 0042100236 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100301.parquet" ]; then
  echo "[1189/1456]  skip 0042100301 (already collected)"
else
  echo "[1189/1456]  2021-22 playoffs  0042100301"
  collect-possessions --game 0042100301 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100302.parquet" ]; then
  echo "[1190/1456]  skip 0042100302 (already collected)"
else
  echo "[1190/1456]  2021-22 playoffs  0042100302"
  collect-possessions --game 0042100302 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100303.parquet" ]; then
  echo "[1191/1456]  skip 0042100303 (already collected)"
else
  echo "[1191/1456]  2021-22 playoffs  0042100303"
  collect-possessions --game 0042100303 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100304.parquet" ]; then
  echo "[1192/1456]  skip 0042100304 (already collected)"
else
  echo "[1192/1456]  2021-22 playoffs  0042100304"
  collect-possessions --game 0042100304 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100305.parquet" ]; then
  echo "[1193/1456]  skip 0042100305 (already collected)"
else
  echo "[1193/1456]  2021-22 playoffs  0042100305"
  collect-possessions --game 0042100305 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100306.parquet" ]; then
  echo "[1194/1456]  skip 0042100306 (already collected)"
else
  echo "[1194/1456]  2021-22 playoffs  0042100306"
  collect-possessions --game 0042100306 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100307.parquet" ]; then
  echo "[1195/1456]  skip 0042100307 (already collected)"
else
  echo "[1195/1456]  2021-22 playoffs  0042100307"
  collect-possessions --game 0042100307 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100311.parquet" ]; then
  echo "[1196/1456]  skip 0042100311 (already collected)"
else
  echo "[1196/1456]  2021-22 playoffs  0042100311"
  collect-possessions --game 0042100311 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100312.parquet" ]; then
  echo "[1197/1456]  skip 0042100312 (already collected)"
else
  echo "[1197/1456]  2021-22 playoffs  0042100312"
  collect-possessions --game 0042100312 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100313.parquet" ]; then
  echo "[1198/1456]  skip 0042100313 (already collected)"
else
  echo "[1198/1456]  2021-22 playoffs  0042100313"
  collect-possessions --game 0042100313 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100314.parquet" ]; then
  echo "[1199/1456]  skip 0042100314 (already collected)"
else
  echo "[1199/1456]  2021-22 playoffs  0042100314"
  collect-possessions --game 0042100314 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100315.parquet" ]; then
  echo "[1200/1456]  skip 0042100315 (already collected)"
else
  echo "[1200/1456]  2021-22 playoffs  0042100315"
  collect-possessions --game 0042100315 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100401.parquet" ]; then
  echo "[1201/1456]  skip 0042100401 (already collected)"
else
  echo "[1201/1456]  2021-22 playoffs  0042100401"
  collect-possessions --game 0042100401 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100402.parquet" ]; then
  echo "[1202/1456]  skip 0042100402 (already collected)"
else
  echo "[1202/1456]  2021-22 playoffs  0042100402"
  collect-possessions --game 0042100402 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100403.parquet" ]; then
  echo "[1203/1456]  skip 0042100403 (already collected)"
else
  echo "[1203/1456]  2021-22 playoffs  0042100403"
  collect-possessions --game 0042100403 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100404.parquet" ]; then
  echo "[1204/1456]  skip 0042100404 (already collected)"
else
  echo "[1204/1456]  2021-22 playoffs  0042100404"
  collect-possessions --game 0042100404 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100405.parquet" ]; then
  echo "[1205/1456]  skip 0042100405 (already collected)"
else
  echo "[1205/1456]  2021-22 playoffs  0042100405"
  collect-possessions --game 0042100405 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042100406.parquet" ]; then
  echo "[1206/1456]  skip 0042100406 (already collected)"
else
  echo "[1206/1456]  2021-22 playoffs  0042100406"
  collect-possessions --game 0042100406 --output-dir "$OUTDIR"
  sleep 0.5
fi

# ── 2022-23 playoffs (84 games) ────────────────────────────
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200101.parquet" ]; then
  echo "[1207/1456]  skip 0042200101 (already collected)"
else
  echo "[1207/1456]  2022-23 playoffs  0042200101"
  collect-possessions --game 0042200101 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200102.parquet" ]; then
  echo "[1208/1456]  skip 0042200102 (already collected)"
else
  echo "[1208/1456]  2022-23 playoffs  0042200102"
  collect-possessions --game 0042200102 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200103.parquet" ]; then
  echo "[1209/1456]  skip 0042200103 (already collected)"
else
  echo "[1209/1456]  2022-23 playoffs  0042200103"
  collect-possessions --game 0042200103 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200104.parquet" ]; then
  echo "[1210/1456]  skip 0042200104 (already collected)"
else
  echo "[1210/1456]  2022-23 playoffs  0042200104"
  collect-possessions --game 0042200104 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200105.parquet" ]; then
  echo "[1211/1456]  skip 0042200105 (already collected)"
else
  echo "[1211/1456]  2022-23 playoffs  0042200105"
  collect-possessions --game 0042200105 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200111.parquet" ]; then
  echo "[1212/1456]  skip 0042200111 (already collected)"
else
  echo "[1212/1456]  2022-23 playoffs  0042200111"
  collect-possessions --game 0042200111 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200112.parquet" ]; then
  echo "[1213/1456]  skip 0042200112 (already collected)"
else
  echo "[1213/1456]  2022-23 playoffs  0042200112"
  collect-possessions --game 0042200112 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200113.parquet" ]; then
  echo "[1214/1456]  skip 0042200113 (already collected)"
else
  echo "[1214/1456]  2022-23 playoffs  0042200113"
  collect-possessions --game 0042200113 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200114.parquet" ]; then
  echo "[1215/1456]  skip 0042200114 (already collected)"
else
  echo "[1215/1456]  2022-23 playoffs  0042200114"
  collect-possessions --game 0042200114 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200115.parquet" ]; then
  echo "[1216/1456]  skip 0042200115 (already collected)"
else
  echo "[1216/1456]  2022-23 playoffs  0042200115"
  collect-possessions --game 0042200115 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200116.parquet" ]; then
  echo "[1217/1456]  skip 0042200116 (already collected)"
else
  echo "[1217/1456]  2022-23 playoffs  0042200116"
  collect-possessions --game 0042200116 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200121.parquet" ]; then
  echo "[1218/1456]  skip 0042200121 (already collected)"
else
  echo "[1218/1456]  2022-23 playoffs  0042200121"
  collect-possessions --game 0042200121 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200122.parquet" ]; then
  echo "[1219/1456]  skip 0042200122 (already collected)"
else
  echo "[1219/1456]  2022-23 playoffs  0042200122"
  collect-possessions --game 0042200122 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200123.parquet" ]; then
  echo "[1220/1456]  skip 0042200123 (already collected)"
else
  echo "[1220/1456]  2022-23 playoffs  0042200123"
  collect-possessions --game 0042200123 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200124.parquet" ]; then
  echo "[1221/1456]  skip 0042200124 (already collected)"
else
  echo "[1221/1456]  2022-23 playoffs  0042200124"
  collect-possessions --game 0042200124 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200131.parquet" ]; then
  echo "[1222/1456]  skip 0042200131 (already collected)"
else
  echo "[1222/1456]  2022-23 playoffs  0042200131"
  collect-possessions --game 0042200131 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200132.parquet" ]; then
  echo "[1223/1456]  skip 0042200132 (already collected)"
else
  echo "[1223/1456]  2022-23 playoffs  0042200132"
  collect-possessions --game 0042200132 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200133.parquet" ]; then
  echo "[1224/1456]  skip 0042200133 (already collected)"
else
  echo "[1224/1456]  2022-23 playoffs  0042200133"
  collect-possessions --game 0042200133 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200134.parquet" ]; then
  echo "[1225/1456]  skip 0042200134 (already collected)"
else
  echo "[1225/1456]  2022-23 playoffs  0042200134"
  collect-possessions --game 0042200134 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200135.parquet" ]; then
  echo "[1226/1456]  skip 0042200135 (already collected)"
else
  echo "[1226/1456]  2022-23 playoffs  0042200135"
  collect-possessions --game 0042200135 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200141.parquet" ]; then
  echo "[1227/1456]  skip 0042200141 (already collected)"
else
  echo "[1227/1456]  2022-23 playoffs  0042200141"
  collect-possessions --game 0042200141 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200142.parquet" ]; then
  echo "[1228/1456]  skip 0042200142 (already collected)"
else
  echo "[1228/1456]  2022-23 playoffs  0042200142"
  collect-possessions --game 0042200142 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200143.parquet" ]; then
  echo "[1229/1456]  skip 0042200143 (already collected)"
else
  echo "[1229/1456]  2022-23 playoffs  0042200143"
  collect-possessions --game 0042200143 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200144.parquet" ]; then
  echo "[1230/1456]  skip 0042200144 (already collected)"
else
  echo "[1230/1456]  2022-23 playoffs  0042200144"
  collect-possessions --game 0042200144 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200145.parquet" ]; then
  echo "[1231/1456]  skip 0042200145 (already collected)"
else
  echo "[1231/1456]  2022-23 playoffs  0042200145"
  collect-possessions --game 0042200145 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200151.parquet" ]; then
  echo "[1232/1456]  skip 0042200151 (already collected)"
else
  echo "[1232/1456]  2022-23 playoffs  0042200151"
  collect-possessions --game 0042200151 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200152.parquet" ]; then
  echo "[1233/1456]  skip 0042200152 (already collected)"
else
  echo "[1233/1456]  2022-23 playoffs  0042200152"
  collect-possessions --game 0042200152 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200153.parquet" ]; then
  echo "[1234/1456]  skip 0042200153 (already collected)"
else
  echo "[1234/1456]  2022-23 playoffs  0042200153"
  collect-possessions --game 0042200153 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200154.parquet" ]; then
  echo "[1235/1456]  skip 0042200154 (already collected)"
else
  echo "[1235/1456]  2022-23 playoffs  0042200154"
  collect-possessions --game 0042200154 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200155.parquet" ]; then
  echo "[1236/1456]  skip 0042200155 (already collected)"
else
  echo "[1236/1456]  2022-23 playoffs  0042200155"
  collect-possessions --game 0042200155 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200156.parquet" ]; then
  echo "[1237/1456]  skip 0042200156 (already collected)"
else
  echo "[1237/1456]  2022-23 playoffs  0042200156"
  collect-possessions --game 0042200156 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200161.parquet" ]; then
  echo "[1238/1456]  skip 0042200161 (already collected)"
else
  echo "[1238/1456]  2022-23 playoffs  0042200161"
  collect-possessions --game 0042200161 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200162.parquet" ]; then
  echo "[1239/1456]  skip 0042200162 (already collected)"
else
  echo "[1239/1456]  2022-23 playoffs  0042200162"
  collect-possessions --game 0042200162 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200163.parquet" ]; then
  echo "[1240/1456]  skip 0042200163 (already collected)"
else
  echo "[1240/1456]  2022-23 playoffs  0042200163"
  collect-possessions --game 0042200163 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200164.parquet" ]; then
  echo "[1241/1456]  skip 0042200164 (already collected)"
else
  echo "[1241/1456]  2022-23 playoffs  0042200164"
  collect-possessions --game 0042200164 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200165.parquet" ]; then
  echo "[1242/1456]  skip 0042200165 (already collected)"
else
  echo "[1242/1456]  2022-23 playoffs  0042200165"
  collect-possessions --game 0042200165 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200166.parquet" ]; then
  echo "[1243/1456]  skip 0042200166 (already collected)"
else
  echo "[1243/1456]  2022-23 playoffs  0042200166"
  collect-possessions --game 0042200166 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200167.parquet" ]; then
  echo "[1244/1456]  skip 0042200167 (already collected)"
else
  echo "[1244/1456]  2022-23 playoffs  0042200167"
  collect-possessions --game 0042200167 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200171.parquet" ]; then
  echo "[1245/1456]  skip 0042200171 (already collected)"
else
  echo "[1245/1456]  2022-23 playoffs  0042200171"
  collect-possessions --game 0042200171 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200172.parquet" ]; then
  echo "[1246/1456]  skip 0042200172 (already collected)"
else
  echo "[1246/1456]  2022-23 playoffs  0042200172"
  collect-possessions --game 0042200172 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200173.parquet" ]; then
  echo "[1247/1456]  skip 0042200173 (already collected)"
else
  echo "[1247/1456]  2022-23 playoffs  0042200173"
  collect-possessions --game 0042200173 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200174.parquet" ]; then
  echo "[1248/1456]  skip 0042200174 (already collected)"
else
  echo "[1248/1456]  2022-23 playoffs  0042200174"
  collect-possessions --game 0042200174 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200175.parquet" ]; then
  echo "[1249/1456]  skip 0042200175 (already collected)"
else
  echo "[1249/1456]  2022-23 playoffs  0042200175"
  collect-possessions --game 0042200175 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200201.parquet" ]; then
  echo "[1250/1456]  skip 0042200201 (already collected)"
else
  echo "[1250/1456]  2022-23 playoffs  0042200201"
  collect-possessions --game 0042200201 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200202.parquet" ]; then
  echo "[1251/1456]  skip 0042200202 (already collected)"
else
  echo "[1251/1456]  2022-23 playoffs  0042200202"
  collect-possessions --game 0042200202 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200203.parquet" ]; then
  echo "[1252/1456]  skip 0042200203 (already collected)"
else
  echo "[1252/1456]  2022-23 playoffs  0042200203"
  collect-possessions --game 0042200203 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200204.parquet" ]; then
  echo "[1253/1456]  skip 0042200204 (already collected)"
else
  echo "[1253/1456]  2022-23 playoffs  0042200204"
  collect-possessions --game 0042200204 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200205.parquet" ]; then
  echo "[1254/1456]  skip 0042200205 (already collected)"
else
  echo "[1254/1456]  2022-23 playoffs  0042200205"
  collect-possessions --game 0042200205 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200206.parquet" ]; then
  echo "[1255/1456]  skip 0042200206 (already collected)"
else
  echo "[1255/1456]  2022-23 playoffs  0042200206"
  collect-possessions --game 0042200206 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200211.parquet" ]; then
  echo "[1256/1456]  skip 0042200211 (already collected)"
else
  echo "[1256/1456]  2022-23 playoffs  0042200211"
  collect-possessions --game 0042200211 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200212.parquet" ]; then
  echo "[1257/1456]  skip 0042200212 (already collected)"
else
  echo "[1257/1456]  2022-23 playoffs  0042200212"
  collect-possessions --game 0042200212 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200213.parquet" ]; then
  echo "[1258/1456]  skip 0042200213 (already collected)"
else
  echo "[1258/1456]  2022-23 playoffs  0042200213"
  collect-possessions --game 0042200213 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200214.parquet" ]; then
  echo "[1259/1456]  skip 0042200214 (already collected)"
else
  echo "[1259/1456]  2022-23 playoffs  0042200214"
  collect-possessions --game 0042200214 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200215.parquet" ]; then
  echo "[1260/1456]  skip 0042200215 (already collected)"
else
  echo "[1260/1456]  2022-23 playoffs  0042200215"
  collect-possessions --game 0042200215 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200216.parquet" ]; then
  echo "[1261/1456]  skip 0042200216 (already collected)"
else
  echo "[1261/1456]  2022-23 playoffs  0042200216"
  collect-possessions --game 0042200216 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200217.parquet" ]; then
  echo "[1262/1456]  skip 0042200217 (already collected)"
else
  echo "[1262/1456]  2022-23 playoffs  0042200217"
  collect-possessions --game 0042200217 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200221.parquet" ]; then
  echo "[1263/1456]  skip 0042200221 (already collected)"
else
  echo "[1263/1456]  2022-23 playoffs  0042200221"
  collect-possessions --game 0042200221 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200222.parquet" ]; then
  echo "[1264/1456]  skip 0042200222 (already collected)"
else
  echo "[1264/1456]  2022-23 playoffs  0042200222"
  collect-possessions --game 0042200222 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200223.parquet" ]; then
  echo "[1265/1456]  skip 0042200223 (already collected)"
else
  echo "[1265/1456]  2022-23 playoffs  0042200223"
  collect-possessions --game 0042200223 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200224.parquet" ]; then
  echo "[1266/1456]  skip 0042200224 (already collected)"
else
  echo "[1266/1456]  2022-23 playoffs  0042200224"
  collect-possessions --game 0042200224 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200225.parquet" ]; then
  echo "[1267/1456]  skip 0042200225 (already collected)"
else
  echo "[1267/1456]  2022-23 playoffs  0042200225"
  collect-possessions --game 0042200225 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200226.parquet" ]; then
  echo "[1268/1456]  skip 0042200226 (already collected)"
else
  echo "[1268/1456]  2022-23 playoffs  0042200226"
  collect-possessions --game 0042200226 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200231.parquet" ]; then
  echo "[1269/1456]  skip 0042200231 (already collected)"
else
  echo "[1269/1456]  2022-23 playoffs  0042200231"
  collect-possessions --game 0042200231 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200232.parquet" ]; then
  echo "[1270/1456]  skip 0042200232 (already collected)"
else
  echo "[1270/1456]  2022-23 playoffs  0042200232"
  collect-possessions --game 0042200232 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200233.parquet" ]; then
  echo "[1271/1456]  skip 0042200233 (already collected)"
else
  echo "[1271/1456]  2022-23 playoffs  0042200233"
  collect-possessions --game 0042200233 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200234.parquet" ]; then
  echo "[1272/1456]  skip 0042200234 (already collected)"
else
  echo "[1272/1456]  2022-23 playoffs  0042200234"
  collect-possessions --game 0042200234 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200235.parquet" ]; then
  echo "[1273/1456]  skip 0042200235 (already collected)"
else
  echo "[1273/1456]  2022-23 playoffs  0042200235"
  collect-possessions --game 0042200235 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200236.parquet" ]; then
  echo "[1274/1456]  skip 0042200236 (already collected)"
else
  echo "[1274/1456]  2022-23 playoffs  0042200236"
  collect-possessions --game 0042200236 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200301.parquet" ]; then
  echo "[1275/1456]  skip 0042200301 (already collected)"
else
  echo "[1275/1456]  2022-23 playoffs  0042200301"
  collect-possessions --game 0042200301 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200302.parquet" ]; then
  echo "[1276/1456]  skip 0042200302 (already collected)"
else
  echo "[1276/1456]  2022-23 playoffs  0042200302"
  collect-possessions --game 0042200302 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200303.parquet" ]; then
  echo "[1277/1456]  skip 0042200303 (already collected)"
else
  echo "[1277/1456]  2022-23 playoffs  0042200303"
  collect-possessions --game 0042200303 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200304.parquet" ]; then
  echo "[1278/1456]  skip 0042200304 (already collected)"
else
  echo "[1278/1456]  2022-23 playoffs  0042200304"
  collect-possessions --game 0042200304 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200305.parquet" ]; then
  echo "[1279/1456]  skip 0042200305 (already collected)"
else
  echo "[1279/1456]  2022-23 playoffs  0042200305"
  collect-possessions --game 0042200305 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200306.parquet" ]; then
  echo "[1280/1456]  skip 0042200306 (already collected)"
else
  echo "[1280/1456]  2022-23 playoffs  0042200306"
  collect-possessions --game 0042200306 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200307.parquet" ]; then
  echo "[1281/1456]  skip 0042200307 (already collected)"
else
  echo "[1281/1456]  2022-23 playoffs  0042200307"
  collect-possessions --game 0042200307 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200311.parquet" ]; then
  echo "[1282/1456]  skip 0042200311 (already collected)"
else
  echo "[1282/1456]  2022-23 playoffs  0042200311"
  collect-possessions --game 0042200311 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200312.parquet" ]; then
  echo "[1283/1456]  skip 0042200312 (already collected)"
else
  echo "[1283/1456]  2022-23 playoffs  0042200312"
  collect-possessions --game 0042200312 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200313.parquet" ]; then
  echo "[1284/1456]  skip 0042200313 (already collected)"
else
  echo "[1284/1456]  2022-23 playoffs  0042200313"
  collect-possessions --game 0042200313 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200314.parquet" ]; then
  echo "[1285/1456]  skip 0042200314 (already collected)"
else
  echo "[1285/1456]  2022-23 playoffs  0042200314"
  collect-possessions --game 0042200314 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200401.parquet" ]; then
  echo "[1286/1456]  skip 0042200401 (already collected)"
else
  echo "[1286/1456]  2022-23 playoffs  0042200401"
  collect-possessions --game 0042200401 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200402.parquet" ]; then
  echo "[1287/1456]  skip 0042200402 (already collected)"
else
  echo "[1287/1456]  2022-23 playoffs  0042200402"
  collect-possessions --game 0042200402 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200403.parquet" ]; then
  echo "[1288/1456]  skip 0042200403 (already collected)"
else
  echo "[1288/1456]  2022-23 playoffs  0042200403"
  collect-possessions --game 0042200403 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200404.parquet" ]; then
  echo "[1289/1456]  skip 0042200404 (already collected)"
else
  echo "[1289/1456]  2022-23 playoffs  0042200404"
  collect-possessions --game 0042200404 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042200405.parquet" ]; then
  echo "[1290/1456]  skip 0042200405 (already collected)"
else
  echo "[1290/1456]  2022-23 playoffs  0042200405"
  collect-possessions --game 0042200405 --output-dir "$OUTDIR"
  sleep 0.5
fi

# ── 2023-24 playoffs (82 games) ────────────────────────────
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300101.parquet" ]; then
  echo "[1291/1456]  skip 0042300101 (already collected)"
else
  echo "[1291/1456]  2023-24 playoffs  0042300101"
  collect-possessions --game 0042300101 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300102.parquet" ]; then
  echo "[1292/1456]  skip 0042300102 (already collected)"
else
  echo "[1292/1456]  2023-24 playoffs  0042300102"
  collect-possessions --game 0042300102 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300103.parquet" ]; then
  echo "[1293/1456]  skip 0042300103 (already collected)"
else
  echo "[1293/1456]  2023-24 playoffs  0042300103"
  collect-possessions --game 0042300103 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300104.parquet" ]; then
  echo "[1294/1456]  skip 0042300104 (already collected)"
else
  echo "[1294/1456]  2023-24 playoffs  0042300104"
  collect-possessions --game 0042300104 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300105.parquet" ]; then
  echo "[1295/1456]  skip 0042300105 (already collected)"
else
  echo "[1295/1456]  2023-24 playoffs  0042300105"
  collect-possessions --game 0042300105 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300111.parquet" ]; then
  echo "[1296/1456]  skip 0042300111 (already collected)"
else
  echo "[1296/1456]  2023-24 playoffs  0042300111"
  collect-possessions --game 0042300111 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300112.parquet" ]; then
  echo "[1297/1456]  skip 0042300112 (already collected)"
else
  echo "[1297/1456]  2023-24 playoffs  0042300112"
  collect-possessions --game 0042300112 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300113.parquet" ]; then
  echo "[1298/1456]  skip 0042300113 (already collected)"
else
  echo "[1298/1456]  2023-24 playoffs  0042300113"
  collect-possessions --game 0042300113 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300114.parquet" ]; then
  echo "[1299/1456]  skip 0042300114 (already collected)"
else
  echo "[1299/1456]  2023-24 playoffs  0042300114"
  collect-possessions --game 0042300114 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300115.parquet" ]; then
  echo "[1300/1456]  skip 0042300115 (already collected)"
else
  echo "[1300/1456]  2023-24 playoffs  0042300115"
  collect-possessions --game 0042300115 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300116.parquet" ]; then
  echo "[1301/1456]  skip 0042300116 (already collected)"
else
  echo "[1301/1456]  2023-24 playoffs  0042300116"
  collect-possessions --game 0042300116 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300121.parquet" ]; then
  echo "[1302/1456]  skip 0042300121 (already collected)"
else
  echo "[1302/1456]  2023-24 playoffs  0042300121"
  collect-possessions --game 0042300121 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300122.parquet" ]; then
  echo "[1303/1456]  skip 0042300122 (already collected)"
else
  echo "[1303/1456]  2023-24 playoffs  0042300122"
  collect-possessions --game 0042300122 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300123.parquet" ]; then
  echo "[1304/1456]  skip 0042300123 (already collected)"
else
  echo "[1304/1456]  2023-24 playoffs  0042300123"
  collect-possessions --game 0042300123 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300124.parquet" ]; then
  echo "[1305/1456]  skip 0042300124 (already collected)"
else
  echo "[1305/1456]  2023-24 playoffs  0042300124"
  collect-possessions --game 0042300124 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300125.parquet" ]; then
  echo "[1306/1456]  skip 0042300125 (already collected)"
else
  echo "[1306/1456]  2023-24 playoffs  0042300125"
  collect-possessions --game 0042300125 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300126.parquet" ]; then
  echo "[1307/1456]  skip 0042300126 (already collected)"
else
  echo "[1307/1456]  2023-24 playoffs  0042300126"
  collect-possessions --game 0042300126 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300131.parquet" ]; then
  echo "[1308/1456]  skip 0042300131 (already collected)"
else
  echo "[1308/1456]  2023-24 playoffs  0042300131"
  collect-possessions --game 0042300131 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300132.parquet" ]; then
  echo "[1309/1456]  skip 0042300132 (already collected)"
else
  echo "[1309/1456]  2023-24 playoffs  0042300132"
  collect-possessions --game 0042300132 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300133.parquet" ]; then
  echo "[1310/1456]  skip 0042300133 (already collected)"
else
  echo "[1310/1456]  2023-24 playoffs  0042300133"
  collect-possessions --game 0042300133 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300134.parquet" ]; then
  echo "[1311/1456]  skip 0042300134 (already collected)"
else
  echo "[1311/1456]  2023-24 playoffs  0042300134"
  collect-possessions --game 0042300134 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300135.parquet" ]; then
  echo "[1312/1456]  skip 0042300135 (already collected)"
else
  echo "[1312/1456]  2023-24 playoffs  0042300135"
  collect-possessions --game 0042300135 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300136.parquet" ]; then
  echo "[1313/1456]  skip 0042300136 (already collected)"
else
  echo "[1313/1456]  2023-24 playoffs  0042300136"
  collect-possessions --game 0042300136 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300137.parquet" ]; then
  echo "[1314/1456]  skip 0042300137 (already collected)"
else
  echo "[1314/1456]  2023-24 playoffs  0042300137"
  collect-possessions --game 0042300137 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300141.parquet" ]; then
  echo "[1315/1456]  skip 0042300141 (already collected)"
else
  echo "[1315/1456]  2023-24 playoffs  0042300141"
  collect-possessions --game 0042300141 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300142.parquet" ]; then
  echo "[1316/1456]  skip 0042300142 (already collected)"
else
  echo "[1316/1456]  2023-24 playoffs  0042300142"
  collect-possessions --game 0042300142 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300143.parquet" ]; then
  echo "[1317/1456]  skip 0042300143 (already collected)"
else
  echo "[1317/1456]  2023-24 playoffs  0042300143"
  collect-possessions --game 0042300143 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300144.parquet" ]; then
  echo "[1318/1456]  skip 0042300144 (already collected)"
else
  echo "[1318/1456]  2023-24 playoffs  0042300144"
  collect-possessions --game 0042300144 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300151.parquet" ]; then
  echo "[1319/1456]  skip 0042300151 (already collected)"
else
  echo "[1319/1456]  2023-24 playoffs  0042300151"
  collect-possessions --game 0042300151 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300152.parquet" ]; then
  echo "[1320/1456]  skip 0042300152 (already collected)"
else
  echo "[1320/1456]  2023-24 playoffs  0042300152"
  collect-possessions --game 0042300152 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300153.parquet" ]; then
  echo "[1321/1456]  skip 0042300153 (already collected)"
else
  echo "[1321/1456]  2023-24 playoffs  0042300153"
  collect-possessions --game 0042300153 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300154.parquet" ]; then
  echo "[1322/1456]  skip 0042300154 (already collected)"
else
  echo "[1322/1456]  2023-24 playoffs  0042300154"
  collect-possessions --game 0042300154 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300155.parquet" ]; then
  echo "[1323/1456]  skip 0042300155 (already collected)"
else
  echo "[1323/1456]  2023-24 playoffs  0042300155"
  collect-possessions --game 0042300155 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300161.parquet" ]; then
  echo "[1324/1456]  skip 0042300161 (already collected)"
else
  echo "[1324/1456]  2023-24 playoffs  0042300161"
  collect-possessions --game 0042300161 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300162.parquet" ]; then
  echo "[1325/1456]  skip 0042300162 (already collected)"
else
  echo "[1325/1456]  2023-24 playoffs  0042300162"
  collect-possessions --game 0042300162 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300163.parquet" ]; then
  echo "[1326/1456]  skip 0042300163 (already collected)"
else
  echo "[1326/1456]  2023-24 playoffs  0042300163"
  collect-possessions --game 0042300163 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300164.parquet" ]; then
  echo "[1327/1456]  skip 0042300164 (already collected)"
else
  echo "[1327/1456]  2023-24 playoffs  0042300164"
  collect-possessions --game 0042300164 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300171.parquet" ]; then
  echo "[1328/1456]  skip 0042300171 (already collected)"
else
  echo "[1328/1456]  2023-24 playoffs  0042300171"
  collect-possessions --game 0042300171 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300172.parquet" ]; then
  echo "[1329/1456]  skip 0042300172 (already collected)"
else
  echo "[1329/1456]  2023-24 playoffs  0042300172"
  collect-possessions --game 0042300172 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300173.parquet" ]; then
  echo "[1330/1456]  skip 0042300173 (already collected)"
else
  echo "[1330/1456]  2023-24 playoffs  0042300173"
  collect-possessions --game 0042300173 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300174.parquet" ]; then
  echo "[1331/1456]  skip 0042300174 (already collected)"
else
  echo "[1331/1456]  2023-24 playoffs  0042300174"
  collect-possessions --game 0042300174 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300175.parquet" ]; then
  echo "[1332/1456]  skip 0042300175 (already collected)"
else
  echo "[1332/1456]  2023-24 playoffs  0042300175"
  collect-possessions --game 0042300175 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300176.parquet" ]; then
  echo "[1333/1456]  skip 0042300176 (already collected)"
else
  echo "[1333/1456]  2023-24 playoffs  0042300176"
  collect-possessions --game 0042300176 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300201.parquet" ]; then
  echo "[1334/1456]  skip 0042300201 (already collected)"
else
  echo "[1334/1456]  2023-24 playoffs  0042300201"
  collect-possessions --game 0042300201 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300202.parquet" ]; then
  echo "[1335/1456]  skip 0042300202 (already collected)"
else
  echo "[1335/1456]  2023-24 playoffs  0042300202"
  collect-possessions --game 0042300202 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300203.parquet" ]; then
  echo "[1336/1456]  skip 0042300203 (already collected)"
else
  echo "[1336/1456]  2023-24 playoffs  0042300203"
  collect-possessions --game 0042300203 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300204.parquet" ]; then
  echo "[1337/1456]  skip 0042300204 (already collected)"
else
  echo "[1337/1456]  2023-24 playoffs  0042300204"
  collect-possessions --game 0042300204 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300205.parquet" ]; then
  echo "[1338/1456]  skip 0042300205 (already collected)"
else
  echo "[1338/1456]  2023-24 playoffs  0042300205"
  collect-possessions --game 0042300205 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300211.parquet" ]; then
  echo "[1339/1456]  skip 0042300211 (already collected)"
else
  echo "[1339/1456]  2023-24 playoffs  0042300211"
  collect-possessions --game 0042300211 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300212.parquet" ]; then
  echo "[1340/1456]  skip 0042300212 (already collected)"
else
  echo "[1340/1456]  2023-24 playoffs  0042300212"
  collect-possessions --game 0042300212 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300213.parquet" ]; then
  echo "[1341/1456]  skip 0042300213 (already collected)"
else
  echo "[1341/1456]  2023-24 playoffs  0042300213"
  collect-possessions --game 0042300213 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300214.parquet" ]; then
  echo "[1342/1456]  skip 0042300214 (already collected)"
else
  echo "[1342/1456]  2023-24 playoffs  0042300214"
  collect-possessions --game 0042300214 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300215.parquet" ]; then
  echo "[1343/1456]  skip 0042300215 (already collected)"
else
  echo "[1343/1456]  2023-24 playoffs  0042300215"
  collect-possessions --game 0042300215 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300216.parquet" ]; then
  echo "[1344/1456]  skip 0042300216 (already collected)"
else
  echo "[1344/1456]  2023-24 playoffs  0042300216"
  collect-possessions --game 0042300216 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300217.parquet" ]; then
  echo "[1345/1456]  skip 0042300217 (already collected)"
else
  echo "[1345/1456]  2023-24 playoffs  0042300217"
  collect-possessions --game 0042300217 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300221.parquet" ]; then
  echo "[1346/1456]  skip 0042300221 (already collected)"
else
  echo "[1346/1456]  2023-24 playoffs  0042300221"
  collect-possessions --game 0042300221 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300222.parquet" ]; then
  echo "[1347/1456]  skip 0042300222 (already collected)"
else
  echo "[1347/1456]  2023-24 playoffs  0042300222"
  collect-possessions --game 0042300222 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300223.parquet" ]; then
  echo "[1348/1456]  skip 0042300223 (already collected)"
else
  echo "[1348/1456]  2023-24 playoffs  0042300223"
  collect-possessions --game 0042300223 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300224.parquet" ]; then
  echo "[1349/1456]  skip 0042300224 (already collected)"
else
  echo "[1349/1456]  2023-24 playoffs  0042300224"
  collect-possessions --game 0042300224 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300225.parquet" ]; then
  echo "[1350/1456]  skip 0042300225 (already collected)"
else
  echo "[1350/1456]  2023-24 playoffs  0042300225"
  collect-possessions --game 0042300225 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300226.parquet" ]; then
  echo "[1351/1456]  skip 0042300226 (already collected)"
else
  echo "[1351/1456]  2023-24 playoffs  0042300226"
  collect-possessions --game 0042300226 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300231.parquet" ]; then
  echo "[1352/1456]  skip 0042300231 (already collected)"
else
  echo "[1352/1456]  2023-24 playoffs  0042300231"
  collect-possessions --game 0042300231 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300232.parquet" ]; then
  echo "[1353/1456]  skip 0042300232 (already collected)"
else
  echo "[1353/1456]  2023-24 playoffs  0042300232"
  collect-possessions --game 0042300232 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300233.parquet" ]; then
  echo "[1354/1456]  skip 0042300233 (already collected)"
else
  echo "[1354/1456]  2023-24 playoffs  0042300233"
  collect-possessions --game 0042300233 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300234.parquet" ]; then
  echo "[1355/1456]  skip 0042300234 (already collected)"
else
  echo "[1355/1456]  2023-24 playoffs  0042300234"
  collect-possessions --game 0042300234 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300235.parquet" ]; then
  echo "[1356/1456]  skip 0042300235 (already collected)"
else
  echo "[1356/1456]  2023-24 playoffs  0042300235"
  collect-possessions --game 0042300235 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300236.parquet" ]; then
  echo "[1357/1456]  skip 0042300236 (already collected)"
else
  echo "[1357/1456]  2023-24 playoffs  0042300236"
  collect-possessions --game 0042300236 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300237.parquet" ]; then
  echo "[1358/1456]  skip 0042300237 (already collected)"
else
  echo "[1358/1456]  2023-24 playoffs  0042300237"
  collect-possessions --game 0042300237 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300301.parquet" ]; then
  echo "[1359/1456]  skip 0042300301 (already collected)"
else
  echo "[1359/1456]  2023-24 playoffs  0042300301"
  collect-possessions --game 0042300301 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300302.parquet" ]; then
  echo "[1360/1456]  skip 0042300302 (already collected)"
else
  echo "[1360/1456]  2023-24 playoffs  0042300302"
  collect-possessions --game 0042300302 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300303.parquet" ]; then
  echo "[1361/1456]  skip 0042300303 (already collected)"
else
  echo "[1361/1456]  2023-24 playoffs  0042300303"
  collect-possessions --game 0042300303 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300304.parquet" ]; then
  echo "[1362/1456]  skip 0042300304 (already collected)"
else
  echo "[1362/1456]  2023-24 playoffs  0042300304"
  collect-possessions --game 0042300304 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300311.parquet" ]; then
  echo "[1363/1456]  skip 0042300311 (already collected)"
else
  echo "[1363/1456]  2023-24 playoffs  0042300311"
  collect-possessions --game 0042300311 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300312.parquet" ]; then
  echo "[1364/1456]  skip 0042300312 (already collected)"
else
  echo "[1364/1456]  2023-24 playoffs  0042300312"
  collect-possessions --game 0042300312 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300313.parquet" ]; then
  echo "[1365/1456]  skip 0042300313 (already collected)"
else
  echo "[1365/1456]  2023-24 playoffs  0042300313"
  collect-possessions --game 0042300313 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300314.parquet" ]; then
  echo "[1366/1456]  skip 0042300314 (already collected)"
else
  echo "[1366/1456]  2023-24 playoffs  0042300314"
  collect-possessions --game 0042300314 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300315.parquet" ]; then
  echo "[1367/1456]  skip 0042300315 (already collected)"
else
  echo "[1367/1456]  2023-24 playoffs  0042300315"
  collect-possessions --game 0042300315 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300401.parquet" ]; then
  echo "[1368/1456]  skip 0042300401 (already collected)"
else
  echo "[1368/1456]  2023-24 playoffs  0042300401"
  collect-possessions --game 0042300401 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300402.parquet" ]; then
  echo "[1369/1456]  skip 0042300402 (already collected)"
else
  echo "[1369/1456]  2023-24 playoffs  0042300402"
  collect-possessions --game 0042300402 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300403.parquet" ]; then
  echo "[1370/1456]  skip 0042300403 (already collected)"
else
  echo "[1370/1456]  2023-24 playoffs  0042300403"
  collect-possessions --game 0042300403 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300404.parquet" ]; then
  echo "[1371/1456]  skip 0042300404 (already collected)"
else
  echo "[1371/1456]  2023-24 playoffs  0042300404"
  collect-possessions --game 0042300404 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042300405.parquet" ]; then
  echo "[1372/1456]  skip 0042300405 (already collected)"
else
  echo "[1372/1456]  2023-24 playoffs  0042300405"
  collect-possessions --game 0042300405 --output-dir "$OUTDIR"
  sleep 0.5
fi

# ── 2024-25 playoffs (84 games) ────────────────────────────
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400101.parquet" ]; then
  echo "[1373/1456]  skip 0042400101 (already collected)"
else
  echo "[1373/1456]  2024-25 playoffs  0042400101"
  collect-possessions --game 0042400101 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400102.parquet" ]; then
  echo "[1374/1456]  skip 0042400102 (already collected)"
else
  echo "[1374/1456]  2024-25 playoffs  0042400102"
  collect-possessions --game 0042400102 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400103.parquet" ]; then
  echo "[1375/1456]  skip 0042400103 (already collected)"
else
  echo "[1375/1456]  2024-25 playoffs  0042400103"
  collect-possessions --game 0042400103 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400104.parquet" ]; then
  echo "[1376/1456]  skip 0042400104 (already collected)"
else
  echo "[1376/1456]  2024-25 playoffs  0042400104"
  collect-possessions --game 0042400104 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400111.parquet" ]; then
  echo "[1377/1456]  skip 0042400111 (already collected)"
else
  echo "[1377/1456]  2024-25 playoffs  0042400111"
  collect-possessions --game 0042400111 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400112.parquet" ]; then
  echo "[1378/1456]  skip 0042400112 (already collected)"
else
  echo "[1378/1456]  2024-25 playoffs  0042400112"
  collect-possessions --game 0042400112 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400113.parquet" ]; then
  echo "[1379/1456]  skip 0042400113 (already collected)"
else
  echo "[1379/1456]  2024-25 playoffs  0042400113"
  collect-possessions --game 0042400113 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400114.parquet" ]; then
  echo "[1380/1456]  skip 0042400114 (already collected)"
else
  echo "[1380/1456]  2024-25 playoffs  0042400114"
  collect-possessions --game 0042400114 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400115.parquet" ]; then
  echo "[1381/1456]  skip 0042400115 (already collected)"
else
  echo "[1381/1456]  2024-25 playoffs  0042400115"
  collect-possessions --game 0042400115 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400121.parquet" ]; then
  echo "[1382/1456]  skip 0042400121 (already collected)"
else
  echo "[1382/1456]  2024-25 playoffs  0042400121"
  collect-possessions --game 0042400121 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400122.parquet" ]; then
  echo "[1383/1456]  skip 0042400122 (already collected)"
else
  echo "[1383/1456]  2024-25 playoffs  0042400122"
  collect-possessions --game 0042400122 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400123.parquet" ]; then
  echo "[1384/1456]  skip 0042400123 (already collected)"
else
  echo "[1384/1456]  2024-25 playoffs  0042400123"
  collect-possessions --game 0042400123 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400124.parquet" ]; then
  echo "[1385/1456]  skip 0042400124 (already collected)"
else
  echo "[1385/1456]  2024-25 playoffs  0042400124"
  collect-possessions --game 0042400124 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400125.parquet" ]; then
  echo "[1386/1456]  skip 0042400125 (already collected)"
else
  echo "[1386/1456]  2024-25 playoffs  0042400125"
  collect-possessions --game 0042400125 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400126.parquet" ]; then
  echo "[1387/1456]  skip 0042400126 (already collected)"
else
  echo "[1387/1456]  2024-25 playoffs  0042400126"
  collect-possessions --game 0042400126 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400131.parquet" ]; then
  echo "[1388/1456]  skip 0042400131 (already collected)"
else
  echo "[1388/1456]  2024-25 playoffs  0042400131"
  collect-possessions --game 0042400131 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400132.parquet" ]; then
  echo "[1389/1456]  skip 0042400132 (already collected)"
else
  echo "[1389/1456]  2024-25 playoffs  0042400132"
  collect-possessions --game 0042400132 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400133.parquet" ]; then
  echo "[1390/1456]  skip 0042400133 (already collected)"
else
  echo "[1390/1456]  2024-25 playoffs  0042400133"
  collect-possessions --game 0042400133 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400134.parquet" ]; then
  echo "[1391/1456]  skip 0042400134 (already collected)"
else
  echo "[1391/1456]  2024-25 playoffs  0042400134"
  collect-possessions --game 0042400134 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400135.parquet" ]; then
  echo "[1392/1456]  skip 0042400135 (already collected)"
else
  echo "[1392/1456]  2024-25 playoffs  0042400135"
  collect-possessions --game 0042400135 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400141.parquet" ]; then
  echo "[1393/1456]  skip 0042400141 (already collected)"
else
  echo "[1393/1456]  2024-25 playoffs  0042400141"
  collect-possessions --game 0042400141 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400142.parquet" ]; then
  echo "[1394/1456]  skip 0042400142 (already collected)"
else
  echo "[1394/1456]  2024-25 playoffs  0042400142"
  collect-possessions --game 0042400142 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400143.parquet" ]; then
  echo "[1395/1456]  skip 0042400143 (already collected)"
else
  echo "[1395/1456]  2024-25 playoffs  0042400143"
  collect-possessions --game 0042400143 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400144.parquet" ]; then
  echo "[1396/1456]  skip 0042400144 (already collected)"
else
  echo "[1396/1456]  2024-25 playoffs  0042400144"
  collect-possessions --game 0042400144 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400151.parquet" ]; then
  echo "[1397/1456]  skip 0042400151 (already collected)"
else
  echo "[1397/1456]  2024-25 playoffs  0042400151"
  collect-possessions --game 0042400151 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400152.parquet" ]; then
  echo "[1398/1456]  skip 0042400152 (already collected)"
else
  echo "[1398/1456]  2024-25 playoffs  0042400152"
  collect-possessions --game 0042400152 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400153.parquet" ]; then
  echo "[1399/1456]  skip 0042400153 (already collected)"
else
  echo "[1399/1456]  2024-25 playoffs  0042400153"
  collect-possessions --game 0042400153 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400154.parquet" ]; then
  echo "[1400/1456]  skip 0042400154 (already collected)"
else
  echo "[1400/1456]  2024-25 playoffs  0042400154"
  collect-possessions --game 0042400154 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400155.parquet" ]; then
  echo "[1401/1456]  skip 0042400155 (already collected)"
else
  echo "[1401/1456]  2024-25 playoffs  0042400155"
  collect-possessions --game 0042400155 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400156.parquet" ]; then
  echo "[1402/1456]  skip 0042400156 (already collected)"
else
  echo "[1402/1456]  2024-25 playoffs  0042400156"
  collect-possessions --game 0042400156 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400157.parquet" ]; then
  echo "[1403/1456]  skip 0042400157 (already collected)"
else
  echo "[1403/1456]  2024-25 playoffs  0042400157"
  collect-possessions --game 0042400157 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400161.parquet" ]; then
  echo "[1404/1456]  skip 0042400161 (already collected)"
else
  echo "[1404/1456]  2024-25 playoffs  0042400161"
  collect-possessions --game 0042400161 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400162.parquet" ]; then
  echo "[1405/1456]  skip 0042400162 (already collected)"
else
  echo "[1405/1456]  2024-25 playoffs  0042400162"
  collect-possessions --game 0042400162 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400163.parquet" ]; then
  echo "[1406/1456]  skip 0042400163 (already collected)"
else
  echo "[1406/1456]  2024-25 playoffs  0042400163"
  collect-possessions --game 0042400163 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400164.parquet" ]; then
  echo "[1407/1456]  skip 0042400164 (already collected)"
else
  echo "[1407/1456]  2024-25 playoffs  0042400164"
  collect-possessions --game 0042400164 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400165.parquet" ]; then
  echo "[1408/1456]  skip 0042400165 (already collected)"
else
  echo "[1408/1456]  2024-25 playoffs  0042400165"
  collect-possessions --game 0042400165 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400171.parquet" ]; then
  echo "[1409/1456]  skip 0042400171 (already collected)"
else
  echo "[1409/1456]  2024-25 playoffs  0042400171"
  collect-possessions --game 0042400171 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400172.parquet" ]; then
  echo "[1410/1456]  skip 0042400172 (already collected)"
else
  echo "[1410/1456]  2024-25 playoffs  0042400172"
  collect-possessions --game 0042400172 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400173.parquet" ]; then
  echo "[1411/1456]  skip 0042400173 (already collected)"
else
  echo "[1411/1456]  2024-25 playoffs  0042400173"
  collect-possessions --game 0042400173 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400174.parquet" ]; then
  echo "[1412/1456]  skip 0042400174 (already collected)"
else
  echo "[1412/1456]  2024-25 playoffs  0042400174"
  collect-possessions --game 0042400174 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400175.parquet" ]; then
  echo "[1413/1456]  skip 0042400175 (already collected)"
else
  echo "[1413/1456]  2024-25 playoffs  0042400175"
  collect-possessions --game 0042400175 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400176.parquet" ]; then
  echo "[1414/1456]  skip 0042400176 (already collected)"
else
  echo "[1414/1456]  2024-25 playoffs  0042400176"
  collect-possessions --game 0042400176 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400177.parquet" ]; then
  echo "[1415/1456]  skip 0042400177 (already collected)"
else
  echo "[1415/1456]  2024-25 playoffs  0042400177"
  collect-possessions --game 0042400177 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400201.parquet" ]; then
  echo "[1416/1456]  skip 0042400201 (already collected)"
else
  echo "[1416/1456]  2024-25 playoffs  0042400201"
  collect-possessions --game 0042400201 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400202.parquet" ]; then
  echo "[1417/1456]  skip 0042400202 (already collected)"
else
  echo "[1417/1456]  2024-25 playoffs  0042400202"
  collect-possessions --game 0042400202 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400203.parquet" ]; then
  echo "[1418/1456]  skip 0042400203 (already collected)"
else
  echo "[1418/1456]  2024-25 playoffs  0042400203"
  collect-possessions --game 0042400203 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400204.parquet" ]; then
  echo "[1419/1456]  skip 0042400204 (already collected)"
else
  echo "[1419/1456]  2024-25 playoffs  0042400204"
  collect-possessions --game 0042400204 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400205.parquet" ]; then
  echo "[1420/1456]  skip 0042400205 (already collected)"
else
  echo "[1420/1456]  2024-25 playoffs  0042400205"
  collect-possessions --game 0042400205 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400211.parquet" ]; then
  echo "[1421/1456]  skip 0042400211 (already collected)"
else
  echo "[1421/1456]  2024-25 playoffs  0042400211"
  collect-possessions --game 0042400211 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400212.parquet" ]; then
  echo "[1422/1456]  skip 0042400212 (already collected)"
else
  echo "[1422/1456]  2024-25 playoffs  0042400212"
  collect-possessions --game 0042400212 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400213.parquet" ]; then
  echo "[1423/1456]  skip 0042400213 (already collected)"
else
  echo "[1423/1456]  2024-25 playoffs  0042400213"
  collect-possessions --game 0042400213 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400214.parquet" ]; then
  echo "[1424/1456]  skip 0042400214 (already collected)"
else
  echo "[1424/1456]  2024-25 playoffs  0042400214"
  collect-possessions --game 0042400214 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400215.parquet" ]; then
  echo "[1425/1456]  skip 0042400215 (already collected)"
else
  echo "[1425/1456]  2024-25 playoffs  0042400215"
  collect-possessions --game 0042400215 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400216.parquet" ]; then
  echo "[1426/1456]  skip 0042400216 (already collected)"
else
  echo "[1426/1456]  2024-25 playoffs  0042400216"
  collect-possessions --game 0042400216 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400221.parquet" ]; then
  echo "[1427/1456]  skip 0042400221 (already collected)"
else
  echo "[1427/1456]  2024-25 playoffs  0042400221"
  collect-possessions --game 0042400221 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400222.parquet" ]; then
  echo "[1428/1456]  skip 0042400222 (already collected)"
else
  echo "[1428/1456]  2024-25 playoffs  0042400222"
  collect-possessions --game 0042400222 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400223.parquet" ]; then
  echo "[1429/1456]  skip 0042400223 (already collected)"
else
  echo "[1429/1456]  2024-25 playoffs  0042400223"
  collect-possessions --game 0042400223 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400224.parquet" ]; then
  echo "[1430/1456]  skip 0042400224 (already collected)"
else
  echo "[1430/1456]  2024-25 playoffs  0042400224"
  collect-possessions --game 0042400224 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400225.parquet" ]; then
  echo "[1431/1456]  skip 0042400225 (already collected)"
else
  echo "[1431/1456]  2024-25 playoffs  0042400225"
  collect-possessions --game 0042400225 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400226.parquet" ]; then
  echo "[1432/1456]  skip 0042400226 (already collected)"
else
  echo "[1432/1456]  2024-25 playoffs  0042400226"
  collect-possessions --game 0042400226 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400227.parquet" ]; then
  echo "[1433/1456]  skip 0042400227 (already collected)"
else
  echo "[1433/1456]  2024-25 playoffs  0042400227"
  collect-possessions --game 0042400227 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400231.parquet" ]; then
  echo "[1434/1456]  skip 0042400231 (already collected)"
else
  echo "[1434/1456]  2024-25 playoffs  0042400231"
  collect-possessions --game 0042400231 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400232.parquet" ]; then
  echo "[1435/1456]  skip 0042400232 (already collected)"
else
  echo "[1435/1456]  2024-25 playoffs  0042400232"
  collect-possessions --game 0042400232 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400233.parquet" ]; then
  echo "[1436/1456]  skip 0042400233 (already collected)"
else
  echo "[1436/1456]  2024-25 playoffs  0042400233"
  collect-possessions --game 0042400233 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400234.parquet" ]; then
  echo "[1437/1456]  skip 0042400234 (already collected)"
else
  echo "[1437/1456]  2024-25 playoffs  0042400234"
  collect-possessions --game 0042400234 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400235.parquet" ]; then
  echo "[1438/1456]  skip 0042400235 (already collected)"
else
  echo "[1438/1456]  2024-25 playoffs  0042400235"
  collect-possessions --game 0042400235 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400301.parquet" ]; then
  echo "[1439/1456]  skip 0042400301 (already collected)"
else
  echo "[1439/1456]  2024-25 playoffs  0042400301"
  collect-possessions --game 0042400301 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400302.parquet" ]; then
  echo "[1440/1456]  skip 0042400302 (already collected)"
else
  echo "[1440/1456]  2024-25 playoffs  0042400302"
  collect-possessions --game 0042400302 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400303.parquet" ]; then
  echo "[1441/1456]  skip 0042400303 (already collected)"
else
  echo "[1441/1456]  2024-25 playoffs  0042400303"
  collect-possessions --game 0042400303 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400304.parquet" ]; then
  echo "[1442/1456]  skip 0042400304 (already collected)"
else
  echo "[1442/1456]  2024-25 playoffs  0042400304"
  collect-possessions --game 0042400304 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400305.parquet" ]; then
  echo "[1443/1456]  skip 0042400305 (already collected)"
else
  echo "[1443/1456]  2024-25 playoffs  0042400305"
  collect-possessions --game 0042400305 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400306.parquet" ]; then
  echo "[1444/1456]  skip 0042400306 (already collected)"
else
  echo "[1444/1456]  2024-25 playoffs  0042400306"
  collect-possessions --game 0042400306 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400311.parquet" ]; then
  echo "[1445/1456]  skip 0042400311 (already collected)"
else
  echo "[1445/1456]  2024-25 playoffs  0042400311"
  collect-possessions --game 0042400311 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400312.parquet" ]; then
  echo "[1446/1456]  skip 0042400312 (already collected)"
else
  echo "[1446/1456]  2024-25 playoffs  0042400312"
  collect-possessions --game 0042400312 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400313.parquet" ]; then
  echo "[1447/1456]  skip 0042400313 (already collected)"
else
  echo "[1447/1456]  2024-25 playoffs  0042400313"
  collect-possessions --game 0042400313 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400314.parquet" ]; then
  echo "[1448/1456]  skip 0042400314 (already collected)"
else
  echo "[1448/1456]  2024-25 playoffs  0042400314"
  collect-possessions --game 0042400314 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400315.parquet" ]; then
  echo "[1449/1456]  skip 0042400315 (already collected)"
else
  echo "[1449/1456]  2024-25 playoffs  0042400315"
  collect-possessions --game 0042400315 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400401.parquet" ]; then
  echo "[1450/1456]  skip 0042400401 (already collected)"
else
  echo "[1450/1456]  2024-25 playoffs  0042400401"
  collect-possessions --game 0042400401 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400402.parquet" ]; then
  echo "[1451/1456]  skip 0042400402 (already collected)"
else
  echo "[1451/1456]  2024-25 playoffs  0042400402"
  collect-possessions --game 0042400402 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400403.parquet" ]; then
  echo "[1452/1456]  skip 0042400403 (already collected)"
else
  echo "[1452/1456]  2024-25 playoffs  0042400403"
  collect-possessions --game 0042400403 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400404.parquet" ]; then
  echo "[1453/1456]  skip 0042400404 (already collected)"
else
  echo "[1453/1456]  2024-25 playoffs  0042400404"
  collect-possessions --game 0042400404 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400405.parquet" ]; then
  echo "[1454/1456]  skip 0042400405 (already collected)"
else
  echo "[1454/1456]  2024-25 playoffs  0042400405"
  collect-possessions --game 0042400405 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400406.parquet" ]; then
  echo "[1455/1456]  skip 0042400406 (already collected)"
else
  echo "[1455/1456]  2024-25 playoffs  0042400406"
  collect-possessions --game 0042400406 --output-dir "$OUTDIR"
  sleep 0.5
fi
if [ -f "/Users/aarondanielson/Dropbox/deep_rapm/data/games/possessions_0042400407.parquet" ]; then
  echo "[1456/1456]  skip 0042400407 (already collected)"
else
  echo "[1456/1456]  2024-25 playoffs  0042400407"
  collect-possessions --game 0042400407 --output-dir "$OUTDIR"
  sleep 0.5
fi

echo "Done."
