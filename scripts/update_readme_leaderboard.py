"""Splice the current leaderboard (top 25 RAPM, top 10 ORAPM, top 10 DRAPM)
into README.md between the leaderboard:start / leaderboard:end markers.

Only players who have appeared in a game within ACTIVE_DAYS of the latest
evaluation date are ranked: the rolling cache keeps every player who ever
cleared the possession filter, including retired ones.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ACTIVE_DAYS = 365
README = Path("README.md")

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--cache", type=Path, default=Path("outputs/rolling_cache.parquet"))
args = parser.parse_args()

df = pd.read_parquet(args.cache)
last = df["date"].max()
cur = df[df["date"] == last].copy()

last_seen: dict[int, pd.Timestamp] = {}
for p in sorted(Path("data").glob("20*/possessions_*.parquet")):
    g = pd.read_parquet(p, columns=["game_date", "offense_player_ids", "defense_player_ids"])
    g["game_date"] = pd.to_datetime(g["game_date"])
    for col in ("offense_player_ids", "defense_player_ids"):
        ids = np.vstack(g[col].values)
        dates = np.repeat(g["game_date"].values, ids.shape[1])
        s = pd.Series(dates, index=ids.ravel()).groupby(level=0).max()
        for pid, d in s.items():
            if pid not in last_seen or d > last_seen[pid]:
                last_seen[pid] = d

cur["last_game"] = cur["player_id"].map(last_seen)
cutoff = last - pd.Timedelta(days=ACTIVE_DAYS)
inactive = cur[cur["last_game"] < cutoff]
cur = cur[cur["last_game"] >= cutoff]


def fmt(x: float) -> str:
    return f"{round(x, 1) + 0.0:+.1f}".replace("-", "−")


def table(frame: pd.DataFrame, key: str, n: int, cols: list[tuple[str, str]]) -> str:
    top = frame.nlargest(n, key)
    head = "| # | Player | " + " | ".join(h for h, _ in cols) + " |"
    sep = "|--:|---|" + "|".join("--:" for _ in cols) + "|"
    lines = [head, sep]
    for i, (_, r) in enumerate(top.iterrows(), 1):
        cells = [f"**{fmt(r[c])}**" if c == key else fmt(r[c]) for _, c in cols]
        lines.append(f"| {i} | {r['player_name']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


asof = f"{last:%B} {last.day}, {last:%Y}"
block = f"""<!-- leaderboard:start -->
The most recent weekly fit, using every regular-season, playoff and preseason possession through **{asof}**. Values are points per 100 possessions relative to an average player; positive DRAPM is good defense. Ranked: players who have appeared in a game in the past year.

### Top 25 overall

{table(cur, "rapm", 25, [("RAPM", "rapm"), ("ORAPM", "orapm"), ("DRAPM", "drapm")])}

<table>
<tr>
<td valign="top" width="50%">

### Top 10 offense

{table(cur, "orapm", 10, [("ORAPM", "orapm"), ("RAPM", "rapm")])}

</td>
<td valign="top" width="50%">

### Top 10 defense

{table(cur, "drapm", 10, [("DRAPM", "drapm"), ("RAPM", "rapm")])}

</td>
</tr>
</table>
<!-- leaderboard:end -->"""

text = README.read_text()
new, n = re.subn(r"<!-- leaderboard:start -->.*?<!-- leaderboard:end -->",
                 lambda _: block, text, flags=re.S)
assert n == 1, "leaderboard markers not found exactly once in README.md"
README.write_text(new)
print(f"README leaderboard updated (as of {last.date()}, {len(cur)} active players ranked)")
top_inactive = inactive.nlargest(3, "rapm")[["player_name", "rapm", "last_game"]]
if len(top_inactive):
    print("excluded as inactive (highest RAPM):")
    print(top_inactive.to_string(index=False))
