"""Bundle a development phase's models, data snapshot, evaluations and code.

    .venv/bin/python scripts/make_release.py --name rapm-phase1_2026-10-26 --freeze-date 2026-10-24

Writes outputs/releases/<name>/:

  models/v2.1-A/      frozen research candidate (checkpoint, cache, forward scores,
                      calibration analysis, manifest) — copied from outputs/candidates/
  models/v1/          the V1 configuration and the production rolling cache at release
  evaluation/         every forward-score parquet in outputs/forward/ (the selection,
                      ablation, holdout and density comparisons)
  calibration/        margin calibration tables and calibration_constants.json — the
                      validation-fitted variance multiplier ĉ and split-conformal |z|
                      quantiles k50 / k90 per model, frozen for prospective use
  data/               the season possession parquets and every reference table the
                      models read (vocab, player table, player info, anthropometrics,
                      PlayerGameLogs, quality table) plus data_manifest.json (rows,
                      last game date, sha256 per file)
  code/               deep_rapm/, scripts/, tests/, pyproject.toml, docs/, git HEAD,
                      git status, pip freeze of the environment
  repro/              the reproduction check of the frozen candidate
  RELEASE.md          what is here and how to reproduce
  MANIFEST.sha256     sha256 of every file in the bundle

The season parquets in data/ may hold games after the freeze date (the
nightly collection keeps running); reproduce on the frozen snapshot with
`--until <freeze date>`.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from deep_rapm.production import V1


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
    else:
        shutil.copy2(src, dst)


def calibration_constants(forward: Path) -> dict:
    m = pd.read_parquet(forward / "margin_calibration.parquet")
    k = pd.read_parquet(forward / "margin_calibration_conformal.parquet")
    out = {}
    for name, g in m.groupby("config"):
        kk = k[(k["config"] == name) & (k["split"] == "val")].iloc[0]
        out[name] = {"c": float(g["c_val"].iloc[0]), "k50": float(kk["k50"]), "k90": float(kk["k90"]),
                     "fitted_on": "validation games 2023-24 and 2024-25, regular season + playoffs"}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--name", required=True)
    p.add_argument("--freeze-date", required=True)
    p.add_argument("--candidate", type=Path, default=Path("outputs/candidates/v2.1-A"))
    args = p.parse_args()
    root = Path("outputs/releases") / args.name
    if root.exists():
        sys.exit(f"{root} exists — releases are immutable; choose a new name")

    copy(args.candidate, root / "models" / args.candidate.name)
    (root / "models/v1").mkdir(parents=True)
    (root / "models/v1/config.json").write_text(json.dumps(asdict(V1), indent=2, default=list))
    copy(Path("outputs/rolling_cache.parquet"), root / "models/v1/rolling_cache_production.parquet")

    for f in sorted(Path("outputs/forward").glob("*.parquet")):
        copy(f, root / "evaluation" / f.name)
    copy(Path("outputs/forward/margin_calibration.parquet"), root / "calibration/margin_calibration.parquet")
    copy(Path("outputs/forward/margin_calibration_conformal.parquet"),
         root / "calibration/margin_calibration_conformal.parquet")
    (root / "calibration/calibration_constants.json").write_text(
        json.dumps(calibration_constants(Path("outputs/forward")), indent=2))

    data_files = sorted(Path("data").glob("20*/possessions_*.parquet")) + [
        Path("data/player_vocab.parquet"), Path("data/players.parquet"), Path("data/player_info.parquet"),
        Path("data/player_anthropometrics.parquet"), Path("data/quality/game_quality.parquet")]
    manifest = []
    for f in data_files:
        copy(f, root / f)
        row = {"path": str(f), "sha256": sha256(f), "bytes": f.stat().st_size}
        if f.name.startswith("possessions_"):
            d = pd.read_parquet(f, columns=["game_id", "game_date"])
            row.update(rows=len(d), games=int(d["game_id"].nunique()), last_game_date=str(d["game_date"].max()),
                       rows_through_freeze=int((pd.to_datetime(d["game_date"]) <= args.freeze_date).sum()))
        manifest.append(row)
    copy(Path("data/raw/player_game_logs"), root / "data/raw/player_game_logs")
    for f in sorted(Path("data/raw/player_game_logs").glob("*.parquet")):
        manifest.append({"path": str(f), "sha256": sha256(f), "bytes": f.stat().st_size})
    (root / "data/data_manifest.json").write_text(json.dumps(
        {"freeze_date": args.freeze_date, "files": manifest}, indent=2))

    for src in ("deep_rapm", "scripts", "tests", "docs", "pyproject.toml", "README.md"):
        copy(Path(src), root / "code" / src)
    run = lambda *cmd: subprocess.run(cmd, capture_output=True, text=True).stdout
    (root / "code/GIT_HEAD.txt").write_text(run("git", "log", "-1", "--format=%H %ad %s", "--date=iso"))
    (root / "code/GIT_STATUS.txt").write_text(run("git", "status", "--porcelain"))
    (root / "code/PIP_FREEZE.txt").write_text(run(sys.executable, "-m", "pip", "freeze"))
    (root / "code/PYTHON.txt").write_text(sys.version)

    repro = Path("outputs/repro_check_2026-10-26")
    if repro.exists():
        copy(repro / "score.log", root / "repro/score.log")
        copy(repro / "comparison.txt", root / "repro/comparison.txt")

    lines = []
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.name != "MANIFEST.sha256":
            lines.append(f"{sha256(f)}  {f.relative_to(root)}")
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    size = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    print(f"{root}: {len(lines)} files, {size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
