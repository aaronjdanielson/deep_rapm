"""
train.py — Deep RAPM training loop.

Usage (CLI):
    train-deep-rapm

Usage (Python):
    from deep_rapm.train import TrainConfig, train_model
    cfg = TrainConfig(test_season="2023-24")
    results = train_model(cfg)

Design notes
------------
- Train/val split is at the game level within the 5 training seasons.
  2023-24 is held out entirely as the test set and never touched during
  training or model selection.
- AdamW with separate param groups: player embeddings get no weight_decay
  (they are regularised via the model's own embedding_penalty); all other
  parameters use weight_decay.
- Embedding mean-centering after every optimizer step (Section 5.3 of the
  paper) to pin the translation-invariant direction and make individual
  player values interpretable.
- Best checkpoint selected by minimum validation loss.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .data.dataset import make_possession_splits, PossessionDataset
from .model import DeepRAPM
from .model_cross_rapm import CrossRAPM
from .model_linear import LinearRAPM


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # ── Data ─────────────────────────────────────────────────────────────────
    train_seasons: list[str] = field(default_factory=lambda: [
        "2018-19", "2019-20", "2020-21", "2021-22", "2022-23",
    ])
    test_season: str = "2023-24"
    data_dir: Path = field(default_factory=lambda: Path("data"))
    player_vocab_path: Path = field(default_factory=lambda: Path("data/player_vocab.parquet"))
    player_table_path: Path = field(default_factory=lambda: Path("data/players.parquet"))
    val_fraction: float = 0.15
    seed: int = 42

    # ── Model ─────────────────────────────────────────────────────────────────
    model_type: str = "deep"   # "deep" = DeepRAPM, "linear" = LinearRAPM, "cross" = CrossRAPM
    d: int = 64
    num_heads: int = 4
    num_layers: int = 2
    head_hidden: int = 128
    ffn_ratio: int = 4
    dropout: float = 0.1
    embedding_reg: float = 1e-4
    score_diff_scale: float = 10.0

    # ── Optimiser ────────────────────────────────────────────────────────────
    batch_size: int = 1024
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 30
    grad_clip: float = 1.0

    # ── Warm-start ───────────────────────────────────────────────────────────
    rapm_dir: Optional[Path] = None   # directory from solve-rapm; None = cold start

    # ── Player features (CrossRAPM only) ─────────────────────────────────────
    box_dir: Optional[Path] = None          # data/box — per-game box score parquets
    feature_half_life: Optional[float] = None  # half-life in days for EWMA; None = position-only

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    log_every: int = 200     # batches between progress lines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loader(ds: PossessionDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,        # keep simple; add pin_memory when moving to GPU
        drop_last=False,
    )


def _center_embeddings(model) -> None:
    """Delegate to model's own centering method (no-op for LinearRAPM)."""
    model.center_embeddings()


@torch.no_grad()
def _warm_start_from_rapm(model, rapm_dir: Path) -> None:
    """
    Initialise model embeddings from the analytical ridge RAPM solution.

    Works for LinearRAPM and CrossRAPM (both have off_embed / def_embed / bias).
    For CrossRAPM with d > 1, only the first latent dimension is set; the rest
    retain their small-random initialisation and are free to learn matchup
    structure.

    off_embed[idx, 0] ← orapm  / 100
    def_embed[idx, 0] ← −drapm / 100
    bias              ← intercept
    """
    import pandas as pd

    rapm_df  = pd.read_parquet(rapm_dir / "rapm.parquet")
    summary  = json.loads((rapm_dir / "rapm_summary.json").read_text())

    model.bias.fill_(summary["intercept"])

    idxs      = torch.tensor(rapm_df["player_idx"].values.tolist(), dtype=torch.long)
    orapm     = torch.tensor((rapm_df["orapm"].values / 100).tolist(), dtype=torch.float32)
    neg_drapm = torch.tensor((-rapm_df["drapm"].values / 100).tolist(), dtype=torch.float32)

    model.off_embed.weight[idxs, 0] = orapm
    model.def_embed.weight[idxs, 0] = neg_drapm

    print(
        f"  Warm-start: intercept={summary['intercept']:.4f}  "
        f"players initialised={len(idxs):,}"
    )


def _extract_features(batch: dict, device: torch.device):
    """Return (off_feats, def_feats) from batch, or (None, None) if absent."""
    off_feats = batch.get("offense_features")
    def_feats = batch.get("defense_features")
    if off_feats is not None:
        off_feats = off_feats.to(device)
        def_feats = def_feats.to(device)
    return off_feats, def_feats


@torch.no_grad()
def _evaluate(model: DeepRAPM, loader: DataLoader, device: torch.device) -> dict:
    """Return val MSE loss and RMSE over the full loader."""
    model.eval()
    total_loss = 0.0
    total_sq_err = 0.0
    total_n = 0

    for batch in loader:
        off_ids = batch["offense_ids"].to(device)
        def_ids = batch["defense_ids"].to(device)
        off_pos = batch["offense_pos"].to(device)
        def_pos = batch["defense_pos"].to(device)
        gs      = batch["gamestate"].to(device)
        targets = batch["target"].to(device)
        off_feats, def_feats = _extract_features(batch, device)

        loss = model.compute_loss(
            off_ids, def_ids, off_pos, def_pos, targets, gs,
            off_feats, def_feats,
        )
        mu = model(off_ids, def_ids, off_pos, def_pos, gs, off_feats, def_feats)

        total_loss    += loss.item() * len(targets)
        total_sq_err  += (mu - targets).pow(2).sum().item()
        total_n       += len(targets)

    model.train()
    return {
        "loss": total_loss / total_n,
        "rmse": (total_sq_err / total_n) ** 0.5,
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_model(cfg: TrainConfig) -> dict:
    """Train DeepRAPM and return final metrics.

    Returns
    -------
    dict with keys: best_val_loss, best_epoch, test_loss, test_rmse,
    checkpoint_path.
    """
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config alongside checkpoints for reproducibility
    config_path = output_dir / "config.json"
    cfg_dict = {k: str(v) if isinstance(v, Path) else v
                for k, v in asdict(cfg).items()}
    config_path.write_text(json.dumps(cfg_dict, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    print("\nBuilding datasets…")
    t0 = time.time()

    # Pre-compute EWMA player feature arrays for CrossRAPM
    off_feat_train = off_feat_test = None
    def_feat_train = def_feat_test = None

    if cfg.model_type == "cross" and cfg.box_dir is not None:
        import pandas as pd
        from .data.feature_lookup import build_feature_arrays

        half_life = cfg.feature_half_life or 180.0

        # Load full training possession data
        _train_frames = []
        for season in cfg.train_seasons:
            season_dir = Path(cfg.data_dir) / season
            parquets = sorted(season_dir.glob("possessions_*.parquet"))
            if parquets:
                _train_frames.append(pd.read_parquet(parquets[0]))
        _train_poss = pd.concat(_train_frames, ignore_index=True)

        _vocab = pd.read_parquet(cfg.player_vocab_path)
        _table = pd.read_parquet(cfg.player_table_path)

        print(f"\nComputing EWMA features (half_life={half_life:.0f}d)…")
        off_feat_train, def_feat_train = build_feature_arrays(
            possession_df=_train_poss,
            box_dir=Path(cfg.box_dir),
            player_vocab=_vocab,
            player_table=_table,
            half_life_days=half_life,
        )

        # Test season: EWMA uses training history + test games (no future leakage
        # since each possession only sees games before its own date).
        _test_dir = Path(cfg.data_dir) / cfg.test_season
        _test_parquets = sorted(_test_dir.glob("possessions_*.parquet"))
        if _test_parquets:
            _test_poss = pd.read_parquet(_test_parquets[0])
            _all_poss = pd.concat([_train_poss, _test_poss], ignore_index=True)
            _all_off, _all_def = build_feature_arrays(
                possession_df=_all_poss,
                box_dir=Path(cfg.box_dir),
                player_vocab=_vocab,
                player_table=_table,
                half_life_days=half_life,
            )
            n_train = len(_train_poss)
            off_feat_test = _all_off[n_train:]
            def_feat_test = _all_def[n_train:]

    train_ds, val_ds = make_possession_splits(
        data_dir=cfg.data_dir,
        seasons=cfg.train_seasons,
        player_vocab_path=cfg.player_vocab_path,
        player_table_path=cfg.player_table_path,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
        score_diff_scale=cfg.score_diff_scale,
        offense_features=off_feat_train,
        defense_features=def_feat_train,
    )
    test_ds, _ = make_possession_splits(
        data_dir=cfg.data_dir,
        seasons=[cfg.test_season],
        player_vocab_path=cfg.player_vocab_path,
        player_table_path=cfg.player_table_path,
        val_fraction=0.0,
        seed=cfg.seed,
        score_diff_scale=cfg.score_diff_scale,
        offense_features=off_feat_test,
        defense_features=def_feat_test,
    )
    print(f"  Train: {len(train_ds):,} possessions")
    print(f"  Val:   {len(val_ds):,} possessions")
    print(f"  Test:  {len(test_ds):,} possessions  (held-out {cfg.test_season})")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    train_loader = _make_loader(train_ds, cfg.batch_size, shuffle=True)
    val_loader   = _make_loader(val_ds,   cfg.batch_size, shuffle=False)
    test_loader  = _make_loader(test_ds,  cfg.batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    import pandas as pd
    vocab = pd.read_parquet(cfg.player_vocab_path)
    num_players = len(vocab) + 1   # +1 for UNKNOWN (index 0)

    # Build player feature tensor for CrossRAPM (one-hot position, etc.)
    player_features = None
    if cfg.model_type == "cross":
        from .data.players import build_player_features
        player_table = pd.read_parquet(cfg.player_table_path)
        player_features = build_player_features(player_table, vocab)
        print(f"\nPlayer features: shape={list(player_features.shape)}"
              f"  (one-hot position per player)")

    model_cls = {"linear": LinearRAPM, "cross": CrossRAPM}.get(cfg.model_type, DeepRAPM)
    model = model_cls(
        num_players=num_players,
        d=cfg.d,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        gamestate_dim=1,          # score_diff / scale
        head_hidden=cfg.head_hidden,
        ffn_ratio=cfg.ffn_ratio,
        dropout=cfg.dropout,
        loss="mse",
        embedding_reg=cfg.embedding_reg,
        player_features=player_features,
    ).to(device)

    if cfg.rapm_dir is not None:
        print("\nWarm-starting from analytical RAPM…")
        _warm_start_from_rapm(model, Path(cfg.rapm_dir))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {n_params:,} parameters  (num_players={num_players})")

    # ── Optimiser — separate param groups ─────────────────────────────────────
    # Player embeddings: no AdamW weight_decay (handled by embedding_penalty)
    # All other params: weight_decay for transformer regularisation
    embed_params = model.embed_parameters()
    embed_param_ids = {id(p) for p in embed_params}
    other_params = [p for p in model.parameters() if id(p) not in embed_param_ids]

    optimizer = AdamW(
        [
            {"params": embed_params, "weight_decay": 0.0},
            {"params": other_params, "weight_decay": cfg.weight_decay},
        ],
        lr=cfg.lr,
    )

    total_steps = len(train_loader) * cfg.max_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=cfg.lr / 20)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_epoch = -1
    checkpoint_path = output_dir / "best.pt"

    print(f"\n{'Epoch':>5}  {'Batch':>6}  {'TrainLoss':>10}  {'ValLoss':>10}  {'ValRMSE':>8}")
    print("-" * 50)

    global_step = 0
    running_loss = 0.0
    running_n = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        epoch_t0 = time.time()

        for batch_idx, batch in enumerate(train_loader, 1):
            off_ids = batch["offense_ids"].to(device)
            def_ids = batch["defense_ids"].to(device)
            off_pos = batch["offense_pos"].to(device)
            def_pos = batch["defense_pos"].to(device)
            gs      = batch["gamestate"].to(device)
            targets = batch["target"].to(device)
            off_feats, def_feats = _extract_features(batch, device)

            optimizer.zero_grad()
            loss = model.compute_loss(
                off_ids, def_ids, off_pos, def_pos, targets, gs,
                off_feats, def_feats,
            )
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            # Mean-center player embeddings after every update
            _center_embeddings(model)

            running_loss += loss.item() * len(targets)
            running_n    += len(targets)
            global_step  += 1

            if batch_idx % cfg.log_every == 0:
                avg_loss = running_loss / running_n
                print(f"{epoch:>5}  {batch_idx:>6}  {avg_loss:>10.4f}", flush=True)
                running_loss = 0.0
                running_n = 0

        # End of epoch: validate
        val_metrics = _evaluate(model, val_loader, device)
        elapsed = time.time() - epoch_t0

        train_avg = running_loss / running_n if running_n > 0 else float("nan")
        print(
            f"{epoch:>5}  {'end':>6}  {train_avg:>10.4f}"
            f"  {val_metrics['loss']:>10.4f}  {val_metrics['rmse']:>7.4f}"
            f"  ({elapsed:.0f}s)",
            flush=True,
        )
        running_loss = 0.0
        running_n = 0

        # Save best checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "val_rmse": val_metrics["rmse"],
                    "config":   cfg_dict,
                },
                checkpoint_path,
            )

    # ── Test evaluation (held-out season) ─────────────────────────────────────
    print(f"\nLoading best checkpoint (epoch {best_epoch}, val_loss={best_val_loss:.4f})…")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    print(f"Evaluating on held-out test season ({cfg.test_season})…")
    test_metrics = _evaluate(model, test_loader, device)

    print(f"\n{'='*50}")
    print(f"Best epoch:   {best_epoch}")
    print(f"Val  loss:    {best_val_loss:.4f}")
    print(f"Test loss:    {test_metrics['loss']:.4f}")
    print(f"Test RMSE:    {test_metrics['rmse']:.4f}")
    print(f"Checkpoint:   {checkpoint_path}")
    print(f"{'='*50}")

    results = {
        "best_epoch":       best_epoch,
        "best_val_loss":    best_val_loss,
        "test_loss":        test_metrics["loss"],
        "test_rmse":        test_metrics["rmse"],
        "checkpoint_path":  str(checkpoint_path),
    }

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))

    return results
