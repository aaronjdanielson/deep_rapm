"""
train_script.py — CLI entry point for Deep RAPM training.

Usage
-----
Defaults (5 train seasons, 2023-24 held out, standard hyperparameters):
    train-deep-rapm

Custom hyperparameters:
    train-deep-rapm --d 128 --num-layers 3 --epochs 50 --batch-size 2048

Quick smoke test (1 epoch, small model):
    train-deep-rapm --epochs 1 --d 32 --num-layers 1

Specific output directory:
    train-deep-rapm --output-dir runs/exp_01
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deep_rapm.train import TrainConfig, train_model


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train-deep-rapm",
        description="Train Deep RAPM on NBA possession data.",
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    data = p.add_argument_group("data")
    data.add_argument(
        "--train-seasons", nargs="+",
        default=["2018-19", "2019-20", "2020-21", "2021-22", "2022-23"],
        metavar="SEASON",
        help="Seasons to use for training/validation (default: 5 seasons).",
    )
    data.add_argument(
        "--test-season", default="2023-24", metavar="SEASON",
        help="Held-out test season (default: 2023-24).",
    )
    data.add_argument(
        "--data-dir", type=Path, default=Path("data"), metavar="DIR",
        help="Root data directory (default: data/).",
    )
    data.add_argument(
        "--player-vocab", type=Path, default=Path("data/player_vocab.parquet"),
        metavar="PATH", help="Player vocab parquet (default: data/player_vocab.parquet).",
    )
    data.add_argument(
        "--player-table", type=Path, default=Path("data/players.parquet"),
        metavar="PATH", help="Player table parquet (default: data/players.parquet).",
    )
    data.add_argument(
        "--val-fraction", type=float, default=0.15,
        help="Fraction of training games held out for validation (default: 0.15).",
    )
    data.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = p.add_argument_group("model")
    model.add_argument("--model", choices=["deep", "linear", "cross"], default="deep",
                       help="Model architecture: 'deep' (Set Transformer), "
                            "'linear' (sum-pooling neural RAPM baseline), or "
                            "'cross' (cross-attention RAPM). (default: deep)")
    model.add_argument("--d", type=int, default=64,
                       help="Embedding / model dimension (default: 64).")
    model.add_argument("--num-heads", type=int, default=4,
                       help="Attention heads (default: 4).")
    model.add_argument("--num-layers", type=int, default=2,
                       help="Transformer layers (default: 2).")
    model.add_argument("--head-hidden", type=int, default=128,
                       help="Prediction head hidden dim (default: 128).")
    model.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate (default: 0.1).")
    model.add_argument("--embedding-reg", type=float, default=1e-4,
                       help="L2 penalty on player embeddings (default: 1e-4).")

    # ── Training ──────────────────────────────────────────────────────────────
    train = p.add_argument_group("training")
    train.add_argument("--batch-size", type=int, default=1024,
                       help="Batch size (default: 1024).")
    train.add_argument("--lr", type=float, default=3e-4,
                       help="Peak learning rate (default: 3e-4).")
    train.add_argument("--weight-decay", type=float, default=1e-4,
                       help="AdamW weight decay on non-embedding params (default: 1e-4).")
    train.add_argument("--epochs", type=int, default=30,
                       help="Training epochs (default: 30).")
    train.add_argument("--grad-clip", type=float, default=1.0,
                       help="Gradient norm clip (default: 1.0).")

    # ── Player features (CrossRAPM) ───────────────────────────────────────────
    feats = p.add_argument_group("player features (cross model only)")
    feats.add_argument(
        "--box-dir", type=Path, default=None, metavar="DIR",
        help="Directory of per-game box score parquets (data/box/).  "
             "Required to enable EWMA player features for --model cross.",
    )
    feats.add_argument(
        "--feature-half-life", type=float, default=None, metavar="DAYS",
        help="Half-life in days for EWMA player feature recency weighting "
             "(default: 180).  Only used when --box-dir is set.",
    )

    # ── Warm-start ────────────────────────────────────────────────────────────
    p.add_argument("--rapm-dir", type=Path, default=None, metavar="DIR",
                   help="Directory from solve-rapm containing rapm.parquet and "
                        "rapm_summary.json.  If set, embeddings are initialised "
                        "from the analytical RAPM solution before training.")

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints"),
                   help="Checkpoint and results directory (default: checkpoints/).")
    p.add_argument("--log-every", type=int, default=200,
                   help="Log training loss every N batches (default: 200).")

    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    cfg = TrainConfig(
        train_seasons=args.train_seasons,
        test_season=args.test_season,
        data_dir=args.data_dir,
        player_vocab_path=args.player_vocab,
        player_table_path=args.player_table,
        val_fraction=args.val_fraction,
        seed=args.seed,
        model_type=args.model,
        d=args.d,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        head_hidden=args.head_hidden,
        dropout=args.dropout,
        embedding_reg=args.embedding_reg,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        grad_clip=args.grad_clip,
        box_dir=args.box_dir,
        feature_half_life=args.feature_half_life,
        rapm_dir=args.rapm_dir,
        output_dir=args.output_dir,
        log_every=args.log_every,
    )

    train_model(cfg)


if __name__ == "__main__":
    main()
