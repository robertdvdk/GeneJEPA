"""
Training script for GeneJEPA using single-cell memmap datasets.

Usage:
    uv run python -m genejepa.train_memmap DATA/brain/processed_data/train

    # With options:
    uv run python -m genejepa.train_memmap DATA/brain/processed_data/train \
        --batch-size 64 --max-epochs 50 --num-workers 4
"""

import argparse
import math
import os
import logging
from dataclasses import asdict
from typing import List

import torch
torch.set_float32_matmul_precision("high")
import os
os.environ["TORCH_LOGS"] = "+sdpa"
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger

try:
    import wandb
except ImportError:
    wandb = None

from .configs import ModelConfig, TrainingConfig, ExperimentConfig
from .train import JepaLightningModule
from .callbacks import EmbeddingQualityValidator, ScibMetricsCallback

from .data_memmap import MemmapDataModule

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train GeneJEPA on memmap data")
    parser.add_argument("data_dir", type=str, help="Path to memmap dataset directory")
    parser.add_argument("--batch-size", type=int, default=92)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/test")
    parser.add_argument("--wandb-project", type=str, default="genejepa")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--neftel-eval-every", type=int, default=5, help="Run Neftel scib-metrics every N epochs")
    parser.add_argument("--subset", type=float, default=1.0, help="Fraction of data to use (e.g. 0.1 for 10%% trial run)")
    parser.add_argument("--encoder-type", type=str, default="transformer", choices=["perceiver", "transformer"],
                        help="Encoder architecture: 'perceiver' (default) or 'transformer'")
    args = parser.parse_args()

    # --- Configs ---
    model_config = ModelConfig(encoder_type=args.encoder_type)
    train_config = TrainingConfig()
    train_config.max_epochs = args.max_epochs
    train_config.learning_rate = args.lr

    exp_config = ExperimentConfig()
    exp_config.checkpoint_dir = args.checkpoint_dir
    exp_config.random_seed = args.seed
    exp_config.wandb_project = args.wandb_project
    exp_config.wandb_run_name = args.wandb_run_name or f"GeneJEPA-memmap-{os.path.basename(args.data_dir)}"

    L.seed_everything(exp_config.random_seed, workers=True)

    # --- Data ---
    datamodule = MemmapDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        seed=args.seed,
        subset_fraction=args.subset,
        encoder_type=args.encoder_type,
        fixed_gene_count=model_config.transformer_fixed_gene_count,
    )
    datamodule.prepare_data()
    datamodule.setup("fit")

    model_config.gene_vocab_size = datamodule.gene_vocab_size
    log.info(f"Gene vocab size: {model_config.gene_vocab_size}")
    log.info(f"Train: {len(datamodule.train_dataset)}, Val: {len(datamodule.val_dataset)}")

    # --- Training steps ---
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    steps_per_epoch = math.ceil(
        len(datamodule.train_dataset) / (args.batch_size * num_devices)
    )
    updates_per_epoch = math.ceil(steps_per_epoch / train_config.accumulate_grad_batches)
    total_steps = updates_per_epoch * train_config.max_epochs
    log.info(f"{steps_per_epoch} batches/epoch, {updates_per_epoch} updates/epoch, {total_steps} total steps")

    # --- Model ---
    trainer_module = JepaLightningModule(model_config, train_config, exp_config, total_steps)

    # --- Logger ---
    logger = None
    if wandb is not None:
        logger = WandbLogger(
            project=exp_config.wandb_project,
            entity=exp_config.wandb_entity,
            name=exp_config.wandb_run_name,
            config={
                "model": asdict(model_config),
                "training": asdict(train_config),
                "data_dir": args.data_dir,
                "batch_size": args.batch_size,
            },
            save_dir="logs/",
        )

    # --- Callbacks ---
    os.makedirs(exp_config.checkpoint_dir, exist_ok=True)
    callbacks: List[Callback] = [
        ModelCheckpoint(
            dirpath=exp_config.checkpoint_dir,
            monitor="val/loss",
            save_top_k=2,
            mode="min",
            save_last=True,
            filename="scjepa-memmap-{epoch:02d}-{val/loss:.3f}",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    if logger:
        callbacks.append(
            EmbeddingQualityValidator(
                num_batches=exp_config.validation_num_batches,
                plot_every_n_epochs=exp_config.validation_plot_every_n_epochs,
            )
        )

    scib_callback = ScibMetricsCallback(
        data_dir=args.data_dir,
        neftel_h5ad_path="DATA/neftel_ss2.h5ad",
        global_mean=datamodule.global_mean,
        global_std=datamodule.global_std,
        val_dataset=datamodule.val_dataset,
        neftel_every_n_epochs=args.neftel_eval_every,
    )
    callbacks.append(scib_callback)

    # --- Trainer ---
    strategy = "auto"
    if num_devices > 1:
        strategy = DDPStrategy(find_unused_parameters=True)

    supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    precision = "bf16-mixed" if supports_bf16 else "16-mixed"

    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        strategy=strategy,
        precision=precision,
        max_epochs=train_config.max_epochs,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=exp_config.log_every_n_steps,
        accumulate_grad_batches=train_config.accumulate_grad_batches,
        gradient_clip_val=train_config.gradient_clip_val,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
    )

    # --- Resume ---
    ckpt_path = args.resume
    if ckpt_path is None:
        last_ckpt = os.path.join(exp_config.checkpoint_dir, "last.ckpt")
        if os.path.isfile(last_ckpt):
            ckpt_path = last_ckpt
            log.info(f"Resuming from {ckpt_path}")

    # --- Train ---
    log.info("Starting training.")
    trainer.fit(trainer_module, datamodule=datamodule, ckpt_path=ckpt_path)
    log.info("Training complete.")


if __name__ == "__main__":
    main()
