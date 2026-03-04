# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GeneJEPA is a self-supervised foundation model for single-cell RNA-seq using a Joint-Embedding Predictive Architecture (JEPA). It learns to predict latent representations of masked gene sets from visible context, trained on the Tahoe-100M atlas (~100M cells). Built with PyTorch Lightning.

## Common Commands

```bash
# Install dependencies (preferred)
uv sync

# Train on Tahoe-100M (streaming from HuggingFace Hub)
uv run -m genejepa.train

# Train on local memmap dataset
uv run -m genejepa.train_memmap DATA/brain/processed_data/train

# train_memmap CLI options:
#   --batch-size 92       --num-workers 4       --max-epochs 50
#   --val-fraction 0.05   --lr 1e-4             --seed 42
#   --checkpoint-dir checkpoints/gene_jepa_memmap
#   --wandb-project genejepa  --wandb-run-name NAME
#   --resume PATH              (or auto-resumes from last.ckpt)
#   --neftel-eval-every 5      (Neftel scib-metrics frequency)
#   --subset 0.1               (use 10% of data for quick trial runs)

# Export gene vocabulary map without training
uv run -m genejepa.train \
  --export-foundation-map hf_data_cache/foundation_gene_map.parquet \
  --export-global-stats hf_data_cache/global_stats.json \
  --foundation-meta hf_data_cache/data/gene_metadata.parquet \
  --export-only
```

HuggingFace authentication is required: `huggingface-cli login` or set `HUGGINGFACE_HUB_TOKEN`.

## Architecture

### Core JEPA Pipeline

The model follows a student-teacher JEPA pattern:

1. **scRNATokenizer** (`tokenizer.py`) — Converts sparse gene vectors into token embeddings. Each token combines a learnable gene identity embedding with Fourier-encoded expression values (configurable 50/50 split of embedding dimension).

2. **GenePerceiverEncoder** (`models.py`) — Perceiver-style encoder that maps variable-length gene token sequences to a fixed-size latent representation via cross-attention with learnable latent tokens (default: 512 latents, 24 transformer blocks). Uses chunked cross-attention with online softmax for memory efficiency.

3. **MLPPredictor** (`models.py`) — BYOL-style predictor (3-layer, 4x expansion) that maps student context embeddings to predicted target representations.

4. **GenePerceiverJEPA** (`models.py`) — Wires everything together: random block masking splits genes into context/target sets, student encodes context, EMA teacher encodes targets, predictor bridges the gap. Loss is VICReg-style (similarity + variance + covariance regularization).

### Data Representation

Cells use a **ragged tensor format**: `(indices, values, offsets)` where indices are gene IDs, values are normalized expression levels (log1p + z-score), and offsets delimit cells in the batch. This efficiently handles variable numbers of expressed genes per cell.

### Training Module

`JepaLightningModule` (`train.py`) wraps the JEPA model with:
- VICReg loss with separate coefficients for similarity, variance, and covariance terms
- EMA teacher with cosine warmup schedule (0.992 → 0.9995)
- Stability gating: gates similarity loss based on teacher dispersion during early training
- AdamW with cosine LR decay and 5% warmup

### Data Modules

- **Tahoe100MDataModule** (`data.py`) — Streams from HuggingFace Hub (3,388 parquet shards). Rank-0 downloads file manifest, computes global normalization stats, then broadcasts to all workers.
- **MemmapDataModule** (`data_memmap.py`) — Loads from local memmap-format datasets via bionemo-scdl's `SingleCellMemMapDataset`. Expects a `DatasetDir` layout: `vocab.json` (gene symbol → token ID), `obs.parquet` (cell metadata), `mem.map` (memmap array), `mapping.json` (memmap schema). Computes global normalization stats via Welford's online algorithm (sampled from up to 50k cells, cached to `global_stats.json`). Uses `random_split` for train/val (default 95/5). Expression values are assumed already log1p-transformed; collation only applies z-score standardization. Supports DDP via `DistributedSampler`.

### Configuration

All hyperparameters live in four dataclasses in `configs.py`: `ModelConfig`, `TrainingConfig`, `DataConfig`, `ExperimentConfig`. No external config files — defaults are modified via CLI args or in code.

### Callbacks

`callbacks.py` provides three validation callbacks (plus helpers `_norm_sym`, `LinearProbeMLP`, `_SubsetByIndices`, `_ListDataset`):

- **EmbeddingQualityValidator** — Collapse metrics (avg cosine similarity, norm std) + UMAP visualization. Reads embeddings from `pl_module._validation_cache` populated during the main val loop. DDP-safe: all-gathers embeddings across ranks with padded tensors, rank-0 computes metrics and plots. UMAP runs on rank-0 local data only (subsampled to 5k points). Logs histograms and scatter plots to W&B.

- **SupervisedValidatorCallback** — Linear probe on external HF Hub datasets (e.g. Human Lung Cell Atlas). Lazy-initializes on rank-0: downloads foundation gene metadata from Tahoe-100M, loads probe AnnData from HF Hub, builds vocabulary overlap map via `_norm_sym` (case-insensitive gene symbol matching). Falls back to scvi-hub denoised expression for minified AnnData (zero-expression). Trains a 2-layer MLP probe (LayerNorm → Linear → GELU → Linear) for a configurable number of epochs, logs `val_probe/accuracy`.

- **ScibMetricsCallback** — Computes scib-metrics (silhouette_label, silhouette_batch, NMI, ARI via k-means) on two data sources. Every epoch: on the memmap validation split (reads batch/label from `obs.parquet`). Every N epochs: on an external Neftel h5ad file (maps gene symbols to `vocab.json` case-insensitively). Lazy-initializes loaders on rank-0. Configurable batch/label keys for both val (`val_batch_key`, `val_label_key`) and Neftel (`neftel_batch_key`, `neftel_label_key`). Max cells capped at 5k by default.

## Key Design Decisions

- **Numerical stability**: Attention computation upcasts to float32; padding masks use -inf carefully
- **Memory efficiency**: Gradient checkpointing in encoder, chunked cross-attention, mixed precision (auto-detects bf16 vs fp16)
- **DDP safety**: Barrier synchronization for file downloads and stats computation; safe worker sharding with single-shard fallback
- **ID-independent masking**: Mask blocks are random over position, not gene ID, to prevent information leakage
