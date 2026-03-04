from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    d: int = 384
    latents_L: int = 256
    blocks_D: int = 12
    heads_h: int = 6
    cross_attn_chunk_size: int = 32
    gene_vocab_size: int = 60_000  # filled later from datamodule

    # --- Encoder selection ---
    encoder_type: str = "perceiver"           # "perceiver" or "transformer"

    # --- Transformer-specific ---
    transformer_num_layers: int = 12
    transformer_num_heads: int = 6
    transformer_fixed_gene_count: int = 768  # total genes selected per cell
    transformer_context_gene_count: int = 384 # genes for student (rest → teacher)

    # --- masking layout (perceiver only) ---
    mask_ratio: float = 0.45
    num_targets: int = 1
    min_context_genes: int = 512
    min_target_genes_per_block: int = 16

    # --- EMA (slower, with warmup) ---
    ema_start_decay: float = 0.992
    ema_end_decay: float = 0.9995
    ema_warmup_epochs: int = 1

    # Tokenizer
    identity_value_split_ratio: float = 0.5
    fourier_num_frequencies: int = 64
    fourier_min_freq: float = 0.1
    fourier_max_freq: float = 100.0
    fourier_freq_scale: float = 1.0

    # Predictor
    predictor_depth: int = 3
    predictor_expansion_factor: int = 4

    def __post_init__(self):
        if self.encoder_type not in ("perceiver", "transformer"):
            raise ValueError(f"encoder_type must be 'perceiver' or 'transformer', got '{self.encoder_type}'")
        if self.encoder_type == "transformer":
            if self.transformer_context_gene_count >= self.transformer_fixed_gene_count:
                raise ValueError(
                    f"transformer_context_gene_count ({self.transformer_context_gene_count}) "
                    f"must be less than transformer_fixed_gene_count ({self.transformer_fixed_gene_count})"
                )


@dataclass
class TrainingConfig:
    sim_coeff: float = 1.0
    var_coeff: float = 25.0
    cov_coeff: float = 1.0

    learning_rate: float = 1e-4
    weight_decay: float = 2e-4
    warmup_ratio: float = 0.05
    max_epochs: int = 50
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.0
    adam_betas: Tuple[float, float] = (0.9, 0.98)


@dataclass
class DataConfig:
    batch_size: int = 128
    num_workers: int = 8
    train_samples: int = 1_000_000
    val_samples: int = 10_000


@dataclass
class ExperimentConfig:
    checkpoint_dir: str = "checkpoints/gene_jepa_tahoe"
    random_seed: int = 42
    log_every_n_steps: int = 10
    wandb_project: str = "genejepa"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = "GeneJEPA-Tahoe-100M"
    # Validation Config
    validation_num_batches: int = 4  # For UMAP/collapse metrics
    validation_plot_every_n_epochs: int = 1
