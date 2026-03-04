"""
LightningDataModule for single-cell memmap datasets.

Reads memmap data via bionemo-scdl and produces the ragged tensor format
(indices, values, offsets) expected by GenePerceiverJEPA.forward().
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, random_split
import lightning as L

from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset

log = logging.getLogger(__name__)


class DatasetDir:
    VOCAB_PATH = "vocab.json"
    MEMMAP_PATH = "mem.map"
    MAPPING_PATH = "mapping.json"
    OBS_PATH = "obs.parquet"

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def validate(self):
        return all(
            [
                self.vocab_path.is_file(),
                self.obs_path.is_file(),
                self.mapping_path.is_file(),
                self.memmap_path.is_file(),
            ]
        )

    def mkdir(self):
        self.data_dir.mkdir(exist_ok=True, parents=True)

    @property
    def memmap_path(self):
        return self.data_dir / self.MEMMAP_PATH

    @property
    def mapping_path(self):
        return self.data_dir / self.MAPPING_PATH

    @property
    def vocab_path(self):
        return self.data_dir / self.VOCAB_PATH

    @property
    def obs_path(self):
        return self.data_dir / self.OBS_PATH


class MemmapCellDataset(Dataset):
    """Returns sparse gene data per cell for ragged-tensor collation."""

    def __init__(self, data_dir: str | Path, min_genes: int = 0):
        self.data_dir = DatasetDir(data_dir)
        self.memmap = SingleCellMemMapDataset(self.data_dir.memmap_path)
        with open(self.data_dir.vocab_path, "r") as f:
            self.vocab = json.load(f)
        self.min_genes = min_genes
        self._rng = np.random.RandomState()

    def __len__(self):
        return self.memmap.number_of_rows()

    def _load_cell(self, idx):
        exp, genes = self.memmap.get_row_padded(
            idx, return_features=True, feature_vars=["_cf_gene_id"]
        )
        gene_ids = genes[0]  # vocab token IDs
        nonzero_mask = exp != 0
        return {
            "gene_indices": gene_ids[nonzero_mask].astype(np.int64),
            "values": exp[nonzero_mask].astype(np.float32),
        }

    def __getitem__(self, idx):
        cell = self._load_cell(idx)
        if self.min_genes > 0:
            max_retries = 50
            attempt = 0
            while len(cell["gene_indices"]) < self.min_genes and attempt < max_retries:
                idx = self._rng.randint(0, len(self))
                cell = self._load_cell(idx)
                attempt += 1
        return cell


class MemmapDataModule(L.LightningDataModule):
    """
    DataModule that reads single-cell memmap data and produces ragged tensor
    batches compatible with GenePerceiverJEPA.

    Expression values in the memmap are assumed to be already log1p-transformed,
    so the collate function only applies z-score standardization.
    """

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 92,
        num_workers: int = 4,
        val_fraction: float = 0.05,
        seed: int = 42,
        subset_fraction: float = 1.0,
        encoder_type: str = "perceiver",
        fixed_gene_count: int = 1024,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.seed = seed
        self.subset_fraction = subset_fraction
        self.encoder_type = encoder_type
        self.fixed_gene_count = fixed_gene_count

        self._dataset_dir = DatasetDir(self.data_dir)
        self.stats_path = self.data_dir / "global_stats.json"

        self.vocab: Optional[dict] = None
        self.global_mean: Optional[float] = None
        self.global_std: Optional[float] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    @property
    def gene_vocab_size(self) -> int:
        if self.vocab is None:
            raise RuntimeError(
                "vocab is not loaded. Call setup() before accessing gene_vocab_size."
            )
        return len(self.vocab)

    def prepare_data(self):
        """Compute and cache global stats (rank 0 only)."""
        if self.stats_path.exists():
            log.info(f"Global stats already cached at {self.stats_path}")
            return

        log.info("Computing global stats via Welford's algorithm...")
        dataset = MemmapCellDataset(self.data_dir)
        n_cells = len(dataset)

        # Sample up to 50k cells for stats computation
        max_samples = min(n_cells, 50_000)
        rng = np.random.RandomState(self.seed)
        sample_indices = rng.choice(n_cells, size=max_samples, replace=False)

        # Welford's online algorithm
        n = 0
        mean = 0.0
        M2 = 0.0

        for i, idx in enumerate(sample_indices):
            if (i + 1) % 10_000 == 0:
                log.info(f"  Stats progress: {i + 1}/{max_samples} cells")

            cell = dataset[int(idx)]
            values = cell["values"]  # already log1p-transformed float32

            new_count = values.size
            if new_count == 0:
                continue

            batch_mean = float(np.mean(values))
            n_old = float(n)
            n_new = n_old + new_count

            delta = batch_mean - mean
            mean += delta * (new_count / n_new)
            M2 += float(np.sum((values - batch_mean) ** 2)) + (
                delta**2
            ) * (n_old * new_count / n_new)
            n += new_count

        if n < 2:
            raise RuntimeError("Not enough data points to compute statistics.")

        variance = M2 / (n - 1)
        std = float(np.sqrt(variance))

        log.info(
            f"Global stats from {n} values ({max_samples} cells): "
            f"mean={mean:.4f}, std={std:.4f}"
        )

        with open(self.stats_path, "w") as f:
            json.dump({"mean": float(mean), "std": std}, f)
        log.info(f"Saved global stats to {self.stats_path}")

    def setup(self, stage: Optional[str] = None):
        """Set up datasets, vocab, and global stats on every rank."""
        # Load vocab
        with open(self._dataset_dir.vocab_path, "r") as f:
            self.vocab = json.load(f)

        # Load global stats
        self._load_stats()

        # Create dataset and split
        min_genes = self.fixed_gene_count if self.encoder_type == "transformer" else 0
        full_dataset = MemmapCellDataset(self.data_dir, min_genes=min_genes)
        n = len(full_dataset)
        n_val = max(1, int(n * self.val_fraction))
        n_train = n - n_val

        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [n_train, n_val], generator=generator
        )

        # Optionally subsample for quick trial runs
        if self.subset_fraction < 1.0:
            n_train_keep = max(1, int(len(self.train_dataset) * self.subset_fraction))
            n_val_keep = max(1, int(len(self.val_dataset) * self.subset_fraction))
            self.train_dataset, _ = random_split(
                self.train_dataset,
                [n_train_keep, len(self.train_dataset) - n_train_keep],
                generator=torch.Generator().manual_seed(self.seed),
            )
            self.val_dataset, _ = random_split(
                self.val_dataset,
                [n_val_keep, len(self.val_dataset) - n_val_keep],
                generator=torch.Generator().manual_seed(self.seed),
            )
            log.info(
                f"Subset to {self.subset_fraction:.0%}: "
                f"{n_train_keep} train, {n_val_keep} val"
            )

        log.info(
            f"Dataset split: {len(self.train_dataset)} train, {len(self.val_dataset)} val "
            f"(total {n}, vocab size {len(self.vocab)})"
        )

    def _load_stats(self):
        """Load global normalization statistics."""
        is_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
        is_rank_zero = not is_ddp or torch.distributed.get_rank() == 0

        if not self.stats_path.exists():
            if is_ddp and not is_rank_zero:
                log.info("Waiting for rank 0 to compute stats...")
                torch.distributed.barrier()
            else:
                self.prepare_data()
                if is_ddp:
                    torch.distributed.barrier()

        with open(self.stats_path, "r") as f:
            stats = json.load(f)
        self.global_mean = float(stats["mean"])
        self.global_std = float(stats["std"])

        if not is_ddp or is_rank_zero:
            log.info(
                f"Loaded global stats: mean={self.global_mean:.4f}, "
                f"std={self.global_std:.4f}"
            )

    def _collate_fn(self, batch: List[Dict]) -> Dict:
        if not batch:
            return {
                "indices": torch.empty(0, dtype=torch.long),
                "values": torch.empty(0, dtype=torch.float),
                "offsets": torch.tensor([0], dtype=torch.long),
                "metadata": [],
            }

        indices = torch.cat([torch.from_numpy(s["gene_indices"]) for s in batch])
        values = torch.cat([torch.from_numpy(s["values"]) for s in batch]).float()

        # Standardize (data is already log1p-transformed in the memmap)
        values = (values - self.global_mean) / (self.global_std + 1e-6)

        offsets = torch.tensor(
            [0] + [len(s["gene_indices"]) for s in batch], dtype=torch.long
        ).cumsum(0)

        return {
            "indices": indices,
            "values": values,
            "offsets": offsets,
            "metadata": [{} for _ in batch],
        }

    def _collate_fn_transformer(self, batch: List[Dict]) -> Dict:
        """Collate for transformer encoder: produces dense [B, N] tensors.

        Filters cells with fewer than fixed_gene_count non-zero genes,
        randomly selects exactly fixed_gene_count genes per cell, and
        shuffles them so the model's column-based context/target split is random.
        """
        N = self.fixed_gene_count

        all_indices = []
        all_values = []

        for sample in batch:
            gene_ids = sample["gene_indices"]
            vals = sample["values"]
            n_genes = len(gene_ids)

            if n_genes < N:
                log.warning(
                    f"Cell with {n_genes} genes < {N} in collate (should not happen). Skipping."
                )
                continue

            # Randomly select N genes and shuffle
            perm = np.random.permutation(n_genes)[:N]
            selected_ids = gene_ids[perm]
            selected_vals = vals[perm]

            all_indices.append(torch.from_numpy(selected_ids.copy()))
            all_values.append(torch.from_numpy(selected_vals.copy()).float())

        if not all_indices:
            return {
                "indices": torch.empty(0, N, dtype=torch.long),
                "values": torch.empty(0, N, dtype=torch.float),
                "metadata": [],
            }

        indices = torch.stack(all_indices)  # [B, N]
        values = torch.stack(all_values)    # [B, N]

        # z-score standardization
        values = (values - self.global_mean) / (self.global_std + 1e-6)

        return {
            "indices": indices,
            "values": values,
            "metadata": [{} for _ in valid_indices],
        }

    def _get_collate_fn(self):
        if self.encoder_type == "transformer":
            return self._collate_fn_transformer
        return self._collate_fn

    def train_dataloader(self) -> DataLoader:
        sampler = None
        shuffle = True
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = DistributedSampler(self.train_dataset, shuffle=True)
            shuffle = False

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=self._get_collate_fn(),
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        sampler = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = DistributedSampler(self.val_dataset, shuffle=False)

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            collate_fn=self._get_collate_fn(),
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )
