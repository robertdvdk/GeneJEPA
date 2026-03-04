# GeneJEPA Architecture Discussion Summary

## Current Architecture

- **Disjoint masking**: Student sees ~55% of genes (context), teacher sees ~45% (target). They never see the same genes.
- **Predictor**: 3-layer MLP with 4× expansion (wide, not a bottleneck). Maps student context embedding → predicted target embedding.
- **Loss**: Cosine similarity (weight 1.0) + VICReg regularization — variance (weight 25.0) + covariance (weight 1.0) on predictions, plus extra variance (weight 20.0) + covariance (weight 1.0) on student context.
- **Teacher**: EMA of student encoder (cosine warmup 0.992 → 0.9995).
- **Inference**: Teacher encodes full (unmasked) input via `get_embedding()`.

## Problems Observed

1. **Sim loss increasing**: VICReg regularization outweighs prediction 46:1. Variance/covariance terms push embeddings to spread out, actively fighting the similarity objective. Predictions diverge from targets over training.

2. **Gradient imbalance**: Value embedding gradients ~25× larger than identity gradients (ratio 0.03–0.05). Identity embeddings barely updating.

3. **Predictor not bottlenecked**: 4× expansion lets it learn shortcuts. `avg_cosine_pred` rising while `avg_cosine_student_ctx` stays low (0.06) — predictor converging while encoder stays healthy.

4. **Dead code**: Stability gate buffers registered but never used. `_debug_cache` never initialized so `teacher_avg_cos` never logged.

5. **Noisy targets from disjoint masking**: Teacher sees only ~45% of genes. Different masks on the same cell → different teacher embeddings. Student chases a moving target.

## Literature Context

- **I-JEPA** (Assran et al., 2023): Teacher sees full input, student sees partial. Collapse prevention via **narrow predictor bottleneck + EMA**. No VICReg.
- **DINO** (Caron et al., 2021): Different augmented views of full input. Collapse prevention via **centering + sharpening**. Less suitable for scRNA-seq — no meaningful augmentations exist for gene expression.
- **Key insight**: VICReg, DINO centering, and I-JEPA bottleneck are three separate collapse prevention strategies. GeneJEPA currently mixes I-JEPA architecture with VICReg — an unusual combination where the two fight each other.

## Why JEPA suits scRNA-seq

- Masking is natural: "given a subset of expressed genes, infer the cell state."
- No need for artificial augmentations (unlike DINO).
- Note: JEPA predicts in **latent space**, not gene expression. It learns that gene subsets describe the same cell state — useful for robustness to dropout/panel differences. It does NOT directly learn gene regulatory relationships (that would require reconstruction).

## Misc Notes

- Flash Attention 3 transformer vs perceiver: For scRNA-seq with 1000–5000 genes/cell and 512 latent tokens, perceiver is ~3–10× faster. FA3 closes the gap but can't overcome the N² vs N·L FLOP difference at these sequence lengths.
- Batch size affects VICReg: variance/covariance are per-batch statistics. Small batches → noisy estimates → variance penalty fires harder.

---

## Things to Try

### 1. Teacher sees full input

**What**: Teacher encodes all genes (unmasked), student encodes a random subset. Student's predictor tries to match the teacher's full-input embedding.

**Why**: Currently the teacher's target is noisy — same cell with different masks produces different teacher embeddings. With full-input teacher, every cell has one stable target regardless of which genes the student sees. This is how I-JEPA works.

**Changes needed**:
- `GenePerceiverJEPA.forward()`: Teacher gets original `(indices, values, offsets)` instead of `target_inputs`.
- `_random_block_masking()`: Simplify — only need to select student's context subset, no separate target set.
- `pred_to_sample_idx` becomes 1:1 (one student embedding → one teacher embedding per sample).
- `get_embedding()`: No change needed — already uses full input.

**Context fraction options**:
- ~55% (current). Safest, isolates the teacher change.
- ~30%. Moderate. ~300 genes for a 1000-gene cell.
- ~10%. Very aggressive. Could curriculum-learn: start at 50%, anneal down to 10%.

**Risk**: Low. Well-understood change aligned with I-JEPA literature. Inference path unchanged.

---

### 2. Transformer instead of Perceiver

**What**: Replace `GenePerceiverEncoder` (cross-attention to fixed latents → self-attention on latents) with a standard transformer encoder (self-attention directly on gene tokens, CLS token for pooling).

**Why**: The perceiver bottlenecks all information through 512 latent tokens via a single cross-attention step. A transformer lets all gene tokens attend to each other directly, preserving more fine-grained gene–gene interactions.

**Trade-offs**:

| | Perceiver (current) | Transformer |
|---|---|---|
| **Complexity** | O(N·L + L²·depth) | O(N²·depth) |
| **Speed at N=3000** | ~3–10× faster | Slower, but FA3 on H100 helps |
| **Information flow** | Bottlenecked through L=512 latents | All tokens attend to all tokens |
| **Memory** | Low (self-attention on 512 tokens) | High (N² attention, need gradient checkpointing) |

**Changes needed**:
- New `GeneTransformerEncoder` class with self-attention + CLS token pooling + Flash Attention 3.
- `ModelConfig`: add `encoder_type: str = "perceiver"` vs `"transformer"`.
- May need to reduce `num_blocks` (24 is a lot for N² attention).

**When to try**: After fixing masking/collapse issues. The perceiver isn't the root cause of training failure.

**Risk**: Medium. Higher memory, slower training. Worth trying once the basic JEPA loop works.

---

### 3. Collapse prevention

Three approaches to fix the 46:1 regularization-vs-prediction imbalance:

#### Option A: Drop VICReg, narrow predictor (I-JEPA style)

Remove all var/cov terms. Shrink predictor to bottleneck (`expansion_factor=0.5`, `depth=2`). Loss becomes just cosine similarity. This is what I-JEPA does — the narrow predictor can't learn trivial mappings.

**Risk**: If bottleneck isn't tight enough, collapse has no safety net. Monitor `avg_cosine_student_ctx`.

#### Option B: Rebalance VICReg coefficients

Keep VICReg but flip weights: `sim_coeff: 25.0`, `var_coeff: 1.0`, reduce `var_s` from 20.0 to ~1.0. Prediction weight 25 vs regularization ~4.

**Risk**: Low, easy to tune. But doesn't address fundamental VICReg-vs-JEPA tension.

#### Option C: Hybrid — narrow predictor + light variance penalty

Narrow predictor (`expansion_factor=0.5`, `depth=2`) + small variance term (`var_coeff=1.0`, `cov_coeff=0.0`). Belt and suspenders.

**Risk**: Lowest. Gets benefits of both.

#### Recommendation
Try **Option A** first (pure I-JEPA). Fall back to **Option C** if collapse appears.

---

### 4. Cell QC filtering in data loader

**What**: Filter out cells with too few expressed genes at data loading time.

**Why**: Cells with <528 genes are silently dropped in `forward()` — wastes compute (tokenized, batched, sent to GPU, then discarded).

**Where**: `MemmapDataModule` in `data_memmap.py`, filter during dataset construction.

**Threshold options**:
- **200 genes**: Permissive. Removes empty droplets/debris. Standard scRNA-seq QC.
- **500 genes**: Matches current effective minimum. Avoids wasted compute.
- **Configurable**: `--min-genes 200` CLI flag.

**Additional QC to consider**:
- Max genes filter (>10k may be doublets)
- Min cells per gene filter (<0.1% of cells = noise)

**Risk**: Very low. No change to model behavior.
