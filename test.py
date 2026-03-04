"""Test GeneTransformerJEPA with random tensors to check SDPA backend selection."""

import argparse
import logging
import os
import time

import torch

# Enable SDPA backend logging so PyTorch prints which kernel is selected
os.environ.setdefault("TORCH_LOGS", "+sdpa")

from genejepa.configs import ModelConfig
from genejepa.models import GeneTransformerJEPA


def main():
    parser = argparse.ArgumentParser(description="Test GeneTransformerJEPA forward pass")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=768)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--bench-iters", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Flash Attention available: {torch.backends.cuda.flash_sdp_enabled()}")
        print(f"Mem-efficient Attention available: {torch.backends.cuda.mem_efficient_sdp_enabled()}")

    # Configure model
    context_genes = args.seq_len // 2
    config = ModelConfig(
        encoder_type="transformer",
        transformer_fixed_gene_count=args.seq_len,
        transformer_context_gene_count=context_genes,
    )
    print(f"\nModelConfig: d={config.d}, layers={config.transformer_num_layers}, "
          f"heads={config.transformer_num_heads}, vocab={config.gene_vocab_size}")
    print(f"Sequence: {args.seq_len} total, {context_genes} context, "
          f"{args.seq_len - context_genes} target")

    model = GeneTransformerJEPA(config).to(device)
    model.train()
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {param_count:.1f}M")

    # Generate random inputs
    indices = torch.randint(0, config.gene_vocab_size, (args.batch_size, args.seq_len), device=device)
    values = torch.randn(args.batch_size, args.seq_len, device=device)

    print(f"\nInput shapes: indices={list(indices.shape)}, values={list(values.shape)}")
    print(f"Running {args.warmup_iters} warmup + {args.bench_iters} bench iterations...\n")

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    dtype_str = "bf16" if use_bf16 else "fp16" if device.type == "cuda" else "fp32"
    print(f"Autocast dtype: {dtype_str}")

    # Warmup
    for i in range(args.warmup_iters):
        with torch.autocast(device.type, dtype=torch.bfloat16 if use_bf16 else torch.float16, enabled=device.type == "cuda"):
            pred, target, student = model(indices, values)
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"  warmup {i+1}: pred={list(pred.shape)}, target={list(target.shape)}, student={list(student.shape)}")

    # Benchmark
    times = []
    for i in range(args.bench_iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.bfloat16 if use_bf16 else torch.float16, enabled=device.type == "cuda"):
            pred, target, student = model(indices, values)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  iter {i+1}: {dt*1000:.1f} ms")

    avg = sum(times) / len(times)
    print(f"\nAverage forward time: {avg*1000:.1f} ms  ({args.batch_size / avg:.0f} samples/s)")

    if device.type == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
