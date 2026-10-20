"""Time training steps for DUST or the backpropagation baseline, without evaluation or checkpointing.

    python benchmark.py --method backprop --tokens 1M --steps 5
    python benchmark.py --method dust --population 4096

Runs on one GPU. Each step is synchronized before and after it is timed.
Warmup steps are recorded but left out of the summary.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch

from baselines.backprop import DEFAULT_CONFIGS as BACKPROP_CONFIGS, make_optimizer as backprop_optimizer, train_step
from dust import DRAW_CHUNK_SIZES, DUST, GPTConfig, load_sequences, make_model, make_optimizer, recipe


GIB = 1024 ** 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['dust', 'backprop'], default='dust')
    parser.add_argument('--tokens', choices=['1M', '10M'], default='1M')
    parser.add_argument('--population', type=int, choices=DRAW_CHUNK_SIZES, default=256)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--steps', type=int, default=5, help='Measured steps.')
    parser.add_argument('--warmup', type=int, default=1, help='Steps run first and excluded from the summary.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, help='Write the per-step records and summary as JSON.')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error('Benchmarking requires a CUDA GPU.')
    if int(os.environ.get('WORLD_SIZE', 1)) != 1:
        parser.error('The benchmark uses one GPU; launch it with python, not torchrun.')
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')

    config = GPTConfig()
    batches = load_sequences(args.data_dir / f'train_{args.tokens}.pt', (args.warmup + args.steps) * 8)
    model = make_model(config, device, args.seed)
    if args.method == 'backprop':
        defaults = BACKPROP_CONFIGS[args.tokens]
        model.requires_grad_(True)
        optimizer = backprop_optimizer(model, defaults['lr'], defaults['momentum'])
        step_fn = lambda x, y: train_step(model, optimizer, x, y)
    else:
        optimizer = make_optimizer(model, recipe(args.tokens, args.population))
        estimator = DUST(model, recipe(args.tokens, args.population), args.seed)
        step_fn = lambda x, y: estimator.step(x, y, optimizer)
    torch.cuda.synchronize()
    setup_allocated = torch.cuda.memory_allocated() / GIB

    records = []
    for step in range(1, args.warmup + args.steps + 1):
        batch = batches[(step - 1) * 8:step * 8].to(device)
        x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss = step_fn(x, y)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        free, total = torch.cuda.mem_get_info()
        record = dict(
            step=step,
            warmup=step <= args.warmup,
            train_loss=loss,
            step_seconds=seconds,
            tokens_per_second=y.numel() / seconds,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / GIB,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / GIB,
            device_used_gib=(total - free) / GIB,
        )
        records.append(record)
        print(json.dumps(record), flush=True)

    if args.method == 'dust':
        estimator.close()
    measured = [r for r in records if not r['warmup']]
    times = [r['step_seconds'] for r in measured]
    summary = dict(
        method=args.method,
        tokens=args.tokens,
        population=args.population if args.method == 'dust' else None,
        gpu_name=torch.cuda.get_device_name(device),
        gpu_total_gib=torch.cuda.get_device_properties(device).total_memory / GIB,
        torch_version=torch.__version__,
        measured_steps=len(measured),
        warmup_steps=args.warmup,
        setup_allocated_gib=setup_allocated,
        mean_step_seconds=statistics.mean(times),
        median_step_seconds=statistics.median(times),
        min_step_seconds=min(times),
        max_step_seconds=max(times),
        mean_tokens_per_second=statistics.mean(r['tokens_per_second'] for r in measured),
        peak_allocated_gib=max(r['peak_allocated_gib'] for r in measured),
        peak_reserved_gib=max(r['peak_reserved_gib'] for r in measured),
    )
    print(json.dumps(summary, indent=2), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(summary=summary, steps=records), indent=2) + '\n')


if __name__ == '__main__':
    main()
