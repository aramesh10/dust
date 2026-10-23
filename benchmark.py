"""Time training steps for DUST or the backpropagation baseline, without evaluation or checkpointing.

    python benchmark.py --method backprop --tokens 1M --steps 5
    python benchmark.py --method dust --population 4096
    python benchmark.py --method optimized --population 4096
    python benchmark.py --method stage2 --population 4096
    python benchmark.py --method noise-cut
    python benchmark.py --method noise-cut-stage3
    python benchmark.py --method noise-cut-opt --eager
    python benchmark.py --method optimized --population 4096 --n-layer 16 --n-embd 1024 --memory-only

The noise methods time dust_noise.py's two phases at its own batch of 4 sequences (the -stageN methods time
stages/noise/stageN.py's, the -opt methods dust_noise_optimized.py's); --population does not apply. --eager runs their compiled helpers eagerly; flex attention
stays compiled, since its eager fallback materializes every attention score.

--n-layer and --n-embd scale the model, at 64 dimensions per head; DUST's 8-block draw recipe is stretched to the
depth. --memory-only runs one chunk of draws per DUST site: the same shapes and peak memory, in far less time. A step
that runs out of memory ends the run, and the summary records it.

Runs on one GPU. Each step is synchronized before and after it is timed.
Warmup steps are recorded but left out of the summary.
"""
import argparse
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import statistics
import time

import torch

from baselines.backprop import DEFAULT_CONFIGS as BACKPROP_CONFIGS, make_optimizer as backprop_optimizer, train_step
from dust import DRAW_CHUNK_SIZES, DUST, GPTConfig, load_sequences, make_model, make_optimizer, recipe
import dust_optimized
from stages.dust import stage1, stage2, stage3, stage4, stage5


GIB = 1024 ** 3
NOISE_MODULES = {'': 'dust_noise', **{f'-stage{n}': f'stages.noise.stage{n}' for n in range(1, 7)},
                 '-opt': 'dust_noise_optimized'}
NOISE_METHODS = [f'noise-{phase}{variant}' for phase in ('boost', 'cut') for variant in NOISE_MODULES]
ESTIMATORS = {
    'dust': DUST, 'stage1': stage1.DUST, 'stage2': stage2.DUST, 'stage3': stage3.DUST, 'stage4': stage4.DUST,
    'stage5': stage5.DUST, 'optimized': dust_optimized.DUST,
}


def depth_recipe(settings, layers, memory_only, vocab_size):
    """The 8-block recipe stretched to `layers` blocks (block i takes block i * 8 // layers's counts). memory_only:
    one chunk of nrep draws per site, and one per vocabulary slab for the head."""
    if memory_only:
        n = settings.nrep
        return replace(settings, writers=[n] * layers, hidden=[n] * layers, hubs=[n] * layers, embedding=n,
                       head=n * vocab_size // 256, local={kind: n for kind in settings.local})
    stretch = lambda counts: [counts[i * len(counts) // layers] for i in range(layers)]
    return replace(settings, writers=stretch(settings.writers), hidden=stretch(settings.hidden),
                   hubs=stretch(settings.hubs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=[*ESTIMATORS, 'backprop', *NOISE_METHODS], default='dust')
    parser.add_argument('--tokens', choices=['1M', '10M'], default='1M')
    parser.add_argument('--population', type=int, choices=DRAW_CHUNK_SIZES, default=256)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--steps', type=int, default=5, help='Measured steps.')
    parser.add_argument('--warmup', type=int, default=1, help='Steps run first and excluded from the summary.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grad-mode', choices=['no_grad', 'inference_mode'], default='no_grad')
    parser.add_argument('--eager', action='store_true', help='Noise methods: run the compiled helpers eagerly.')
    parser.add_argument('--n-layer', type=int, default=8)
    parser.add_argument('--n-embd', type=int, default=512)
    parser.add_argument('--memory-only', action='store_true', help='DUST methods: one chunk of draws per site.')
    parser.add_argument('--output', type=Path, help='Write the per-step records and summary as JSON.')
    parser.add_argument(
        '--memory-snapshot', action='store_true',
        help='Record CUDA allocations during the last step and save them next to --output as .pickle.',
    )
    args = parser.parse_args()
    if args.memory_snapshot and not args.output:
        parser.error('--memory-snapshot requires --output.')

    if not torch.cuda.is_available():
        parser.error('Benchmarking requires a CUDA GPU.')
    if int(os.environ.get('WORLD_SIZE', 1)) != 1:
        parser.error('The benchmark uses one GPU; launch it with python, not torchrun.')
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')

    heads = args.n_embd // 64
    config = GPTConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=heads, n_kv_head=heads)
    noise = args.method.startswith('noise')
    rows = 4 if noise else 8
    batches = load_sequences(args.data_dir / f'train_{args.tokens}.pt', (args.warmup + args.steps) * rows)
    if noise:
        compile = torch.compile
        if args.eager:  # the modules apply torch.compile when imported
            torch.compile = lambda fn=None, **options: fn if fn is not None else (lambda f: f)
        try:
            import dust_noise  # imported only here: it switches model.py's attention to flex attention
            phase, _, variant = args.method.removeprefix('noise-').partition('-')
            module = importlib.import_module(NOISE_MODULES[variant and '-' + variant])
        finally:
            torch.compile = compile
        if args.eager:
            from torch.nn.attention.flex_attention import flex_attention
            dust_noise.flex = torch.compile(flex_attention)
        settings = module.Settings(model=config)
        model = dust_noise.make_model(device, args.seed, settings.model, settings.projection_std, settings.head_std)
        optimizer = dust_noise.make_optimizer(model, settings, args.warmup + args.steps)
        estimator = (module.Boost if phase == 'boost' else module.Cut)(model, settings, args.seed, 0, 1)
        step_fn = lambda x, y: estimator.step(x, y, optimizer)
    elif args.method == 'backprop':
        model = make_model(config, device, args.seed)
        defaults = BACKPROP_CONFIGS[args.tokens]
        model.requires_grad_(True)
        optimizer = backprop_optimizer(model, defaults['lr'], defaults['momentum'])
        step_fn = lambda x, y: train_step(model, optimizer, x, y)
    else:
        model = make_model(config, device, args.seed)
        settings = depth_recipe(recipe(args.tokens, args.population), args.n_layer, args.memory_only, config.vocab_size)
        optimizer = make_optimizer(model, settings)
        estimator = ESTIMATORS[args.method](model, settings, args.seed)
        step_fn = lambda x, y: estimator.step(x, y, optimizer)
    torch.cuda.synchronize()
    setup_allocated = torch.cuda.memory_allocated() / GIB

    records, out_of_memory = [], None
    for step in range(1, args.warmup + args.steps + 1):
        batch = batches[(step - 1) * rows:step * rows].to(device)
        x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        snapshot = args.memory_snapshot and step == args.warmup + args.steps
        if snapshot:
            torch.cuda.memory._record_memory_history(max_entries=1_000_000, stacks='python')
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            with torch.inference_mode(args.grad_mode == 'inference_mode'):
                loss = step_fn(x, y)
        except torch.OutOfMemoryError:
            out_of_memory = dict(step=step, peak_allocated_gib=torch.cuda.max_memory_allocated() / GIB,
                                 peak_reserved_gib=torch.cuda.max_memory_reserved() / GIB)
            print(json.dumps(dict(out_of_memory=out_of_memory)), flush=True)
            break
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        if snapshot:
            torch.cuda.memory._dump_snapshot(str(args.output.with_suffix('.pickle')))
            torch.cuda.memory._record_memory_history(enabled=None)
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

    if args.method != 'backprop':
        estimator.close()
    measured = [r for r in records if not r['warmup']]
    times = [r['step_seconds'] for r in measured]
    peaks = [r for r in records + [out_of_memory] if r]
    summary = dict(
        method=args.method,
        grad_mode=args.grad_mode,
        eager=args.eager,
        tokens=args.tokens,
        population=args.population if args.method != 'backprop' and not noise else None,
        batch_sequences=rows,
        gpu_name=torch.cuda.get_device_name(device),
        gpu_total_gib=torch.cuda.get_device_properties(device).total_memory / GIB,
        torch_version=torch.__version__,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        parameters=sum(p.numel() for p in model.parameters()),
        memory_only=args.memory_only,
        out_of_memory=out_of_memory,
        measured_steps=len(measured),
        warmup_steps=args.warmup,
        setup_allocated_gib=setup_allocated,
        **(dict(
            mean_step_seconds=statistics.mean(times),
            median_step_seconds=statistics.median(times),
            min_step_seconds=min(times),
            max_step_seconds=max(times),
            mean_tokens_per_second=statistics.mean(r['tokens_per_second'] for r in measured),
        ) if times else {}),
        peak_allocated_gib=max(r['peak_allocated_gib'] for r in peaks),
        peak_reserved_gib=max(r['peak_reserved_gib'] for r in peaks),
    )
    print(json.dumps(summary, indent=2), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(summary=summary, steps=records), indent=2) + '\n')


if __name__ == '__main__':
    main()
