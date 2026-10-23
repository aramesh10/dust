"""Check that dust_noise_optimized.py (or a stage) estimates dust_noise.py's gradients, on one batch of 4 sequences.

Each phase is estimated three times from the same weights and seeds: twice by dust_noise.py, whose difference is the
run-to-run floor (the second recompiles; embedding scatters and kernel choices are not bitwise deterministic), and once by the optimized
estimators. Prints each parameter's relative difference from the first run.

    python check_noise.py
    python check_noise.py --module stages.noise.stage2
"""
import argparse
import importlib
from pathlib import Path

import torch

import dust
import dust_noise


def estimate(module, phase, batch, device, **overrides):
    settings = module.Settings(**overrides)
    model = dust_noise.make_model(device, 0, settings.model, settings.projection_std, settings.head_std)
    estimator = getattr(module, phase)(model, settings, 0, 0, 1)
    try:
        return estimator.estimate(batch[:, :-1].contiguous(), batch[:, 1:].contiguous())
    finally:
        estimator.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--module', default='dust_noise_optimized', help='The estimators to check.')
    parser.add_argument('--chain-tokens', type=int, help="Override the checked replay's token block.")
    args = parser.parse_args()
    device = torch.device('cuda', 0)
    torch.set_float32_matmul_precision('high')
    batch = dust.load_sequences(args.data_dir / 'train_1M.pt', 4).to(device)
    for phase in ('Boost', 'Cut'):
        reference = estimate(dust_noise, phase, batch, device)
        torch._dynamo.reset()  # the floor run recompiles, as the optimized one does
        again = estimate(dust_noise, phase, batch, device)
        overrides = {} if args.chain_tokens is None else dict(chain_tokens=args.chain_tokens)
        optimized = estimate(importlib.import_module(args.module), phase, batch, device, **overrides)
        difference = lambda g, name: ((g[name] - reference[name]).norm() / reference[name].norm().clamp_min(1e-30)).item()
        print(f'\n{phase}: relative difference from dust_noise.py      floor  optimized')
        for name in reference:
            print(f'  {name:42s} {difference(again, name):9.2e}  {difference(optimized, name):9.2e}')


if __name__ == '__main__':
    main()
