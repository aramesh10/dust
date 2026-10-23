"""Optimization 3 on stage 2: smaller token blocks in the replay (Cut), 256 tokens instead of 512, so its moved streams
and its readout logits are half the size. This changes the shapes of the replay's bfloat16 matrix products, so from
here on the estimates differ from dust_noise.py's in rounding (about 1e-5 relative).
"""
from dataclasses import dataclass

from stages.noise import stage2

Boost, Cut = stage2.Boost, stage2.Cut


@dataclass
class Settings(stage2.Settings):
    chain_tokens: int = 256
