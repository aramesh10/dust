"""Optimization 5 on stage 4: fewer embedding moves at once (Boost), 4 instead of 16, so the first attention stage's
readout never holds the full float32 logits of 16 moved copies of the batch.
"""
from dataclasses import dataclass

from stages.noise import stage4

Boost, Cut = stage4.Boost, stage4.Cut


@dataclass
class Settings(stage4.Settings):
    emb_chunk: int = 4
