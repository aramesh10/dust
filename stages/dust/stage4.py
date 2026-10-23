"""Optimization 4 on stage 3: the head and cross-entropy run LOSS_CHUNK tokens at a time, so no pass holds the
full float32 logits. Each token's loss depends only on its own logits.
"""
import torch

from stages.dust import stage3

LOSS_CHUNK = 2048


class DUST(stage3.DUST):
    def token_losses(self, rows, targets):
        return torch.cat([
            super(DUST, self).token_losses(r, t) for r, t in zip(rows.split(LOSS_CHUNK), targets.split(LOSS_CHUNK))
        ])

    def slab_sums(self, rows, targets):
        chunks = [super(DUST, self).slab_sums(r, t) for r, t in zip(rows.split(LOSS_CHUNK), targets.split(LOSS_CHUNK))]
        return torch.cat([s for s, _ in chunks]), torch.cat([t for _, t in chunks])
