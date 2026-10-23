"""Optimization 4 on stage 3: attention maps one sequence at a time, never cached. dust_noise.py built every layer's map
for the whole batch during the query probes and kept them until the end of the step (2 GiB), although only the value
credit reads them. Here the value credit builds each sequence's map when it needs it.
"""
import math

import torch

from model import apply_rotary_emb, norm
from stages.noise import stage3

Settings = stage3.Settings


class Boost(stage3.Boost):
    def query_errors(self, draws, targets, layers=None):
        """The parent builds every layer's attention map here only to cache it."""
        self.attention_maps = lambda i: None
        try:
            return super().query_errors(draws, targets, layers)
        finally:
            del self.attention_maps

    def sequence_map(self, i, r):
        """dust_noise.Boost.attention_maps for sequence r alone, A[1, h, query, source]."""
        d, model = self.capture, self.model
        cfg = model.config
        t, h, hd = d.length, cfg.n_head, cfg.n_embd // cfg.n_head
        cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
        prefix = f'transformer.h.{i}.attn.'
        q, k = (norm(apply_rotary_emb(d.outputs[prefix + name][r:r + 1].float().view(1, t, h, hd), cos, sin))
                for name in ('c_q', 'c_k'))
        logits = q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2) / math.sqrt(hd)
        positions = torch.arange(t, device=self.device)
        lag = positions[:, None] - positions[None, :]
        logits = logits.masked_fill(~((lag >= 0) & (lag <= model.window_sizes[i][0])), float('-inf'))
        return logits.softmax(-1).to(torch.bfloat16)

    def value_credit(self, i, target):
        d, cfg = self.capture, self.model.config
        b, t, h = d.batch, d.length, cfg.n_head
        hub = target.reshape(b, t, h, cfg.n_embd // h).transpose(1, 2).float()
        return torch.cat([self.sequence_map(i, r).float().transpose(-1, -2) @ hub[r:r + 1]
                          for r in range(b)]).transpose(1, 2)


class Cut(Boost, stage3.Cut):
    pass
