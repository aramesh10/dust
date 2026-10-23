"""dust_noise.py with memory optimizations. Same estimators and update; only what is kept in memory changes.

  1. Routed keys by index (Cut). The own-query response moves each query's 256 strongest key logits. dust_noise.py
     copied those keys and values out for every query, 1 GiB per layer for one sequence, all of it alive through the
     sequence's replay. Here each query keeps only the indices of its keys, and the routed kernel reads them from the
     layer's clean keys and values.
  2. MLP units gathered per layer (Cut). The replay restricts each MLP to its token's top-512 units. dust_noise.py
     gathered those units' weights for every block of tokens and every layer at once (4 GiB); here each layer's are
     gathered inside the replay when it is entered, and dropped when it is left.
  3. Smaller token blocks in the replay (Cut): 256 tokens instead of 512, so its moved streams and its readout logits
     are half the size.
  4. Attention maps one sequence at a time, never cached. dust_noise.py built every layer's map for the whole batch
     during the query probes and kept them until the end of the step (2 GiB), although only the value credit reads
     them. Here the value credit builds each sequence's map when it needs it.
  5. Fewer embedding moves at once (Boost): 4 instead of 16, so the first attention stage's readout never holds the
     full float32 logits of 16 moved copies of the batch.
  6. Captured layer inputs keep their dtype. dust_noise.py stored a float32 copy of every bfloat16 layer input (the
     inputs of both c_proj); here they are stored as they are and widened when read.
  7. Query probe draws signed in place. Each batch of 64 draws is gathered Hadamard rows times two sign patterns;
     multiplying in place keeps one copy of the batch instead of three.

Optimization 3 changes the shapes of the replay's bfloat16 matrix products, so its estimates differ from
dust_noise.py's in rounding (about 1e-5 relative); with chain_tokens=512 they match it exactly (check_noise.py).

stages/noise/ adds these one at a time: stages/noise/stageN.py has optimizations 1 to N, and this file is stage 7.

    python benchmark.py --method noise-cut-opt
"""
import torch

from dust_noise import hadamard
from stages.noise import stage6

Settings = stage6.Settings


class SignedCapture(stage6.LeanCapture):
    def orthogonal_noise(self, site, draws, chunk, n, b, t, d, dtype=None):
        """Optimization 7: Capture.orthogonal_noise, signing the gathered rows in place."""
        if site not in self.frames:
            cycles = -(-self.world * draws // d)
            self.frames[site] = (self.signs(cycles, d, shared=True),
                                 torch.randint(0, d, (b, t), device=self.device, generator=self.shared_generator))
        patterns, offset = self.frames[site]
        index = self.rank * draws + chunk * n + torch.arange(n, device=self.device)
        rows = (offset.unsqueeze(0) + index[:, None, None]) % d
        a = hadamard(d, self.device, self.dtype)[rows]
        a.mul_(patterns[index // d][:, None, None, :]).mul_(self.signs(n, b, t, 1))
        return a if dtype in (None, self.dtype) else a.to(dtype)


class Boost(stage6.Boost):
    capture_class = SignedCapture


class Cut(Boost, stage6.Cut):
    pass
