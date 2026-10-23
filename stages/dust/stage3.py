"""Optimization 3 on stage 2: direct draws keep a seed instead of their noise.

The noise is regenerated from the seed in the forward hook and again when the centered rewards are
accumulated, so no draw's noise outlives its use. The other estimators use each draw's noise as soon as it is
drawn, so they have none to keep.
"""
import torch

from stages.dust import stage2


class DUST(stage2.DUST):
    def __init__(self, model, settings, seed=42, rank=0, world=1):
        super().__init__(model, settings, seed, rank, world)
        self.replay, self.seed, self.draw_id = torch.Generator(self.device), seed + 3 + 7777 * rank, 0

    def output_hook(self, name):
        def hook(module, args, output):
            if name in self.capturing:
                self.inputs[name] = args[0].float() if args[0].is_floating_point() else args[0]
                self.outputs[name] = output
            if name in self.jitter:
                return output + self.regenerate(name, output)

        return hook

    def input_hook(self, name):
        def hook(module, args):
            if name in self.jitter:
                return (args[0] + self.regenerate(name, args[0]),)

        return hook

    def next_seed(self):
        self.draw_id += 1
        return (self.seed << 40) | self.draw_id

    def seeded_noise(self, seed, shape):
        self.replay.manual_seed(seed)
        return torch.randn(*shape, device=self.device, dtype=self.dtype, generator=self.replay)

    def regenerate(self, name, like):
        seed, sigma = self.jitter[name]
        return (sigma * self.seeded_noise(seed, like.shape)).to(like.dtype)

    def direct_errors(self, sites, draws, sigma, start=None):
        """Stage 2's direct_errors, with a seed per (draw, site) in place of its noise."""
        n, width = self.settings.nrep, self.width
        errors = {s: torch.zeros(self.batch, self.length, width(s), device=self.device) for s in sites}

        for _ in range(draws // n):
            seeds = [{s: self.next_seed() for s in sites} for _ in range(n)]
            perturbed = []
            for draw in seeds:
                self.jitter = {s: (seed, sigma) for s, seed in draw.items()}
                try:
                    perturbed.append(self.losses(self.x, self.y, start))
                finally:
                    self.jitter = {}

            reward = self.clean_loss.unsqueeze(0) - torch.stack(perturbed)
            if n > 1:
                reward = reward - reward.mean(0, keepdim=True)
            # regenerate each draw's noise once its centered reward is known
            for r, draw in zip(reward, seeds):
                for site, seed in draw.items():
                    a = self.seeded_noise(seed, (self.batch, self.length, width(site)))
                    errors[site].addcmul_(r.unsqueeze(-1), a, value=-1 / sigma)  # errors -= r·a/σ

        return {s: error / draws for s, error in errors.items()}
