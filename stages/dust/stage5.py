"""Optimization 5 on stage 4: each gradient is folded into its SGD momentum buffer as soon as it is estimated and
then dropped, instead of every gradient being kept until optimizer.step(). The weights all move together at the
end of the step, so every draw sees the same weights. With plain momentum SGD (no dampening, Nesterov or weight
decay) this is exactly torch.optim.SGD.step().
"""
import torch

from dust import average_ranks
from model import has_ve
from stages.dust import stage4


class DUST(stage4.DUST):
    def __init__(self, model, settings, seed=42, rank=0, world=1):
        super().__init__(model, settings, seed, rank, world)
        self.parameters = dict(model.named_parameters())

    def fold(self, name, gradient, rows=slice(None)):
        """Average over GPUs and fold into the SGD momentum buffer; the weight moves in apply()."""
        parameter = self.parameters[name]
        # Dividing the summed estimates by the token count gives the gradient of the mean loss.
        gradient = average_ranks((gradient / self.valid_tokens).to(parameter.dtype))
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(f'Non-finite estimate for {name}')
        state = self.optimizer.state[parameter]
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(parameter)
        state['momentum_buffer'][rows].mul_(self.groups[parameter]['momentum']).add_(gradient)
        self.folded.add(name)

    def fold_errors(self, errors):
        """Turn each site's error into its weight gradient and fold it, then drop the errors.
        Emptying the dict frees them even though the caller still holds it, so none outlives its gradient."""
        for site, error in errors.items():
            self.fold(site + '.weight', self.reassemble(site, error))
        errors.clear()

    def apply(self):
        """SGD's p -= lr * buffer, once every parameter has been folded this step."""
        missing = set(self.parameters) - self.folded
        if missing:
            raise KeyError(f'No estimate for {sorted(missing)}')
        for group in self.optimizer.param_groups:
            for parameter in group['params']:
                parameter.add_(self.optimizer.state[parameter]['momentum_buffer'], alpha=-group['lr'])

    @torch.no_grad()
    def step(self, x, y, optimizer):
        """Stage 2's step, folding each gradient as soon as it is estimated."""
        self.capture(x, y)
        r, world, layers = self.settings, self.world, self.model.config.n_layer
        self.optimizer, self.folded, self.valid_tokens = optimizer, set(), (y != -1).sum()
        self.groups = {p: group for group in optimizer.param_groups for p in group['params']}
        for layer in range(layers):
            prefix = f'transformer.h.{layer}.'
            groups = [
                ([prefix + 'attn.c_proj', prefix + 'mlp.c_proj'], r.writers[layer]),
                ([prefix + 'mlp.c_fc'], r.hidden[layer]),
            ]
            for sites, count in groups:
                errors = self.direct_errors(sites, count // world, 0.2, layer)
                self.capture_block(layer, sites)
                self.fold_errors(errors)
                self.inputs, self.outputs = {}, {}

        site = 'transformer.wte'
        errors = self.direct_errors([site], r.embedding // world, 0.2)
        self.inputs[site] = self.x  # the embedding's input is the token ids
        self.fold_errors(errors)

        targets = {}
        for layer in range(layers):
            site, sigma = f'o@{layer}', 0.2 if layer < layers // 2 else 0.4
            targets[layer] = average_ranks(self.direct_errors([site], r.hubs[layer] // world, sigma, layer)[site])

        for rows, gradient in self.head_slabs(r.head // world):
            self.fold('lm_head.weight', gradient, rows)

        for layer in range(layers):
            prefix = f'transformer.h.{layer}.'
            ve = has_ve(layer, layers)
            internals = [prefix + 'attn.' + s for s in ('c_q', 'c_k', 'c_v', 've_gate')[:4 if ve else 3]]
            self.capture_block(layer, internals + ([f'value_embeds.{layer}'] if ve else []))
            for kind, count in r.local.items():
                if ve or kind not in ('ve', 'gate'):
                    self.fold_errors(self.local_errors(kind, count // world, layer, targets[layer]))
            self.inputs, self.outputs = {}, {}
        for name, gradient in self.scalar_gradients(8 // world).items():
            self.fold(name, gradient)

        optimizer.zero_grad(set_to_none=True)
        self.apply()
        return self.clean_loss[y != -1].mean().item()
