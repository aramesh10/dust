"""Optimization 2 on stage 1: block checkpoints.

The clean pass keeps only the residual stream entering each block (its checkpoint) and the stream leaving the
last one, instead of every layer's input and output. After a group of draws, its block is recomputed from the
checkpoint keeping only that group's layers, which are dropped once their gradients are taken. The attention
internals are estimated one block at a time. The head recomputes its logits from the last stream and
reassembles one slab at a time instead of keeping the logits and a full-vocabulary error. The same checkpoints
let a draw that jitters a site in block L rerun the network from block L's checkpoint, since the blocks before
it are unchanged.
"""
import torch
from torch import nn

from dust import average_ranks
from model import attention, apply_rotary_emb, has_ve, norm
from stages.dust import stage1


class DUST(stage1.DUST):
    def __init__(self, model, settings, seed=42, rank=0, world=1):
        super().__init__(model, settings, seed, rank, world)
        self.capturing, self.checkpointing, self.streams = set(), False, []

    def output_hook(self, name):
        def hook(module, args, output):
            if name in self.capturing:
                self.inputs[name] = args[0].float() if args[0].is_floating_point() else args[0]
                self.outputs[name] = output
            if name in self.jitter:
                return output + self.jitter[name].to(output.dtype)

        return hook

    def logits(self, rows):
        raw = self.model.lm_head(rows)[..., :self.model.config.vocab_size].float()
        return 15 * torch.tanh(raw / 15)

    def token_losses(self, rows, targets):
        return nn.functional.cross_entropy(self.logits(rows), targets, ignore_index=-1, reduction='none')

    def losses(self, x, y, start=None):
        """GPT.forward; with `start`, it resumes from the checkpoint entering block `start`."""
        model = self.model
        with self.autocast():
            cos_sin = (model.cos[:, :x.shape[1]], model.sin[:, :x.shape[1]])
            if start is None:
                h = x0 = norm(model.transformer.wte(x))
            else:
                h, x0 = self.streams[start], self.x0  # already mixed with x0 for block `start`
            for i in range(start or 0, model.config.n_layer):
                if i != start:
                    h = model.resid_lambdas[i] * h + model.x0_lambdas[i] * x0
                if self.checkpointing:
                    self.streams[i] = h
                ve = model.value_embeds[str(i)](x) if str(i) in model.value_embeds else None
                h = model.transformer.h[i](h, ve, cos_sin, model.window_sizes[i])
            if self.checkpointing:
                self.x0, self.streams[model.config.n_layer] = x0, h
            losses = self.token_losses(norm(h).flatten(0, 1), y.flatten())
        return losses.view_as(y).float()

    def capture(self, x, y):
        """Keep only the stream entering each block and the stream leaving the last."""
        self.x, self.y = x, y
        self.batch, self.length = x.shape
        self.streams, self.checkpointing = [None] * (self.model.config.n_layer + 1), True
        try:
            self.clean_loss = self.losses(x, y)
        finally:
            self.checkpointing = False

    def capture_block(self, layer, sites):
        """Recompute one block from its checkpoint, keeping only the given sites' inputs and outputs.
        Called after the draws that need them, so no draw runs while they are held."""
        model = self.model
        self.inputs, self.outputs = {}, {}
        self.capturing = set(sites)
        try:
            with self.autocast():
                ve = model.value_embeds[str(layer)](self.x) if str(layer) in model.value_embeds else None
                cos_sin = (model.cos[:, :self.length], model.sin[:, :self.length])
                model.transformer.h[layer](self.streams[layer], ve, cos_sin, model.window_sizes[layer])
        finally:
            self.capturing = set()

    def width(self, site):
        """A site's output width, read from the model so that no output has to be captured for it."""
        if site.startswith('o@'):
            return self.model.config.n_embd
        module = self.modules[site]
        return module.embedding_dim if isinstance(module, nn.Embedding) else module.out_features

    def direct_errors(self, sites, draws, sigma, start=None):
        """Stage 1's direct_errors, with each draw rerun from block `start`, or from the embedding if None."""
        n, width = self.settings.nrep, self.width
        errors = {s: torch.zeros(self.batch, self.length, width(s), device=self.device) for s in sites}

        for _ in range(draws // n):
            noise = {s: self.noise(n, self.batch, self.length, width(s)) for s in sites}
            perturbed = []
            for draw in range(n):
                self.jitter = {s: sigma * a[draw] for s, a in noise.items()}
                try:
                    perturbed.append(self.losses(self.x, self.y, start))
                finally:
                    self.jitter = {}

            reward = self.clean_loss.unsqueeze(0) - torch.stack(perturbed)
            if n > 1:
                reward = reward - reward.mean(0, keepdim=True)
            for site, a in noise.items():
                errors[site] -= torch.einsum('nbt,nbtd->btd', reward.to(self.dtype), a).float() / sigma

        return {s: error / draws for s, error in errors.items()}

    def slab_sums(self, rows, targets):
        """The clean logits' per-slab exp sums and target logits."""
        logits = self.logits(rows)
        sums = logits.exp().view(-1, self.model.config.vocab_size // 256, 256).sum(-1)
        return sums, logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    def head_slabs(self, draws, sigma=0.05):
        """Stage 1's head_error, with the logits recomputed from the last stream, and each slab's rows of the
        gradient yielded as soon as its draws are done."""
        b, t = self.batch, self.length
        weight = self.model.lm_head.weight
        vocab, slab = self.model.config.vocab_size, 256
        slabs = vocab // slab
        target = self.y.clamp(min=0)
        with self.autocast():
            x = norm(self.streams[-1]).float()
            slab_sums, clean_target = self.slab_sums(x.flatten(0, 1), target.flatten())
        slab_sums, clean_target = slab_sums.view(b, t, slabs), clean_target.view(b, t)
        total = slab_sums.sum(-1)

        for block in range(slabs):
            lo = block * slab
            with self.autocast():
                raw = nn.functional.linear(x, weight[lo:lo + slab]).float()
            in_slab = (target >= lo) & (target < lo + slab)
            index = (target - lo).clamp(0, slab - 1).unsqueeze(-1)
            error = torch.zeros(b, t, slab, device=self.device)
            for _ in range(draws // slabs):
                a = self.noise(b, t, slab)
                changed = 15 * torch.tanh((raw + sigma * a.float()) / 15)
                partition = (total - slab_sums[..., block]) + changed.exp().sum(-1)
                target_logit = torch.where(in_slab, changed.gather(-1, index).squeeze(-1), clean_target)
                reward = torch.where(self.y != -1, self.clean_loss - (partition.log() - target_logit), 0)
                if b > 1:
                    reward = reward - reward.mean(0, keepdim=True)
                error.addcmul_(reward.unsqueeze(-1), a, value=-1 / sigma)  # error -= r·a/σ
            yield slice(lo, lo + slab), torch.einsum('btd,bti->di', error / (draws // slabs), x)
        if weight.shape[0] > vocab:
            yield slice(vocab, None), torch.zeros_like(weight[vocab:], dtype=torch.float32)

    def local_errors(self, kind, draws, layer, target, sigma=0.05):
        """Stage 1's local_errors for one block, from capture_block(layer)."""
        cfg = self.model.config
        b, t, h, d = self.batch, self.length, cfg.n_head, cfg.n_embd // cfg.n_head
        site_suffix = {'q': 'attn.c_q', 'k': 'attn.c_k', 'v': 'attn.c_v', 'gate': 'attn.ve_gate'}
        site = f'value_embeds.{layer}' if kind == 've' else f'transformer.h.{layer}.{site_suffix[kind]}'
        width = h if kind == 'gate' else cfg.n_embd
        error = torch.zeros(b, t, width, device=self.device)
        cos, sin = self.model.cos[:, :t].float(), self.model.sin[:, :t].float()

        def processed(raw):
            return norm(apply_rotary_emb(raw, cos, sin)).to(self.dtype)

        prefix = f'transformer.h.{layer}.attn.'
        q, k, v = [self.outputs[prefix + 'c_' + name].float().view(b, t, h, d) for name in ('q', 'k', 'v')]
        z = self.outputs[prefix + 've_gate'].float() if has_ve(layer, cfg.n_layer) else None
        ve = self.outputs[f'value_embeds.{layer}'].float().view(b, t, h, d) if z is not None else None
        values = v if z is None else v + (2 * torch.sigmoid(z)).unsqueeze(-1) * ve
        qp, kp = processed(q), processed(k)
        window = self.model.window_sizes[layer][0]
        baseline = attention(qp, kp, values.to(self.dtype), window)
        positions = torch.arange(t, device=self.device)
        lag = positions[None, :] - positions[:, None]  # [perturbation position, output position]
        # credit[t, s]: how much of output s's score goes to the jitter at t, inside the causal window
        credit = ((lag >= 0) & (lag <= window)).float() * self.settings.gamma ** lag.clamp(min=0)

        for _ in range(draws):
            a = self.noise(b, t, width, dtype=torch.float32)
            divisor = sigma
            if kind in ('q', 'k'):
                raw = q if kind == 'q' else k
                perturbed = processed(raw + sigma * a.view(b, t, h, d))
                qa, ka = (perturbed, kp) if kind == 'q' else (qp, perturbed)
                delta = attention(qa, ka, values.to(self.dtype), window) - baseline
            else:
                if kind == 'gate':
                    delta_gate = 2 * torch.sigmoid(z + sigma * a) - 2 * torch.sigmoid(z - sigma * a)
                    delta_value = delta_gate.unsqueeze(-1) * ve
                    divisor = 2 * sigma
                else:
                    delta_value = sigma * a.view(b, t, h, d)
                    if kind == 've':
                        delta_value = delta_value * (2 * torch.sigmoid(z)).unsqueeze(-1)
                delta = attention(qp, kp, delta_value.to(self.dtype), window)

            score = torch.einsum('bthd,bthd->bth', target.view(b, t, h, d).to(self.dtype), delta)
            if kind != 'q':
                score = torch.einsum('bsh,ts->bth', score.float(), credit)
            estimate = score.float().unsqueeze(-1) * a.view(b, t, h, -1)
            error += estimate.reshape(b, t, width) / divisor

        return {site: error / draws}

    @torch.no_grad()
    def step(self, x, y, optimizer):
        """dust.py's step, with each block's layers recomputed from its checkpoint after the draws that need them."""
        self.capture(x, y)
        r, world, layers = self.settings, self.world, self.model.config.n_layer
        gradients = {}
        for layer in range(layers):
            prefix = f'transformer.h.{layer}.'
            groups = [
                ([prefix + 'attn.c_proj', prefix + 'mlp.c_proj'], r.writers[layer]),
                ([prefix + 'mlp.c_fc'], r.hidden[layer]),
            ]
            for sites, count in groups:
                errors = self.direct_errors(sites, count // world, 0.2, layer)
                self.capture_block(layer, sites)
                for site, error in errors.items():
                    gradients[site + '.weight'] = self.reassemble(site, error)
                self.inputs, self.outputs = {}, {}

        site = 'transformer.wte'
        error = self.direct_errors([site], r.embedding // world, 0.2)[site]
        self.inputs[site] = self.x  # the embedding's input is the token ids
        gradients[site + '.weight'] = self.reassemble(site, error)

        targets = {}
        for layer in range(layers):
            site, sigma = f'o@{layer}', 0.2 if layer < layers // 2 else 0.4
            targets[layer] = average_ranks(self.direct_errors([site], r.hubs[layer] // world, sigma, layer)[site])

        gradient = torch.zeros_like(self.model.lm_head.weight, dtype=torch.float32)
        for rows, slab_gradient in self.head_slabs(r.head // world):
            gradient[rows] = slab_gradient
        gradients['lm_head.weight'] = gradient

        for layer in range(layers):
            prefix = f'transformer.h.{layer}.'
            ve = has_ve(layer, layers)
            internals = [prefix + 'attn.' + s for s in ('c_q', 'c_k', 'c_v', 've_gate')[:4 if ve else 3]]
            self.capture_block(layer, internals + ([f'value_embeds.{layer}'] if ve else []))
            for kind, count in r.local.items():
                if ve or kind not in ('ve', 'gate'):
                    for site, error in self.local_errors(kind, count // world, layer, targets[layer]).items():
                        gradients[site + '.weight'] = self.reassemble(site, error)
            self.inputs, self.outputs = {}, {}
        gradients.update(self.scalar_gradients(8 // world))

        # Dividing the summed estimates by the token count gives the gradient of the mean loss.
        valid_tokens = (y != -1).sum()
        optimizer.zero_grad(set_to_none=True)
        for name, parameter in self.model.named_parameters():
            gradient = average_ranks((gradients[name] / valid_tokens).to(parameter.dtype))
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f'Non-finite estimate for {name}')
            parameter.grad = gradient
        optimizer.step()
        return self.clean_loss[y != -1].mean().item()
