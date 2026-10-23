"""dust.py with memory optimizations. Same estimator and update; only what is kept in memory changes.

  1. Every estimator runs its draws one at a time instead of nrep together, so no pass holds nrep copies of
     the batch, its logits or its attention.
  2. The clean pass keeps only the residual stream entering each block (its checkpoint) and the stream leaving
     the last one, instead of every layer's input and output. After a group of draws, its block is recomputed
     from the checkpoint keeping only that group's layers, which are dropped once their gradients are taken.
     The attention internals are estimated one block at a time. The head recomputes its logits from the last
     stream and reassembles one slab at a time instead of keeping the logits and a full-vocabulary error.
     The same checkpoints let a draw that jitters a site in block L rerun the network from block L's
     checkpoint, since the blocks before it are unchanged.
  3. Direct draws keep a seed instead of their noise. The noise is regenerated from the seed in the forward
     hook and again when the centered rewards are accumulated, so no draw's noise outlives its use. The other
     estimators use each draw's noise as soon as it is drawn, so they have none to keep.
  4. The head and cross-entropy run LOSS_CHUNK tokens at a time, so no pass holds the full float32 logits.
  5. Each gradient is folded into its SGD momentum buffer as soon as it is estimated and then dropped. The
     weights all move together at the end of the step, so every draw sees the same weights.
  6. Each block's attention-output target is estimated in that block's pass and used right away by its
     attention internals, instead of all blocks' targets being kept until the end.

stages/dust/ adds these one at a time: stages/dust/stageN.py has optimizations 1 to N, and this file is stage 6.

    python dust_optimized.py --tokens 1M --population 256 --output runs/opt-1m-p256
"""
import torch
from torch import nn

import dust
from dust import average_ranks
from model import attention, apply_rotary_emb, has_ve, norm

LOSS_CHUNK = 2048  # Optimization 4: tokens per head and cross-entropy chunk


class DUST(dust.DUST):
    def __init__(self, model, settings, seed=42, rank=0, world=1):
        super().__init__(model, settings, seed, rank, world)
        self.capturing, self.checkpointing, self.streams = set(), False, []  # Optimization 2
        self.replay, self.seed, self.draw_id = torch.Generator(self.device), seed + 3 + 7777 * rank, 0  # Optimization 3
        self.parameters = dict(model.named_parameters())  # Optimization 5

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

    # Optimization 3: each (draw, site) gets its own seed, and its noise is rebuilt from it when needed
    def next_seed(self):
        self.draw_id += 1
        return (self.seed << 40) | self.draw_id

    def seeded_noise(self, seed, shape):
        self.replay.manual_seed(seed)
        return torch.randn(*shape, device=self.device, dtype=self.dtype, generator=self.replay)

    def regenerate(self, name, like):
        seed, sigma = self.jitter[name]
        return (sigma * self.seeded_noise(seed, like.shape)).to(like.dtype)

    def logits(self, rows):
        raw = self.model.lm_head(rows)[..., :self.model.config.vocab_size].float()
        return 15 * torch.tanh(raw / 15)

    def token_losses(self, rows, targets):
        """Optimization 4: cross-entropy LOSS_CHUNK tokens at a time."""
        return torch.cat([
            nn.functional.cross_entropy(self.logits(r), t, ignore_index=-1, reduction='none')
            for r, t in zip(rows.split(LOSS_CHUNK), targets.split(LOSS_CHUNK))
        ])

    def losses(self, x, y, start=None):
        """GPT.forward; with `start`, it resumes from the checkpoint entering block `start` (Optimization 2)."""
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
        """Optimization 2: keep only the stream entering each block and the stream leaving the last."""
        self.x, self.y = x, y
        self.batch, self.length = x.shape
        self.streams, self.checkpointing = [None] * (self.model.config.n_layer + 1), True
        try:
            self.clean_loss = self.losses(x, y)
        finally:
            self.checkpointing = False

    def capture_block(self, layer, sites):
        """Optimization 2: recompute one block from its checkpoint, keeping only the given sites' inputs and outputs.
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
        """One-sided loss drops, centered across each chunk of draws. Each draw has its own forward pass
        (Optimization 1), from block `start` (Optimization 2), or from the embedding if None."""
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
            # Optimization 3: regenerate each draw's noise once its centered reward is known
            for r, draw in zip(reward, seeds):
                for site, seed in draw.items():
                    a = self.seeded_noise(seed, (self.batch, self.length, width(site)))
                    errors[site].addcmul_(r.unsqueeze(-1), a, value=-1 / sigma)  # errors -= r·a/σ

        return {s: error / draws for s, error in errors.items()}

    def slab_sums(self, rows, targets):
        """Optimization 4: the clean logits' per-slab exp sums and target logits, LOSS_CHUNK tokens at a time."""
        slabs = self.model.config.vocab_size // 256
        sums, logits_at = [], []
        for r, t in zip(rows.split(LOSS_CHUNK), targets.split(LOSS_CHUNK)):
            logits = self.logits(r)
            sums.append(logits.exp().view(-1, slabs, 256).sum(-1))
            logits_at.append(logits.gather(-1, t.unsqueeze(-1)).squeeze(-1))
        return torch.cat(sums), torch.cat(logits_at)

    def head_slabs(self, draws, sigma=0.05):
        """Perturb one vocabulary slab at a time; recompute CE from its changed logits.

        Each draw jitters one 256-column slab of the logits, and the loss is recomputed from the clean sums of
        the other slabs. Optimization 2: the logits are recomputed from the last stream, and each slab's rows of
        the gradient are yielded as soon as its draws are done.
        """
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
        """Score attention-output changes against estimated output errors (no derivatives).

        Each draw recomputes the block's attention output from the captured clean activations and scores the
        change per head against the output's estimated error (`target`, computed by direct_errors on the
        o@{layer} site). Queries keep their own token's score; the other kinds sum the scores of later
        tokens, decayed by gamma per token. Optimization 2: one block at a time, from capture_block(layer).
        """
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

    def fold(self, name, gradient, rows=slice(None)):
        """Optimization 5: average over GPUs and fold into the SGD momentum buffer; the weight moves in apply()."""
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
        """Optimization 5: turn each site's error into its weight gradient and fold it, then drop the errors.
        Emptying the dict frees them even though the caller still holds it, so none outlives its gradient."""
        for site, error in errors.items():
            self.fold(site + '.weight', self.reassemble(site, error))
        errors.clear()

    def apply(self):
        """Optimization 5: SGD's p -= lr * buffer, once every parameter has been folded this step."""
        missing = set(self.parameters) - self.folded
        if missing:
            raise KeyError(f'No estimate for {sorted(missing)}')
        for group in self.optimizer.param_groups:
            for parameter in group['params']:
                parameter.add_(self.optimizer.state[parameter]['momentum_buffer'], alpha=-group['lr'])

    @torch.no_grad()
    def step(self, x, y, optimizer):
        """One clean pass, then block by block: direct-loss and local attention estimates. Then the embedding,
        head and mixing scalars, and an SGD update."""
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

            # Attention output. The deeper half moves the loss too little at 0.2 for a readable reward, so it
            # uses 0.4. The target is averaged over GPUs so that every GPU scores the attention internals
            # against the estimate from the whole population.
            # Optimization 6: this block's target is estimated here and used right away by its attention internals
            site, sigma = f'o@{layer}', 0.2 if layer < layers // 2 else 0.4
            target = average_ranks(self.direct_errors([site], r.hubs[layer] // world, sigma, layer)[site])
            ve = has_ve(layer, layers)
            internals = [prefix + 'attn.' + s for s in ('c_q', 'c_k', 'c_v', 've_gate')[:4 if ve else 3]]
            self.capture_block(layer, internals + ([f'value_embeds.{layer}'] if ve else []))
            for kind, count in r.local.items():
                if ve or kind not in ('ve', 'gate'):
                    self.fold_errors(self.local_errors(kind, count // world, layer, target))
            del target
            self.inputs, self.outputs = {}, {}

        site = 'transformer.wte'
        errors = self.direct_errors([site], r.embedding // world, 0.2)
        self.inputs[site] = self.x  # the embedding's input is the token ids
        self.fold_errors(errors)
        for rows, gradient in self.head_slabs(r.head // world):
            self.fold('lm_head.weight', gradient, rows)
        for name, gradient in self.scalar_gradients(8 // world).items():
            self.fold(name, gradient)

        optimizer.zero_grad(set_to_none=True)
        self.apply()
        return self.clean_loss[y != -1].mean().item()


if __name__ == '__main__':
    dust.DUST = DUST  # dust.main() builds dust.DUST
    dust.main()
