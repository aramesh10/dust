"""Optimization 1 on dust.py: every estimator runs its draws one at a time instead of nrep together, so no
pass holds nrep copies of the batch, its logits or its attention. The rewards of each nrep draws are still
centered together.
"""
import torch

import dust
from model import attention, apply_rotary_emb, has_ve, norm


class DUST(dust.DUST):
    def direct_errors(self, sites, draws, sigma):
        """One-sided loss drops, centered across each chunk of draws. Each draw has its own forward pass."""
        n = self.settings.nrep
        width = lambda s: self.model.config.n_embd if s.startswith('o@') else self.outputs[s].shape[-1]
        errors = {s: torch.zeros(self.batch, self.length, width(s), device=self.device) for s in sites}

        for _ in range(draws // n):
            noise = {s: self.noise(n, self.batch, self.length, width(s)) for s in sites}
            perturbed = []
            for draw in range(n):
                self.jitter = {s: sigma * a[draw] for s, a in noise.items()}
                try:
                    perturbed.append(self.losses(self.x, self.y))
                finally:
                    self.jitter = {}

            reward = self.clean_loss.unsqueeze(0) - torch.stack(perturbed)
            if n > 1:
                reward = reward - reward.mean(0, keepdim=True)
            for site, a in noise.items():
                errors[site] -= torch.einsum('nbt,nbtd->btd', reward.to(self.dtype), a).float() / sigma

        return {s: error / draws for s, error in errors.items()}

    def head_error(self, draws, sigma=0.05):
        """Perturb one vocabulary slab at a time; recompute CE from its changed logits, one draw at a time,
        visiting the slabs in the same order as dust.py."""
        n, b, t = self.settings.nrep, self.batch, self.length
        raw = self.outputs['lm_head']
        vocab, slab = self.model.config.vocab_size, 256
        slabs = vocab // slab
        logits = 15 * torch.tanh(raw[..., :vocab].float() / 15)
        slab_sums = logits.exp().view(b, t, slabs, slab).sum(-1)
        total = slab_sums.sum(-1)
        target = self.y.clamp(min=0)
        clean_target = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        error = torch.zeros_like(raw, dtype=torch.float32)

        for draw in range(draws):
            block = (draw // n) % slabs
            lo = block * slab
            a = self.noise(b, t, slab)
            changed = 15 * torch.tanh((raw[..., lo:lo + slab].float() + sigma * a.float()) / 15)
            partition = (total - slab_sums[..., block]) + changed.exp().sum(-1)
            in_slab = (target >= lo) & (target < lo + slab)
            index = (target - lo).clamp(0, slab - 1).unsqueeze(-1)
            target_logit = torch.where(in_slab, changed.gather(-1, index).squeeze(-1), clean_target)
            reward = torch.where(self.y != -1, self.clean_loss - (partition.log() - target_logit), 0)
            if b > 1:
                reward = reward - reward.mean(0, keepdim=True)
            error[..., lo:lo + slab].addcmul_(reward.unsqueeze(-1), a, value=-1 / sigma)  # error -= r·a/σ

        return error / (draws // slabs)

    def local_errors(self, kind, draws, targets, sigma=0.05):
        """dust.py's local_errors with one draw at a time."""
        cfg = self.model.config
        b, t, h, d = self.batch, self.length, cfg.n_head, cfg.n_embd // cfg.n_head
        site_suffix = {'q': 'attn.c_q', 'k': 'attn.c_k', 'v': 'attn.c_v', 'gate': 'attn.ve_gate'}
        layers = [i for i in range(cfg.n_layer) if kind not in ('ve', 'gate') or has_ve(i, cfg.n_layer)]
        sites = {
            i: f'value_embeds.{i}' if kind == 've' else f'transformer.h.{i}.{site_suffix[kind]}'
            for i in layers
        }
        width = h if kind == 'gate' else cfg.n_embd
        errors = {i: torch.zeros(b, t, width, device=self.device) for i in layers}
        cos, sin = self.model.cos[:, :t].float(), self.model.sin[:, :t].float()

        def processed(raw):
            return norm(apply_rotary_emb(raw, cos, sin)).to(self.dtype)

        clean = {}
        for i in layers:
            prefix = f'transformer.h.{i}.attn.'
            q, k, v = [self.outputs[prefix + 'c_' + name].float().view(b, t, h, d) for name in ('q', 'k', 'v')]
            z = self.outputs[prefix + 've_gate'].float() if has_ve(i, cfg.n_layer) else None
            ve = self.outputs[f'value_embeds.{i}'].float().view(b, t, h, d) if z is not None else None
            values = v if z is None else v + (2 * torch.sigmoid(z)).unsqueeze(-1) * ve
            qp, kp = processed(q), processed(k)
            window = self.model.window_sizes[i][0]
            baseline = attention(qp, kp, values.to(self.dtype), window)
            positions = torch.arange(t, device=self.device)
            lag = positions[None, :] - positions[:, None]  # [perturbation position, output position]
            # credit[t, s]: how much of output s's score goes to the jitter at t, inside the causal window
            credit = ((lag >= 0) & (lag <= window)).float() * self.settings.gamma ** lag.clamp(min=0)
            clean[i] = q, k, z, ve, qp, kp, values, baseline, credit

        for _ in range(draws):
            for i in layers:
                a = self.noise(b, t, width, dtype=torch.float32)
                q, k, z, ve, qp, kp, values, baseline, credit = clean[i]
                window = self.model.window_sizes[i][0]
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

                score = torch.einsum('bthd,bthd->bth', targets[i].view(b, t, h, d).to(self.dtype), delta)
                if kind != 'q':
                    score = torch.einsum('bsh,ts->bth', score.float(), credit)
                estimate = score.float().unsqueeze(-1) * a.view(b, t, h, -1)
                errors[i] += estimate.reshape(b, t, width) / divisor

        return {sites[i]: errors[i] / draws for i in layers}
