"""Optimization 1 on dust_noise.py: routed keys by index (Cut). The own-query response moves each query's 256 strongest
key logits. dust_noise.py copied those keys and values out for every query, 1 GiB per layer for one sequence, all of it
alive through the sequence's replay. Here each query keeps only the indices of its keys, and the routed kernel reads
them from the layer's clean keys and values.
"""
from functools import cache

import torch
from torch.nn import functional as F

import dust_noise
from dust_noise import EPS, norm, norm_difference

Settings, Boost = dust_noise.Settings, dust_noise.Boost


@cache
def indexed_kernel():
    """dust_noise.finite_attention_kernels' routed kernel, reading each query's keys and values through its indices."""
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def indexed(Q, K, V, I, P, Rest, Out, Length,
                Probes: tl.constexpr, Heads: tl.constexpr, Keys: tl.constexpr, D: tl.constexpr,
                BP: tl.constexpr, BK: tl.constexpr):
        token, head = tl.program_id(1) // Heads, tl.program_id(1) % Heads
        probes = tl.program_id(0) * BP + tl.arange(0, BP)
        keys, features = tl.arange(0, BK), tl.arange(0, D)
        q = tl.load(Q + ((token * Probes + probes[:, None]) * Heads + head) * D + features[None, :],
                    probes[:, None] < Probes, 0)
        index = tl.load(I + (token * Heads + head) * Keys + keys, keys < Keys, 0)
        offsets = (head * Length + index[:, None]) * D + features[None, :]
        k = tl.load(K + offsets, keys[:, None] < Keys, 0)
        v = tl.load(V + offsets, keys[:, None] < Keys, 0)
        p = tl.load(P + (token * Heads + head) * Keys + keys, keys < Keys, 0)
        ds = tl.dot(q, tl.trans(k)).to(q.dtype).to(tl.float32) * (D ** -.5)
        dlse = libdevice.log1p(tl.sum(p[None, :] * libdevice.expm1(ds), 1))
        weights = (p[None, :] * libdevice.expm1(ds - dlse[:, None])).to(v.dtype)
        moved = tl.dot(weights, v).to(v.dtype).to(tl.float32)
        rest = tl.load(Rest + (token * Heads + head) * D + features)
        moved += libdevice.expm1(-dlse)[:, None] * rest[None, :]
        tl.store(Out + ((token * Probes + probes[:, None]) * Heads + head) * D + features[None, :],
                 moved, probes[:, None] < Probes)

    return indexed


@torch.library.custom_op('noise_record::indexed_response', mutates_args=())
def indexed_response(dq: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, indices: torch.Tensor,
                     probabilities: torch.Tensor, rest: torch.Tensor) -> torch.Tensor:
    import triton
    m, probes, heads, width = dq.shape
    count = indices.shape[2]
    out = torch.empty((m, probes, heads, width), device=dq.device, dtype=torch.float32)
    indexed_kernel()[(triton.cdiv(probes, 32), m * heads)](
        dq.contiguous(), keys, values, indices.contiguous(), probabilities.contiguous(), rest.contiguous(), out,
        keys.shape[1], probes, heads, count, width, 32, max(16, triton.next_power_of_2(count)), num_warps=4)
    return out


@indexed_response.register_fake
def _(dq, keys, values, indices, probabilities, rest):
    return torch.empty(dq.shape, device=dq.device, dtype=torch.float32)


def query_change_indexed(x, delta, pos, w_q, keys, values, indices, top_p, rest, w_proj_t, cos, sin):
    """dust_noise.query_change_routed with the selected keys read through their indices."""
    m, probes, _ = delta.shape
    heads, _, width = keys.shape
    c, s = cos[pos][:, None, None], sin[pos][:, None, None]
    rotate = lambda u: torch.cat([u[..., :width // 2] * c + u[..., width // 2:] * s,
                                  u[..., :width // 2] * -s + u[..., width // 2:] * c], -1)
    r0 = rotate((norm(x) @ w_q.T).view(m, 1, heads, width))
    dr = rotate((norm_difference(x, delta) @ w_q.T).view(m, probes, heads, width))
    sq = r0.square().sum(-1, keepdim=True)
    inv0 = torch.rsqrt(sq / width + EPS)
    inv = torch.rsqrt((sq + 2 * (r0 * dr).sum(-1, keepdim=True) + dr.square().sum(-1, keepdim=True)) / width + EPS)
    dq = dr * inv + r0 * (inv - inv0)
    dout = indexed_response(dq.to(keys.dtype), keys, values, indices, top_p, rest)
    return dout.flatten(-2) @ w_proj_t


@torch.compile(dynamic=False)
def chain_chunk(site, move, scale, e, x0, mids, xL, idx, hs, lam_r, lam_0, w_fc, w_out_t, weight_t, base,
                target_logit_w, y, own, packed):
    """dust_noise.chain_chunk_eager with indexed own-query keys. packed: w_fc and w_out_t hold each token's selected
    units, else every unit, gathered here as each layer is entered."""
    pos, inputs, w_q, keys, values, indices, top_p, rest, w_proj_t, (cos, sin) = own
    if site < 0:
        dx0 = norm(e.unsqueeze(1) + move) - x0.unsqueeze(1)
        delta, first = (lam_r[0] + lam_0[0]) * dx0, 0
    else:
        dx0, delta, first = None, move, site
    for b in range(first, len(mids)):
        if b > first:
            delta = lam_r[b] * delta + (lam_0[b] * dx0 if dx0 is not None else 0)
        if site < 0 or b > first:
            delta = delta + query_change_indexed(inputs[b], delta, pos, w_q[b], keys[b], values[b], indices[b],
                                                 top_p[b], rest[b], w_proj_t[b], cos, sin)
        dz = norm_difference(mids[b], delta)
        fc, wo = (w_fc[b], w_out_t[b]) if packed else (w_fc[b][idx[b]], w_out_t[b][idx[b]])
        dh = torch.bmm(dz.to(fc.dtype), fc.transpose(1, 2)).float()
        h = hs[b].unsqueeze(1)
        up, down = F.relu(h + dh), F.relu(h)
        delta = delta + torch.bmm(((up - down) * (up + down)).to(wo.dtype), wo).float()
    sq = xL.square().sum(-1, keepdim=True)
    dots = torch.einsum('md,mpd->mp', xL, delta)
    inv = torch.rsqrt((sq + 2 * dots + delta.square().sum(-1)) / xL.shape[-1] + EPS)
    z = (base.unsqueeze(1) + (delta.to(torch.bfloat16) @ weight_t).float()) * inv.unsqueeze(-1)
    zy = (base.gather(1, y.clamp(min=0).unsqueeze(1)) + (delta * target_logit_w.unsqueeze(1)).sum(-1)) * inv
    moved = torch.exp(15 * torch.tanh(z / 15)).sum(-1).log() - 15 * torch.tanh(zy / 15)
    inv0 = torch.rsqrt(sq / xL.shape[-1] + EPS)
    clean = torch.exp(15 * torch.tanh(base * inv0 / 15)).sum(-1, keepdim=True).log() - 15 * torch.tanh(
        base.gather(1, y.clamp(min=0).unsqueeze(1)) * inv0 / 15)
    return (moved - clean) / scale * (y >= 0).unsqueeze(1)


class Cut(dust_noise.Cut):
    def indexed_keys(self, sequence, k):
        """dust_noise.Cut.strong_keys keeping each query's key indices instead of copies."""
        pos, inputs, w_q, keys, values, w_proj_t, cos, sin, windows = self.own_query_inputs(sequence)
        length = len(pos)
        indices, top_p, rests = [], [], []
        for x, wq, kk, vv, window in zip(inputs, w_q, keys, values, windows):
            heads, _, width = kk.shape
            c, s = cos[:, None], sin[:, None]
            r = (norm(x) @ wq.T).view(length, heads, width)
            q = norm(torch.cat([r[..., :width // 2] * c + r[..., width // 2:] * s,
                                r[..., :width // 2] * -s + r[..., width // 2:] * c], -1))
            lag = pos[:, None] - pos[None]
            allowed = (lag >= 0) & (lag <= window)
            prob = torch.softmax((torch.einsum('thd,hsd->hts', q.to(kk.dtype), kk).float() * width ** -0.5)
                                 .masked_fill(~allowed, float('-inf')), -1)
            p, idx = prob.topk(min(k, length), dim=-1)
            out = torch.einsum('hts,hsd->htd', prob.to(vv.dtype), vv).float()
            del prob
            head = torch.arange(heads, device=self.device)[:, None, None]
            rest = torch.cat([out[:, part] - torch.einsum('htk,htkd->htd', p[:, part], vv[head, idx[:, part]].float())
                              for part in torch.arange(length, device=self.device).split(dust_noise.KEY_BUCKET)], 1)
            indices.append(idx.transpose(0, 1).int().contiguous())
            top_p.append(p.transpose(0, 1).contiguous())
            rests.append(rest.transpose(0, 1).contiguous())
        return pos, inputs, w_q, keys, values, indices, top_p, rests, w_proj_t, (cos, sin)

    def mlp_weights(self, w_fc, w_out_t, routes, sl):
        """Each MLP's weights for one token block's replay, and whether they are its tokens' selected units."""
        return ([w[r[0][sl]] for w, r in zip(w_fc, routes)], [w[r[0][sl]] for w, r in zip(w_out_t, routes)], True)

    def chain_credits(self, e, x0, mids, outs, routes, y):
        """dust_noise.Cut.chain_credits with indexed own-query keys."""
        model, s = self.model, self.s
        L = model.config.n_layer
        vocab = model.config.vocab_size
        weight = model.lm_head.weight.float()[:vocab]
        weight_t = weight.T.to(torch.bfloat16).contiguous()
        w_fc = [blk.mlp.c_fc.weight.to(s.chain_dtype) for blk in model.transformer.h]
        w_out_t = [blk.mlp.c_proj.weight.to(s.chain_dtype).T.contiguous() for blk in model.transformer.h]
        lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        xL = outs[-1]
        base = xL @ weight.T
        target_w = weight[y.clamp(min=0)]
        n, m, h = e.shape[0], s.chain_tokens, s.scalar_step
        probes = []
        lo, hi = self.share(self.width)
        P, scalars_here = hi - lo, self.part == 0
        for site in range(-1, L):
            H = self.basis()[lo:hi]
            probed = e if site < 0 else mids[site]
            steps = s.chain_sigma * probed.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
            probes.append((site, H, steps, torch.empty(n, self.width, device=self.device),
                           torch.zeros(2, device=self.device)))
        length = self.capture.length
        for start in range(0, n, length):
            pos, inputs, w_q, keys, values, indices, top_p, rests, w_proj_t, cos_sin = self.indexed_keys(
                start // length, s.query_keys)
            for t0 in range(start, start + length, m):
                t1 = min(t0 + m, start + length)
                sl, local = slice(t0, t1), slice(t0 - start, t1 - start)
                own = (pos[local], [z[local] for z in inputs], w_q, keys, values, [z[local] for z in indices],
                       [z[local] for z in top_p], [z[local] for z in rests], w_proj_t, cos_sin)
                fc, out, packed = self.mlp_weights(w_fc, w_out_t, routes, sl)
                for site, H, steps, credit, lam in probes:
                    move = steps[sl].unsqueeze(-1) * H.unsqueeze(0)
                    scale = steps[sl].expand(-1, P)
                    if site >= 0 and scalars_here:  # symmetric probes of both linear mixing scalars
                        prev = (x0 if site == 0 else outs[site - 1])[sl].unsqueeze(1)
                        move = torch.cat([move, h * prev, h * x0[sl].unsqueeze(1),
                                          -h * prev, -h * x0[sl].unsqueeze(1)], 1)
                        scale = torch.cat([scale, torch.full_like(scale[:, :2], h),
                                           torch.full_like(scale[:, :2], -h)], 1)
                    derivative = chain_chunk(
                        site, move, scale, e[sl], x0[sl], [z[sl] for z in mids], xL[sl], [r[0][sl] for r in routes],
                        [r[1][sl] for r in routes], lam_r, lam_0, fc, out, weight_t, base[sl], target_w[sl], y[sl],
                        own, packed)
                    credit[sl] = derivative[:, :P] @ H / self.width
                    if site >= 0 and scalars_here:
                        lam += (derivative[:, P:P + 2] + derivative[:, P + 2:]).sum(0) / 2
                del fc, out
            del own, inputs, keys, values, indices, top_p, rests
        credits = {}
        for site, _, _, credit, lam in probes:
            credits['emb' if site < 0 else site] = self.group_sum(credit)
            if site >= 0:
                credits['lam', site] = self.group_sum(lam)
        return credits
