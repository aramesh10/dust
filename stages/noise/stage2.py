"""Optimization 2 on stage 1: MLP units gathered per layer (Cut). The replay restricts each MLP to its token's top-512
units. Stage 1 gathered those units' weights for every block of tokens and every layer at once (4 GiB); here each
layer's are gathered inside the replay when it is entered, and dropped when it is left.
"""
from stages.noise import stage1

Settings, Boost = stage1.Settings, stage1.Boost


class Cut(stage1.Cut):
    def mlp_weights(self, w_fc, w_out_t, routes, sl):
        return w_fc, w_out_t, False
