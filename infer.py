"""Online per-image inference.

For each image: encode -> latent complexity -> budgeted bit policy ->
set per-block realized bits (clipping auto-selects the eps for that bit) ->
quantized UNet forward -> decode.
"""
import torch

from .complexity import complexity_features, ComplexityScore


class AdaptiveRunner:
    def __init__(self, qmodel, policy, cscore):
        self.qmodel = qmodel
        self.policy = policy
        self.cscore = cscore
        self.last_bits = None

    @torch.no_grad()
    def run(self, latent, forward_fn):
        z = latent if latent.dim() == 4 else latent.unsqueeze(0)
        feats = complexity_features(z)
        c = self.cscore.score(feats).item()
        bits = self.policy.assign(c)
        self.qmodel.set_block_bits(bits)
        self.last_bits = bits
        return forward_fn(self.qmodel.model, z)

    def realized_avg_bit(self):
        return self.policy.realized_avg_bit(self.last_bits)


def load_cscore(config):
    cs = ComplexityScore()
    cs.mean = torch.tensor(config["cscore_mean"])
    cs.std = torch.tensor(config["cscore_std"])
    return cs
