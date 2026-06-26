"""Spatial (conv-path) adaptive quantization — extends adabm_diff.

The global gate averaged each block's output over H,W and found no per-image
variance. This module targets the dimension averaging hid: WITHIN-latent spatial
variance. Only convolutional (ResNet) layers — spatially local — get region
-adaptive bits; attention layers keep the static per-block bit from the spine.

Workflow (same as the rest of the project):
  1. PROBE first: spatial_sensitivity + analyze_spatial -> is there within-latent
     variance, and does local complexity predict WHERE quantization hurts?
  2. Only if the probe is positive, BUILD: tag conv vs attn layers, drive conv
     activation quantizers with a per-image spatial bit map under a budget.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from .quantizers import fake_quant_asym, QConv2d, QLinear


# ----------------------------------------------------------------------------
# Spatial complexity (per-patch entropy + high-frequency energy)
# ----------------------------------------------------------------------------
def spatial_complexity_map(z, grid=8):
    """z: [1,C,H,W] -> torch [grid,grid] combined complexity (standardized).
    Uses local variance + high-frequency energy: both rise with how hard a region
    is to quantize. (Per-patch entropy is avoided: normalizing a flat patch makes
    it look complex.)"""
    _, C, H, W = z.shape
    ph, pw = H // grid, W // grid
    std = torch.zeros(grid, grid)
    hf = torch.zeros(grid, grid)
    fy = torch.fft.fftfreq(ph).abs().view(-1, 1)
    fx = torch.fft.fftfreq(pw).abs().view(1, -1)
    mask = (torch.sqrt(fy ** 2 + fx ** 2) > 0.25 * 0.5).float().to(z.device)
    for i in range(grid):
        for j in range(grid):
            patch = z[0, :, i*ph:(i+1)*ph, j*pw:(j+1)*pw]
            std[i, j] = patch.std()
            mag = torch.fft.fft2(patch).abs() ** 2
            hf[i, j] = (mag * mask).sum() / (mag.sum() + 1e-8)

    def zsc(a):
        return (a - a.mean()) / (a.std() + 1e-8)
    return (zsc(std) + zsc(hf)).to(z.device)   # [grid,grid]


# ----------------------------------------------------------------------------
# PROBE: does sensitivity vary spatially, and does complexity localize it?
# ----------------------------------------------------------------------------
@torch.no_grad()
def spatial_sensitivity(qm, latents, forward_fn, block_id, grid=8, probe_bit=4):
    """Per-patch quant error on `block_id`'s output, per image, with the matching
    per-patch latent complexity. Returns err_maps, comp_maps : [N,grid,grid]."""
    feat = {}

    def hook(m, i, o):
        feat['o'] = (o[0] if isinstance(o, (tuple, list)) else o).detach()
    h = qm.blocks[block_id][-1].register_forward_hook(hook)

    err_maps, comp_maps = [], []
    for z in latents:
        qm.set_enabled(False); forward_fn(qm.model, z); fp = feat['o'].clone()
        qm.set_enabled(True); qm.set_all_bits(probe_bit); forward_fn(qm.model, z); q = feat['o']
        e = (fp - q).pow(2).mean(dim=1)[0]            # [H,W] mean over channels
        H, W = e.shape; ph, pw = H // grid, W // grid
        em = e[:ph*grid, :pw*grid].reshape(grid, ph, grid, pw).mean(dim=(1, 3))
        err_maps.append(em.cpu().numpy())
        comp_maps.append(spatial_complexity_map(z, grid).cpu().numpy())
    h.remove()
    return np.array(err_maps), np.array(comp_maps)


def analyze_spatial(err_maps, comp_maps):
    per_img_cov = [em.std() / (em.mean() + 1e-12) for em in err_maps]
    spatial_cov = float(np.mean(per_img_cov))
    rho = spearmanr(comp_maps.reshape(-1), err_maps.reshape(-1)).correlation
    print("=" * 56); print("SPATIAL SENSITIVITY PROBE"); print("=" * 56)
    print(f"within-latent spatial CoV of quant error: {spatial_cov:.3f}")
    print(f"  -> {'spatial variance EXISTS' if spatial_cov > 0.3 else 'flat: spatial wont help'}")
    print(f"Spearman(local complexity, local error):  {rho:.3f}")
    print(f"  -> {'complexity localizes sensitivity (BUILD it)' if abs(rho) > 0.2 else 'metric does not localize'}")
    print("=" * 56)
    return dict(spatial_cov=spatial_cov, rho=rho)


# ----------------------------------------------------------------------------
# BUILD: conv/attn tagging + spatially-varying activation quantization
# ----------------------------------------------------------------------------
def tag_conv_attn(qm):
    """Mark which quant layers are conv (spatial-eligible) vs attn/linear (static)."""
    for name, q in qm.qmods.items():
        q.spatial_eligible = isinstance(q, QConv2d)   # ResNet convs only
    return qm


def _eps_for(aq, b):
    name = f"eps_{b}"
    return getattr(aq, name) if hasattr(aq, name) else torch.tensor(1.0)


def set_spatial_bits(aq, bit_map):
    """Attach a [gh,gw] bit map to an ActQuant and switch it to spatial mode."""
    aq._spatial = bit_map

def _spatial_forward(aq, x):
    bm = aq._spatial.to(x.device).float()
    H, W = x.shape[-2:]
    big = F.interpolate(bm[None, None], size=(H, W), mode='nearest')[0, 0]
    out = x
    for b in aq.legal_bits:
        m = (big == b)
        if m.any():
            e = _eps_for(aq, b).to(x.device)
            xq = fake_quant_asym(x, aq.lower * e, aq.upper * e, float(b))
            out = torch.where(m[None, None], xq, out)
    return out


# install spatial forward on ActQuant (keeps scalar behavior when no map set)
from .quantizers import ActQuant
_orig_actquant_forward = ActQuant.forward
def _actquant_forward(self, x):
    if getattr(self, '_spatial', None) is not None and self.enabled and self.bit < 16:
        return _spatial_forward(self, x)
    return _orig_actquant_forward(self, x)
ActQuant.forward = _actquant_forward


# ----------------------------------------------------------------------------
# Spatial budgeted policy: complexity map -> per-patch bits over {4,8}
# ----------------------------------------------------------------------------
class SpatialBudgetedPolicy:
    def __init__(self, legal_bits=(4, 8), frac_high=0.5):
        self.legal = sorted(int(b) for b in legal_bits)
        self.lo, self.hi = self.legal[0], self.legal[-1]
        self.frac_high = frac_high           # fraction of patches at the high bit

    def assign(self, comp_map):
        """comp_map: torch/np [gh,gw] -> torch [gh,gw] of bits in {lo,hi}."""
        c = comp_map if torch.is_tensor(comp_map) else torch.tensor(comp_map)
        flat = c.reshape(-1)
        k = max(1, int(round(self.frac_high * flat.numel())))
        thr = torch.topk(flat, k).values.min()
        return torch.where(c >= thr, torch.tensor(float(self.hi)),
                           torch.tensor(float(self.lo)))

    def avg_bit(self, bit_map):
        return float(bit_map.float().mean())
