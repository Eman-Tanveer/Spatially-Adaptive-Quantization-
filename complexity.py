"""Latent complexity signal that drives the online bit policy.

Two features per latent: Shannon entropy of the value distribution and the
fraction of spectral energy in high frequencies. A ComplexityScore standardizes
them using calibration-set statistics and returns one scalar per image.

NOTE: this metric must be *validated* by sensitivity.py before you trust it.
On OSEDiff the latent is the clean VAE encoding of the LR image, so this has to
beat gradient density on the LR image to justify operating in latent space.
"""
import torch


def _entropy(z, n_bins=64):
    # z: [B, C, H, W] -> [B] entropy of per-image value histogram
    B = z.shape[0]
    out = []
    for i in range(B):
        v = z[i].reshape(-1)
        v = (v - v.min()) / (v.max() - v.min() + 1e-8)
        hist = torch.histc(v, bins=n_bins, min=0.0, max=1.0)
        p = hist / (hist.sum() + 1e-8)
        p = p[p > 0]
        out.append(-(p * p.log()).sum())
    return torch.stack(out)


def _hf_energy(z, cutoff=0.25):
    # fraction of 2D spectral energy with radius > cutoff*Nyquist, mean over channels
    B, C, H, W = z.shape
    fy = torch.fft.fftfreq(H, device=z.device).abs().view(H, 1)
    fx = torch.fft.fftfreq(W, device=z.device).abs().view(1, W)
    radius = torch.sqrt(fy ** 2 + fx ** 2)
    hf_mask = (radius > cutoff * 0.5).float()
    mag = torch.fft.fft2(z).abs() ** 2  # [B,C,H,W]
    hf = (mag * hf_mask).sum(dim=(-1, -2))
    tot = mag.sum(dim=(-1, -2)) + 1e-8
    return (hf / tot).mean(dim=1)  # [B]


def complexity_features(z):
    """z: [B,C,H,W] -> [B,2] raw (entropy, hf_energy)."""
    return torch.stack([_entropy(z), _hf_energy(z)], dim=1)


class ComplexityScore:
    """Standardizes the two features using calibration statistics."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, feats):
        self.mean = feats.mean(dim=0)
        self.std = feats.std(dim=0) + 1e-8
        return self

    def score(self, feats):
        z = (feats - self.mean) / self.std
        return z.sum(dim=1)  # [B] combined scalar


def lr_gradient_density(img):
    """Baseline metric to beat: AdaBM's gradient density on the (LR) image.
    img: [B,C,H,W] in [0,1] -> [B]."""
    gray = img.mean(dim=1, keepdim=True)
    gx = gray[..., :, 1:] - gray[..., :, :-1]
    gy = gray[..., 1:, :] - gray[..., :-1, :]
    return gx.abs().mean(dim=(-1, -2, -3)) + gy.abs().mean(dim=(-1, -2, -3))
