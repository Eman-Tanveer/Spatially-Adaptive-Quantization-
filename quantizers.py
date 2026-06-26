"""Fake-quantization modules.

Key correction over AdaBM/PassionSR: the activation clipping range is selected
for the *realized* bit-width chosen per image (not just the calibration-time bit).
Each quantizer stores an OMSE clip scale `eps_b` per legal bit; when the policy
sets bit=b at inference, the range becomes [eps_b*lower, eps_b*upper].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g  # straight-through estimator


_round = _Round.apply


def fake_quant_asym(x, lo, hi, bit):
    """Asymmetric uniform fake-quant (activations)."""
    levels = (2.0 ** bit) - 1.0
    hi = torch.maximum(hi, lo + 1e-6)
    s = (hi - lo) / levels
    xc = torch.clamp(x, min=lo, max=hi)
    return _round((xc - lo) / s) * s + lo


def fake_quant_sym(w, u, bit):
    """Symmetric uniform fake-quant (weights)."""
    levels = (2.0 ** bit) - 1.0
    u = torch.clamp(u, min=1e-6)
    s = (2.0 * u) / levels
    wc = torch.clamp(w, min=-u, max=u)
    return _round((wc + u) / s) * s - u


class ActQuant(nn.Module):
    def __init__(self, legal_bits=(4, 8), default_bit=8):
        super().__init__()
        self.legal_bits = tuple(sorted(int(b) for b in legal_bits))
        self.lower = nn.Parameter(torch.tensor(-1.0))
        self.upper = nn.Parameter(torch.tensor(1.0))
        for b in self.legal_bits:
            self.register_buffer(f"eps_{b}", torch.tensor(1.0))
        self.bit = int(default_bit)
        self.enabled = True
        self._ema_min = None
        self._ema_max = None

    @torch.no_grad()
    def observe(self, x, beta=0.9):
        mn, mx = x.min().detach(), x.max().detach()
        if self._ema_min is None:
            self._ema_min, self._ema_max = mn.clone(), mx.clone()
        else:
            self._ema_min = beta * self._ema_min + (1 - beta) * mn
            self._ema_max = beta * self._ema_max + (1 - beta) * mx

    @torch.no_grad()
    def finalize_range(self):
        if self._ema_min is not None:
            self.lower.data = self._ema_min.clone()
            self.upper.data = self._ema_max.clone()

    @torch.no_grad()
    def calibrate_bac(self, x, steps=100):
        """OMSE search for eps per legal bit, minimizing quant MSE at that bit."""
        lo0, hi0 = self.lower.data, self.upper.data
        for b in self.legal_bits:
            best, best_e = float("inf"), 1.0
            for i in range(steps):
                e = 1.0 - i * (0.9 / steps)  # 1.0 .. 0.1
                xq = fake_quant_asym(x, lo0 * e, hi0 * e, float(b))
                err = (x - xq).pow(2).mean().item()
                if err < best:
                    best, best_e = err, e
            getattr(self, f"eps_{b}").data = torch.tensor(best_e)

    def set_bit(self, b):
        self.bit = int(b)

    def _eps(self, device):
        name = f"eps_{self.bit}"
        return getattr(self, name) if hasattr(self, name) else torch.tensor(1.0, device=device)

    def forward(self, x):
        if not self.enabled or self.bit >= 16:
            return x
        e = self._eps(x.device)
        return fake_quant_asym(x, self.lower * e, self.upper * e, float(self.bit))


class WeightQuant(nn.Module):
    def __init__(self, legal_bits=(4, 8), default_bit=8):
        super().__init__()
        self.legal_bits = tuple(sorted(int(b) for b in legal_bits))
        self.upper = nn.Parameter(torch.tensor(1.0))
        for b in self.legal_bits:
            self.register_buffer(f"eps_{b}", torch.tensor(1.0))
        self.bit = int(default_bit)
        self.enabled = True

    @torch.no_grad()
    def calibrate(self, w, steps=100):
        u0 = w.abs().max()
        self.upper.data = u0.clone()
        for b in self.legal_bits:
            best, best_e = float("inf"), 1.0
            for i in range(steps):
                e = 1.0 - i * (0.9 / steps)
                wq = fake_quant_sym(w, u0 * e, float(b))
                err = (w - wq).pow(2).mean().item()
                if err < best:
                    best, best_e = err, e
            getattr(self, f"eps_{b}").data = torch.tensor(best_e)

    def set_bit(self, b):
        self.bit = int(b)

    def _eps(self, device):
        name = f"eps_{self.bit}"
        return getattr(self, name) if hasattr(self, name) else torch.tensor(1.0, device=device)

    def forward(self, w):
        if not self.enabled or self.bit >= 16:
            return w
        return fake_quant_sym(w, self.upper * self._eps(w.device), float(self.bit))


class QConv2d(nn.Module):
    """Wraps a frozen nn.Conv2d with activation + weight quantizers."""

    def __init__(self, conv, legal_bits=(4, 8), default_bit=8):
        super().__init__()
        self.conv = conv
        for p in self.conv.parameters():
            p.requires_grad_(False)
        self.aq = ActQuant(legal_bits, default_bit)
        self.wq = WeightQuant(legal_bits, default_bit)
        self.block_id = None

    def set_bit(self, b):
        self.aq.set_bit(b)
        self.wq.set_bit(b)

    def forward(self, x):
        xq = self.aq(x)
        wq = self.wq(self.conv.weight)
        c = self.conv
        return F.conv2d(xq, wq, c.bias, c.stride, c.padding, c.dilation, c.groups)


class QLinear(nn.Module):
    def __init__(self, lin, legal_bits=(4, 8), default_bit=8):
        super().__init__()
        self.lin = lin
        for p in self.lin.parameters():
            p.requires_grad_(False)
        self.aq = ActQuant(legal_bits, default_bit)
        self.wq = WeightQuant(legal_bits, default_bit)
        self.block_id = None

    def set_bit(self, b):
        self.aq.set_bit(b)
        self.wq.set_bit(b)

    def forward(self, x):
        return F.linear(self.aq(x), self.wq(self.lin.weight), self.lin.bias)
