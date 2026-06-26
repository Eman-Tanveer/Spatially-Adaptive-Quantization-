"""Wrap a (diffusers) UNet's Conv2d/Linear layers with quantizers, grouped by block.

Backbone-agnostic: works on any nn.Module tree. `default_block_id` groups by the
diffusers naming convention (down_blocks.N / mid_block / up_blocks.N). Override
`block_id_fn` for other backbones.
"""
import re
import torch
import torch.nn as nn

from .quantizers import QConv2d, QLinear, ActQuant


def default_block_id(name):
    # diffusers: 'down_blocks.0.resnets.1.conv1' -> 'down_blocks.0'
    m = re.match(r"(down_blocks\.\d+|up_blocks\.\d+|mid_block)", name)
    if m:
        return m.group(1)
    # fallback: first two dotted tokens
    toks = name.split(".")
    return ".".join(toks[:2]) if len(toks) >= 2 else name


def default_skip(name):
    # keep stems / time embedding at full precision (sensitive, cheap)
    return any(s in name for s in ("conv_in", "conv_out", "time_emb", "time_embedding"))


class QuantizedModel:
    def __init__(self, model, legal_bits=(4, 8), default_bit=8,
                 block_id_fn=default_block_id, skip_fn=default_skip):
        self.model = model
        self.legal_bits = tuple(legal_bits)
        self.qmods = {}          # qualified_name -> QConv2d/QLinear
        self.blocks = {}         # block_id -> [qmods]
        self._wrap(block_id_fn, skip_fn, default_bit)

    def _wrap(self, block_id_fn, skip_fn, default_bit):
        targets = []
        for name, mod in self.model.named_modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)) and not skip_fn(name):
                targets.append((name, mod))
        for name, mod in targets:
            if isinstance(mod, nn.Conv2d):
                q = QConv2d(mod, self.legal_bits, default_bit)
            else:
                q = QLinear(mod, self.legal_bits, default_bit)
            q.block_id = block_id_fn(name)
            self._set_submodule(name, q)
            self.qmods[name] = q
            self.blocks.setdefault(q.block_id, []).append(q)

    def _set_submodule(self, name, new):
        parent = self.model
        toks = name.split(".")
        for t in toks[:-1]:
            parent = getattr(parent, t) if not t.isdigit() else parent[int(t)]
        last = toks[-1]
        if last.isdigit():
            parent[int(last)] = new
        else:
            setattr(parent, last, new)

    # ---- bit control ----
    def set_all_bits(self, b):
        for q in self.qmods.values():
            q.set_bit(b)

    def set_block_bits(self, bits):  # {block_id: bit}
        for bid, b in bits.items():
            for q in self.blocks[bid]:
                q.set_bit(b)

    def set_enabled(self, flag):
        for q in self.qmods.values():
            q.aq.enabled = flag
            q.wq.enabled = flag

    def block_ids(self):
        return list(self.blocks.keys())

    def block_cost(self):
        """Relative compute weight per block = sum of weight elements (proxy for BitOPs/bit)."""
        cost = {}
        for bid, qs in self.blocks.items():
            c = 0
            for q in qs:
                w = q.conv.weight if isinstance(q, QConv2d) else q.lin.weight
                c += w.numel()
            cost[bid] = float(c)
        return cost

    # ---- calibration ----
    @torch.no_grad()
    def calibrate_ranges(self, batches, forward_fn):
        """Collect activation ranges (EMA) + weight ranges + per-bit BaC eps.
        batches: iterable of model inputs; forward_fn(model, batch) runs a pass."""
        # 1) activation ranges via observe hooks
        handles = []
        for q in self.qmods.values():
            handles.append(q.aq.register_forward_pre_hook(
                lambda m, inp: m.observe(inp[0])))
        self.set_enabled(False)  # FP forward while only observing
        for batch in batches:
            forward_fn(self.model, batch)
        for h in handles:
            h.remove()
        for q in self.qmods.values():
            q.aq.finalize_range()
            w = q.conv.weight if isinstance(q, QConv2d) else q.lin.weight
            q.wq.calibrate(w)
        # 2) BaC eps per bit: grab one real activation per quantizer
        captured = {}
        handles = []
        for name, q in self.qmods.items():
            handles.append(q.aq.register_forward_pre_hook(
                (lambda nm: (lambda m, inp: captured.__setitem__(nm, inp[0].detach())))(name)))
        first = next(iter(batches)) if not isinstance(batches, list) else batches[0]
        forward_fn(self.model, first)
        for h in handles:
            h.remove()
        for name, q in self.qmods.items():
            if name in captured:
                q.aq.calibrate_bac(captured[name])
        self.set_enabled(True)
