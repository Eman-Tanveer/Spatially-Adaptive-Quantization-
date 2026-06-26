"""Offline phase driver.

1. wrap + calibrate quantizers (ranges, weights, per-bit BaC eps)
2. probe sensitivity -> per-block sensitivity scores
3. ILP/greedy init of per-block base bits under the budget
4. fine-tune ONLY the clipping ranges by distilling the FP UNet (LR-only)
5. save the config consumed online

Bits are discrete and hardware-legal; we fine-tune clipping, not bit-widths
(cleaner and more HW-honest than learning a continuous bit via STE).
"""
import json
import numpy as np
import torch

from .qunet import QuantizedModel
from .complexity import complexity_features, ComplexityScore, lr_gradient_density
from . import sensitivity as S
from .ilp_init import allocate_bits_greedy, allocate_bits_ilp, avg_bit
from .policy import BudgetedBitPolicy


def build_offline(model, latents, forward_fn, legal_bits=(4, 8),
                  target_avg_bit=5.0, probe_bit=None, lr_images=None,
                  finetune_steps=200, lr=1e-3, use_ilp=False,
                  feat_weight=10.0, run_gate=True):
    """
    model:      FP UNet (will be wrapped in place)
    latents:    list of single-image UNet inputs for calibration (batch=1)
    forward_fn: forward_fn(model, x) -> latent-space output tensor
    lr_images:  optional [N,C,H,W] LR images to compute the gradient-density baseline
    returns: (qmodel, policy, complexity_score, config_dict)
    """
    probe_bit = probe_bit or min(legal_bits)
    qmodel = QuantizedModel(model, legal_bits=legal_bits, default_bit=max(legal_bits))
    qmodel.calibrate_ranges(latents, forward_fn)

    # complexity scorer
    feats = torch.cat([complexity_features(x if x.dim() == 4 else x.unsqueeze(0))
                       for x in latents], dim=0)
    cscore = ComplexityScore().fit(feats)
    comps = cscore.score(feats).cpu().numpy()

    # sensitivity probe (the gate)
    M, block_ids = S.block_sensitivity_matrix(qmodel, latents, forward_fn, probe_bit)
    if run_gate:
        grad = None
        if lr_images is not None:
            grad = lr_gradient_density(lr_images).cpu().numpy()
        S.analyze(M, comps, grad)
        try:
            S.plot(M, block_ids, comps)
        except Exception as e:
            print("plot skipped:", e)

    sensitivity = {bid: float(M[:, j].mean()) for j, bid in enumerate(block_ids)}
    cost = qmodel.block_cost()

    # init base bits under budget
    alloc = allocate_bits_ilp if use_ilp else allocate_bits_greedy
    base_bits = alloc(sensitivity, cost, legal_bits, target_avg_bit)
    print(f"init avg bit = {avg_bit(base_bits, cost):.3f} (budget {target_avg_bit})")
    qmodel.set_block_bits(base_bits)

    # fine-tune clipping ranges by FP distillation (LR-only)
    _finetune_clipping(qmodel, latents, forward_fn, steps=finetune_steps,
                       lr=lr, feat_weight=feat_weight)

    # policy
    policy = BudgetedBitPolicy(sensitivity, cost, legal_bits, target_avg_bit)
    policy.set_complexity_range(float(np.percentile(comps, 10)),
                                float(np.percentile(comps, 90)))

    config = dict(
        legal_bits=list(legal_bits), target_avg_bit=target_avg_bit,
        base_bits=base_bits, sensitivity=sensitivity, cost=cost,
        comp_lo=policy.comp_lo, comp_hi=policy.comp_hi,
        cscore_mean=cscore.mean.tolist(), cscore_std=cscore.std.tolist(),
    )
    return qmodel, policy, cscore, config


def _finetune_clipping(qmodel, latents, forward_fn, steps, lr, feat_weight):
    # optimize only clipping params (lower/upper, weight upper); weights frozen
    params = []
    for q in qmodel.qmods.values():
        params += [q.aq.lower, q.aq.upper, q.wq.upper]
    opt = torch.optim.Adam(params, lr=lr)
    n = len(latents)
    for step in range(steps):
        x = latents[step % n]
        qmodel.set_enabled(False)
        with torch.no_grad():
            fp = forward_fn(qmodel.model, x).detach()
        qmodel.set_enabled(True)
        q = forward_fn(qmodel.model, x)
        loss = (q - fp).abs().mean()
        if feat_weight:
            loss = loss + feat_weight * (1 - torch.cosine_similarity(
                q.flatten(1), fp.flatten(1)).mean())
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 5) == 0:
            print(f"  ft step {step:4d}  loss {loss.item():.4f}")


def save_config(config, path):
    def _ser(o):
        if isinstance(o, dict):
            return {str(k): _ser(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_ser(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    with open(path, "w") as f:
        json.dump(_ser(config), f, indent=2)
    print(f"saved config -> {path}")
