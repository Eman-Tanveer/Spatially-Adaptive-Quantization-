"""End-to-end spatial-adaptive SR inference + the ablation that proves the map.

Compares, at a MATCHED average activation bit on the target (up-)blocks:
  uniform : target conv activations all at `uniform_bit`
  random  : target conv activations get a random {lo,hi} spatial map
  spatial : target conv activations get the complexity-driven {lo,hi} map
All other layers (down-blocks, attention) and all weights are held identical
across modes, so any quality difference is due ONLY to where the bits go.
Fidelity is measured against the FP output (needs no HR ground truth); pass
hr_images to also score against ground truth.
"""
import numpy as np
import torch

from . import spatial
from .quantizers import QConv2d


# ----------------------------------------------------------------------------
def _apply(qm, target_blocks, act_map, static_act=8, weight_bit=8):
    """act_map: scalar bit (uniform) OR torch [g,g] map (random/spatial)."""
    for bid, qs in qm.blocks.items():
        is_target = bid in target_blocks
        for q in qs:
            q.wq.set_bit(weight_bit)                       # weights constant everywhere
            if is_target and isinstance(q, QConv2d):
                if torch.is_tensor(act_map):
                    spatial.set_spatial_bits(q.aq, act_map)
                else:
                    q.aq._spatial = None; q.aq.set_bit(int(act_map))
            else:
                q.aq._spatial = None; q.aq.set_bit(static_act)


def _avg_target_bit(qm, target_blocks, act_map):
    if torch.is_tensor(act_map):
        return float(act_map.float().mean())
    return float(act_map)


@torch.no_grad()
def sr_image(model, qm, lr, prompt, mode, target_blocks,
             grid=8, frac_high=0.5, legal=(4, 8), uniform_bit=6,
             static_act=8, weight_bit=8):
    vae = model.vae; t = model.timesteps; sf = vae.config.scaling_factor
    ns = model.noise_scheduler
    z = vae.encode(lr).latent_dist.sample() * sf

    if mode == 'fp':
        qm.set_enabled(False)
    else:
        qm.set_enabled(True)
        pol = spatial.SpatialBudgetedPolicy(legal, frac_high)
        if mode == 'uniform':
            _apply(qm, target_blocks, uniform_bit, static_act, weight_bit)
        elif mode == 'random':
            bm = pol.assign(torch.rand(grid, grid))
            _apply(qm, target_blocks, bm, static_act, weight_bit)
        elif mode == 'spatial':
            bm = pol.assign(spatial.spatial_complexity_map(z, grid))
            _apply(qm, target_blocks, bm, static_act, weight_bit)
        else:
            raise ValueError(mode)

    pred = qm.model(z, t, encoder_hidden_states=prompt).sample
    x0 = ns.step(pred, t, z, return_dict=True).prev_sample
    hr = vae.decode(x0 / sf).sample.clamp(-1, 1)
    return ((hr + 1) / 2)[0].permute(1, 2, 0).float().cpu().numpy()   # HxWx3 in [0,1]


# ----------------------------------------------------------------------------
def evaluate(model, qm, lrs, prompt, target_blocks, n=30,
             hr_images=None, lpips_fn=None, **kw):
    """Prints the ablation table. Metrics are vs FP output (quant fidelity);
    if hr_images given, also vs ground truth."""
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    modes = ['uniform', 'random', 'spatial']
    R = {m: {'psnr': [], 'ssim': [], 'lpips': [], 'gtpsnr': []} for m in modes}

    for i, lr in enumerate(lrs[:n]):
        ref = sr_image(model, qm, lr, prompt, 'fp', target_blocks, **kw)
        gt = None
        if hr_images is not None:
            gt = hr_images[i].permute(1, 2, 0).float().cpu().numpy()
        for m in modes:
            out = sr_image(model, qm, lr, prompt, m, target_blocks, **kw)
            R[m]['psnr'].append(psnr(ref, out, data_range=1.0))
            R[m]['ssim'].append(ssim(ref, out, channel_axis=2, data_range=1.0))
            if gt is not None:
                R[m]['gtpsnr'].append(psnr(gt, out, data_range=1.0))
            if lpips_fn is not None:
                a = torch.tensor(out).permute(2, 0, 1)[None] * 2 - 1
                b = torch.tensor(ref).permute(2, 0, 1)[None] * 2 - 1
                R[m]['lpips'].append(float(lpips_fn(a.cuda(), b.cuda())))

    print(f"avg target-block activation bit ~ {kw.get('uniform_bit',6)} (matched across modes)\n")
    hdr = f"{'mode':9s} {'PSNR(vsFP)':11s} {'SSIM(vsFP)':11s}"
    if hr_images is not None: hdr += f" {'PSNR(vsGT)':11s}"
    if lpips_fn is not None: hdr += f" {'LPIPS(vsFP)':11s}"
    print(hdr); print('-'*len(hdr))
    for m in modes:
        line = f"{m:9s} {np.mean(R[m]['psnr']):8.2f}    {np.mean(R[m]['ssim']):.4f}     "
        if hr_images is not None: line += f"{np.mean(R[m]['gtpsnr']):8.2f}    "
        if lpips_fn is not None: line += f"{np.mean(R[m]['lpips']):.4f}"
        print(line)
    print("\nspatial should beat random (same bit budget, better placement)")
    print("and ideally beat uniform-%d (adaptive {4,8} > uniform-6)." % kw.get('uniform_bit', 6))
    return R
