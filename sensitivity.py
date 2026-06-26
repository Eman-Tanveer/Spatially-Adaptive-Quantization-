"""THE GATE. Run this before building anything else.

Measures, on the FP vs statically-quantized UNet:
  - per-block x per-image quantization error matrix M
  - whether per-image variance is large enough to be worth adapting    (Q1)
  - whether layer-order is image-invariant (AdaBM's independence claim) (Q2)
  - whether your latent complexity metric predicts image sensitivity,
    and whether it beats gradient density on the LR image               (Q3)

Decision:
  * Q1 small  -> kill the per-image axis; make bits static, pivot novelty
                 (prompt/cross-attention-conditioned allocation).
  * Q1 large + Q3 good -> the c-teal column is justified; build it.
  * Q1 large + Q3 poor -> keep adapting but use gradient density (cheaper),
                          or design a better metric.
"""
import numpy as np
import torch
from scipy.stats import spearmanr


@torch.no_grad()
def block_sensitivity_matrix(qmodel, latents, forward_fn, probe_bit=4):
    """
    qmodel: QuantizedModel
    latents: list of single-image model inputs (batch size 1 each)
    forward_fn(model, x) -> runs UNet, return value ignored (we hook blocks)
    returns: M [n_images, n_blocks], block_ids
    """
    block_ids = qmodel.block_ids()
    feats = {bid: None for bid in block_ids}

    def make_hook(bid):
        def hook(m, inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            feats[bid] = o.detach().reshape(-1)
        return hook

    # hook the last quant layer of each block
    handles = []
    for bid, qs in qmodel.blocks.items():
        handles.append(qs[-1].register_forward_hook(make_hook(bid)))

    n_img, n_blk = len(latents), len(block_ids)
    M = np.zeros((n_img, n_blk))
    for i, x in enumerate(latents):
        qmodel.set_enabled(False)              # FP
        forward_fn(qmodel.model, x)
        fp = {bid: feats[bid].clone() for bid in block_ids}
        qmodel.set_enabled(True)
        qmodel.set_all_bits(probe_bit)         # static low-bit
        forward_fn(qmodel.model, x)
        for j, bid in enumerate(block_ids):
            M[i, j] = (fp[bid] - feats[bid]).pow(2).mean().item()
    for h in handles:
        h.remove()
    return M, block_ids


def _mean_pairwise_cosine(rows):
    R = rows / (np.linalg.norm(rows, axis=1, keepdims=True) + 1e-12)
    S = R @ R.T
    iu = np.triu_indices(len(R), k=1)
    return float(S[iu].mean())


def analyze(M, complexity, lr_grad=None):
    """M [n_img,n_blk]; complexity [n_img]; lr_grad [n_img] optional."""
    img_mean = M.mean(axis=1)                       # per-image overall sensitivity
    q1 = float(img_mean.std() / (img_mean.mean() + 1e-12))   # coefficient of variation
    q2_layer_invariance = _mean_pairwise_cosine(M)           # rows = per-image layer profiles
    q2_image_invariance = _mean_pairwise_cosine(M.T)         # cols = per-layer image profiles
    rho_metric = spearmanr(complexity, img_mean).correlation
    rho_grad = spearmanr(lr_grad, img_mean).correlation if lr_grad is not None else None

    print("=" * 60)
    print("SENSITIVITY PROBE RESULTS")
    print("=" * 60)
    print(f"Q1  per-image variance (CoV of image sensitivity): {q1:.3f}")
    print(f"    -> {'enough to adapt' if q1 > 0.15 else 'TOO SMALL: make bits static, pivot'}")
    print(f"Q2  layer-order invariance across images (cos):    {q2_layer_invariance:.3f}")
    print(f"    image-order invariance across layers (cos):    {q2_image_invariance:.3f}")
    print(f"    -> {'decomposition holds' if q2_layer_invariance > 0.9 else 'weak: per-block per-image needed'}")
    print(f"Q3  Spearman(latent complexity, sensitivity):      {rho_metric:.3f}")
    if rho_grad is not None:
        print(f"    Spearman(LR gradient density, sensitivity):    {rho_grad:.3f}")
        better = "latent metric WINS" if abs(rho_metric) > abs(rho_grad) else "gradient density is as good/better -> use it (cheaper)"
        print(f"    -> {better}")
    print("=" * 60)
    return dict(q1_cov=q1, layer_invariance=q2_layer_invariance,
               image_invariance=q2_image_invariance,
               rho_metric=rho_metric, rho_grad=rho_grad)


def plot(M, block_ids, complexity, out_path="sensitivity_probe.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    im = ax[0].imshow(M, aspect="auto", cmap="magma")
    ax[0].set_xlabel("block"); ax[0].set_ylabel("image")
    ax[0].set_title("per-image x per-block quant MSE")
    fig.colorbar(im, ax=ax[0])
    ax[1].scatter(complexity, M.mean(axis=1))
    ax[1].set_xlabel("latent complexity"); ax[1].set_ylabel("mean quant MSE")
    ax[1].set_title("does complexity predict sensitivity?")
    fig.tight_layout(); fig.savefig(out_path, dpi=120)
    print(f"saved {out_path}")
