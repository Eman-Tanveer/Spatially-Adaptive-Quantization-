"""Entry point.

  python -m adabm_diff.run_probe --mock      # verify wiring end-to-end on a tiny UNet
  python -m adabm_diff.run_probe --osediff    # run on the real OSEDiff UNet (fill TODOs)

Run --mock FIRST. It exercises every module (wrap, calibrate, probe, ILP init,
fine-tune, policy, inference) in a few seconds with no checkpoints, so you can
confirm the pipeline before plugging in OSEDiff.
"""
import argparse
import torch
import torch.nn as nn

from .calibrate import build_offline, save_config
from .infer import AdaptiveRunner


# ----------------------------------------------------------------------------
# Mock UNet: a few "blocks" of conv/linear so module surgery + block grouping run.
# ----------------------------------------------------------------------------
class MockUNet(nn.Module):
    def __init__(self, c=4, ch=32):
        super().__init__()
        self.conv_in = nn.Conv2d(c, ch, 3, padding=1)          # skipped (stem)
        self.down_blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(ch, ch, 3, padding=1)) for _ in range(3)])
        self.mid_block = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
                                       nn.Conv2d(ch, ch, 3, padding=1))
        self.up_blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(ch, ch, 3, padding=1)) for _ in range(3)])
        self.conv_out = nn.Conv2d(ch, c, 3, padding=1)         # skipped (tail)

    def forward(self, z):
        h = self.conv_in(z)
        for b in self.down_blocks:
            h = h + b(h)
        h = h + self.mid_block(h)
        for b in self.up_blocks:
            h = h + b(h)
        return self.conv_out(h)


def run_mock():
    torch.manual_seed(0)
    model = MockUNet()
    # forward_fn returns the latent-space output we distill against
    forward_fn = lambda m, x: m(x)
    # synthetic calibration latents with varied complexity
    latents = []
    for i in range(24):
        scale = 0.2 + 1.5 * (i / 24)            # vary frequency content
        z = torch.randn(1, 4, 16, 16) * scale
        latents.append(z)
    lr_images = torch.rand(24, 3, 64, 64)

    qmodel, policy, cscore, config = build_offline(
        model, latents, forward_fn,
        legal_bits=(4, 8), target_avg_bit=5.0,
        lr_images=lr_images, finetune_steps=40, run_gate=True)

    save_config(config, "adabm_diff_config.json")

    runner = AdaptiveRunner(qmodel, policy, cscore)
    print("\nper-image realized bits (online):")
    for i in range(0, 24, 6):
        out = runner.run(latents[i], forward_fn)
        bits = {k: v for k, v in runner.last_bits.items()}
        print(f"  img {i:2d}  avg_bit {runner.realized_avg_bit():.2f}  "
              f"n8={sum(v == 8 for v in bits.values())}  out {tuple(out.shape)}")
    print("\nMOCK SELF-TEST OK")


def run_osediff():
    """
    TODO (you fill these — needs your SD2.1 base + OSEDiff LoRA + RAM):

      from osediff import OSEDiff_gen           # repo: cswry/OSEDiff
      gen = OSEDiff_gen(args, accelerator); unet = gen.unet
      t = gen.timesteps                          # fixed [999]
      # build calibration latents from ~100 LR images:
      latents = [gen.vae.encode(lr).latent_dist.sample()*gen.vae.config.scaling_factor
                 for lr in calib_lr]
      prompt = gen.encode_prompt(["clean, high quality"])   # cache one cond
      forward_fn = lambda m, z: m(z, t, encoder_hidden_states=prompt).sample

      qmodel, policy, cscore, cfg = build_offline(
          unet, latents, forward_fn, legal_bits=(4,8),
          target_avg_bit=5.0, lr_images=calib_lr_tensor, use_ilp=True)
      save_config(cfg, "osediff_q_config.json")

    Notes:
      * default_skip keeps conv_in/conv_out/time_emb FP. Add cross-attn keys to a
        custom skip_fn ONLY if you want them always-FP; otherwise the ILP will
        already give them high bits because they score as sensitive.
      * forward_fn must return a tensor (use .sample on the UNet output).
    """
    raise SystemExit("Fill in run_osediff() with your checkpoint paths (see docstring).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--osediff", action="store_true")
    a = ap.parse_args()
    if a.osediff:
        run_osediff()
    else:
        run_mock()
