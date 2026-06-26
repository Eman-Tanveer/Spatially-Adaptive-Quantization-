# Spatially-Adaptive Quantization for One-Step Diffusion Super-Resolution

Post-training quantization (PTQ) of **OSEDiff** (one-step diffusion real-world image super-resolution) with **per-region adaptive bit allocation** on the decoder, plus a lightweight **LoRA adapter** to recover perceptual quality at low bit-widths.

> **Domain:** Model compression & efficient inference for diffusion-based image super-resolution.
> **Core idea:** quantization sensitivity is spatially non-uniform — so allocate precision per region instead of uniformly.

---

## Highlights

- **Sensitivity decomposition** for one-step diffusion SR: shows per-image adaptation is ineffective, per-layer sensitivity is stable, and spatial sensitivity is concentrated in the decoder up-blocks.
- **Spatially-adaptive bit allocation** on the decoder convolutional path, driven by a variance + frequency complexity map; attention layers kept static for hardware realizability.
- **Parameter-efficient adapter** (rank-8 LoRA, ~1.18M params) trained by perceptual distillation to recover quality at ultra-low bit-widths.
- **Official-protocol evaluation** on RealSR, reproducing OSEDiff's published full-precision numbers (PSNR ~= 25.3) so results are comparable to prior work.

---

## Architecture

![Architecture](architecture_current.png)

LR image -> DAPE/RAM (prompt) + VAE encoder (latent z) -> **Quantized UNet** (one step, t=999; encoder/attention static, decoder up-blocks spatially-adaptive) -> **LoRA adapter** (low-bit recovery) -> scheduler -> VAE decode -> AdaIN color fix -> SR output. A spatial complexity map and budgeted bit policy assign per-region bits to the decoder; offline calibration supplies the quantization config.

---

## Setup

OSEDiff's vendored model files require an older `diffusers`. Use a clean environment with the pinned versions below.

```bash
# 1. Environment
conda create -n ptq_osediff python=3.10 -y
conda activate ptq_osediff

# 2. PyTorch (match your CUDA; example: CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Pinned dependencies (required by OSEDiff)
pip install diffusers==0.25.0 transformers==4.44.2 huggingface_hub==0.25.2 \
            tokenizers==0.19.1 accelerate==0.33.0 peft==0.11.1

# 4. Eval / utility deps
pip install pyiqa lpips piq scikit-image scipy pulp matplotlib Pillow tqdm opencv-python

# 5. OSEDiff backbone
git clone https://github.com/cswry/OSEDiff
```

### Weights
- **Stable Diffusion 2.1-base** (diffusers format, safetensors only — not the `.ckpt`).
- **OSEDiff weights:** `osediff.pkl` (ships at `OSEDiff/preset/models/osediff.pkl`).
- **RAM / DAPE prompt model:** `ram_swin_large_14m.pth` (public, from the Recognize Anything project) and `DAPE.pth`.

> Note: GPU with >= 16 GB VRAM recommended. Adapter training uses gradient checkpointing on the UNet (and VAE for pixel-LPIPS) to fit 16 GB.

---

## Repository structure

```
adabm_diff/
├── quantizers.py    # fake-quant (STE), ActQuant/WeightQuant/QConv2d/QLinear, per-bit OMSE clipping (eps)
├── qunet.py         # QuantizedModel: in-place wrapping, block grouping, set bits, calibrate, block_cost
├── complexity.py    # complexity measures
├── sensitivity.py   # sensitivity decomposition (image / layer / spatial diagnostic)
├── spatial.py       # spatial complexity map (variance + frequency), spatial allocation, policy
├── policy.py        # budgeted bit policy (average-bit budget -> per-region bits)
├── ilp_init.py      # greedy / ILP per-block bit initialization
├── calibrate.py     # offline calibration driver (OMSE eps for {2,4,8}), save config
├── infer.py         # inference helpers
└── eval_sr.py       # evaluation (PSNR/SSIM/LPIPS/DISTS), ablation modes
```

Phase scripts (`phaseN_*.py`) orchestrate the pipeline (setup, calibration, ceiling, ablation, adapter, re-base).

---

## Usage

The pipeline runs in gated phases. Run one phase, verify its output, then proceed.

```bash
python phase1_setup.py        # build model, load data, encode calibration latents
python phase2_calibrate.py    # OMSE clipping for {2,4,8}; verify eps (none = 1.0)
python phase3_ceiling.py      # full-precision ceiling under official protocol
python phase4_ablation.py     # matched-budget: uniform vs random vs spatial
python phase7_adapter.py      # train LoRA adapter (pixel-LPIPS), evaluate
```

**Key configuration**
- Legal activation bits: `{2, 4, 8}`; weights static 8-bit (W8).
- Target blocks (adaptive): `up_blocks.1`, `up_blocks.2` (decoder).
- Calibration on RealSR **train**, evaluation on RealSR **test**.
- Inference timestep: `t = 999` (OSEDiff design — do not change).

---

## Evaluation

- **Datasets:** RealSR (V3) x4, Canon + Nikon (100 test pairs); DRealSR planned.
- **Metrics:** Y-channel PSNR/SSIM, LPIPS, DISTS (via `pyiqa`), reported vs. full-precision and vs. ground truth.
- **Protocol:** OSEDiff's official test pipeline — arbitrary-resolution LR, x4 upscale, Lanczos multiple-of-8 alignment, RAM prompts, AdaIN color fix.
- **Efficiency:** average bit-width / relative BitOPs (theoretical cost; no wall-clock claims, since quantization is simulated).

**Full-precision ceiling (reference):** PSNR ~= 25.34 · SSIM ~= 0.7395 · LPIPS ~= 0.2969 · DISTS ~= 0.1791.

> Quantized results are being re-computed under the official protocol; see `complete_documentation.md` for current findings and the project document for full framing.

---

## Status & limitations

- Adaptive precision is applied to **activations**; weights are static 8-bit. Adaptive **weight** quantization is planned.
- The bit allocator is currently **binary** (high/low); a **multi-level {2,4,8}** allocator is the next upgrade.
- The complexity map is a hand-crafted proxy (Spearman ~= 0.3–0.56 with true sensitivity); a learned, perceptually-aligned map is future work.
- Efficiency is reported at the BitOPs level; realized speedup needs low-bit kernel support.
- Two ablations are honest **negatives**: learned-clipping fine-tune (negligible vs. OMSE) and MSE-based error compensation (degrades perceptual quality).

---

## Acknowledgments

Built on **OSEDiff** (Wu et al.) and Stable Diffusion 2.1. Quantization design draws on AdaBM, PassionSR, MixDQ, and PTQD; the adapter follows the QLoRA family. Prompts use the Recognize Anything Model (RAM/DAPE).

---

## License

For academic / research use. Respect the licenses of OSEDiff, Stable Diffusion, and RAM.
