---
status: accepted
---

# Train on the RX 7900 XTX through native Windows ROCm, not DirectML or Colab

Voice Models are trained locally so the user never has to open Colab. PyTorch-DirectML crashes on the HiFi-GAN discriminator backward pass, so we use AMD's official Windows ROCm wheels (ROCm 7.2.1, torch 2.9.1, Python 3.12, driver ≥ 26.2.2) instead. WSL2 + ROCm was the fallback we rejected so the desktop app doesn't have to bridge Windows and Linux for every job. Accepted 2026-10-30: `checks/gpu_smoke_test.py` passed in fp32 and fp16 on the RX 7900 XTX (driver 32.0.32015.7029). The first run takes ~8 s/step while MIOpen tunes kernels; after that it's ~0.8 s/step, using 2.1 GiB of VRAM.
