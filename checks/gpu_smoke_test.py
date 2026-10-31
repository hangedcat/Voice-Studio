"""Check that the GPU can run the parts of RVC training that broke on DirectML.

Mirrors the HiFi-GAN discriminators (multi-period + multi-scale) and the
mel-spectrogram loss used by RVC/Applio, then runs a few optimizer steps.
Exit code 0 = every check passed.
"""
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm, weight_norm


class DiscriminatorP(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        chans = [1, 32, 128, 512, 1024, 1024]
        self.convs = nn.ModuleList(
            weight_norm(nn.Conv2d(chans[i], chans[i + 1], (5, 1), (3 if i < 4 else 1, 1), padding=(2, 0)))
            for i in range(5)
        )
        self.post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        b, c, t = x.shape
        if t % self.period:
            pad = self.period - t % self.period
            x = F.pad(x, (0, pad), "reflect")
            t += pad
        x = x.view(b, c, t // self.period, self.period)
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
        return self.post(x).flatten(1)


class DiscriminatorS(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1, 16, 15, 1, padding=7)),
            norm(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.post = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
        return self.post(x).flatten(1)


class MultiDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discs = nn.ModuleList(
            [DiscriminatorS(spectral_norm)] + [DiscriminatorP(p) for p in (2, 3, 5, 7, 11, 17, 23, 37)]
        )

    def forward(self, x):
        return [d(x) for d in self.discs]


def mel_loss(y_hat, y, device):
    window = torch.hann_window(1024, device=device)
    spec = lambda a: torch.stft(a.squeeze(1), 1024, 256, 1024, window, return_complex=True).abs()
    return F.l1_loss(torch.log(spec(y_hat) + 1e-5), torch.log(spec(y) + 1e-5))


def check(name, fn):
    start = time.perf_counter()
    try:
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"PASS  {name}  ({time.perf_counter() - start:.1f}s)")
        return True
    except Exception as e:
        print(f"FAIL  {name}: {type(e).__name__}: {e}")
        return False


def main():
    if not torch.cuda.is_available():
        print("FAIL  torch.cuda.is_available() is False - PyTorch cannot see the GPU")
        return 1
    device = torch.device("cuda")
    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")

    disc = MultiDiscriminator().to(device)
    gen = nn.Sequential(nn.Conv1d(1, 64, 7, padding=3), nn.Tanh(), nn.Conv1d(64, 1, 7, padding=3)).to(device)
    opt_d = torch.optim.AdamW(disc.parameters(), 2e-4, betas=(0.8, 0.99))
    opt_g = torch.optim.AdamW(gen.parameters(), 2e-4, betas=(0.8, 0.99))
    real = torch.randn(4, 1, 12800, device=device)  # 4 x 0.32 s slices at 40 kHz, like RVC segments

    def step(amp):
        with torch.autocast("cuda", enabled=amp):
            fake = gen(real)
            d_loss = sum(torch.mean((1 - r) ** 2) + torch.mean(f ** 2)
                         for r, f in zip(disc(real), disc(fake.detach())))
        opt_d.zero_grad()
        d_loss.backward()  # the backward pass that crashed on DirectML
        opt_d.step()
        with torch.autocast("cuda", enabled=amp):
            fake = gen(real)
            g_loss = sum(torch.mean((1 - f) ** 2) for f in disc(fake))
        g_loss = g_loss + 45 * mel_loss(fake.float(), real, device)
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
        if not (torch.isfinite(d_loss) and torch.isfinite(g_loss)):
            raise RuntimeError(f"non-finite loss d={d_loss.item()} g={g_loss.item()}")

    results = [
        check("STFT mel loss forward/backward",
              lambda: mel_loss(real.clone().requires_grad_(), real, device).backward()),
        check("discriminator backward, fp32 x5", lambda: [step(False) for _ in range(5)]),
        check("discriminator backward, fp16 autocast x5", lambda: [step(True) for _ in range(5)]),
    ]
    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
