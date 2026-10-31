"""Breath Restoration: put the source singer's breaths back into AI Vocals.

RVC has no pitch to follow on breaths, so it turns them into crackle. Wherever the
Clean Acapella has no pitch for a breath-sized stretch that sounds like a breath,
crossfade from the AI Vocals to the Clean Acapella's breath, made to fit the AI Vocals:
  - EQ'd toward the AI Vocals' tone (the source singer's breaths are much brighter), and
  - scaled to the AI Vocals' own level at that spot, so loudness doesn't jump.
s/sh sounds inside words are left to RVC: swapping them mid-word is audible as a seam.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 44100
MIN_BREATH = 0.08  # s; shorter pitch gaps are consonants inside words, left to RVC's protect
MAX_BREATH = 1.5   # s; a longer "breath" is more likely a sung note the pitch tracker lost
FADE = 0.03        # s; crossfade on each side of a breath
# Share of a segment's energy above 2 kHz, measured on real Clean Acapellas: sung frames ~0.04,
# breaths 0.33-0.96, s/sh sounds 0.98-1.00. Below MIN it's likely a lost sung note; above MAX an s/sh.
MIN_NOISE_SHARE = 0.2
MAX_NOISE_SHARE = 0.97
N_FFT, HOP = 2048, 512


@dataclass
class Breath:
    start: float  # s
    end: float


def find_breaths(voiced: np.ndarray, frame: float) -> list[Breath]:
    """Runs of unvoiced frames between MIN_BREATH and MAX_BREATH long."""
    edges = np.flatnonzero(np.diff(np.r_[1, voiced.astype(int), 1]))  # starts/ends of unvoiced runs
    runs = [Breath(s * frame, e * frame) for s, e in zip(edges[::2], edges[1::2])]
    return [b for b in runs if MIN_BREATH <= b.end - b.start <= MAX_BREATH]


def high_share(segment: np.ndarray, sr: int = SR) -> float:
    """Fraction of a segment's energy above 2 kHz."""
    power = np.abs(np.fft.rfft(segment)) ** 2
    freqs = np.fft.rfftfreq(len(segment), 1 / sr)
    return float(power[freqs > 2000].sum() / (power.sum() + 1e-12))


def ramp_window(s: int, e: int, n: int, sr: int = SR) -> np.ndarray:
    """1 on [s, e), ramping to 0 over FADE on each side."""
    w = np.zeros(n)
    fade = int(FADE * sr)
    w[s:e] = 1
    a, z = max(0, s - fade), min(n, e + fade)
    w[a:s] = np.linspace(0, 1, fade, endpoint=False)[fade - (s - a):]
    w[e:z] = np.linspace(1, 0, fade + 1)[1:][:z - e]
    return w


def match_tone(src: np.ndarray, ai: np.ndarray, spans: list[tuple[int, int]], sr: int = SR) -> np.ndarray:
    """EQ src so its average 1/3-octave spectrum over spans matches ai's there (cut <= 24 dB, boost <= 6 dB)."""
    import librosa

    S = librosa.stft(src, n_fft=N_FFT, hop_length=HOP)
    A = librosa.stft(ai, n_fft=N_FFT, hop_length=HOP)
    frames = np.zeros(S.shape[1], bool)
    for s, e in spans:
        frames[s // HOP:e // HOP + 1] = True
    p_src = (np.abs(S[:, frames]) ** 2).mean(1)
    p_ai = (np.abs(A[:, frames]) ** 2).mean(1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    edges = 100 * 2 ** np.arange(0, np.log2(sr / 2 / 100) + 1 / 3, 1 / 3)
    centers, db = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = (freqs >= lo) & (freqs < hi)
        if band.any():
            centers.append(np.sqrt(lo * hi))
            db.append(10 * np.log10((p_ai[band].sum() + 1e-12) / (p_src[band].sum() + 1e-12)))
    curve = np.interp(np.log(np.maximum(freqs, 1)), np.log(centers), np.clip(db, -24, 6))
    return librosa.istft(S * 10 ** (curve[:, None] / 20), hop_length=HOP, length=len(src))


def restore(ai: np.ndarray, src: np.ndarray, voiced: np.ndarray, frame: float, level: float = 1.0,
            sr: int = SR) -> tuple[np.ndarray, list[Breath]]:
    """Mono AI Vocals and source at the same rate and alignment -> AI Vocals with the source's breaths."""
    n = len(ai)
    src = np.pad(src, (0, max(0, n - len(src))))[:n]
    span = lambda b: (int(b.start * sr), min(n, int(b.end * sr)))
    breaths = [b for b in find_breaths(voiced, frame)
               if MIN_NOISE_SHARE <= high_share(src[slice(*span(b))], sr) < MAX_NOISE_SHARE]
    if not breaths:
        return ai.copy(), []
    spans = [span(b) for b in breaths]
    rms = lambda x: np.sqrt(np.mean(x ** 2)) + 1e-9
    frame_of = np.minimum((np.arange(n) / sr / frame).astype(int), len(voiced) - 1)
    sung = voiced[frame_of]
    sung_ratio = rms(ai[sung]) / rms(src[sung])  # AI singing level / source singing level
    src = match_tone(src, ai, spans, sr)
    out = ai.copy()
    for s, e in spans:
        w = ramp_window(s, e, n, sr)
        # The AI's own level there, unless that's a loud crack: then no louder than the source
        # breath is relative to the source singing.
        gain = level * min(rms(ai[s:e]) / rms(src[s:e]), sung_ratio)
        out = out * (1 - w) + gain * src * w
    return out, breaths


def restore_breaths(ai_vocals: Path, clean_acapella: Path, out: Path, level: float = 1.0) -> list[Breath]:
    """File version of restore(): out is a 44.1 kHz mono WAV."""
    import librosa
    import soundfile as sf

    from .pipeline import pitch_track

    ai, _ = librosa.load(ai_vocals, sr=SR, mono=True)
    src, _ = librosa.load(clean_acapella, sr=SR, mono=True)
    f0, frame = pitch_track(clean_acapella)
    mixed, breaths = restore(ai, src, f0 > 0, frame, level)
    sf.write(out, np.clip(mixed, -1, 1), SR, subtype="PCM_16")
    return breaths
