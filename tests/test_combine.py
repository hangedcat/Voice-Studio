import numpy as np
import soundfile as sf

from voice_studio.pipeline import combine


def tone(path, seconds, amp, sr):
    t = np.arange(int(seconds * sr)) / sr
    sf.write(path, amp * np.sin(2 * np.pi * 440 * t), sr)


def test_mix_applies_gains_and_runs_as_long_as_the_longer_file(tmp_path):
    tone(tmp_path / "a.wav", 2.0, 0.2, 44100)
    tone(tmp_path / "b.wav", 1.0, 0.2, 40000)  # different rate, like Applio's AI Vocals
    out = combine(tmp_path / "a.wav", tmp_path / "b.wav", tmp_path / "mix.wav", (1.0, 0.5))
    mix, sr = sf.read(out)
    assert sr == 44100 and abs(len(mix) / sr - 2.0) < 0.05
    assert abs(np.abs(mix[sr // 4: sr // 2]).max() - 0.3) < 0.02  # 0.2 + 0.5 * 0.2, in phase
    assert abs(np.abs(mix[int(1.5 * sr):]).max() - 0.2) < 0.02    # only A after B ends


def test_loud_sum_is_limited_not_clipped(tmp_path):
    tone(tmp_path / "a.wav", 1.0, 0.9, 44100)
    tone(tmp_path / "b.wav", 1.0, 0.9, 44100)
    mix, _ = sf.read(combine(tmp_path / "a.wav", tmp_path / "b.wav", tmp_path / "mix.wav"))
    assert np.abs(mix).max() <= 0.99
