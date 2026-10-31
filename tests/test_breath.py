import numpy as np

from voice_studio.breath import FADE, SR, find_breaths, restore

FRAME = 0.01  # s per pitch frame, like FCPE
GAP = (1.0, 1.3)
S, E = int(GAP[0] * SR), int(GAP[1] * SR)
rng = np.random.default_rng(0)


def voiced_with_gap(seconds: float) -> np.ndarray:
    voiced = np.ones(int(seconds / FRAME), bool)
    voiced[int(GAP[0] / FRAME):int(GAP[1] / FRAME)] = False
    return voiced


def tone(hz, seconds, amp):
    return amp * np.sin(2 * np.pi * hz * np.arange(int(seconds * SR)) / SR)


def high_passed_noise(n, cutoff_hz):
    spec = np.fft.rfft(rng.normal(0, 1, n))
    spec[np.fft.rfftfreq(n, 1 / SR) < cutoff_hz] = 0
    return np.fft.irfft(spec, n)


def rms(x):
    return np.sqrt(np.mean(x ** 2))


def test_only_breath_sized_gaps_count():
    voiced = np.ones(400, bool)
    voiced[50:53] = False    # 30 ms: a consonant inside a word
    voiced[100:120] = False  # 200 ms: a breath
    voiced[200:380] = False  # 1.8 s: probably a lost sung note
    assert [(round(b.start, 2), round(b.end, 2)) for b in find_breaths(voiced, FRAME)] == [(1.0, 1.2)]


def test_breath_replaces_crackle_at_the_ai_level():
    src = tone(440, 2, 0.4)
    src[S:E] = rng.normal(0, 0.2, E - S)          # a loud breath (white noise)
    crackle = tone(1000, 2, 0.05)[: E - S]        # what RVC made of it
    ai = tone(220, 2, 0.3)
    ai[S:E] = crackle

    out, breaths = restore(ai, src, voiced_with_gap(2), FRAME)
    assert len(breaths) == 1
    mid = slice(S + int(FADE * SR), E - int(FADE * SR))
    assert abs(20 * np.log10(rms(out[mid]) / rms(crackle))) < 1.5      # no loudness jump
    assert abs(np.corrcoef(out[mid], ai[mid])[0, 1]) < 0.5             # crackle replaced
    fade = int(FADE * SR) + 1
    np.testing.assert_array_equal(out[: S - fade], ai[: S - fade])     # singing untouched
    np.testing.assert_array_equal(out[E + fade:], ai[E + fade:])


def test_a_loud_crack_is_not_matched():
    src = tone(440, 2, 0.4)
    src[S:E] = rng.normal(0, 0.02, E - S)  # a quiet breath, ~-23 dB under the singing
    ai = tone(220, 2, 0.4)
    ai[S:E] = tone(1000, 2, 0.4)[: E - S]  # RVC cracked it as loud as the singing
    out, _ = restore(ai, src, voiced_with_gap(2), FRAME)
    mid = slice(S + int(FADE * SR), E - int(FADE * SR))
    assert rms(out[mid]) < rms(ai[mid]) / 5  # at least 14 dB quieter than the crack


def test_s_sounds_are_left_to_rvc():
    src = tone(440, 2, 0.4)
    src[S:E] = high_passed_noise(E - S, 4000)  # all energy up high: an s/sh
    ai = tone(220, 2, 0.3)
    out, breaths = restore(ai, src, voiced_with_gap(2), FRAME)
    assert breaths == []
    np.testing.assert_array_equal(out, ai)


def test_tonal_gap_is_left_alone():
    """A pitchless stretch that sounds like singing is a lost note, not a breath."""
    src = tone(300, 2, 0.4)  # the tracker said "no pitch", but it's a 300 Hz note
    ai = tone(150, 2, 0.4)
    out, breaths = restore(ai, src, voiced_with_gap(2), FRAME)
    assert breaths == []
    np.testing.assert_array_equal(out, ai)
