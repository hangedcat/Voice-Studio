from voice_studio.pipeline import suggest_pitch

VOICE = 176.3  # my-voice median pitch


def test_female_song_drops_an_octave():
    assert suggest_pitch(380.0, VOICE) == -12


def test_song_in_voice_range_is_unchanged():
    assert suggest_pitch(VOICE, VOICE) == 0


def test_low_song_goes_up_an_octave():
    assert suggest_pitch(88.0, VOICE) == 12


def test_result_is_always_whole_octaves_within_range():
    for hz in (40, 60, 120, 260, 440, 900, 2000):
        shift = suggest_pitch(hz, VOICE)
        assert shift % 12 == 0 and -24 <= shift <= 24
