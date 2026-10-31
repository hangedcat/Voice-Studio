# Voice Studio: Handoff

## Goal

A Windows desktop app that runs the AI vocal backing (垫音) workflow from `D:\Project\AI_Singing_Voice_Conversion_Guide.md` entirely on the local PC (RX 7900 XTX), with no Colab. It turns a song into `instrumental.wav` + `ai_vocals.wav` in the user's voice, and trains new voice models. Terms are defined in [CONTEXT.md](CONTEXT.md), and decisions are in [docs/adr/](docs/adr/).

v1 scope (user's choice): a Cover queue (batch), Cleanup (de-reverb), pitch auto-suggest, Preview Mix and a Train tab, in a native PySide6 window.

## Current progress (2026-10-31)

Steps 1–4 of the plan are done and verified. Step 5 (launcher, review, handoff) is done too.

| Piece | File | Verified |
|---|---|---|
| ROCm GPU training | `checks/gpu_smoke_test.py`, ADR 0001 | HiFi-GAN discriminator backward in fp32 and fp16 on the 7900 XTX |
| Cover pipeline | `voice_studio/pipeline.py` | `song.wav` → stems in 103–158 s, pitch auto −12 |
| Training | `voice_studio/training.py` | 200 epochs in ~2 h; one Voice Model exported every 20 epochs |
| Desktop window | `voice_studio/app.py`, `Voice Studio.bat` | full Cover Job run by pressing the real button; preview loads; cancel-on-quit kills the process tree |
| Separate tab | `separate_song()` in `pipeline.py`, `SeparateTab` in `app.py` | Song → `vocals.wav` + `instrumental.wav` (+ optional `vocals_no_reverb.wav`) in 85–90 s, via both the GUI and the CLI |
| Breath Restoration | `voice_studio/breath.py`, Cover option "Restore original breaths" / CLI `--no-breaths`, `--breath-level` | **on by default** (v2, level 1.0; the user approved it by ear); A/B files in `output/<song>/compare/` (`breaths_off`, `breaths_v2`, `breaths_v2_quiet`, `mix_*`); `tests/test_breath.py` |
| Combine tab | `combine()` in `pipeline.py` (ffmpeg amix + limiter), `CombineTab` in `app.py` | exports at the Preview Mix fader levels to WAV/MP3/FLAC; `tests/test_combine.py` checks gains, length, sample-rate mismatch, no clipping |
| CLI | `python -m voice_studio cover SONG --voice NAME` / `separate SONG [--clean]` / `combine A B --out MIX [--gains 1.0 0.7]` / `train DATASET --name NAME` | all four |
| Tests | `tests/test_pipeline.py` (pitch suggestion) | 4 passed |

Voice Models live in `models/voices/<name>/` (`model.pth`, `model.index`, `voice.json` holding the median pitch). The current best candidates are `my-voice-v2-{100,140,180,200}e`, trained from the normalized `datasets/my-voice/test1.wav`. A comparison render of each is in `output/song/compare/`. **The user hasn't picked a winner yet.**

Conversion defaults (CoverSettings), chosen by the user by ear ("variant G" + Breath Restoration at level 1.0): Cleanup on (MelBand Roformer de-reverb), f0 `fcpe`, protect 0.15, index rate 0.5, pitch auto (whole octaves).

## What worked

- AMD's official Windows ROCm wheels (ROCm 7.2.1, torch 2.9.1, Python 3.12) train RVC fine. The first run is ~8 s/step while MIOpen tunes kernels; after that it's ~0.8 s/step.
- Driving pinned Applio (`vendor/Applio` @ `59c1131`) through `core.py` subprocesses.
- Using audio-separator for both Separation (BS-Roformer `model_bs_roformer_ep_317_sdr_12.9755`) and Cleanup, with `--custom_output_names` (plus `--single_stem` when only one stem is wanted).
- Rebuilding a lost `.index` without retraining: preprocess, then extract, then `core.py index`.
- Breath cracks: an index + lower protect + FCPE + de-reverb each helped. More epochs helped further (the user confirmed 200e > 60e).

## What didn't work (don't repeat)

- **Never let pip replace ROCm torch.** Always install with `-c constraints-rocm.txt`. Applio's requirements pin CUDA torch and NVIDIA libraries, so `requirements-applio.txt` is a filtered copy of them.
- **Demucs** fails to save under ROCm torchaudio 2.9 (it needs `torchcodec`). That's why Separation moved to audio-separator.
- **Applio `--save-every-weights`** is a click flag with `default=True`, so *passing* it turns it OFF. Leave it out.
- **Applio preprocessing defaults to no normalization.** The quiet dataset (peak 0.091) gave a model ~22 dB too quiet. Use `--normalization-mode pre`, not `post`: per-slice normalization boosts breath-only slices.
- **Applio `hybrid[...]` f0 methods** crash at this commit (not implemented in the pipeline). Don't offer them.
- **Don't feed songs straight to audio-separator.** It reads files with soundfile, which silently stops partway through some MP3s (a 33 s Douyin download came out as 15 s). `decode()` converts every song to WAV with ffmpeg first.
- **Reusing a training name** makes Applio silently resume the old checkpoint. `train_voice()` refuses existing names.

## Rebuild the environment

```powershell
cd D:\Project\voice-studio
uv venv --python 3.12 .venv
# AMD ROCm SDK + torch wheels (7.2.1): steps 1-2 of
# https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html
uv pip install --python .venv\Scripts\python.exe -c constraints-rocm.txt -r requirements-applio.txt audio-separator PySide6 pytest
cd vendor\Applio; ..\..\.venv\Scripts\python.exe core.py prerequisites --pretraineds-hifigan --models --no-exe
```
This also needs graphics driver ≥ 26.2.2 and ffmpeg on PATH (installed via winget).

## Next steps

1. **The user picks the best epoch** from `output/song/compare/v2_{100,140,180,200}e.wav`. Then delete the other `models/voices/my-voice-v2-*` folders and the old `my-voice-60e` / `my-voice-200e` (the quiet one) to keep the model list short.
2. **Breath Restoration v2 is on by default** (the user preferred level 1.0 over 0.5). v1 (sung-level gain, s/sh included, 15 ms fades) sounded "really unnatural": pasted breaths came in 3-11 dB louder and much brighter than the AI Vocals around them. v2 replaces only breaths, meaning an FCPE pitch gap of 80 ms-1.5 s with 20-97% of its energy above 2 kHz (s/sh sit at 98-100%, singing at ~4%). It EQs the source toward the AI's 1/3-octave tone, matches the AI's local level (capped at the sung-level ratio, so loud cracks aren't copied) and fades over 30 ms. If it's still unnatural, the next idea is to drop the pasting and just turn the AI down in those gaps (duck the crackle).
3. **Possible improvements, not requested:** a "delete model" button, per-song output-name collisions (two different songs with the same file name share `output/<name>`), streaming per-step progress inside a Cover Job.
4. **voice-studio isn't committed** to git. Ask the user before committing (it would also need a `.gitignore` for `.venv`, `vendor`, `models`, `output`, `datasets`).
