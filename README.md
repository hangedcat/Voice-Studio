# Voice Studio

A Windows desktop app for making AI singing covers locally, with no Colab: split a song into vocals and music, re-sing the vocals in your own trained voice, and mix the result. It also trains new voice models on your own GPU.

Built for an AMD RX 7900 XTX using AMD's native Windows ROCm PyTorch. It drives [Applio](https://github.com/IAHispano/Applio) (RVC) and [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) under the hood.

## Tabs

| Tab | What it does |
|---|---|
| **Cover** | A queue of songs → `instrumental.wav` + `ai_vocals.wav`. Steps: Separation (BS-Roformer), Cleanup (de-reverb), pitch auto-suggest in whole octaves, Conversion (RVC), Breath Restoration. Includes a two-fader Preview Mix. |
| **Separate** | A queue of songs → `vocals.wav` + `instrumental.wav`, with an optional no-reverb copy of the vocals. |
| **Combine** | Mixes two audio files at the levels you set by ear, then exports WAV, MP3 or FLAC. |
| **Train** | A folder of your recordings → one voice model every N epochs, so you can pick the best one by ear. |

Everything lands in `output/<song name>/`, and voice models in `models/voices/<name>/`.

## Setup

Requirements: Windows 11, an AMD Radeon RX 7000-series GPU with graphics driver ≥ 26.2.2, Python 3.12, [uv](https://docs.astral.sh/uv/), and ffmpeg on PATH (`winget install ffmpeg`).

```powershell
uv venv --python 3.12 .venv
# Install AMD's ROCm SDK + torch wheels (ROCm 7.2.1), steps 1-2 of:
# https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html
uv pip install --python .venv\Scripts\python.exe -c constraints-rocm.txt -r requirements-applio.txt audio-separator PySide6 pytest

git clone https://github.com/IAHispano/Applio vendor\Applio
git -C vendor\Applio checkout 59c113106c0dfff647324ab1a80c286f17b51929
cd vendor\Applio; ..\..\.venv\Scripts\python.exe core.py prerequisites --pretraineds-hifigan --models --no-exe
```

Always install with `-c constraints-rocm.txt`, so pip can't swap ROCm torch for a CUDA build. To confirm the GPU can train, run `.venv\Scripts\python.exe checks\gpu_smoke_test.py`.

## Use

Double-click `Voice Studio.bat`. Or use the command line:

```powershell
python -m voice_studio cover song.mp3 --voice my-voice-v2-200e
python -m voice_studio separate song.mp3 --clean
python -m voice_studio combine instrumental.wav ai_vocals.wav --out mix.mp3 --gains 1.0 0.7
python -m voice_studio train datasets\my-voice --name my-voice-v3
```

Tests: `python -m pytest`

## Docs

- [CONTEXT.md](CONTEXT.md): the project's vocabulary (Acapella, Clean Acapella, Cover Job, ...).
- [docs/adr/](docs/adr/): the reasons behind native ROCm and driving Applio headless.
- [HANDOFF.md](HANDOFF.md): current state, what worked, pitfalls to avoid, next steps.
