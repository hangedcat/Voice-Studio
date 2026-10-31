"""Cover Job pipeline: Song -> Separation -> Cleanup -> Conversion -> Instrumental + AI Vocals.

Heavy steps run as subprocesses (see docs/adr/0002) so GPU memory is released
between steps and a crash in one tool can't take the app down.
"""
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APPLIO = ROOT / "vendor" / "Applio"
VOICES_DIR = ROOT / "models" / "voices"
SEPARATOR_MODELS = ROOT / "models" / "separator"
PYTHON = sys.executable

SEPARATION_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
DEREVERB_MODEL = "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"


@dataclass
class VoiceModel:
    name: str
    pth: Path
    index: Path | None
    median_hz: float | None  # the speaker's median sung/spoken pitch, for pitch suggestion

    @classmethod
    def load(cls, folder: Path) -> "VoiceModel":
        meta = folder / "voice.json"
        median = json.loads(meta.read_text())["median_hz"] if meta.exists() else None
        index = folder / "model.index"
        return cls(folder.name, folder / "model.pth", index if index.exists() else None, median)


def list_voice_models() -> list[VoiceModel]:
    if not VOICES_DIR.exists():
        return []
    return [VoiceModel.load(d) for d in sorted(VOICES_DIR.iterdir()) if (d / "model.pth").exists()]


@dataclass
class CoverSettings:
    """Defaults are the "G" settings the user picked by ear on 2026-10-30."""
    pitch: int | None = None  # semitones; None = use suggest_pitch()
    cleanup: bool = True
    f0_method: str = "fcpe"
    protect: float = 0.15
    index_rate: float = 0.5
    breaths: bool = True       # Breath Restoration (see breath.py); v2 at level 1.0 approved by ear 2026-10-31
    breath_level: float = 1.0  # restored breaths' volume relative to the AI Vocals' level


@dataclass
class CoverResult:
    instrumental: Path
    ai_vocals: Path
    acapella: Path
    clean_acapella: Path | None
    pitch: int


class StepFailed(RuntimeError):
    pass


_running: list[subprocess.Popen] = []


def start(cmd: list[str], cwd: Path | None = None) -> subprocess.Popen:
    """Start a step's subprocess and register it so cancel_running() can stop it."""
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    _running.append(proc)
    return proc


def finish(proc: subprocess.Popen) -> None:
    if proc in _running:
        _running.remove(proc)


def cancel_running() -> None:
    """Kill every running step, including the child processes Applio training spawns."""
    for proc in list(_running):
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)


def _run(step: str, cmd: list[str], cwd: Path | None = None) -> None:
    proc = start(cmd, cwd)
    try:
        output, _ = proc.communicate()
    finally:
        finish(proc)
    if proc.returncode != 0:
        tail = "\n".join(output.strip().splitlines()[-15:])
        raise StepFailed(f"{step} failed (exit {proc.returncode}):\n{tail}")


def _audio_separator(step: str, audio: Path, model: str, work: Path, names: dict[str, str]) -> list[Path]:
    """Run audio-separator, writing each wanted stem to work/<name>.wav."""
    exe = Path(PYTHON).with_name("audio-separator.exe")
    cmd = [str(exe), str(audio), "-m", model, "--model_file_dir", str(SEPARATOR_MODELS),
           "--output_dir", str(work), "--output_format", "WAV", "--custom_output_names", json.dumps(names)]
    if len(names) == 1:
        cmd += ["--single_stem", next(iter(names))]
    _run(step, cmd)
    paths = [work / f"{name}.wav" for name in names.values()]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise StepFailed(f"{step}: audio-separator did not write {missing}")
    return paths


def decode(song: Path, work: Path) -> Path:
    """Decode any input to work/input.wav with ffmpeg.

    audio-separator reads files with soundfile, which silently stops partway through some
    MP3s (a 33 s Douyin download came out as 15 s); ffmpeg reads them whole.
    """
    work.mkdir(parents=True, exist_ok=True)
    wav = work / "input.wav"
    _run("Decode", ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(song),
                    "-vn", "-c:a", "pcm_f32le", str(wav)])
    return wav


def separate(song: Path, work: Path) -> tuple[Path, Path]:
    """Separation: split a Song into its Acapella and Instrumental."""
    song = decode(song, work)
    acapella, instrumental = _audio_separator("Separation", song, SEPARATION_MODEL, work,
                                              {"Vocals": "acapella", "Instrumental": "instrumental"})
    return acapella, instrumental


def cleanup(acapella: Path, work: Path, name: str = "clean_acapella") -> Path:
    """Cleanup: remove reverb from an Acapella so breaths aren't converted into crackle."""
    (clean,) = _audio_separator("Cleanup", acapella, DEREVERB_MODEL, work, {"noreverb": name})
    return clean


@dataclass
class SeparationResult:
    instrumental: Path
    acapella: Path
    clean_acapella: Path | None


def separate_song(song: Path, out_dir: Path, clean: bool = False, on_stage=lambda stage: None) -> SeparationResult:
    """Separation Job: a Song -> out_dir/vocals.wav + instrumental.wav (+ vocals_no_reverb.wav)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    on_stage("Separation")
    song = decode(song, out_dir / "work")
    acapella, instrumental = _audio_separator("Separation", song, SEPARATION_MODEL, out_dir,
                                              {"Vocals": "vocals", "Instrumental": "instrumental"})
    clean_acapella = None
    if clean:
        on_stage("Cleanup")
        clean_acapella = cleanup(acapella, out_dir, "vocals_no_reverb")
    return SeparationResult(instrumental, acapella, clean_acapella)


def combine(first: Path, second: Path, out: Path, gains: tuple[float, float] = (1.0, 1.0)) -> Path:
    """Mix two audio files at linear gains into out (format from its extension).

    Runs as long as the longer file; a limiter keeps the sum from clipping.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    graph = (f"[0:a]volume={gains[0]:.3f}[a];[1:a]volume={gains[1]:.3f}[b];"
             "[a][b]amix=inputs=2:duration=longest:normalize=0,alimiter=limit=0.98:level=0")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(first), "-i", str(second),
           "-filter_complex", graph, "-ar", "44100"]
    if out.suffix.lower() == ".mp3":
        cmd += ["-b:a", "320k"]
    _run("Combine", cmd + [str(out)])
    return out


def pitch_track(audio: Path):
    """FCPE pitch of a vocal recording: (f0 in Hz per frame, 0 = no pitch; frame length in seconds)."""
    import librosa
    import torch
    from torchfcpe import spawn_bundled_infer_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wav, sr = librosa.load(audio, sr=16000, mono=True)
    model = spawn_bundled_infer_model(device=device)
    f0 = model.infer(torch.from_numpy(wav).float()[None, :, None].to(device), sr=sr,
                     f0_min=50, f0_max=1100).squeeze().cpu().numpy()
    return f0, len(wav) / sr / len(f0)


def median_pitch_hz(audio: Path) -> float:
    """Median pitch of the voiced frames of a vocal recording, via FCPE."""
    f0, _ = pitch_track(audio)
    voiced = f0[f0 > 0]
    if voiced.size == 0:
        raise StepFailed(f"no sung pitch found in {audio.name}")
    return float(sorted(voiced)[voiced.size // 2])


def suggest_pitch(song_median_hz: float, voice_median_hz: float) -> int:
    """Pitch Shift that moves the song toward the voice's range, in whole octaves so it stays in key."""
    semitones = 12 * math.log2(voice_median_hz / song_median_hz)
    return max(-24, min(24, 12 * round(semitones / 12)))


def convert(clean_acapella: Path, voice: VoiceModel, settings: CoverSettings, pitch: int, out: Path) -> Path:
    """Conversion: re-sing an acapella with a Voice Model through Applio."""
    out = out.resolve()  # Applio runs from its own folder, so relative paths would break
    _run("Conversion", [PYTHON, "core.py", "infer",
                        "--input-path", str(clean_acapella.resolve()), "--output-path", str(out),
                        "--pth-path", str(voice.pth), "--index-path", str(voice.index or ""),
                        "--pitch", str(pitch), "--index-rate", str(settings.index_rate if voice.index else 0),
                        "--protect", str(settings.protect), "--f0-method", settings.f0_method,
                        "--export-format", "WAV"], cwd=APPLIO)
    return out


def cover(song: Path, voice: VoiceModel, out_dir: Path, settings: CoverSettings | None = None,
          on_stage=lambda stage: None) -> CoverResult:
    """Run one Cover Job. on_stage(name) is called as each step starts."""
    settings = settings or CoverSettings()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "work"
    work.mkdir(exist_ok=True)

    on_stage("Separation")
    acapella, instrumental = separate(song, work)

    clean = None
    if settings.cleanup:
        on_stage("Cleanup")
        clean = cleanup(acapella, work)

    pitch = settings.pitch
    if pitch is None:
        if voice.median_hz is None:
            raise StepFailed(f"voice model '{voice.name}' has no voice.json; set the pitch by hand")
        on_stage("Pitch suggestion")
        pitch = suggest_pitch(median_pitch_hz(acapella), voice.median_hz)

    on_stage("Conversion")
    ai_vocals = out_dir / "ai_vocals.wav"
    if settings.breaths:
        from .breath import restore_breaths

        raw = convert(clean or acapella, voice, settings, pitch, work / "ai_vocals_raw.wav")
        on_stage("Breath restoration")
        restore_breaths(raw, clean or acapella, ai_vocals, settings.breath_level)
    else:
        convert(clean or acapella, voice, settings, pitch, ai_vocals)

    final_instrumental = out_dir / "instrumental.wav"
    shutil.move(instrumental, final_instrumental)
    return CoverResult(final_instrumental, ai_vocals.resolve(), acapella, clean, pitch)
