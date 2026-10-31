"""Training Job: Voice Dataset -> preprocess -> extract -> index -> train -> Voice Models.

Every saved epoch is registered as its own Voice Model (models/voices/<name>-<N>e)
so the user can pick the best-sounding one by ear.
"""
import json
import re
import shutil
from pathlib import Path

import numpy as np

from .pipeline import APPLIO, PYTHON, VOICES_DIR, StepFailed, _run, finish, start

SAMPLE_RATE = 40000
EPOCH_LINE = re.compile(r"epoch=(\d+) \| step=(\d+)")


def _core(*args: str) -> list[str]:
    return [PYTHON, "core.py", *args]


def train_voice(name: str, dataset: Path, log: Path, epochs: int = 200, save_every: int = 20,
                batch_size: int = 8, on_stage=lambda stage: None, on_epoch=lambda epoch, total: None
                ) -> list[Path]:
    """Run one Training Job and return the Voice Model folders it created, oldest epoch first."""
    if (APPLIO / "logs" / name).exists():
        # Applio would silently resume that run's checkpoint instead of training a new model
        raise StepFailed(f"a training run named '{name}' already exists; pick a new name")
    on_stage("Preprocess")
    # "pre" normalizes the whole recording once: quiet datasets otherwise train a quiet model,
    # and per-slice "post" would boost breath-only slices to full volume.
    _run("Preprocess", _core("preprocess", "--model-name", name, "--dataset-path", str(dataset.resolve()),
                             "--sample-rate", str(SAMPLE_RATE), "--cpu-cores", "8",
                             "--normalization-mode", "pre"), cwd=APPLIO)
    on_stage("Feature extraction")
    _run("Feature extraction", _core("extract", "--model-name", name, "--f0-method", "rmvpe", "--gpu", "0",
                                     "--sample-rate", str(SAMPLE_RATE), "--embedder-model", "contentvec",
                                     "--cpu-cores", "8"), cwd=APPLIO)
    on_stage("Index")
    _run("Index", _core("index", "--model-name", name, "--index-algorithm", "Auto"), cwd=APPLIO)

    on_stage("Training")
    # No --save-every-weights: it defaults to on, and passing the flag turns it OFF (click quirk).
    cmd = _core("train", "--model-name", name, "--vocoder", "HiFi-GAN", "--total-epoch", str(epochs),
                "--save-every-epoch", str(save_every), "--save-only-latest", "--sample-rate", str(SAMPLE_RATE),
                "--batch-size", str(batch_size), "--gpu", "0", "--pretrained", "--cache-data-in-gpu",
                "--index-algorithm", "Auto")
    proc = start(cmd, cwd=APPLIO)
    try:
        with open(log, "w", encoding="utf-8") as f:
            for line in proc.stdout:
                f.write(line)
                f.flush()
                if m := EPOCH_LINE.search(line):
                    on_epoch(int(m.group(1)), epochs)
        proc.wait()
    finally:
        finish(proc)
    if proc.returncode != 0:
        raise StepFailed(f"Training failed (exit {proc.returncode}); see {log}")

    on_stage("Export")
    return export_voice_models(name)


def export_voice_models(name: str) -> list[Path]:
    """Register every exported epoch of a training run as a Voice Model folder."""
    logs = APPLIO / "logs" / name
    index = logs / f"{name}.index"
    f0 = np.concatenate([np.load(p) for p in (logs / "f0_voiced").glob("*.npy")])
    meta = json.dumps({"median_hz": round(float(np.median(f0[f0 > 0])), 1)})

    pattern = re.compile(rf"{re.escape(name)}_(\d+)e_\d+s\.pth")
    weights = sorted((int(m.group(1)), p) for p in logs.glob("*.pth") if (m := pattern.fullmatch(p.name)))
    if not weights:
        raise StepFailed(f"Export: no trained weights found in {logs}")
    folders = []
    for epoch, pth in weights:
        folder = VOICES_DIR / f"{name}-{epoch}e"
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pth, folder / "model.pth")
        shutil.copy2(index, folder / "model.index")
        (folder / "voice.json").write_text(meta)
        folders.append(folder)
    return folders
