"""Command line: python -m voice_studio cover SONG --voice NAME [--pitch N] [--no-cleanup]
                             separate SONG [--clean]
                             combine A B --out MIX [--gains 1.0 0.7]"""
import argparse
import sys
from pathlib import Path

from .pipeline import ROOT, CoverSettings, StepFailed, combine, cover, list_voice_models, separate_song


def main() -> int:
    parser = argparse.ArgumentParser(prog="voice_studio")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("cover", help="turn a Song into instrumental.wav + ai_vocals.wav")
    c.add_argument("song", type=Path)
    c.add_argument("--voice", required=True, help="folder name under models/voices")
    c.add_argument("--pitch", type=int, help="semitones; omit to auto-suggest")
    c.add_argument("--no-cleanup", action="store_true", help="skip de-reverb")
    c.add_argument("--no-breaths", action="store_true", help="skip Breath Restoration")
    c.add_argument("--breath-level", type=float, default=1.0, help="restored breaths' volume (default 1.0)")
    c.add_argument("--out", type=Path, help="default: output/<song name>")
    t = sub.add_parser("train", help="train Voice Models from a Voice Dataset folder")
    t.add_argument("dataset", type=Path)
    t.add_argument("--name", required=True)
    t.add_argument("--epochs", type=int, default=200)
    t.add_argument("--save-every", type=int, default=20)
    s = sub.add_parser("separate", help="split a Song into vocals.wav + instrumental.wav")
    s.add_argument("song", type=Path)
    s.add_argument("--clean", action="store_true", help="also write vocals_no_reverb.wav")
    s.add_argument("--out", type=Path, help="default: output/<song name>")
    m = sub.add_parser("combine", help="mix two audio files into one")
    m.add_argument("first", type=Path)
    m.add_argument("second", type=Path)
    m.add_argument("--out", type=Path, required=True, help=".wav, .mp3 or .flac")
    m.add_argument("--gains", type=float, nargs=2, default=(1.0, 1.0), metavar=("A", "B"),
                   help="linear volume of each file, e.g. 1.0 0.7")
    args = parser.parse_args()

    if args.command == "train":
        return train(args)
    if args.command == "separate":
        return separate(args)
    if args.command == "combine":
        try:
            print(combine(args.first, args.second, args.out, tuple(args.gains)))
        except StepFailed as e:
            print(e, file=sys.stderr)
            return 1
        return 0

    voices = {v.name: v for v in list_voice_models()}
    if args.voice not in voices:
        print(f"unknown voice '{args.voice}'; available: {', '.join(voices) or 'none'}", file=sys.stderr)
        return 2
    out = args.out or ROOT / "output" / args.song.stem
    settings = CoverSettings(pitch=args.pitch, cleanup=not args.no_cleanup, breaths=not args.no_breaths,
                             breath_level=args.breath_level)
    try:
        result = cover(args.song.resolve(), voices[args.voice], out, settings,
                       on_stage=lambda s: print(f"-> {s}", flush=True))
    except StepFailed as e:
        print(e, file=sys.stderr)
        return 1
    print(f"pitch {result.pitch:+d}\ninstrumental: {result.instrumental}\nai vocals:    {result.ai_vocals}")
    return 0


def separate(args) -> int:
    out = args.out or ROOT / "output" / args.song.stem
    try:
        result = separate_song(args.song.resolve(), out, args.clean, on_stage=lambda s: print(f"-> {s}", flush=True))
    except StepFailed as e:
        print(e, file=sys.stderr)
        return 1
    print(f"vocals:       {result.acapella}\ninstrumental: {result.instrumental}")
    if result.clean_acapella:
        print(f"no reverb:    {result.clean_acapella}")
    return 0


def train(args) -> int:
    from .training import train_voice

    log = ROOT / "output" / f"train_{args.name}.log"
    log.parent.mkdir(exist_ok=True)
    try:
        folders = train_voice(args.name, args.dataset, log, args.epochs, args.save_every,
                              on_stage=lambda s: print(f"-> {s}", flush=True),
                              on_epoch=lambda e, n: print(f"   epoch {e}/{n}", flush=True))
    except StepFailed as e:
        print(e, file=sys.stderr)
        return 1
    print("voice models:", *(f.name for f in folders), sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
