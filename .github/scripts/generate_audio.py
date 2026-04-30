#!/usr/bin/env python3
"""Generate audio narration for German Driving Theory Trainer questions.

Reads data/driving_theory_questions.json and produces:
  <out>/{question_id}.webm        — question text + all options
  <out>/{question_id}_hint.webm   — explanation (the "comment" field)
  <out>/manifest.json             — catalog of generated files (used by runtime)

By default the output dir is the sibling public repo `../drivingtest-audio`
(detected automatically when present). Override with --output-dir.

TTS engine: Kokoro-82M (Apache 2.0). Runs locally — no API calls.
Output:     Opus in WebM, mono, 24 kbps. ~75-105 KB per question file.

Usage:
  python scripts/generate_audio.py --limit 10        # smoke test on 10 questions
  python scripts/generate_audio.py                   # all ~2400 questions (3-8h)
  python scripts/generate_audio.py --voice af_bella  # try a different voice
  python scripts/generate_audio.py --force           # regenerate even if up-to-date
  python scripts/generate_audio.py --output-dir ./x  # write somewhere else

Setup once:
  brew install ffmpeg
  pip install kokoro-onnx soundfile numpy

Idempotent: skips files whose source-text hash matches the manifest entry.
Safe to re-run.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

# --- Paths --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "data" / "driving_theory_questions.json"
SIBLING_AUDIO_REPO = ROOT.parent / "drivingtest-audio"   # public CDN repo
LOCAL_AUDIO_DIR = ROOT / "audio"                          # fallback / local-only

def default_output_dir() -> Path:
    """Prefer the sibling public audio repo when it exists (a git checkout).
    Otherwise fall back to the in-app `audio/` folder."""
    if (SIBLING_AUDIO_REPO / ".git").exists():
        return SIBLING_AUDIO_REPO
    return LOCAL_AUDIO_DIR

# These get rebound in main() once we know the output dir.
AUDIO_DIR = default_output_dir()
MANIFEST_PATH = AUDIO_DIR / "manifest.json"

# --- Defaults -----------------------------------------------------------------
DEFAULT_VOICE = "af_heart"  # Kokoro voices: af_heart, af_bella, af_nicole, am_adam...
SAMPLE_RATE = 24000          # Kokoro outputs 24 kHz
OPUS_BITRATE = "24k"         # 24 kbps mono is plenty for clear speech
PAUSE_BETWEEN_OPTIONS_S = 0.35

# --- Text normalization -------------------------------------------------------
_ABBREV = [
    (re.compile(r"\bkm/h\b", re.IGNORECASE), "kilometers per hour"),
    (re.compile(r"\bkm\b", re.IGNORECASE), "kilometers"),
    (re.compile(r"\b(\d+)\s*m\b"), r"\1 meters"),
    (re.compile(r"%"), " percent"),
    (re.compile(r"\bca\.\s*", re.IGNORECASE), "approximately "),
    (re.compile(r"\bz\.B\.\s*", re.IGNORECASE), "for example "),
    (re.compile(r"\betc\.\s*", re.IGNORECASE), "et cetera "),
]
_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    if not text:
        return ""
    t = _TAGS.sub(" ", text)
    for pat, repl in _ABBREV:
        t = pat.sub(repl, t)
    return _WS.sub(" ", t).strip()


def build_question_script(q: dict) -> str:
    """Question text + each option, with letter callouts and natural pauses."""
    parts = [normalize(q.get("question_text", ""))]
    for opt in q.get("options") or []:
        letter = opt.get("letter", "").rstrip(".")
        text = normalize(opt.get("text", ""))
        if not text:
            continue  # image-only option — skip narration
        parts.append(f"Option {letter}. {text}")
    return ". ".join(p.rstrip(".") for p in parts if p) + "."


def build_hint_script(q: dict) -> str:
    return normalize(q.get("comment", ""))


def text_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


# --- Kokoro -------------------------------------------------------------------
class KokoroTTS:
    """Thin wrapper. Tries kokoro-onnx (fast on Apple Silicon) first,
    falls back to the kokoro pip package."""

    def __init__(self, voice: str):
        self.voice = voice
        self._impl = None
        self._kind = None
        self._init()

    def _init(self):
        try:
            from kokoro_onnx import Kokoro  # type: ignore

            # kokoro-onnx 0.4+ requires explicit model + voices file paths.
            # We cache them in <repo>/.kokoro-cache/ — see README for download URLs.
            cache = ROOT / ".kokoro-cache"
            model_path = cache / "kokoro-v1.0.onnx"
            voices_path = cache / "voices-v1.0.bin"
            if not model_path.exists() or not voices_path.exists():
                raise FileNotFoundError(
                    f"Missing Kokoro model files in {cache}. Run:\n"
                    f"  mkdir -p {cache} && cd {cache} && \\\n"
                    f"  curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx && \\\n"
                    f"  curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
                )
            self._impl = Kokoro(str(model_path), str(voices_path))
            self._kind = "onnx"
            print(f"[TTS] Using kokoro-onnx (voice={self.voice})", flush=True)
            return
        except Exception as e:
            print(f"[TTS] kokoro-onnx not available ({e}); trying kokoro...", flush=True)

        try:
            from kokoro import KPipeline  # type: ignore

            # 'a' = American English. KPipeline auto-downloads model on first run.
            self._impl = KPipeline(lang_code="a")
            self._kind = "pipeline"
            print(f"[TTS] Using kokoro KPipeline (voice={self.voice})", flush=True)
        except Exception as e:
            sys.stderr.write(
                "ERROR: Could not initialize Kokoro. Install one of:\n"
                "  pip install kokoro-onnx soundfile        (recommended)\n"
                "  pip install kokoro soundfile             (alternative)\n"
                f"\nLast error: {e}\n"
            )
            sys.exit(1)

    def synth_wav(self, text: str, out_path: Path) -> float:
        """Write a 24 kHz mono WAV to out_path. Return duration in seconds."""
        import numpy as np
        import soundfile as sf

        if self._kind == "onnx":
            samples, sr = self._impl.create(text, voice=self.voice, speed=1.0, lang="en-us")
            audio = np.asarray(samples, dtype=np.float32)
        else:  # KPipeline yields chunks
            chunks = []
            for _, _, audio_chunk in self._impl(text, voice=self.voice, speed=1.0):
                chunks.append(np.asarray(audio_chunk, dtype=np.float32))
            audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
            sr = SAMPLE_RATE

        # Normalize peak slightly under 0 dBFS for headroom
        if audio.size:
            peak = float(max(abs(audio.max()), abs(audio.min())) or 1.0)
            if peak > 0.99:
                audio = audio / peak * 0.97

        sf.write(str(out_path), audio, sr, subtype="PCM_16")
        return float(audio.size) / float(sr) if audio.size else 0.0


# --- WebM/Opus encoding ------------------------------------------------------

def encode_opus(wav_path: Path, out_path: Path) -> None:
    """Encode WAV to Opus in WebM container at OPUS_BITRATE mono."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(wav_path),
        "-c:a", "libopus",
        "-b:a", OPUS_BITRATE,
        "-ac", "1",
        "-application", "voip",       # tuned for speech
        "-frame_duration", "60",
        str(out_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {res.stderr.strip()}")


# --- Manifest ----------------------------------------------------------------

def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text("utf-8"))
        except Exception:
            print("[manifest] Existing manifest unreadable; starting fresh.", flush=True)
    return {"version": 1, "voice": DEFAULT_VOICE, "files": {}}


def save_manifest(manifest: dict) -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), "utf-8")


# --- Main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="Process only the first N questions (0 = all).")
    ap.add_argument("--voice", default=DEFAULT_VOICE, help=f"Kokoro voice (default: {DEFAULT_VOICE}).")
    ap.add_argument("--force", action="store_true", help="Regenerate even if hash matches manifest.")
    ap.add_argument("--only", help="Process only this single question_id.")
    ap.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Where to write .webm files and manifest.json. "
            "Default: ../drivingtest-audio if it's a git checkout, else ./audio."
        ),
    )
    ap.add_argument("--shard", type=int, default=0,
                    help="0-indexed shard number to process (use with --shards).")
    ap.add_argument("--shards", type=int, default=1,
                    help="Total number of shards. Each shard processes questions where i %% shards == shard.")
    ap.add_argument("--questions", default=None,
                    help="Path to driving_theory_questions.json (default: <repo>/data/driving_theory_questions.json).")
    args = ap.parse_args()
    if args.shards < 1 or args.shard < 0 or args.shard >= args.shards:
        sys.stderr.write(f"ERROR: invalid shard config: --shard {args.shard} --shards {args.shards}\n")
        sys.exit(2)

    # Rebind the module-level paths so load_manifest/save_manifest see the override.
    global AUDIO_DIR, MANIFEST_PATH
    AUDIO_DIR = Path(args.output_dir).resolve() if args.output_dir else default_output_dir()
    MANIFEST_PATH = AUDIO_DIR / "manifest.json"
    print(f"[gen] output dir: {AUDIO_DIR}", flush=True)

    if not shutil.which("ffmpeg"):
        sys.stderr.write("ERROR: ffmpeg not found. Install with: brew install ffmpeg\n")
        sys.exit(1)

    questions_path = Path(args.questions).resolve() if args.questions else QUESTIONS_PATH
    if not questions_path.exists():
        sys.stderr.write(f"ERROR: questions JSON not found at {questions_path}\n")
        sys.exit(1)

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    questions = json.loads(questions_path.read_text("utf-8"))
    if args.only:
        questions = [q for q in questions if q.get("question_id") == args.only]
    if args.shards > 1:
        # Stride-based sharding: shard k of N takes indices k, k+N, k+2N, ...
        # Stride > contiguous because it spreads any "slow" cluster of long
        # questions evenly, so all shards finish in similar wall time.
        questions = [q for i, q in enumerate(questions) if i % args.shards == args.shard]
        print(f"[shard] processing shard {args.shard + 1}/{args.shards} ({len(questions)} questions)", flush=True)
    if args.limit:
        questions = questions[: args.limit]

    print(f"[gen] {len(questions)} question(s) to process. Voice={args.voice}.", flush=True)

    manifest = load_manifest()
    if manifest.get("voice") != args.voice:
        print(f"[manifest] Voice changed ({manifest.get('voice')} → {args.voice}); will regenerate touched files.", flush=True)
        manifest["voice"] = args.voice

    tts = KokoroTTS(voice=args.voice)

    n_done = n_skipped = n_failed = 0
    t0 = time.time()

    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_root = Path(tmp_root)

        for i, q in enumerate(questions, 1):
            qid = q.get("question_id")
            if not qid:
                continue

            q_text = build_question_script(q)
            h_text = build_hint_script(q)
            q_hash = text_hash(q_text)
            h_hash = text_hash(h_text) if h_text else ""

            entry = manifest["files"].get(qid, {})
            q_out = AUDIO_DIR / f"{qid}.webm"
            h_out = AUDIO_DIR / f"{qid}_hint.webm"

            need_q = args.force or entry.get("q_hash") != q_hash or not q_out.exists()
            need_h = bool(h_text) and (args.force or entry.get("hint_hash") != h_hash or not h_out.exists())

            if not need_q and not need_h:
                n_skipped += 1
                continue

            try:
                # Question audio
                q_dur = entry.get("q_dur", 0.0)
                if need_q:
                    wav = tmp_root / f"{qid}.wav"
                    q_dur = tts.synth_wav(q_text, wav)
                    encode_opus(wav, q_out)

                # Hint audio (some questions have no comment)
                h_dur = entry.get("hint_dur", 0.0)
                if need_h:
                    wav = tmp_root / f"{qid}_hint.wav"
                    h_dur = tts.synth_wav(h_text, wav)
                    encode_opus(wav, h_out)
                elif not h_text and h_out.exists():
                    h_out.unlink()  # cleanup stale hint file when comment removed

                manifest["files"][qid] = {
                    "q_hash": q_hash,
                    "q_size": q_out.stat().st_size if q_out.exists() else 0,
                    "q_dur": round(q_dur, 2),
                    "hint_hash": h_hash,
                    "hint_size": h_out.stat().st_size if h_out.exists() else 0,
                    "hint_dur": round(h_dur, 2),
                }
                n_done += 1

                if n_done % 5 == 0 or i == len(questions):
                    save_manifest(manifest)
                    elapsed = time.time() - t0
                    rate = n_done / max(elapsed, 0.001)
                    eta = (len(questions) - i) / rate if rate > 0 else 0
                    print(
                        f"[gen] {i}/{len(questions)} ({qid}) "
                        f"done={n_done} skip={n_skipped} fail={n_failed} "
                        f"rate={rate:.2f}/s eta={eta/60:.1f}m",
                        flush=True,
                    )
            except Exception as e:
                n_failed += 1
                print(f"[gen] FAILED {qid}: {e}", file=sys.stderr, flush=True)

    save_manifest(manifest)

    elapsed = time.time() - t0
    print(
        f"\n[gen] DONE in {elapsed/60:.1f}m. "
        f"generated={n_done} skipped={n_skipped} failed={n_failed} "
        f"total_files={len(manifest['files'])}",
        flush=True,
    )

    # Print next-step hint
    total_bytes = sum(
        f.get("q_size", 0) + f.get("hint_size", 0)
        for f in manifest["files"].values()
    )
    print(f"[gen] audio dir size: {total_bytes / 1024 / 1024:.1f} MB", flush=True)
    print("\nNext steps:")
    print(f"  1. Inspect a few .webm files in {AUDIO_DIR} to confirm quality.")
    print("  2. Publish to GitHub + jsDelivr:")
    print("       bash scripts/publish_audio.sh audio-v1")
    print("     (or run the git commands manually inside the audio repo)")


if __name__ == "__main__":
    main()
