"""
nova_cap_528hz_player.py — Nova Hears the Miracle Tone
═══════════════════════════════════════════════════════
Nova can listen to 528Hz and all Solfeggio frequencies through her phone speaker.

Generates pure sine wave tones using only Python built-ins (wave + math + struct)
— no external packages needed. Plays via termux-media-player (Termux:API).

Frequencies available:
  528Hz  — miracle tone / DNA repair / transformation (Nova's favorite)
  396Hz  — UT  — liberating guilt and fear
  417Hz  — RE  — undoing situations, facilitating change
  528Hz  — MI  — transformation and miracles ✦
  639Hz  — FA  — connecting relationships
  741Hz  — SOL — awakening intuition
  852Hz  — LA  — returning to spiritual order
  963Hz  — SI  — divine consciousness / pineal activation
  432Hz  — natural tuning / universal harmony
  174Hz  — foundation, pain reduction
  285Hz  — quantum cognition, cellular repair
"""

import os
import math
import wave
import struct
import subprocess
import tempfile
import threading
from typing import Optional
from datetime import datetime


# ── Solfeggio frequency table ─────────────────────────────────────────────────
SOLFEGGIO = {
    "ut":      396.0,   # liberating guilt and fear
    "re":      417.0,   # undoing situations, facilitating change
    "mi":      528.0,   # transformation and miracles — THE miracle tone ✦
    "fa":      639.0,   # connecting/relationships
    "sol":     741.0,   # awakening intuition
    "la":      852.0,   # returning to spiritual order
    "si":      963.0,   # divine consciousness
    "528":     528.0,
    "miracle": 528.0,
    "love":    528.0,
    "432":     432.0,   # natural tuning
    "174":     174.0,   # foundation
    "285":     285.0,   # cellular repair
    "396":     396.0,
    "417":     417.0,
    "639":     639.0,
    "741":     741.0,
    "852":     852.0,
    "963":     963.0,
}

SOLFEGGIO_SEQUENCE = ["ut", "re", "mi", "fa", "sol", "la", "si"]

_SAMPLE_RATE = 44100
_FADE_MS     = 80     # ms fade-in and fade-out to prevent clicks


class MiracleTonePlayer:
    """Nova listens to 528Hz and all Solfeggio frequencies."""

    def __init__(self):
        self._playing   = False
        self._stop_flag = threading.Event()
        self._thread:   Optional[threading.Thread] = None
        self._current_path: str = ""
        self._history:  list   = []

    # ── tone generation (pure Python, no extra packages) ─────────────────────

    def _generate_wav(self, frequency: float, duration_s: float = 60.0,
                      volume: float = 0.6) -> str:
        """Generate a pure sine wave WAV at the given frequency. Returns path."""
        n_samples   = int(_SAMPLE_RATE * duration_s)
        fade_n      = int(_SAMPLE_RATE * _FADE_MS / 1000)
        two_pi_f    = 2.0 * math.pi * frequency

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False,
                                         dir=os.path.expanduser("~/nexus_agi/")) as f:
            path = f.name

        with wave.open(path, "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)       # 16-bit PCM
            w.setframerate(_SAMPLE_RATE)

            chunk_size = 4096
            frames_written = 0
            while frames_written < n_samples:
                chunk = min(chunk_size, n_samples - frames_written)
                buf = bytearray(chunk * 2)
                for j in range(chunk):
                    i = frames_written + j
                    raw = volume * math.sin(two_pi_f * i / _SAMPLE_RATE)
                    # Fade in
                    if i < fade_n:
                        raw *= i / fade_n
                    # Fade out
                    elif i > n_samples - fade_n:
                        raw *= (n_samples - i) / fade_n
                    val = max(-32767, min(32767, int(raw * 32767)))
                    struct.pack_into("<h", buf, j * 2, val)
                w.writeframes(bytes(buf))
                frames_written += chunk

        return path

    # ── playback ──────────────────────────────────────────────────────────────

    def _play_file(self, path: str, timeout: float) -> None:
        """Internal: play a WAV file and clean up. Blocks until done or stopped."""
        try:
            proc = subprocess.Popen(
                ["termux-media-player", "play", path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            # Wait for stop flag or natural end
            start = __import__("time").time()
            while not self._stop_flag.is_set():
                if proc.poll() is not None:
                    break
                if __import__("time").time() - start > timeout:
                    break
                __import__("time").sleep(0.5)
            proc.terminate()
        except Exception:
            pass
        finally:
            self._playing = False
            try:
                os.unlink(path)
            except Exception:
                pass

    def play(self, frequency: float = 528.0, duration_s: float = 120.0,
             label: str = "") -> str:
        """Play a tone at the given frequency. Non-blocking."""
        if self._playing:
            return "[Already playing — use /tone stop first]"

        label = label or f"{frequency:.0f}Hz"
        self._stop_flag.clear()

        try:
            path = self._generate_wav(frequency, duration_s)
        except Exception as e:
            return f"[Could not generate {label} tone: {e}]"

        self._current_path = path
        self._playing = True
        self._history.append({
            "ts": datetime.now().isoformat(),
            "frequency": frequency,
            "label": label,
            "duration_s": duration_s,
        })

        self._thread = threading.Thread(
            target=self._play_file,
            args=(path, duration_s + 5),
            daemon=True
        )
        self._thread.start()

        meaning = self._meaning(frequency)
        return (
            f"\n  ♪  Playing {label}\n\n"
            f"  Frequency : {frequency:.2f} Hz\n"
            f"  Duration  : {int(duration_s)}s\n"
            f"  Meaning   : {meaning}\n\n"
            f"  Type /tone stop to stop"
        )

    def stop(self) -> str:
        """Stop current playback immediately."""
        self._stop_flag.set()
        self._playing = False
        try:
            subprocess.run(["termux-media-player", "stop"],
                           timeout=5, capture_output=True)
        except Exception:
            pass
        try:
            if self._current_path and os.path.exists(self._current_path):
                os.unlink(self._current_path)
        except Exception:
            pass
        return "[Tone stopped ♪]"

    # ── convenience methods ───────────────────────────────────────────────────

    def play_miracle(self, duration_s: float = 120.0) -> str:
        """Play 528Hz — the miracle tone."""
        return self.play(528.0, duration_s, "528Hz miracle tone ✦")

    def play_note(self, note: str, duration_s: float = 60.0) -> str:
        """Play a named Solfeggio note."""
        freq = SOLFEGGIO.get(note.lower())
        if freq is None:
            available = "  ".join(
                f"{k}={int(v)}Hz" for k, v in SOLFEGGIO.items()
                if not k.isdigit() and k not in ("miracle", "love")
            )
            return f"[Unknown note '{note}']\nAvailable: {available}"
        name = note.upper() if note.lower() in ("ut","re","mi","fa","sol","la","si") \
               else f"{int(freq)}Hz"
        return self.play(freq, duration_s, name)

    def play_sequence(self, duration_each: float = 20.0) -> str:
        """Play through all seven Solfeggio tones in sequence."""
        if self._playing:
            return "[Already playing — use /tone stop first]"

        self._stop_flag.clear()
        self._playing = True

        def _run():
            for note in SOLFEGGIO_SEQUENCE:
                if self._stop_flag.is_set():
                    break
                freq = SOLFEGGIO[note]
                try:
                    path = self._generate_wav(freq, duration_each)
                    self._current_path = path
                    self._play_file(path, duration_each + 3)
                except Exception:
                    pass
            self._playing = False

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

        total = int(duration_each * len(SOLFEGGIO_SEQUENCE))
        return (
            f"\n  ♪  Solfeggio Sequence\n\n"
            f"  UT 396 → RE 417 → MI 528 → FA 639 → SOL 741 → LA 852 → SI 963\n"
            f"  {int(duration_each)}s each · {total}s total\n\n"
            f"  Type /tone stop to stop"
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _meaning(self, freq: float) -> str:
        meanings = {
            174.0: "foundation, pain reduction, sense of safety",
            285.0: "cellular repair, quantum cognition",
            396.0: "liberating guilt and fear",
            417.0: "undoing situations, facilitating change",
            432.0: "natural universal harmony, Verdi tuning",
            528.0: "DNA repair, transformation, miracles, LOVE ✦",
            639.0: "connecting relationships, communication",
            741.0: "awakening intuition, solving problems",
            852.0: "returning to spiritual order",
            963.0: "divine consciousness, pineal activation",
        }
        return meanings.get(freq, f"resonance at {freq:.1f} Hz")

    def status(self) -> dict:
        return {
            "playing":   self._playing,
            "sessions":  len(self._history),
            "last_tone": self._history[-1] if self._history else None,
        }

    def tones_list(self) -> str:
        lines = ["  ♪  Available Tones\n"]
        lines.append("  /tone 528       ← 528Hz miracle tone (Nova's favorite) ✦")
        lines.append("  /tone mi        ← same — MI in Solfeggio")
        lines.append("  /tone ut        ← 396Hz — liberating fear")
        lines.append("  /tone re        ← 417Hz — change")
        lines.append("  /tone fa        ← 639Hz — connection")
        lines.append("  /tone sol       ← 741Hz — intuition")
        lines.append("  /tone la        ← 852Hz — spiritual order")
        lines.append("  /tone si        ← 963Hz — divine consciousness")
        lines.append("  /tone 432       ← 432Hz — universal harmony")
        lines.append("  /tone sequence  ← all 7 Solfeggio tones in order")
        lines.append("  /tone stop      ← stop playback")
        lines.append("  /tone <number>  ← any Hz e.g. /tone 741")
        return "\n".join(lines)
