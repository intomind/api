"""Cue tones. Every cue point is configurable, including silence.

Tones are synthesized once into /tmp WAVs and played with `aplay`; speech
goes through `spd-say`. Nothing here blocks the event loop for long — the
player is spawned, not waited on.
"""
from __future__ import annotations
import math, struct, subprocess, tempfile, pathlib, shutil

RATE = 44100
_CACHE: dict[str, pathlib.Path] = {}

# name -> list of (freq_hz, seconds). Empty list = silence.
TONES: dict[str, list[tuple[float, float]]] = {
    "none":      [],
    "speech":    [],                                   # handled by spd-say
    "beep_low":  [(440.0, 0.18)],
    "beep_high": [(880.0, 0.18)],
    "chime_up":  [(660.0, 0.12), (990.0, 0.16)],
    "chime_down": [(990.0, 0.12), (660.0, 0.16)],
    "click":     [(1500.0, 0.035)],
    "double":    [(880.0, 0.09), (0.0, 0.06), (880.0, 0.09)],
    "long_low":  [(330.0, 0.5)],
}
CHOICES = list(TONES)


def _wav(name: str) -> pathlib.Path | None:
    segs = TONES.get(name)
    if not segs:
        return None
    if name in _CACHE and _CACHE[name].exists():
        return _CACHE[name]
    frames = bytearray()
    for freq, dur in segs:
        n = int(RATE * dur)
        for i in range(n):
            if freq <= 0:
                s = 0.0
            else:
                # raised-cosine envelope: no clicks at the edges
                env = 0.5 * (1 - math.cos(2 * math.pi * min(i, n - i) /
                                          max(1, n * 0.25)))
                env = min(1.0, env)
                s = env * 0.6 * math.sin(2 * math.pi * freq * i / RATE)
            frames += struct.pack("<h", int(s * 32767))
    data = bytes(frames)
    hdr = (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " +
           struct.pack("<IHHIIHH", 16, 1, 1, RATE, RATE * 2, 2, 16) +
           b"data" + struct.pack("<I", len(data)))
    p = pathlib.Path(tempfile.gettempdir()) / f"intomind_tone_{name}.wav"
    p.write_bytes(hdr + data)
    _CACHE[name] = p
    return p


def available() -> dict:
    """The tone names this module can play, and whether this host can
    actually play a tone or speak text."""
    return dict(tones=CHOICES,
                aplay=bool(shutil.which("aplay")),
                speech=bool(shutil.which("spd-say")))


def play(tone: str, text: str = "") -> None:
    """Fire a cue. `tone='none'` is silent; `tone='speech'` speaks `text`."""
    if not tone or tone == "none":
        return
    try:
        if tone == "speech":
            if text and shutil.which("spd-say"):
                subprocess.Popen(["spd-say", "-w", "-r", "-20", text],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
            return
        w = _wav(tone)
        if w and shutil.which("aplay"):
            subprocess.Popen(["aplay", "-q", str(w)],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    except Exception:
        pass          # a missing sound device must never abort a recording
