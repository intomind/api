"""Shared fakes. Kept out of the test modules so both suites use one device."""
from __future__ import annotations
import numpy as np

from intomind import instrument as I

FS = 500.0


class FakeShell:
    dev = None
    available = True

    def __init__(self, mv: float = 1657.0):
        self.mv = mv

    async def arun(self, cmds, **kw):
        return [(cmds[0], f"RLDOUT = {int(self.mv)} mV  (n=16)")]

    async def aregs(self):
        return {}


class FakeDev:
    connected = True
    name = "IntoMind-FAKE"

    def __init__(self, dc: float = 1500.0, gaps: int = 0, leadoff: int = 0x00):
        self.on_sample = None
        self.dc, self.gaps = dc, gaps
        # The lead-off status byte a device reports. 0x00 = every electrode on.
        # Contact is decoded from this, never from `dc`.
        self.leadoff = leadoff
        self.streaming = False
        self.commands: list = []
        self.gaps_announced = 0
        # The API-layer config cache the hub renders from.
        self.regs: dict = {}
        self.config: dict = {}
        self.filter = None

    async def refresh_config(self):
        return self.config

    async def start(self, samples_per_packet: int = 25):
        self.streaming = True

    async def stop(self):
        # The firmware answers ST_NOT_STREAMING if it is not streaming, and
        # Device.command raises on any non-zero status. Model that.
        if not self.streaming:
            raise RuntimeError("op 0x02 -> status 4")
        self.streaming = False

    async def command(self, op, arg=None, timeout=6.0):
        self.commands.append((op, arg))
        return b""



async def fake_probe(dev, seconds: float = 3.0) -> I.Probe:
    """A probe from a device whose electrodes are on, unless the test says
    otherwise via `dev.leadoff`.

    Contact now comes from the device's own electrode-off detection, so
    a probe with no lead-off bits means "not measured", and that is deliberately
    NOT a pass. Tests that want a healthy instrument must say so.
    """
    n = 1500
    uv = np.stack([np.full(n, dev.dc), np.full(n, dev.dc)])
    counts = np.full_like(uv, 0.1 * (1 << 23))
    loff = np.full(n, getattr(dev, "leadoff", 0x00), dtype=np.uint8)
    return I.Probe(uv=uv, counts=counts, gaps=dev.gaps, fs=FS, seconds=n / FS,
                   leadoff=loff)
