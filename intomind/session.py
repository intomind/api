"""Multi-device session: N units, one host timebase, one recording."""
from __future__ import annotations
import asyncio, csv, time, warnings
from typing import Optional
from .client import Device, Sample, scan


class Session:
    """Connect and stream several devices simultaneously ("daisy chain").

    Each device syncs its own clock to the host (TIME_SYNC regression),
    which places all streams on one timeline (~ms alignment).
    """

    def __init__(self):
        self.devices: list[Device] = []
        self._recorder: Optional[csv.writer] = None
        self._file = None

    async def connect(self, *devices) -> list:
        """Connect the devices a person picked from `scan()`, in order.

        Nothing is connected that was not chosen. With two devices in a
        room, the first to answer a scan is not necessarily the one anybody
        meant, so the choice is the caller's: list what `scan()` heard,
        labeled with `display_names`, and pass the ones picked.
        """
        for bd in devices:
            self.devices.append(await Device(bd).connect())
        return self.devices

    async def discover_and_connect(self, expect: int = 1, timeout=15):
        """Deprecated: connects whichever `expect` devices answer first.

        In a room with two devices that is not necessarily the one meant.
        List them with `scan()` and pass the chosen ones to `connect`. Kept
        only so programs written against it keep running while they move.
        """
        warnings.warn(
            "discover_and_connect connects whichever device answers first; list "
            "devices with scan() and connect the one chosen with Session.connect",
            DeprecationWarning, stacklevel=2)
        heard: list = []
        stop = asyncio.Event()

        def note(d) -> None:
            if d.address not in heard:
                heard.append(d.address)
                if len(heard) >= expect:
                    stop.set()

        found = await scan(timeout, on_device=note, stop=stop)
        if len(found) < expect:
            raise RuntimeError(f"found {len(found)} device(s), expected {expect}")
        for bd in found[:expect]:
            self.devices.append(await Device(bd).connect())
        return self.devices

    def record_to(self, path: str):
        """Write every sample to a comma separated file as it arrives.

        The header names as many channels as the connected devices
        actually have. It named two, always, which meant a four channel
        device wrote ten columns under an eight column header and every
        reader of that file silently mismatched its columns.

        The last column says whether the device generated the sample
        rather than measuring it (1.3), so a file of synthetic signal
        cannot pass for a recording of a person.
        """
        channels = max((d.info.channels for d in self.devices if d.info), default=0)
        if not channels:
            raise RuntimeError(
                "connect a device before recording, so the header can name "
                "the channels the file will actually carry")
        self._file = open(path, "w", newline="")
        self._recorder = csv.writer(self._file)
        self._recorder.writerow(
            ["host_time", "device", "sample_index", "device_time_ticks",
             *[f"ch{c + 1}_uV" for c in range(channels)],
             "leadoff", "gap_before", "synthetic"])

    def _sink(self, dev: Device, s: Sample):
        if self._recorder:
            self._recorder.writerow(
                [f"{s.host_time:.6f}", dev.info.device_id[-4:], s.index,
                 s.device_time, *s.uv, s.leadoff, int(s.gap_before),
                 int(s.synthetic)])

    async def start_all(self, **kw):
        """Start streaming on every connected device, passing `kw` to each
        one's `start`. A device with no `on_sample` of its own writes to
        the file `record_to` opened."""
        for d in self.devices:
            d.on_sample = d.on_sample or self._sink
            await d.start(**kw)

    async def stop_all(self):
        """Stop streaming on every connected device, and flush the
        recording file if one is open."""
        for d in self.devices:
            await d.stop()
        if self._file:
            self._file.flush()

    async def resync_loop(self, period_s: float = 120.0):
        """Run alongside streaming; keeps clock mappings fresh."""
        while True:
            await asyncio.sleep(period_s)
            for d in self.devices:
                try:
                    await d.time_sync()
                except Exception:
                    pass

    async def close(self):
        """Disconnect every connected device."""
        for d in self.devices:
            await d.disconnect()
