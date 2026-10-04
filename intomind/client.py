"""One device: discovery, control, streaming, and the clock that ties a
sample to a wall clock.

The wire lives in `protocol`, which is held to the contract's own
conformance vectors. Nothing here decodes bytes by hand, and nothing here
knows what converter a device is built on. What a device can do is what it
says it can do.

Hardware this library was not written for is supported by an adapter,
which is a separate package that claims a device and hangs its own methods
off `device.extra`. See `adapters`.
"""
from __future__ import annotations
import asyncio
import warnings, struct, time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Optional
from bleak import BleakScanner, BleakClient
from bleak.exc import BleakError

from . import adapters, pairing, protocol as P, timebase
from .protocol import DeviceInfo, ProtocolError


class Sync(NamedTuple):
    """One TIME_SYNC exchange, read on both host clocks.

    `mono` is what the device clock is regressed against, because it cannot
    step. `host` is the wall clock read next to it, and the pair is the anchor
    that carries the fit back into the timebase the record is written in.
    """
    host: float               # CLOCK_REALTIME, midpoint of the exchange
    device: float             # device uptime, seconds
    rtt: float                # round trip; half of it bounds the uncertainty
    mono: float               # CLOCK_MONOTONIC, midpoint of the exchange

# WHAT A CLOCK SKEW IS ALLOWED TO BE, and how well it has to be known before
# it is used instead of assuming the two clocks run at the same rate. See
# `Device._reject_skew`, which is the only reader of either.
MAX_SKEW_PPM = 1000.0          # two crystals, generously
MAX_SKEW_STDERR_PPM = 100.0    # and it has to be measured, not drawn

# The characteristics, and the operations, from the contract.
SVC, INFO, CTL, RSP, DATA, STAT = (
    P.SERVICE, P.DEVICE_INFO, P.CONTROL, P.CONTROL_RESPONSE, P.EEG_DATA, P.STATUS)
UPDATE_CONTROL, UPDATE_DATA, PREDICTIONS, EMBEDDINGS = (
    P.UPDATE_CONTROL, P.UPDATE_DATA, P.PREDICTIONS, P.EMBEDDINGS)

# The names the rest of this library and applications already use. The
# numbers are the contract's, in one place, in `protocol`.
OP = {name.upper(): code for name, code in P.OPCODES.items()}
OP["START"] = P.OPCODES["start_stream"]
OP["STOP"] = P.OPCODES["stop_stream"]
OP["SET_SPP"] = P.OPCODES["set_samples_per_packet"]

GAIN_BY_CODE = P.GAIN_BY_CODE
RATE_BY_CODE = P.RATE_BY_CODE
RATE_BY_INFO_BIT = P.RATE_BY_INFO_BIT
MODES = P.MODES

# Status characteristic (v0.1 contract §Status). Both firmwares encode the same
# 12 bytes, so this is the one configuration read every device has -- a device
# with no register getter can still say what it is set to.
STATUS_LEN = P.STATUS_LEN


# `Status`, `DeviceInfo`, and the packet decoding all live in `protocol`,
# which is held to the contract's own conformance vectors. They are named
# here so that code written against this module keeps working.
Status = P.Status


@dataclass
class Sample:
    """One multi-channel sample with device- and host-time."""
    index: int
    device_time: int          # ticks (see DeviceInfo.tick_hz)
    host_time: float          # mapped via TIME_SYNC regression
    channels: tuple           # raw int counts
    uv: tuple                 # scaled microvolts
    leadoff: int
    gap_before: bool = False
    #: 1.3: the device generated this sample rather than measuring it. It
    #: came in a synthetic packet (type 0x04), and nothing else sets it.
    synthetic: bool = False



#: What a repeated packet means, in words for whoever is watching. Measured
#: 2026-09-28: the system's Bluetooth service sends each notification once
#: per program subscribed, and every subscriber hears all of the copies.
ANOTHER_LISTENER = ("another program on this computer is also listening to this device: "
                    "each packet arrives twice, and the second copy is dropped")

#: How long after the last repeat another program still counts as listening.
_LISTENER_HOLD_S = 2.0


async def scan(timeout: float = 10.0, on_device: Optional[Callable] = None,
               stop: Optional[asyncio.Event] = None) -> list:
    """Find devices that speak this protocol.

    The service identifier is what a device advertises to say what it is,
    so that is what is asked for. A name is a label a user can change and a
    prefix is a guess, and both were how this used to work.

    The scan listens for `timeout` seconds at most. A device advertising
    several times a second is heard in well under one, so nothing that
    wants a device should wait the timeout out. `on_device`, if given, is
    called with each device the moment it is heard, and once more if its
    name arrives later, because the name travels in the scan response that
    follows the advertisement. `stop`, if given, ends the scan as soon as it
    is set. A list a person picks from fills as devices answer, and a
    program that wants one device stops the moment it has it.

    A device the computer is already connected to does not advertise, so
    no scan can hear it. That is every device paired a moment ago from the
    system's Bluetooth settings, which connect as they pair. The system
    knows them, so they are reported first, before the radio listens, and a
    client connected through the system shares the link. The list comes
    back in the order heard.
    """
    found: list = []
    by_address: dict = {}

    def heard(d) -> None:
        known = by_address.get(d.address)
        if known is None:
            found.append(d)
        elif d.name and not known.name:
            found[found.index(known)] = d
        else:
            return
        by_address[d.address] = d
        if on_device is not None:
            on_device(d)

    for d in await pairing.held_devices():
        props = d.details.get("props", {}) if isinstance(d.details, dict) else {}
        if _speaks(props.get("UUIDs"), d.name):
            heard(d)

    def advertised(d, adv) -> None:
        if _speaks(adv.service_uuids, d.name or adv.local_name):
            heard(d)

    scanner = BleakScanner(detection_callback=advertised)
    await scanner.start()
    try:
        await asyncio.wait_for((stop or asyncio.Event()).wait(), timeout)
    except asyncio.TimeoutError:
        pass
    finally:
        await scanner.stop()
    return found


def _speaks(uuids, name) -> bool:
    if P.SERVICE in {str(u).lower() for u in (uuids or [])}:
        return True
    # Without the service in the advertisement, the name: a 1.3 device's
    # composed name ends with the product, a 1.2 device's began with a prefix.
    n = name or ""
    return n.startswith(P.NAME_PREFIX) or n.endswith(P.PRODUCT_NAME)


def display_names(devices) -> dict:
    """A label per device address for a list a person picks from: the
    device's own name, and when two devices in the list share one, a
    number after the later ones (`Ada's Blue IntoMind One 2`). The number is
    the host's and lasts as long as the list: the device knows nothing of
    it, which is the point, since two devices cannot agree on who is second
    without a host to see them both."""
    labels, seen = {}, {}
    for d in devices:
        name = d.name or "IntoMind One"
        seen[name] = seen.get(name, 0) + 1
        labels[d.address] = name if seen[name] == 1 else f"{name} {seen[name]}"
    return labels



class Refused(RuntimeError):
    """The device answered a control request with a status other than ok.

    `status` is the number and `status_name` its name in the contract. The
    message keeps the form `op 0x.. -> status N`, and says what to do when
    there is something a person can do.
    """

    def __init__(self, op: int, status: int):
        self.op = op
        self.status = status
        self.status_name = P.STATUS_CODES.get(status, f"undefined status {status}")
        text = f"op 0x{op:02x} -> status {status} ({self.status_name})"
        if status == P.STATUS_USB_POWER:
            text += ": the device does not record while plugged in. Unplug it to record"
        super().__init__(text)


class Device:
    """One unit: control plane + sample stream + clock mapping.

    This is also the reference ADAPTER. An adapter owns one transport and one
    protocol and presents them uniformly to the rest of the application, so that
    hardware other than ours can be supported by code we neither write nor
    maintain. Every adapter identifies itself, and that identity is written into
    each recording: a capture made through code we did not ship has to be
    identifiable as such forever, not only while someone remembers.

    The interface an adapter must present is NOT yet stable. It has exactly one
    implementation, which is the condition under which an abstraction is usually
    wrong, so it will change when the second one arrives.
    """

    ADAPTER = dict(name="intomind.ble", version="1", first_party=True,
                   transport="bluetooth-le")

    # CLASS DEFAULTS, so that a Device is answerable about its clock before
    # anything has happened to it. `_map_frozen` is what `start` raises to
    # stop the fit moving under a running stream and `_learned` is the fit
    # that keeps being taken meanwhile; both have a meaning on a device that
    # has never streamed, and that meaning is "no, and nothing yet".
    _map_frozen = False
    _learned: dict = {}

    # What this device can do, declared by the adapter that speaks to it. The
    # application renders its controls from this rather than from knowledge of
    # any particular device, so an adapter for other hardware describes itself
    # the same way and gets the same interface.
    #
    # No model id and no component part numbers. Both are internal, and naming
    # the front end identifies the design as surely as a model number does.
    # `hardware` is left for the application to compose from what the firmware
    # reports, which is what a user actually needs to know.
    def profile(self) -> dict:
        """What this device can do, for an application that renders
        controls from it.

        Composed from the device's own capability bits and its own rate
        mask. A table of any particular hardware here would be a defect,
        and was one: a fixed gain list and a fixed control set described
        the first device this library met and quietly described every
        other one wrongly.
        """
        if self.info is None:
            return dict(vendor="IntoMind", hardware=None, controls={})
        info = self.info
        controls = dict(
            gain=dict(supported=True, options=list(P.GAIN_BY_CODE)),
            sample_rate=dict(supported=True, options=info.rates),
        )
        if info.can("test_signal") or info.can("input_short") or info.can("synthetic"):
            options = ["normal"]
            if info.can("test_signal"):
                options.append("test")
            if info.can("input_short"):
                options.append("short")
            if info.can("synthetic"):
                # 1.3: the device generates the signal, converter off.
                options.append("synthetic")
            # What the amplifier is listening to. A bench diagnostic rather
            # than an everyday knob.
            controls["signal_source"] = dict(supported=True, tier="advanced",
                                             options=options)
        if info.can("leadoff"):
            controls["leadoff"] = dict(supported=True, options=[False, True])
        if info.can("model"):
            controls["predictions"] = dict(
                supported=True, ready=info.can("model_ready"),
                head_slots=info.head_slots, embedding=info.model_embed_dim)
        if info.can("pipeline"):
            # The chain the device runs on its signal: kinds from the
            # device's catalog, the device's own default in basic mode, the
            # whole chain in advanced mode.
            controls["processing"] = dict(supported=True, tier="basic",
                                          catalog="pipeline_catalog()")
        if info.can("bias_drive"):
            controls["bias"] = dict(supported=True, tier="advanced",
                                    options=list(P.BIAS_MODES))
        if info.can("indicator"):
            # The status lamp's level: a user's choice the device keeps
            # across power cycles, and an identify blink on request.
            controls["indicator"] = dict(supported=True, options=list(P.INDICATOR_LEVELS),
                                         identify=True)
        if info.can("converter_registers"):
            # A diagnostic read, in words, for the converter family the
            # device declares. Nothing to set.
            controls["converter_registers"] = dict(supported=True, tier="advanced", read_only=True)
        if info.can("embeddings"):
            # The encoder's output for each window, on request, in the
            # forms the contract defines.
            controls["embeddings"] = dict(supported=True,
                                          forms=[f for f in P.EMBEDDING_FORMS if f != "off"])
        if info.can("model_cadence"):
            # 1.3: how often the model describes a window, seconds; 0 is
            # every window the device can. The device states its minimum.
            controls["model_cadence"] = dict(supported=True, unit="seconds", minimum="model_interval()")
        if info.can("device_name"):
            # 1.3: a name and an adjective the device keeps and advertises,
            # composed within what the air carries.
            controls["name"] = dict(supported=True, parts=["name", "adjective"],
                                    product=P.PRODUCT_NAME, max_bytes=P.NAME_MAX_COMPOSED)
        hardware = ("{}.{}.{}".format(*info.hardware) if info.hardware else None)
        return dict(vendor="IntoMind", hardware=hardware, controls=controls)

    def __init__(self, ble_device, on_disconnect: Optional[Callable] = None):
        self._bd = ble_device
        self._on_disconnect = on_disconnect
        self._client: Optional[BleakClient] = None
        self.info: Optional[DeviceInfo] = None
        # A connected device streams. Nothing "owns" the stream; a recording
        # only decides whether the samples are also written to disk.
        self.streaming = False
        self.name = ble_device.name or ble_device.address
        self._rsp: asyncio.Queue = asyncio.Queue()
        self.on_sample: Optional[Callable[[ "Device", Sample], None]] = None
        self.on_packet: Optional[Callable[["Device", P.Packet], None]] = None
        #: 1.4: set when the device ended the stream because USB power
        #: appeared; it does not record while plugged in. `start` clears it.
        self.ended_by_usb = False
        #: Called once, with the device, when that happens. From the
        #: Bluetooth library's thread, like `on_packet`.
        self.on_usb_stop: Optional[Callable[["Device"], None]] = None
        #: One window's model output, on a device that carries a model and
        #: has been told to predict.
        self.on_prediction: Optional[Callable[["Device", P.Prediction], None]] = None
        #: One whole window of the encoder's output, on a device that sends
        #: embeddings and has been asked to (`set_embeddings`).
        self.on_embedding: Optional[Callable[["Device", P.EmbeddingWindow], None]] = None
        #: Every embedding notification as it arrives, before assembly.
        self.on_embedding_packet: Optional[Callable[["Device", P.Embedding], None]] = None
        self._assembler: Optional[P.EmbeddingAssembler] = None
        self._embeddings_subscribed = False
        self._embeddings_form = "off"
        #: Notifications that were not a message. Counted rather than
        #: guessed at, because guessing turns a bad byte into a plausible
        #: sample.
        self.undecodable = 0
        self.status: Optional[Status] = None
        #: The adapter that claimed this device, if any, and whatever it
        #: offers. None on a device the contract already covers.
        self.adapter = None
        self.extra = None
        self._predictions = False
        self._update_rsp: Optional[asyncio.Queue] = None
        #: The chain in force when the stream last started, on a device that
        #: runs one: what a recording stores to say what produced its samples.
        self.pipeline_at_start: Optional["P.PipelineState"] = None
        self._catalog_cache: Optional[list] = None
        # What the device is set to, from the device. There is no register
        # cache here any more: the status characteristic is the whole
        # answer under this contract.
        self.config: dict = {}
        # Clock mapping: mono ≈ device*skew + offset (regressed).
        #
        # Regressed against CLOCK_MONOTONIC, never against the wall clock. An
        # NTP step does not merely displace events, it BENDS a realtime-fitted
        # line -- so the fit that dates every sample would be corrupted by
        # something entirely outside this instrument. Monotonic cannot step.
        # Realtime is recovered afterwards through the anchors. See timebase.py.
        self._syncs: list[Sync] = []
        self._offset = 0.0
        self._skew = 1.0
        # Why the slope was not taken, when it was not. None means it was.
        self._skew_pinned: Optional[str] = None
        # (realtime - monotonic) at the FIRST anchor. Every sample's host_time
        # is placed on that one frame, so the sample column stays a continuous
        # ruler even across a step -- the step is recorded, not smeared into
        # the data. Events carry both clocks and are mapped onto the same frame
        # by timebase.align_events.
        self._frame_offset = 0.0
        self._frame_mono = None
        self._gain = 6
        self._expected_idx = None
        self.samples_total = 0
        self.gaps_announced = 0
        self.duplicates = 0          # packets delivered twice by the host's stack
        self._last_duplicate: Optional[float] = None
        # The first index and device time of the last few packets passed
        # on, what a copy is recognized by.
        self._recent: deque = deque(maxlen=8)

    @property
    def another_listener(self) -> bool:
        """Whether another program on this computer is listening to the
        device now: a packet arrived twice within the last two seconds. The
        copies are dropped, so what is recorded is intact either way."""
        return (self._last_duplicate is not None
                and time.monotonic() - self._last_duplicate < _LISTENER_HOLD_S)

    @property
    def connected(self) -> bool:
        """Link liveness. `hub.dev is not None` is NOT liveness."""
        return bool(self._client and self._client.is_connected)

    def _disconnected(self, _client):
        if self._on_disconnect:
            try:
                self._on_disconnect(self)
            except Exception:
                pass

    async def connect(self):
        """Connect to this device, pairing first if needed, read its
        Device Info, and subscribe to its notifications. Drops any
        connection this object already holds, and returns self."""
        # A SECOND CONNECT MUST NOT LEAVE THE FIRST ONE SUBSCRIBED. This used
        # to overwrite `self._client` and walk away from whatever was in it.
        # The old BleakClient stayed connected and stayed notified on DATA, so
        # every packet arrived twice -- and both copies were stored, because
        # each was a well-formed packet with a valid header. Two connects gave
        # a capture at 2x its sample rate with every device tick repeated;
        # three gave 3x. Nothing failed, and the manifest recorded a rate the
        # samples did not have.
        if self._client is not None:
            await self.disconnect()
        self._client = BleakClient(self._bd, timeout=30,
                                   disconnected_callback=self._disconnected)
        await self._client.__aenter__()
        # PAIR BEFORE THE FIRST ENCRYPTED OPERATION. Every characteristic but
        # Device Info needs an encrypted, bonded link (Just Works, LE Secure
        # Connections). On Linux the system pairs only if a program has
        # registered an agent to answer for the host, so this library
        # registers one and pairs over the same connection, and the system
        # uses that agent for this pairing and nothing else (`pairing`). A
        # device paired from the desktop's settings first is found paired and
        # nothing is asked again. Elsewhere the system pairs by itself, and
        # pairing an already-bonded device raises on some platforms; that is
        # fine: the first encrypted operation below tells, and it is bounded,
        # because a link that cannot be secured used to hang there forever on
        # a pairing that had already failed.
        details = getattr(self._bd, "details", None)
        path = details.get("path") if isinstance(details, dict) else None
        pairing_note = await pairing.pair(path, 20) if path else None
        if pairing_note is None:
            pairing_note = "pairing completed"
            try:
                await asyncio.wait_for(self._client.pair(), 20)
            except Exception as e:                               # noqa: BLE001
                pairing_note = f"pairing reported {type(e).__name__}: {e}"
        self.info = P.decode_device_info(await self._client.read_gatt_char(INFO))
        try:
            await asyncio.wait_for(self._client.start_notify(RSP, self._on_rsp), 15)
        except asyncio.TimeoutError:
            await self.disconnect()
            raise ConnectionError(
                "the link could not be secured: the device's answers could not be "
                f"subscribed to within 15 s ({pairing_note}). The system's Bluetooth "
                "service did not complete pairing. Remove the device from the "
                f"system (on Linux: bluetoothctl remove {self._bd.address}) and "
                "connect again.") from None
        await self._client.start_notify(DATA, self._on_data)
        if self.info.can("model"):
            try:
                await self._client.start_notify(PREDICTIONS, self._on_prediction)
            except Exception:                            # noqa: BLE001
                # A device that claims a model but will not let us listen
                # is still a device that streams. The claim is recorded and
                # the stream is what matters.
                pass
        # What this device can do is what it says it can do. The bits are
        # in Device Info and nothing here infers a capability from a
        # version number or from which characteristics happen to exist,
        # both of which are questions that only correlate with the answer.
        self.adapter = adapters.for_device(self.info)
        self.extra = self.adapter.attach(self) if self.adapter else None
        self.ADAPTER = adapters.identity(self.adapter)
        await self.time_sync()
        return self

    def can(self, capability: str) -> bool:
        """Whether the connected device claims a capability."""
        return bool(self.info and self.info.can(capability))

    def _require(self, capability: str, what: str) -> None:
        if not self.can(capability):
            raise RuntimeError(
                f"this device does not {what}: it does not claim the "
                f"{capability} capability")

    async def disconnect(self):
        """Drop the link. Safe to call on a link that is already down.

        The client is cleared even when the teardown raises: a client that
        could not be closed is still one this object must stop treating as
        its own, and holding the reference is how a failed disconnect turns
        into a second delivery path on the next connect.
        """
        c, self._client = self._client, None
        if c is not None:
            try:
                await c.__aexit__(None, None, None)
            except Exception:                            # noqa: BLE001
                pass
        self.streaming = False
        # The device turns embeddings off at disconnect; so does this side.
        self._embeddings_subscribed = False
        self._embeddings_form = "off"
        self._assembler = None

    def _on_rsp(self, _h, d):
        self._rsp.put_nowait(bytes(d))

    async def command(self, op: int, arg: Optional[int] = None,
                      timeout: float = 6.0, data: Optional[bytes] = None) -> bytes:
        """One control exchange. `arg` is the one-byte form most opcodes use;
        `data` carries the variable-length body of the few that take one,
        such as SET_PIPELINE's chain and SET_NAME's parts."""
        if not self.connected:
            raise ConnectionError("BLE link is down")
        if data is not None:
            payload = bytes([op]) + bytes(data)
        else:
            payload = bytes([op] if arg is None else [op, arg])
        await self._client.write_gatt_char(CTL, payload, response=True)
        while True:
            r = await asyncio.wait_for(self._rsp.get(), timeout)
            if r[0] == op:
                if r[1] != 0:
                    raise Refused(op, r[1])
                return r

    async def _payload(self, name: str, arg: Optional[int] = None,
                       data: Optional[bytes] = None) -> bytes:
        """One control exchange, answering with the payload alone. An answer
        is `[opcode, status, payload]`, and every decoder in `protocol` takes
        the payload; handing it the whole answer reads the opcode as data."""
        r = await self.command(P.OPCODES[name], arg, data=data)
        return bytes(r[2:])

    async def time_sync(self):
        """One sync exchange. Refit the clock map over every exchange so far.

        Previously this fitted a line through the *first and last* exchanges
        only, and `resync_loop` -- the thing that would ever produce a second
        exchange -- was never started by the hub. In practice the map was a
        single point with skew pinned at 1.0, and nothing measured how wrong it
        was. `docs/TIMING.md` described a regression that did not run.

        Both clocks are read on each side of the exchange. The monotonic pair
        is what the device clock is regressed against; the realtime pair is the
        anchor that carries the result back into the record's timebase.
        """
        m1 = time.monotonic()
        t1 = time.time()
        r = await self.command(OP["TIME_SYNC"])
        m2 = time.monotonic()
        t2 = time.time()
        td = int.from_bytes(r[2:10], "little") / self.info.tick_hz
        # The device answered somewhere inside the exchange. The midpoint is
        # the best estimate; half the round trip is the irreducible uncertainty.
        self._syncs.append(Sync(host=(t1 + t2) / 2, device=td,
                                rtt=t2 - t1, mono=(m1 + m2) / 2))
        self._fit_clock()
        return td

    def _fit_clock(self) -> None:
        """Least squares over all exchanges: mono = skew * device + offset.

        Fitted against monotonic, so a wall-clock step leaves it untouched.

        A SLOPE IS ONLY TAKEN WHEN THE BASELINE CAN CARRY ONE. Every exchange
        is a BLE round trip, and the midpoint of a round trip is only as good
        as the trip is symmetric -- tens of milliseconds of jitter on a busy
        radio. `calibrate_clock` spaces its exchanges 0.15 s apart, so the
        whole baseline is under a second, and least squares will happily draw
        a line of ANY slope through six points scattered by 90 ms across it.
        One four-channel session did exactly that and fitted skew = 1.6986:
        every sample stamped over 3304 s of an hour that lasted 1945, with
        nothing in the manifest saying the number was unusable.

        Two crystals are the physical claim being measured, and two crystals
        differ by parts per million. So the slope has to clear both bars
        before it is believed: it must be physically possible, and its own
        standard error must be small enough for it to beat the assumption it
        is replacing. Otherwise skew is pinned at 1.0 and only the offset is
        fitted -- the assumption, stated, rather than noise wearing a slope.
        """
        n = len(self._syncs)
        if n == 0:
            return
        first = min(self._syncs, key=lambda s: s.mono)
        pinned, skew, offset = None, 1.0, 0.0
        if n == 1:
            offset = self._syncs[0].mono - self._syncs[0].device
            pinned = "one exchange: a point has no slope"
        else:
            hs = [s.mono for s in self._syncs]
            ds = [s.device for s in self._syncs]
            dbar, hbar = sum(ds) / n, sum(hs) / n
            sdd = sum((d - dbar) ** 2 for d in ds)
            sdh = sum((d - dbar) * (h - hbar) for d, h in zip(ds, hs))
            skew = (sdh / sdd) if sdd > 0 else 1.0
            offset = hbar - skew * dbar
            pinned = self._reject_skew(skew, offset, sdd, n)
            if pinned:
                skew, offset = 1.0, hbar - dbar
        # What the exchanges say, always. What the samples are stamped with,
        # only while nothing is being stamped -- see `start`.
        self._learned = dict(skew=skew, offset=offset, pinned=pinned,
                             n_syncs=n)
        if self._map_frozen:
            return
        self._frame_mono = first.mono
        self._frame_offset = first.host - first.mono
        self._skew, self._offset, self._skew_pinned = skew, offset, pinned

    def _reject_skew(self, skew: float, offset: float,
                     sdd: float, n: int) -> Optional[str]:
        """Why this slope cannot be believed, or None if it can.

        Two tests, and a slope has to pass both.

        IS IT PHYSICALLY POSSIBLE. `MAX_SKEW_PPM` is what two oscillators can
        differ by. A watch crystal is typically specified at +-20 ppm and
        ages to perhaps +-100, so 1000 ppm is an order of magnitude of
        headroom already, and anything past it is a fit artifact rather than
        a clock.

        IS IT MEASURED. The slope's own standard error, taken from the
        residuals of this very fit. This is the test that catches the near
        misses -- 772 ppm fitted across 0.87 s reads as a plausible crystal
        and is actually round-trip jitter, and the standard error says so
        (+-250 ppm) where the value alone does not. Below `MAX_SKEW_STDERR_PPM`
        needs a baseline of minutes, which is what `Session.resync_loop`
        produces and a burst of exchanges at connect never can.
        """
        if sdd <= 0:
            return "every exchange landed on the same device time"
        err_ppm = abs(skew - 1.0) * 1e6
        if err_ppm > MAX_SKEW_PPM:
            return (f"fitted {err_ppm:,.0f} ppm from unity, past the "
                    f"{MAX_SKEW_PPM:,.0f} ppm two oscillators can differ by")
        if n <= 2:
            return "two exchanges: a slope with no residual to check it"
        resid = [(s.mono - (skew * s.device + offset)) for s in self._syncs]
        sse = sum(r * r for r in resid)
        se_ppm = (((sse / (n - 2)) / sdd) ** 0.5) * 1e6   # stderr of the slope
        if se_ppm > MAX_SKEW_STDERR_PPM:
            span = max(s.device for s in self._syncs) - min(
                s.device for s in self._syncs)
            return (f"slope {err_ppm:,.0f} +- {se_ppm:,.0f} ppm over a "
                    f"{span:.2f} s baseline: not measured well enough to "
                    f"beat assuming 1.0")
        return None

    def anchors(self) -> list[tuple[float, float]]:
        """(realtime, monotonic) pairs — the map between the two clocks."""
        return [(s.host, s.mono) for s in self._syncs]

    def clock_quality(self) -> dict:
        """How much to trust `host_time`. Reported, never assumed.

        `residual_ms` is the RMS deviation of the sync exchanges from the fitted
        line. It cannot be computed from a capture's stored arrays, because
        `host_time` there is *derived* from `device_time` through this very map
        -- a residual taken from those columns is identically zero and would be
        a fabricated number.

        `max_half_rtt_ms` bounds what the residual can ever mean: a BLE control
        round trip is milliseconds and one-sided, so a residual below it is
        transport jitter, not clock error.

        `anchors` and `wall_clock_steps` are what let a reader put events and
        samples on one ruler afterwards. A step is REPORTED, never repaired in
        silence: `timebase.align_events` uses these to undo it, and a capture
        that suffered one says so in its own manifest.
        """
        n = len(self._syncs)
        anchors = self.anchors()
        steps = timebase.find_steps(anchors)
        out = dict(n_syncs=n, skew=self._skew, offset=self._offset,
                   skew_pinned=self._skew_pinned is not None,
                   skew_pinned_why=self._skew_pinned,
                   residual_ms=None, max_half_rtt_ms=None, span_s=None,
                   fitted_against="monotonic",
                   anchors=anchors, frame_mono=self._frame_mono,
                   wall_clock_steps=steps,
                   wall_clock_stepped=bool(steps))
        if n:
            out["max_half_rtt_ms"] = max(s.rtt for s in self._syncs) / 2 * 1e3
        if n >= 2:
            out["span_s"] = self._syncs[-1].device - self._syncs[0].device
            errs = [m - (self._skew * d + self._offset)
                    for m, d in ((s.mono, s.device) for s in self._syncs)]
            out["residual_ms"] = (sum(e * e for e in errs) / n) ** 0.5 * 1e3
        # WHAT THE EXCHANGES SAY VERSUS WHAT THE SAMPLES WERE STAMPED WITH.
        # They are the same number except across a recording, where the map is
        # held still while the exchanges keep arriving (see `start`). The
        # difference is not an error -- it is the drift the sample column
        # still carries, stated in seconds over the span the exchanges cover,
        # so an analysis can correct `device_ticks` or decide it does not
        # matter, rather than having to wonder.
        got = self._learned or {}
        out["skew_measured"] = got.get("skew")
        out["skew_measured_pinned_why"] = got.get("pinned")
        out["map_frozen_at_stream_start"] = bool(self._map_frozen)
        if got.get("skew") and out["span_s"]:
            out["uncorrected_drift_s"] = (got["skew"] - self._skew) \
                * out["span_s"]
        return out

    async def calibrate_clock(self, n: int = 5, gap_s: float = 0.15) -> dict:
        """Several exchanges up front, so the map starts with an error bar.

        It does NOT start with a usable slope, and it never could: a second of
        baseline cannot measure parts per million, so `_fit_clock` will pin
        skew at 1.0 and say so. What this buys is the offset, averaged over
        several round trips instead of taken from one, and a residual that
        states how good that offset is. The slope arrives later or not at all,
        from `Session.resync_loop` running across the recording itself.
        """
        for i in range(max(1, n)):
            if i:
                await asyncio.sleep(gap_s)
            await self.time_sync()
        return self.clock_quality()

    def mono_time(self, device_ticks: int) -> float:
        """Device ticks on CLOCK_MONOTONIC. The step-proof form."""
        return (device_ticks / self.info.tick_hz) * self._skew + self._offset

    def host_time(self, device_ticks: int) -> float:
        """Device ticks in the record's realtime frame.

        Pinned to the offset measured at the FIRST anchor, so this column stays
        a continuous ruler for the whole recording. If the wall clock steps
        mid-capture the samples do not jump with it -- the step is recorded in
        `clock_quality()["wall_clock_steps"]` instead, where it can be read and
        applied deliberately rather than being baked into the data.
        """
        return self.mono_time(device_ticks) + self._frame_offset

    async def set_gain(self, gain: int):
        """Set the analog gain. Raises ValueError if `gain` is not one
        of `gain_options()`."""
        if gain not in self.gain_options():
            raise ValueError(f"this device has no gain {gain}× "
                             f"(it has {self.gain_options()})")
        await self.command(OP["SET_GAIN"], GAIN_BY_CODE.index(gain))
        self._gain = gain

    async def set_rate(self, sps: int):
        """Set the sample rate, in samples per second, to one of
        `rate_options()`."""
        code = {v: k for k, v in RATE_BY_CODE.items()}[sps]
        await self.command(OP["SET_RATE"], code)

    async def set_mode(self, mode):
        """The signal source: normal (the electrodes), test (the converter's
        test signal), short (inputs shorted), or, on a device that claims
        it, synthetic (1.3: the converter is not driven and the device
        generates the signal at 500 samples per second). A name or a code.
        The device refuses any mode while streaming, and synthetic at a rate
        other than 500; a converter mode leaves synthetic. Never persisted.
        """
        if isinstance(mode, str):
            if mode not in P.MODES:
                raise ValueError(f"the mode is one of {list(P.MODES)}, not {mode!r}")
            mode = P.MODES[mode]
        if mode == P.MODES["synthetic"]:
            self._require("synthetic", "generate a synthetic signal")
        await self.command(OP["SET_MODE"], int(mode))

    # The register level is not part of this contract. A device built on a
    # converter whose registers someone needs reaches it through an
    # adapter, which is a separate package, and `device.extra` is where it
    # appears. Nothing in the public protocol reads or writes a register,
    # and the numbers those operations used are reserved forever so an old
    # host talking to a new device fails loudly rather than being misread.

    async def set_leadoff(self, on: bool):
        """Turn the lead-off comparators on or off.

        They tell a host which electrodes are attached, and they inject a
        small current to do it, which is why they are not on by default.
        """
        self._require("leadoff", "detect lead-off")
        await self.command(OP["SET_LEADOFF"], 1 if on else 0)

    async def battery(self) -> "P.Battery":
        """What the battery is doing, when the device can measure it."""
        self._require("battery_voltage", "report its battery")
        return P.decode_battery(await self._payload("get_battery"))

    async def boot_info(self) -> "P.BootInfo":
        """Which firmware the device started, why, and whether that image
        has confirmed itself. An image that has not is on trial and will be
        rolled back if it cannot."""
        return P.decode_boot_info(await self._payload("get_boot_info"))

    async def clear_bonds(self):
        """Forget every host but this one. Takes effect at the next
        disconnect, which is also when the device stops answering to the
        others."""
        await self.command(P.OPCODES["clear_bonds"])

    # --- the model, on devices that carry one --------------------------------

    async def model_info(self) -> "P.ModelInfo":
        """What the device's model can do and is doing: its state,
        active head, encoder, weights version, and timing."""
        self._require("model", "run a model")
        return P.decode_model_info(await self._payload("get_model_info"))

    async def heads(self) -> tuple[int | None, list]:
        """The head slots, which one is selected, and the encoder each head
        names."""
        self._require("heads", "hold heads")
        active, heads = P.decode_heads(await self._payload("list_heads"))
        if self.info.protocol >= (1, 4):
            try:
                encoders = P.decode_head_encoders(await self._payload("list_head_encoders"))
            except Refused:
                # A device that refuses this has not said which encoder its
                # heads name, so each is shown as a head that does not say.
                encoders = {}
            heads = P.with_encoders(heads, encoders)
        return active, heads

    async def select_head(self, slot: int):
        """Select the head in `slot` as the one that produces
        predictions."""
        self._require("heads", "hold heads")
        await self.command(P.OPCODES["select_head"], slot)

    async def remove_head(self, slot: int):
        """Erase the head in `slot`."""
        self._require("heads", "hold heads")
        await self.command(P.OPCODES["remove_head"], slot)

    # --- processing, on devices that run a chain -----------------------------
    #
    # The device preprocesses its own signal. When a chain is in force only
    # its output streams, and the chain is readable at any time, so a
    # recording can say exactly what produced its samples. The rules a chain
    # must meet are the device's, restated in `protocol.check_chain`, so a
    # chain that cannot run is refused here in words before anything is
    # sent. The device refuses a change while streaming: stop, set, start.

    async def _rate_now(self) -> int:
        rate = self.config.get("rate_sps") if self.config else None
        if not rate:
            rate = (await self.refresh_config()).get("rate_sps")
        return int(rate or 500)

    async def _catalog(self) -> list:
        if self._catalog_cache is None:
            self._catalog_cache = await self.pipeline_catalog()
        return self._catalog_cache

    async def pipeline_catalog(self) -> list:
        """The stage kinds this device runs, from the device."""
        self._require("pipeline", "run a processing chain")
        return P.decode_catalog(await self._payload("get_pipeline_catalog"))

    async def pipeline(self) -> "P.PipelineState":
        """The chain in force, and whether it is the device's own default for
        its current rate or one a host set."""
        self._require("pipeline", "run a processing chain")
        return P.decode_pipeline_state(await self._payload("get_pipeline"))

    async def set_pipeline(self, stages) -> None:
        """Put a chain on the device's signal path. Build stages with
        `protocol.highpass`, `protocol.lowpass`, and `protocol.notch`."""
        self._require("pipeline", "run a processing chain")
        stages = list(stages)
        P.check_chain(stages, await self._rate_now(), await self._catalog())
        await self.command(P.OPCODES["set_pipeline"], data=P.encode_chain(stages))

    async def clear_pipeline(self) -> None:
        """The natural signal, as this host's choice."""
        self._require("pipeline", "run a processing chain")
        await self.command(P.OPCODES["clear_pipeline"])

    async def restore_pipeline_default(self) -> None:
        """The device's own default chain for its current rate."""
        self._require("pipeline", "run a processing chain")
        await self.command(P.OPCODES["restore_pipeline_default"])

    async def prediction_input(self) -> "P.PredictionInput":
        """Where the model's input comes from, with the chain in effect."""
        self._require("model", "run a model")
        return P.decode_prediction_input(await self._payload("get_prediction_input"))

    async def set_prediction_input(self, source: str, stages=()) -> None:
        """Point the model at the stream as it is, at the natural signal, or at
        a chain of its own, without touching the stream. Holds until changed
        or until predictions are turned off. Allowed while streaming."""
        self._require("model", "run a model")
        stages = list(stages)
        if source == "own_chain":
            P.check_chain(stages, await self._rate_now(), await self._catalog())
        await self.command(P.OPCODES["set_prediction_input"],
                           data=P.encode_prediction_input(source, stages))

    async def set_bias(self, mode: str) -> None:
        """The bias drive: off, on, or loop open, on a device that claims
        one. The IntoMind One does not."""
        self._require("bias_drive", "drive a bias electrode")
        if mode not in P.BIAS_MODES:
            raise ValueError(f"bias mode is one of {list(P.BIAS_MODES)}, not {mode!r}")
        await self.command(P.OPCODES["set_bias"], P.BIAS_MODES[mode])

    async def bias_diagnostic(self) -> "P.BiasDiagnostic":
        """The bias output measured, on a device that claims a bias drive."""
        self._require("bias_drive", "drive a bias electrode")
        return P.decode_bias_diagnostic(await self._payload("get_bias_diagnostic"))

    # --- the lamp, the converter's registers, embeddings (1.2) ---------------

    async def indicator(self) -> str:
        """The status lamp's level: silent, reserved, or verbose."""
        self._require("indicator", "have a status lamp a host can set")
        level = (await self._payload("get_indicator"))[0]
        return P.INDICATOR_LEVEL_NAMES.get(level, f"level {level}")

    async def set_indicator(self, level: str) -> None:
        """Put the lamp at a level. The device keeps it across power cycles.

        silent shows nothing; reserved, the default, shows low battery and a
        converter fault at power on; verbose adds waiting for a host,
        connected, streaming, and update in progress. A blink means the same
        thing at every level that shows it.
        """
        self._require("indicator", "have a status lamp a host can set")
        if level not in P.INDICATOR_LEVELS:
            raise ValueError(f"the lamp's level is one of {list(P.INDICATOR_LEVELS)}, not {level!r}")
        await self.command(P.OPCODES["set_indicator"], P.INDICATOR_LEVELS[level])

    async def identify(self, seconds: int = 5) -> None:
        """Blink the lamp so this device can be told from others in the room,
        for 1 to 30 seconds, at every level. 0 stops one in progress."""
        self._require("indicator", "have a status lamp a host can set")
        if not 0 <= int(seconds) <= P.IDENTIFY_MAX_SECONDS:
            raise ValueError(f"identify runs 1 to {P.IDENTIFY_MAX_SECONDS} seconds, or 0 to stop")
        await self.command(P.OPCODES["identify"], int(seconds))

    # --- 1.3: the model's cadence and the device's name ----------------------

    async def model_interval(self) -> "P.ModelInterval":
        """How often the model describes a window, and the least interval
        the device keeps, in seconds. Zero in force is every window the
        device can, each the newest complete one."""
        self._require("model_cadence", "let a host set the model's cadence")
        return P.decode_model_interval(await self._payload("get_model_interval"))

    async def set_model_interval(self, seconds: int) -> None:
        """One described window every `seconds`, on a grid from the stream's
        start; 0 is every window the device can. The device refuses a value
        below its minimum and keeps the setting across power cycles."""
        self._require("model_cadence", "let a host set the model's cadence")
        await self.command(OP["SET_MODEL_INTERVAL"], data=P.encode_model_interval(seconds))

    async def get_name(self) -> tuple[str, str]:
        """The name and the adjective a host set, each possibly empty. The
        device composes them as `compose_name` does and advertises the
        result, which is `device.name` once a scan has heard it."""
        self._require("device_name", "be named")
        return P.decode_name_parts(await self._payload("get_name"))

    async def set_name(self, name: str, adjective: str) -> None:
        """Name the device. It advertises `name's adjective IntoMind One`
        from the next advertisement, shows it to a connected host after its
        next restart, and keeps both parts across power cycles. The
        composition must fit the 29 bytes a scan response carries, so every
        scan list shows the whole of it; `P.name_fits` says whether it does.
        """
        self._require("device_name", "be named")
        if not P.name_fits(name, adjective):
            raise ValueError(
                f"{P.compose_name(name, adjective)!r} does not fit the {P.NAME_MAX_COMPOSED} bytes the air "
                "carries, or a part has a leading or trailing space or a control character")
        await self.command(OP["SET_NAME"], data=P.encode_name_parts(name, adjective))

    async def converter_registers(self) -> "P.ConverterRegisters":
        """The analog converter's configuration registers, raw, for debugging.

        Read only, and only while not streaming: the device answers busy
        while it converts. `.describe()` puts the bytes in words for the
        converter families this version knows.
        """
        self._require("converter_registers", "read its converter's registers")
        return P.decode_converter_registers(await self._payload("get_converter_registers"))

    async def set_embeddings(self, form: str, on_window=None) -> None:
        """Ask the device for its encoder's output for each window while it
        streams.

        `form` is "window" for the one vector a head consumes, "tokens" for
        the per-slice vectors reconstruction consumes, "both", or "off".
        Whole windows arrive through `on_embedding(device, window)`; pass
        `on_window` to set it here. No head needs to be selected. The
        weights never leave the device: these are its outputs.
        """
        self._require("embeddings", "send its encoder's output")
        if form not in P.EMBEDDING_FORMS:
            raise ValueError(f"the form is one of {list(P.EMBEDDING_FORMS)}, not {form!r}")
        if form != "off":
            info = await self.model_info()
            if form in ("tokens", "both") and info.tokens_per_channel == 0:
                raise RuntimeError("this device's model declares no tokens, so it cannot send the token form")
            self._assembler = P.EmbeddingAssembler(self.info.channels, info.tokens_per_channel, form)
            if on_window is not None:
                self.on_embedding = on_window
            if not self._embeddings_subscribed:
                try:
                    await self._client.start_notify(EMBEDDINGS, self._on_embedding)
                except Exception as e:                       # noqa: BLE001
                    raise RuntimeError(
                        "this device claims embeddings, and the system's Bluetooth "
                        "service shows no characteristic for them: it is holding an "
                        "older description of the device from before its firmware "
                        f"was updated. Remove the device from the system (on Linux: "
                        f"bluetoothctl remove {self._bd.address}) and connect again. "
                        f"({type(e).__name__}: {e})") from None
                self._embeddings_subscribed = True
        await self.command(P.OPCODES["set_embeddings"], P.EMBEDDING_FORMS[form])
        self._embeddings_form = form
        if form == "off":
            self._assembler = None

    async def collect_embeddings(self, windows: int, form: str = "window",
                                 timeout: float | None = None) -> list:
        """Gather whole windows of the encoder's output while streaming, then
        turn the form off again. The stream must already be running.

        A window is four seconds of signal, and the device may describe
        fewer than every window (`model_interval`), so unless `timeout` says
        otherwise the wait allows thirty seconds a window and a minute more.
        """
        if timeout is None:
            timeout = 60.0 + 30.0 * windows
        got: list = []
        done = asyncio.Event()

        def take(_dev, w):
            got.append(w)
            if len(got) >= windows:
                done.set()

        previous = self.on_embedding
        await self.set_embeddings(form, on_window=take)
        try:
            await asyncio.wait_for(done.wait(), timeout)
        finally:
            self.on_embedding = previous
            try:
                await self.set_embeddings("off")
            except Exception:                            # noqa: BLE001
                pass
        return got[:windows]

    def _on_embedding(self, _h, d):
        if self._assembler is None:
            return
        try:
            e = P.decode_embedding(bytes(d))
        except P.ProtocolError:
            self.undecodable += 1
            return
        if self.on_embedding_packet:
            self.on_embedding_packet(self, e)
        w = self._assembler.feed(e)
        if w is not None and self.on_embedding:
            self.on_embedding(self, w)

    # --- carrying an update, and uploading a head ---------------------------

    async def _update(self, request: bytes, timeout: float = 5.0) -> "P.UpdateResponse":
        """One update service exchange."""
        if self._update_rsp is None:
            self._update_rsp = asyncio.Queue()
            await self._client.start_notify(
                UPDATE_CONTROL, lambda _h, d: self._update_rsp.put_nowait(bytes(d)))
        while not self._update_rsp.empty():
            self._update_rsp.get_nowait()
        await self._client.write_gatt_char(UPDATE_CONTROL, request, response=True)
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"the update service did not answer within {timeout} seconds")
            data = await asyncio.wait_for(self._update_rsp.get(), timeout=left)
            r = P.decode_update_response(data)
            if r.op == request[0]:
                return r

    #: Data writes between progress checks during a transfer, and how many
    #: times a transfer may carry on from the device's own count.
    TRANSFER_BURST = 8
    TRANSFER_RESUMES = 64

    async def _update_retrying(self, request: bytes, attempts: int = 3) -> "P.UpdateResponse":
        """An update exchange that a device still writing its flash may turn
        away: tried again after a pause, a few times, then given up."""
        for attempt in range(attempts):
            try:
                return await self._update(request)
            except (TimeoutError, asyncio.TimeoutError, BleakError):
                if attempt == attempts - 1:
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))

    async def transfer(self, image: bytes, target: str = "app", slot: int = 0,
                       progress=None) -> str:
        """Carry a signed image to the device and have it verified there.

        A host carries an update without being able to read or forge one:
        everything past the envelope is encrypted and means nothing here.
        The device writes what arrives, checks the signature against what
        is in its own flash, and says what it found. Nothing is activated
        by this call.

        The device's own count and checksum are compared against ours as
        the transfer runs, so a truncated or reordered write is caught
        where it happened rather than at the end.
        """
        import binascii, hashlib

        self._require("update", "accept updates")
        transfer_id = hashlib.sha256(image).digest()[:8]
        r = await self._update(P.encode_update_start(target, slot, len(image), transfer_id))
        if not r.ok:
            raise RuntimeError(f"the device refused to start the transfer: {r.status_name}")
        _target_slot, chunk_max, offset = r.start()
        chunk = min(chunk_max, self.info.update_chunk_max or chunk_max, 244)
        # SHORT BURSTS, AND THE DEVICE'S OWN COUNT. Writes without a response
        # have no flow control, and a device busy writing its flash drops
        # some. Asking after every eight keeps the queue shallow, and when
        # the device has fewer bytes than were sent the transfer carries on
        # from the device's count, which is what the resume offset is for.
        # Bounded: a transfer that stops advancing is a fault, not a delay.
        since_query, resumes, resumed_at = 0, 0, -1
        while offset < len(image):
            end = min(offset + chunk, len(image))
            await self._client.write_gatt_char(UPDATE_DATA, image[offset:end], response=False)
            offset = end
            since_query += 1
            if since_query >= self.TRANSFER_BURST or offset == len(image):
                since_query = 0
                q = await self._update_retrying(P.encode_update_op("query"))
                if not q.ok:
                    raise RuntimeError(f"the device refused a progress check: {q.status_name}")
                _state, device_offset, device_crc = q.query()
                if device_offset != offset:
                    resumes += 1
                    if resumes > self.TRANSFER_RESUMES or device_offset <= resumed_at:
                        raise RuntimeError(
                            f"the device accepted {device_offset} bytes where {offset} were sent, "
                            f"and the transfer is not advancing")
                    resumed_at = offset = device_offset
                    continue
                ours = binascii.crc32(image[:offset]) & 0xFFFFFFFF
                if device_crc != ours:
                    raise RuntimeError(
                        f"the device's checksum over {offset} bytes is {device_crc:#010x} "
                        f"and ours is {ours:#010x}")
                if progress:
                    progress(offset, len(image))
        f = await self._update(P.encode_update_op("finish"), timeout=20.0)
        result = f.verify_result
        if not f.ok:
            raise RuntimeError(f"the device refused the image: {result or f.status_name}")
        return result or "verified"

    async def abort_transfer(self):
        """Cancel an update transfer in progress. Any error from the
        device is ignored."""
        try:
            await self._update(P.encode_update_op("abort"))
        except Exception:                                # noqa: BLE001
            pass

    async def activate(self):
        """Put a transferred image in force. The device resets, so the link
        drops and everything this object knows about it is stale."""
        r = await self._update(P.encode_update_op("activate"))
        if not r.ok:
            raise RuntimeError(f"the device refused to activate the image: {r.status_name}")
        self.streaming = False

    async def upload_head(self, blob: bytes, slot: int, *, select: bool = False) -> str:
        """Put a head in a slot, and optionally select it.

        The blob carries its own hash, which the device checks, so a head
        that arrived wrong is refused rather than run.
        """
        self._require("heads", "hold heads")
        head = P.decode_head(blob)
        if self.info.model_embed_dim and head.in_dim != self.info.model_embed_dim:
            raise ValueError(
                f"this head takes an embedding of {head.in_dim} and the device's "
                f"model produces {self.info.model_embed_dim}")
        # 1.3: a head names the encoder it was trained beside. A mismatch is
        # said out loud and nothing stops: the head is yours, the device
        # stores it, and only you know whether it still means anything.
        if head.encoder_id != P.NO_ENCODER_ID and self.info.can("model"):
            try:
                running = (await self.model_info()).encoder_id
            except Exception:                       # noqa: BLE001
                running = None
            if running and running != head.encoder_id:
                warnings.warn(
                    f"this head was trained beside encoder {head.encoder_id} and the device runs "
                    f"{running}; it is stored and will run, and its outputs may mean nothing",
                    stacklevel=2)
        await self.transfer(blob, target="head", slot=slot)
        if select:
            await self.select_head(slot)
        return head.head_id

    async def set_predictions(self, on: bool):
        """Turn the model's output on or off.

        A device refuses this until a head is selected, because saying yes
        would promise notifications that never arrive.
        """
        self._require("model", "run a model")
        await self.command(P.OPCODES["set_predictions"], 1 if on else 0)
        self._predictions = bool(on)

    async def read_status(self) -> "Status":
        """The Status characteristic: state, mode, gain, rate, contact, buffer.

        Present in the v0.1 contract and implemented by every firmware, so it is
        the one configuration read that does not depend on a capability. Read,
        never subscribed: a caller that wants the heartbeat can notify on STAT
        itself.
        """
        b = await self._client.read_gatt_char(STAT)
        if len(b) < STATUS_LEN:
            raise RuntimeError(f"status characteristic returned {len(b)} bytes, "
                               f"expected {STATUS_LEN}")
        self.status = P.decode_status(bytes(b))
        return self.status

    async def refresh_config(self) -> dict:
        """What the device is set to, read from the device.

        The status characteristic is the one configuration read every
        device has, and it is the whole answer under this contract. There
        is no register path here, and a device that wants one brings an
        adapter.
        """
        st = await self.read_status()
        self.config = dict(gain=st.gain, rate_sps=st.sample_rate_hz,
                           mode=st.mode, leadoff=bool(st.leadoff))
        self._gain = st.gain or self._gain
        if self.can("pipeline"):
            chain = await self.pipeline()
            self.config["processing"] = dict(
                origin=chain.origin, stages=[s.as_dict() for s in chain.stages],
                description=P.describe_chain(chain.stages, st.sample_rate_hz))
        if self.can("indicator"):
            self.config["indicator"] = await self.indicator()
        if self.can("model_cadence"):
            mi = await self.model_interval()
            self.config["model_interval_s"] = mi.interval_s
            self.config["model_interval_min_s"] = mi.minimum_s
        if self.can("device_name"):
            name, adjective = await self.get_name()
            self.config["name"] = name
            self.config["adjective"] = adjective
            self.config["composed_name"] = P.compose_name(name, adjective)
        return dict(self.config)

    def gain_options(self) -> list:
        """The gains this contract defines. Every device that speaks it
        accepts all of them, so there is nothing to ask."""
        return list(P.GAIN_BY_CODE)

    def rate_options(self) -> list:
        """The sample rates this device declares in its own Device Info."""
        rates = self.info.rates if self.info else []
        return rates or sorted(RATE_BY_CODE.values())

    async def start(self, samples_per_packet: Optional[int] = None, new_epoch=True):
        """Start streaming. `samples_per_packet` is a preference, clamped
        to what fits one notification on a 247-byte link, and the default
        is as many as fit. Resets the sample index first unless `new_epoch`
        is false."""
        # A data packet must fit one notification: 247-byte MTU - 3 ATT
        # header - 20 packet header, over 3 bytes per channel per sample.
        # The firmware enforces its own cap (INVALID_ARG) and a request
        # sized for 2 channels (25) is over the 4-channel limit (18), so
        # the caller's number is a preference, not a command.
        #
        # The default is as many as fit, which is the device's own default.
        # Fewer per packet means more packets a second, and a link that
        # cannot carry them costs samples: measured on a unit at 500 samples
        # a second, 10 per packet lost one sample in six for five minutes,
        # 18 per packet lost none. A caller who wants lower latency asks.
        fit = (247 - 3 - 20) // (3 * self.info.channels)
        samples_per_packet = fit if samples_per_packet is None else max(1, min(samples_per_packet, fit))
        self.ended_by_usb = False
        await self.command(OP["SET_SPP"], samples_per_packet)
        if new_epoch:
            await self.reset_epoch()
        # What produces the samples about to arrive. It cannot change while
        # streaming, so one read here is the truth for the whole stream.
        self.pipeline_at_start = await self.pipeline() if self.can("pipeline") else None
        await self.command(OP["START"])
        # THE MAP THAT DATES SAMPLES STOPS MOVING WHILE SAMPLES ARE ARRIVING.
        # Exchanges keep happening and keep being recorded -- they are how the
        # skew becomes measurable at all, since a baseline of minutes is the
        # only thing that can measure parts per million -- but re-fitting the
        # line mid-stream would date the second half of a recording on a
        # different ruler from the first, and put a step in the middle of it
        # exactly where nothing happened. The fit that was in force when the
        # stream started is the fit the whole stream is stamped with; what was
        # learned afterwards is REPORTED, in `clock_quality`, for an analysis
        # to apply to `device_ticks` if it wants to.
        self._map_frozen = True
        self.streaming = True

    async def reset_epoch(self):
        """Zero the sample-index counter. Does NOT stop the stream -- it only
        re-bases the index, so a recording can establish a clean per-capture
        epoch on the already-running live stream."""
        await self.command(OP["RESET_EPOCH"])
        self._expected_idx = None

    async def stop(self):
        """Stop streaming. The device is marked as stopped even if the
        command itself fails."""
        try:
            await self.command(OP["STOP"])
        finally:
            self.streaming = False
            self._map_frozen = False

    def _on_data(self, _h, d):
        try:
            pkt = P.decode_packet(bytes(d), self.info.channels)
        except ProtocolError:
            # A notification that is not a packet is dropped and counted,
            # never guessed at. Guessing is how a bad byte becomes a
            # plausible sample.
            self.undecodable += 1
            return
        self._gain = pkt.gain              # from the wire, not from memory
        self._rate_wire = pkt.sample_rate_hz or 0
        ns = pkt.n_samples
        # A packet already delivered is dropped and counted. The system's
        # Bluetooth service sends each notification once per program
        # subscribed, and every subscriber hears all of the copies (measured
        # 2026-09-28); counted as samples they doubled the rate and flagged
        # a gap before every packet. A copy is the same packet: the same
        # first index and the same device time. A device that restarts its
        # count carries a new device time, so a restart, flagged or not, is
        # never taken for a copy and is judged as the contract says.
        key = (pkt.index, pkt.device_time)
        if key in self._recent:
            self.duplicates += 1
            if not self.another_listener:
                # Said once as it starts, not once a packet.
                warnings.warn(ANOTHER_LISTENER, RuntimeWarning, stacklevel=2)
            self._last_duplicate = time.monotonic()
            return
        gap = pkt.discontinuity or (self._expected_idx is not None
                                    and pkt.index != self._expected_idx)
        if gap:
            self.gaps_announced += 1
        self._recent.append(key)
        self._expected_idx = (pkt.index + ns) & 0xFFFFFFFF
        # 1.4: a device that does not record while plugged in ends a stream
        # of the natural signal when USB power appears, and its packets say
        # so from then on. Their samples were converted before, and are
        # delivered. The generated signal runs on USB power and is not ended.
        if (pkt.usb_present and not pkt.synthetic and self.streaming
                and self.info.protocol >= (1, 4)):
            self.streaming = False
            self._map_frozen = False
            self.ended_by_usb = True
            if self.on_usb_stop:
                self.on_usb_stop(self)
        if self.on_packet:
            self.on_packet(self, pkt)
        if self.on_sample:
            # The device reports its own resolution and reference. Nothing
            # here assumes either, which is how a comparison between two
            # devices stays a comparison.
            lsb_uv = self.info.microvolts_per_count(pkt.gain)
            tick_per = self.info.tick_hz / pkt.sample_rate_hz if pkt.sample_rate_hz else 0
            for i, row in enumerate(pkt.counts):
                dt = int(pkt.device_time + i * tick_per)
                self.on_sample(self, Sample(
                    pkt.index + i, dt, self.host_time(dt), tuple(row),
                    tuple(round(c * lsb_uv, 3) for c in row),
                    pkt.leadoff, gap_before=(gap and i == 0),
                    synthetic=pkt.synthetic))
        self.samples_total += ns

    def _on_prediction(self, _h, d):
        """One window's model output, if anything is listening."""
        if not self.on_prediction:
            return
        try:
            self.on_prediction(self, P.decode_prediction(bytes(d)))
        except ProtocolError:
            self.undecodable += 1

    def fs_effective(self, packets: list) -> float:
        """True sample rate from (index, device_time) regression."""
        if len(packets) < 2:
            return 0.0
        p0, p1 = packets[0], packets[-1]
        span = (p1.device_time - p0.device_time) / self.info.tick_hz
        return (p1.first_index - p0.first_index) / span if span > 0 else 0.0
