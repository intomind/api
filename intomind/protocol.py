"""The IntoMind BLE protocol, version 1.4.

This module is the wire and nothing else. It has no device in it, no
transport, no state, and no opinion about what a caller does with a
message. Every function takes bytes and returns what the contract says
those bytes mean, or raises `ProtocolError` saying why they are not a
message.

It is held to `contract/conformance.json`, which the firmware emits from
the codec it runs. Every message here decodes those vectors to the stated
fields and refuses the malformed ones for the stated reason, so this
implementation and the device cannot drift apart without a test failing.

Three things are worth knowing before reading further.

**Capabilities are the device's answer.** What a device can do is in the
bits it reports, never in a table of hardware here. A device that does not
claim a capability is not asked for it.

**The timeline is never silently repaired.** A forward step in the sample
index is a loss with an exact count. A step that is not forward is a break
whose extent is not a number, and this module says so rather than
returning a count that would be a fiction.

**Nothing is assumed about width.** The number of channels, the converter
resolution, the reference voltage, and the tick rate all come from the
device, and the scaling helper takes them as arguments.
"""
from __future__ import annotations

import dataclasses
import struct
from dataclasses import dataclass, field

VERSION = (1, 4)

UUID_BASE = "f3a1{:04x}-2c4b-4d1e-9a6f-1b2c3d4e5f60"


def uuid(fill: int) -> str:
    """The full identifier for one characteristic."""
    return UUID_BASE.format(fill)


SERVICE = uuid(0x0001)
DEVICE_INFO = uuid(0x0002)
CONTROL = uuid(0x0003)
CONTROL_RESPONSE = uuid(0x0004)
EEG_DATA = uuid(0x0005)
STATUS = uuid(0x0006)
UPDATE_CONTROL = uuid(0x0008)
UPDATE_DATA = uuid(0x0009)
PREDICTIONS = uuid(0x000A)
#: 1.2: the encoder's output for each window.
EMBEDDINGS = uuid(0x000B)

#: The name a device advertises under, before its own identifier.
NAME_PREFIX = "IntoMind-"
#: 1.3: the product name a composed device name ends with. A 1.3 device
#: advertises `[name's] [adjective] IntoMind One`; a 1.2 device advertised
#: `IntoMind-` and four hex digits.
PRODUCT_NAME = "IntoMind One"


class ProtocolError(Exception):
    """Bytes that are not a message, with the reason they are not."""


class Truncated(ProtocolError):
    """The message ends before the fields it promises."""


class Invalid(ProtocolError):
    """A field holds a value the contract does not define."""


class Reserved(ProtocolError):
    """A number the contract reserves. A device answers these unsupported
    rather than treating them as malformed, and so does a host."""


# --- the control point ------------------------------------------------------

OPCODES = {
    "start_stream": 0x01,
    "stop_stream": 0x02,
    "set_rate": 0x10,
    "set_gain": 0x11,
    "set_mode": 0x20,
    "set_leadoff": 0x30,
    "time_sync": 0x40,
    "set_samples_per_packet": 0x41,
    "get_battery": 0x42,
    "get_boot_info": 0x43,
    "reset_epoch": 0x50,
    "clear_bonds": 0x53,
    "set_predictions": 0x80,
    "select_head": 0x81,
    "list_heads": 0x82,
    "remove_head": 0x83,
    "get_model_info": 0x84,
    # New in 1.1.
    "set_prediction_input": 0x85,
    "get_prediction_input": 0x86,
    "set_bias": 0x32,
    "get_bias_diagnostic": 0x33,
    "get_pipeline_catalog": 0x90,
    "get_pipeline": 0x91,
    "set_pipeline": 0x92,
    "clear_pipeline": 0x93,
    "restore_pipeline_default": 0x94,
    # New in 1.2.
    "get_converter_registers": 0x34,
    "get_indicator": 0x44,
    "set_indicator": 0x45,
    "identify": 0x46,
    "set_embeddings": 0x87,
    # New in 1.3.
    "set_model_interval": 0x88,
    "get_model_interval": 0x89,
    "get_name": 0x47,
    "set_name": 0x48,
    # New in 1.4.
    "list_head_encoders": 0x8A,
    "soft_reset": 0xF0,
}
NAME_BY_OPCODE = {v: k for k, v in OPCODES.items()}

#: Opcodes that take one argument byte, and require it.
TAKES_ARG = {"set_rate", "set_gain", "set_mode", "set_leadoff",
             "set_samples_per_packet", "set_predictions", "select_head",
             "remove_head", "set_bias", "set_indicator", "identify",
             "set_embeddings"}

#: Opcodes new in 1.1 that carry a payload after the opcode byte, laid out
#: as the processing section below says. These are the only requests longer
#: than two bytes.
TAKES_PAYLOAD = {"set_pipeline", "set_prediction_input", "set_model_interval", "set_name"}

#: Opcodes a 1.0 device does not know. A host sends one only to a device
#: whose Device Info claims the capability behind it.
NEW_IN_1_1 = {"set_prediction_input", "get_prediction_input", "set_bias",
              "get_bias_diagnostic", "get_pipeline_catalog", "get_pipeline",
              "set_pipeline", "clear_pipeline", "restore_pipeline_default"}

#: Opcodes a 1.1 device does not know, likewise gated on a capability.
NEW_IN_1_2 = {"get_converter_registers", "get_indicator", "set_indicator",
              "identify", "set_embeddings"}
#: Opcodes a 1.2 device does not know: the model's cadence (capability bit
#: 15) and the device's name (capability bit 17, in the second word).
NEW_IN_1_3 = {"set_model_interval", "get_model_interval", "get_name", "set_name"}

#: Numbers an earlier development firmware used, and numbers the contract
#: keeps for factory and bench builds. Never reused for anything else.
RESERVED_OPCODES = frozenset({0x12, 0x13, 0x31, 0x70, 0x71} | set(range(0x60, 0x70)))

STATUS_CODES = {0: "ok", 1: "invalid argument", 2: "unsupported", 3: "busy",
                4: "not streaming", 5: "hardware", 7: "usb power"}
#: 1.4: a stream of the natural signal refused while USB power is present.
#: 6 stays unassigned because the Update service uses it.
STATUS_USB_POWER = 7

GAIN_BY_CODE = [1, 2, 4, 6, 8, 12, 24]
CODE_BY_GAIN = {g: i for i, g in enumerate(GAIN_BY_CODE)}
RATE_BY_CODE = {4: 1000, 5: 500, 6: 250}
CODE_BY_RATE = {v: k for k, v in RATE_BY_CODE.items()}
#: The bit in Device Info's rate mask that stands for each rate.
RATE_BY_INFO_BIT = {0: 250, 1: 500, 2: 1000}

#: 1.3 adds synthetic: the converter is not driven and the device generates
#: the signal, at 500 samples per second, on a device that claims it.
MODES = {"normal": 0, "test": 1, "short": 2, "synthetic": 3}
MODE_BY_CODE = {v: k for k, v in MODES.items()}

CHARGER_STATES = {0: "no input", 1: "charging", 2: "complete", 3: "fault",
                  4: "standby"}

BOOT_REASONS = {0: "power on", 1: "reset pin", 2: "software", 3: "watchdog",
                4: "fault", 5: "update", 6: "rollback", 0xFF: "unknown"}


def encode_request(opcode, arg=None) -> bytes:
    """One control point write.

    `opcode` is a name or a number. An opcode that takes an argument
    requires one, and one that does not takes none, because the device
    refuses either mistake and a host should not make it.
    """
    if isinstance(opcode, str):
        name, code = opcode, OPCODES[opcode]
    else:
        code, name = opcode, NAME_BY_OPCODE.get(opcode)
    if name is None:
        raise Invalid(f"{code:#04x} is not an opcode this contract defines")
    if name in TAKES_PAYLOAD:
        if not isinstance(arg, (bytes, bytearray)):
            raise Invalid(f"{name} takes a payload of bytes")
        return bytes([code]) + bytes(arg)
    if (name in TAKES_ARG) != (arg is not None):
        want = "an argument" if name in TAKES_ARG else "no argument"
        raise Invalid(f"{name} takes {want}")
    return bytes([code]) if arg is None else bytes([code, arg])


def decode_request(data: bytes) -> tuple[str, int | bytes | None]:
    """A control point write, as the device reads it. For tests and for
    anything that has to speak both sides of the contract. For an opcode
    that takes a payload, the second value is the payload's bytes, which
    the processing section's decoders judge."""
    if not data:
        raise Truncated("an empty write is not a request")
    code, rest = data[0], data[1:]
    if code in RESERVED_OPCODES:
        raise Reserved(f"{code:#04x} is reserved")
    name = NAME_BY_OPCODE.get(code)
    if name is None:
        raise Invalid(f"{code:#04x} is not an opcode this contract defines")
    if name in TAKES_PAYLOAD:
        if name == "set_model_interval" and len(rest) != 2:
            raise Invalid("set_model_interval takes two bytes, the seconds little-endian")
        if name == "set_name":
            decode_name_parts(bytes(rest))
        return name, bytes(rest)
    if name in TAKES_ARG:
        if len(rest) != 1:
            raise Invalid(f"{name} takes exactly one argument byte")
        return name, rest[0]
    if rest:
        raise Invalid(f"{name} takes no argument")
    return name, None


@dataclass(frozen=True)
class Response:
    """One answer to a control request: an opcode, a status, and whatever
    payload goes with it."""
    opcode: int
    status: int
    payload: bytes

    @property
    def name(self) -> str | None:
        """The name of the opcode this responds to, or None if the
        contract does not define it."""
        return NAME_BY_OPCODE.get(self.opcode)

    @property
    def ok(self) -> bool:
        """Whether the request succeeded (status 0)."""
        return self.status == 0

    @property
    def status_name(self) -> str:
        """The status in words, or "undefined status N" if the contract
        does not define it."""
        return STATUS_CODES.get(self.status, f"undefined status {self.status}")


def decode_response(data: bytes) -> Response:
    """One Control Response message, decoded into a `Response`. Raises
    Truncated if `data` is under two bytes."""
    if len(data) < 2:
        raise Truncated("a response is at least an opcode and a status")
    return Response(data[0], data[1], bytes(data[2:]))


# --- device info ------------------------------------------------------------

#: What a device can do, as the bits it reports.
CAPABILITIES = {
    "battery_voltage": 1 << 0,
    "battery_low_flag": 1 << 1,
    "leadoff": 1 << 2,
    "test_signal": 1 << 3,
    "dcdc_mode": 1 << 4,
    "input_short": 1 << 5,
    "update": 1 << 6,
    "model": 1 << 7,
    "model_ready": 1 << 8,
    "heads": 1 << 9,
    # New in 1.1.
    "pipeline": 1 << 10,
    "bias_drive": 1 << 11,
    # New in 1.2.
    "embeddings": 1 << 12,
    "indicator": 1 << 13,
    "converter_registers": 1 << 14,
    # New in 1.3.
    "model_cadence": 1 << 15,
}
#: 1.3: capability bits 16 and up, the second word of Device Info at
#: offset 76. Bit 16 of the contract is bit 0 here.
CAPABILITIES_HIGH = {
    "synthetic": 1 << 0,
    "device_name": 1 << 1,
}

# --- the status lamp (1.2) ---------------------------------------------------
#
# One language at three levels of verbosity. A blink means the same thing at
# every level that shows it; the level only decides what is shown. The
# blink shapes are the device's and documented on its page; a host explains
# the lamp in words from this table and never decodes a light.

INDICATOR_LEVELS = {"silent": 0, "reserved": 1, "verbose": 2}
INDICATOR_LEVEL_NAMES = {v: k for k, v in INDICATOR_LEVELS.items()}
#: What each level shows, in the order the lamp prefers when several are
#: true (most important first).
INDICATOR_SHOWS = {
    "silent": [],
    "reserved": ["converter fault", "low battery"],
    "verbose": ["converter fault", "low battery", "update in progress", "streaming", "connected", "waiting for a host"],
}
#: Low battery begins below this percent of the device's own estimate and
#: ends above the clear percent, or on external power.
LOW_BATTERY_PERCENT = 20
LOW_BATTERY_CLEAR_PERCENT = 25
IDENTIFY_MAX_SECONDS = 30

# --- the encoder's output over the air (1.2) ---------------------------------

EMBEDDING_FORMS = {"off": 0, "window": 1, "tokens": 2, "both": 3}

_INFO_HEAD = struct.Struct("<BBBBBBBIIHB")
INFO_V01_LEN = 27
INFO_LEN_1_0 = 76
#: 1.3: the 1.0 layout with a second capability word appended.
INFO_LEN = 80
BATTERY_UNKNOWN = 0xFF


@dataclass(frozen=True)
class DeviceInfo:
    """Everything a device says about itself, and nothing a host assumed."""
    protocol: tuple[int, int]
    firmware: tuple[int, int, int]
    channels: int
    adc_bits: int
    tick_hz: int
    vref_uv: int
    capabilities: int
    supported_rates: int
    device_id: str
    #: Present on a device that reports the 1.0 layout.
    hardware: tuple[int, int, int] | None = None
    build_id: str | None = None
    model_embed_dim: int = 0
    model_native_sps: int = 0
    model_window_samples: int = 0
    head_slots: int = 0
    head_max_outputs: int = 0
    head_slot_bytes: int = 0
    update_chunk_max: int = 0
    app_slot_bytes: int = 0
    weights_image_bytes: int = 0
    #: 1.3: capability bits 16 and up. Zero on a device whose info ends at
    #: 76 bytes.
    capabilities_high: int = 0

    def can(self, capability: str) -> bool:
        """Whether the device claims a capability, from either word. Unknown
        names are False rather than an error, so a host written against a
        later version of this contract still runs."""
        if capability in CAPABILITIES_HIGH:
            return bool(self.capabilities_high & CAPABILITIES_HIGH[capability])
        return bool(self.capabilities & CAPABILITIES.get(capability, 0))

    @property
    def rates(self) -> list[int]:
        """The rates the device declares, from its own mask."""
        return [sps for bit, sps in sorted(RATE_BY_INFO_BIT.items())
                if self.supported_rates & (1 << bit)]

    @property
    def protocol_string(self) -> str:
        """The protocol version as "major.minor"."""
        return f"{self.protocol[0]}.{self.protocol[1]}"

    @property
    def firmware_string(self) -> str:
        """The firmware version as "major.minor.patch"."""
        return "{}.{}.{}".format(*self.firmware)

    # The names a manifest and a report already use. Kept so that a capture
    # recorded before this module existed reads the same way afterwards.
    @property
    def proto(self) -> str:
        """Alias for `protocol_string`, for a manifest that already uses
        this name."""
        return self.protocol_string

    @property
    def proto_version(self) -> tuple[int, int]:
        """Alias for `protocol`, for a manifest that already uses this
        name."""
        return self.protocol

    @property
    def fw(self) -> str:
        """Alias for `firmware_string`, for a manifest that already uses
        this name."""
        return self.firmware_string

    def microvolts_per_count(self, gain: int) -> float:
        """One count, in microvolts, at a gain. From the device's own
        reference and width."""
        return (2.0 * self.vref_uv) / (gain * (1 << self.adc_bits))


def decode_device_info(data: bytes) -> DeviceInfo:
    """The Device Info characteristic, decoded into a `DeviceInfo`, reading
    whichever layout length the device itself declares."""
    if len(data) < INFO_V01_LEN:
        raise Truncated(f"device info is at least {INFO_V01_LEN} bytes")
    (pj, pn, fj, fn, fp, channels, bits, tick, vref, caps, rates) = _INFO_HEAD.unpack_from(data, 0)
    info = dict(protocol=(pj, pn), firmware=(fj, fn, fp), channels=channels,
                adc_bits=bits, tick_hz=tick, vref_uv=vref, capabilities=caps,
                supported_rates=rates, device_id=data[19:27].hex())
    if len(data) == INFO_V01_LEN:
        return DeviceInfo(**info)
    info_len = data[27]
    if info_len < INFO_LEN_1_0:
        raise Invalid(f"a device speaking {pj}.{pn} states its info is {info_len} bytes, "
                      f"which is shorter than the {INFO_LEN_1_0} the 1.0 layout defines")
    if len(data) < INFO_LEN_1_0:
        raise Truncated(f"device info promises {info_len} bytes and carries {len(data)}")
    # 1.3: the second capability word, when the device says its info reaches it.
    caps_high = struct.unpack_from("<I", data, 76)[0] if info_len >= INFO_LEN and len(data) >= INFO_LEN else 0
    (hw_major, hw_minor, hw_patch) = data[28:31]
    (embed, native, window, slots, max_out, slot_bytes, chunk_max) = struct.unpack_from("<HHHBBHH", data, 40)
    (app_bytes, weights_bytes) = struct.unpack_from("<II", data, 52)
    return DeviceInfo(**info, hardware=(hw_major, hw_minor, hw_patch),
                      build_id=data[32:40].hex(), model_embed_dim=embed,
                      model_native_sps=native, model_window_samples=window,
                      head_slots=slots, head_max_outputs=max_out,
                      head_slot_bytes=slot_bytes, update_chunk_max=chunk_max,
                      app_slot_bytes=app_bytes, weights_image_bytes=weights_bytes,
                      capabilities_high=caps_high)


# --- status -----------------------------------------------------------------

_STATUS = struct.Struct("<BBBBBBBBHH")
STATUS_LEN = _STATUS.size
STATUS_FLAGS = {"usb_present": 1 << 0, "buffer_high_watermark": 1 << 1}


@dataclass(frozen=True)
class Status:
    """The Status characteristic, decoded: streaming state, mode, gain,
    rate, contact, and buffer."""
    streaming: bool
    mode: str
    gain: int
    sample_rate_hz: int | None
    charger: str
    #: None when the device has no battery telemetry, never 255 percent.
    battery_percent: int | None
    leadoff: int
    usb_present: bool
    buffer_high_watermark: bool
    #: Telemetry. The account of what a host missed is the sample index.
    dropped_total: int
    buffer_fill: int


def decode_status(data: bytes) -> Status:
    """The Status characteristic, decoded into a `Status`. Raises
    Truncated if `data` is shorter than `STATUS_LEN`."""
    if len(data) < STATUS_LEN:
        raise Truncated(f"status is {STATUS_LEN} bytes")
    (state, mode, gain_code, rate_code, charger, batt, loff, flags,
     dropped, fill) = _STATUS.unpack_from(data, 0)
    return Status(
        streaming=state == 1,
        mode=MODE_BY_CODE.get(mode, f"mode {mode}"),
        gain=GAIN_BY_CODE[gain_code] if gain_code < len(GAIN_BY_CODE) else None,
        sample_rate_hz=RATE_BY_CODE.get(rate_code),
        charger=CHARGER_STATES.get(charger, f"charger state {charger}"),
        battery_percent=None if batt == BATTERY_UNKNOWN else batt,
        leadoff=loff,
        usb_present=bool(flags & STATUS_FLAGS["usb_present"]),
        buffer_high_watermark=bool(flags & STATUS_FLAGS["buffer_high_watermark"]),
        dropped_total=dropped, buffer_fill=fill)


# --- the stream -------------------------------------------------------------

PACKET_EEG = 0x01
PACKET_PREDICTION = 0x02
#: 1.3: samples the device generated with its converter off, in exactly the
#: layout of `PACKET_EEG`. A 1.2 host refuses it and so never stores one as a
#: measurement.
PACKET_SYNTHETIC = 0x04
DATA_HEADER_LEN = 20
_DATA = struct.Struct("<BBHIQBBBB")
DATA_FLAGS = {"discontinuity": 1 << 0, "leadoff_active": 1 << 1,
              "usb_present": 1 << 4}
MODE_SHIFT, MODE_MASK = 2, 0b11 << 2
LOST_SATURATED = 0xFFFF


@dataclass(frozen=True)
class Packet:
    """One notification of samples, decoded but not scaled.

    `counts` is a list of rows, each row one sample across the channels, in
    converter counts. Scaling is the caller's, with the device's own
    reference and width, because this module assumes nothing about either.
    """
    discontinuity: bool
    leadoff_active: bool
    mode: str
    usb_present: bool
    samples_lost_before: int
    index: int
    device_time: int
    leadoff: int
    gain: int
    sample_rate_hz: int | None
    counts: list[list[int]]
    #: 1.3: the samples were generated by the device, not measured (packet
    #: type 0x04). Never store one as a measurement.
    synthetic: bool = False

    @property
    def n_samples(self) -> int:
        """How many samples this packet carries."""
        return len(self.counts)


def _int24(b: bytes) -> int:
    v = (b[0] << 16) | (b[1] << 8) | b[2]
    return v - 0x1000000 if v & 0x800000 else v


def decode_packet(data: bytes, channels: int) -> Packet:
    """One EEG notification. `channels` comes from the device."""
    if len(data) < DATA_HEADER_LEN:
        raise Truncated("a data packet is at least its header")
    (ptype, flags, lost, index, device_time, n, loff, gain_code, rate_code) = _DATA.unpack_from(data, 0)
    if ptype not in (PACKET_EEG, PACKET_SYNTHETIC):
        raise Invalid(f"packet type {ptype:#04x} is not sample data")
    if gain_code >= len(GAIN_BY_CODE):
        raise Invalid(f"gain code {gain_code} is not one this contract defines")
    body = data[DATA_HEADER_LEN:]
    want = n * channels * 3
    if len(body) != want:
        raise Truncated(f"{n} samples of {channels} channels is {want} bytes, and this packet carries {len(body)}")
    counts = [[_int24(body[(i * channels + c) * 3:(i * channels + c) * 3 + 3])
               for c in range(channels)] for i in range(n)]
    return Packet(
        discontinuity=bool(flags & DATA_FLAGS["discontinuity"]),
        leadoff_active=bool(flags & DATA_FLAGS["leadoff_active"]),
        mode=MODE_BY_CODE.get((flags & MODE_MASK) >> MODE_SHIFT, "unknown"),
        usb_present=bool(flags & DATA_FLAGS["usb_present"]),
        samples_lost_before=lost, index=index, device_time=device_time,
        leadoff=loff, gain=GAIN_BY_CODE[gain_code],
        sample_rate_hz=RATE_BY_CODE.get(rate_code), counts=counts,
        synthetic=ptype == PACKET_SYNTHETIC)


@dataclass(frozen=True)
class Continuity:
    """What one packet's indexing says happened since the one before it."""
    #: "continuous", "loss", or "rebase".
    kind: str
    #: Samples missing. None for a re-base, whose extent is not a number.
    lost: int | None

    @property
    def broken(self) -> bool:
        """Whether the timeline broke: a loss or a re-base, rather than
        continuing cleanly."""
        return self.kind != "continuous"


def continuity(previous_index: int, previous_n: int, index: int, discontinuity: bool) -> Continuity:
    """Judge two packets by the contract's own rule.

    The subtraction is signed, so the index's own wrap at two to the
    thirty second is exact. A step that is not forward is a break in the
    timeline rather than a loss, and reporting the unsigned difference
    there claims a loss of four billion samples that never happened.
    """
    expected = (previous_index + previous_n) & 0xFFFFFFFF
    step = (index - expected) & 0xFFFFFFFF
    if step >= 0x80000000:
        step -= 0x100000000
    if step == 0 and not discontinuity:
        return Continuity("continuous", 0)
    if step > 0:
        return Continuity("loss", step)
    return Continuity("rebase", None)


# --- predictions ------------------------------------------------------------

PREDICTION_HEADER_LEN = 28
_PREDICTION = struct.Struct("<BBBBIQH")
PREDICTION_FLAGS = {"gap_in_window": 1 << 0, "duty_reduced": 1 << 1,
                    "leadoff_in_window": 1 << 2}


@dataclass(frozen=True)
class Prediction:
    """One head's output for one window, decoded from a Predictions
    notification."""
    head_slot: int
    head_id: str
    index: int
    device_time: int
    window_samples: int
    gap_in_window: bool
    duty_reduced: bool
    leadoff_in_window: bool
    outputs: list[float]
    #: 1.1: what the window was taken from: the stream, the natural signal,
    #: or the model's own chain. A 1.0 device says the stream.
    input_source: str = "stream"


def decode_prediction(data: bytes) -> Prediction:
    """One Predictions notification, decoded into a `Prediction`."""
    if len(data) < PREDICTION_HEADER_LEN:
        raise Truncated("a prediction is at least its header")
    (ptype, flags, slot, n, index, device_time, window) = _PREDICTION.unpack_from(data, 0)
    if ptype != PACKET_PREDICTION:
        raise Invalid(f"packet type {ptype:#04x} is not a prediction")
    body = data[PREDICTION_HEADER_LEN:]
    if len(body) != n * 4:
        raise Truncated(f"{n} outputs is {n * 4} bytes, and this packet carries {len(body)}")
    return Prediction(
        head_slot=slot, head_id=data[18:26].hex(), index=index,
        device_time=device_time, window_samples=window,
        gap_in_window=bool(flags & PREDICTION_FLAGS["gap_in_window"]),
        duty_reduced=bool(flags & PREDICTION_FLAGS["duty_reduced"]),
        leadoff_in_window=bool(flags & PREDICTION_FLAGS["leadoff_in_window"]),
        outputs=list(struct.unpack_from(f"<{n}f", body, 0)),
        input_source=INPUT_SOURCE_NAMES.get(data[26], f"source {data[26]}"))


# --- what the device reports about itself beyond status ---------------------

@dataclass(frozen=True)
class Battery:
    """What the device reports about its battery: millivolts, percent,
    and charger state."""
    millivolts: int
    percent: int | None
    charger: str


def decode_battery(payload: bytes) -> Battery:
    """A battery reading, decoded into a `Battery`. Raises Truncated if
    `payload` is under four bytes."""
    if len(payload) < 4:
        raise Truncated("a battery reading is four bytes")
    mv, pct, charger = struct.unpack_from("<HBB", payload, 0)
    return Battery(mv, None if pct == BATTERY_UNKNOWN else pct,
                   CHARGER_STATES.get(charger, f"charger state {charger}"))


@dataclass(frozen=True)
class BootInfo:
    """Which firmware slot the device started from, why, whether that
    image has confirmed itself, and how many times it has booted."""
    slot: int
    reason: str
    confirmed: bool
    boot_count: int


def decode_boot_info(payload: bytes) -> BootInfo:
    """A boot information reading, decoded into a `BootInfo`. Raises
    Truncated if `payload` is under eight bytes."""
    if len(payload) < 8:
        raise Truncated("boot information is eight bytes")
    slot, reason, state, _, count = struct.unpack_from("<BBBBI", payload, 0)
    return BootInfo(slot, BOOT_REASONS.get(reason, f"reason {reason}"),
                    state == 1, count)


HEAD_STATES = {0: "empty", 1: "valid", 2: "invalid", 3: "width mismatch"}
MODEL_STATES = {0: "no runtime", 1: "no weights", 2: "ready", 3: "updating"}
NO_HEAD = 0xFF


@dataclass(frozen=True)
class Head:
    """One head slot: its state, output width, id, name, and the encoder
    it names."""
    slot: int
    state: str
    out_dim: int
    head_id: str
    name: str
    #: 1.3: the encoder the head was trained beside, as the head file
    #: recorded it; sixteen zeros when the trainer did not say, or from a
    #: 1.2 device. The device stores and reports it and never judges it.
    encoder_id: str = "0000000000000000"

    @property
    def usable(self) -> bool:
        """Whether this slot holds a usable head (its state is
        "valid")."""
        return self.state == "valid"

    def trained_beside(self, encoder_id: str) -> bool | None:
        """Whether this head names `encoder_id` as the encoder it was trained
        beside: None when the head does not say."""
        if self.encoder_id == "0000000000000000":
            return None
        return self.encoder_id == encoder_id


def decode_heads(payload: bytes) -> tuple[int | None, list[Head]]:
    """The head slots, and which one is selected. None means none is.

    The heads say nothing here of the encoder they name: from 1.4 that is
    `decode_head_encoders`, which `with_encoders` folds in."""
    if len(payload) < 2:
        raise Truncated("a head list is at least its count")
    active, n = payload[0], payload[1]
    records_end = 2 + n * 30
    # 1.3 defined one eight byte encoder id per record after the records.
    # No device sent it, because the IntoMind One's list was too long with
    # it, and 1.4 withdraws it. One that comes is still read.
    if len(payload) not in (records_end, records_end + n * 8):
        raise Invalid(f"a list of {n} heads is {records_end} bytes, or {records_end + n * 8} with its "
                      f"encoder ids, and this one is {len(payload)}")
    heads = []
    for i in range(n):
        r = payload[2 + i * 30: 32 + i * 30]
        slot, state, out_dim = r[0], r[1], struct.unpack_from("<H", r, 2)[0]
        encoder = payload[records_end + i * 8: records_end + i * 8 + 8].hex() if len(payload) > records_end else "0000000000000000"
        heads.append(Head(slot, HEAD_STATES.get(state, f"state {state}"), out_dim,
                          r[4:12].hex(), r[12:28].rstrip(b"\0").decode("utf-8", "replace"), encoder))
    return (None if active == NO_HEAD else active), heads


def decode_head_encoders(payload: bytes) -> dict[int, str]:
    """1.4: the encoder each head names, by slot, from LIST_HEAD_ENCODERS:
    sixteen zeros for an empty slot or a head that does not say."""
    if len(payload) < 1:
        raise Truncated("a list of head encoders is at least its count")
    n = payload[0]
    if len(payload) != 1 + n * 9:
        raise Invalid(f"a list of {n} head encoders is {1 + n * 9} bytes, and this one is {len(payload)}")
    return {payload[1 + i * 9]: payload[2 + i * 9: 10 + i * 9].hex() for i in range(n)}


def with_encoders(heads: list[Head], encoders: dict[int, str]) -> list[Head]:
    """The heads, each with the encoder `encoders` says it names. A slot
    missing from it keeps what it had."""
    return [dataclasses.replace(h, encoder_id=encoders.get(h.slot, h.encoder_id)) for h in heads]


@dataclass(frozen=True)
class ModelInfo:
    """What the device's model can do and is doing: its state, active
    head, encoder, weights version, and timing."""
    state: str
    active_head: int | None
    predictions_on: bool
    encoder_id: str
    weights_version: tuple[int, int, int]
    #: 1.1: the signal classes the loaded model declares it takes, as
    #: `INPUT_CLASSES` bits. A 1.0 device sends zero, which reads as the
    #: time domain.
    input_classes: int = 1
    #: 1.2: tokens the loaded model produces per channel per window, the
    #: shape of the token form of embeddings. A 1.1 device's message ends
    #: before this byte and reads as zero: it sends no tokens.
    tokens_per_channel: int = 0
    #: 1.3: the time one pass of the encoder takes on the device, in
    #: milliseconds, as measured; zero before the first pass.
    pass_ms: int = 0
    #: 1.3: the interval between described windows in force, seconds; zero
    #: is every window the device can.
    interval_s: int = 0
    #: 1.3: whether the device holds a generator, so it can offer the
    #: synthetic signal.
    generator: bool = False

    @property
    def ready(self) -> bool:
        """Whether the model is loaded and ready to run (its state is
        "ready")."""
        return self.state == "ready"

    def takes(self, input_class: str) -> bool:
        """Whether the loaded model accepts `input_class` as input."""
        return bool(self.input_classes & INPUT_CLASSES.get(input_class, 0))


def decode_model_info(payload: bytes) -> ModelInfo:
    """GET_MODEL_INFO's answer, decoded into a `ModelInfo`. Raises
    Truncated if `payload` is under sixteen bytes."""
    if len(payload) < 16:
        raise Truncated("model information is at least sixteen bytes")
    state, active, on = payload[0], payload[1], payload[2]
    pass_ms, interval_s, generator = 0, 0, False
    if len(payload) >= 22:
        pass_ms, interval_s = struct.unpack_from("<HH", payload, 17)
        generator = payload[21] != 0
    return ModelInfo(MODEL_STATES.get(state, f"state {state}"),
                     None if active == NO_HEAD else active, bool(on),
                     payload[4:12].hex(), tuple(payload[12:15]),
                     payload[15] or INPUT_CLASSES["time_domain"],
                     payload[16] if len(payload) >= 17 else 0,
                     pass_ms, interval_s, generator)


# --- the model's cadence and the device's name (1.3) --------------------------

@dataclass(frozen=True)
class ModelInterval:
    """GET_MODEL_INTERVAL's answer: the interval in force between described
    windows, and the least interval the device keeps, both in seconds. Zero
    in force is every window the device can, each the newest complete one."""
    interval_s: int
    minimum_s: int


def decode_model_interval(payload: bytes) -> ModelInterval:
    """GET_MODEL_INTERVAL's answer, decoded into a `ModelInterval`. Raises
    Truncated if `payload` is under four bytes."""
    if len(payload) < 4:
        raise Truncated("a model interval is four bytes")
    return ModelInterval(*struct.unpack_from("<HH", payload, 0))


def encode_model_interval(seconds: int) -> bytes:
    """SET_MODEL_INTERVAL's payload."""
    if not 0 <= int(seconds) <= 0xFFFF:
        raise Invalid("an interval is 0 to 65535 seconds")
    return struct.pack("<H", int(seconds))


#: The most a composed name may be, in bytes: what a scan response carries,
#: so every scan list shows the whole of it. The same limit bounds each part.
NAME_MAX_COMPOSED = 29


def name_part_ok(part: str) -> bool:
    """A name or an adjective: UTF-8 of at most `NAME_MAX_COMPOSED` bytes,
    no control characters, no leading or trailing space. Empty is allowed."""
    b = part.encode("utf-8")
    if len(b) > NAME_MAX_COMPOSED:
        return False
    if any(c < 0x20 or c == 0x7F for c in b):
        return False
    return part == part.strip(" ")


def compose_name(name: str, adjective: str, product: str = PRODUCT_NAME) -> str:
    """The device's name as it composes it: `name's adjective product`, the
    possessive only with a name, the adjective only when given."""
    parts = []
    if name:
        parts.append(name + "'s")
    if adjective:
        parts.append(adjective)
    parts.append(product)
    return " ".join(parts)


def name_fits(name: str, adjective: str, product: str = PRODUCT_NAME) -> bool:
    """Whether the composition fits the air, so the device will accept it."""
    return (name_part_ok(name) and name_part_ok(adjective)
            and len(compose_name(name, adjective, product).encode("utf-8")) <= NAME_MAX_COMPOSED)


def encode_name_parts(name: str, adjective: str) -> bytes:
    """SET_NAME's payload, and GET_NAME's answer: each part as a length and
    its UTF-8 bytes."""
    n, a = name.encode("utf-8"), adjective.encode("utf-8")
    if len(n) > NAME_MAX_COMPOSED or len(a) > NAME_MAX_COMPOSED:
        raise Invalid(f"a name and an adjective are each at most {NAME_MAX_COMPOSED} bytes")
    return bytes([len(n)]) + n + bytes([len(a)]) + a


def decode_name_parts(payload: bytes) -> tuple[str, str]:
    """GET_NAME's answer, or SET_NAME's payload, decoded back into
    (name, adjective)."""
    if not payload:
        raise Truncated("a name carries at least its lengths")
    nl = payload[0]
    if len(payload) < nl + 2:
        raise Invalid("a name that ends inside its own parts")
    name = payload[1:1 + nl]
    al = payload[1 + nl]
    adjective = payload[2 + nl:]
    if len(adjective) != al:
        raise Invalid("a name whose adjective is not the length it declares")
    return name.decode("utf-8", "replace"), adjective.decode("utf-8", "replace")


# --- the converter's registers, read only (1.2) ------------------------------
#
# The raw configuration bytes of the analog converter, for debugging. The
# contract carries the bytes and the family of chip they belong to; what
# they mean is the family's datasheet, keyed here on the family the device
# declares and never on a device name. There is no write.

#: The contract names a converter family by a number. Family one is the
#: Texas Instruments ADS1299 family (ADS1299, ADS1299-4, ADS1299-6); the
#: IntoMind One carries the ADS1299-4, as its datasheet says. This library
#: puts that family's registers into words.
CONVERTER_FAMILIES = {1: "ads1299"}


@dataclass(frozen=True)
class ConverterRegisters:
    """The converter's raw registers: which family, where they start,
    and their values."""
    family: str
    family_code: int
    first: int
    values: bytes

    def describe(self) -> dict:
        """The registers in words, for the families this version knows."""
        if self.family_code == 1 and self.first == 0:
            return describe_ads1299_registers(self.values)
        return {"family": self.family, "values": self.values.hex()}


def decode_converter_registers(payload: bytes) -> ConverterRegisters:
    """The GET_CONVERTER_REGISTERS answer, decoded into a
    `ConverterRegisters`. Raises Truncated if `payload` is under three
    bytes."""
    if len(payload) < 3:
        raise Truncated("registers are at least a family, a first address, and a count")
    family, first, count = payload[0], payload[1], payload[2]
    if count == 0 or len(payload) != 3 + count:
        raise Invalid(f"{count} registers declared and {len(payload) - 3} carried")
    return ConverterRegisters(CONVERTER_FAMILIES.get(family, f"family {family}"), family, first,
                              bytes(payload[3:]))


_FAMILY_1_RATES = {0: 16000, 1: 8000, 2: 4000, 3: 2000, 4: 1000, 5: 500, 6: 250}
_FAMILY_1_GAINS = {0: 1, 1: 2, 2: 4, 3: 6, 4: 8, 5: 12, 6: 24}
_FAMILY_1_INPUTS = {0: "electrode", 1: "shorted", 2: "bias measurement", 3: "supply",
                    4: "temperature", 5: "test signal", 6: "bias drive, positive",
                    7: "bias drive, negative"}
_FAMILY_1_LOFF_THRESHOLD = [95, 92.5, 90, 87.5, 85, 80, 75, 70]
_FAMILY_1_LOFF_CURRENT = ["6 nA", "24 nA", "6 uA", "24 uA"]
_FAMILY_1_LOFF_FREQUENCY = ["dc", "7.8 Hz", "31.2 Hz", "data rate / 4"]


def _channels_of(mask: int) -> list[int]:
    return [c + 1 for c in range(8) if mask & (1 << c)]


def describe_ads1299_registers(values: bytes) -> dict:
    """An ADS1299 family register file in words, from address 0: the
    identity, the data rate, the test signal, the reference and bias
    settings, lead-off detection, each channel's gain and input, and the
    bias and lead-off sense masks. Every field is the datasheet's (Texas
    Instruments SBAS499C, Register Maps). A file shorter than 24 bytes is
    described as far as it goes.
    """
    def at(i: int) -> int | None:
        return values[i] if i < len(values) else None

    out: dict = {"family": CONVERTER_FAMILIES[1]}
    ident = at(0)
    if ident is not None:
        out["identity"] = {
            "device": "ADS1299" if (ident & 0x1C) == 0x1C else f"not an ADS1299 ({ident:#04x})",
            "family_member": (ident & 0x1C) == 0x1C,
            "revision": (ident >> 5) & 0x7,
            "channels": {0: 4, 1: 6, 2: 8}.get(ident & 0x3, f"code {ident & 0x3}"),
        }
    c1 = at(1)
    if c1 is not None:
        out["data_rate_sps"] = _FAMILY_1_RATES.get(c1 & 0x7)
        out["multiple_readback"] = bool(c1 & 0x40)
        out["clock_output"] = bool(c1 & 0x20)
    c2 = at(2)
    if c2 is not None:
        out["test_signal"] = {
            "internal": bool(c2 & 0x10),
            "amplitude": 2 if c2 & 0x04 else 1,
            "frequency": {0: "fclk / 2^21", 1: "fclk / 2^20", 3: "dc"}.get(c2 & 0x3, "do not use"),
        }
    c3 = at(3)
    if c3 is not None:
        out["reference_buffer_on"] = bool(c3 & 0x80)
        out["bias"] = {
            "amplifier_on": bool(c3 & 0x04),
            "reference_internal": bool(c3 & 0x08),
            "measurement": bool(c3 & 0x10),
            "sense": bool(c3 & 0x02),
            "connected": not (c3 & 0x01),
        }
    lo = at(4)
    if lo is not None:
        out["lead_off"] = {
            "threshold_percent": _FAMILY_1_LOFF_THRESHOLD[(lo >> 5) & 0x7],
            "current": _FAMILY_1_LOFF_CURRENT[(lo >> 2) & 0x3],
            "frequency": _FAMILY_1_LOFF_FREQUENCY[lo & 0x3],
        }
    channels = []
    for i in range(8):
        ch = at(5 + i)
        if ch is None:
            break
        channels.append({
            "channel": i + 1,
            "powered_down": bool(ch & 0x80),
            "gain": _FAMILY_1_GAINS.get((ch >> 4) & 0x7, "do not use"),
            "reference_switch": bool(ch & 0x08),
            "input": _FAMILY_1_INPUTS[ch & 0x7],
        })
    if channels:
        out["channels"] = channels
    for i, name in ((13, "bias_sense_positive"), (14, "bias_sense_negative"),
                    (15, "lead_off_sense_positive"), (16, "lead_off_sense_negative"),
                    (17, "lead_off_flip"), (18, "lead_off_status_positive"),
                    (19, "lead_off_status_negative")):
        v = at(i)
        if v is not None:
            out[name] = _channels_of(v)
    m1 = at(21)
    if m1 is not None:
        out["common_reference_switch"] = bool(m1 & 0x20)
    c4 = at(23)
    if c4 is not None:
        out["single_shot"] = bool(c4 & 0x08)
        out["lead_off_comparators_on"] = bool(c4 & 0x02)
    return out


# --- embeddings (1.2) --------------------------------------------------------
#
# The encoder's output for one window, in two forms. The window embedding is
# the mean over live channels and over time of the tokens, and is what a
# head consumes. The tokens are the encoder's output before that pooling,
# one vector per channel per slice of the window, and are what
# reconstruction turns back into signal. Both are quantized the way a head
# receives an embedding (times 4096), so what a host trains on is what the
# device runs. The weights never leave the device; these are its outputs.

PACKET_EMBEDDING = 0x03
EMBEDDING_HEADER_LEN = 28
WINDOW_TOKEN = 0xFF
EMBEDDING_FLAGS = {"gap_in_window": 1 << 0, "duty_reduced": 1 << 1,
                   "leadoff_in_window": 1 << 2, "more_parts": 1 << 3}
_EMBEDDING = struct.Struct("<BBBBIQH8sBB")


@dataclass(frozen=True)
class Embedding:
    """One notification: a whole vector, or one part of one."""
    index: int
    device_time: int
    window_samples: int
    encoder_id: str
    input_source: str
    embed_dim: int
    #: Index of the first value carried here.
    first: int
    #: None for the window embedding; otherwise the token's index,
    #: channel-major over all channels: channel × tokens_per_channel + slice.
    token: int | None
    more_parts: bool
    gap_in_window: bool
    duty_reduced: bool
    leadoff_in_window: bool
    #: Quantized values, the encoder's output times 4096.
    values: list[int]

    @property
    def is_window_embedding(self) -> bool:
        """Whether this is the window embedding rather than one
        token's."""
        return self.token is None


def decode_embedding(data: bytes) -> Embedding:
    """One Embeddings notification, decoded into an `Embedding`."""
    if len(data) < EMBEDDING_HEADER_LEN:
        raise Truncated("an embedding is at least its header")
    (ptype, flags, dim, first, index, device_time, window, encoder, source, token) = \
        _EMBEDDING.unpack_from(data, 0)
    if ptype != PACKET_EMBEDDING:
        raise Invalid(f"packet type {ptype:#04x} is not an embedding")
    body = data[EMBEDDING_HEADER_LEN:]
    if not body or len(body) % 2:
        raise Truncated("an embedding carries whole sixteen-bit values, at least one")
    n = len(body) // 2
    if first + n > dim:
        raise Invalid(f"values {first} to {first + n} of a vector of {dim}")
    return Embedding(
        index=index, device_time=device_time, window_samples=window,
        encoder_id=encoder.hex(), input_source=INPUT_SOURCE_NAMES.get(source, f"source {source}"),
        embed_dim=dim, first=first, token=None if token == WINDOW_TOKEN else token,
        more_parts=bool(flags & EMBEDDING_FLAGS["more_parts"]),
        gap_in_window=bool(flags & EMBEDDING_FLAGS["gap_in_window"]),
        duty_reduced=bool(flags & EMBEDDING_FLAGS["duty_reduced"]),
        leadoff_in_window=bool(flags & EMBEDDING_FLAGS["leadoff_in_window"]),
        values=list(struct.unpack_from(f"<{n}h", body, 0)))


@dataclass
class EmbeddingWindow:
    """One window's embeddings, assembled from its notifications."""
    index: int
    device_time: int
    window_samples: int
    encoder_id: str
    input_source: str
    embed_dim: int
    gap_in_window: bool
    duty_reduced: bool
    leadoff_in_window: bool
    #: The window embedding, `embed_dim` quantized values, when asked for.
    embedding: list[int] | None = None
    #: The tokens, `channels × tokens_per_channel` lists of `embed_dim`
    #: quantized values, channel-major, when asked for.
    tokens: list[list[int]] | None = None

    def embedding_floats(self) -> list[float]:
        """The window embedding, dequantized back to floats."""
        return [v / EMBED_SCALE for v in (self.embedding or [])]

    def token_array(self, channels: int, tokens_per_channel: int):
        """The tokens as `(channels, tokens_per_channel, embed_dim)` floats,
        what the reconstruction head takes."""
        import numpy as np
        t = np.asarray(self.tokens, dtype=np.float32) / EMBED_SCALE
        return t.reshape(channels, tokens_per_channel, self.embed_dim)


class EmbeddingAssembler:
    """Puts notifications back together into whole windows.

    Built for a device's channel count, the model's tokens per channel, and
    the form asked for; `feed` takes each decoded notification and returns
    the window when its last piece arrives. Windows older than the two most
    recent that never completed are dropped and counted in `incomplete`, so
    a lost notification costs one window and not the memory of the host.
    """

    def __init__(self, channels: int, tokens_per_channel: int, form: str = "both"):
        if form not in EMBEDDING_FORMS or form == "off":
            raise ValueError(f"form is window, tokens, or both, not {form!r}")
        self.channels, self.tokens_per_channel, self.form = channels, tokens_per_channel, form
        self.n_tokens = channels * tokens_per_channel
        self.incomplete = 0
        self._windows: dict[int, dict] = {}

    def _want_embedding(self) -> bool:
        return self.form in ("window", "both")

    def _want_tokens(self) -> bool:
        return self.form in ("tokens", "both")

    def feed(self, e: Embedding) -> EmbeddingWindow | None:
        """Feed one decoded notification in. Returns the assembled
        `EmbeddingWindow` once its last piece has arrived, otherwise
        None."""
        w = self._windows.get(e.index)
        if w is None:
            w = self._windows[e.index] = {
                "meta": e, "embedding": [None] * e.embed_dim,
                "tokens": [[None] * e.embed_dim for _ in range(self.n_tokens)],
            }
            # Two windows can be in flight; anything older never finished.
            for old in sorted(self._windows):
                if len(self._windows) <= 3:
                    break
                del self._windows[old]
                self.incomplete += 1
        target = w["embedding"] if e.token is None else (w["tokens"][e.token] if e.token < self.n_tokens else None)
        if target is None:
            return None
        target[e.first:e.first + len(e.values)] = e.values
        emb_done = not self._want_embedding() or None not in w["embedding"]
        tok_done = not self._want_tokens() or all(None not in t for t in w["tokens"])
        if not (emb_done and tok_done):
            return None
        del self._windows[e.index]
        m = w["meta"]
        return EmbeddingWindow(
            index=m.index, device_time=m.device_time, window_samples=m.window_samples,
            encoder_id=m.encoder_id, input_source=m.input_source, embed_dim=m.embed_dim,
            gap_in_window=m.gap_in_window, duty_reduced=m.duty_reduced,
            leadoff_in_window=m.leadoff_in_window,
            embedding=list(w["embedding"]) if self._want_embedding() else None,
            tokens=[list(t) for t in w["tokens"]] if self._want_tokens() else None)


# --- processing (1.1) --------------------------------------------------------
#
# The device preprocesses its own signal. A chain is an ordered list of
# stages, each a kind from the device's catalog with up to four sixteen-bit
# parameters whose units the kind defines. The empty chain is the natural
# signal. When a chain is in force only its output streams, and the chain
# is what a recording stores to say what produced its samples: there is no
# raw-or-filtered flag anywhere, because a flag says nothing about what a
# signal is.

STAGE_KINDS = {"highpass": 1, "lowpass": 2, "notch": 3}
KIND_NAMES = {v: k for k, v in STAGE_KINDS.items()}
STAGE_CLASSES = {"map": 0, "representation": 1, "detector": 2}
CLASS_NAMES = {v: k for k, v in STAGE_CLASSES.items()}
#: 1.3 adds synthetic: reported by a device generating its signal, never
#: requested.
INPUT_SOURCES = {"stream": 0, "natural": 1, "own_chain": 2, "synthetic": 3}
INPUT_SOURCE_NAMES = {v: k for k, v in INPUT_SOURCES.items()}
INPUT_CLASSES = {"time_domain": 1 << 0, "representation": 1 << 1}
PIPELINE_ORIGINS = {"default": 0, "host": 1}
ORIGIN_NAMES = {v: k for k, v in PIPELINE_ORIGINS.items()}
BIAS_MODES = {"off": 0, "on": 1, "loop_open": 2}
MAX_STAGES = 12
MAX_PARAMS = 4
MAX_NOTCHES = 8
#: A corner or band edge at or above this share of Nyquist is refused.
NYQUIST_SHARE_MAX = 0.9
#: The automatic low-pass corner, as a share of the sample rate.
LOW_PASS_AUTO_SHARE = 0.4
DEFAULT_HIGH_PASS_DHZ = 5
#: Both mains fundamentals, second and third harmonics, four hertz wide,
#: the device's default until a region is chosen.
DEFAULT_MAINS_BANDS_DHZ = ((480, 520), (980, 1020), (1480, 1520), (580, 620), (1180, 1220), (1780, 1820))


@dataclass(frozen=True)
class Stage:
    """One stage of a chain: a kind and its parameters, in tenths of a hertz
    for every kind this version defines."""
    kind: int
    params: tuple[int, ...] = ()

    @property
    def name(self) -> str:
        """This stage's kind in words, or "kind N" if the contract does
        not define it."""
        return KIND_NAMES.get(self.kind, f"kind {self.kind}")

    def as_dict(self) -> dict:
        """This stage as a plain dict, with its kind, name, and
        parameters."""
        return dict(kind=self.kind, name=self.name, params=list(self.params))


def _dhz(hz) -> int:
    v = int(round(float(hz) * 10))
    if not 0 <= v <= 0xFFFF:
        raise Invalid(f"{hz} Hz is outside what a stage parameter can hold")
    return v


def highpass(corner_hz) -> Stage:
    """A high-pass at a corner, at least 0.1 Hz."""
    return Stage(STAGE_KINDS["highpass"], (_dhz(corner_hz),))


def lowpass(corner_hz=None) -> Stage:
    """A low-pass at a corner, or the automatic corner at four tenths of
    the sample rate when none is given."""
    return Stage(STAGE_KINDS["lowpass"], (0 if corner_hz is None else _dhz(corner_hz),))


def notch(low_hz, high_hz) -> Stage:
    """A notch over the band between two edges. Any band."""
    return Stage(STAGE_KINDS["notch"], (_dhz(low_hz), _dhz(high_hz)))


def encode_chain(stages) -> bytes:
    """`u8 n_stages`, then per stage `u8 kind, u8 n_params, u16[n_params]`."""
    stages = list(stages)
    if len(stages) > MAX_STAGES:
        raise Invalid(f"a chain holds at most {MAX_STAGES} stages")
    out = bytearray([len(stages)])
    for s in stages:
        if not 1 <= int(s.kind) <= 255:
            raise Invalid("a stage kind is one byte and never zero")
        if len(s.params) > MAX_PARAMS:
            raise Invalid(f"a stage takes at most {MAX_PARAMS} parameters")
        out += bytes([int(s.kind), len(s.params)])
        for p in s.params:
            out += struct.pack("<H", int(p))
    return bytes(out)


def _decode_chain_prefix(data: bytes) -> tuple[list[Stage], bytes]:
    if not data:
        raise Truncated("a chain is at least its stage count")
    n = data[0]
    if n > MAX_STAGES:
        raise Invalid(f"{n} stages is more than a chain holds")
    at, stages = 1, []
    for _ in range(n):
        if len(data) < at + 2:
            raise Truncated("a chain ends inside a stage")
        kind, n_params = data[at], data[at + 1]
        if kind == 0:
            raise Invalid("stage kind zero is not a kind")
        if n_params > MAX_PARAMS:
            raise Invalid(f"a stage takes at most {MAX_PARAMS} parameters")
        at += 2
        if len(data) < at + 2 * n_params:
            raise Truncated("a stage promises parameters it does not carry")
        params = struct.unpack_from(f"<{n_params}H", data, at) if n_params else ()
        at += 2 * n_params
        stages.append(Stage(kind, tuple(params)))
    return stages, bytes(data[at:])


def decode_chain(data: bytes) -> list[Stage]:
    """A chain descriptor that is the whole of `data`."""
    stages, rest = _decode_chain_prefix(data)
    if rest:
        raise Invalid("bytes after the last stage")
    return stages


@dataclass(frozen=True)
class CatalogEntry:
    """One kind a device runs: its number, class, parameter count, how many
    instances a chain may hold, and its name."""
    kind: int
    cls: str
    n_params: int
    max_instances: int
    name: str


#: The kinds this version of the contract defines, with their limits. A
#: device's own catalog is the authority; this is what a host checks
#: against when it has not read one.
CONTRACT_CATALOG = [
    CatalogEntry(STAGE_KINDS["highpass"], "map", 1, 1, "highpass"),
    CatalogEntry(STAGE_KINDS["lowpass"], "map", 1, 1, "lowpass"),
    CatalogEntry(STAGE_KINDS["notch"], "map", 2, MAX_NOTCHES, "notch"),
]


def decode_catalog(payload: bytes) -> list[CatalogEntry]:
    """GET_PIPELINE_CATALOG: `u8 n_kinds`, then records of twelve bytes."""
    if not payload:
        raise Truncated("a catalog is at least its count")
    n = payload[0]
    if len(payload) != 1 + 12 * n:
        raise Invalid("a catalog is one byte and twelve per kind, exactly")
    entries = []
    for i in range(n):
        b = payload[1 + 12 * i:13 + 12 * i]
        name = b[4:12].split(b"\0", 1)[0].decode("ascii", "replace")
        entries.append(CatalogEntry(b[0], CLASS_NAMES.get(b[1], f"class {b[1]}"), b[2], b[3], name))
    return entries


@dataclass(frozen=True)
class PipelineState:
    """The chain in force, and whether it is the device's own default for
    its current rate or one a host set."""
    origin: str
    stages: list

    @property
    def is_default(self) -> bool:
        """Whether the chain in force is the device's own default,
        rather than one a host set."""
        return self.origin == "default"


def decode_pipeline_state(payload: bytes) -> PipelineState:
    """GET_PIPELINE's answer, decoded into a `PipelineState`. Raises
    Truncated if `payload` is empty."""
    if not payload:
        raise Truncated("a pipeline state is at least its origin")
    if payload[0] not in ORIGIN_NAMES:
        raise Invalid(f"origin {payload[0]} is not one this contract defines")
    return PipelineState(ORIGIN_NAMES[payload[0]], decode_chain(payload[1:]))


@dataclass(frozen=True)
class PredictionInput:
    """Where the model's input comes from: the stream as it is, the natural
    signal, or a chain of the model's own. In an answer the stages are the
    chain in effect for the model."""
    source: str
    stages: list


def encode_prediction_input(source: str, stages=()) -> bytes:
    """SET_PREDICTION_INPUT's payload."""
    if source not in INPUT_SOURCES:
        raise Invalid(f"{source!r} is not a source this contract defines")
    stages = list(stages)
    if source != "own_chain" and stages:
        raise Invalid("only the model's own chain carries stages")
    return bytes([INPUT_SOURCES[source]]) + encode_chain(stages)


def decode_prediction_input(payload: bytes) -> PredictionInput:
    """GET_PREDICTION_INPUT's answer, decoded into a `PredictionInput`.
    Raises Truncated if `payload` is empty."""
    if not payload:
        raise Truncated("a prediction input is at least its source")
    # Synthetic (3) is reported by a device generating its signal, never
    # requested, so a request naming it is not one.
    if payload[0] not in INPUT_SOURCE_NAMES or payload[0] == INPUT_SOURCES["synthetic"]:
        raise Invalid(f"source {payload[0]} is not one this contract defines")
    return PredictionInput(INPUT_SOURCE_NAMES[payload[0]], decode_chain(payload[1:]))


@dataclass(frozen=True)
class BiasDiagnostic:
    """The bias output over the device's own window, in millivolts."""
    mean_mv: int
    sd_mv: int
    min_mv: int
    max_mv: int


def decode_bias_diagnostic(payload: bytes) -> BiasDiagnostic:
    """GET_BIAS_DIAGNOSTIC's answer, decoded into a `BiasDiagnostic`.
    Raises Truncated if `payload` is under eight bytes."""
    if len(payload) < 8:
        raise Truncated("a bias diagnostic is eight bytes")
    return BiasDiagnostic(*struct.unpack_from("<4h", payload, 0))


# The rules, the same ones the device applies, so a host can say before
# sending whether a chain runs at a rate and explain a refusal in words.

def low_pass_auto_dhz(rate_sps) -> int:
    """The automatic low-pass corner for a rate, in tenths of a hertz."""
    return int(rate_sps) * 4


def corner_limit_dhz(rate_sps) -> int:
    """The first corner a rate refuses, in tenths of a hertz: nine tenths
    of Nyquist."""
    return int(rate_sps) * 45 // 10


def check_chain(stages, rate_sps, catalog=None) -> None:
    """Whether a chain can run at a rate. Raises `Invalid` saying why not."""
    by_kind = {e.kind: e for e in (catalog or CONTRACT_CATALOG)}
    limit = corner_limit_dhz(rate_sps)
    counts: dict[int, int] = {}
    hp = lp = None
    for s in stages:
        e = by_kind.get(s.kind)
        if e is None:
            raise Invalid(f"kind {s.kind} is not in the device's catalog")
        if len(s.params) != e.n_params:
            raise Invalid(f"{e.name} takes {e.n_params} parameter(s), not {len(s.params)}")
        counts[s.kind] = counts.get(s.kind, 0) + 1
        if counts[s.kind] > e.max_instances:
            raise Invalid(f"a chain holds at most {e.max_instances} {e.name} stage(s)")
        p = s.params
        if s.kind == STAGE_KINDS["highpass"]:
            if p[0] == 0:
                raise Invalid("a high-pass corner is at least 0.1 Hz")
            if p[0] >= limit:
                raise Invalid(f"a high-pass at {p[0] / 10:g} Hz is at or above nine tenths of Nyquist at {rate_sps} SPS")
            hp = p[0]
        elif s.kind == STAGE_KINDS["lowpass"]:
            c = p[0] or low_pass_auto_dhz(rate_sps)
            if c >= limit:
                raise Invalid(f"a low-pass at {c / 10:g} Hz is at or above nine tenths of Nyquist at {rate_sps} SPS")
            lp = c
        elif s.kind == STAGE_KINDS["notch"]:
            if p[0] == 0:
                raise Invalid("a band starts above zero")
            if p[1] <= p[0]:
                raise Invalid("a band's high edge is above its low edge")
            if p[1] >= limit:
                raise Invalid(f"a band up to {p[1] / 10:g} Hz is at or above nine tenths of Nyquist at {rate_sps} SPS")
    if hp is not None and lp is not None and hp >= lp:
        raise Invalid("the high-pass meets the low-pass: the band passes nothing")


def default_chain(rate_sps) -> list[Stage]:
    """The device's own default for a rate: a 0.5 Hz high-pass, the mains
    bands the rate can represent, the automatic low-pass."""
    limit = corner_limit_dhz(rate_sps)
    stages = [Stage(STAGE_KINDS["highpass"], (DEFAULT_HIGH_PASS_DHZ,))]
    stages += [Stage(STAGE_KINDS["notch"], (lo, hi)) for lo, hi in DEFAULT_MAINS_BANDS_DHZ if hi < limit]
    stages.append(Stage(STAGE_KINDS["lowpass"], (0,)))
    return stages


def mains_bands(mains_hz: int, rate_sps=None) -> list[Stage]:
    """The notch stages for a region's mains: the fundamental, its second
    and its third harmonic, four hertz wide. Given a rate, only the bands
    that rate can represent: at 250 samples a second the device refuses
    the higher harmonics."""
    f = int(mains_hz) * 10
    limit = corner_limit_dhz(rate_sps) if rate_sps else None
    return [Stage(STAGE_KINDS["notch"], (k * f - 20, k * f + 20)) for k in (1, 2, 3)
            if limit is None or k * f + 20 < limit]


def output_class(stages, catalog=None) -> str:
    """The class of signal a chain produces: the time domain unless a stage
    is a representation change."""
    by_kind = {e.kind: e for e in (catalog or CONTRACT_CATALOG)}
    for s in stages:
        e = by_kind.get(s.kind)
        if e is not None and e.cls == "representation":
            return "representation"
    return "time_domain"


def describe_chain(stages, rate_sps=None) -> str:
    """A chain in words, for a manifest, a header, or a person."""
    stages = list(stages)
    if not stages:
        return "natural signal"
    words = []
    for s in stages:
        p = s.params
        if s.kind == STAGE_KINDS["highpass"] and p:
            words.append(f"high-pass {p[0] / 10:g} Hz")
        elif s.kind == STAGE_KINDS["lowpass"] and p:
            if p[0]:
                words.append(f"low-pass {p[0] / 10:g} Hz")
            elif rate_sps:
                words.append(f"low-pass {low_pass_auto_dhz(rate_sps) / 10:g} Hz (automatic at {rate_sps} SPS)")
            else:
                words.append("low-pass automatic")
        elif s.kind == STAGE_KINDS["notch"] and len(p) >= 2:
            words.append(f"notch {p[0] / 10:g} to {p[1] / 10:g} Hz")
        else:
            words.append(f"{s.name} {list(p)}")
    return ", ".join(words)


# --- the update service -----------------------------------------------------

UPDATE_OPS = {"start": 0x01, "query": 0x02, "finish": 0x03, "activate": 0x04,
              "abort": 0x05}
UPDATE_TARGETS = {"app": 1, "weights": 2, "head": 3}
UPDATE_STATUS = {0: "ok", 1: "invalid argument", 2: "unsupported", 3: "busy",
                 4: "no transfer", 5: "flash", 6: "verification failed"}
UPDATE_STATES = {0: "idle", 1: "receiving", 2: "complete", 3: "failed"}
VERIFY_RESULTS = {
    0: "verified", 1: "length", 2: "malformed", 3: "target", 4: "slot",
    5: "hash", 6: "signature", 7: "security counter", 8: "head shape",
    9: "flash", 10: "key id",
}
ENVELOPE_LEN = 32
ENVELOPE_MAGIC = b"IMUP"
SLOT_LINK_NONE = 0xFF


def encode_update_start(target: str, slot: int, total_len: int, transfer_id: bytes) -> bytes:
    """The Update Control write that starts a transfer: target, slot,
    length, and transfer id."""
    if target not in UPDATE_TARGETS:
        raise Invalid(f"{target} is not something this contract transfers")
    if len(transfer_id) != 8:
        raise Invalid("a transfer identifier is eight bytes")
    return struct.pack("<BBBI", UPDATE_OPS["start"], UPDATE_TARGETS[target], slot, total_len) + bytes(transfer_id)


def encode_update_op(op: str) -> bytes:
    """The Update Control write for an operation with no further
    argument: query, finish, activate, or abort."""
    return bytes([UPDATE_OPS[op]])


@dataclass(frozen=True)
class UpdateResponse:
    """One answer from the Update service: an operation, a status, and
    whatever payload goes with it."""
    op: int
    status: int
    payload: bytes

    @property
    def ok(self) -> bool:
        """Whether the operation succeeded (status 0)."""
        return self.status == 0

    @property
    def status_name(self) -> str:
        """The status in words, or "undefined status N" if the contract
        does not define it."""
        return UPDATE_STATUS.get(self.status, f"undefined status {self.status}")

    @property
    def verify_result(self) -> str | None:
        """The verdict on an image, which only a finish answer carries. A
        start answer's first payload byte is a slot number, and reading it
        as a verdict would name a result the device never gave."""
        if self.op != UPDATE_OPS["finish"] or not self.payload:
            return None
        return VERIFY_RESULTS.get(self.payload[0], f"result {self.payload[0]}")

    def start(self) -> tuple[int, int, int]:
        """Which slot the device wants, the largest write it takes, and how
        much of this transfer it already has."""
        if len(self.payload) < 7:
            raise Truncated("a start answer is seven bytes")
        slot, chunk_max, resume = struct.unpack_from("<BHI", self.payload, 0)
        return slot, chunk_max, resume

    def query(self) -> tuple[str, int, int]:
        """The transfer's state, how many bytes the device has, and
        their checksum. Raises Truncated if the payload is under nine
        bytes."""
        if len(self.payload) < 9:
            raise Truncated("a query answer is nine bytes")
        state, offset, crc = struct.unpack_from("<BII", self.payload, 0)
        return UPDATE_STATES.get(state, f"state {state}"), offset, crc


_UPDATE_OP_NUMBERS = frozenset(UPDATE_OPS.values())


def decode_update_response(data: bytes) -> UpdateResponse:
    """One Update Control response, decoded into an `UpdateResponse`."""
    if len(data) < 2:
        raise Truncated("an update answer is at least an operation and a status")
    op, status = data[0], data[1]
    if op not in _UPDATE_OP_NUMBERS:
        raise Invalid(f"{op:#04x} is not an update operation this contract defines")
    if status not in UPDATE_STATUS:
        raise Invalid(f"status {status} is not one this contract defines")
    return UpdateResponse(op, status, bytes(data[2:]))


@dataclass(frozen=True)
class Envelope:
    """The plain head of a wrapped image. Everything after it is encrypted
    and means nothing to a host, which is the point: a host carries an
    update without being able to read or forge one."""
    target: str
    slot_link: int
    key_id: int
    nonce: bytes
    plain_len: int

    @property
    def total_len(self) -> int:
        """The whole wrapped image's length: the envelope plus the
        encrypted body."""
        return ENVELOPE_LEN + self.plain_len


def decode_envelope(data: bytes) -> Envelope:
    """An image's envelope header, decoded into an `Envelope`."""
    if len(data) < ENVELOPE_LEN:
        raise Truncated("an envelope is thirty two bytes")
    if data[0:4] != ENVELOPE_MAGIC or data[4] != 1:
        raise Invalid("not an image envelope this contract knows")
    target = {1: "app", 2: "weights"}.get(data[5])
    if target is None:
        raise Invalid(f"target {data[5]} is not something an envelope carries")
    slot_link = data[6]
    if target == "app" and slot_link > 1:
        raise Invalid("an application image is built for slot zero or one")
    if target == "weights" and slot_link != SLOT_LINK_NONE:
        raise Invalid("weights are not built for a slot")
    return Envelope(target, slot_link, data[7], bytes(data[8:24]),
                    struct.unpack_from("<I", data, 24)[0])


# --- heads ------------------------------------------------------------------

HEAD_MAGIC = b"IMHD"
HEAD_HEADER_LEN = 32
#: 1.3: format 2 appends the eight byte id of the encoder the head was
#: trained beside. The device stores and reports it and never judges it; a
#: host warns on a mismatch, and nothing blocks.
HEAD_HEADER_LEN_2 = 40
HEAD_HASH_LEN = 32
HEAD_KIND_LINEAR = 1
NO_ENCODER_ID = "0000000000000000"
#: Counts per unit of embedding. Part of the contract: changing it would
#: change every head ever trained.
EMBED_SCALE = 4096.0


def head_blob_len(in_dim: int, out_dim: int, version: int = 2) -> int:
    """The exact byte length of a head blob with `in_dim` inputs and
    `out_dim` outputs, in format `version`."""
    header = HEAD_HEADER_LEN_2 if version == 2 else HEAD_HEADER_LEN
    return header + out_dim * in_dim + 8 * out_dim + HEAD_HASH_LEN


def quantize_embedding(embedding) -> list[int]:
    """The integers a head receives for a float embedding. Saturates rather
    than wrapping, so an extreme value is clipped and never reversed."""
    out = []
    for v in embedding:
        scaled = float(v) * EMBED_SCALE
        if scaled >= 32767:
            out.append(32767)
        elif scaled <= -32768:
            out.append(-32768)
        else:
            out.append(int(scaled + 0.5) if scaled >= 0 else int(scaled - 0.5))
    return out


def build_head(weights, bias, scale, *, name: str = "", in_dim: int | None = None,
               encoder_id: str | None = None, version: int = 2):
    """A head blob, ready to upload.

    `weights` is one row per output, each row `in_dim` values in the range
    a signed byte holds. `bias` is one integer per output and `scale` one
    float per output, which together carry whatever range the outputs
    need. The hash is computed here, because a head the device cannot
    check is a head the device refuses.

    `encoder_id` names the encoder the head was trained beside (1.3, head
    format 2), so a host can warn when it is uploaded to a device running
    another. None writes the format 2 header with no encoder named.
    `version` 1 writes the 1.0 header, for a device that predates format 2.
    """
    import hashlib

    rows = [list(r) for r in weights]
    out_dim = len(rows)
    if out_dim == 0:
        raise Invalid("a head has at least one output")
    in_dim = in_dim if in_dim is not None else len(rows[0])
    if any(len(r) != in_dim for r in rows):
        raise Invalid("every row of a head is the same width")
    if len(bias) != out_dim or len(scale) != out_dim:
        raise Invalid("one bias and one scale per output")
    for r in rows:
        if any(not -128 <= int(v) <= 127 for v in r):
            raise Invalid("head weights are signed bytes, so they are trained or scaled to fit")
    encoder = bytes.fromhex(encoder_id or NO_ENCODER_ID)
    if len(encoder) != 8:
        raise Invalid("an encoder id is eight bytes, sixteen hex digits")
    if version not in (1, 2):
        raise Invalid("a head is written in format 1 or 2")
    if version == 1 and encoder != bytes(8):
        raise Invalid("format 1 has no room for an encoder id")
    encoded = bytearray()
    encoded += HEAD_MAGIC + bytes([version, HEAD_KIND_LINEAR])
    encoded += struct.pack("<HH", in_dim, out_dim)
    encoded += name.encode("utf-8")[:16].ljust(16, b"\0")
    encoded += b"\0" * 6
    if version == 2:
        encoded += encoder
    for r in rows:
        encoded += bytes((int(v) & 0xFF) for v in r)
    for b in bias:
        encoded += struct.pack("<i", int(b))
    for s in scale:
        encoded += struct.pack("<f", float(s))
    digest = hashlib.sha256(bytes(encoded)).digest()
    return bytes(encoded) + digest


@dataclass(frozen=True)
class HeadBlob:
    """A head blob, parsed into its shape, name, id, weights, bias, and
    scale."""
    in_dim: int
    out_dim: int
    name: str
    head_id: str
    weights: list[list[int]]
    bias: list[int]
    scale: list[float]
    #: The format version the file was written in: 1, or 2 with an encoder id.
    version: int = 2
    #: 1.3: the encoder the head was trained beside, sixteen zeros when the
    #: file does not say (every format 1 file, and a format 2 file whose
    #: trainer did not know).
    encoder_id: str = NO_ENCODER_ID


def decode_head(blob: bytes) -> HeadBlob:
    """Read a head back, and check it the way the device does."""
    import hashlib

    if len(blob) < HEAD_HEADER_LEN + HEAD_HASH_LEN:
        raise Truncated("a head is at least a header and a hash")
    if blob[0:4] != HEAD_MAGIC or blob[4] not in (1, 2):
        raise Invalid("not a head this contract knows")
    version = blob[4]
    if blob[5] != HEAD_KIND_LINEAR:
        raise Invalid(f"head kind {blob[5]} is not one this contract defines")
    in_dim, out_dim = struct.unpack_from("<HH", blob, 6)
    if len(blob) != head_blob_len(in_dim, out_dim, version):
        raise Invalid(f"a head of {in_dim} by {out_dim} is {head_blob_len(in_dim, out_dim, version)} bytes, "
                      f"and this one is {len(blob)}")
    body, digest = blob[:-HEAD_HASH_LEN], blob[-HEAD_HASH_LEN:]
    if hashlib.sha256(body).digest() != digest:
        raise Invalid("a head whose hash does not match its weights is refused, as the device refuses it")
    encoder_id = blob[32:40].hex() if version == 2 else NO_ENCODER_ID
    at = HEAD_HEADER_LEN_2 if version == 2 else HEAD_HEADER_LEN
    weights = [[int.from_bytes(blob[at + o * in_dim + i:at + o * in_dim + i + 1], "little", signed=True)
                for i in range(in_dim)] for o in range(out_dim)]
    at += out_dim * in_dim
    bias = list(struct.unpack_from(f"<{out_dim}i", blob, at))
    at += out_dim * 4
    scale = list(struct.unpack_from(f"<{out_dim}f", blob, at))
    return HeadBlob(in_dim, out_dim, blob[10:26].rstrip(b"\0").decode("utf-8", "replace"),
                    digest[:8].hex(), weights, bias, scale, version, encoder_id)


def evaluate_head(head: HeadBlob, embedding) -> list[float]:
    """What a head produces, computed the way the device computes it: the
    accumulation is exact in integers and only the scale is a float."""
    ints = quantize_embedding(embedding) if any(isinstance(v, float) for v in embedding) else list(embedding)
    if len(ints) != head.in_dim:
        raise Invalid(f"this head takes an embedding of {head.in_dim}, and it was given {len(ints)}")
    out = []
    for o in range(head.out_dim):
        acc = head.bias[o] + sum(w * e for w, e in zip(head.weights[o], ints))
        out.append(head.scale[o] * acc)
    return out
