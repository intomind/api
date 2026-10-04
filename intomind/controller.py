"""Game controller input, read straight from the kernel.

The label stream for the game paradigm: what the player did, stamped precisely
enough to sit beside the neural samples. Reading `/dev/input/eventN` directly
with `struct` rather than through `python-evdev`, because `host/` is the product
and runs on other people's benches -- a package that has to be installed and
compiled to record a button press is a dependency this does not need.

**Timestamps come from the kernel, not from Python.** The kernel stamps an
input event when the interrupt is handled; anything measured after the read
returns has a scheduler between it and the truth. That difference does not
matter for a cue spoken by the protocol and matters a great deal for the exact
instant a thumb moved.

Which clock the kernel uses is asked for and then *recorded*: `EVIOCSCLOCKID`
switches the device to CLOCK_MONOTONIC, which is what alignment wants (see
timebase.py -- monotonic cannot step). If the switch is refused, the stamps stay
on CLOCK_REALTIME and the recording says so in `kernel_clock`, so a reader can
tell a precise monotonic stamp from one that had to be approximated.

Storage: controller events go to a compact array file, not into `events.json`.
An analog stick generates a couple of hundred events a second and an hour of
play is most of a million of them -- as JSON that is a hundred megabytes of
record for a few kilobytes of meaning. `events.json` keeps the protocol's own
sparse cues. **One timebase, two containers, chosen by rate**; both are joined
by `analysis.inputs()`.
"""
from __future__ import annotations

import array
import fcntl
import os
import pathlib
import re
import struct
import time

from . import timebase

# ---------------------------------------------------------------- event codes
# linux/input-event-codes.h. Only what this module names; the raw type/code
# integers are recorded regardless, so an unrecognized control is still data.
EV_SYN, EV_KEY, EV_REL, EV_ABS, EV_MSC = 0x00, 0x01, 0x02, 0x03, 0x04
SYN_REPORT = 0

BTN_NAMES = {
    0x130: "BTN_SOUTH", 0x131: "BTN_EAST", 0x132: "BTN_C", 0x133: "BTN_NORTH",
    0x134: "BTN_WEST", 0x135: "BTN_Z", 0x136: "BTN_TL", 0x137: "BTN_TR",
    0x138: "BTN_TL2", 0x139: "BTN_TR2", 0x13a: "BTN_SELECT",
    0x13b: "BTN_START", 0x13c: "BTN_MODE", 0x13d: "BTN_THUMBL",
    0x13e: "BTN_THUMBR",
}
ABS_NAMES = {
    0x00: "ABS_X", 0x01: "ABS_Y", 0x02: "ABS_Z", 0x03: "ABS_RX",
    0x04: "ABS_RY", 0x05: "ABS_RZ", 0x10: "ABS_HAT0X", 0x11: "ABS_HAT0Y",
}
EV_NAMES = {EV_SYN: "SYN", EV_KEY: "KEY", EV_REL: "REL", EV_ABS: "ABS",
            EV_MSC: "MSC"}


def code_name(ev_type: int, code: int) -> str:
    """A readable name, falling back to the number. Never invents a meaning."""
    if ev_type == EV_KEY:
        return BTN_NAMES.get(code, f"KEY_{code:#x}")
    if ev_type == EV_ABS:
        return ABS_NAMES.get(code, f"ABS_{code:#x}")
    return f"{EV_NAMES.get(ev_type, ev_type)}_{code:#x}"


# ---------------------------------------------------------------- the struct
# struct input_event { struct timeval time; __u16 type, code; __s32 value; }
# Sizes are computed, never assumed: `timeval` is two longs, which is 16 bytes
# on LP64 and 8 on a 32-bit kernel, and a hardcoded 24 silently misparses every
# event on the wrong one.
_FMT = "@llHHi"
_SIZE = struct.calcsize(_FMT)

# EVIOCSCLOCKID, derived rather than pasted as a magic number:
#   _IOW(type, nr, size) = (_IOC_WRITE << 30) | (size << 16) | (type << 8) | nr
#   _IOC_WRITE = 1, type = 'E', nr = 0xa0, size = sizeof(int) = 4
_IOC_WRITE = 1
EVIOCSCLOCKID = (_IOC_WRITE << 30) | (4 << 16) | (ord("E") << 8) | 0xA0
CLOCK_MONOTONIC = 1

DEVICES_PROC = pathlib.Path("/proc/bus/input/devices")
INPUT_DIR = pathlib.Path("/dev/input")

# EVIOCGBIT(ev, len) = _IOC(_IOC_READ, 'E', 0x20 + ev, len). Derived from the
# macro for the same reason EVIOCSCLOCKID is: a wrong constant returns an
# empty bitmap, and an empty bitmap looks exactly like "this pad has no
# buttons" rather than like a bug.
_IOC_READ = 2
KEY_MAX, ABS_MAX = 0x2FF, 0x3F


def _eviocgbit(ev: int, length: int) -> int:
    return (_IOC_READ << 30) | (length << 16) | (ord("E") << 8) | (0x20 + ev)


# EVIOCGABS(abs) = _IOR('E', 0x40 + abs, struct input_absinfo). Same derivation
# again. `struct input_absinfo` is six __s32: value, minimum, maximum, fuzz,
# flat, resolution.
#
# Why this is worth asking for: an analysis of stick movement needs a DEADZONE,
# and without absinfo the only options are to guess a fraction of full scale or
# to infer the range from whatever the player happened to push. The kernel
# already knows -- `flat` IS the driver's declared deadzone and `minimum`/
# `maximum` are the true full scale. The first real session (2026-08-06) went to
# disk without them, so its deadzone has to be inferred; no later one should.
_ABSINFO_FMT = "@6i"
_ABSINFO_SIZE = struct.calcsize(_ABSINFO_FMT)


def _eviocgabs(code: int) -> int:
    return ((_IOC_READ << 30) | (_ABSINFO_SIZE << 16) | (ord("E") << 8)
            | (0x40 + code))


def axis_ranges(fd: int, codes) -> dict:
    """Each axis's real range and the driver's own deadzone, from the kernel."""
    out = {}
    for code in codes:
        buf = array.array("i", [0] * 6)
        try:
            fcntl.ioctl(fd, _eviocgabs(int(code)), buf)
        except OSError:
            continue
        value, minimum, maximum, fuzz, flat, resolution = buf
        out[code_name(EV_ABS, int(code))] = dict(
            min=int(minimum), max=int(maximum), fuzz=int(fuzz),
            flat=int(flat), resolution=int(resolution))
    return out


# ---------------------------------------------------------------- discovery
def _blocks(text: str):
    for blk in text.split("\n\n"):
        if blk.strip():
            yield blk


def list_devices(proc_text: str | None = None) -> list[dict]:
    """Every input device the kernel is advertising.

    Parsed from `/proc/bus/input/devices` rather than by opening each node,
    because opening a device to find out what it is requires permission to read
    it, and this has to be able to report "found it, cannot open it" as a
    distinct outcome from "not plugged in".
    """
    if proc_text is None:
        if not DEVICES_PROC.exists():
            return []
        proc_text = DEVICES_PROC.read_text(errors="replace")
    out = []
    for blk in _blocks(proc_text):
        d = dict(name=None, vendor=None, product=None, handlers=[],
                 ev_bits=0, phys=None, uniq=None)
        for line in blk.splitlines():
            if line.startswith("I:"):
                for k, v in re.findall(r"(\w+)=([0-9a-fA-F]+)", line):
                    if k in ("Vendor", "Product"):
                        d[k.lower()] = v
            elif line.startswith("N: Name="):
                d["name"] = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("P: Phys="):
                d["phys"] = line.split("=", 1)[1].strip()
            elif line.startswith("U: Uniq="):
                d["uniq"] = line.split("=", 1)[1].strip() or None
            elif line.startswith("H: Handlers="):
                d["handlers"] = line.split("=", 1)[1].split()
            elif line.startswith("B: EV="):
                try:
                    d["ev_bits"] = int(line.split("=", 1)[1].strip(), 16)
                except ValueError:
                    pass
        ev = next((h for h in d["handlers"] if h.startswith("event")), None)
        d["event_node"] = f"/dev/input/{ev}" if ev else None
        # The kernel's joydev driver binds a `js*` handler to joysticks and
        # gamepads and to nothing else, which is a far better discriminator
        # than guessing from capability bits: a keyboard also reports EV_KEY
        # and a tablet also reports EV_ABS.
        d["is_gamepad"] = any(h.startswith("js") for h in d["handlers"])
        out.append(d)
    return out


def find_gamepads() -> list[dict]:
    """Every discovered input device the kernel classifies as a gamepad
    and that has an event node to read."""
    return [d for d in list_devices() if d["is_gamepad"] and d["event_node"]]


def _bits(fd: int, ev: int, maxcode: int) -> set[int]:
    n = (maxcode // 8) + 1
    buf = array.array("B", [0] * n)
    try:
        fcntl.ioctl(fd, _eviocgbit(ev, n), buf)
    except OSError:
        return set()
    return {c for c in range(maxcode + 1) if buf[c // 8] >> (c % 8) & 1}


def capabilities(node: str) -> dict:
    """Every control this pad actually has, asked of the kernel.

    Not inferred from what someone happened to press. A pad's d-pad may be a
    hat, two axes or four buttons depending on the model and its mode switch,
    and a control map written against the wrong one silently labels nothing.
    Reported as names, with the raw codes kept, so an unrecognized control is
    still visible rather than dropped.
    """
    try:
        fd = os.open(node, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as e:
        return dict(node=node, error=str(e), buttons=[], axes=[], codes={})
    try:
        keys = _bits(fd, EV_KEY, KEY_MAX)
        axes = _bits(fd, EV_ABS, ABS_MAX)
        ranges = axis_ranges(fd, sorted(axes))
    finally:
        os.close(fd)
    return dict(
        node=node, error=None,
        buttons=sorted(code_name(EV_KEY, c) for c in keys),
        axes=sorted(code_name(EV_ABS, c) for c in axes),
        codes=dict(key=sorted(keys), abs=sorted(axes)),
        # Full scale and the driver's declared deadzone, per axis. An analysis
        # that has to guess these is an analysis whose threshold nobody can
        # check afterwards.
        axis_ranges=ranges,
    )


def check_map(controls: dict, caps: dict) -> dict:
    """Compare a game's control map against the pad that is plugged in.

    Two failures this catches, both of which otherwise show up only as missing
    labels in an hour of data:

      * `missing` -- the map names a control this pad does not have, so that
        binding can never fire.
      * `unmapped` -- the pad has a control the map says nothing about. Not an
        error (a pad has buttons a game does not use) but worth seeing, since
        it is also what a wrong d-pad mode looks like.
    """
    have = set(caps.get("buttons") or []) | set(caps.get("axes") or [])
    named = set(controls or {})
    # Which game ACTIONS this pad cannot produce. That is the question that
    # matters: a shoulder mapped from both a digital button and an analog
    # axis is reachable whenever the pad has either, and reporting the absent
    # one as a fault would be noise. An action with no surviving control is a
    # real hole -- it is a binding that can never fire.
    by_action: dict[str, set] = {}
    for ctl_name, action in (controls or {}).items():
        by_action.setdefault(action, set()).add(ctl_name)
    unreachable = sorted(a for a, ctls in by_action.items()
                         if not (ctls & have))
    return dict(missing=sorted(named - have), unmapped=sorted(have - named),
                matched=sorted(named & have), unreachable_actions=unreachable)


def describe_availability() -> dict:
    """What is present, and if nothing is, why — plainly.

    **A gamepad needs no setup of any kind.** udev tags every joystick
    `uaccess` (`70-uaccess.rules`: `SUBSYSTEM=="input",
    ENV{ID_INPUT_JOYSTICK}=="?*", TAG+="uaccess"`) and logind then puts an ACL
    on the node for whoever is logged in at the seat. Plug the pad in and it is
    readable.

    So this never asks the user to change a group or a permission. Blanket
    `/dev/input` access would also pick up the keyboard, which is out of scope
    here and not worth asking for.
    """
    if not DEVICES_PROC.exists():
        return dict(ok=False, reason="no /proc/bus/input/devices (not Linux?)",
                    pads=[])
    pads = find_gamepads()
    if not pads:
        return dict(ok=False, pads=[],
                    reason="no gamepad detected — plug one in (USB or its "
                           "2.4 GHz dongle; not Bluetooth, which would be a "
                           "second radio client on this host)")
    readable = [p for p in pads if os.access(p["event_node"], os.R_OK)]
    if not readable:
        nodes = ", ".join(p["event_node"] for p in pads)
        return dict(ok=False, pads=pads,
                    reason=f"found {len(pads)} gamepad(s) but cannot read "
                           f"{nodes}. That is unusual: udev normally grants "
                           f"the seat's user an ACL on any joystick. Likely a "
                           f"session without a seat (ssh, or a service), or a "
                           f"pad udev did not classify as a joystick. Check "
                           f"`getfacl {nodes.split(',')[0]}`")
    return dict(ok=True, reason=None, pads=readable)


# ---------------------------------------------------------------- reading
class Controller:
    """One open input device, read event by event.

    Non-blocking: `poll()` drains whatever is ready and returns it. The caller
    decides when to poll, so this does not own a thread or an event loop and
    cannot wedge a recording by blocking on a pad that stopped sending.
    """

    def __init__(self, node: str, *, name: str | None = None):
        self.node = str(node)
        self.name = name
        self.fd = os.open(self.node, os.O_RDONLY | os.O_NONBLOCK)
        self.kernel_clock = "realtime"
        try:
            # Ask for monotonic stamps. See timebase.py: monotonic is the clock
            # alignment must use, because it cannot step.
            fcntl.ioctl(self.fd, EVIOCSCLOCKID,
                        array.array("i", [CLOCK_MONOTONIC]))
            self.kernel_clock = "monotonic"
        except OSError:
            # Old kernel, or a device that refuses. Recorded, not fatal:
            # realtime stamps are still precise, they just need the anchors to
            # be aligned, and `kernel_clock` tells the reader which case it is.
            pass
        self._buf = b""

    def close(self) -> None:
        """Close the device node, if it is still open."""
        if self.fd is not None:
            try:
                os.close(self.fd)
            finally:
                self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def poll(self) -> list[dict]:
        """Every input event available now, oldest first.

        A short read cannot split an event in the kernel's own API, but a
        partial buffer is kept anyway: this costs nothing and the alternative
        is silently discarding a byte boundary if that ever stops being true.
        """
        try:
            data = os.read(self.fd, _SIZE * 256)
        except BlockingIOError:
            return []
        except OSError:
            return []
        self._buf += data
        n = len(self._buf) // _SIZE
        raw, self._buf = self._buf[: n * _SIZE], self._buf[n * _SIZE:]
        now = timebase.stamp()
        out = []
        for i in range(n):
            sec, usec, etype, code, value = struct.unpack_from(
                _FMT, raw, i * _SIZE)
            if etype == EV_SYN:
                continue            # frame delimiter, carries no state
            k = sec + usec / 1e6
            if self.kernel_clock == "monotonic":
                mono, host = k, now["host_time"] - (now["mono"] - k)
            else:
                mono, host = now["mono"] - (now["host_time"] - k), k
            out.append(dict(mono=mono, host_time=host, type=etype, code=code,
                            value=value, name=code_name(etype, code)))
        return out


class InputRecorder:
    """Accumulates controller events as columns, for a compact array file.

    Columns rather than dicts because the volume is real: an analog stick
    alone produces a few hundred events a second. The arrays are the record;
    `summary()` is what goes in the manifest.
    """

    def __init__(self):
        self.mono: list[float] = []
        self.host: list[float] = []
        self.type: list[int] = []
        self.code: list[int] = []
        self.value: list[int] = []
        self.sources: list[dict] = []
        self.dropped = 0

    def add_source(self, c: Controller, controls: dict | None = None) -> None:
        """Record a pad, what it can produce, and whether the map fits it.

        The capability set goes into the capture because it bounds what the
        session could possibly have labeled: an action with no control behind
        it on this pad never fired, and six months later that is indistinguish-
        able from an action the player simply never used.
        """
        caps = capabilities(c.node)
        self.sources.append(dict(
            node=c.node, name=c.name, kernel_clock=c.kernel_clock,
            buttons=caps["buttons"], axes=caps["axes"],
            # Full scale and the driver's declared deadzone per axis. Without
            # these an analysis of stick movement has to guess its own
            # threshold from whatever the player happened to push, and a
            # threshold nobody can trace is one nobody can check.
            axis_ranges=caps.get("axis_ranges") or {},
            capability_error=caps["error"],
            map_check=(check_map(controls, caps) if controls else None)))

    def feed(self, events) -> int:
        """Append events, as `Controller.poll` returns them, to the
        recorder's columns. Returns the total event count so far."""
        for e in events:
            self.mono.append(e["mono"])
            self.host.append(e["host_time"])
            self.type.append(e["type"])
            self.code.append(e["code"])
            self.value.append(e["value"])
        return len(self.mono)

    def __len__(self) -> int:
        return len(self.mono)

    def summary(self) -> dict:
        """What the manifest records about the input stream.

        `buttons` and `axes` are counted separately because they answer
        different questions: a session with axis traffic and zero button events
        is a stick being waggled, and a session with neither is a controller
        that was connected and never touched. Both look identical in a single
        total, and both are worth knowing before analyzing an hour of it.
        """
        btn = sum(1 for t in self.type if t == EV_KEY)
        axis = sum(1 for t in self.type if t == EV_ABS)
        span = (max(self.mono) - min(self.mono)) if self.mono else 0.0
        return dict(sources=self.sources, n_events=len(self.mono),
                    n_button_events=btn, n_axis_events=axis,
                    span_s=span, dropped=self.dropped,
                    kernel_clocks=sorted({s["kernel_clock"]
                                          for s in self.sources}))

    def arrays(self) -> dict:
        """The recorded columns as numpy arrays, ready to save into a
        capture's input file."""
        import numpy as np
        return dict(
            input_mono=np.asarray(self.mono, dtype=np.float64),
            input_host_time=np.asarray(self.host, dtype=np.float64),
            input_type=np.asarray(self.type, dtype=np.int16),
            input_code=np.asarray(self.code, dtype=np.int32),
            input_value=np.asarray(self.value, dtype=np.int32),
        )
