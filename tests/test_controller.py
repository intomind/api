"""Controller capture: decoding, discovery, and the honesty of both.

    python3 tests/test_controller.py

No gamepad is attached and this machine grants no access to `/dev/input`, so
the read path is exercised through a FIFO instead: `Controller` opens a path
and calls `os.read`, which works on any character stream. That runs the real
constructor, the real `EVIOCSCLOCKID` attempt (a FIFO refuses it, which is the
fallback branch), and the real decode over real bytes.

Deliberately NOT tested by opening `/dev/input/event3`: that node is the
keyboard, and a test that reads it is a test that records keystrokes.
"""
from __future__ import annotations
import os, pathlib, shutil, struct, sys, tempfile, traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from intomind import controller as C      # noqa: E402

_TMP: pathlib.Path | None = None

PROC_SAMPLE = """\
I: Bus=0011 Vendor=0001 Product=0001 Version=ab41
N: Name="AT Translated Set 2 keyboard"
P: Phys=isa0060/serio0/input0
H: Handlers=sysrq kbd event3 leds
B: EV=120013

I: Bus=0018 Vendor=04f3 Product=312b Version=0100
N: Name="ELAN0504:01 04F3:312B Touchpad"
P: Phys=i2c-ELAN0504:01
H: Handlers=mouse0 event6
B: EV=b

I: Bus=0003 Vendor=2dc8 Product=3106 Version=0111
N: Name="8BitDo Ultimate 2C Wireless Controller"
P: Phys=usb-0000:00:14.0-1/input0
U: Uniq=e4:17:d8:00:00:01
H: Handlers=event12 js0
B: EV=20000b
"""


def use_tmp():
    global _TMP
    _TMP = pathlib.Path(tempfile.mkdtemp(prefix="intomind_ctl_"))


def _ev(sec, usec, etype, code, value):
    return struct.pack(C._FMT, sec, usec, etype, code, value)


def _fifo_controller():
    """A real Controller reading a real fd, with no hardware involved."""
    p = _TMP / f"pad{len(list(_TMP.iterdir()))}"
    os.mkfifo(p)
    c = C.Controller(str(p), name="fake pad")
    w = os.open(str(p), os.O_WRONLY | os.O_NONBLOCK)
    return c, w


# ---------------------------------------------------------------- the struct
def test_input_event_size_is_computed_not_assumed():
    """A hardcoded 24 misparses every event on a kernel whose timeval is 8
    bytes. The size must come from the format."""
    assert C._SIZE == struct.calcsize(C._FMT)
    assert C._SIZE == 24, "expected 24 on LP64; the format string is wrong"


def test_evioc_constant_matches_the_documented_value():
    """Derived from the _IOW macro rather than pasted. If the derivation is
    wrong the ioctl silently fails and every stamp quietly falls back to
    realtime, which is exactly the degradation this is meant to avoid."""
    assert C.EVIOCSCLOCKID == 0x400445A0, hex(C.EVIOCSCLOCKID)


# ---------------------------------------------------------------- discovery
def test_the_gamepad_is_found_by_its_js_handler():
    devs = C.list_devices(PROC_SAMPLE)
    pads = [d for d in devs if d["is_gamepad"]]
    assert len(pads) == 1, [d["name"] for d in pads]
    assert pads[0]["name"] == "8BitDo Ultimate 2C Wireless Controller"
    assert pads[0]["event_node"] == "/dev/input/event12"
    assert pads[0]["uniq"] == "e4:17:d8:00:00:01"


def test_a_keyboard_is_not_a_gamepad():
    """A keyboard reports EV_KEY too. Capability bits are not a discriminator;
    the joydev handler is."""
    kbd = [d for d in C.list_devices(PROC_SAMPLE) if "keyboard" in d["name"]][0]
    assert kbd["is_gamepad"] is False
    assert kbd["event_node"] == "/dev/input/event3"


def test_a_touchpad_is_not_a_gamepad():
    """And a touchpad reports EV_ABS."""
    tp = [d for d in C.list_devices(PROC_SAMPLE) if "Touchpad" in d["name"]][0]
    assert tp["is_gamepad"] is False


def test_devices_without_an_event_node_are_not_offered():
    d = C.list_devices('N: Name="odd"\nH: Handlers=js9\nB: EV=b\n')[0]
    assert d["event_node"] is None


# ---------------------------------------------------------------- reading
def test_poll_decodes_real_bytes_off_a_real_fd():
    c, w = _fifo_controller()
    try:
        os.write(w, _ev(100, 500000, C.EV_KEY, 0x130, 1)
                 + _ev(100, 500000, C.EV_SYN, 0, 0)
                 + _ev(100, 750000, C.EV_ABS, 0x00, -32768))
        ev = c.poll()
        assert len(ev) == 2, ev            # SYN dropped
        assert ev[0]["name"] == "BTN_SOUTH" and ev[0]["value"] == 1
        assert ev[1]["name"] == "ABS_X" and ev[1]["value"] == -32768
    finally:
        os.close(w)
        c.close()


def test_a_fifo_refuses_the_clock_ioctl_and_that_is_recorded():
    """The fallback branch. A device that will not switch keeps realtime
    stamps, and the recording has to say which it got."""
    c, w = _fifo_controller()
    try:
        assert c.kernel_clock == "realtime"
    finally:
        os.close(w)
        c.close()


def test_an_unknown_control_is_still_data():
    """A pad with a button this module has never heard of must record the
    press, not drop it for lacking a name."""
    c, w = _fifo_controller()
    try:
        os.write(w, _ev(1, 0, C.EV_KEY, 0x2ff, 1))
        ev = c.poll()
        assert len(ev) == 1 and ev[0]["code"] == 0x2FF
        assert ev[0]["name"] == "KEY_0x2ff"
    finally:
        os.close(w)
        c.close()


def test_monotonic_stamps_are_used_directly():
    """When the kernel is stamping monotonic, that value IS `mono` — no
    host-side arithmetic between the interrupt and the record."""
    c, w = _fifo_controller()
    try:
        c.kernel_clock = "monotonic"
        os.write(w, _ev(4242, 250000, C.EV_KEY, 0x131, 1))
        e = c.poll()[0]
        assert abs(e["mono"] - 4242.25) < 1e-9, e["mono"]
    finally:
        os.close(w)
        c.close()


def test_realtime_stamps_are_carried_onto_monotonic():
    """With realtime stamps, `host_time` is the precise one and `mono` is
    derived from it. The pair must stay consistent, because alignment uses
    `mono` and a reader has to be able to trust the relationship."""
    import time as _t
    c, w = _fifo_controller()
    try:
        now = _t.time()
        os.write(w, _ev(int(now), 0, C.EV_KEY, 0x130, 1))
        e = c.poll()[0]
        assert abs(e["host_time"] - int(now)) < 1e-6
        # the offset between the two clocks is the same one the host sees now
        assert abs((e["host_time"] - e["mono"]) - (_t.time() - _t.monotonic())) < 0.5
    finally:
        os.close(w)
        c.close()


def test_nothing_ready_is_not_an_error():
    c, w = _fifo_controller()
    try:
        assert c.poll() == []
    finally:
        os.close(w)
        c.close()


def test_a_partial_event_is_held_not_misparsed():
    """Half an event now and half later must decode to one event, not two
    wrong ones."""
    c, w = _fifo_controller()
    try:
        blob = _ev(7, 0, C.EV_ABS, 0x01, 1234)
        os.write(w, blob[:10])
        assert c.poll() == []
        os.write(w, blob[10:])
        ev = c.poll()
        assert len(ev) == 1 and ev[0]["value"] == 1234
    finally:
        os.close(w)
        c.close()


# ---------------------------------------------------------------- recorder
def test_buttons_and_axes_are_counted_separately():
    """A session with axis traffic and no buttons is a stick being waggled; a
    session with neither is a pad nobody touched. One total hides both."""
    r = C.InputRecorder()
    r.feed([dict(mono=1.0, host_time=1.0, type=C.EV_KEY, code=0x130, value=1),
            dict(mono=1.5, host_time=1.5, type=C.EV_ABS, code=0, value=10),
            dict(mono=2.0, host_time=2.0, type=C.EV_ABS, code=1, value=-10)])
    s = r.summary()
    assert s["n_events"] == 3
    assert s["n_button_events"] == 1 and s["n_axis_events"] == 2
    assert abs(s["span_s"] - 1.0) < 1e-9


def test_the_recorder_names_which_clock_each_source_used():
    c, w = _fifo_controller()
    try:
        r = C.InputRecorder()
        r.add_source(c)
        assert r.summary()["kernel_clocks"] == ["realtime"]
        assert r.summary()["sources"][0]["name"] == "fake pad"
    finally:
        os.close(w)
        c.close()


def test_arrays_round_trip_through_numpy():
    import numpy as np
    r = C.InputRecorder()
    r.feed([dict(mono=1.0, host_time=2.0, type=C.EV_KEY, code=0x130, value=1)])
    a = r.arrays()
    assert a["input_mono"].dtype == np.float64
    assert int(a["input_code"][0]) == 0x130


def test_nothing_ever_asks_for_blanket_input_access():
    """A gamepad needs no setup: udev tags joysticks `uaccess` and logind puts
    an ACL on the node for the seat's user.

    Anything granting blanket `/dev/input` access also grants the keyboard,
    which is the one device a gamepad recorder must never be able to read. So
    no message this module produces may suggest it."""
    import inspect
    src = inspect.getsource(C.describe_availability)
    advice = src[src.index('if not DEVICES_PROC.exists()'):]
    reason = (C.describe_availability().get("reason") or "")
    for bad in ("usermod", "input' group", 'input" group', "input group",
                "chmod", "rules.d"):
        assert bad not in advice, f"the code suggests {bad!r}"
        assert bad not in reason, f"the message suggests {bad!r}"


def test_no_gamepad_is_reported_as_no_gamepad():
    """Not as a permission problem. This machine has none attached, so this
    is the live case."""
    got = C.describe_availability()
    if not got["ok"] and not got["pads"]:
        assert "no gamepad detected" in got["reason"]
        assert "Bluetooth" in got["reason"]      # the single-BLE-link contract


def test_availability_says_which_problem_it_is():
    """'Not plugged in' and 'plugged in, no permission' need different fixes,
    so they must not collapse into one message."""
    got = C.describe_availability()
    assert got["ok"] in (True, False)
    if not got["ok"]:
        assert got["reason"], "an unavailable controller must say why"


# --- the map has to fit the pad that is actually plugged in ------------------
def test_eviocgbit_is_derived_from_the_macro():
    """A wrong constant returns an empty bitmap, and an empty bitmap looks
    exactly like 'this pad has no buttons' rather than like a bug."""
    # EVIOCGBIT(EV_KEY, 96) = _IOC(_IOC_READ,'E',0x20+1,96)
    assert C._eviocgbit(C.EV_KEY, 96) == (2 << 30) | (96 << 16) | (0x45 << 8) | 0x21
    assert C._eviocgbit(C.EV_ABS, 8) == (2 << 30) | (8 << 16) | (0x45 << 8) | 0x23


def test_an_action_reachable_by_either_form_is_not_a_hole():
    """A shoulder arrives as a digital button on one pad and an analog axis
    on another. Mapping both and having one absent is normal, not a fault."""
    caps = dict(buttons=["BTN_TL"], axes=["ABS_Z"])
    r = C.check_map({"BTN_TL2": "L2", "ABS_Z": "L2"}, caps)
    assert r["unreachable_actions"] == []
    assert "BTN_TL2" in r["missing"]          # informational, not a hole


def test_an_action_with_no_surviving_control_is_reported():
    """The real failure: a binding that can never fire on this pad."""
    caps = dict(buttons=["BTN_SOUTH"], axes=[])
    r = C.check_map({"BTN_TL2": "L2", "BTN_SOUTH": "Cross"}, caps)
    assert r["unreachable_actions"] == ["L2"]


def test_capabilities_on_a_missing_node_is_reported_not_empty():
    """An unreadable pad must not come back looking like a pad with no
    buttons -- that is the failure mode this whole check exists to prevent."""
    caps = C.capabilities(str(_TMP / "nope"))
    assert caps["error"] and caps["buttons"] == [] and caps["axes"] == []


def main() -> int:
    use_tmp()
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            failed.append(t.__name__)
            print(f"  FAIL  {t.__name__}: {e}")
            if "-v" in sys.argv:
                traceback.print_exc()
    if _TMP and _TMP.exists():
        shutil.rmtree(_TMP)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("failed:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
