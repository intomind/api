"""The experiments module keeps the protocol module reachable, and every
recording arms on a fresh count.

    python3 tests/test_experiments.py

The module imports `protocol` and also registers an experiment called
"protocol". The experiment's function once carried that same name, which
rebound the module-level name, so `_processing` and `_apply` found a
function where they expected the protocol module. The first test holds the
line. The second runs both callers against a stand-in device.

Five recordings armed the recorder before resetting the count, so a run
started on a live stream saved the end of that stream first. The last three
tests hold the order in `_arm` and hold every recording to it.
"""
import asyncio
import inspect
import pathlib
import re
import sys
import traceback
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from intomind import experiments as X  # noqa: E402
from intomind import protocol as P  # noqa: E402


class _Device:
    """Enough of a device for `_processing` and `_apply`."""

    config = {"rate_sps": 500}
    mode = None

    def can(self, name):
        return name == "pipeline"

    async def pipeline(self):
        return types.SimpleNamespace(origin="default", stages=[])

    async def set_mode(self, code):
        self.mode = code

    async def refresh_config(self):
        return {"mode_code": self.mode}


def test_the_protocol_module_is_not_shadowed():
    assert X.protocol is P, f"experiments.protocol is {X.protocol!r}, not the protocol module"
    assert "protocol" in X.REGISTRY, "the cued protocol experiment is still registered under its name"


def test_the_two_callers_reach_the_protocol_module():
    dev = _Device()
    proc = asyncio.run(X._processing(dev))
    assert proc["origin"] == "default" and "description" in proc, proc
    cfg = asyncio.run(X._apply(dev, None, "normal", None))
    assert cfg["mode"] == "normal" and dev.mode == P.MODES["normal"], cfg


class _Stream:
    """A device and a recorder in one, writing down the order they were
    used in."""

    def __init__(self, streaming):
        self.streaming = streaming
        self.calls = []

    async def reset_epoch(self):
        self.calls.append("reset")

    async def start(self, samples_per_packet=None):
        self.calls.append("start")

    def arm(self, _dev):
        self.calls.append("arm")


def test_a_running_stream_is_reset_before_the_recorder_arms():
    s = _Stream(streaming=True)
    started = asyncio.run(X._arm(s, s))
    assert s.calls == ["reset", "arm"], s.calls
    assert started is False, "a stream that was already running was started again"


def test_a_stopped_stream_is_armed_then_started():
    s = _Stream(streaming=False)
    started = asyncio.run(X._arm(s, s))
    assert s.calls == ["arm", "start"], s.calls
    assert started is True


def test_every_recording_arms_through_one_function():
    """Arming anywhere else is how five recordings came to arm before the
    reset, so the module is read for any other call."""
    calls = re.findall(r"\brec\.arm\(", inspect.getsource(X))
    assert len(calls) == 1 and re.search(r"\brec\.arm\(", inspect.getsource(X._arm)), \
        f"the recorder is armed in {len(calls)} places, not only in _arm"
    for name in ("_record", "berger", "record", "cued_protocol", "_cued_trials", "saccade"):
        fn = inspect.unwrap(getattr(X, name))
        assert "await _arm(dev, rec" in inspect.getsource(fn), name


def main() -> int:
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
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("failed:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
