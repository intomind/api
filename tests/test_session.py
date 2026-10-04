"""The multi-device session connects as soon as its devices are heard."""
from __future__ import annotations
import asyncio, pathlib, sys, time, traceback
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from intomind import session as S  # noqa: E402


class FakeDevice:
    def __init__(self, bd):
        self.bd = bd

    async def connect(self):
        return self


def fake_scan(devices, timeout_when_unheard):
    """A scan with the api's own contract: `found` per device as it is
    heard, `stop` ending the scan early, the whole timeout otherwise."""
    async def scan(timeout, on_device=None, stop=None):
        for d in devices:
            await asyncio.sleep(0.01)          # heard, one after another
            if on_device is not None:
                on_device(d)
        try:
            await asyncio.wait_for((stop or asyncio.Event()).wait(), timeout)
        except asyncio.TimeoutError:
            timeout_when_unheard.append(timeout)
        return list(devices)
    return scan


def test_the_session_connects_the_moment_enough_devices_are_heard():
    two = [SimpleNamespace(address="AA:00", name="one"), SimpleNamespace(address="BB:00", name="two")]
    waited_out = []
    S.scan, S.Device = fake_scan(two, waited_out), FakeDevice
    t0 = time.monotonic()
    devices = asyncio.run(S.Session().discover_and_connect(expect=2, timeout=15))
    took = time.monotonic() - t0
    assert [d.bd.address for d in devices] == ["AA:00", "BB:00"]
    assert took < 1.0, f"two devices heard in 20 ms took {took:.2f} s to connect"
    assert waited_out == [], "the scan should have been stopped, not waited out"


def test_a_missing_device_still_waits_the_timeout_and_then_says_so():
    one = [SimpleNamespace(address="AA:00", name="one")]
    waited_out = []
    S.scan, S.Device = fake_scan(one, waited_out), FakeDevice
    try:
        asyncio.run(S.Session().discover_and_connect(expect=2, timeout=0.2))
    except RuntimeError as e:
        assert "found 1 device(s), expected 2" in str(e)
    else:
        raise AssertionError("one device heard of two expected should be an error")
    assert waited_out == [0.2], "with too few devices the scan runs to its timeout"


def test_the_recorders_file_says_which_samples_were_generated():
    import csv, tempfile
    from intomind.client import Sample
    s = S.Session()
    dev = SimpleNamespace(info=SimpleNamespace(channels=2, device_id="00000000c0de0001"))
    s.devices = [dev]
    path = pathlib.Path(tempfile.mkdtemp(prefix="intomind_session_")) / "r.csv"
    s.record_to(str(path))
    s._sink(dev, Sample(0, 0, 0.0, (1, 2), (0.1, 0.2), 0))
    s._sink(dev, Sample(1, 2, 0.002, (1, 2), (0.1, 0.2), 0, synthetic=True))
    s._file.close()
    rows = list(csv.reader(path.open()))
    assert rows[0][-1] == "synthetic" and len(rows[0]) == len(rows[1])
    assert [r[-1] for r in rows[1:]] == ["0", "1"]


def test_connect_connects_exactly_the_devices_chosen():
    three = [SimpleNamespace(address=a, name="IntoMind One") for a in ("AA:00", "BB:00", "CC:00")]
    S.Device = FakeDevice
    s = S.Session()
    asyncio.run(s.connect(three[1]))
    assert [d.bd.address for d in s.devices] == ["BB:00"], "only the one picked is connected"


def test_connecting_whatever_answers_first_is_deprecated():
    import warnings
    one = [SimpleNamespace(address="AA:00", name="one")]
    S.scan, S.Device = fake_scan(one, []), FakeDevice
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(S.Session().discover_and_connect(expect=1, timeout=1))
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


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
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
