"""EDF+ export. Read the bytes back and check them against the standard.

    python3 tests/test_edf.py

The point of an export is that software we did not write can read it, so the
tests parse the header the way a stranger would: fixed-width ASCII fields at
fixed offsets, then int16 data records.
"""
from __future__ import annotations
import datetime as _dt
import pathlib, shutil, sys, tempfile, traceback

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from intomind import edf                     # noqa: E402

START = _dt.datetime(2026, 7, 10, 4, 3, 10)
FS = 498.80
_TMP: pathlib.Path | None = None


def tmp() -> pathlib.Path:
    global _TMP
    if _TMP is None:
        _TMP = pathlib.Path(tempfile.mkdtemp(prefix="intomind_edf_"))
    return _TMP


def parse_header(b: bytes) -> dict:
    f = lambda o, n: b[o:o + n].decode("ascii").strip()   # noqa: E731
    ns = int(f(252, 4))
    h = dict(version=f(0, 8), patient=f(8, 80), recording=f(88, 80),
             startdate=f(168, 8), starttime=f(176, 8), header_bytes=int(f(184, 8)),
             reserved=f(192, 44), n_records=int(f(236, 8)),
             duration=float(f(244, 8)), ns=ns)
    o = 256
    h["labels"] = [f(o + i * 16, 16) for i in range(ns)]
    o += 16 * ns
    h["transducer"] = [f(o + i * 80, 80) for i in range(ns)]
    o += 80 * ns
    h["dim"] = [f(o + i * 8, 8) for i in range(ns)]
    o += 8 * ns
    h["phys_min"] = [float(f(o + i * 8, 8)) for i in range(ns)]
    o += 8 * ns
    h["phys_max"] = [float(f(o + i * 8, 8)) for i in range(ns)]
    o += 8 * ns
    h["dig_min"] = [int(f(o + i * 8, 8)) for i in range(ns)]
    o += 8 * ns
    h["dig_max"] = [int(f(o + i * 8, 8)) for i in range(ns)]
    o += 8 * ns
    h["prefilter"] = [f(o + i * 80, 80) for i in range(ns)]
    o += 80 * ns
    h["nsr"] = [int(f(o + i * 8, 8)) for i in range(ns)]
    return h


def write(name, n=3000, index=None, events=None, fs=FS):
    t = np.arange(n) / fs
    sigs = {"ch1": 1500.0 + 20 * np.sin(2 * np.pi * 10 * t),
            "ch2": -900.0 + 15 * np.sin(2 * np.pi * 10 * t)}
    p = tmp() / name
    info = edf.write_edf(p, signals=sigs, fs=fs, start=START,
                         index=index, events=events)
    return p, info, sigs


# ---------------------------------------------------------------- header
def test_header_is_exactly_the_right_size_and_shape():
    p, info, _ = write("a.edf")
    b = p.read_bytes()
    h = parse_header(b)
    assert h["version"] == "0"
    assert h["ns"] == 3                       # ch1, ch2, EDF Annotations
    assert h["header_bytes"] == 256 * 4 == len(b) - h["n_records"] * (
        2 * (h["nsr"][0] + h["nsr"][1] + h["nsr"][2]))
    assert h["labels"][2] == "EDF Annotations"
    assert h["dim"][:2] == ["uV", "uV"]
    assert h["dig_min"] == [-32768] * 3 and h["dig_max"] == [32767] * 3


def test_startdate_and_time_are_edf_formatted():
    p, _, _ = write("b.edf")
    h = parse_header(p.read_bytes())
    assert h["startdate"] == "10.07.26"
    assert h["starttime"] == "04.03.10"


def test_non_integer_rate_lives_in_the_record_duration():
    """498.80 SPS. Rounding to 500 would misplace a 10 Hz rhythm by 0.024 Hz."""
    p, info, _ = write("c.edf")
    h = parse_header(p.read_bytes())
    assert h["nsr"][0] == 499
    assert abs(h["duration"] - 499 / FS) < 1e-6
    assert abs(info["record_duration_s"] * info["fs"] - 499) < 1e-9


# ---------------------------------------------------------------- continuity
def test_gapless_capture_is_edf_c():
    p, info, _ = write("d.edf", index=np.arange(3000))
    assert info["continuity"] == "EDF+C"
    assert parse_header(p.read_bytes())["reserved"] == "EDF+C"


def digital(p: pathlib.Path, si: int = 0) -> np.ndarray:
    """Every stored integer of one signal, records end to end."""
    b = p.read_bytes()
    h = parse_header(b)
    rec = 2 * sum(h["nsr"])
    pre = 2 * sum(h["nsr"][:si])
    return np.concatenate([
        np.frombuffer(b, "<i2", count=h["nsr"][si],
                      offset=h["header_bytes"] + r * rec + pre)
        for r in range(h["n_records"])]).astype(np.int64)


def annotations(p: pathlib.Path) -> list[tuple]:
    """(onset, duration, text) for every annotation with text."""
    b = p.read_bytes()
    h = parse_header(b)
    rec = 2 * sum(h["nsr"])
    out = []
    for r in range(h["n_records"]):
        o = h["header_bytes"] + r * rec + 2 * sum(h["nsr"][:-1])
        for tal in b[o:o + 2 * h["nsr"][-1]].split(b"\x00"):
            parts = tal.split(b"\x14")
            head = parts[0].split(b"\x15")
            for text in parts[1:]:
                if text:
                    out.append((float(head[0]),
                                float(head[1]) if len(head) > 1 else None,
                                text.decode()))
    return out


def test_a_hole_is_filled_and_every_sample_keeps_its_time():
    """A sample written straight on from the ones before a hole sits early by
    the length of the hole, so the hole is filled and labeled instead."""
    idx = np.concatenate([np.arange(1500), np.arange(1500) + 2500])
    p, info, sigs = write("e.edf", index=idx)
    h = parse_header(p.read_bytes())
    assert info["continuity"] == h["reserved"] == "EDF+C"
    assert info["gaps"] == 1 and info["samples_filled"] == 1000, info

    # The hole is 1000 places at the lowest number the file can store, and no
    # recorded sample is that number.
    d = digital(p)
    hole = np.zeros(len(d), bool)
    hole[1500:2500] = True
    assert np.all(d[hole] == -32768) and not np.any(d[~hole] == -32768)

    # Every recorded sample sits at its own index, in the file and in time.
    pmin, pmax = h["phys_min"][0], h["phys_max"][0]
    uv = (d - (-32768)) * (pmax - pmin) / 65535 + pmin
    kept = idx < len(d)
    assert np.max(np.abs(uv[idx[kept]] - sigs["ch1"][kept])) <= (pmax - pmin) / 65535

    skip = [(t, dur) for t, dur, text in annotations(p) if text == edf.SKIP_LABEL]
    assert len(skip) == 1, skip
    assert abs(skip[0][0] - 1500 / FS) < 1e-5 and abs(skip[0][1] - 1000 / FS) < 1e-5, skip


def test_a_sample_index_that_steps_back_is_refused():
    """A break whose length is not known has no place on a timeline."""
    idx = np.concatenate([np.arange(1500), np.arange(1400, 2900)])
    try:
        write("e2.edf", index=idx)
    except ValueError as e:
        assert "steps back" in str(e), e
        return
    raise AssertionError("a timeline that steps back was written")


def test_zero_for_holes_is_one_setting_and_costs_range():
    """A build that prefers zero changes `GAP_FILL_UV`, and the range then
    stretches to hold it, as the module description says."""
    idx = np.concatenate([np.arange(1500), np.arange(1500) + 2100])
    before = edf.GAP_FILL_UV
    edf.GAP_FILL_UV = 0.0
    try:
        p, info, _ = write("e3.edf", index=idx)
    finally:
        edf.GAP_FILL_UV = before
    h = parse_header(p.read_bytes())
    d = digital(p)
    pmin, pmax = h["phys_min"][0], h["phys_max"][0]
    step = (pmax - pmin) / 65535
    uv = (d - (-32768)) * step + pmin
    assert pmin <= 0.0 and np.all(np.abs(uv[1500:2100]) <= step / 2), (pmin, uv[1500])
    assert step > 1500 / 65535, "the range did not stretch to hold zero"


def test_no_recorded_sample_can_be_the_hole_value():
    """The margin below each signal's lowest sample keeps the lowest number
    for holes, at the extremes too: a flat signal and a signal at the very
    bottom of its own range."""
    for name, x in (("flat", np.full(1500, 42.0)),
                    ("ramp", np.linspace(-5000.0, 5000.0, 1500))):
        idx = np.concatenate([np.arange(750), np.arange(750) + 760])
        p = tmp() / f"e4_{name}.edf"
        edf.write_edf(p, signals={"ch1": x}, fs=FS, start=START, index=idx)
        d = digital(p)
        assert np.sum(d == -32768) == 10, (name, np.sum(d == -32768))


def test_the_stated_range_holds_every_sample_however_far_from_zero():
    """The header states each bound in eight characters, which cannot hold
    every number exactly. A bound rounded toward the samples can land inside
    the margin, and the lowest sample is then written as the bound, or, with
    a hole in the file, as the hole. The bounds are moved outward instead."""
    rng = np.random.default_rng(7)
    idx = np.concatenate([np.arange(750), np.arange(750) + 760])
    for name, x in (("quiet far below zero", -150000.4 + rng.normal(0, 0.3, 1500)),
                    ("quiet at five digits", -12346.04 + rng.normal(0, 0.3, 1500)),
                    ("quiet far above zero", 123456.7 + rng.normal(0, 0.3, 1500))):
        p = tmp() / "e5.edf"
        edf.write_edf(p, signals={"ch1": x}, fs=FS, start=START, index=idx)
        h = parse_header(p.read_bytes())
        pmin, pmax = h["phys_min"][0], h["phys_max"][0]
        assert pmin <= x.min() and pmax >= x.max(), (name, pmin, x.min(), x.max(), pmax)
        d = digital(p)
        assert np.sum(d == -32768) == 10, (name, np.sum(d == -32768))
        uv = (d - (-32768)) * (pmax - pmin) / 65535 + pmin
        kept = idx < len(d)
        assert np.max(np.abs(uv[idx[kept]] - x[kept])) <= (pmax - pmin) / 65535, name


# ---------------------------------------------------------------- data
def test_samples_round_trip_within_quantization():
    """16-bit over the signal's own range: error must be under one LSB."""
    p, info, sigs = write("f.edf")
    b = p.read_bytes()
    h = parse_header(b)
    nsr, ann = h["nsr"][0], h["nsr"][2]
    rec = 2 * (nsr * 2 + ann)
    got = []
    for r in range(h["n_records"]):
        o = h["header_bytes"] + r * rec
        got.append(np.frombuffer(b, "<i2", count=nsr, offset=o))
    d = np.concatenate(got)

    pmin, pmax = h["phys_min"][0], h["phys_max"][0]
    uv = (d.astype(float) - (-32768)) * (pmax - pmin) / (32767 - (-32768)) + pmin
    want = sigs["ch1"][:info["samples_written"]]
    lsb = (pmax - pmin) / 65535
    assert np.max(np.abs(uv - want)) <= lsb, np.max(np.abs(uv - want))


def test_export_is_declared_lossy():
    _, info, _ = write("g.edf")
    assert info["lossy"] is True and info["bits"] == 16


def test_trailing_partial_record_is_dropped_not_padded():
    """Zero-padding a short record would write samples that never existed."""
    _, info, _ = write("h.edf", n=1200)     # 2 records of 499 = 998
    assert info["n_records"] == 2
    assert info["samples_written"] == 998
    assert info["samples_dropped"] == 202


def test_events_land_in_the_right_record():
    ev = [dict(t=0.0, label="closed"), dict(t=2.5, label="open")]
    p, info, _ = write("i.edf", events=ev)
    assert info["n_annotations"] == 2
    b = p.read_bytes()
    assert b"closed" in b and b"open" in b


def test_too_short_capture_is_refused():
    try:
        write("j.edf", n=100)
    except ValueError as e:
        assert "shorter than one" in str(e)
        return
    raise AssertionError("a sub-record capture was exported")


def test_mismatched_signal_lengths_are_refused():
    try:
        edf.write_edf(tmp() / "k.edf",
                      signals={"a": np.zeros(1000), "b": np.zeros(999)},
                      fs=FS, start=START)
    except ValueError as e:
        assert "expected" in str(e)
        return
    raise AssertionError("ragged signals were exported")


def test_flat_signal_does_not_produce_a_degenerate_range():
    p, _, _ = edf.write_edf(tmp() / "l.edf",
                            signals={"ch1": np.full(1500, 42.0)},
                            fs=FS, start=START), None, None
    h = parse_header((tmp() / "l.edf").read_bytes())
    assert h["phys_max"][0] > h["phys_min"][0]


def test_mne_can_read_what_we_wrote():
    """The only test that matters for interoperability: software we did not
    write, reading our bytes. Skipped if MNE is absent."""
    try:
        import mne
    except ImportError:
        print("       (skipped: mne not installed)")
        return
    import warnings
    warnings.filterwarnings("ignore")

    p, info, sigs = write("m.edf", n=3000, events=[dict(t=1.0, label="closed")])
    r = mne.io.read_raw_edf(p, preload=True, verbose="ERROR")
    assert r.ch_names == ["ch1", "ch2"], r.ch_names
    assert abs(r.info["sfreq"] - FS) < 0.01, r.info["sfreq"]
    assert r.n_times == info["samples_written"]

    ann = mne.read_annotations(p)
    assert "closed" in set(map(str, ann.description))

    got = r.get_data(picks=[0])[0] * 1e6            # MNE returns volts
    want = sigs["ch1"][:info["samples_written"]]
    lsb = (want.max() - want.min()) * 1.02 / 65535
    assert np.max(np.abs(got - want)) <= lsb * 1.05


def test_quantization_step_scales_with_the_signals_own_range():
    """The 16-bit step is range/65535, where `range` is what the signal actually
    spans in this file -- not the DC value. A quiet 40 uV excursion quantizes
    finely; a recording whose electrode offset drifts 8 mV does not, and there
    the step (~0.13 uV) approaches the instrument's 0.29 uV RMS noise floor.
    Anyone working at the noise floor must read the .npz."""
    p, info, _ = write("n.edf")                     # ch1 spans ~40 uV
    h = parse_header(p.read_bytes())
    narrow = (h["phys_max"][0] - h["phys_min"][0]) / 65535
    assert narrow < 0.002, narrow

    t = np.arange(3000) / FS
    wide = {"ch1": 8000.0 * t / t[-1] + 20 * np.sin(2 * np.pi * 10 * t)}
    q = tmp() / "n2.edf"
    edf.write_edf(q, signals=wide, fs=FS, start=START)
    hw = parse_header(q.read_bytes())
    step = (hw["phys_max"][0] - hw["phys_min"][0]) / 65535
    assert 0.10 < step < 0.16, step                 # ~0.12 uV, as on the bench
    assert step > narrow * 50
    assert info["lossy"] is True


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
    if _TMP and _TMP.exists():
        shutil.rmtree(_TMP)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("failed:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
