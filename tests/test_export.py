"""The six export formats, read back the way a stranger would read them.

    python3 tests/test_export.py

An export exists to be opened by software we did not write, so these tests
parse the bytes rather than trust the summary: fixed-width ASCII and packed
integers for EDF+ and BDF+, the csv module for the text formats, np.load and
h5py for the two archives. Every capture here is fabricated, so the right
answer is known before the file is written.

No pytest, no network, no device, and nothing written inside a captures
directory.
"""
from __future__ import annotations
import csv
import json
import pathlib
import shutil
import sys
import tempfile
import traceback

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from intomind import analysis as A          # noqa: E402
from intomind import edf                    # noqa: E402
from intomind import export as X            # noqa: E402
from intomind import provenance as prov     # noqa: E402

FS = 500.0
VREF_UV = 2_400_000
GAIN = 12
ADC_BITS = 24
#: 0.024 µV per count, which is what decides whether a format carried the
#: record: a file that lands every sample inside half of this returns the
#: counts that were recorded.
LSB_UV = (2.0 * VREF_UV) / (GAIN * (1 << ADC_BITS))

_TMP: pathlib.Path | None = None


# ---------------------------------------------------------------- fixtures
def tmp() -> pathlib.Path:
    """One temporary tree per run, with the captures directory redirected
    into it before anything can write to the real one."""
    global _TMP
    if _TMP is None:
        _TMP = pathlib.Path(tempfile.mkdtemp(prefix="intomind_export_"))
        prov.use_captures_dir(_TMP / "captures")
        for name in ("CAPTURES", "SESSIONS", "LEGACY", "MONTAGE_BACKFILL"):
            p = pathlib.Path(getattr(prov, name)).resolve()
            assert str(p).startswith(str(_TMP)), f"prov.{name} escaped: {p}"
        assert pathlib.Path(A.CAPTURES).resolve() == (_TMP / "captures").resolve()
    return _TMP


#: Redirected HERE, at import, and not on the first call that happens to want
#: a directory. A helper that writes a capture reaches for `A.CAPTURES` before
#: any test asks for a path, and with the redirect deferred the first test to
#: run put its fixture in the installed default instead of the temporary tree.
tmp()


def work(name: str) -> pathlib.Path:
    """A directory for one test's output files."""
    d = tmp() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def counts_for(n: int, nch: int) -> np.ndarray:
    """A DC-coupled channel as the instrument actually sees one: millivolts of
    electrode offset drifting under a small rhythm. The excursion is what
    decides how coarsely a 16-bit container quantizes, so a fixture that sat
    at zero would make EDF+ look lossless."""
    t = np.arange(n) / FS
    span = np.linspace(0.0, 8000.0, n) if n > 1 else np.zeros(n)
    rows = [(1500.0 * (c + 1) + span + 20.0 * np.sin(2 * np.pi * 10 * t))
            for c in range(nch)]
    return np.round(np.vstack(rows) / LSB_UV).astype(np.int64)


def fake_capture(label: str = "t_export", n: int = 1500, nch: int = 2, *,
                 ticks: bool = True, index: np.ndarray | None = None,
                 meta_extra: dict | None = None):
    """A Capture of the real class, so `uv` is the method it really is."""
    idx = np.arange(n, dtype=np.int64) if index is None else np.asarray(index)
    meta = dict(label=label, n=n, fs_effective=FS, vref_uv=VREF_UV, gain=GAIN,
                adc_bits=ADC_BITS, device_name="fake instrument",
                board_revision="unknown",
                montage=dict(electrode="fixture electrode"))
    meta.update(meta_extra or {})
    return A.Capture(label, counts_for(n, nch), idx, idx / FS, meta, [],
                     idx.copy() if ticks else None)


def on_disk(label: str, cap, *, checksum: bool = True):
    """Write a capture where `analysis.load` will find it, marks included."""
    n = cap.counts.shape[1]
    np.savez_compressed(
        pathlib.Path(A.CAPTURES) / f"{label}.npz",
        counts=cap.counts, index=cap.index, device_ticks=cap.device_ticks,
        host_time=cap.host_time,
        leadoff=np.full(n, 0x12, np.uint8),
        gap_before=np.zeros(n, bool))
    meta = dict(cap.meta, label=label)
    (pathlib.Path(A.CAPTURES) / f"{label}.meta.json").write_text(json.dumps(meta))
    if checksum:
        prov.write_checksums(label)
    return label


def parse_header(b: bytes) -> dict:
    """The fixed-width header, at the offsets the standard puts it. Decoded as
    latin-1 because a BDF+ version field starts with a byte that is not
    ASCII."""
    f = lambda o, n: b[o:o + n].decode("latin-1").strip()    # noqa: E731
    ns = int(f(252, 4))
    h = dict(version=b[:8], patient=f(8, 80), recording=f(88, 80),
             startdate=f(168, 8), starttime=f(176, 8),
             header_bytes=int(f(184, 8)), reserved=f(192, 44),
             n_records=int(f(236, 8)), duration=float(f(244, 8)), ns=ns)
    o = 256
    h["labels"] = [f(o + i * 16, 16) for i in range(ns)]
    o += 16 * ns
    h["transducer"] = [f(o + i * 80, 80) for i in range(ns)]
    o += 80 * ns + 8 * ns
    h["phys_min"] = [float(f(o + i * 8, 8)) for i in range(ns)]
    o += 8 * ns
    h["phys_max"] = [float(f(o + i * 8, 8)) for i in range(ns)]
    o += 8 * ns
    h["dig_min"] = [int(f(o + i * 8, 8)) for i in range(ns)]
    o += 8 * ns
    h["dig_max"] = [int(f(o + i * 8, 8)) for i in range(ns)]
    o += 8 * ns + 80 * ns
    h["nsr"] = [int(f(o + i * 8, 8)) for i in range(ns)]
    return h


def read_signals(path: pathlib.Path, sample_bytes: int):
    """Every data signal in µV, rebuilt the way the standard says to."""
    b = path.read_bytes()
    h = parse_header(b)
    rec = sum(n * sample_bytes for n in h["nsr"])
    out = {}
    for si, lbl in enumerate(h["labels"]):
        if lbl.endswith("Annotations"):
            continue
        pre = sum(h["nsr"][j] for j in range(si)) * sample_bytes
        got = []
        for r in range(h["n_records"]):
            o = h["header_bytes"] + r * rec + pre
            raw = b[o:o + h["nsr"][si] * sample_bytes]
            if sample_bytes == 2:
                d = np.frombuffer(raw, "<i2").astype(np.int64)
            else:
                u = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int64)
                d = u[:, 0] | (u[:, 1] << 8) | (u[:, 2] << 16)
                d = np.where(d >= (1 << 23), d - (1 << 24), d)
            got.append(d)
        d = np.concatenate(got).astype(np.float64)
        pmin, pmax = h["phys_min"][si], h["phys_max"][si]
        dmin, dmax = h["dig_min"][si], h["dig_max"][si]
        out[lbl] = (d - dmin) * (pmax - pmin) / (dmax - dmin) + pmin
    return h, out


def rows_of(path: pathlib.Path, delimiter: str):
    with path.open(newline="") as f:
        return list(csv.reader(f, delimiter=delimiter))


def have_h5py() -> bool:
    try:
        import h5py                                          # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------- the shape
def test_every_format_writes_a_file_and_the_same_summary():
    cap = fake_capture()
    d = work("shape")
    for fmt in X.FORMATS:
        if fmt == "mat" and not have_h5py():
            print("       (mat skipped: h5py not installed)")
            continue
        fn = getattr(X, f"export_{fmt}")
        s = fn("t_export", d / f"a{X.SUFFIX[fmt]}", load=lambda _l: cap)
        for key in ("label", "path", "format", "channels", "samples",
                    "sample_rate_hz"):
            assert key in s, (fmt, key)
        assert s["label"] == "t_export" and s["format"] == fmt
        assert s["channels"] == 2
        assert abs(s["sample_rate_hz"] - FS) < 0.01, s
        p = pathlib.Path(s["path"])
        assert p.exists() and p.stat().st_size > 0, fmt
        assert 0 < s["samples"] <= 1500, (fmt, s["samples"])


def test_export_dispatches_by_name_to_every_format():
    """The Command Center offers `FORMATS` without knowing what any of them
    is, so every name in it has to resolve."""
    cap = fake_capture()
    d = work("dispatch")
    assert len(X.FORMATS) == 6 and len(set(X.FORMATS)) == 6
    for fmt in X.FORMATS:
        if fmt == "mat" and not have_h5py():
            continue
        s = X.export("t_export", fmt, d, load=lambda _l: cap)
        assert s["format"] == fmt
        assert pathlib.Path(s["path"]).name == f"t_export{X.SUFFIX[fmt]}"


def test_an_unknown_format_names_the_ones_that_exist():
    try:
        X.export("t_export", "wav", load=lambda _l: fake_capture())
    except ValueError as e:
        assert "wav" in str(e) and "npz" in str(e) and "bdf" in str(e)
        return
    raise AssertionError("an unknown format was accepted")


def test_the_file_lands_exactly_where_it_was_asked_to():
    """`savez` adds `.npz` to a name that does not end in it, which would
    leave the summary naming a file that is not there. A directory means the
    default name inside it, and a path whose parent is missing gets one."""
    cap = fake_capture(n=600)
    d = work("paths")
    s = X.export_npz("t_export", d / "weird.bin", load=lambda _l: cap)
    assert s["path"] == str(d / "weird.bin") and (d / "weird.bin").exists()
    assert not (d / "weird.bin.npz").exists()

    s = X.export_tsv("t_export", d, load=lambda _l: cap)
    assert pathlib.Path(s["path"]) == d / "t_export.tsv"

    s = X.export_csv("t_export", d / "deep" / "a.csv", load=lambda _l: cap)
    assert pathlib.Path(s["path"]).exists()


def test_exports_never_land_in_the_captures_directory():
    """captures/ is checksummed and frozen. A derived file in it would be
    hashed by nothing and would make the directory listing lie."""
    cap = fake_capture()
    s = X.export_npz("t_export", load=lambda _l: cap)
    p = pathlib.Path(s["path"]).resolve()
    caps = pathlib.Path(prov.CAPTURES).resolve()
    assert caps not in p.parents, p
    assert p.parent == caps.parent / "exports", p


# ---------------------------------------------------------------- npz
def test_npz_returns_every_sample_to_the_count_it_was_recorded_at():
    """float32 microvolts, and the test is whether the counts come back."""
    cap = fake_capture()
    s = X.export_npz("t_export", work("npz") / "a.npz", load=lambda _l: cap)
    with np.load(s["path"]) as z:
        for c in range(cap.nch):
            uv = z[f"ch{c + 1}_uv"]
            assert uv.dtype == np.float32, uv.dtype
            back = np.round(uv.astype(np.float64) / LSB_UV).astype(np.int64)
            assert np.array_equal(back, cap.counts[c]), c
    assert s["counts_recovered"] is True and s["lossy"] is False
    assert s["max_error_uv"] < 0.5 * LSB_UV, s["max_error_uv"]


def test_npz_carries_both_clocks_and_the_manifest():
    cap = fake_capture()
    s = X.export_npz("t_export", work("npz2") / "a.npz", load=lambda _l: cap)
    with np.load(s["path"]) as z:
        assert np.array_equal(z["index"], cap.index)
        assert np.array_equal(z["device_ticks"], cap.device_ticks)
        assert np.allclose(z["host_time"], cap.host_time)
        meta = json.loads(str(z["manifest_json"]))
    assert meta == cap.meta
    assert meta["adc_bits"] == ADC_BITS and meta["gain"] == GAIN


def test_npz_takes_the_marks_from_the_capture_archive():
    """`Capture` does not carry lead-off or the gap mark today, and both are
    recorded per sample. They come out of the capture's own .npz, and only
    when they line up sample for sample with what was loaded."""
    label = on_disk("t_marks", fake_capture("t_marks", n=600))
    s = X.export_npz(label, work("marks") / "a.npz")
    with np.load(s["path"]) as z:
        assert np.array_equal(z["leadoff"], np.full(600, 0x12, np.uint8))
        assert z["gap_before"].dtype == bool and not z["gap_before"].any()
    assert s["omitted"] == [], s["omitted"]


def test_npz_names_a_per_sample_array_it_could_not_write():
    """A missing array is reported, never left out in silence."""
    cap = fake_capture("t_noticks", ticks=False)
    s = X.export_npz("t_noticks", work("noticks") / "a.npz",
                     load=lambda _l: cap)
    with np.load(s["path"]) as z:
        assert "device_ticks" not in z.files
    said = " ".join(s["omitted"])
    assert "device_ticks" in said and "leadoff" in said and "gap_before" in said
    assert "index" not in said and "host_time" not in said


# ---------------------------------------------------------------- bdf
def test_bdf_is_a_biosemi_file_with_24_bit_samples():
    cap = fake_capture()
    s = X.export_bdf("t_export", work("bdf") / "a.bdf", load=lambda _l: cap)
    h, _ = read_signals(pathlib.Path(s["path"]), 3)
    assert h["version"] == b"\xffBIOSEMI", h["version"]
    assert h["reserved"] == "BDF+C", h["reserved"]
    assert h["labels"] == ["ch1", "ch2", "BDF Annotations"], h["labels"]
    assert h["dig_min"] == [-8388608] * 3, h["dig_min"]
    assert h["dig_max"] == [8388607] * 3, h["dig_max"]
    assert s["bits"] == 24 and s["family"] == "BDF+"
    # The file is exactly the header plus three-byte samples, which is the
    # only thing that tells a reader it is not an EDF.
    size = pathlib.Path(s["path"]).stat().st_size
    assert size == h["header_bytes"] + h["n_records"] * 3 * sum(h["nsr"])


def test_bdf_returns_every_sample_to_the_count_it_was_recorded_at():
    """The reason to offer BDF+ at all: 24 bits is as wide as the record."""
    cap = fake_capture()
    s = X.export_bdf("t_export", work("bdf2") / "a.bdf", load=lambda _l: cap)
    _, sigs = read_signals(pathlib.Path(s["path"]), 3)
    for c in range(cap.nch):
        back = np.round(sigs[f"ch{c + 1}"] / LSB_UV).astype(np.int64)
        assert np.array_equal(back, cap.counts[c][:s["samples"]]), c
    assert s["counts_recovered"] is True and s["lossy"] is False
    assert s["max_error_uv"] < 0.5 * LSB_UV, s["max_error_uv"]


def test_edf_is_sixteen_bits_and_says_so():
    """Same capture, narrower container: the samples do not survive it, and
    the summary says lossy before anyone measures anything."""
    cap = fake_capture()
    s = X.export_edf("t_export", work("edf") / "a.edf", load=lambda _l: cap)
    h, sigs = read_signals(pathlib.Path(s["path"]), 2)
    assert h["version"] == b"0       " and h["reserved"] == "EDF+C"
    assert h["labels"][-1] == "EDF Annotations"
    assert h["dig_min"] == [-32768] * 3 and h["dig_max"] == [32767] * 3
    assert s["lossy"] is True and s["bits"] == 16
    # On a capture with 8 mV of electrode drift the 16-bit step is coarser
    # than one ADC count, and this is the measured number, not an assertion
    # about the format.
    assert s["max_error_uv"] > 0.5 * LSB_UV, s["max_error_uv"]
    assert s["counts_recovered"] is False
    back = np.round(sigs["ch1"] / LSB_UV).astype(np.int64)
    assert not np.array_equal(back, cap.counts[0][:s["samples"]])


def test_a_synthetic_capture_says_so_in_every_format():
    """A generated signal is labeled as generated wherever it goes: in the
    labels of the columns that hold it, and first in an EDF+ or BDF+
    recording field, which software that never reads the manifest reads."""
    cap = fake_capture(meta_extra=dict(synthetic=True, signal_source="synthetic"))
    d = work("synthetic")
    for fmt, write in (("edf", X.export_edf), ("bdf", X.export_bdf)):
        s = write("t_export", d / f"a.{fmt}", load=lambda _l: cap)
        h = parse_header(pathlib.Path(s["path"]).read_bytes())
        assert h["recording"].startswith("SYNTHETIC, not measured: fake instrument"), h["recording"]
        assert h["labels"][:2] == ["ch1 synthetic", "ch2 synthetic"], h["labels"]
    for fmt, write, sep in (("csv", X.export_csv, ","), ("tsv", X.export_tsv, "\t")):
        s = write("t_export", d / f"a.{fmt}", load=lambda _l: cap)
        head = pathlib.Path(s["path"]).read_text().splitlines()[0].split(sep)
        assert head[2:] == ["ch1_synthetic_uv", "ch2_synthetic_uv"], head
    s = X.export_npz("t_export", d / "a.npz", load=lambda _l: cap)
    with np.load(s["path"]) as z:
        assert {"ch1_synthetic_uv", "ch2_synthetic_uv"} <= set(z.files) and "ch1_uv" not in z.files
    if have_h5py():
        import h5py
        s = X.export_mat("t_export", d / "a.mat", load=lambda _l: cap)
        with h5py.File(s["path"], "r") as f:
            assert "synthetic_uv" in f and "uv" not in f
    # A measured capture keeps its plain labels.
    plain = fake_capture()
    s = X.export_edf("t_export", d / "b.edf", load=lambda _l: plain)
    h = parse_header(pathlib.Path(s["path"]).read_bytes())
    assert "SYNTHETIC" not in h["recording"] and h["labels"][:2] == ["ch1", "ch2"]
    s = X.export_csv("t_export", d / "b.csv", load=lambda _l: plain)
    assert pathlib.Path(s["path"]).read_text().splitlines()[0].split(",")[2:] == ["ch1_uv", "ch2_uv"]


def test_a_gap_is_discontinuous_in_both_containers():
    """Concatenating across a hole would invent the samples that were lost."""
    idx = np.concatenate([np.arange(750), np.arange(750) + 755])
    cap = fake_capture("t_gap", n=1500, index=idx)
    d = work("gap")
    assert X.export_edf("t_gap", d / "a.edf",
                        load=lambda _l: cap)["continuity"] == "EDF+D"
    assert X.export_bdf("t_gap", d / "a.bdf",
                        load=lambda _l: cap)["continuity"] == "BDF+D"


def test_the_last_partial_record_is_reported_not_padded():
    cap = fake_capture("t_tail", n=1200)
    s = X.export_bdf("t_tail", work("tail") / "a.bdf", load=lambda _l: cap)
    assert s["samples"] == 1000 and s["samples_dropped"] == 200, s


# ---------------------------------------------------------------- csv, tsv
def test_csv_and_tsv_differ_only_by_their_delimiter():
    cap = fake_capture()
    d = work("delim")
    a = X.export_csv("t_export", d / "a.csv", load=lambda _l: cap)
    b = X.export_tsv("t_export", d / "a.tsv", load=lambda _l: cap)
    rows_a = rows_of(pathlib.Path(a["path"]), ",")
    rows_b = rows_of(pathlib.Path(b["path"]), "\t")
    assert rows_a == rows_b, "the two text formats are not the same file"
    assert len(rows_a) == 1501
    assert "\t" not in pathlib.Path(a["path"]).read_text()
    assert "," not in pathlib.Path(b["path"]).read_text()
    assert a["delimiter"] == "," and b["delimiter"] == "\t"


def test_csv_columns_are_both_clocks_then_one_per_channel():
    cap = fake_capture(nch=4)
    s = X.export_csv("t_export", work("cols") / "a.csv", load=lambda _l: cap)
    rows = rows_of(pathlib.Path(s["path"]), ",")
    assert rows[0] == ["host_time_s", "device_time",
                       "ch1_uv", "ch2_uv", "ch3_uv", "ch4_uv"], rows[0]
    for i in (0, 37, 1499):
        r = rows[i + 1]
        assert abs(float(r[0]) - cap.host_time[i]) < 1e-6
        assert int(r[1]) == int(cap.device_ticks[i])
        for c in range(4):
            assert abs(float(r[2 + c]) - cap.uv(c)[i]) < 0.5 * LSB_UV


def test_the_text_formats_carry_more_resolution_than_one_count():
    """Four decimals is right for this LSB and wrong for a device with a
    smaller one, so the digits are derived from the capture."""
    cap = fake_capture()
    s = X.export_csv("t_export", work("res") / "a.csv", load=lambda _l: cap)
    assert s["resolution_uv"] < LSB_UV, (s["resolution_uv"], LSB_UV)
    assert s["counts_recovered"] is True and s["lossy"] is False
    rows = rows_of(pathlib.Path(s["path"]), ",")
    back = np.array([round(float(r[2]) / LSB_UV) for r in rows[1:]], np.int64)
    assert np.array_equal(back, cap.counts[0])
    assert X._decimals(LSB_UV) == 4
    assert X._decimals(LSB_UV / 1000.0) == 7


def test_csv_export_does_not_reference_an_unimported_name():
    """`edf.export_csv` read a module-level CAPTURES that was never imported
    into that file, so the first call it ever got raised NameError. It had no
    test, which is how it survived."""
    label = on_disk("t_defect1", fake_capture("t_defect1", n=600))
    s = edf.export_csv(label)
    p = pathlib.Path(s["path"])
    assert p.exists() and p.stat().st_size > 0
    assert len(rows_of(p, ",")) == 601


def test_csv_export_calls_uv_as_the_method_it_is():
    """`Capture.uv` takes a channel index. The old writer asked for
    `len(cap.uv[0])` and iterated `cap.uv` as if it were the channels, which
    cannot work on a method and was never run."""
    label = on_disk("t_defect2", fake_capture("t_defect2", n=600, nch=4))
    s = edf.export_csv(label)
    rows = rows_of(pathlib.Path(s["path"]), ",")
    assert len(rows[0]) == 6, rows[0]          # two clocks, four channels
    cap = A.load(label)
    assert cap.nch == 4
    for c in range(4):
        assert abs(float(rows[1][2 + c]) - cap.uv(c)[0]) < 0.5 * LSB_UV
        assert abs(float(rows[600][2 + c]) - cap.uv(c)[599]) < 0.5 * LSB_UV


# ---------------------------------------------------------------- the gate
def test_a_capture_that_fails_verification_is_refused_by_every_format():
    """An export is regenerable and the capture it came from is not, so a
    recording that reports damage is not copied into six files that no longer
    report it."""
    label = on_disk("t_bad", fake_capture("t_bad", n=600))
    with (pathlib.Path(A.CAPTURES) / f"{label}.npz").open("ab") as f:
        f.write(b"\x00")                       # the bytes are not the bytes
    ok, problems = prov.verify(label)
    assert not ok and "CHECKSUM MISMATCH" in problems[0]

    cap = fake_capture(label, n=600)           # a loader that would succeed
    d = work("refused")
    for fmt in X.FORMATS:
        out = d / f"x{X.SUFFIX[fmt]}"
        try:
            X.export(label, fmt, out, load=lambda _l: cap)
        except ValueError as e:
            assert "refusing to export" in str(e), (fmt, e)
            assert "CHECKSUM MISMATCH" in str(e), (fmt, e)
            assert not out.exists(), f"{fmt} wrote a file anyway"
            continue
        raise AssertionError(f"{fmt} exported a capture that failed verify")


def test_the_gate_lets_a_sound_capture_through():
    """A gate never seen to pass is as useless as one never seen to fail."""
    label = on_disk("t_good", fake_capture("t_good", n=600))
    ok, _ = prov.verify(label)
    assert ok
    s = X.export_csv(label, work("good") / "a.csv")
    assert pathlib.Path(s["path"]).exists()


# ---------------------------------------------------------------- mat
def test_mat_says_what_to_install_when_h5py_is_absent():
    """A host without the extra gets a line to type, not an import
    traceback."""
    cap = fake_capture()
    saved = sys.modules.get("h5py", "absent")
    sys.modules["h5py"] = None                 # import h5py now raises
    try:
        X.export_mat("t_export", work("noh5") / "a.mat", load=lambda _l: cap)
    except RuntimeError as e:
        assert "h5py" in str(e) and "pip install intomind[mat]" in str(e)
        assert not (work("noh5") / "a.mat").exists()
        return
    finally:
        if saved == "absent":
            sys.modules.pop("h5py", None)
        else:
            sys.modules["h5py"] = saved
    raise AssertionError("export_mat ran without h5py")


def test_mat_is_an_hdf5_file_that_matlab_will_open():
    if not have_h5py():
        print("       (skipped: h5py not installed)")
        return
    import h5py
    cap = fake_capture()
    s = X.export_mat("t_export", work("mat") / "a.mat", load=lambda _l: cap)
    p = pathlib.Path(s["path"])

    head = p.read_bytes()[:128]
    assert head.startswith(b"MATLAB 7.3 MAT-file"), head[:32]
    assert head[124:128] == b"\x00\x02IM", head[124:128]

    with h5py.File(p, "r") as f:
        uv = f["uv"]
        assert uv.shape == (2, 1500), uv.shape   # MATLAB reads 1500 by 2
        assert uv.attrs["MATLAB_class"] == b"double"
        assert f["manifest_json"].attrs["MATLAB_class"] == b"char"
        meta = json.loads(np.asarray(f["manifest_json"]).astype(np.uint16)
                          .tobytes().decode("utf-16-le"))
        for c in range(cap.nch):
            back = np.round(uv[c] / LSB_UV).astype(np.int64)
            assert np.array_equal(back, cap.counts[c]), c
        assert np.array_equal(f["index"][0], cap.index)
    assert meta == cap.meta
    assert s["lossy"] is False and s["stored_dtype"] == "float64"


# ---------------------------------------------------------------- outsiders
def test_mne_reads_both_containers():
    """The only test that matters for interoperability: software we did not
    write, reading our bytes. Skipped if MNE is absent."""
    try:
        import mne
    except ImportError:
        print("       (skipped: mne not installed)")
        return
    import warnings
    warnings.filterwarnings("ignore")

    cap = fake_capture()
    d = work("mne")
    edf_s = X.export_edf("t_export", d / "a.edf", load=lambda _l: cap)
    bdf_s = X.export_bdf("t_export", d / "a.bdf", load=lambda _l: cap)
    for s, reader in ((edf_s, mne.io.read_raw_edf), (bdf_s, mne.io.read_raw_bdf)):
        r = reader(s["path"], preload=True, verbose="ERROR")
        assert r.ch_names == ["ch1", "ch2"], (s["format"], r.ch_names)
        assert abs(r.info["sfreq"] - FS) < 0.01
        assert r.n_times == s["samples"]
        got = r.get_data(picks=[0])[0] * 1e6            # MNE returns volts
        err = float(np.max(np.abs(got - cap.uv(0)[:s["samples"]])))
        # What the summary promised is what a foreign reader measures.
        assert err <= s["max_error_uv"] * 1.01 + 1e-9, (s["format"], err)


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
