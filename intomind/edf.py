"""EDF+ and BDF+ export. Interoperability, and a narrower view of the record.

EDF+ stores 16-bit integers. Our captures are 24-bit ADC counts. **Export is
therefore lossy by construction** — it is a way to open a recording in EDFbrowser,
MNE, or a colleague's software, not a substitute for `captures/`. The `.npz` is
the record. Every exported file says so in its recording identifier.

The same writer also produces BDF+, which is the same container with 24-bit
samples and the BioSemi version identifier. At 24 bits the step is narrower
than one ADC count on any capture that does not span the whole converter, so a
BDF+ carries the sample values a reader needs and the reconstruction error is
measured and returned rather than asserted. Pass `bits=24`.

How lossy depends on the signal's own excursion in that file, not on its DC
value: the step is `range / 65535`. A quiet 40 µV trace quantizes finely. A real
recording whose electrode offset drifts several mV does not — on
`berger_on_g12_20260710-040310` the step is **0.13 µV**, against an instrument
noise floor of 0.29 µV RMS. Noise-floor work reads the `.npz`.

Two things this writer refuses to fake:

* **Time.** Every sample is written at the time it was recorded, counted from
  the sample index. Where samples were lost, the file holds one placeholder
  for each lost sample, so everything after a hole stays where it belongs,
  and each hole carries a `BAD_ACQ_SKIP` annotation over exactly its span.
  That is the label MNE gives a hole in its own files: it filters each side
  of one separately, rejects an epoch that overlaps one, and returns NaN there
  when asked (`get_data(reject_by_annotation="NaN")`). Writing the samples
  after a hole straight on from the ones before it would put them early by
  the length of the hole, and so would a record onset that skips the hole,
  because a reader such as MNE takes the records end to end. A sample index
  that steps back is a break whose length is not known, and the file is
  refused rather than given a timeline the recording does not have.

  A placeholder is the lowest number the file can store, not zero and not
  NaN. The format holds whole numbers only, so NaN cannot be written. Each
  signal's range is fitted to that signal with a margin below its lowest
  sample, so no recorded sample can be the lowest number, and using it costs
  nothing. Zero costs precision: most recordings sit thousands of microvolts
  from zero, and fitting zero into a signal's range stretches it, so every
  recorded sample in a 16-bit file is rounded more coarsely. `GAP_FILL_UV`
  holds the choice, for a build that prefers zero and accepts that cost.

* **The sample rate.** `fs_effective` is not an integer (498.80 SPS on this
  bench). EDF requires an integer number of samples per data record, so the
  record *duration* carries the remainder. Rounding the rate to 500 would
  misplace a 10 Hz rhythm by 0.024 Hz.

Reference: Kemp & Olivan 2003, "European data format 'plus' (EDF+)".
"""
from __future__ import annotations
import datetime as _dt
import math
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import numpy as np


def _device_of(meta: dict) -> str:
    """What made this capture, in the words the capture itself carries.

    The name the device advertises where one was recorded, its id where one was
    not, and "unknown device" where neither is. Never a name written into this
    library: it talks to whatever is connected.
    """
    return (meta.get("device_name")
            or meta.get("device_id")
            or "unknown device")

ANNOT_LABEL = "EDF Annotations"
DIG_MIN, DIG_MAX = -32768, 32767

#: What differs between the two widths of one container, and nothing else
#: does. EDF+ writes 2-byte samples under version "0", BDF+ writes 3-byte
#: samples under the BioSemi identifier, and each names its own annotation
#: signal. A reader picks the sample width from the file, so the two cannot be
#: mixed in one file and are never guessed here.
WIDTHS = {
    16: dict(dig_min=-32768, dig_max=32767, sample_bytes=2,
             version=b"0       ", annot="EDF Annotations", family="EDF+"),
    24: dict(dig_min=-8388608, dig_max=8388607, sample_bytes=3,
             version=b"\xffBIOSEMI", annot="BDF Annotations", family="BDF+"),
}
SUFFIX = {16: ".edf", 24: ".bdf"}

#: The annotation over each hole where samples were lost. It is the label MNE
#: gives an acquisition gap in its own files, so MNE skips the span when it
#: filters and rejects an epoch that overlaps it.
SKIP_LABEL = "BAD_ACQ_SKIP"

#: What a lost sample is written as. None writes the lowest number the file
#: can store, which no recorded sample can be and which costs no precision. A
#: number of microvolts writes that instead, and the range of every signal is
#: stretched to hold it wherever there is a hole. 0.0 writes zeros, and a
#: recording that sits thousands of microvolts from zero is then rounded more
#: coarsely in a 16-bit file, every recorded sample included.
GAP_FILL_UV: float | None = None


def _fmt(value, width: int) -> bytes:
    """ASCII, left-justified, exactly `width` bytes. EDF is fixed-width."""
    s = str(value)
    if len(s) > width:
        s = s[:width]
    b = s.ljust(width).encode("ascii", "replace")
    if len(b) != width:                       # non-ascii collapsed to '?'
        b = b[:width].ljust(width, b" ")
    return b


def _num(x: float, width: int = 8) -> bytes:
    """Fit a number into `width` ASCII chars without losing more than we must."""
    s = f"{x:.{width}g}"
    if len(s) > width:
        for prec in range(width - 2, 0, -1):
            s = f"{x:.{prec}f}"
            if len(s) <= width:
                break
    if len(s) > width:
        s = s[:width]
    return _fmt(s, width)


def _written(x: float) -> float:
    """A physical bound as the eight-character header field will hold it.

    Anything `_num` cannot round-trip through `float` is left alone, which
    leaves the pre-existing behavior in place for a value no EDF field could
    have carried in the first place.
    """
    try:
        return float(_num(x).decode().strip())
    except ValueError:
        return float(x)


def _bound(x: float, *, below: bool) -> float:
    """A physical bound as the eight-character header field will hold it,
    moved away from the samples where the field cannot hold it exactly: at
    or below `x` for the lower bound, at or above for the upper.

    `_written` gives the nearest value the field holds, and past six digits
    the value truncated toward zero, and either can land inside the margin
    `_scale` leaves. The extreme sample is then outside the range the header
    states, and a reader gets it back clipped to the bound.
    """
    if not math.isfinite(x):
        return _written(x)
    d = Decimal(repr(float(x)))
    rounding = ROUND_FLOOR if below else ROUND_CEILING
    for places in range(7, -1, -1):
        s = f"{d.quantize(Decimal(1).scaleb(-places), rounding=rounding):f}"
        if len(s) <= 8 and _written(float(s)) == float(s):
            return float(s)
    return _written(x)


def _tal(onset: float, duration: float | None, texts: list[str]) -> bytes:
    """One time-stamped annotation list."""
    out = f"{'+' if onset >= 0 else '-'}{abs(onset):.6f}".encode()
    if duration is not None:
        out += b"\x15" + f"{duration:.6f}".encode()
    for t in texts or [""]:
        out += b"\x14" + t.encode("utf-8", "replace")
    return out + b"\x14\x00"


def _scale(x: np.ndarray) -> tuple[float, float]:
    """Physical range for one signal, symmetric and never degenerate."""
    lo, hi = float(np.min(x)), float(np.max(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    pad = 0.01 * (hi - lo)
    return lo - pad, hi + pad


def _digitize(x: np.ndarray, pmin: float, pmax: float,
              dmin: int = DIG_MIN, dmax: int = DIG_MAX) -> np.ndarray:
    g = (dmax - dmin) / (pmax - pmin)
    d = np.round((x - pmin) * g + dmin)
    return np.clip(d, dmin, dmax).astype("<i4")


def _undigitize(d: np.ndarray, pmin: float, pmax: float,
                dmin: int, dmax: int) -> np.ndarray:
    """What a reader gets back, by the formula in the standard. Used to
    measure the error this file actually carries rather than describe it."""
    return (d.astype(np.float64) - dmin) * (pmax - pmin) / (dmax - dmin) + pmin


def _pack(d: np.ndarray, sample_bytes: int) -> bytes:
    """Integer samples as the container stores them, little-endian.

    Two bytes go out as int16. Three bytes are the low three of a little-endian
    int32, which is the same two's complement the 24-bit format defines, so a
    negative sample keeps its sign without a second encoding step.
    """
    if sample_bytes == 2:
        return d.astype("<i2").tobytes()
    return d.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3].tobytes()


def _transducer_of(meta: dict) -> str:
    """The electrode, in the words the capture itself carries.

    Read from the recorded montage, never written into this library: what was
    on the head is a fact about one session on one bench, and a sentence typed
    here would ride into the header of every export from every device. An
    unrecorded electrode leaves the field empty, which is what EDF+ means by
    unspecified.
    """
    m = meta.get("montage") or {}
    return str(m.get("electrode") or "")


def contiguous(index: np.ndarray) -> bool:
    """Whether `index` climbs by exactly one sample with no break, start
    to end."""
    return index.size < 2 or bool(np.all(np.diff(index) == 1))


def _prefilter_of(meta: dict) -> str:
    """What the device did to the signal, for the header's prefilter field:
    the chain in words, from the capture's own manifest. A capture made
    before the manifest carried a chain said only raw or filtered, and that
    word is kept as it was."""
    proc = meta.get("processing")
    if isinstance(proc, dict) and proc.get("description"):
        return str(proc["description"])[:80]
    sig = meta.get("signal")
    if sig in (None, "", "raw"):
        return "none (natural signal)"
    return str(sig)[:80]


def write_edf(path, *, signals: dict[str, np.ndarray], fs: float,
              start: _dt.datetime, index: np.ndarray | None = None,
              events: list[dict] | None = None, patient: str = "X X X X",
              recording: str = "", prefilter: str = "",
              transducer: str = "", bits: int = 16) -> dict:
    """Write one EDF+ or BDF+ file. Returns a summary of what was written.

    `signals` maps label -> µV samples, all the same length. `index` is the
    capture's sample index, and it places every sample in time: where it
    skips ahead, the file holds a placeholder for each lost sample and a
    `BAD_ACQ_SKIP` annotation over the hole, and where it steps back, the
    file is refused. `samples_filled` and `gaps` in the summary count them.

    `bits` selects the width of the container: 16 for EDF+, 24 for BDF+. Both
    quantize the µV handed in onto an integer grid, so both report the step
    they used (`quantum_uv`) and the largest error that step actually produced
    in this file (`max_error_uv`), measured by reconstructing every sample the
    way a reader will. `lossy` says that the file holds integers on that grid
    rather than the µV themselves, which is true of both widths. Whether the
    grid was fine enough to carry a given recording is a question about that
    recording, and `export.py` answers it against the capture's own LSB.
    """
    if bits not in WIDTHS:
        raise ValueError(f"bits must be 16 (EDF+) or 24 (BDF+), not {bits}")
    w = WIDTHS[bits]
    dmin, dmax, sample_bytes = w["dig_min"], w["dig_max"], w["sample_bytes"]

    labels = list(signals)
    if not labels:
        raise ValueError("no signals")
    n = len(signals[labels[0]])
    for lbl, x in signals.items():
        if len(x) != n:
            raise ValueError(f"signal {lbl} has {len(x)} samples, expected {n}")
    if n == 0:
        raise ValueError("no samples")

    idx = np.arange(n) if index is None else np.asarray(index, dtype=np.int64)
    if len(idx) != n:
        raise ValueError(f"index has {len(idx)} entries for {n} samples")
    steps = np.diff(idx)
    back = np.nonzero(steps < 1)[0]
    if back.size:
        raise ValueError(
            f"refusing to write {w['family']}: the sample index steps back "
            f"after sample {int(back[0])}, a break whose length is not known, "
            f"so the samples after it have no place on this file's timeline")

    # Where each sample belongs, counted in samples from the first. A hole is
    # a run of places no sample holds, and each place gets a placeholder, so
    # every sample after a hole keeps its time in any reader.
    pos = idx - idx[0]
    span = int(pos[-1]) + 1

    # Integer samples per record; the remainder lives in the record duration.
    nsr = max(1, int(round(fs)))
    dur = nsr / float(fs)
    n_rec = span // nsr
    if n_rec == 0:
        raise ValueError(f"capture shorter than one {dur:.3f} s record")
    used = n_rec * nsr
    kept = pos < used
    holes = [(int(pos[i]) + 1, min(int(steps[i]) - 1, used - int(pos[i]) - 1))
             for i in np.nonzero(steps > 1)[0] if pos[i] + 1 < used]

    # Annotations, bucketed into the record whose time span contains them:
    # the events handed in, and one over each hole.
    ann: list[list[tuple]] = [[] for _ in range(n_rec)]
    n_events = 0
    for e in events or []:
        t = e.get("t")
        if t is None:
            continue
        r = min(n_rec - 1, max(0, int(t / dur)))
        ann[r].append((float(t), None, str(e.get("label", ""))))
        n_events += 1
    for first, length in holes:
        ann[first // nsr].append((first / fs, length / fs, SKIP_LABEL))
    # widest annotation record decides the annotation signal's size
    annot_bytes = []
    for r in range(n_rec):
        blob = _tal(r * nsr / float(fs), None, [""])
        for t, d, label in ann[r]:
            blob += _tal(t, d, [label])
        annot_bytes.append(blob)
    widest = max(len(b) for b in annot_bytes)
    ann_nsr = max(1, (widest + sample_bytes - 1) // sample_bytes + 1)

    ns = len(labels) + 1
    hdr_bytes = 256 * (ns + 1)
    # The records follow on from each other in time, holes included, so every
    # file this writer makes is the continuous kind.
    reserved = w["family"][:3] + "+C"

    # The physical bounds AS THE HEADER WILL HOLD THEM. EDF gives each bound
    # eight ASCII characters, and a reader rebuilds every sample from what it
    # finds there, so digitizing against the full-precision bound puts the
    # writer on one grid and the reader on another. On a capture spanning
    # 8 mV that disagreement was 4e-4 µV, three times the quantization step of
    # the 24-bit container it was supposed to be measuring.
    #
    # Each range is fitted to the recorded samples alone. A placeholder in
    # microvolts (`GAP_FILL_UV`) has to fit too, and stretches the range.
    fill = GAP_FILL_UV if holes else None
    ranges = {}
    for lbl in labels:
        x = np.asarray(signals[lbl], dtype=np.float64)[kept]
        if fill is not None:
            x = np.append(x, fill)
        lo, hi = _scale(x)
        lo_w, hi_w = _bound(lo, below=True), _bound(hi, below=False)
        ranges[lbl] = (lo_w, hi_w) if hi_w > lo_w else (lo, hi)

    head = b"".join([
        w["version"],
        _fmt(patient, 80),
        _fmt(recording or f"Startdate {start:%d-%b-%Y}".upper(), 80),
        _fmt(f"{start:%d.%m.%y}", 8),
        _fmt(f"{start:%H.%M.%S}", 8),
        _fmt(hdr_bytes, 8),
        _fmt(reserved, 44),
        _fmt(n_rec, 8),
        _num(dur, 8),
        _fmt(ns, 4),
    ])
    assert len(head) == 256, len(head)

    def per_signal(fn) -> bytes:
        return b"".join(fn(lbl) for lbl in labels) + fn(None)

    sig = b"".join([
        per_signal(lambda l: _fmt(l if l else w["annot"], 16)),
        per_signal(lambda l: _fmt(transducer if l else "", 80)),
        per_signal(lambda l: _fmt("uV" if l else "", 8)),
        per_signal(lambda l: _num(ranges[l][0]) if l else _fmt(-1, 8)),
        per_signal(lambda l: _num(ranges[l][1]) if l else _fmt(1, 8)),
        per_signal(lambda l: _fmt(dmin, 8)),
        per_signal(lambda l: _fmt(dmax, 8)),
        per_signal(lambda l: _fmt(prefilter if l else "", 80)),
        per_signal(lambda l: _fmt(nsr if l else ann_nsr, 8)),
        per_signal(lambda l: _fmt("", 32)),
    ])
    assert len(sig) == 256 * ns, len(sig)

    # Every recorded sample goes where it belongs, and every place in a hole
    # holds the placeholder.
    dig, real = {}, {}
    for lbl in labels:
        pmin, pmax = ranges[lbl]
        real[lbl] = _digitize(np.asarray(signals[lbl])[kept], pmin, pmax,
                              dmin=dmin, dmax=dmax)
        if fill is None:
            hole = dmin
            # The margin below the lowest sample keeps every recorded sample
            # off the lowest number. One on it could not be told from a hole,
            # so that is checked here rather than assumed.
            if holes and np.any(real[lbl] == dmin):
                raise RuntimeError(f"signal {lbl}: a recorded sample landed on "
                                   f"the number that marks a hole")
        else:
            hole = int(_digitize(np.array([fill]), pmin, pmax,
                                 dmin=dmin, dmax=dmax)[0])
        dig[lbl] = np.full(used, hole, dtype="<i4")
        dig[lbl][pos[kept]] = real[lbl]

    # What this file cost, measured on the file itself: the step each signal
    # was put on, and the worst distance between a sample handed in and the
    # sample a reader gets back. Placeholders are not samples and are left out.
    quantum = 0.0
    max_err = 0.0
    for lbl in labels:
        pmin, pmax = ranges[lbl]
        quantum = max(quantum, (pmax - pmin) / (dmax - dmin))
        back = _undigitize(real[lbl], pmin, pmax, dmin, dmax)
        src = np.asarray(signals[lbl], dtype=np.float64)[kept]
        max_err = max(max_err, float(np.max(np.abs(back - src))))

    with open(path, "wb") as f:
        f.write(head); f.write(sig)
        for r in range(n_rec):
            a, b = r * nsr, (r + 1) * nsr
            for lbl in labels:
                f.write(_pack(dig[lbl][a:b], sample_bytes))
            f.write(annot_bytes[r].ljust(ann_nsr * sample_bytes, b"\x00"))

    written = int(kept.sum())
    return dict(path=str(path), n_records=n_rec, samples_per_record=nsr,
                record_duration_s=dur, continuity=reserved, fs=float(fs),
                samples_written=written, samples_dropped=n - written,
                samples_filled=sum(length for _, length in holes),
                gaps=len(holes), n_annotations=n_events,
                lossy=True, bits=bits, family=w["family"],
                quantum_uv=quantum, max_error_uv=max_err)


def export_capture(label: str, path=None, *, load=None, bits: int = 16) -> dict:
    """Export one capture. µV are recomputed from counts, as everywhere else.

    `bits=16` writes EDF+ and `bits=24` writes BDF+, into `.edf` and `.bdf`
    respectively when no path is given.

    A capture whose sample index steps back within its first second begins
    with samples from before the recording's count was reset. The file
    starts at the reset, and `samples_before_restart` says how many samples
    it left out.
    """
    from . import analysis as A
    from . import provenance as prov

    cap = (load or A.load)(label)
    ok, problems = prov.verify(label)
    if not ok:
        raise ValueError(f"refusing to export {label}: {'; '.join(problems)}")

    started = cap.meta.get("started_utc")
    try:
        start = _dt.datetime.strptime(started, "%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        start = _dt.datetime(2000, 1, 1)

    # A recording that armed before it reset the count holds, first, samples
    # still numbered from the stream before it, and then a count that starts
    # again at zero. Those samples come from before the recording began, and
    # how long the break after them lasted is not known, so the file starts
    # at the reset. A step back any later is a break inside the recording,
    # and write_edf refuses it.
    index = np.asarray(cap.index, dtype=np.int64)
    back = np.nonzero(np.diff(index) < 1)[0] + 1
    skip = int(back[-1]) if back.size and back[-1] < max(1, int(round(cap.fs or 0))) else 0

    events = []
    ev_path = prov.capture_path(label, ".events.json")
    if ev_path.exists():
        import json
        raw = json.loads(ev_path.read_text())
        t0 = float(cap.host_time[skip]) if len(cap.host_time) else 0.0
        for e in raw:
            if "host_time" in e:
                events.append(dict(t=float(e["host_time"]) - t0,
                                   label=str(e.get("cond", e.get("kind", "")))))

    # A generated signal's channels say so in their own labels too.
    label_of = (lambda c: f"ch{c+1} synthetic") if cap.meta.get("synthetic") else (lambda c: f"ch{c+1}")
    signals = {label_of(c): cap.uv(c)[skip:] for c in range(cap.nch)}
    rev = cap.meta.get("board_revision", "unknown")
    # Say "lossy" in the file itself when the container is narrower than the
    # resolution the capture was recorded at, and only then. The width of the
    # record is the manifest's answer, not this module's.
    rec_bits = cap.meta.get("adc_bits")
    note = " lossy" if rec_bits and bits < int(rec_bits) else ""
    # A generated signal says so first, where no truncation of the field can
    # take it away: the file is opened by software that never reads the
    # manifest.
    generated = "SYNTHETIC, not measured: " if cap.meta.get("synthetic") else ""
    if path is None:
        # Derived artifacts never live beside the record. captures/ is frozen,
        # checksummed, and committed; exports are regenerable and are not.
        out = prov.CAPTURES.parent / "exports"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{label}{SUFFIX.get(bits, '.edf')}"
    out = write_edf(
        path, signals=signals, fs=cap.fs, start=start, index=index[skip:],
        events=events, bits=bits,
        # The DEVICE that made it, from the capture's own manifest. This used
        # to be one hardcoded name, which put it in the header of every
        # export, including recordings from a different device.
        recording=f"{generated}{_device_of(cap.meta)} {rev} {bits}-bit{note}; "
                  f"raw record is {label}.npz",
        prefilter=_prefilter_of(cap.meta),
        # The electrode the capture recorded, or nothing. This used to be one
        # hardcoded electrode, for the same reason and with the same defect.
        transducer=_transducer_of(cap.meta))
    out["label"] = label
    out["samples_before_restart"] = skip
    out["board_revision"] = rev
    return out


# ---------------------------------------------------------------------------
def export_csv(label: str, path=None, *, load=None) -> dict:
    """Export a capture to CSV. Kept here for callers that already know the
    name, and implemented once in `export.py` with the other five formats.

    This function used to hold a second implementation, which had never run:
    it read a module-level `CAPTURES` that was never imported into this file,
    and it indexed `Capture.uv` as if it were a sequence when it is a method
    taking a channel. Both were untested, and one delegation is the way they
    stay fixed.
    """
    from . import export as _export
    return _export.export_csv(label, path, load=load)
