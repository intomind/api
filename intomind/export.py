"""Six ways out of a capture, and one shape for all of them.

    export_<format>(label, path=None, *, load=None) -> dict
    export(label, format, path=None, *, load=None) -> dict

Every summary carries the same six keys, so a caller can offer a list of
formats without knowing what any of them is:

    label, path, format, channels, samples, sample_rate_hz

What the formats do not share is what they cost, and each summary says that in
numbers measured on the file that was just written rather than in a sentence
about the format in general:

| `npz` | float32 microvolts per channel, the clocks, the marks, the manifest |
| `edf` | EDF+, 16-bit. The narrowest container here and the most widely read |
| `bdf` | BDF+, 24-bit. The same container, wide enough for a 24-bit record |
| `csv` | plain text, one row per sample |
| `tsv` | the same rows, tab separated |
| `mat` | MATLAB v7.3, which is HDF5. Needs h5py |

Three rules hold across all six.

A capture that does not pass `provenance.verify` is not exported at all. An
export can always be made again and the capture it came from cannot, so a
recording that reports damage is not copied into six new files that no longer
report it.

Exports are written outside the captures directory. A capture is checksummed
and frozen, an export is regenerable, and a directory holding both invites a
ledger that counts the wrong files.

Nothing per-sample is dropped in silence. An array the capture does not carry,
or one whose length does not line up with the samples that were loaded, is
named in the summary's `omitted` list with the reason.

`load=` stands in for `analysis.load`, which is how the tests run with no
captures on disk.
"""
from __future__ import annotations
import csv as _csv
import datetime as _dt
import json
import pathlib
import sys

import numpy as np

from . import edf as _edf
from . import provenance as prov

#: Every format this module writes, in the order a caller may as well offer
#: them: the record first, then the two clinical containers, then text, then
#: the one that needs an extra install.
FORMATS = ("npz", "edf", "bdf", "csv", "tsv", "mat")

SUFFIX = dict(npz=".npz", edf=".edf", bdf=".bdf",
              csv=".csv", tsv=".tsv", mat=".mat")

#: What to say when the optional dependency is absent. A bare ImportError
#: tells the caller a module name, and this tells them the line to type.
MAT_MISSING = ("MATLAB v7.3 is an HDF5 file and needs h5py, which is not "
               "installed. Install it with: pip install intomind[mat]")

#: h5py requires a userblock of at least 512 bytes and a power of two. The
#: MATLAB header is 128 bytes and sits at the front of it.
MAT_USERBLOCK = 512

_MAT_CLASS = {"float64": b"double", "float32": b"single",
              "int64": b"int64", "int32": b"int32", "int16": b"int16",
              "uint64": b"uint64", "uint32": b"uint32", "uint16": b"uint16",
              "uint8": b"uint8"}


# ---------------------------------------------------------------- the gate
def _analysis():
    """`analysis` imported late, because it imports scipy and scipy is an
    extra. Writing a capture out as text must not require the DSP stack.
    """
    from . import analysis
    return analysis


def _prepare(label: str, path, load, fmt: str):
    """Refuse, load, and decide where the file goes, in that order.

    The integrity check comes first so that it does not depend on a loader
    succeeding, and so the refusal says the same thing whichever format asked.
    """
    ok, problems = prov.verify(label)
    if not ok:
        raise ValueError(f"refusing to export {label}: " + "; ".join(problems))
    cap = (load or _analysis().load)(label)
    return cap, _path_for(label, path, fmt), list(problems)


def _path_for(label: str, path, fmt: str) -> pathlib.Path:
    """The caller's path, a name inside the caller's directory, or the
    default `exports/` beside the captures directory."""
    if path is None:
        out = pathlib.Path(prov.CAPTURES).parent / "exports"
        out.mkdir(parents=True, exist_ok=True)
        return out / f"{prov.check_label(label)}{SUFFIX[fmt]}"
    p = pathlib.Path(path)
    if p.is_dir():
        return p / f"{prov.check_label(label)}{SUFFIX[fmt]}"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _summary(label, path, fmt, cap, *, samples, problems, **extra) -> dict:
    """The six keys every format answers with, plus that format's own."""
    out = dict(label=label, path=str(path), format=fmt,
               channels=int(cap.nch), samples=int(samples),
               sample_rate_hz=float(cap.fs or 0.0),
               provenance_problems=list(problems))
    out.update(extra)
    return out


# ---------------------------------------------------------------- the data
def _n(cap) -> int:
    """How many samples there are, counted on the counts, which are the
    record. Everything else is checked against this."""
    return int(np.asarray(cap.counts).shape[1])


def _synthetic(cap) -> bool:
    """Whether the device generated this capture's signal (protocol 1.3)."""
    return bool((cap.meta or {}).get("synthetic"))


def _column(cap, c: int) -> str:
    """A channel's label: `ch1_uv`, or `ch1_synthetic_uv` when the device
    generated the signal, so the columns that hold a generated signal say
    so wherever the file goes without its manifest."""
    return f"ch{c + 1}_synthetic_uv" if _synthetic(cap) else f"ch{c + 1}_uv"


def _uv(cap) -> list:
    """Microvolts per channel.

    `Capture.uv` is a method taking a channel index. Treating it as a sequence
    of channels is one of the two defects this module was written to replace,
    and both have a test.
    """
    return [np.asarray(cap.uv(c), dtype=np.float64) for c in range(cap.nch)]


def _aligned(name: str, a, n: int):
    """A per-sample array, or the reason it cannot be written.

    Length is not a formality here. A capture repaired on load has fewer
    samples than its file does, so pairing a full-length array against the
    loaded samples would put every value against the wrong sample.
    """
    if a is None:
        return None, f"{name}: the capture does not carry it"
    arr = np.asarray(a)
    if arr.ndim == 0 or arr.shape[0] != n:
        got = 0 if arr.ndim == 0 else arr.shape[0]
        return None, (f"{name}: {got} values for {n} samples, so it cannot be "
                      f"lined up with them")
    return arr, None


def _archive(label: str) -> dict:
    """The per-sample marks from the capture's own archive.

    `analysis.Capture` holds the counts, the sample index, the two clocks and
    the manifest. The lead-off byte and the gap mark are recorded per sample
    and are in the capture's `.npz`, so they are read from there when the
    loaded object does not expose them. Read only, and only these two.
    """
    try:
        p = prov.capture_path(label, ".npz")
        if not p.exists():
            return {}
        with np.load(p) as z:
            return {k: z[k] for k in ("leadoff", "gap_before") if k in z.files}
    except (OSError, ValueError, KeyError):
        return {}


def _per_sample(cap, label: str, n: int):
    """Every per-sample array except the microvolts, with what was left out."""
    found, omitted = {}, []
    archive = None
    for name in ("index", "device_ticks", "host_time", "leadoff", "gap_before"):
        a = getattr(cap, name, None)
        if a is None and name in ("leadoff", "gap_before"):
            if archive is None:
                archive = _archive(label)
            a = archive.get(name)
        arr, why = _aligned(name, a, n)
        if arr is None:
            omitted.append(why)
        else:
            found[name] = arr
    return found, omitted


def _decimals(lsb_uv: float) -> int:
    """Decimal places that carry one ADC count with room to spare.

    Text holds as many digits as it is given, so the number of digits is
    derived from the capture's own LSB rather than fixed at four. The step
    written is at most a hundredth of one count, which is what lets a value
    read back land on the count it came from.
    """
    if not lsb_uv or lsb_uv <= 0 or not np.isfinite(lsb_uv):
        return 6
    return int(min(12, max(3, np.ceil(-np.log10(float(lsb_uv) / 100.0)))))


# ---------------------------------------------------------------- npz
def export_npz(label: str, path=None, *, load=None) -> dict:
    """The capture as a numpy archive: microvolts, both clocks, the marks,
    and the manifest as JSON text.

    Microvolts are float32, which holds a 24-bit count to well under half an
    LSB anywhere but the last decade of full scale. The summary says what the
    conversion actually cost on this capture rather than assuming it was free.
    """
    cap, out, problems = _prepare(label, path, load, "npz")
    n = _n(cap)
    uv = _uv(cap)
    arrays = {_column(cap, c): u.astype(np.float32) for c, u in enumerate(uv)}
    found, omitted = _per_sample(cap, label, n)
    arrays.update(found)
    arrays["manifest_json"] = np.array(
        json.dumps(cap.meta, sort_keys=True, default=str))

    # Written through an open file rather than by name, because savez appends
    # its own suffix to a name that does not end in .npz and the summary would
    # then name a file that is not there.
    with open(out, "wb") as f:
        np.savez_compressed(f, **arrays)

    lsb = float(cap.lsb_uv)
    err = 0.0
    for c, u in enumerate(uv):
        stored = arrays[_column(cap, c)].astype(np.float64)
        err = max(err, float(np.max(np.abs(stored - u))) if n else 0.0)
    recovered = bool(err < 0.5 * lsb)
    return _summary(label, out, "npz", cap, samples=n, problems=problems,
                    lossy=not recovered, counts_recovered=recovered,
                    max_error_uv=err, lsb_uv=lsb, omitted=omitted,
                    arrays=sorted(arrays))


# ---------------------------------------------------------------- edf, bdf
def _export_edf_family(label: str, path, load, bits: int) -> dict:
    """EDF+ and BDF+ are one container at two widths, so they are one path
    here and one writer in `edf.py`."""
    fmt = "edf" if bits == 16 else "bdf"
    cap, out, problems = _prepare(label, path, load, fmt)
    fs = float(cap.fs or 0.0)
    if fs <= 0:
        raise ValueError(
            f"refusing to export {label} as {fmt.upper()}: the capture states "
            f"no sample rate, and this container places every sample by rate")

    info = _edf.export_capture(label, out, load=lambda _label: cap, bits=bits)
    lsb = float(cap.lsb_uv)
    recovered = bool(info["max_error_uv"] < 0.5 * lsb)
    return _summary(
        label, info["path"], fmt, cap,
        samples=info["samples_written"], problems=problems,
        # EDF+ carries 16 bits where a 24-bit record has 24, so it is reported
        # as lossy whatever one quiet capture happens to fit into. BDF+ is as
        # wide as the record, so there the claim is the measurement.
        lossy=(True if bits == 16 else not recovered),
        counts_recovered=recovered, max_error_uv=info["max_error_uv"],
        quantum_uv=info["quantum_uv"], lsb_uv=lsb, bits=info["bits"],
        family=info["family"], continuity=info["continuity"],
        samples_dropped=info["samples_dropped"],
        n_annotations=info["n_annotations"])


def export_edf(label: str, path=None, *, load=None) -> dict:
    """EDF+: 16-bit, and the format the rest of the world reads.

    Sixteen bits over the signal's own excursion, against a record of 24. The
    summary reports it lossy for that reason, and `max_error_uv` is what the
    quantization cost this file. A capture whose samples must survive
    unchanged goes out as `bdf` or `npz`.

    Samples are written in whole data records, so a tail shorter than one
    record is not in the file and `samples_dropped` says how many that was.
    """
    return _export_edf_family(label, path, load, 16)


def export_bdf(label: str, path=None, *, load=None) -> dict:
    """BDF+: the same container at 24 bits, which is why it is offered.

    At 24 bits the step is finer than one ADC count on any capture that does
    not span the whole converter, so the values come back as they went in.
    That is measured per file: `counts_recovered` is true when the largest
    reconstruction error is under half of this capture's own LSB, and `lossy`
    follows it rather than the other way around.

    As with EDF+, whole data records only, and `samples_dropped` says what the
    last partial record would have held.
    """
    return _export_edf_family(label, path, load, 24)


# ---------------------------------------------------------------- csv, tsv
def _export_delimited(label: str, path, load, fmt: str, delimiter: str) -> dict:
    """One writer, two delimiters. The columns, the rows, the digits and the
    order are the same file either way.
    """
    cap, out, problems = _prepare(label, path, load, fmt)
    n, nch = _n(cap), cap.nch
    uv = _uv(cap)
    lsb = float(cap.lsb_uv)
    dec = _decimals(lsb)
    found, omitted = _per_sample(cap, label, n)
    host = found.get("host_time")
    ticks = found.get("device_ticks")

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f, delimiter=delimiter, lineterminator="\n")
        w.writerow(["host_time_s", "device_time"]
                   + [_column(cap, c) for c in range(nch)])
        for i in range(n):
            w.writerow([f"{float(host[i]):.6f}" if host is not None else "",
                        int(ticks[i]) if ticks is not None else "",
                        *(f"{float(uv[c][i]):.{dec}f}" for c in range(nch))])

    resolution = 10.0 ** -dec
    recovered = bool(resolution / 2.0 < 0.5 * lsb)
    # Only the two clocks can go missing from a row. The marks are not columns
    # here, so they are not reported as omitted from a file that never has them.
    omitted = [o for o in omitted
               if o.startswith("host_time") or o.startswith("device_ticks")]
    return _summary(label, out, fmt, cap, samples=n, problems=problems,
                    delimiter=delimiter, decimals=dec,
                    resolution_uv=resolution, lsb_uv=lsb,
                    counts_recovered=recovered, lossy=not recovered,
                    omitted=omitted)


def export_csv(label: str, path=None, *, load=None) -> dict:
    """Comma separated text: one header row, then one row per sample of host
    time, device time, and a microvolt column per channel.

    Both clocks are columns because both are what makes a gap visible in a
    text file. The rows are the samples that were loaded, in order, and
    nothing is inserted to bridge a hole in them.
    """
    return _export_delimited(label, path, load, "csv", ",")


def export_tsv(label: str, path=None, *, load=None) -> dict:
    """The same file as `export_csv`, separated by tabs.

    Same columns, same rows, same digits, one writer. Tabs are here because a
    microvolt column is a decimal number and a comma is a decimal separator in
    much of the world.
    """
    return _export_delimited(label, path, load, "tsv", "\t")


# ---------------------------------------------------------------- mat
def _h5py():
    """h5py, or a sentence saying what to install."""
    try:
        import h5py
    except ImportError:
        raise RuntimeError(MAT_MISSING) from None
    return h5py


def _mat_header(when: str) -> bytes:
    """The 128 bytes MATLAB looks at before the HDF5 file begins: 116 bytes of
    description, 8 of subsystem offset, then the version and the byte order
    the description is written in."""
    text = (f"MATLAB 7.3 MAT-file, Platform: {sys.platform}, "
            f"Created on: {when} HDF5 schema 1.00 .")
    return (text.encode("ascii", "replace")[:116].ljust(116, b" ")
            + b"\x00" * 8 + b"\x00\x02" + b"IM")


def _mat_write(f, name: str, a) -> None:
    """One MATLAB variable.

    MATLAB varies the first dimension fastest and numpy varies the last, so an
    array written here as (channels, samples) opens in MATLAB as samples by
    channels, which is the shape its own code expects. Vectors are written as
    one row and open as one column.
    """
    a = np.asarray(a)
    decode = None
    if a.dtype == bool:
        stored, cls, decode = a.astype(np.uint8), b"logical", 1
    else:
        cls = _MAT_CLASS.get(a.dtype.name)
        stored = a if cls else a.astype(np.float64)
        cls = cls or b"double"
    ds = f.create_dataset(name, data=stored)
    ds.attrs["MATLAB_class"] = np.bytes_(cls)
    if decode is not None:
        ds.attrs["MATLAB_int_decode"] = np.int64(decode)


def _mat_string(f, name: str, s: str) -> None:
    """A MATLAB char array, which is UTF-16 code units and an attribute."""
    a = np.frombuffer(s.encode("utf-16-le"), dtype=np.uint16).reshape(1, -1)
    ds = f.create_dataset(name, data=a)
    ds.attrs["MATLAB_class"] = np.bytes_(b"char")
    ds.attrs["MATLAB_int_decode"] = np.int64(2)


def export_mat(label: str, path=None, *, load=None) -> dict:
    """MATLAB v7.3, which is an HDF5 file with MATLAB's attributes on it.

    Microvolts go in as one float64 matrix that opens as samples by channels,
    with the clocks and the marks beside it as columns and the manifest as a
    char array. The values are written at the width they are already computed
    at, so nothing is rounded on the way out.

    v7.3 rather than v7: v7 needs scipy and a MAT-file writer, and it holds
    2 GB. v7.3 is HDF5, which h5py writes directly and which has no such
    limit. h5py is an extra, and a host without it gets a sentence saying so.
    """
    cap, out, problems = _prepare(label, path, load, "mat")
    h5py = _h5py()
    n = _n(cap)
    uv = _uv(cap)
    found, omitted = _per_sample(cap, label, n)

    matrix = "synthetic_uv" if _synthetic(cap) else "uv"
    with h5py.File(out, "w", userblock_size=MAT_USERBLOCK) as f:
        _mat_write(f, matrix, np.vstack(uv) if uv else np.zeros((0, n)))
        for name, arr in found.items():
            _mat_write(f, name, np.asarray(arr).reshape(1, -1))
        _mat_string(f, "manifest_json",
                    json.dumps(cap.meta, sort_keys=True, default=str))
    when = _dt.datetime.now(_dt.timezone.utc).strftime("%a %b %d %H:%M:%S %Y")
    with open(out, "r+b") as f:
        f.write(_mat_header(when))

    lsb = float(cap.lsb_uv)
    return _summary(label, out, "mat", cap, samples=n, problems=problems,
                    lossy=False, counts_recovered=True, max_error_uv=0.0,
                    lsb_uv=lsb, stored_dtype="float64", omitted=omitted,
                    variables=sorted([matrix, "manifest_json", *found]))


# ---------------------------------------------------------------- dispatch
_BY_NAME = dict(npz=export_npz, edf=export_edf, bdf=export_bdf,
                csv=export_csv, tsv=export_tsv, mat=export_mat)


def export(label: str, format: str, path=None, *, load=None) -> dict:
    """Export by name, so a caller can offer `FORMATS` without knowing any of
    them. An unknown name is refused with the list of the ones that exist."""
    name = str(format).strip().lower().lstrip(".")
    fn = _BY_NAME.get(name)
    if fn is None:
        raise ValueError(f"unknown export format {format!r}. "
                         f"Known formats: {', '.join(FORMATS)}")
    return fn(label, path, load=load)


__all__ = ["FORMATS", "export", "export_npz", "export_edf", "export_bdf",
           "export_csv", "export_tsv", "export_mat"]
