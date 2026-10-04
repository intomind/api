"""Canonical analysis for captures.

Every caller, application and CLI alike, comes through here, so there
is exactly one implementation of every statistic.

Conventions enforced here, learned the hard way:
  * Raw ADC counts are the record. uV is always derived from meta at
    analysis time, never trusted from the wire.
  * DC and AC are reported separately. A DC-coupled AFE sees tens of mV of
    electrode half-cell offset; folding that into an "RMS" measures the
    electrode, not the signal.
  * Spectra are computed only over contiguous sample runs. A dropped packet
    is a discontinuity, and a discontinuity splatters broadband power.
"""
from __future__ import annotations
import json, pathlib, itertools, warnings
from dataclasses import dataclass, field
import numpy as np
try:
    from scipy import signal, stats
except ImportError as _e:                    # pragma: no cover
    raise ImportError(
        "the analyses need scipy, which this package declares optional. "
        "Install it with: pip install 'intomind[analysis]'") from _e

from . import provenance as prov
from . import timebase

ANALYSIS_VERSION = 2   # bump on any change that can alter a reported number

# Where captures live. `provenance` owns the answer, including the
# environment variable and the redirect an application can make, and this
# name follows it. Inferring a path from where this file happens to sit was
# the old behavior and it guessed wrong the moment the library was
# installed rather than checked out.
CAPTURES = prov.CAPTURES
EEG_BANDS = dict(delta=(1, 4), theta=(4, 8), alpha=(8, 13),
                 beta=(13, 30), gamma=(30, 45))

# Nothing about the signal chain is assumed. Mains is 50 Hz in much of the
# world; band edges are exactly the thing an experimenter needs to sweep.
# Every one of these is overridable per call, from the API and the hub.
DEFAULTS = dict(hp=1.0, lp=45.0, notch=60.0, notch_q=30.0, trim=0.5)

# How far a stored clock skew may sit from 1.0 before `load` stops believing
# it. Deliberately looser than `client.MAX_SKEW_PPM`, which decides whether to
# FIT one: this decides whether to overrule what a past recording already
# wrote down, and that is the more conservative act of the two.
MAX_TRUSTED_SKEW_PPM = 5000.0


@dataclass
class Capture:
    label: str
    counts: np.ndarray          # (nch, n) int64 ADC counts
    index: np.ndarray
    host_time: np.ndarray
    meta: dict
    # What `load` had to correct before this was usable, in the words a
    # report can print. Empty on a capture that needed nothing, which is
    # every capture recorded by code that has the fixes in it.
    repairs: list = field(default_factory=list)
    device_ticks: np.ndarray | None = None
    #: Which electrodes the device believed were attached, one byte per
    #: sample. None for a capture recorded before this was stored.
    leadoff: np.ndarray | None = None
    #: Where the timeline broke, one flag per sample. A True marks the
    #: first sample after a break.
    gap_before: np.ndarray | None = None

    @property
    def fs(self) -> float:
        """Effective sample rate, guarded against a poisoned stored value.

        `fs_effective` in the manifest is normally correct, but a capture whose
        sample index reset mid-recording (pre-fix) stamped a bogus rate -- the
        berger_on_g12 capture stored 162 SPS for a true 494. The per-sample
        host clock is monotonic and independent of the index, so if the stored
        rate disagrees with it by more than 5%, trust the clock.
        """
        fs = self.meta.get("fs_effective") or 0.0
        ht = self.host_time
        if ht is not None and len(ht) > 1 and ht[-1] > ht[0]:
            fs_ht = (len(ht) - 1) / (ht[-1] - ht[0])
            if fs <= 0 or abs(fs - fs_ht) / fs_ht > 0.05:
                return fs_ht
        return fs

    @property
    def lsb_uv(self) -> float:
        """Microvolts per count, from what the device said it was.

        The resolution is the device's, read from the manifest. Assuming
        twenty four bits here would silently scale every number an analysis
        produces on any device that is not that wide.
        """
        bits = self.meta.get("adc_bits")
        if not bits:
            raise ValueError(
                f"{self.meta.get('label', 'this capture')} does not record adc_bits, "
                "so the microvolts it holds cannot be computed. Manifests from "
                "version 2 onward record it.")
        return (2.0 * self.meta["vref_uv"]) / (self.meta["gain"] * (1 << bits))

    @property
    def nch(self) -> int:
        return self.counts.shape[0]

    def uv(self, c: int) -> np.ndarray:
        return self.counts[c].astype(np.float64) * self.lsb_uv


def list_captures() -> list[dict]:
    """Every capture in the collection, newest first.

    THE LABEL IS CHECKED ON THE WAY IN, EVEN HERE. A capture directory is a
    directory: anything with write access can leave a `.meta.json` under a
    name of its choosing, so a label read off disk is the same untrusted thing
    as a label arriving in a request, and it gets the same rule. It used to be
    taken as given, and `verify` below then raised on it, which cost the reader
    the whole collection over one file.

    A rejected entry is skipped and counted rather than raised, so one file
    nobody can name still does not take the list down, and the count is warned
    about rather than swallowed.
    """
    out, rejected = [], []
    for m in sorted(CAPTURES.glob("*.meta.json"), key=lambda p: p.stat().st_mtime,
                    reverse=True):
        label = m.name[:-len(".meta.json")]
        try:
            prov.check_label(label)
        except ValueError as e:
            rejected.append(f"{m.name}: {e}")
            continue
        try:
            d = json.loads(m.read_text())
            if not isinstance(d, dict):
                raise ValueError("a manifest is an object")
        except Exception:
            continue
        d["label"] = label
        d["has_events"] = (CAPTURES / f"{d['label']}.events.json").exists()
        ok, problems = prov.verify(d["label"])
        first = problems[0] if problems else ""
        d["integrity"] = (
            "ok" if ok and not problems else
            "incomplete" if ok and first.startswith("provenance incomplete") else
            "unverified" if ok else
            "ABORTED" if d.get("aborted") else "FAILED")
        d["integrity_detail"] = first
        out.append(d)
    if rejected:
        warnings.warn(f"{len(rejected)} file(s) in {CAPTURES} are not named "
                      f"like captures and were skipped: " + "; ".join(rejected))
    return out


# REPAIRS MADE ON READ, NEVER ON DISK.
#
# A capture is a record. What was written is what was recorded, and rewriting
# it would destroy the evidence of the bug that made it -- so both of these
# correct the arrays in memory, on every load, and put a sentence in
# `Capture.repairs` saying what they did. Every one is a defect that has since
# been fixed at its source; these exist so the sessions already on disk are
# not lost to it.

def _dedupe_delivery(counts, index, host_time, ticks, marks=None):
    """Drop packets that were delivered more than once.

    Two BLE subscriptions on one peripheral both receive every notification,
    and both copies are well-formed, so the recorder stored each sample as
    many times as there were subscriptions. The signature is exact and cannot
    occur any other way: a device time repeated, carrying byte-identical
    counts. A real sample cannot share a tick with another real sample, and a
    coincidence cannot reproduce every channel.

    Both halves of that signature are required. If ticks repeat but the counts
    behind them differ, this is something else -- a counter that wrapped, a
    firmware fault -- and it is reported rather than silently halved.
    """
    if ticks is None or len(ticks) < 2:
        return None
    u, first, inv, cnt = np.unique(ticks, return_index=True,
                                   return_inverse=True, return_counts=True)
    if len(u) == len(ticks):
        return None
    n_extra = len(ticks) - len(u)
    if not np.array_equal(counts[:, first][:, inv], counts):
        return (None, f"{n_extra:,} samples repeat a device time with "
                      f"DIFFERENT counts behind it. Not duplicate delivery, "
                      f"and not something this can correct: the sample "
                      f"stream is unusable as a timeline.")
    keep = np.sort(first)                       # first arrival, original order
    marks = marks or {}
    if not np.all(np.diff(ticks[keep]) > 0):
        return (None, f"{n_extra:,} duplicated samples, and removing them "
                      f"still does not leave the device clock increasing.")
    dupes = round(len(ticks) / len(u), 3)
    return (dict(counts=counts[:, keep], index=index[keep],
                 host_time=host_time[keep], ticks=ticks[keep],
                 **{k: v[keep] for k, v in marks.items() if v is not None}),
            f"every packet was delivered {dupes:g} times over, so {n_extra:,} "
            f"of {len(ticks):,} samples were copies. Removed: {len(u):,} "
            f"samples remain. (Two BLE subscriptions were open on the "
            f"instrument at once.)")


def _repin_clock(meta, ticks):
    """Restamp samples whose `host_time` came from an impossible clock skew.

    `client._fit_clock` used to regress the device clock against the host's
    over whatever baseline the exchanges at connect happened to span -- under
    a second. Round-trip jitter across that is enough to fit any slope at all;
    one session got 1.6986, and every sample in it is stamped over 3304 s of
    an hour that lasted 1945.

    A slope that far from unity is not a clock, so the samples are restamped
    with skew pinned at 1.0 and the offset refitted from the same exchanges.
    Their device times are recoverable from the fit that is stored, which is
    what makes this exact rather than approximate.

    It needs the sync anchors. Without them there is no honest offset to fit,
    and this says the times are wrong instead of inventing a correction: an
    unusable timeline that says so is recoverable, and one quietly shifted by
    140 s is not.
    """
    c = meta.get("clock") or {}
    skew, off = c.get("skew"), c.get("offset")
    anchors, tick_hz = c.get("anchors") or [], meta.get("tick_hz") or 0
    if not skew or abs(skew - 1.0) * 1e6 <= MAX_TRUSTED_SKEW_PPM:
        return None
    ppm = abs(skew - 1.0) * 1e6
    if off is None or len(anchors) < 2 or not tick_hz or ticks is None:
        return (None, f"the clock was fitted at skew {skew:.4f} "
                      f"({ppm:,.0f} ppm), which no pair of oscillators can "
                      f"do, and this capture does not carry the exchanges "
                      f"needed to refit it. Every sample time is stretched by "
                      f"that factor. Counts are unaffected.")
    real = np.array([a[0] for a in anchors], float)
    mono = np.array([a[1] for a in anchors], float)
    dev = (mono - off) / skew              # undo the fit, recover the device
    frame = real[np.argmin(mono)] - mono.min()
    ht = np.asarray(ticks, float) / tick_hz + (mono.mean() - dev.mean()) + frame
    return (dict(host_time=ht),
            f"the clock was fitted at skew {skew:.4f} ({ppm:,.0f} ppm) over "
            f"a {c.get('span_s') or 0:.2f} s baseline, stretching every "
            f"sample time by that factor. Restamped from the device clock "
            f"with the skew pinned at 1.0; the samples now span "
            f"{(ht[-1] - ht[0]):,.0f} s.")


def load(label: str, verify: bool = True, repair: bool = True) -> Capture:
    """Load a capture, refusing silently-corrupted bytes.

    A capture recorded before P0 has no checksum file; it loads with a warning
    rather than an error, because most of this project's data predates it.
    A capture *with* a checksum that no longer matches is a hard error: an
    analysis must never quietly read bytes other than the ones recorded.

    `repair` corrects two recording defects in memory -- duplicate packet
    delivery and an impossible clock skew -- and lists what it did on
    `Capture.repairs`. Pass False to see exactly what is on disk.
    """
    if verify:
        ok, problems = prov.verify(label)
        if not ok:
            raise IOError(f"{label}: integrity check FAILED: {problems}")
        for p in problems:
            warnings.warn(f"{label}: {p}")
    z = np.load(prov.capture_path(label, ".npz"))
    meta = json.loads(prov.capture_path(label, ".meta.json").read_text())
    counts, index = z["counts"], z["index"]
    host_time = z["host_time"]
    ticks = z["device_ticks"] if "device_ticks" in z.files else None
    # Per sample marks: which electrodes the device believed were attached,
    # and where the timeline broke. They belong to the samples they were
    # recorded beside, so a repair that drops samples drops these with them.
    # Reading them separately and pairing them by position afterwards would
    # put every mark against the wrong sample.
    leadoff = z["leadoff"] if "leadoff" in z.files else None
    gap_before = z["gap_before"] if "gap_before" in z.files else None
    notes = []

    def apply(got):
        """Take a repair's answer: its sentence always, its arrays if it made
        any. Run in order, each on what the one before it left, because
        restamping the clock has to restamp the samples that survived the
        de-duplication rather than the ones that did not."""
        nonlocal counts, index, host_time, ticks, leadoff, gap_before
        if got is None:
            return
        fixed, said = got
        notes.append(said)
        if not fixed:
            return
        counts = fixed.get("counts", counts)
        index = fixed.get("index", index)
        host_time = fixed.get("host_time", host_time)
        ticks = fixed.get("ticks", ticks)
        leadoff = fixed.get("leadoff", leadoff)
        gap_before = fixed.get("gap_before", gap_before)

    if repair:
        apply(_dedupe_delivery(counts, index, host_time, ticks,
                               dict(leadoff=leadoff, gap_before=gap_before)))
        apply(_repin_clock(meta, ticks))
    return Capture(label, counts, index, host_time, meta, notes, ticks,
                   leadoff, gap_before)


def events(label: str) -> list[dict]:
    """The capture's events, on the same ruler as its sample `host_time`.

    **`host_time` here is the ALIGNED time**, which is what every analysis wants
    because it is what indexes correctly into `Capture.host_time`. The clock the
    event was actually stamped with is kept as `host_time_recorded`, and
    `aligned` says whether a correction was possible at all.

    Why the loader and not each caller: if the wall clock steps mid-recording,
    a raw event stamp and the sample column no longer share an origin, and
    every `searchsorted` against them is silently off by the size of the jump.
    Making the corrected value the default is what stops a future analysis from
    getting it wrong by simply not knowing to ask. The file on disk is never
    rewritten -- it stays the record of what was stamped.
    """
    p = prov.capture_path(label, ".events.json")
    if not p.exists():
        return []
    raw = json.loads(p.read_text())
    m = prov.capture_path(label, ".meta.json")
    meta = json.loads(m.read_text()) if m.exists() else {}
    out = []
    for e in timebase.align_events(meta, raw):
        e["host_time_recorded"] = e.get("host_time")
        e["host_time"] = e.pop("host_time_aligned")
        out.append(e)
    return out


def inputs(label: str, *, kinds=("button", "axis")) -> list[dict]:
    """Controller events, on the same ruler as the capture's samples.

    Stored as arrays rather than as JSON because of their rate, and returned
    here as the same event shape `events()` returns so an analysis does not
    have to care which container a label came out of. `host_time` is the
    ALIGNED time, for the same reason and with the same guarantee as in
    `events()`.

    `kinds` filters by what the control is: `button` for EV_KEY (the sparse,
    discrete labels most analyses want) and `axis` for EV_ABS (continuous).
    They are separated because a button press is an event and a stick position
    is a signal, and treating a hundred axis samples a second as if each were a
    discrete label is how a corpus becomes unusable.
    """
    from . import controller as ctl
    p = prov.capture_path(label, ".inputs.npz")
    if not p.exists():
        return []
    m = prov.capture_path(label, ".meta.json")
    meta = json.loads(m.read_text()) if m.exists() else {}
    z = np.load(p)
    want = set(kinds)
    raw = []
    for mono, ht, t, c, v in zip(z["input_mono"], z["input_host_time"],
                                 z["input_type"], z["input_code"],
                                 z["input_value"]):
        kind = ("button" if int(t) == ctl.EV_KEY
                else "axis" if int(t) == ctl.EV_ABS else "other")
        if kind not in want:
            continue
        raw.append(dict(kind=kind, cond=ctl.code_name(int(t), int(c)),
                        host_time=float(ht), mono=float(mono),
                        type=int(t), code=int(c), value=int(v)))
    out = []
    for e in timebase.align_events(meta, raw):
        e["host_time_recorded"] = e.get("host_time")
        e["host_time"] = e.pop("host_time_aligned")
        out.append(e)
    return out


def contiguous_runs(index: np.ndarray, minlen: int) -> list[tuple[int, int]]:
    brk = np.where(np.diff(index) != 1)[0]
    e = np.concatenate(([0], brk + 1, [len(index)]))
    return [(a, b) for a, b in zip(e[:-1], e[1:]) if b - a >= minlen]


def _filters(fs, hp, lp, notch, notch_q):
    sos_hp = signal.butter(4, hp, "hp", fs=fs, output="sos") if hp else None
    sos_lp = signal.butter(4, lp, "lp", fs=fs, output="sos") if lp else None
    bn = signal.iirnotch(notch, notch_q, fs) if notch else None
    return sos_hp, sos_lp, bn


def preprocess(x, fs, hp=DEFAULTS["hp"], lp=DEFAULTS["lp"],
               notch=DEFAULTS["notch"], notch_q=DEFAULTS["notch_q"],
               trim=DEFAULTS["trim"]):
    """Filter a trace. hp/lp/notch of 0 or None disable that stage."""
    sos_hp, sos_lp, bn = _filters(fs, hp, lp, notch, notch_q)
    if sos_hp is not None:
        x = signal.sosfiltfilt(sos_hp, x)
    if bn is not None:
        x = signal.filtfilt(bn[0], bn[1], x)
    if sos_lp is not None:
        x = signal.sosfiltfilt(sos_lp, x)
    k = int(trim * fs)
    return x[k:len(x) - k] if k and len(x) > 2 * k + 8 else x


def welch_psd(cap: Capture, c: int, nperseg=2048, filtered=False, **kw):
    """Mean PSD over contiguous runs. Returns (f, pxx)."""
    runs = contiguous_runs(cap.index, nperseg + int(cap.fs))
    if not runs:
        runs = [(0, cap.counts.shape[1])]
    acc, f = [], None
    for a, b in runs:
        x = cap.uv(c)[a:b]
        x = preprocess(x, cap.fs, **kw) if filtered else x - x.mean()
        if len(x) < nperseg:
            continue
        f, p = signal.welch(x, fs=cap.fs, nperseg=nperseg)
        acc.append(p)
    if not acc:
        x = cap.uv(c)
        f, p = signal.welch(x - x.mean(), fs=cap.fs,
                            nperseg=min(nperseg, len(x)))
        acc = [p]
    return f, np.mean(acc, axis=0)


def _band(f, p, lo, hi):
    k = (f >= lo) & (f < hi)
    return float(np.trapezoid(p[k], f[k])) if k.any() else 0.0


def spectral_slope(f, p, lo=2.0, hi=40.0, exclude=(7.0, 14.0)):
    """log-log slope of the background. EEG is ~ -1..-2; ~0 means white."""
    k = (f >= lo) & (f <= hi) & ~((f >= exclude[0]) & (f <= exclude[1])) & (p > 0)
    if k.sum() < 8:
        return float("nan")
    return float(np.polyfit(np.log10(f[k]), np.log10(p[k]), 1)[0])


def summary(label: str, line_hz: float = 60.0, nperseg: int = 2048,
            slope_lo: float = 2.0, slope_hi: float = 40.0) -> dict:
    """DC, AC, band split, spectral slope, noise floor. Unfiltered by design:
    this is the view that must show you what the instrument actually produced.
    `line_hz` only selects which mains line to report, it removes nothing."""
    cap = load(label)
    fullscale = (1 << 23) - 1
    runs = contiguous_runs(cap.index, 2048)
    out = dict(label=label, meta=cap.meta, fs=cap.fs, lsb_uv=cap.lsb_uv,
               n=int(cap.counts.shape[1]),
               n_runs=len(runs),
               kept_frac=float(sum(b - a for a, b in runs) /
                               max(1, cap.counts.shape[1])),
               channels=[])
    for c in range(cap.nch):
        x = cap.uv(c)
        raw = cap.counts[c].astype(np.float64)
        f, p = welch_psd(cap, c, nperseg=nperseg)
        ac = x - x.mean()
        med = lambda lo, hi: float(np.median(p[(f >= lo) & (f <= hi)]))
        k = (f >= 1) & (f <= 45)
        tot = _band(f, p, 1, 45)
        ch = dict(
            dc_uv=float(x.mean()),
            ac_rms_uv=float(ac.std()),
            ac_pp_uv=float(ac.max() - ac.min()),
            pp_over_rms=float((ac.max() - ac.min()) / ac.std()) if ac.std() else 0.0,
            saturation_pct=float(np.mean(np.abs(raw) > 0.99 * fullscale) * 100),
            rms_1_45_uv=float(np.sqrt(np.trapezoid(p[k], f[k]))),
            psd_5_45=med(5, 45), psd_60_90=med(60, 90), psd_100_200=med(100, 200),
            slope_2_40=spectral_slope(f, p, slope_lo, slope_hi),
            peak_hz=float(f[1:][np.argmax(p[1:])]),
            bands={n: (_band(f, p, lo, hi) / tot * 100 if tot else 0.0)
                   for n, (lo, hi) in EEG_BANDS.items()},
            # absolute: the mains line sits outside 1-45 Hz, so normalizing it
            # by the in-band total explodes when the in-band signal is quiet.
            psd_60=med(line_hz - 0.6, line_hz + 0.6),
            line_hz=line_hz,
        )
        out["channels"].append(ch)
    return out


def psd_curve(label: str, filtered=False, fmax=100.0, nperseg=2048,
              hp=DEFAULTS["hp"], lp=DEFAULTS["lp"], notch=DEFAULTS["notch"],
              notch_q=DEFAULTS["notch_q"]) -> dict:
    cap = load(label)
    curves = []
    kw = dict(hp=hp, lp=lp, notch=notch, notch_q=notch_q) if filtered else {}
    for c in range(cap.nch):
        f, p = welch_psd(cap, c, nperseg=nperseg, filtered=filtered, **kw)
        k = (f > 0) & (f <= fmax)
        curves.append(dict(f=f[k].tolist(), p=p[k].tolist()))
    return dict(label=label, fs=cap.fs, gain=cap.meta.get("gain"),
                rld_sens=cap.meta.get("rld_sens"), curves=curves,
                # The same table `compare` returns, so a one-capture PSD has
                # numbers to show and not just a picture. Without this the
                # result panel rendered empty for the default analysis, which
                # is the "results box is always blank" that was reported.
                table=[_psd_row(label)])


def _psd_row(label: str) -> dict:
    s = summary(label)
    return {"label": label, **{f"ch{c+1}_psd_5_45": ch["psd_5_45"]
                               for c, ch in enumerate(s["channels"])}}


def compare(labels: list[str], fmax=100.0, **kw) -> dict:
    """Overlay PSDs of several captures; the RLD ladder view."""
    return dict(kind="compare",
                series=[psd_curve(l, fmax=fmax, **kw) for l in labels],
                table=[_psd_row(l) for l in labels])


def clean_epochs(cap: Capture, c: int, epoch_s=2.0, ppmax=150.0,
                 sel: np.ndarray | None = None, **kw):
    """Artifact-rejected epochs. Returns (list_of_epochs, n_total)."""
    L = int(epoch_s * cap.fs)
    keep, total = [], 0
    idx = cap.index if sel is None else cap.index[sel]
    dat = cap.uv(c) if sel is None else cap.uv(c)[sel]
    for a, b in contiguous_runs(idx, L + int(cap.fs)):
        x = preprocess(dat[a:b], cap.fs, **kw)
        for k in range(len(x) // L):
            e = x[k * L:(k + 1) * L]
            total += 1
            if e.max() - e.min() <= ppmax:
                keep.append(e)
    return keep, total


def berger(label: str, epoch_s=2.0, ppmax=150.0, settle=None,
           alpha_lo=8.0, alpha_hi=13.0, ref_lo=4.0, ref_hi=30.0,
           hp=DEFAULTS["hp"], lp=DEFAULTS["lp"], notch=DEFAULTS["notch"],
           notch_q=DEFAULTS["notch_q"],
           allow_pseudoreplication: bool = False) -> dict:
    """Alpha, eyes closed vs open.

    The default test is a **paired, block-level Wilcoxon signed-rank**. Epochs
    within a block are not independent; pooling them is pseudoreplication and it
    inflates significance -- on 2026-07-09 it turned p = 0.45 into p = 0.12.
    Pooling is available only behind `allow_pseudoreplication=True`, and the
    result is then stamped `pseudoreplicated: true`.

    Both **absolute** and **relative** band power are reported. Relative alpha
    can move purely because its 4-30 Hz denominator moved -- on 2026-07-10 the
    only "significant" result was exactly that: eyes-closed cut theta by 40%,
    inflating alpha's share while absolute alpha sat still (ratio 0.989).
    """
    cap, ev = load(label), events(label)
    if not ev:
        raise ValueError(f"{label} has no events.json -- not a cued experiment")
    settle = cap.meta.get("settle", 3.0) if settle is None else settle
    block = cap.meta.get("block", 20.0)
    ht = cap.host_time
    filt = dict(hp=hp, lp=lp, notch=notch, notch_q=notch_q)
    params = dict(epoch_s=epoch_s, ppmax=ppmax, settle=settle,
                  alpha=[alpha_lo, alpha_hi], reference=[ref_lo, ref_hi],
                  filters=filt, allow_pseudoreplication=allow_pseudoreplication)

    out = dict(label=label, analysis="berger",
               analysis_version=ANALYSIS_VERSION, params=params,
               capture_sha256=prov.sha256_file(prov.capture_path(label, ".npz")),
               block=block, settle=settle, pseudoreplicated=False,
               n_reps=len(ev) // 2, channels=[])

    for c in range(cap.nch):
        per_block, pooled = [], {"closed": [], "open": []}
        for e in ev:
            sel = np.where((ht >= e["host_time"] + settle) &
                           (ht < e["host_time"] + block))[0]
            if len(sel) < int(epoch_s * cap.fs) + int(cap.fs):
                continue
            eps, tot = clean_epochs(cap, c, epoch_s, ppmax, sel=sel, **filt)
            rel, absol, refs = [], [], []
            for seg in eps:
                f, p = signal.welch(seg, fs=cap.fs, nperseg=len(seg))
                num = _band(f, p, alpha_lo, alpha_hi)
                den = _band(f, p, ref_lo, ref_hi)
                absol.append(num); refs.append(den)
                rel.append(num / den * 100 if den else 0.0)
            if not eps:
                continue
            pooled[e["cond"]].extend(rel)
            per_block.append(dict(rep=e["rep"], cond=e["cond"],
                                  alpha_abs=float(np.mean(absol)),
                                  alpha_rel=float(np.mean(rel)),
                                  reference_abs=float(np.mean(refs)),
                                  clean=len(eps), total=tot))

        res = dict(blocks=per_block)
        for key, tag in (("alpha_abs", "absolute"), ("alpha_rel", "relative")):
            cl = np.array([b[key] for b in per_block if b["cond"] == "closed"])
            op = np.array([b[key] for b in per_block if b["cond"] == "open"])
            n = min(len(cl), len(op))
            if n < 2:
                res[tag] = dict(n_blocks=n, note="too few blocks to test")
                continue
            cl, op = cl[:n], op[:n]
            w, pw = stats.wilcoxon(cl, op, alternative="greater")
            t, pt = stats.ttest_rel(cl, op)
            pt = pt / 2 if t > 0 else 1 - pt / 2
            d = float(np.mean(cl - op) / np.std(cl - op, ddof=1)) if n > 1 else 0.0
            res[tag] = dict(
                n_blocks=n, closed=float(cl.mean()), open=float(op.mean()),
                ratio=float(cl.mean() / op.mean()) if op.mean() else 0.0,
                wins=int((cl > op).sum()),
                test="wilcoxon_signed_rank_paired_blocks",
                p=float(pw), p_paired_t=float(pt), cohens_dz=d,
                p_floor=float(2.0 ** -n),   # smallest attainable p at this n
            )

        if allow_pseudoreplication and pooled["closed"] and pooled["open"]:
            u, pp = stats.mannwhitneyu(pooled["closed"], pooled["open"],
                                       alternative="greater")
            res["pooled_epochs"] = dict(
                n_closed=len(pooled["closed"]), n_open=len(pooled["open"]),
                p=float(pp), test="mannwhitneyu_pooled_epochs",
                warning="PSEUDOREPLICATED: epochs within a block are not "
                        "independent; this p-value is inflated")
            out["pseudoreplicated"] = True

        # A relative-alpha effect with no absolute-alpha effect is a moving
        # denominator, not a Berger effect. Say so, rather than leaving it to
        # the reader (this is exactly what happened on 2026-07-10).
        ab, rl = res.get("absolute", {}), res.get("relative", {})
        if "p" in ab and "p" in rl:
            refc = np.array([b["reference_abs"] for b in per_block
                             if b["cond"] == "closed"])
            refo = np.array([b["reference_abs"] for b in per_block
                             if b["cond"] == "open"])
            nref = min(len(refc), len(refo))
            ref_ratio = (float(refc[:nref].mean() / refo[:nref].mean())
                         if nref and refo[:nref].mean() else None)
            res["reference_band_ratio"] = ref_ratio
            if rl["p"] < 0.05 and ab["p"] > 0.05:
                res["denominator_artifact"] = True
                res["interpretation"] = (
                    "Relative alpha moved but ABSOLUTE alpha did not "
                    f"(ratio {ab['ratio']:.3f}, p={ab['p']:.3f}). The "
                    f"{ref_lo}-{ref_hi} Hz reference band changed by "
                    f"{ref_ratio:.3f}x. This is a denominator effect, not a "
                    "Berger effect.")
            else:
                res["denominator_artifact"] = False

        # is there an alpha peak at all, or just band power on a 1/f slope?
        allc = clean_epochs(cap, c, epoch_s, ppmax, **filt)[0]
        if allc:
            E = np.array(allc)
            f, p = signal.welch(E, fs=cap.fs, nperseg=E.shape[1], axis=1)
            p = p.mean(axis=0)
            pk_lo, pk_hi = alpha_lo - 1.0, alpha_hi + 1.0
            res["slope_2_40"] = spectral_slope(f, p, exclude=(pk_lo, pk_hi))
            kk = (f >= 2) & (f <= 40) & ~((f >= pk_lo) & (f <= pk_hi)) & (p > 0)
            if kk.sum() >= 8:
                A = np.polyfit(np.log10(f[kk]), np.log10(p[kk]), 1)
                ka = (f >= pk_lo) & (f <= pk_hi)
                resid = p[ka] / 10 ** np.polyval(A, np.log10(f[ka]))
                res["alpha_peak_ratio"] = float(resid.max())
                res["alpha_peak_hz"] = float(f[ka][np.argmax(resid)])
                res["has_alpha_peak"] = bool(resid.max() >= 1.5)
            res["n_epochs"] = len(allc)
        out["channels"].append(res)

    # a summary a human cannot misread
    n_tests = 2 * cap.nch
    out["multiple_comparisons"] = dict(
        n_tests=n_tests, bonferroni_alpha=0.05 / n_tests)
    return out


def saccade(label: str, pre_s=0.5, post_s=1.5, hp=0.1, lp=10.0,
            notch=60.0, notch_q=30.0,
            min_deflection_uv=50.0, max_corr=-0.5) -> dict:
    """V2 validation: does the instrument see the ocular dipole?

    An eye is a dipole -- cornea positive, retina negative. With ch1 on the left
    temple and ch2 on the right, a leftward saccade must deflect the two channels
    in OPPOSITE directions, and a rightward saccade must reverse both signs.

    Nothing else in this instrument can produce that. Noise does not invert
    across channels; mains does not; a gain error does not; a wiring or sign
    error cannot survive it. It simultaneously proves the channels are truly
    differential, correctly wired, correctly signed, and recording a real
    physiological dipole.

    The `sham` condition (cue plays, subject stays still) is the control for the
    cue itself. A deflection on sham trials means the response is not ocular.

    Acceptance (preregistered):
      * ch1-ch2 correlation over the response window <= -0.5, for left and right
      * the sign of each channel reverses between left and right
      * |deflection| >= 50 µV
      * sham shows no deflection above 3x the pre-cue baseline SD
    """
    cap, ev = load(label), events(label)
    if not ev:
        raise ValueError(f"{label} has no events.json")
    fs = cap.fs
    npre, npost = int(pre_s * fs), int(post_s * fs)
    filt = dict(hp=hp, lp=lp, notch=notch, notch_q=notch_q, trim=0.0)
    sig = [preprocess(cap.uv(c), fs, **filt) for c in range(cap.nch)]
    ht = cap.host_time

    def epochs_for(cond):
        out = [[] for _ in range(cap.nch)]
        for e in ev:
            if e.get("cond") != cond:
                continue
            i = int(np.searchsorted(ht, e["host_time"]))
            if i - npre < 0 or i + npost >= len(ht):
                continue
            for c in range(cap.nch):
                seg = sig[c][i - npre:i + npost]
                out[c].append(seg - seg[:npre].mean())     # baseline-correct
        return [np.array(o) for o in out]

    res = dict(label=label, analysis="saccade",
               analysis_version=ANALYSIS_VERSION,
               capture_sha256=prov.sha256_file(prov.capture_path(label, ".npz")),
               params=dict(pre_s=pre_s, post_s=post_s, filters=filt,
                           min_deflection_uv=min_deflection_uv,
                           max_corr=max_corr),
               conditions={}, criteria={}, verdict=None)

    resp = slice(npre, npre + int(1.0 * fs))    # 0..1 s after the cue
    means = {}
    for cond in ("left", "right", "sham"):
        E = epochs_for(cond)
        if not len(E[0]):
            continue
        m = [e.mean(axis=0) for e in E]
        means[cond] = m
        base_sd = float(np.mean([e[:, :npre].std() for e in E]))
        entry = dict(n_trials=int(len(E[0])), baseline_sd_uv=base_sd,
                     channels=[])
        for c in range(cap.nch):
            w = m[c][resp]
            peak = float(w[np.argmax(np.abs(w))])
            entry["channels"].append(dict(
                deflection_uv=peak, sign=int(np.sign(peak)),
                snr=float(abs(peak) / base_sd) if base_sd else 0.0))
        if cap.nch >= 2:
            entry["ch1_ch2_corr"] = float(
                np.corrcoef(m[0][resp], m[1][resp])[0, 1])
        res["conditions"][cond] = entry

    crit = res["criteria"]
    have_lr = "left" in means and "right" in means
    if have_lr:
        L, R = res["conditions"]["left"], res["conditions"]["right"]
        crit["polarity_inverted_left"] = bool(L.get("ch1_ch2_corr", 0) <= max_corr)
        crit["polarity_inverted_right"] = bool(R.get("ch1_ch2_corr", 0) <= max_corr)
        crit["sign_reverses_with_direction"] = bool(all(
            L["channels"][c]["sign"] == -R["channels"][c]["sign"] != 0
            for c in range(min(2, cap.nch))))
        crit["deflection_big_enough"] = bool(all(
            abs(d["channels"][c]["deflection_uv"]) >= min_deflection_uv
            for d in (L, R) for c in range(min(2, cap.nch))))
    if "sham" in res["conditions"]:
        S = res["conditions"]["sham"]
        crit["sham_is_null"] = bool(all(
            ch["snr"] < 3.0 for ch in S["channels"]))

    if have_lr:
        required = ["polarity_inverted_left", "polarity_inverted_right",
                    "sign_reverses_with_direction", "deflection_big_enough"]
        if "sham_is_null" in crit:
            required.append("sham_is_null")
        res["verdict"] = "PASS" if all(crit[k] for k in required) else "FAIL"
        res["failed_criteria"] = [k for k in required if not crit[k]]
    return res


def _cue_epochs(cap, ev, cond, npre, npost, sig, ht, baseline=True):
    out = [[] for _ in range(cap.nch)]
    for e in ev:
        if e.get("cond") != cond:
            continue
        i = int(np.searchsorted(ht, e["host_time"]))
        if i - npre < 0 or i + npost >= len(ht):
            continue
        for c in range(cap.nch):
            seg = sig[c][i - npre:i + npost]
            out[c].append(seg - seg[:npre].mean() if baseline else seg)
    return [np.array(o) for o in out]


def blink(label: str, pre_s=0.5, post_s=1.5, hp=0.1, lp=10.0,
          notch=60.0, notch_q=30.0, min_pp_uv=100.0,
          min_detect_frac=0.9, max_latency_sd_s=0.25,
          sham_max_pp_ratio=1.5) -> dict:
    """V3 validation: cued blinks.

    A blink drives the corneo-retinal dipole vertically; frontopolar electrodes
    see 100-300 µV. Unlike alpha, it is large, time-locked, and under the
    subject's control, so a sham cue is a real control: if a deflection appears
    when the cue plays and the subject does nothing, the response is not ocular.

    Per-trial detection matters more than the average. A single 3 mV movement
    artifact can manufacture a beautiful grand average from twenty null trials,
    which is why `detected_frac` is a criterion and the average alone is not.
    """
    cap = load(label)
    ev = events(label)
    if not ev:
        raise ValueError(f"{label} has no events.json")
    fs = cap.fs
    npre, npost = int(pre_s * fs), int(post_s * fs)
    filt = dict(hp=hp, lp=lp, notch=notch, notch_q=notch_q, trim=0.0)
    sig = [preprocess(cap.uv(c), fs, **filt) for c in range(cap.nch)]
    ht = cap.host_time
    resp = slice(npre, npre + int(0.8 * fs))

    res = dict(label=label, analysis="blink", analysis_version=ANALYSIS_VERSION,
               capture_sha256=prov.sha256_file(prov.capture_path(label, ".npz")),
               params=dict(pre_s=pre_s, post_s=post_s, filters=filt,
                           min_pp_uv=min_pp_uv, min_detect_frac=min_detect_frac,
                           max_latency_sd_s=max_latency_sd_s,
                           sham_max_pp_ratio=sham_max_pp_ratio),
               conditions={}, criteria={}, verdict=None)

    for cond in ("blink", "sham"):
        E = _cue_epochs(cap, ev, cond, npre, npost, sig, ht)
        if not len(E[0]):
            continue
        entry = dict(n_trials=int(len(E[0])), channels=[])
        for c in range(cap.nch):
            trials = E[c][:, resp]
            pps = np.ptp(trials, axis=1)
            base_pp = np.ptp(E[c][:, :npre], axis=1)
            base_sd = float(E[c][:, :npre].std())
            lat = np.argmax(np.abs(trials), axis=1) / fs
            # Compare peak-to-peak against peak-to-peak. The peak-to-peak of
            # Gaussian noise over a 400-sample window is already ~6 sigma, so a
            # pp/sd "SNR" can never approach 1 even on a perfectly null trial.
            entry["channels"].append(dict(
                mean_pp_uv=float(pps.mean()),
                baseline_pp_uv=float(base_pp.mean()),
                pp_ratio=float(pps.mean() / base_pp.mean())
                if base_pp.mean() else 0.0,
                detected_frac=float(np.mean(pps >= min_pp_uv)),
                latency_s=float(lat.mean()),
                latency_sd_s=float(lat.std()),
                baseline_sd_uv=base_sd))
        res["conditions"][cond] = entry

    crit = res["criteria"]
    if "blink" in res["conditions"]:
        B = res["conditions"]["blink"]
        crit["amplitude"] = bool(any(ch["mean_pp_uv"] >= min_pp_uv
                                     for ch in B["channels"]))
        crit["reliable_across_trials"] = bool(any(
            ch["detected_frac"] >= min_detect_frac for ch in B["channels"]))
        crit["consistent_latency"] = bool(any(
            ch["latency_sd_s"] <= max_latency_sd_s for ch in B["channels"]))
    if "sham" in res["conditions"]:
        # Null means: no cue-locked excursion above the baseline's own noise,
        # and nothing a human would call a blink.
        crit["sham_is_null"] = bool(all(
            ch["pp_ratio"] < sham_max_pp_ratio and ch["mean_pp_uv"] < min_pp_uv
            for ch in res["conditions"]["sham"]["channels"]))

    if crit:
        required = [k for k in ("amplitude", "reliable_across_trials",
                                "consistent_latency", "sham_is_null")
                    if k in crit]
        res["verdict"] = "PASS" if all(crit[k] for k in required) else "FAIL"
        res["failed_criteria"] = [k for k in required if not crit[k]]
    return res


def jaw(label: str, window_s=1.0, lo_hz=20.0, hi_hz=200.0,
        min_ratio=10.0, alpha=0.001) -> dict:
    """V4 validation: cued jaw clench.

    The temporal poles sit over temporalis. Clenching produces broadband EMG,
    an order of magnitude above baseline in 20-200 Hz — a band where cortical
    EEG contributes almost nothing at this montage.

    Paired over trials: each clench window is compared with the rest window that
    precedes it. Pooling the epochs would be pseudoreplication, which turned a
    real p = 0.45 into p = 0.12 elsewhere in this project.
    """
    cap = load(label)
    ev = events(label)
    if not ev:
        raise ValueError(f"{label} has no events.json")
    fs = cap.fs
    if hi_hz >= fs / 2:
        hi_hz = fs / 2 - 1.0
    n = int(window_s * fs)
    ht = cap.host_time
    raw = [cap.uv(c) for c in range(cap.nch)]

    def band_power(x):
        if len(x) < 32:
            return float("nan")
        f, p = signal.welch(x - x.mean(), fs=fs, nperseg=min(256, len(x)))
        return float(_band(f, p, lo_hz, hi_hz))

    pairs = [[] for _ in range(cap.nch)]        # (rest, clench) per trial
    for e in ev:
        if e.get("cond") != "clench":
            continue
        i = int(np.searchsorted(ht, e["host_time"]))
        if i - n < 0 or i + n >= len(ht):
            continue
        for c in range(cap.nch):
            pairs[c].append((band_power(raw[c][i - n:i]),
                             band_power(raw[c][i:i + n])))

    res = dict(label=label, analysis="jaw", analysis_version=ANALYSIS_VERSION,
               capture_sha256=prov.sha256_file(prov.capture_path(label, ".npz")),
               params=dict(window_s=window_s, band=[lo_hz, hi_hz],
                           min_ratio=min_ratio, alpha=alpha,
                           test="wilcoxon_paired_trials"),
               channels=[], criteria={}, verdict=None)

    for c in range(cap.nch):
        pr = [p for p in pairs[c] if np.isfinite(p[0]) and np.isfinite(p[1])]
        if len(pr) < 3:
            res["channels"].append(dict(n_trials=len(pr), note="too few trials"))
            continue
        rest = np.array([p[0] for p in pr])
        clench = np.array([p[1] for p in pr])
        try:
            p_val = float(stats.wilcoxon(clench, rest, alternative="greater").pvalue)
        except ValueError:
            p_val = 1.0
        res["channels"].append(dict(
            n_trials=len(pr), rest=float(rest.mean()), clench=float(clench.mean()),
            ratio=float(clench.mean() / rest.mean()) if rest.mean() else float("inf"),
            wins=int(np.sum(clench > rest)), p=p_val,
            p_floor=float(2.0 ** -len(pr))))

    good = [ch for ch in res["channels"] if "p" in ch]
    if good:
        res["criteria"]["power_ratio"] = bool(any(ch["ratio"] >= min_ratio
                                                  for ch in good))
        res["criteria"]["significant"] = bool(any(ch["p"] < alpha for ch in good))
        req = list(res["criteria"])
        res["verdict"] = "PASS" if all(res["criteria"][k] for k in req) else "FAIL"
        res["failed_criteria"] = [k for k in req if not res["criteria"][k]]
    return res


def contact(label: str, dc_min_uv: float = 1000.0,
            ac_min_uv: float = 2.0) -> dict:
    """Is each channel actually on skin?

    Two independent indicators:
      * DC half-cell offset. A dry electrode on skin develops a mV-scale
        electrode potential. Tens of µV means it is not touching skin (or the
        pair is shorted together).
      * The device's own lead-off comparators, carried per packet.
    Reported separately, never merged: they can disagree, and that is
    information.
    """
    cap = load(label)
    z = np.load(prov.capture_path(label, ".npz"))
    lo = z["leadoff"]
    out = dict(label=label, analysis="contact",
               analysis_version=ANALYSIS_VERSION,
               dc_min_uv=dc_min_uv, ac_min_uv=ac_min_uv,
               leadoff_bytes=sorted(int(v) for v in set(lo.tolist())),
               leadoff_nonzero_frac=float(np.mean(lo != 0)),
               channels=[])
    for c in range(cap.nch):
        x = cap.uv(c)
        dc, ac = float(x.mean()), float((x - x.mean()).std())
        has_dc = abs(dc) >= dc_min_uv
        has_ac = ac >= ac_min_uv
        out["channels"].append(dict(
            dc_uv=dc, ac_rms_uv=ac, dc_ok=has_dc, ac_ok=has_ac,
            verdict=("contact" if (has_dc and has_ac) else
                     "NO CONTACT (no half-cell offset)" if not has_dc else
                     "suspect (offset but near-floor AC)")))
    return out


def _n(name, default, **kw):
    return dict(name=name, type="number", default=default, **kw)


ANALYSES = {
    "summary": dict(
        fn=summary, needs_events=False,
        doc="DC/AC split, band powers, spectral slope, noise floor (unfiltered)",
        params=[_n("line_hz", 60.0), _n("nperseg", 2048),
                _n("slope_lo", 2.0), _n("slope_hi", 40.0)]),
    "psd": dict(
        fn=psd_curve, needs_events=False,
        doc="Power spectral density (multi-select overlays)",
        params=[dict(name="filtered", type="select", default=0, choices=[0, 1]),
                _n("fmax", 100.0), _n("nperseg", 2048),
                _n("hp", DEFAULTS["hp"]), _n("lp", DEFAULTS["lp"]),
                _n("notch", DEFAULTS["notch"]), _n("notch_q", DEFAULTS["notch_q"])]),
    "contact": dict(
        fn=contact, needs_events=False,
        doc="Electrode contact check: DC half-cell offset + lead-off comparators",
        params=[_n("dc_min_uv", 1000.0), _n("ac_min_uv", 2.0)]),
    "saccade": dict(
        fn=saccade, needs_events=True,
        doc="V2 validation: ocular dipole -- ch1/ch2 must invert, sign must reverse",
        params=[_n("pre_s", 0.5), _n("post_s", 1.5),
                _n("hp", 0.1), _n("lp", 10.0),
                _n("notch", 60.0), _n("notch_q", 30.0),
                _n("min_deflection_uv", 50.0), _n("max_corr", -0.5)]),
    "blink": dict(
        fn=blink, needs_events=True,
        doc="V3 validation: cued blinks -- amplitude, per-trial reliability, "
            "latency consistency, sham null",
        params=[_n("pre_s", 0.5), _n("post_s", 1.5),
                _n("hp", 0.1), _n("lp", 10.0),
                _n("notch", 60.0), _n("notch_q", 30.0),
                _n("min_pp_uv", 100.0), _n("min_detect_frac", 0.9),
                _n("max_latency_sd_s", 0.25), _n("sham_max_pp_ratio", 1.5)]),
    "jaw": dict(
        fn=jaw, needs_events=True,
        doc="V4 validation: jaw clench -- 20-200 Hz EMG, paired over trials",
        params=[_n("window_s", 1.0), _n("lo_hz", 20.0), _n("hi_hz", 200.0),
                _n("min_ratio", 10.0), _n("alpha", 0.001)]),
    "berger": dict(
        fn=berger, needs_events=True,
        doc="Alpha closed-vs-open: paired block-level test, absolute + relative, 1/f peak",
        params=[_n("epoch_s", 2.0), _n("ppmax", 150.0),
                _n("alpha_lo", 8.0), _n("alpha_hi", 13.0),
                _n("ref_lo", 4.0), _n("ref_hi", 30.0),
                _n("hp", DEFAULTS["hp"]), _n("lp", DEFAULTS["lp"]),
                _n("notch", DEFAULTS["notch"]), _n("notch_q", DEFAULTS["notch_q"]),
                dict(name="allow_pseudoreplication", type="select", default=0,
                     choices=[0, 1])]),
}
