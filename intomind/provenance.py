"""Provenance and integrity for captures.

Answers, for any capture: which firmware, which circuit revision, which code,
which register state, and are these the bytes that were recorded?

Layout note. Content-addressed capture *directories* were specified.
This module implements the same guarantees additively over the existing
`<label>.npz` / `<label>.meta.json` layout, so the running hub and every existing
analysis keep working. Migration to directories is mechanical once the manifest
is authoritative; the checksum file is already the hinge.
"""
from __future__ import annotations
import hashlib, json, os, pathlib, platform, re, subprocess, sys, time

from . import montage
from . import placement as _placement

# WHERE THINGS ARE, AND WHY NONE OF IT IS FOUND BY LOOKING UPWARDS.
#
# These were `parents[2] / "captures"` and `parents[2] / "config/site.json"`,
# resolved from this file's own location, back when the library lived inside
# one device's repository and there was only one of everything. That is how a
# device-agnostic library came to read one particular bench's electrode
# description and stamp it into every capture, whatever device was connected:
# one device's recordings all claimed the electrode prose of a different
# bench, hardware that never touched them.
#
# So nothing is inferred from the filesystem any more. An application declares
# its own data directory and its own site; a library that guesses either one
# guesses wrong the moment there is a second application or a second device.
CAPTURES = pathlib.Path(
    os.environ.get("INTOMIND_CAPTURES")
    or pathlib.Path.home() / ".local" / "share" / "intomind" / "captures")
SESSIONS = CAPTURES / "SESSIONS.jsonl"
# Retroactive hashes of captures recorded before checksums existed. Deliberately
# NOT the per-capture `<label>.sha256`, whose presence means "finalized at write
# time". These bytes were first attested long after they were recorded, and the
# distinction must survive: they are tamper-evident from the marking date
# onward, and unattested before it.
LEGACY = CAPTURES / "LEGACY.sha256"
# Montages asserted for captures recorded before the montage was structured.
# A SIDECAR, deliberately, for the same reason LEGACY.sha256 is one: a
# capture's own manifest is the record of what was known WHEN IT WAS RECORDED,
# and rewriting it would erase the difference between a fact the instrument
# captured and a fact a human attached afterwards. It would also break the
# capture's checksum and require unfreezing a finalized file, which is exactly
# the property that makes captures trustworthy. Nothing here touches a capture.
MONTAGE_BACKFILL = CAPTURES / "MONTAGE_BACKFILL.json"


#: What a capture may be called. A label becomes a filename, and an
#: application takes it from wherever its users come from, so it is
#: checked here rather than at each of the places that join it to a path.
LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def check_label(label: str) -> str:
    """Refuse a label that is not a plain name.

    A label reaches this library from a command line, a web request, or a
    filename somebody typed. Joined to a path without checking, `../` in
    one of them reads and writes outside the collection, and a leading dot
    hides the result. The rule is deliberately narrow: letters, digits,
    and the three separators that read well, starting with a letter or a
    digit.
    """
    if not isinstance(label, str) or not LABEL.match(label):
        raise ValueError(
            f"{label!r} is not a capture label. A label is letters, digits, "
            "dots, dashes and underscores, starts with a letter or a digit, "
            "and is at most 128 characters.")
    if ".." in label:
        raise ValueError(f"{label!r} is not a capture label: it walks up a directory")
    return label


def capture_path(label: str, suffix: str) -> pathlib.Path:
    """The path of one of a capture's files, with the label checked.

    Every place that turns a label into a path goes through here, so
    there is one rule and it cannot be forgotten in one of them.
    """
    return captures_dir() / f"{check_label(label)}{suffix}"


def captures_dir() -> pathlib.Path:
    """Where captures live, right now.

    Ask rather than remember. A module that binds this at import time holds
    a stale answer the moment an application redirects, which is how a
    recorder once wrote into one directory while the reader looked in
    another and reported an empty collection rather than an error.
    """
    return CAPTURES


def use_captures_dir(path) -> pathlib.Path:
    """Point every capture path at `path`, and return it.

    An application built ON this library keeps its own data: the joystick
    application that records game sessions keeps them in its own collection,
    not in
    the bench's `captures/`, so that neither collection can be mistaken for the
    other and neither ledger accumulates the other's rows.

    A single function rather than five module globals for callers to set
    individually, because that is exactly how `prov.LEGACY` was once left
    pointing at the real ledger while the other four were redirected, and a
    test suite wrote its own hashes into the project's record.
    """
    global CAPTURES, SESSIONS, LEGACY, MONTAGE_BACKFILL
    CAPTURES = pathlib.Path(path)
    CAPTURES.mkdir(parents=True, exist_ok=True)
    SESSIONS = CAPTURES / "SESSIONS.jsonl"
    LEGACY = CAPTURES / "LEGACY.sha256"
    MONTAGE_BACKFILL = CAPTURES / "MONTAGE_BACKFILL.json"
    # `analysis` keeps a module level name for the callers that already use
    # it, so it is rebound here too. Everything written since asks
    # `captures_dir()` instead, which cannot go stale.
    from . import analysis
    analysis.CAPTURES = CAPTURES
    return CAPTURES

# v2 (2026-07-11): records adc_bits, channels and device_proto. v1 manifests do
#   not state the resolution they were recorded at, so they carry an assumption
#   rather than a fact.
# v3 (2026-08-04): records what produced the signal and where it was
#   conditioned (`signal_chain`).
# v4 (2026-08-06): records the montage as DATA (`montage`), so a channel can be
#   selected by what it is rather than by index, and the clock block carries
#   the realtime/monotonic anchors needed to align events after a wall-clock
#   step. Older manifests remain readable; they simply cannot answer those two
#   questions, and say null rather than guessing.
# v5 (2026-08-20): records WHERE THE ELECTRODES WERE PUT. `electrode_layout` is
#   what the device offers, `placement` is where this run put it including the
#   note and what each electrode sat over, and `montage_source` says whether
#   the montage was generated from that placement or copied out of a bench
#   file. Before this, every recording copied one bench's montage whatever was
#   plugged in, which is why a four-channel IntoMind One capture from
#   2026-08-19 carries a two-channel temple montage it was never near. Older
#   manifests read as `montage_source: null` and are not rewritten.
# v6 (2026-09-28): records WHAT THE SAMPLES WERE, read off the packets that
#   carried them: `signal_source` is the mode every packet declared, and
#   `synthetic` is true when the device generated the signal rather than
#   measuring it (protocol 1.3). Before this a generated recording said so
#   only in the configuration its recorder happened to read. Older manifests
#   read as null for both.
MANIFEST_VERSION = 6


# ---------------------------------------------------------------- hashing
def sha256_file(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def checksum_path(label: str) -> pathlib.Path:
    return capture_path(label, ".sha256")


def write_checksums(label: str) -> dict[str, str]:
    """Hash every artifact of a capture. Written last, so its presence means
    the capture was finalized."""
    sums = {}
    for suffix in (".npz", ".meta.json", ".events.json", ".inputs.npz"):
        p = capture_path(label, suffix)
        if p.exists():
            sums[p.name] = sha256_file(p)
    checksum_path(label).write_text(
        "".join(f"{v}  {k}\n" for k, v in sorted(sums.items())))
    return sums


def aborted_reason(label: str) -> str | None:
    m = capture_path(label, ".meta.json")
    if not m.exists():
        return None
    try:
        return json.loads(m.read_text()).get("aborted")
    except Exception:
        return None


def _check_lines(lines) -> list[str]:
    problems = []
    for line in lines:
        if not line.strip():
            continue
        want, name = line.split(None, 1)
        p = CAPTURES / name.strip()
        if not p.exists():
            problems.append(f"missing: {name.strip()}")
        elif sha256_file(p) != want:
            problems.append(f"CHECKSUM MISMATCH: {name.strip()}")
    return problems


def legacy_ledger() -> dict[str, list[str]]:
    """label -> its retroactive hash lines."""
    if not LEGACY.exists():
        return {}
    out: dict[str, list[str]] = {}
    for line in LEGACY.read_text().splitlines():
        if not line.strip():
            continue
        _, name = line.split(None, 1)
        label = name.strip().rsplit(".", 1)[0]
        for suffix in (".meta", ".events"):        # <label>.meta.json etc.
            if label.endswith(suffix):
                label = label[: -len(suffix)]
        out.setdefault(label, []).append(line)
    return out


def verify(label: str) -> tuple[bool, list[str]]:
    """(ok, problems).

    Four distinct states, never conflated:
      * checksums present and matching -> finalized at write time, intact
      * checksums absent, `aborted` set -> the run died; NOT a complete capture
      * checksums absent, in the legacy ledger -> recorded before checksums
        existed. Marked `provenance: incomplete`. Tamper-evident only from the
        marking date; the bytes as recorded were never attested.
      * checksums absent, unmarked -> unverified, and nobody has looked
    A checksum that no longer matches is always a hard failure.
    """
    cp = checksum_path(label)
    if cp.exists():
        problems = _check_lines(cp.read_text().splitlines())
        return not problems, problems

    why = aborted_reason(label)
    if why:
        return False, [f"ABORTED ({why}): never finalized, do not analyze"]

    lines = legacy_ledger().get(label)
    if lines:
        problems = _check_lines(lines)
        if problems:
            return False, problems
        return True, ["provenance incomplete: recorded before checksums "
                      "existed; unchanged since it was marked, but its bytes "
                      "at recording time were never attested"]
    return True, ["unverified: no checksum file, not marked (recorded before P0)"]


def mark_legacy(label: str, board_revision: str | None = None) -> list[str]:
    """Stamp a pre-checksum capture as `provenance: incomplete` and record its
    hashes in the legacy ledger.

    This is what the integrity phase's acceptance criterion actually asks for:
    *every existing capture is migrated or explicitly marked.* Inferring
    "unverified" from a missing file is not a marking -- a capture whose
    `.sha256` was deleted would look identical to one recorded before the scheme
    existed, and those are very different claims.
    """
    m = capture_path(label, ".meta.json")
    if checksum_path(label).exists():
        raise ValueError(f"{label} is finalized; nothing to mark")
    if not m.exists():
        raise FileNotFoundError(m)

    try:
        os.chmod(m, 0o644)
    except OSError:
        pass
    meta = json.loads(m.read_text())
    meta["provenance"] = "incomplete"
    meta["provenance_note"] = (
        "Recorded before capture checksums existed. Hashed retroactively on "
        "first marking: tamper-evident from then on, unattested before. "
        "Board revision and register state may be absent.")
    # Never guess a board revision. A capture pooled across the RLD rework is
    # worse than a capture nobody trusts, and `unknown` is a fact whereas
    # A guessed revision is a fabrication that would survive into someone's analysis.
    if not meta.get("board_revision"):
        meta["board_revision"] = board_revision or "unknown"
        if board_revision:
            meta["board_revision_source"] = "asserted by a human at marking time"
    m.write_text(json.dumps(meta, indent=2))

    lines = []
    for suffix in (".npz", ".meta.json", ".events.json"):
        p = capture_path(label, suffix)
        if p.exists():
            lines.append(f"{sha256_file(p)}  {p.name}")
    existing = LEGACY.read_text().splitlines() if LEGACY.exists() else []
    keep = [l for l in existing if not l.strip().endswith(tuple(
        f"{label}{s}" for s in (".npz", ".meta.json", ".events.json")))]
    LEGACY.write_text("\n".join(sorted(keep + lines)) + "\n")
    freeze(label)
    return lines


def assert_revision(label: str, revision: str, source: str) -> list[str]:
    """Replace an `unknown` board revision on an already-marked capture, and
    re-hash it into the legacy ledger.

    `mark_legacy` refuses to guess, and rightly: it stamps `unknown` and stops.
    But `unknown` is a hole, not an answer, and a capture nobody can place on
    one side of the RLD rework is a capture nobody can use. This closes the hole
    ONLY where the revision is *derivable from an artifact*, and it records that
    artifact in `board_revision_source` so the claim carries its own citation.

    It will not overwrite a revision that is already stated, and it will not
    touch a capture finalized at write time (its meta.json is inside a checksum
    the recorder attested; re-writing it would forge that attestation). The
    `.npz` is never modified -- only the manifest, and the ledger that guards it.
    """
    if not source:
        raise ValueError("a revision without its source is a guess; give --source")
    m = capture_path(label, ".meta.json")
    if checksum_path(label).exists():
        raise ValueError(
            f"{label} was finalized at write time; its manifest is attested. "
            "Do not rewrite it.")
    if not m.exists():
        raise FileNotFoundError(m)
    if not legacy_ledger().get(label):
        raise ValueError(f"{label} is not marked; run mark_legacy first")

    meta = json.loads(m.read_text())
    have = meta.get("board_revision")
    if have and have != "unknown":
        raise ValueError(f"{label} already states board_revision={have!r}")

    try:
        os.chmod(m, 0o644)
    except OSError:
        pass
    meta["board_revision"] = revision
    meta["board_revision_source"] = source
    m.write_text(json.dumps(meta, indent=2))

    lines = []
    for suffix in (".npz", ".meta.json", ".events.json"):
        p = capture_path(label, suffix)
        if p.exists():
            lines.append(f"{sha256_file(p)}  {p.name}")
    existing = LEGACY.read_text().splitlines() if LEGACY.exists() else []
    keep = [l for l in existing if not l.strip().endswith(tuple(
        f"{label}{s}" for s in (".npz", ".meta.json", ".events.json")))]
    LEGACY.write_text("\n".join(sorted(keep + lines)) + "\n")
    freeze(label)
    return lines


def freeze(label: str) -> None:
    """Make a finalized capture read-only. Best effort; never fatal."""
    for suffix in (".npz", ".meta.json", ".events.json", ".inputs.npz",
                   ".sha256"):
        p = capture_path(label, suffix)
        if p.exists():
            try:
                os.chmod(p, 0o444)
            except OSError:
                pass


# ---------------------------------------------------------------- environment
# WHICH SOFTWARE MADE THIS CAPTURE, as the application declares it.
#
# This used to be the repository the library happened to sit in, which was the
# right answer only while there was exactly one of everything. A capture made
# by one application recorded a different repository's commit, which
# describes neither the recorder nor the device.
#
# None until an application says. `git_state()` then reports nulls, which is
# the truth: nothing said what version of what made this.
_REPO: pathlib.Path | None = None


def use_repo(path) -> None:
    """Declare the source tree of the application doing the recording."""
    global _REPO
    _REPO = pathlib.Path(path).resolve() if path else None


def _git(*args: str) -> str | None:
    if _REPO is None:
        return None
    try:
        return subprocess.run(["git", "-C", str(_REPO), *args],
                              capture_output=True, text=True, timeout=5,
                              check=True).stdout.strip()
    except Exception:
        return None


def git_state() -> dict:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return dict(git_commit=commit,
                git_dirty=bool(status) if status is not None else None)


def host_versions() -> dict:
    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None
    return dict(python=sys.version.split()[0], platform=platform.platform(),
                numpy=ver("numpy"), scipy=ver("scipy"), bleak=ver("bleak"))


# The setup this software is being used on, as the APPLICATION declares it.
# None until something declares one, and that is the honest default: this
# library talks to whatever device is connected, and it cannot know what is
# plugged into what, which board revision it is, or what is on anybody's head.
_SITE: dict | None = None


def use_site(source) -> dict:
    """Declare the setup: a dict, or a path to a JSON file holding one.

    Called by the application that knows. A bench tool points this at that
    bench's `config/site.json`; an application that talks to whatever device is
    in front of it declares nothing, and its captures say so.

    This used to be a constant path into one device's repository, read whatever
    was connected. See the note on CAPTURES above for what that cost.
    """
    global _SITE
    if source is None:
        _SITE = None
    elif isinstance(source, dict):
        _SITE = dict(source)
    else:
        _SITE = json.loads(pathlib.Path(source).read_text())
    return site()


def site() -> dict:
    """Facts about the setup that no code can discover: which circuit revision
    is on the bench, what the mains frequency is, what the PSU is set to.

    Unknown is returned as unknown. A capture from a session that declared no
    site says `board_revision: "unknown"` and `electrode_montage: null`, which
    is true, rather than borrowing some other bench's answer, which is not.
    """
    known = dict(board_revision="unknown", mains_hz=None, psu_volts=None,
                 firmware_sha256=None, firmware_artifact=None,
                 electrode_montage=None, subject_id=None, notes="")
    known.update(_SITE or {})
    return known


# ---------------------------------------------------------------- quality
def gap_ledger(index) -> dict:
    """Where the discontinuities are, not just how many.

    Robust to an epoch reset in the middle of a recording: a reset shows as a
    negative index step and is NOT lost data, so it is counted separately and
    never inflates `samples_lost`. Only FORWARD gaps (a jump > 1) are missing
    samples. The old `expected = idx[-1] - idx[0] + 1` produced a negative
    `samples_lost` the moment the index reset (a 494 SPS capture read -68775).
    """
    import numpy as np
    idx = np.asarray(index)
    if idx.size == 0:
        return dict(gaps=[], n_gaps=0, epoch_resets=0, samples_received=0,
                    samples_expected=0, samples_lost=0)
    d = np.diff(idx)
    fwd = np.where(d > 1)[0]                       # real gaps: missing samples
    resets = int(np.sum(d < 0))                    # epoch re-base, not loss
    gaps = [dict(after_sample=int(idx[b]), lost=int(d[b] - 1)) for b in fwd]
    lost = int(sum(g["lost"] for g in gaps))
    received = int(idx.size)
    return dict(gaps=gaps, n_gaps=len(gaps), epoch_resets=resets,
                samples_received=received,
                samples_expected=received + lost,
                samples_lost=lost)


def fs_from_ticks(device_ticks, tick_hz) -> float:
    """Effective sample rate from the DEVICE clock, not the packet index.

    The device time-stamp is monotonic device uptime -- it survives an epoch
    reset that would break an index-based regression. This is the trustworthy
    rate: on the berger_on_g12 capture the index method gave 162 SPS where the
    ticks (and the host clock) agreed on 494.
    """
    import numpy as np
    t = np.asarray(device_ticks, dtype=np.float64)
    if t.size < 2 or not tick_hz:
        return 0.0
    span = (t[-1] - t[0]) / float(tick_hz)
    return (t.size - 1) / span if span > 0 else 0.0


# ---------------------------------------------------------------- manifest
def _montage_fields(site_cfg: dict, dev_channels: int | None, *,
                    layout: dict | None = None,
                    place: dict | None = None) -> dict:
    """The structured montage, plus whether it agrees with the instrument.

    TWO WAYS IN, and the recording says which it used. A caller that knows
    where the electrodes were put passes the device's `layout` and this run's
    `placement`, and the montage is generated from them. A caller that does
    not falls back to whatever the bench's site file declares, which is how
    every capture before 2026-08-20 was made and is why an IntoMind One
    recording from 2026-08-19 carries a two-channel temple montage it was
    never anywhere near.

    A montage describing two channels attached to an eight-channel device is
    not a small inconsistency -- it means the recording and the description of
    it disagree about what was measured, and every per-channel result from it
    is suspect. It is REPORTED rather than raised: refusing here would destroy
    a capture that has already been made, which is a worse outcome than a
    capture that states its own defect. A malformed montage is likewise
    recorded as an error string rather than silently dropped.
    """
    def _ok(m):
        n = montage.channel_count(m)
        return None if dev_channels is None else (n == int(dev_channels))

    if layout is not None and place is not None:
        try:
            lay = _placement.layout_from(layout)
            m = _placement.to_montage(lay, place)
            r = _placement.resolve(lay, place)
        except (_placement.PlacementError, montage.MontageError) as e:
            return dict(montage=None, montage_error=str(e),
                        montage_matches_device=False,
                        montage_source="placement",
                        electrode_layout=layout, placement=None)
        return dict(montage=m, montage_error=None,
                    montage_matches_device=_ok(m),
                    montage_source="placement",
                    electrode_layout=layout, placement=r)
    try:
        m = montage.from_site(site_cfg)
    except montage.MontageError as e:
        return dict(montage=None, montage_error=str(e),
                    montage_matches_device=False, montage_source="site",
                    electrode_layout=None, placement=None)
    if m is None:
        return dict(montage=None, montage_error=None,
                    montage_matches_device=None, montage_source=None,
                    electrode_layout=None, placement=None)
    return dict(montage=m, montage_error=None, montage_matches_device=_ok(m),
                montage_source="site", electrode_layout=None, placement=None)


def montage_backfill() -> dict:
    """Montages asserted after the fact, keyed by capture label."""
    if MONTAGE_BACKFILL.exists():
        try:
            return json.loads(MONTAGE_BACKFILL.read_text())
        except Exception:
            pass
    return {}


def montage_for(label: str, meta: dict | None = None) -> dict:
    """The montage of a capture, and how we come to know it.

    `provenance` is the whole point of this function:

      * `recorded`   -- the capture's own manifest carries it. It was true at
                        the moment of recording and nothing was inferred.
      * `backfilled` -- asserted later from a cited source, and readable, but
                        it is a claim about the past rather than an observation
                        of it. An analysis may use it; a result that turns on
                        it should say so.
      * `unknown`    -- nobody has stated it. Not guessed, not defaulted.

    Pooling `recorded` and `backfilled` without distinction is how a montage
    nobody verified ends up underwriting a per-channel claim.
    """
    if meta is None:
        p = capture_path(label, ".meta.json")
        meta = json.loads(p.read_text()) if p.exists() else {}
    m = (meta or {}).get("montage")
    if m:
        return dict(montage=m, provenance="recorded", source=m.get("source"))
    entry = montage_backfill().get(label)
    if entry:
        return dict(montage=entry.get("montage"), provenance="backfilled",
                    source=entry.get("source"), marked_utc=entry.get("marked_utc"))
    return dict(montage=None, provenance="unknown", source=None)


def signal_chain(dev) -> dict:
    """Which adapter produced this recording, and where it was filtered.

    Reported, never assumed: an adapter that does not identify itself is
    recorded as unknown rather than as ours.
    """
    ad = dict(getattr(dev, "ADAPTER", None)
              or dict(name="unknown", version=None, first_party=False,
                      transport=None))
    f = getattr(dev, "filter", None) or {}
    on = bool(f.get("on"))
    # Where the conditioning happened. Today only a device can filter, so a
    # configured filter means the device did it. An adapter that filters in the
    # application will set this to "host", and a capture will then say that the
    # signal it holds was not the one the device emitted.
    where = getattr(dev, "FILTERS_IN", "device")
    return dict(
        adapter=ad,
        filtering=(where if on else "none"),
        filters=(dict(hp_hz=f.get("hp_hz"), lp_hz=f.get("lp_hz"),
                      bands=[list(b) for b in (f.get("bands") or [])])
                 if on else None),
    )


def signal_source_of(packets) -> dict:
    """What the samples were, off the wire rather than from a setting.

    `signal_source` is the mode the packets declared: "normal" for the
    electrodes, "test" for the converter's test signal, "short" for shorted
    inputs, "synthetic" for a signal the device generated. A device refuses
    a change of mode while it streams, so one run holds one mode. If one
    ever holds more, all of them are listed, in order, rather than one
    chosen. `synthetic` is true when any sample was generated. Both are null
    when no packet was kept.
    """
    modes = sorted({p.mode for p in packets})
    if not modes:
        return dict(signal_source=None, synthetic=None)
    return dict(signal_source=modes[0] if len(modes) == 1 else modes,
                synthetic=any(p.synthetic for p in packets))


def build_manifest(*, label, dev, packets, regs_before, regs_after,
                   protocol, params, extra, index, device_ticks=None,
                   gaps_in_run=0, config_after=None) -> dict:
    s = site()
    # WHERE THE ELECTRODES WERE PUT, if the caller knows. Taken out of `extra`
    # rather than added as arguments so that every existing caller keeps
    # working unchanged and records exactly what it recorded before.
    _layout = extra.pop("electrode_layout", None)
    _place = extra.pop("placement", None)
    # Rate from the device clock (reset-proof); fall back to the packet-index
    # regression only if the ticks are unavailable.
    fs_eff = fs_from_ticks(device_ticks, dev.info.tick_hz) \
        if device_ticks is not None and len(device_ticks) >= 2 \
        else dev.fs_effective(packets)
    # Did the run change the instrument? Registers answer that where a
    # device has a register path, which this contract does not: only an
    # adapter supplies them. Otherwise the answer is the configuration the
    # device reports, read before the run and again after it.
    config_before = extra.get("config_before")
    diff = {}
    if regs_before and regs_after:
        diff = {k: [regs_before.get(k), regs_after.get(k)]
                for k in set(regs_before) | set(regs_after)
                if regs_before.get(k) != regs_after.get(k)}
    elif config_before and config_after:
        diff = {k: [config_before.get(k), config_after.get(k)]
                for k in set(config_before) | set(config_after)
                if config_before.get(k) != config_after.get(k)}
    m = dict(
        manifest_version=MANIFEST_VERSION,
        label=label,
        started_utc=extra.pop("started_utc", None),
        finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        captured_unix=time.time(),
        # device
        device_id=dev.info.device_id, firmware_version=dev.info.fw,
        firmware_sha256=s.get("firmware_sha256"),
        firmware_artifact=s.get("firmware_artifact"),
        tick_hz=dev.info.tick_hz, vref_uv=dev.info.vref_uv,
        # What made this recording. Resolution is a property of the DEVICE, not
        # a constant: a recording that does not state its own resolution cannot
        # be safely compared against one from a different converter.
        adc_bits=getattr(dev.info, "adc_bits", None),
        channels=getattr(dev.info, "channels", None),
        device_proto=getattr(dev.info, "proto", None),
        # circuit + environment: facts only the bench knows
        board_revision=s.get("board_revision"),
        electrode_montage=s.get("electrode_montage"),
        # WHERE THE ELECTRODES WERE, as data rather than as a sentence.
        #
        # `electrode_montage` above is the prose a human wrote and is kept
        # verbatim; `montage` is the same fact in a form an analysis can act
        # on. Index is not identity: `ch1` on a two-channel bipolar layout and
        # `ch1` on an 8-channel referential build are different measurements,
        # and a corpus meant to be re-analyzed across that change cannot be
        # selected by position. A bench that has not been described
        # structurally records null -- unknown, written as unknown.
        **_montage_fields(s, getattr(dev.info, "channels", None),
                          layout=_layout, place=_place),
        mains_hz=s.get("mains_hz"), psu_volts=s.get("psu_volts"),
        subject_id=s.get("subject_id"),
        # config, and whether the run mutated the instrument
        regs_before=regs_before, regs_after=regs_after, regs_diff=diff,
        # What the device reported it was set to, after the run. `regs_*`
        # stay null unless an adapter read registers, which is honest: this
        # contract has no register path.
        config_after=config_after,
        device_config_changed_by_run=bool(diff),
        gain=packets[-1].gain if packets else None,
        rate_sps=packets[-1].sample_rate_hz if packets else None,
        fs_effective=fs_eff,
        # protocol
        protocol_id=protocol, protocol_params=params,
        # WHAT PRODUCED THIS SIGNAL, and where it was conditioned.
        #
        # A capture used to record the device and the firmware but not whether
        # the signal had been filtered, so a recording could not say what it
        # contained. On 2026-08-04 that had to be inferred from the spectrum.
        #
        # `filtering` is where the conditioning happened, which matters more
        # once adapters exist: a device that cannot filter in its own firmware
        # is filtered by the application instead, and then the capture holds
        # something the device never produced. It has to say so.
        signal_chain=signal_chain(dev),
        # Whether the samples were measured at all, and from what.
        **signal_source_of(packets),
        # quality. The ledger is authoritative: it says *where* every
        # discontinuity is. `gaps_announced` is the device's own count for this
        # run only -- never the session-cumulative counter.
        quality=gap_ledger(index),
        gaps_announced=int(gaps_in_run),
        # How much host_time can be trusted. Reported, never assumed.
        clock=(dev.clock_quality() if hasattr(dev, "clock_quality") else None),
        # software
        **git_state(), host_versions=host_versions(),
    )
    # A protocol's parameters may add to the manifest; they may not ERASE a
    # measurement. `gain` and `rate` arrive in every data packet from the
    # device -- they are facts off the wire -- and an extra carrying gain=None
    # used to overwrite one, which is how the first game session ended up with
    # 467k samples it could not be scaled from. A None never wins against a
    # value that was actually measured.
    for k, v in extra.items():
        if v is None and m.get(k) is not None:
            continue
        m[k] = v
    return m


def append_session(manifest: dict) -> None:
    CAPTURES.mkdir(exist_ok=True)
    row = {k: manifest.get(k) for k in
           ("label", "started_utc", "finished_utc", "protocol_id",
            "board_revision", "firmware_version", "gain", "rate_sps",
            "device_config_changed_by_run", "git_commit")}
    row["n"] = manifest.get("n")
    row["samples_lost"] = manifest.get("quality", {}).get("samples_lost")
    with open(SESSIONS, "a") as f:
        f.write(json.dumps(row) + "\n")
