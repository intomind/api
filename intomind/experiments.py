"""Experiment registry + runner.

An experiment is an async generator that yields event dicts (status, cue,
progress, saved). The hub streams those to the browser; the CLI prints them.
Every experiment persists raw counts + meta + events BEFORE any analysis,
and records the full configuration (gain, rate, mode, bias drive) in meta.
"""
from __future__ import annotations
import asyncio, json, time, subprocess
import numpy as np

from . import audio
from . import protocol
from . import provenance as prov
from . import timebase
# NOTE: nothing here imports any game library. An application that records
# gameplay is built ON this package, not a part of it, and the
# dependency runs one way only -- an unused `from . import games` sat here and
# would have made the instrument library unimportable without it.

REGISTRY: dict[str, dict] = {}


def experiment(name, doc, params, *, enforced=None, prereg=None):
    """Register a protocol.

    A protocol declares its parameters, an optional `enforced` precondition (a
    genuinely unsafe condition that REFUSES the run), and its `prereg` analysis
    plan. It does NOT declare instrument settings: gain, rate, bias drive and
    signal source describe the instrument and are chosen by the user within
    their control tier -- never imposed by the experiment. A recording captures
    whatever the instrument is currently set to.
    """
    def deco(fn):
        REGISTRY[name] = dict(name=name, doc=doc, params=params, fn=fn,
                              enforced=dict(enforced or {}),
                              prereg=dict(prereg or {}))
        return fn
    return deco


# Every subject-facing protocol requires this. It is the standing order from
# The real guard against a recording being sprung on the operator is structural:
# experiments start only from the Command Center, never the agent, never the CLI.
# A per-run consent token on top of that was pure friction for a self-recording
# tool, so it is gone. Safety
# information (the bias drive is on by default, what it does, current injection on
# contact_check) is surfaced once at connect time by the CC's connection notice.
# This dict is retained empty so `enforced=CONSENT` sites read cleanly.
CONSENT: dict = {}

# THERE ARE NO PRE-FLIGHT GATES, DELIBERATELY.
#
# Every experiment used to declare a `` list -- rld_regulating, contact,
# no_saturation, link_healthy -- and the comment here claimed that "contact and
# rld_regulating are the two that would have caught the failures that actually
# happened."
#
# NOTHING EVALUATED THEM. `prepare_run()` never read `spec["gates"]`; it says so
# in its own docstring ("There is no pre-flight and nothing refuses the run"). The
# list was shipped to the browser by the API and the browser rendered an empty div.
# A safety mechanism that does not run, taking credit for catching failures it
# cannot catch, is worse than no safety mechanism: it is a lie a reader will
# believe. A rule with no check is deleted, not documented.
#
# What actually exists, and is honest about itself: electrode contact is an
# on-demand button (Hub.check_contact), the electrode-off comparators can be left
# on continuously (Hub.enable_contact_detection), and the connection LED is a
# connection indicator. Nothing refuses a recording, and that is deliberate.


def cue_sound(tone: str, text: str = ""):
    """One cue point. `tone` may be 'none' (silent), 'speech', or a tone name."""
    audio.play(tone, text)


def say(text: str):
    """Speak `text` aloud through the host's audio output."""
    audio.play("speech", text)


TONE_PARAMS = [
    dict(name="tone_start", type="select", default="speech",
         choices=audio.CHOICES),
    dict(name="tone_change", type="select", default="speech",
         choices=audio.CHOICES),
    dict(name="tone_end", type="select", default="speech",
         choices=audio.CHOICES),
]


class Recorder:
    """Accumulates raw samples from a Device while armed."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear every buffered sample and packet, and leave the recorder
        disarmed."""
        self.idx, self.dt, self.ht = [], [], []
        self.ch, self.lo, self.gp = [], [], []
        self.packets = []
        self.armed = False
        self.gaps_at_arm = 0

    def arm(self, dev):
        """Snapshot the device's cumulative gap counter, then arm.

        `Device.gaps_announced` counts for the life of the connection, so
        reporting it verbatim made a 20 s capture inherit every gap the session
        had ever seen. Only the delta belongs to this recording, and the
        gap *ledger* -- which says where each discontinuity is -- is the number
        that actually means something.
        """
        self.reset()
        self.gaps_at_arm = getattr(dev, "gaps_announced", 0)
        self.armed = True

    def on_sample(self, _dev, s):
        """Buffer one sample, while armed. Set as a device's `on_sample`
        callback."""
        if not self.armed:
            return
        self.idx.append(s.index); self.dt.append(s.device_time)
        self.ht.append(s.host_time); self.ch.append(s.channels)
        self.lo.append(s.leadoff); self.gp.append(s.gap_before)

    def on_packet(self, _dev, p):
        """Buffer one packet, while armed. Set as a device's `on_packet`
        callback."""
        if self.armed:
            self.packets.append(p)

    def save(self, dev, label, extra_meta, events=None, regs_after=None,
             aborted: str | None = None, inputs=None, config_after=None) -> dict:
        """Persist raw first, then the manifest, then the checksums.

        Checksums are written LAST: their presence is what marks a capture as
        finalized. A run that dies mid-save leaves no .sha256 and is therefore
        never mistaken for a complete capture.
        """
        prov.captures_dir().mkdir(exist_ok=True)
        index = np.array(self.idx, dtype=np.int64)
        ticks = np.array(self.dt, dtype=np.int64)
        counts = np.array(self.ch, dtype=np.int64).T
        np.savez_compressed(
            prov.capture_path(label, ".npz"), counts=counts, index=index,
            device_ticks=ticks,
            host_time=np.array(self.ht, dtype=np.float64),
            leadoff=np.array(self.lo, dtype=np.uint8),
            gap_before=np.array(self.gp, dtype=bool))

        # Controller input goes to its own array file, not into events.json.
        # An analog stick emits a couple of hundred events a second, so an
        # hour of play is most of a million of them: as JSON that is ~100 MB of
        # record. One timebase, two containers, chosen by rate. `inputs()` in
        # analysis.py joins them back into one stream.
        if inputs is not None and len(inputs):
            np.savez_compressed(prov.capture_path(label, ".inputs.npz"),
                                **inputs.arrays())

        p = self.packets
        extra = dict(extra_meta)
        protocol = extra.pop("experiment", None)
        regs_before = extra.pop("regs_before", None)
        meta = prov.build_manifest(
            label=label, dev=dev, packets=p,
            regs_before=regs_before, regs_after=regs_after,
            protocol=protocol, params=dict(extra), extra=extra, index=index,
            device_ticks=ticks, config_after=config_after,
            gaps_in_run=max(0, getattr(dev, "gaps_announced", 0)
                            - self.gaps_at_arm))
        meta["n"] = len(self.idx)
        # What the input stream was, even when it was empty: a session that
        # recorded zero controller events is a different fact from one that
        # never had a controller, and only the summary can tell them apart.
        if inputs is not None:
            meta["inputs"] = inputs.summary()
        # legacy keys: every existing analysis and tool reads these
        meta.setdefault("fw", dev.info.fw)
        meta.setdefault("experiment", protocol)
        meta.setdefault("regs_before", regs_before)

        if aborted:
            meta["aborted"] = aborted
        prov.capture_path(label, ".meta.json").write_text(json.dumps(meta, indent=2))
        if events is not None:
            prov.capture_path(label, ".events.json").write_text(
                json.dumps(events, indent=2))
        # Checksums mark a capture finalized. An aborted run never gets them,
        # so it can never be mistaken for a complete recording.
        if not aborted:
            prov.write_checksums(label)
            prov.freeze(label)
        prov.append_session(meta)
        return meta


async def _record(dev, rec, label, seconds, extra, emit, events=None, spp=25,
                  tick=0.25, ctx=None):
    """Record from the stream that is already running.

    A connected device streams. A recording does not start, stop, or take
    custody of that stream: it only decides whether the samples are *also*
    written to disk. Stopping the stream to record was an ownership model I
    invented, and it is what left the live trace dark whenever a run ended or
    was refused.
    """
    # Establish a clean per-recording epoch BEFORE arming, so the stored index
    # starts at 0 and is monotonic. Resetting the counter does not stop the
    # stream. The old flow armed on the live stream (index at whatever the
    # session had reached) and THEN, trusting a stale `streaming` flag, called
    # start() -- which reset the epoch MID-CAPTURE. That discontinuity poisoned
    # the sample-rate regression (a 494 SPS capture was stamped 162) and the gap
    # accounting (negative "lost").
    was_streaming = bool(getattr(dev, "streaming", False))
    if was_streaming:
        await dev.reset_epoch()
        await asyncio.sleep(0.05)          # let in-flight old-index packets drain
    rec.arm(dev)
    extra = dict(extra)
    extra["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_here = not was_streaming
    if started_here:
        await dev.start(samples_per_packet=spp)   # not streaming: start fresh (resets epoch)
    t0 = time.time()
    while time.time() - t0 < seconds:
        if ctx and ctx.stop.is_set():
            break
        await asyncio.sleep(tick)
        await emit(dict(type="progress", frac=(time.time() - t0) / seconds))
    # Only stop what we started. Otherwise the trace goes dark the moment a
    # recording ends, and the device is still streaming, so that would be a lie.
    link_ok = await _safe_stop(dev) if started_here \
        else bool(getattr(dev, "connected", True))
    await asyncio.sleep(0.3)
    rec.armed = False
    config_after = await _config_after(dev, link_ok)
    # Stamp WHAT signal this capture is: the chain the device ran on it, in
    # the device's own words, or the natural signal. Never a raw-or-filtered
    # flag: a flag says nothing about what a signal is.
    proc = await _processing(dev)
    if proc:
        extra["processing"] = proc
    extra["signal"] = proc["description"] if proc else "unknown"
    meta = rec.save(dev, label, extra, events, config_after=config_after,
                    aborted=None if link_ok else "link_lost")
    await emit(dict(type="saved", label=label, meta=_jsonable(meta)))
    return meta


async def _safe_stop(dev) -> bool:
    """Stop streaming; report whether the link was still alive. A capture is
    only finalized if it was."""
    try:
        await dev.stop()
    except Exception:
        return False
    return bool(getattr(dev, "connected", True))


def _jsonable(d):
    return {k: (None if isinstance(v, float) and v != v else v)
            for k, v in d.items()}


async def _config_after(dev, link_ok: bool) -> dict | None:
    """What the device says it is set to, read after a run.

    This contract has no register path, so there is nothing else to read and
    nothing to fall back to. A link that dropped answers nothing, and the
    manifest records that as nothing rather than as the settings the run was
    asked for.
    """
    if not link_ok:
        return None
    try:
        return await dev.refresh_config()
    except Exception:
        return None


async def _processing(dev) -> dict | None:
    """What the device did to the signal, as the device reports it.

    A device that claims a processing chain answers for itself: the chain in
    force, and whether it was the device's default or a host's. A device
    that claims none streamed its natural signal, unless an adapter for it
    knows better. Only a link failure leaves this unknown.
    """
    can = getattr(dev, "can", None)
    if callable(can) and dev.can("pipeline"):
        try:
            st = await dev.pipeline()
        except Exception:
            return None
        rate = (getattr(dev, "config", None) or {}).get("rate_sps")
        return dict(origin=st.origin, stages=[s.as_dict() for s in st.stages],
                    description=protocol.describe_chain(st.stages, rate))
    extra = getattr(dev, "extra", None)
    if extra is not None and hasattr(extra, "get_filters"):
        try:
            f = await extra.get_filters()
        except Exception:
            return None
        on = bool(f and f.get("on"))
        return dict(origin="adapter", stages=[], adapter=f,
                    description="adapter filters " + str(f) if on else "natural signal")
    return dict(origin="none", stages=[], description="natural signal")


async def _apply(dev, gain, mode, emit):
    """Configure the device over the one control link, then read back what
    it is set to.

    Only a sweep uses this. A normal recording never changes a setting, so
    that what it captured is what the device was already doing.
    """
    cfg = {}
    if gain:
        await dev.set_gain(int(gain)); cfg["gain_requested"] = int(gain)
    if mode:
        await dev.set_mode(protocol.MODES[mode]); cfg["mode"] = mode
    cfg.update(await dev.refresh_config())
    return cfg


def _ts():
    return time.strftime("%Y%m%d-%H%M%S")


class RunCtx:
    """Live handle on the running experiment: stop it, or mark an event.

    `cfg` is filled in by the runner *before* the experiment body executes: the
    instrument has already been configured, read back, and gated. An experiment
    never configures the device it is recording from -- that is what made a
    protocol's stale default silently override the live device state.
    """

    def __init__(self, emit, cfg: dict | None = None):
        self.emit = emit
        self.stop = asyncio.Event()
        self.events: list[dict] = []
        self.cfg: dict = dict(cfg or {})

    # Every event carries BOTH clocks. `host_time` is the record's timebase and
    # `mono` is the witness that says whether the wall clock moved underneath
    # it; `timebase.align_events` needs the pair to put events and samples on
    # one ruler. See timebase.py for why neither clock alone is enough.
    def marker(self, name: str, **kw) -> dict:
        """Record a timestamped marker event named `name` and return it."""
        e = dict(cond=name, kind="marker", **timebase.stamp(), **kw)
        self.events.append(e)
        return e

    def cue(self, cond: str, **kw) -> dict:
        """Record a timestamped cue event for condition `cond` and return
        it."""
        e = dict(cond=cond, kind="cue", **timebase.stamp(), **kw)
        self.events.append(e)
        return e

    def event(self, kind: str, cond: str, **kw) -> dict:
        """A stamped event of any kind — the path external sources use.

        `marker` and `cue` are the operator's and the protocol's. A controller
        button, an emulator state change, anything arriving from outside the
        protocol comes through here, so every source lands in one stream with
        one timebase rather than each inventing its own.
        """
        e = dict(cond=cond, kind=kind, **timebase.stamp(), **kw)
        self.events.append(e)
        return e


RLD_CHOICES = ["on", "loop_open", "off"]   # canonical names only


def _cfg(ctx) -> dict:
    """The run's provenance block: what the instrument actually was."""
    return dict(ctx.cfg)


def _lbl(ctx, prefix: str) -> str:
    c = ctx.cfg
    return f"{prefix}_{c.get('rld', 'na')}_g{c.get('gain', 'na')}_{_ts()}"

# ---------------------------------------------------------------- experiments

@experiment("baseline", "Fixed-duration resting capture.",
            [dict(name="seconds", type="number", default=30),
             dict(name="note", type="text", default="")] + TONE_PARAMS,
            enforced=CONSENT,)
async def baseline(dev, rec, emit, p, ctx):
    """A fixed-duration resting recording."""
    label = _lbl(ctx, "baseline")
    await emit(dict(type="note", text=f"recording {p['seconds']}s → {label}"))
    cue_sound(p.get("tone_start", "speech"), "Recording.")
    await _record(dev, rec, label, float(p["seconds"]),
                  dict(experiment="baseline", note=p.get("note", ""), **_cfg(ctx)),
                  emit, events=ctx.events, ctx=ctx)
    cue_sound(p.get("tone_end", "speech"), "Done.")
    return [label]


@experiment("berger", "Cued eyes-closed / eyes-open alpha test.",
            [dict(name="block", type="number", default=20),
             dict(name="reps", type="number", default=8),
             dict(name="settle", type="number", default=3)] + TONE_PARAMS,
            enforced=CONSENT,
            prereg=dict(name="berger", alpha_band=[8.0, 13.0],
                        reference_band=[4.0, 30.0], epoch_s=2.0,
                        reject_pp_uv=150.0, test="wilcoxon_paired_blocks",
                        allow_pseudoreplication=False))
async def berger(dev, rec, emit, p, ctx):
    """Cued eyes-closed and eyes-open blocks, alternating for a set number
    of repetitions."""
    cfg = _cfg(ctx)
    label = _lbl(ctx, "berger")
    block, reps = float(p["block"]), int(p["reps"])
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec.arm(dev)
    await dev.start(samples_per_packet=25)
    try:
        cue_sound(p.get("tone_start", "speech"),
                  "Starting. Sit still, breathe normally.")
        await emit(dict(type="cue", text="Sit still — starting in 6 s"))
        await asyncio.sleep(6)
        total = reps * 2 * block
        done = 0.0
        for r in range(reps):
            for cond in ("closed", "open"):
                txt = "Close your eyes" if cond == "closed" else "Open your eyes"
                cue_sound(p.get("tone_change", "speech"), txt)
                ctx.cue(cond, rep=r)
                await emit(dict(type="cue", text=f"{txt}  (rep {r+1}/{reps})"))
                t0 = time.time()
                while time.time() - t0 < block and not ctx.stop.is_set():
                    await asyncio.sleep(0.25)
                    await emit(dict(type="progress",
                                    frac=(done + time.time() - t0) / total))
                done += block
        cue_sound(p.get("tone_end", "speech"), "Done. You can relax.")
        await emit(dict(type="cue", text="Done — relax"))
    finally:
        link_ok = await _safe_stop(dev)
        await asyncio.sleep(0.3)
        rec.armed = False
        config_after = await _config_after(dev, link_ok)
        meta = rec.save(dev, label, dict(
            experiment="berger", block=block, reps=reps, started_utc=started,
            settle=float(p["settle"]), **cfg), ctx.events, config_after=config_after,
            aborted=None if link_ok else "link_lost")
        await emit(dict(type="saved", label=label, meta=_jsonable(meta)))
    return [label]


# Three protocols that existed to vary a register (the bias ladder, the
# contact sweep, and the input multiplexer matrix) left this library on
# 2026-09-19. Each one's declared purpose was to characterize one
# converter, which is knowledge that belongs to an adapter and not to a
# library whose rule is that the device is the authority on what the
# device is. They are registered by that adapter if it is installed, so a
# bench that needs them still has them, and a product does not carry them.

@experiment("record",
            "Open-ended recording. Runs until you press Stop. Use the marker "
            "buttons to timestamp events as they happen.",
            [dict(name="markers", type="text",
                  default="event 1,event 2,event 3,event 4,event 5"),
             dict(name="max_seconds", type="number", default=600),
             dict(name="note", type="text", default="")] + TONE_PARAMS,
            enforced=CONSENT,
            # No `contact` gate: a free-form recording with the electrodes off
            # is a legitimate thing to want, and blocking it would be the same
            # over-reach as forcing `rld: on`.
)
async def record(dev, rec, emit, p, ctx):
    """An open-ended recording that runs until stopped, with marker
    buttons to timestamp events as they happen."""
    cfg = _cfg(ctx)
    label = _lbl(ctx, "record")
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    await emit(dict(type="note", text=f"recording → {label} (Stop to end)"))
    await emit(dict(type="markers",
                    names=[m.strip() for m in str(p["markers"]).split(",")
                           if m.strip()]))
    rec.arm(dev)
    await dev.start(samples_per_packet=25)
    cue_sound(p.get("tone_start", "speech"), "Recording.")
    t0 = time.time()
    limit = float(p["max_seconds"])
    try:
        while not ctx.stop.is_set() and time.time() - t0 < limit:
            await asyncio.sleep(0.25)
            await emit(dict(type="elapsed", seconds=time.time() - t0,
                            markers=len(ctx.events)))
    finally:
        link_ok = await _safe_stop(dev)
        await asyncio.sleep(0.3)
        rec.armed = False
        cue_sound(p.get("tone_end", "speech"), "Done.")
        config_after = await _config_after(dev, link_ok)
        meta = rec.save(dev, label, dict(
            experiment="record", note=p.get("note", ""), started_utc=started,
            duration_s=time.time() - t0, **cfg), ctx.events, config_after=config_after,
            aborted=None if link_ok else "link_lost")
        await emit(dict(type="saved", label=label, meta=_jsonable(meta)))
    return [label]


@experiment("protocol",
            "Custom cued protocol. One step per line: 'cue text | seconds'. "
            "Repeated 'reps' times. Each cue is spoken and timestamped.",
            [dict(name="steps", type="textarea",
                  default="Close your eyes | 20\nOpen your eyes | 20"),
             dict(name="reps", type="number", default=3),
             dict(name="settle", type="number", default=3),
             dict(name="lead_in", type="number", default=6),
             dict(name="note", type="text", default="")] + TONE_PARAMS,
            enforced=CONSENT,)
async def cued_protocol(dev, rec, emit, p, ctx):
    """A custom cued protocol: one step per line, each spoken and
    timestamped, repeated for a set number of repetitions."""
    steps = []
    for line in str(p["steps"]).splitlines():
        if not line.strip():
            continue
        if "|" not in line:
            raise ValueError(f"step needs 'cue | seconds': {line!r}")
        text, secs = line.rsplit("|", 1)
        steps.append((text.strip(), float(secs)))
    if not steps:
        raise ValueError("no steps")

    cfg = _cfg(ctx)
    label = _lbl(ctx, "protocol")
    reps = int(p["reps"])
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec.arm(dev)
    await dev.start(samples_per_packet=25)
    try:
        cue_sound(p.get("tone_start", "speech"), "Starting.")
        await emit(dict(type="cue", text=f"Starting in {p['lead_in']} s"))
        await asyncio.sleep(float(p["lead_in"]))
        total = reps * sum(s for _, s in steps)
        done = 0.0
        for r in range(reps):
            for text, secs in steps:
                if ctx.stop.is_set():
                    break
                cue_sound(p.get("tone_change", "speech"), text)
                ctx.cue(text, rep=r, planned_s=secs)
                await emit(dict(type="cue", text=f"{text}  (rep {r+1}/{reps})"))
                t0 = time.time()
                while time.time() - t0 < secs and not ctx.stop.is_set():
                    await asyncio.sleep(0.25)
                    await emit(dict(type="progress",
                                    frac=(done + time.time() - t0) / total))
                done += secs
        cue_sound(p.get("tone_end", "speech"), "Done.")
        await emit(dict(type="cue", text="Done"))
    finally:
        link_ok = await _safe_stop(dev)
        await asyncio.sleep(0.3)
        rec.armed = False
        config_after = await _config_after(dev, link_ok)
        meta = rec.save(dev, label, dict(
            experiment="protocol", steps=steps, reps=reps, started_utc=started,
            settle=float(p["settle"]), block=max(s for _, s in steps),
            note=p.get("note", ""), **cfg), ctx.events, config_after=config_after,
            aborted=None if link_ok else "link_lost")
        await emit(dict(type="saved", label=label, meta=_jsonable(meta)))
    return [label]


async def _cued_trials(dev, rec, emit, p, ctx, *, label, conds, texts,
                       lead_in=6.0, rest_s=2.0, hold_s=2.0, intro="",
                       rest_text="Rest", extra=None):
    """One cue per trial, timestamped at the cue instant, with rest between.

    Shared by every event-locked validation protocol. The cue order is shuffled
    with a fixed seed and written into the manifest, because the order is part
    of the record: a subject who can predict the next cue is a different subject.

    BOTH EDGES ARE SPOKEN. `rest_text` is voiced at the start of every rest, the
    way `saccade` voices "Look at the center". Without it the subject hears
    "Clench your jaw", silence, "Clench your jaw" -- nothing ever says stop, so
    the obvious reading is to hold the action throughout, which destroys the
    contrast the analysis is built on. That happened on 2026-08-04 and the run
    was unusable. A cue that starts a state must have a cue that ends it.
    """
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec.arm(dev)
    await dev.start(samples_per_packet=25)
    try:
        cue_sound(p.get("tone_start", "speech"), intro)
        await emit(dict(type="cue", text=f"{intro}  starting in {lead_in:.0f} s"))
        await asyncio.sleep(lead_in)

        for i, cond in enumerate(conds):
            if ctx.stop.is_set():
                break
            cue_sound(p.get("tone_change", "speech"), rest_text)
            await emit(dict(type="cue", text=f"{rest_text}  ({i+1}/{len(conds)})"))
            t0 = time.time()
            while time.time() - t0 < rest_s and not ctx.stop.is_set():
                await asyncio.sleep(0.05)

            cue_sound(p.get("tone_change", "speech"), texts[cond])
            ctx.cue(cond, trial=i)          # timestamped at the cue instant
            await emit(dict(type="cue", text=f"{texts[cond]}  ({i+1}/{len(conds)})"))
            t0 = time.time()
            while time.time() - t0 < hold_s and not ctx.stop.is_set():
                await asyncio.sleep(0.05)
                await emit(dict(type="progress", frac=(i + 1) / len(conds)))

        cue_sound(p.get("tone_end", "speech"), "Done. You can relax.")
        await emit(dict(type="cue", text="Done"))
    finally:
        link_ok = await _safe_stop(dev)
        await asyncio.sleep(0.3)
        rec.armed = False
        config_after = await _config_after(dev, link_ok)
        meta = rec.save(dev, label, dict(
            started_utc=started, order=list(conds), rest_s=rest_s,
            hold_s=hold_s, note=p.get("note", ""), **(extra or {}), **_cfg(ctx)),
            ctx.events, config_after=config_after,
            aborted=None if link_ok else "link_lost")
        await emit(dict(type="saved", label=label, meta=_jsonable(meta)))
    return [label]


@experiment("blink",
            "V3 validation: cued blinks. Frontopolar electrodes see a 100-300 "
            "uV blink. Sham trials ('stay still') control for the cue itself.",
            [dict(name="trials", type="number", default=30),
             dict(name="hold_s", type="number", default=1.5),
             dict(name="rest_s", type="number", default=2.0),
             dict(name="include_sham", type="select", default=1, choices=[0, 1]),
             dict(name="note", type="text", default="")] + TONE_PARAMS,
            enforced=CONSENT,
            prereg=dict(name="blink", pre_s=0.5, post_s=1.5, hp=0.1, lp=10.0,
                        min_pp_uv=100.0, min_detect_frac=0.9,
                        max_latency_sd_s=0.25, sham_must_be_null=True))
async def blink(dev, rec, emit, p, ctx):
    """Cued blink trials, with interleaved sham trials that control for
    the cue itself."""
    import random
    trials = int(p["trials"])
    conds = ["blink"] * trials
    if int(p["include_sham"]):
        conds += ["sham"] * max(2, trials // 4)
    random.Random(24601).shuffle(conds)
    return await _cued_trials(
        dev, rec, emit, p, ctx, label=_lbl(ctx, "blink"), conds=conds,
        texts={"blink": "Blink", "sham": "Stay still"},
        rest_s=float(p["rest_s"]), hold_s=float(p["hold_s"]), rest_text="Rest",
        intro="Blink test. Blink once, hard, when told, then rest.",
        extra=dict(experiment="blink", trials=trials))


@experiment("jaw_clench",
            "V4 validation: cued jaw clench. The temporal poles sit over "
            "temporalis, so a clench is broadband EMG an order of magnitude "
            "above baseline in 20-200 Hz.",
            [dict(name="trials", type="number", default=20),
             dict(name="hold_s", type="number", default=2.0),
             dict(name="rest_s", type="number", default=3.0),
             dict(name="include_sham", type="select", default=1, choices=[0, 1]),
             dict(name="note", type="text", default="")] + TONE_PARAMS,
            enforced=CONSENT,
            prereg=dict(name="jaw", window_s=1.0, lo_hz=20.0, hi_hz=200.0,
                        min_ratio=10.0, alpha=0.001,
                        test="wilcoxon_paired_trials"))
async def jaw_clench(dev, rec, emit, p, ctx):
    """Cued jaw-clench trials, with interleaved sham trials that control
    for the cue itself."""
    import random
    trials = int(p["trials"])
    conds = ["clench"] * trials
    if int(p["include_sham"]):
        conds += ["sham"] * max(2, trials // 4)
    random.Random(31337).shuffle(conds)
    return await _cued_trials(
        dev, rec, emit, p, ctx, label=_lbl(ctx, "jaw"), conds=conds,
        texts={"clench": "Clench your jaw", "sham": "Stay still"},
        rest_s=float(p["rest_s"]), hold_s=float(p["hold_s"]), rest_text="Relax",
        intro="Jaw clench test. Clench when told, and relax when told.",
        extra=dict(experiment="jaw_clench", trials=trials))


@experiment("saccade",
            "V2 validation: cued left/right eye movements. An eye is a dipole, "
            "so L-temple and R-temple must deflect in OPPOSITE directions, and "
            "the sign must reverse with direction. Nothing else can fake that.",
            [dict(name="trials", type="number", default=20),
             dict(name="hold_s", type="number", default=2.0),
             dict(name="center_s", type="number", default=2.0),
             dict(name="include_sham", type="select", default=1, choices=[0, 1]),
             dict(name="note", type="text", default="")] + TONE_PARAMS,
            enforced=CONSENT,
            # Fixed before the first confirmatory run, and stamped into the
            # manifest. Changing any of these after seeing the data
            # makes the result exploratory, and it will say so.
            prereg=dict(name="saccade", window_s=[0.1, 0.9], baseline_s=[-0.5, 0.0],
                        min_deflection_uv=50.0, require_opposite_polarity=True,
                        require_sign_reversal=True, sham_must_be_null=True,
                        test="wilcoxon_paired_trials"))
async def saccade(dev, rec, emit, p, ctx):
    """Left/right saccades, with an interleaved sham condition.

    The sham cue ("stay still") is the control for the cue itself: the speaker,
    the audio driver, and any cue-locked electrical artifact. If a deflection
    appears on sham trials, the "saccade" response is not ocular.
    """
    import random
    cfg = _cfg(ctx)
    label = _lbl(ctx, "saccade")
    trials = int(p["trials"])
    hold = float(p["hold_s"])
    # `centre_s` until 2026-08-20. Three captures carry the old
    # spelling and a saved form may still send it, so it is
    # accepted on the way in and never written on the way out.
    center = float(p.get("center_s", p.get("centre_s", 2.0)))
    sham = bool(int(p["include_sham"]))

    conds = (["left", "right"] * ((trials + 1) // 2))[:trials]
    if sham:
        conds += ["sham"] * max(2, trials // 4)
    rnd = random.Random(12345)          # fixed: the order is part of the record
    rnd.shuffle(conds)

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec.arm(dev)
    await dev.start(samples_per_packet=25)
    try:
        cue_sound(p.get("tone_start", "speech"),
                  "Saccade test. Keep your head still. Move only your eyes.")
        await emit(dict(type="cue", text="Head still — starting in 6 s"))
        await asyncio.sleep(6)

        for i, cond in enumerate(conds):
            if ctx.stop.is_set():
                break
            cue_sound(p.get("tone_change", "speech"), "Look at the center")
            await emit(dict(type="cue", text=f"Center  ({i+1}/{len(conds)})"))
            t0 = time.time()
            while time.time() - t0 < center and not ctx.stop.is_set():
                await asyncio.sleep(0.1)

            text = {"left": "Look left", "right": "Look right",
                    "sham": "Stay still"}[cond]
            cue_sound(p.get("tone_change", "speech"), text)
            ctx.cue(cond, trial=i)          # timestamped at the cue instant
            await emit(dict(type="cue", text=f"{text}  ({i+1}/{len(conds)})"))
            t0 = time.time()
            while time.time() - t0 < hold and not ctx.stop.is_set():
                await asyncio.sleep(0.1)
                await emit(dict(type="progress", frac=(i + 1) / len(conds)))

        cue_sound(p.get("tone_end", "speech"), "Done. You can relax.")
        await emit(dict(type="cue", text="Done"))
    finally:
        link_ok = await _safe_stop(dev)
        await asyncio.sleep(0.3)
        rec.armed = False
        config_after = await _config_after(dev, link_ok)
        meta = rec.save(dev, label, dict(
            experiment="saccade", trials=trials, hold_s=hold, center_s=center,
            include_sham=sham, order=conds, started_utc=started,
            note=p.get("note", ""), **cfg), ctx.events, config_after=config_after,
            aborted=None if link_ok else "link_lost")
        await emit(dict(type="saved", label=label, meta=_jsonable(meta)))
    return [label]
