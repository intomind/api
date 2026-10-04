"""One reading, two clocks — so a recording can survive the wall clock moving.

Every timestamp in this project has always been `time.time()`, CLOCK_REALTIME.
That is the right choice for the *record*: it is the only clock that means
anything across processes, across a reboot, or against a file's mtime. The
kernel's input timestamps land on it and so does a capture's `captured_unix`.

It is also the only clock that can silently **jump**. NTP steps it. When it
steps mid-session, every stamp taken afterwards is displaced relative to every
stamp taken before, by an amount nothing in the record states — and worse, the
device→host clock fit in `client.py` is itself regressed against realtime, so a
step does not merely shift the events, it *bends the line that dates every
sample*. A 20-minute cued trial never noticed. An hour of gameplay will.

CLOCK_MONOTONIC cannot jump and cannot run backwards. It also means nothing
outside this boot, so it cannot be the record's timebase on its own.

Neither clock is sufficient and the answer is not to choose between them:

  * **Fit against monotonic.** The device↔host regression uses the monotonic
    reading of each exchange, so a wall-clock step cannot corrupt it.
  * **Anchor to realtime.** Every exchange also records the realtime reading
    taken next to it. The pair (realtime, monotonic) is an *anchor*, and the
    sequence of anchors is what converts a monotonic instant back into the
    realtime the record is written in.
  * **A step is then visible rather than silent.** Between two anchors, realtime
    and monotonic must elapse by the same amount. When they do not, the wall
    clock moved, and the record says by how much and when.

Nothing here needs the step to be *absent*. It needs the step to be *stated*.
"""
from __future__ import annotations

import time

# What separates a wall clock being *disciplined* from a wall clock being
# *stepped*. NTP slews at up to 500 ppm to walk out a small error, so the
# tolerance has to grow with the gap between anchors or a long quiet interval
# reads as a jump. The 50 ms floor covers scheduling jitter between the two
# clock reads themselves; NTP's own step threshold is 128 ms, so a real step
# clears this comfortably at any interval.
SLEW_PPM = 500e-6
STEP_FLOOR_S = 0.050


def stamp() -> dict:
    """Both clocks, read adjacently, as a dict ready to merge into an event.

    Monotonic is read first and realtime second, always in that order, so the
    tiny interval between them is a fixed bias rather than a coin flip.
    """
    m = time.monotonic()
    t = time.time()
    return dict(host_time=t, mono=m)


def anchor() -> tuple[float, float]:
    """One (realtime, monotonic) pair. The unit the mapping is built from."""
    m = time.monotonic()
    t = time.time()
    return (t, m)


def step_tolerance(dt_s: float) -> float:
    """How far realtime may legitimately drift from monotonic across `dt_s`."""
    return STEP_FLOOR_S + SLEW_PPM * abs(dt_s)


def find_steps(anchors) -> list[dict]:
    """Wall-clock steps visible between consecutive anchors.

    `anchors` is a sequence of (realtime, monotonic). Returns one entry per
    detected step, each carrying the monotonic instant the step is bracketed
    by and the size of the jump, so a reader can both locate and undo it.
    """
    out = []
    a = sorted(anchors, key=lambda x: x[1])
    for (t0, m0), (t1, m1) in zip(a, a[1:]):
        drift = (t1 - t0) - (m1 - m0)
        tol = step_tolerance(m1 - m0)
        if abs(drift) > tol:
            out.append(dict(after_mono=m0, before_mono=m1,
                            jump_s=drift, tolerance_s=tol,
                            realtime_before=t0, realtime_after=t1))
    return out


def offset_at(anchors, mono: float) -> float:
    """The realtime−monotonic offset in force at a given monotonic instant.

    Piecewise by construction: the offset that applied at `mono` is the one
    measured by the newest anchor at or before it. Before the first anchor the
    first anchor's offset is used, which is the only honest extrapolation
    available — and it is exact whenever no step occurred.
    """
    if not anchors:
        raise ValueError("no anchors: cannot map monotonic onto realtime")
    a = sorted(anchors, key=lambda x: x[1])
    best = a[0]
    for t, m in a:
        if m <= mono:
            best = (t, m)
        else:
            break
    return best[0] - best[1]


def to_realtime(anchors, mono: float) -> float:
    """A monotonic instant expressed in the record's realtime frame."""
    return mono + offset_at(anchors, mono)


def to_frame(anchors, mono: float, frame_mono: float | None = None) -> float:
    """A monotonic instant in ONE realtime frame — the frame at `frame_mono`.

    `to_realtime` answers "what did the wall clock read at that moment", which
    is what a human wants. This answers "where does this sit on a single
    continuous timeline", which is what alignment wants, because a step means
    those two questions have different answers. Pass the frame anchor (normally
    the start of the recording) and every instant lands on one ruler with the
    jump removed.
    """
    if not anchors:
        raise ValueError("no anchors: cannot map monotonic onto realtime")
    if frame_mono is None:
        frame_mono = min(m for _, m in anchors)
    return mono + offset_at(anchors, frame_mono)


def align_events(meta: dict, events: list[dict]) -> list[dict]:
    """Put every event on the recording's own realtime frame.

    This is the function an analysis calls. It takes a capture's manifest and
    its events, and returns the events with `host_time_aligned` added: the
    instant each event occurred, on the same continuous ruler the sample
    `host_time` column uses, with any wall-clock step undone.

    An event that predates dual-stamping has no `mono` and cannot be corrected;
    it keeps its original `host_time` and is marked `aligned=False` rather than
    silently passed off as corrected. A capture with no steps needs no
    correction and says so by leaving every event `aligned=True` and unchanged.
    """
    clock = (meta or {}).get("clock") or {}
    anchors = [tuple(a) for a in (clock.get("anchors") or [])]
    frame = clock.get("frame_mono")
    out = []
    for e in events:
        e = dict(e)
        m = e.get("mono")
        if m is None or not anchors:
            e["host_time_aligned"] = e.get("host_time")
            e["aligned"] = False
        else:
            e["host_time_aligned"] = to_frame(anchors, m, frame)
            e["aligned"] = True
        out.append(e)
    return out
