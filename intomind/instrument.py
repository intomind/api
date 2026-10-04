"""Instrument layer: explicit state, contact, register decoding.

**State.** `disconnected -> connected -> configured -> armed -> streaming ->
stopped`. The device is the only source of truth for its own state: a setting
is written, read back, and compared. A recording that starts in an unknown
configuration is a recording that cannot be interpreted, and we made several
of those.

**No gates.** There is no pre-flight. Nothing checks the instrument before a
recording and nothing refuses one. Press Run and it records. Electrode contact is
an on-demand check the operator asks for, not a background service: it works by
pushing current through the electrodes, which disturbs the signal they are
measuring.

**Experiments never set instrument settings.** The settings the user configured
-- within their control tier -- are exactly what a recording uses. An experiment
records at the current configuration and stamps it into the manifest; it does
not apply, correct, or override anything. The only precondition mechanism left
is `enforced`, which REFUSES an unsafe run rather than reconfiguring the device.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------- contact
# Contact comes from the device's own electrode-off detection, carried in the
# stream, never from a DC heuristic.
#
# The old test was `|DC| >= 1 mV`, on the theory that skin gives a half-cell
# offset and a floating input sits near zero. That theory is wrong in both
# directions. A floating input is high impedance: it drifts and routinely parks
# above 1 mV, so electrodes lying on the table read as "on skin". Touch one and
# you inject charge that decays slowly through the 10k series resistor and the
# input caps, so it keeps reading "on" long after your finger is gone. That was
# observed exactly this. The heuristic is deleted.
#
# Lead-off detection is electrode-off detection: the device pushes a small
# current into each input, an electrode that is off cannot sink it, the input
# is driven to a rail, and a comparator flags it. The result rides in every
# sample packet, so contact is already streaming, per electrode.
#
# The contract specifies `loff_statp`: bit n is channel n+1, the positive side
# only, one bit per channel, for as many channels as the device has. The
# negative side is not on the wire, so it is not reported. A SET bit means
# that electrode is OFF.
#
# A device whose own packing differs is decoded by its adapter, not here. The
# byte's values are indistinguishable between packings, so reading one as the
# other reports the wrong electrodes and says nothing about it. This library
# decodes the contract it publishes, and nothing else.

# Contact must hold for most of the window, not for one lucky sample: a single
# comparator blip should not flip the indicator.
CONTACT_MIN_FRACTION = 0.75


def contact_channels(dev, default: int = 2) -> int:
    """How many channels to decode contact for -- the device's own count."""
    return int(getattr(getattr(dev, "info", None), "channels", None) or default)


def contact_from_leadoff(loff, n_channels: int = 2) -> list[bool]:
    """Per-channel contact, decoded from the lead-off bits in the stream.

    `loff` is one status byte or an array of them, one per sample. A channel
    has contact when the electrode the contract reports for it is on for at
    least CONTACT_MIN_FRACTION of the samples.
    """
    a = np.atleast_1d(np.asarray(loff, dtype=np.uint8))
    if a.size == 0:
        return [False] * n_channels
    out = []
    for ch in range(n_channels):
        mask = 1 << ch          # one bit per channel, positive side only
        out.append(bool(((a & mask) == 0).mean() >= CONTACT_MIN_FRACTION))
    return out



async def quiesce(dev) -> None:
    """Force a known state on a freshly-connected device.

    A device left mid-capture by a dropped link should come to rest before a
    new session drives it.

    And lead-off is turned OFF, explicitly. It is a device setting, not a
    host one: it survives a host restart and a reconnect. The detection
    current polarizes a dry electrode, adds DC drift, and degrades
    common-mode rejection. "Force a known state" has to mean it.

    Lives here, in the product, because BOTH the Command Center and the laboratory
    need it and a second copy would drift.
    """
    try:
        await dev.stop()
    except Exception:
        pass          # ST_NOT_STREAMING is the expected happy path
    from intomind.client import OP
    try:
        await dev.command(OP["SET_LEADOFF"], 0)
    except Exception as e:
        print(f"could not clear lead-off: {e}")


class StateError(RuntimeError):
    """An operation was attempted from a state that does not allow it."""


# ---------------------------------------------------------------- setting tiers
#
# The tiers gate what the user may CHANGE, in the Command Center UI; they no
# longer let an experiment impose settings. An experiment never sets instrument
# settings -- the settings the user configured, within their tier, are exactly
# what a recording uses. (This retired the `advised`/`resolve_advised`
# auto-apply, where running a protocol silently reconfigured the hardware to the
# protocol's opinion: "the settings the user sets are the settings they
# get when they hit record.")
TIERS = ("basic", "intermediate", "advanced")


class EnforcedViolation(RuntimeError):
    """A genuinely unsafe precondition refuses the run at every tier. This is
    NOT a setting an experiment imposes -- it refuses, it never reconfigures."""


def check_enforced(spec: dict, params: dict) -> None:
    """`enforced` preconditions refuse at every tier, Advanced included.

    Retained as a mechanism; the registry declares nothing enforced today. If a
    future protocol ever needs a truly unoverridable precondition, declaring it
    `enforced` still refuses here.
    """
    for key, requirement in (spec.get("enforced") or {}).items():
        got = params.get(key)
        if requirement == "required" and not got:
            raise EnforcedViolation(
                f"'{key}' is required for '{spec['name']}' and cannot be "
                f"overridden at any tier")
        if requirement not in ("required",) and got != requirement:
            raise EnforcedViolation(
                f"'{key}' must be {requirement!r} for '{spec['name']}'")


def effective_settings(spec: dict, actual: dict, applied: dict,
                       params: dict) -> dict:
    """Every setting's effective value, for the manifest.

    Invariant: a run must be reconstructible from its manifest alone. The values
    are the instrument's ACTUAL configuration (read from silicon), plus the
    experiment's parameters -- never a default the experiment wished for.
    """
    eff = dict(actual)
    eff.update(applied)
    for key, value in params.items():
        eff[f"param.{key}"] = value
    return eff
