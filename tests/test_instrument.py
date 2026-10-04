"""Tests for the instrument layer: gates and setting classes.

    python3 tests/test_instrument.py

No pytest, no network, no device. Every test names the incident it guards
against; several of them would have failed on 2026-07-09 and caught a defect
that instead cost two days and a set of confounded recordings.
"""
from __future__ import annotations
import asyncio, inspect, pathlib, sys, traceback

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from intomind import instrument as I          # noqa: E402

FS = 500.0


def approx(a, b, tol=1e-6):
    assert abs(a - b) < tol, f"{a} != {b}"


# ---------------------------------------------------------------- gates












SPEC = dict(name="berger", enforced={"consent": "required"})

def test_enforced_refuses_without_consent():
    """The standing order, as code: no measurement without an explicit go."""
    try:
        I.check_enforced(SPEC, {})
    except I.EnforcedViolation as e:
        assert "consent" in str(e)
        return
    raise AssertionError("a run without consent was permitted")


def test_enforced_refuses_at_every_tier():
    for tier in I.TIERS:
        try:
            I.check_enforced(SPEC, {"tier": tier})
        except I.EnforcedViolation:
            continue
        raise AssertionError(f"tier {tier} bypassed an enforced setting")


def test_enforced_passes_with_consent():
    I.check_enforced(SPEC, {"consent": {"who": "ada", "when": "now"}})


def test_the_advice_auto_apply_is_gone():
    """Experiments never set settings. The resolve_advised / AdviceViolation
    machinery that silently reconfigured the hardware to a protocol's opinion
    is removed, root and branch."""
    assert not hasattr(I, "resolve_advised")
    assert not hasattr(I, "AdviceViolation")
    assert not hasattr(I, "Advice")


def test_effective_settings_records_the_actual_configuration():
    """The manifest records what the instrument IS, plus the experiment's
    parameters -- never a default the experiment wished for."""
    eff = I.effective_settings(SPEC, {"rld": "off", "mode": "normal", "gain": 6},
                               {}, {"reps": 8})
    assert eff["rld"] == "off"             # the ACTUAL setting, not an opinion
    assert eff["gain"] == 6                # from silicon
    assert eff["param.reps"] == 8          # the experiment's parameter
    assert eff["param.reps"] == 8


# ---------------------------------------------------------------- preflight
class FakeShell:
    available = True

    def __init__(self, mv=1657.0):
        self.mv = mv

    async def arun(self, cmds):
        return [(cmds[0], f"RLDOUT = {int(self.mv)} mV  (n=16)")]



def test_no_protocol_takes_gain_or_rld_as_a_parameter():
    """Device state is not an experiment parameter. A stale protocol
    default silently overrode the live device configuration and produced a
    berger run with the right-leg drive down."""
    from intomind import experiments as X
    for name, spec in X.REGISTRY.items():
        names = {p["name"] for p in spec["params"]}
        assert "gain" not in names, f"{name} still takes gain"
        assert "rld" not in names, f"{name} still takes rld"


def test_no_protocol_declares_a_per_run_enforced_setting():
    """The per-run consent/leadoff gates are gone. The guard against a
    recording being sprung on the operator is structural -- runs start only from
    the Command Center -- not a per-run token."""
    from intomind import experiments as X
    for name, spec in X.REGISTRY.items():
        assert not spec["enforced"], \
            f"{name} still declares an enforced per-run setting: {spec['enforced']}"


def test_no_protocol_declares_a_setting_opinion():
    """Experiments never set settings, so none declares advised/free settings.
    A protocol that needs the bias drive on documents that for the user; it does
    not silently force it: the settings the user sets are what they get."""
    from intomind import experiments as X
    for name, spec in X.REGISTRY.items():
        assert "advised" not in spec, f"{name} still declares advised settings"
        assert "free" not in spec, f"{name} still declares free settings"



def test_every_cued_state_has_a_spoken_end():
    """A cue that starts a state must have a spoken cue that ends it.

    `_cued_trials` used to voice only the action and leave the rest period
    silent, so a jaw_clench run sounded like "Clench your jaw", silence,
    "Clench your jaw" -- nothing said stop, the subject held the clench for the
    whole run, and the clench/sham contrast the analysis needs was gone. The
    text was on screen, which is no use to someone following a voice."""
    import inspect
    from intomind import experiments as X
    src = inspect.getsource(X._cued_trials)
    body = src[src.index("for i, cond in enumerate(conds)"):]
    rest = body[:body.index("ctx.cue(")]
    assert "cue_sound(" in rest, \
        "_cued_trials no longer speaks anything during the rest period"
    assert "rest_text" in inspect.signature(X._cued_trials).parameters, \
        "_cued_trials lost its rest_text cue"
    # and every protocol that uses it must supply one
    for name in ("blink", "jaw_clench"):
        fn_src = inspect.getsource(X.REGISTRY[name]["fn"])
        assert "rest_text=" in fn_src, f"{name} does not supply a spoken rest cue"


def test_confirmatory_protocols_carry_a_preregistered_analysis():
    from intomind import experiments as X
    for name in ("berger", "saccade"):
        pr = X.REGISTRY[name]["prereg"]
        assert pr and pr.get("test"), f"{name} has no preregistered analysis"


def _sync(host, device, rtt, mono=None):
    """One exchange. `mono` defaults to `host` — the no-step case."""
    from intomind.client import Sync
    return Sync(host=host, device=device, rtt=rtt,
                mono=host if mono is None else mono)


def _dev_with_syncs(syncs):
    from intomind.client import Device
    d = Device.__new__(Device)
    d._syncs = [s if hasattr(s, "mono") else _sync(*s) for s in syncs]
    d._skew, d._offset = 1.0, 0.0
    d._frame_offset, d._frame_mono = 0.0, None
    d._fit_clock()
    return d


def test_clock_single_sync_has_no_slope_and_no_residual():
    """One exchange cannot measure skew. It must not pretend to."""
    d = _dev_with_syncs([(1000.0, 10.0, 0.004)])
    q = d.clock_quality()
    assert q["n_syncs"] == 1 and q["skew"] == 1.0
    assert q["residual_ms"] is None
    approx(q["max_half_rtt_ms"], 2.0, 1e-9)


def test_clock_recovers_a_known_skew():
    """mono = 1.000_02 * device + 500, sampled cleanly."""
    skew, off = 1.00002, 500.0
    syncs = [(skew * d + off, float(d), 0.004) for d in (0, 30, 60, 90, 120)]
    q = _dev_with_syncs(syncs).clock_quality()
    approx(q["skew"], skew, 1e-9)
    approx(q["offset"], off, 1e-6)
    assert q["residual_ms"] < 1e-6
    approx(q["span_s"], 120.0)


def test_clock_residual_reports_scatter():
    """Perturb one exchange by 5 ms; the residual must see it."""
    syncs = [(float(d), float(d), 0.004) for d in (0, 30, 60, 90)]
    syncs[2] = (syncs[2][0] + 0.005, syncs[2][1], 0.004)
    q = _dev_with_syncs(syncs).clock_quality()
    assert 1.0 < q["residual_ms"] < 5.0, q["residual_ms"]


def test_clock_residual_is_not_derivable_from_a_capture():
    """host_time in a capture is host_time(device_ticks) -- the map itself.
    Regressing one against the other returns exactly zero, which is why the
    residual must come from the sync exchanges and not from the stored arrays."""
    d = _dev_with_syncs([(0.0, 0.0, 0.004), (60.0, 60.0, 0.004)])
    d.info = type("I", (), {"tick_hz": 32768.0})()
    ticks = [0, 32768, 65536, 98304]
    hosts = [d.host_time(t) for t in ticks]
    pred = [d._skew * (t / 32768.0) + d._offset for t in ticks]
    assert all(abs(h - p) < 1e-12 for h, p in zip(hosts, pred))


def test_a_slope_fitted_across_a_second_of_jitter_is_refused():
    """`calibrate_clock` spaces six exchanges 0.15 s apart, and a BLE round
    trip is jittery by tens of milliseconds. Least squares will draw a line of
    any slope through that: one four-channel session fitted skew = 1.6986 and
    stamped 1945 s of recording across 3304 s, with nothing saying the number
    was unusable. A second cannot measure parts per million, so it must not
    be allowed to try."""
    jitter = [0.0, 0.09, -0.06, 0.05, -0.08, 0.10]
    syncs = [_sync(host=d, device=d, rtt=0.06, mono=d + j)
             for d, j in zip((0.0, 0.15, 0.30, 0.45, 0.60, 0.75), jitter)]
    q = _dev_with_syncs(syncs).clock_quality()
    assert q["skew"] == 1.0, q["skew"]
    assert q["skew_pinned"] is True
    assert q["skew_pinned_why"], "pinned without saying why"


def test_a_slope_measured_across_minutes_is_kept():
    """The refusal is about the BASELINE, not about slopes. Exchanges spread
    across a recording -- which is what `Session.resync_loop` produces -- can
    measure a crystal, and then the measurement is used."""
    skew = 1.00004
    syncs = [_sync(host=skew * d, device=float(d), rtt=0.06,
                   mono=skew * d + j)
             for d, j in zip(range(0, 1800, 120),
                             [0.002, -0.001, 0.0015, -0.002] * 4)]
    q = _dev_with_syncs(syncs).clock_quality()
    assert q["skew_pinned"] is False, q["skew_pinned_why"]
    approx(q["skew"], skew, 1e-6)


def test_a_physically_impossible_slope_is_refused_whatever_its_error_bar():
    """Two oscillators differ by parts per million. A clean fit of 1.7 is a
    clean fit of something that is not a clock."""
    syncs = [_sync(host=float(d), device=float(d), rtt=0.004,
                   mono=1.6986 * d) for d in (0, 300, 600, 900, 1200)]
    q = _dev_with_syncs(syncs).clock_quality()
    assert q["skew"] == 1.0
    assert "ppm" in (q["skew_pinned_why"] or "")


def test_the_sample_map_is_held_still_while_a_stream_runs():
    """Re-fitting mid-stream would date the second half of a recording on a
    different ruler from the first, and put a step in the middle of it where
    nothing happened. What is learned meanwhile is reported, not applied."""
    d = _dev_with_syncs([_sync(host=t, device=t, rtt=0.004, mono=t)
                         for t in (0.0, 120.0)])
    frozen_skew, frozen_offset = d._skew, d._offset
    d._map_frozen = True
    d._syncs += [_sync(host=1.00004 * t, device=float(t), rtt=0.004,
                       mono=1.00004 * t) for t in (600, 1200, 1800)]
    d._fit_clock()
    assert (d._skew, d._offset) == (frozen_skew, frozen_offset)
    q = d.clock_quality()
    assert q["map_frozen_at_stream_start"] is True
    assert q["skew_measured"] != q["skew"], "nothing was learned"
    assert q["uncorrected_drift_s"] > 0, q["uncorrected_drift_s"]


# --- the wall clock moving under a recording ---------------------------------
# Every test below fails if the fit is regressed against realtime instead of
# monotonic, which is the whole reason the basis was changed.

def test_clock_fit_is_immune_to_a_wall_clock_step():
    """NTP steps realtime +30 s mid-session. The device fit must not move.

    The monotonic readings are clean and evenly spaced; the realtime readings
    jump. A realtime-fitted line would absorb the jump as skew and mis-date
    every sample in the recording.
    """
    clean = [_sync(host=float(d), device=float(d), rtt=0.004, mono=float(d))
             for d in (0, 30, 60, 90, 120)]
    stepped = [_sync(host=s.host + (30.0 if s.mono >= 60 else 0.0),
                     device=s.device, rtt=s.rtt, mono=s.mono) for s in clean]
    q_clean = _dev_with_syncs(clean).clock_quality()
    q_step = _dev_with_syncs(stepped).clock_quality()
    approx(q_step["skew"], q_clean["skew"], 1e-12)
    approx(q_step["offset"], q_clean["offset"], 1e-9)
    assert q_step["residual_ms"] < 1e-6, q_step["residual_ms"]
    assert q_step["fitted_against"] == "monotonic"


def test_clock_reports_the_step_it_survived():
    """Surviving it is not enough — the capture has to say it happened."""
    syncs = [_sync(host=float(d) + (30.0 if d >= 60 else 0.0),
                   device=float(d), rtt=0.004, mono=float(d))
             for d in (0, 30, 60, 90, 120)]
    q = _dev_with_syncs(syncs).clock_quality()
    assert q["wall_clock_stepped"] is True
    assert len(q["wall_clock_steps"]) == 1, q["wall_clock_steps"]
    approx(q["wall_clock_steps"][0]["jump_s"], 30.0, 1e-9)


def test_clock_does_not_cry_step_at_ordinary_ntp_slew():
    """400 ppm of disciplining across 120 s is not a step. It must not read
    as one, or the flag means nothing and gets ignored."""
    syncs = [_sync(host=float(d) * (1 + 400e-6), device=float(d),
                   rtt=0.004, mono=float(d))
             for d in (0, 30, 60, 90, 120)]
    q = _dev_with_syncs(syncs).clock_quality()
    assert q["wall_clock_stepped"] is False, q["wall_clock_steps"]


def test_sample_times_stay_continuous_across_a_step():
    """The sample column is one ruler. A step is recorded, never smeared in."""
    syncs = [_sync(host=float(d) + (30.0 if d >= 60 else 0.0),
                   device=float(d), rtt=0.004, mono=float(d))
             for d in (0, 30, 60, 90, 120)]
    d = _dev_with_syncs(syncs)
    d.info = type("I", (), {"tick_hz": 1000000})()
    ts = [d.host_time(int(s * 1e6)) for s in (0, 30, 60, 90, 120)]
    steps = [b - a for a, b in zip(ts, ts[1:])]
    assert all(abs(s - 30.0) < 1e-6 for s in steps), steps


def test_align_events_undoes_the_step_and_says_so():
    """Two events 1 s apart in real time, with a +30 s jump between them."""
    from intomind import timebase
    meta = dict(clock=dict(anchors=[(0.0, 0.0), (30.0, 30.0),
                                    (90.0, 60.0), (120.0, 90.0)],
                           frame_mono=0.0))
    events = [dict(kind="marker", cond="a", host_time=10.0, mono=10.0),
              dict(kind="marker", cond="b", host_time=41.0, mono=11.0)]
    out = timebase.align_events(meta, events)
    assert all(e["aligned"] for e in out)
    approx(out[1]["host_time_aligned"] - out[0]["host_time_aligned"], 1.0, 1e-9)


def test_align_events_refuses_to_fake_a_correction():
    """An event recorded before dual-stamping has no `mono`. It must be marked
    uncorrected rather than passed off as aligned."""
    from intomind import timebase
    meta = dict(clock=dict(anchors=[(0.0, 0.0), (30.0, 30.0)], frame_mono=0.0))
    out = timebase.align_events(meta, [dict(kind="cue", cond="old",
                                            host_time=12.0)])
    assert out[0]["aligned"] is False
    approx(out[0]["host_time_aligned"], 12.0)


def test_manifest_carries_clock_quality():
    from intomind import provenance as prov
    d = _dev_with_syncs([(0.0, 0.0, 0.004), (60.0, 60.0, 0.006)])
    d.info = type("I", (), {"tick_hz": 32768.0, "fw": "1.1.0",
                            "device_id": "x", "vref_uv": 2420000})()
    d.gaps_announced = 0
    d.fs_effective = lambda pkts: 500.0
    m = prov.build_manifest(label="t", dev=d, packets=[], regs_before={},
                            regs_after={}, protocol="t", params={}, extra={},
                            index=np.arange(10), gaps_in_run=0)
    assert m["clock"]["n_syncs"] == 2
    approx(m["clock"]["max_half_rtt_ms"], 3.0, 1e-9)


def test_a_manifest_reads_the_gain_and_rate_off_the_packets_it_was_given():
    """The empty-packet path was the only one a test had ever walked, and
    the other one read a field the packet does not carry, so every recording
    would have failed at the moment it was saved. The packets here are the
    real decoded shape, built by the protocol's own decoder, so a change to
    that shape breaks this test rather than a recording."""
    from intomind import provenance as prov, protocol as P
    d = _dev_with_syncs([(0.0, 0.0, 0.004), (60.0, 60.0, 0.006)])
    d.info = type("I", (), {"tick_hz": 32768.0, "fw": "1.1.0",
                            "device_id": "x", "vref_uv": 2420000})()
    d.gaps_announced = 0
    d.fs_effective = lambda pkts: 500.0
    header = P._DATA.pack(1, 0, 0, 0, 1234, 2, 0, 6, 5)
    body = b"".join(int(7).to_bytes(3, "big", signed=True) for _ in range(2 * 4))
    packets = [P.decode_packet(header + body, 4)]
    assert packets[0].gain == 24 and packets[0].sample_rate_hz == 500
    m = prov.build_manifest(label="t", dev=d, packets=packets, regs_before={},
                            regs_after={}, protocol="t", params={}, extra={},
                            index=np.arange(10), gaps_in_run=0)
    assert m["gain"] == 24, "the gain the packets carry"
    assert m["rate_sps"] == 500, "the rate the packets carry"


def test_a_manifest_says_what_the_samples_were_off_the_packets():
    """A generated signal is marked where it was recorded, from the packets
    that carried it, not from a setting a recorder may not have read."""
    from intomind import provenance as prov, protocol as P
    from test_protocol import by_kind
    d = _dev_with_syncs([(0.0, 0.0, 0.004), (60.0, 60.0, 0.006)])
    d.info = type("I", (), {"tick_hz": 32768.0, "fw": "1.3.1",
                            "device_id": "x", "vref_uv": 2420000})()
    d.gaps_announced = 0
    d.fs_effective = lambda pkts: 500.0
    decoded = [P.decode_packet(bytes.fromhex(v["bytes"]), 4) for v in by_kind("eeg_data")]
    generated = [p for p in decoded if p.synthetic]
    # The contract's measured vector is the converter's test signal, which is
    # as much a fact about the samples as the electrodes would be.
    measured = [p for p in decoded if not p.synthetic]
    assert generated and measured, "vectors of both kinds"

    def manifest(packets):
        return prov.build_manifest(label="t", dev=d, packets=packets, regs_before={},
                                   regs_after={}, protocol="t", params={}, extra={},
                                   index=np.arange(10), gaps_in_run=0)
    m = manifest(generated[:1])
    assert (m["signal_source"], m["synthetic"]) == ("synthetic", True)
    assert m["manifest_version"] == 6
    m = manifest(measured[:1])
    assert (m["signal_source"], m["synthetic"]) == ("test", False)
    m = manifest(measured[:1] + generated[:1])
    assert (m["signal_source"], m["synthetic"]) == (["synthetic", "test"], True), \
        "a run holding two modes lists both rather than choosing one"
    m = manifest([])
    assert (m["signal_source"], m["synthetic"]) == (None, None)


# ---------------------------------------------------------------- runner
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
    if failed:
        print("failed:", ", ".join(failed))
    return 1 if failed else 0


# ------------------------------------------------- contact, from LOFF (not DC)

def test_contact_decodes_the_loff_bits_per_channel():
    """The contract's packing: bit n is channel n+1, set means that electrode
    is off. One bit per channel, for as many channels as the device has.

    This used to decode a different device's packing, which puts two bits per
    channel in the same byte. The chooser picked that one for any protocol at
    or above 0.2, so every 1.0 device -- every device this library is for --
    decoded the wrong bits and reported electrodes that were never read. The
    packing a device uses is not something to infer from a version number:
    this library decodes the contract it publishes, and an adapter decodes
    its own."""
    from intomind.instrument import contact_from_leadoff as c
    assert c(0x00, 4) == [True, True, True, True]    # nothing flagged
    assert c(0x01, 4) == [False, True, True, True]   # channel 1 off
    assert c(0x02, 4) == [True, False, True, True]   # channel 2 off
    assert c(0x04, 4) == [True, True, False, True]   # channel 3 off
    assert c(0x08, 4) == [True, True, True, False]   # channel 4 off
    assert c(0x0F, 4) == [False, False, False, False]
    assert c(0x00, 2) == [True, True]                # a two-channel device
    assert c(0x02, 2) == [True, False]





def test_contact_needs_to_hold_not_just_blip():
    """One stray comparator sample must not flip the indicator."""
    import numpy as np
    from intomind.instrument import contact_from_leadoff
    mostly_on = np.zeros(100, dtype=np.uint8); mostly_on[:5] = 0x01
    assert contact_from_leadoff(mostly_on) == [True, True]
    mostly_off = np.full(100, 0x01, dtype=np.uint8); mostly_off[:5] = 0x00
    assert contact_from_leadoff(mostly_off) == [False, True]


def test_contact_is_decoded_for_the_channels_the_device_has():
    """A four-channel device gets four answers, and the fourth is read from
    the fourth bit rather than from a byte that never carried it."""
    from intomind.instrument import contact_from_leadoff
    assert len(contact_from_leadoff(0x00, 4)) == 4
    assert len(contact_from_leadoff(0x00, 8)) == 8
    assert contact_from_leadoff(0x80, 8)[7] is False


if __name__ == "__main__":
    sys.exit(main())
