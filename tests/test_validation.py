"""V3 (blink) and V4 (jaw clench) analyses, on synthetic signals.

    python3 tests/test_validation.py

Every fixture has a known right answer. A validation test that cannot FAIL on
fabricated data validates nothing, so each positive case is paired with the
negative that must be rejected: a sham that responds, an average carried by one
outlier trial, an EMG burst that is really a DC step.
"""
from __future__ import annotations
import json, pathlib, shutil, sys, tempfile, traceback

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from intomind import analysis as A          # noqa: E402
from intomind import provenance as prov     # noqa: E402

FS = 500.0
VREF_UV, GAIN = 2_400_000, 12
LSB_UV = (2.0 * VREF_UV) / (GAIN * (1 << 24))
_TMP: pathlib.Path | None = None


def use_tmp():
    global _TMP
    _TMP = pathlib.Path(tempfile.mkdtemp(prefix="intomind_val_"))
    A.CAPTURES = _TMP
    prov.CAPTURES = _TMP
    prov.SESSIONS = _TMP / "SESSIONS.jsonl"
    prov.LEGACY = _TMP / "LEGACY.sha256"


def write(label, ch1, ch2, events):
    n = len(ch1)
    counts = np.vstack([np.round(np.asarray(ch1) / LSB_UV),
                        np.round(np.asarray(ch2) / LSB_UV)]).astype(np.int64)
    idx = np.arange(n, dtype=np.int64)
    np.savez_compressed(A.CAPTURES / f"{label}.npz", counts=counts, index=idx,
                        device_ticks=idx, host_time=idx / FS,
                        leadoff=np.zeros(n, np.uint8),
                        gap_before=np.zeros(n, bool))
    (A.CAPTURES / f"{label}.meta.json").write_text(json.dumps(
        dict(label=label, n=n, fs_effective=FS, vref_uv=VREF_UV, gain=GAIN,
              adc_bits=24)))
    (A.CAPTURES / f"{label}.events.json").write_text(json.dumps(events))
    prov.write_checksums(label)
    return label


def pink(n, rng, slope=-1.5):
    f = np.fft.rfftfreq(n, 1 / FS); f[0] = f[1]
    s = np.fft.rfft(rng.normal(0, 1, n)) * f ** (slope / 2)
    x = np.fft.irfft(s, n)
    return x / (x.std() or 1.0)


# ---------------------------------------------------------------- blink
def _blink_capture(label, rng, *, amp=180.0, sham_amp=0.0, trials=12,
                   jitter_s=0.02, one_outlier=False):
    pre, post = int(2.0 * FS), int(2.0 * FS)
    conds = ["blink"] * trials + ["sham"] * 4
    ch1, ch2, ev, t = [], [], [], 0
    for i, cond in enumerate(conds):
        n = pre + post
        b1, b2 = 6.0 * pink(n, rng), 6.0 * pink(n, rng)
        a = {"blink": amp, "sham": sham_amp}[cond]
        if one_outlier and cond == "blink":
            a = amp * 12 if i == 0 else 2.0     # one huge trial, rest null
        w = np.zeros(n)
        onset = pre + int(rng.normal(0, jitter_s) * FS)
        k = np.arange(int(0.25 * FS))
        w[onset:onset + len(k)] = a * np.sin(np.pi * k / len(k))   # one hump
        ch1.append(b1 + w)
        ch2.append(b2 + 0.9 * w)               # blinks are common-mode-ish
        ev.append(dict(cond=cond, trial=i, kind="cue", host_time=(t + pre) / FS))
        t += n
    return write(label, np.concatenate(ch1), np.concatenate(ch2), ev)


def test_blink_passes_on_real_blinks():
    r = A.blink(_blink_capture("b_ok", np.random.default_rng(1)))
    assert r["verdict"] == "PASS", r.get("failed_criteria")
    assert r["conditions"]["blink"]["channels"][0]["mean_pp_uv"] > 100


def test_blink_fails_when_too_small():
    r = A.blink(_blink_capture("b_small", np.random.default_rng(2), amp=30.0))
    assert r["verdict"] == "FAIL"
    assert "amplitude" in r["failed_criteria"]


def test_blink_fails_when_the_sham_responds():
    """If a deflection appears when the cue plays and the subject does nothing,
    the response is not ocular. It is the speaker, or the audio driver."""
    r = A.blink(_blink_capture("b_sham", np.random.default_rng(3),
                               sham_amp=200.0))
    assert r["verdict"] == "FAIL"
    assert "sham_is_null" in r["failed_criteria"]


def test_blink_fails_when_one_outlier_trial_carries_the_average():
    """A single movement artifact can manufacture a beautiful grand average out
    of twenty null trials. Per-trial detection is what catches it."""
    r = A.blink(_blink_capture("b_out", np.random.default_rng(4),
                               one_outlier=True))
    ch = r["conditions"]["blink"]["channels"][0]
    assert ch["mean_pp_uv"] > 100, "the average looks fine, as designed"
    assert ch["detected_frac"] < 0.9
    assert r["verdict"] == "FAIL"
    assert "reliable_across_trials" in r["failed_criteria"]


def test_blink_fails_on_scattered_latency():
    r = A.blink(_blink_capture("b_jit", np.random.default_rng(5),
                               jitter_s=0.6), max_latency_sd_s=0.1)
    assert "consistent_latency" in r.get("failed_criteria", [])


# ---------------------------------------------------------------- jaw
def _jaw_capture(label, rng, *, burst=12.0, sham_burst=0.0, trials=10,
                 dc_step=0.0):
    pre, post = int(2.0 * FS), int(2.0 * FS)
    conds = ["clench"] * trials + ["sham"] * 3
    ch1, ch2, ev, t = [], [], [], 0
    for i, cond in enumerate(conds):
        n = pre + post
        b1, b2 = 4.0 * pink(n, rng), 4.0 * pink(n, rng)
        amp = {"clench": burst, "sham": sham_burst}[cond]
        emg = np.zeros(n)
        seg = rng.normal(0, 1, post)
        # broadband 20-200 Hz: white noise high-passed by differencing
        emg[pre:] = amp * np.diff(np.concatenate([[0.0], seg]))
        if dc_step and cond == "clench":
            emg[pre:] += dc_step               # a DC step, not EMG
        ch1.append(b1 + emg)
        ch2.append(b2 + emg)
        ev.append(dict(cond=cond, trial=i, kind="cue", host_time=(t + pre) / FS))
        t += n
    return write(label, np.concatenate(ch1), np.concatenate(ch2), ev)


def test_jaw_passes_on_real_emg():
    r = A.jaw(_jaw_capture("j_ok", np.random.default_rng(6)))
    assert r["verdict"] == "PASS", r.get("failed_criteria")
    ch = r["channels"][0]
    assert ch["ratio"] >= 10.0 and ch["p"] < 0.001
    assert ch["wins"] == ch["n_trials"]


def test_jaw_fails_on_a_quiet_clench():
    r = A.jaw(_jaw_capture("j_weak", np.random.default_rng(7), burst=0.6))
    assert r["verdict"] == "FAIL"
    assert "power_ratio" in r["failed_criteria"]


def test_jaw_is_not_fooled_by_a_dc_step():
    """A DC step has no 20-200 Hz content. If the band-power criterion could be
    satisfied by an offset, every electrode pop would read as a clench."""
    r = A.jaw(_jaw_capture("j_dc", np.random.default_rng(8), burst=0.0,
                           dc_step=5000.0))
    assert r["verdict"] == "FAIL", r["channels"][0]


def test_jaw_p_floor_is_reported():
    """With 10 trials no paired test can ever beat p = 2^-10."""
    r = A.jaw(_jaw_capture("j_floor", np.random.default_rng(9), trials=10))
    ch = r["channels"][0]
    assert ch["p_floor"] == 2.0 ** -10
    assert ch["p"] >= ch["p_floor"] / 2


def test_jaw_needs_enough_trials():
    r = A.jaw(_jaw_capture("j_few", np.random.default_rng(10), trials=2))
    assert r["channels"][0].get("note") == "too few trials"
    assert r["verdict"] is None


def test_jaw_band_is_clipped_below_nyquist():
    r = A.jaw(_jaw_capture("j_nyq", np.random.default_rng(11)), hi_hz=400.0)
    assert r["params"]["band"][1] < FS / 2


# ---------------------------------------------------------------- registry
def test_protocols_and_analyses_are_registered_and_preregistered():
    from intomind import experiments as X
    for name, ana in (("blink", "blink"), ("jaw_clench", "jaw")):
        assert name in X.REGISTRY, name
        pr = X.REGISTRY[name]["prereg"]
        assert pr and pr["name"] == ana
        assert not X.REGISTRY[name]["enforced"]   # no per-run gate
        assert ana in A.ANALYSES


def test_validation_protocols_include_a_sham_condition():
    """Every positive is worthless without its negative control (V5)."""
    from intomind import experiments as X
    for name in ("blink", "jaw_clench", "saccade"):
        names = {p["name"] for p in X.REGISTRY[name]["params"]}
        assert "include_sham" in names, name


def test_timing_metadata_survives_an_epoch_reset():
    """A recording whose sample index resets mid-capture (a stale epoch, as in
    berger_on_g12) must still report the correct rate and non-negative loss.
    The index method gave 162 SPS and -68775 lost; ticks + robust ledger fix it."""
    # 205 s at 494 SPS; index starts high (stale epoch), resets to 0 at sample 74.
    n = 101270
    fs_true = 494.0
    ticks = (np.arange(n) * (1_000_000 / fs_true)).astype(np.int64)   # monotonic uptime
    index = np.arange(n, dtype=np.int64)
    index[:74] += 69650                     # stale-epoch head, then reset to 0
    index[300:] += 25                        # one real forward gap of 25 lost

    q = prov.gap_ledger(index)
    assert q["samples_lost"] == 25, q          # only the forward gap, never negative
    assert q["epoch_resets"] == 1, q           # the reset counted, not as loss
    assert q["samples_expected"] == n + 25, q

    fs = prov.fs_from_ticks(ticks, 1_000_000)
    assert abs(fs - fs_true) < 1.0, fs         # 494, not 162


def main() -> int:
    use_tmp()
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


def test_a_capture_records_what_produced_its_signal():
    """A recording used to state the device and the firmware but not whether it
    had been filtered, so it could not say what it contained. On 2026-08-04 that
    had to be inferred from the spectrum of the file itself."""
    import sys, pathlib as _p
    sys.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
    from intomind import provenance as prov
    from intomind.client import Device

    class Ours:
        ADAPTER = Device.ADAPTER
        filter = dict(on=True, hp_hz=0.5, lp_hz=131.0, bands=[(58.0, 62.0)])

    sc = prov.signal_chain(Ours())
    assert sc["adapter"]["first_party"] is True
    assert sc["filtering"] == "device"
    assert sc["filters"]["bands"] == [[58.0, 62.0]]

    class Unfiltered:
        ADAPTER = Device.ADAPTER
        filter = dict(on=False)

    assert prov.signal_chain(Unfiltered())["filtering"] == "none"
    assert prov.signal_chain(Unfiltered())["filters"] is None


def test_an_unidentified_adapter_is_not_recorded_as_ours():
    """Reported, never assumed. Code we did not ship must be identifiable in a
    recording forever, not only while someone remembers."""
    import sys, pathlib as _p
    sys.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
    from intomind import provenance as prov

    class Anon:
        filter = None

    ad = prov.signal_chain(Anon())["adapter"]
    assert ad["first_party"] is False, "an unidentified adapter passed as ours"
    assert ad["name"] == "unknown"


def test_host_side_filtering_says_so():
    """A device that cannot filter in firmware is filtered by the application,
    and the capture then holds a signal the device never produced."""
    import sys, pathlib as _p
    sys.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
    from intomind import provenance as prov

    class Third:
        ADAPTER = dict(name="x", version="1", first_party=False, transport="serial")
        FILTERS_IN = "host"
        filter = dict(on=True, hp_hz=1.0, lp_hz=40.0, bands=[])

    assert prov.signal_chain(Third())["filtering"] == "host"


if __name__ == "__main__":
    sys.exit(main())
