"""Test suite over SYNTHETIC signals with known ground truth.

    python3 tests/test_analysis.py

No pytest, no network, no device. Every capture is fabricated so the right
answer is known in advance. Each test names the defect it guards against.

Run this before trusting any number the analysis layer prints.
"""
from __future__ import annotations
import json, pathlib, shutil, sys, tempfile, traceback

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from intomind import analysis as A          # noqa: E402
from intomind import provenance as prov     # noqa: E402

FS = 500.0
VREF_UV = 2_400_000
GAIN = 12
LSB_UV = (2.0 * VREF_UV) / (GAIN * (1 << 24))     # 0.02404 µV / count

_TMP: pathlib.Path | None = None


# ---------------------------------------------------------------- fixtures
def use_tmp_captures():
    global _TMP
    _TMP = pathlib.Path(tempfile.mkdtemp(prefix="intomind_tests_"))
    A.CAPTURES = _TMP
    prov.CAPTURES = _TMP
    prov.SESSIONS = _TMP / "SESSIONS.jsonl"
    # Every module-level path must be redirected. LEGACY was not, and these
    # tests wrote test hashes straight into the project's real ledger.
    prov.LEGACY = _TMP / "LEGACY.sha256"
    _assert_no_real_paths()


def _assert_no_real_paths() -> None:
    """A test that writes into captures/ is a test that corrupts the record."""
    real = pathlib.Path(__file__).resolve().parent.parent / "captures"
    for name in ("CAPTURES", "SESSIONS", "LEGACY"):
        p = pathlib.Path(getattr(prov, name)).resolve()
        assert real not in [p, *p.parents], \
            f"prov.{name} still points inside the real captures/: {p}"
    assert pathlib.Path(A.CAPTURES).resolve() != real


def cleanup():
    if _TMP and _TMP.exists():
        shutil.rmtree(_TMP)


def pink(n, rng, slope=-1.0):
    """1/f^|slope/2| amplitude noise, unit RMS."""
    f = np.fft.rfftfreq(n, 1 / FS)
    amp = np.ones_like(f)
    amp[1:] = f[1:] ** (slope / 2.0)
    amp[0] = 0.0
    ph = rng.uniform(0, 2 * np.pi, len(f))
    x = np.fft.irfft(amp * np.exp(1j * ph), n)
    return x / x.std()


def write_capture(label, uv_ch1, uv_ch2, *, events=None, index=None,
                  checksum=True, meta_extra=None):
    n = len(uv_ch1)
    counts = np.vstack([np.round(np.asarray(uv_ch1) / LSB_UV),
                        np.round(np.asarray(uv_ch2) / LSB_UV)]).astype(np.int64)
    idx = np.arange(n, dtype=np.int64) if index is None else np.asarray(index)
    np.savez_compressed(A.CAPTURES / f"{label}.npz", counts=counts, index=idx,
                        device_ticks=idx, host_time=idx / FS,
                        leadoff=np.zeros(n, np.uint8),
                        gap_before=np.zeros(n, bool))
    meta = dict(label=label, n=n, fs_effective=FS, vref_uv=VREF_UV, gain=GAIN,
                adc_bits=24,
                settle=0.0, block=10.0, gaps_announced=0)
    meta.update(meta_extra or {})
    (A.CAPTURES / f"{label}.meta.json").write_text(json.dumps(meta))
    if events is not None:
        (A.CAPTURES / f"{label}.events.json").write_text(json.dumps(events))
    if checksum:
        prov.write_checksums(label)
    return label


# ---------------------------------------------------------------- tests
def test_uv_scaling_round_trip():
    """A known-amplitude sine survives counts -> µV unchanged.

    NOTE: this is round-trip CONSISTENCY, not physical truth. The fixture
    encodes counts with the same lsb formula the analysis decodes with, so a
    wrong formula would pass. Only V1 (calibrated injection of a known µV signal
    into the electrode leads) can validate the scaling
    against reality. That has never been done.
    """
    n = int(20 * FS)
    t = np.arange(n) / FS
    amp = 50.0                                    # µV peak
    x = amp * np.sin(2 * np.pi * 10.0 * t)
    write_capture("t_scale", x, x)
    s = A.summary("t_scale")
    rms = s["channels"][0]["ac_rms_uv"]
    want = amp / np.sqrt(2)
    assert abs(rms - want) / want < 0.02, f"rms {rms:.3f} != {want:.3f}"


def test_alpha_is_recovered():
    """A 10 µV, 10 Hz alpha on a pink background must be found, at the right
    frequency, with a peak clearly above the 1/f fit.
    Guards: the whole spectral pipeline."""
    rng = np.random.default_rng(1)
    n = int(60 * FS)
    t = np.arange(n) / FS
    bg = 20.0 * pink(n, rng, slope=-2.0)
    alpha = 10.0 * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 6.28))
    write_capture("t_alpha", bg + alpha, bg + alpha)
    cap = A.load("t_alpha")
    eps, _ = A.clean_epochs(cap, 0, 2.0, 1e9)
    from scipy import signal as sg
    E = np.array(eps)
    f, p = sg.welch(E, fs=FS, nperseg=E.shape[1], axis=1)
    p = p.mean(0)
    kk = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14)) & (p > 0)
    c = np.polyfit(np.log10(f[kk]), np.log10(p[kk]), 1)
    ka = (f >= 7) & (f <= 14)
    resid = p[ka] / 10 ** np.polyval(c, np.log10(f[ka]))
    peak_hz = f[ka][np.argmax(resid)]
    assert resid.max() > 3.0, f"alpha peak only {resid.max():.2f}x"
    assert abs(peak_hz - 10.0) < 0.6, f"peak at {peak_hz:.2f} Hz"


def test_white_noise_has_no_peak_and_flat_slope():
    """Guards: reporting band power as 'alpha' when the spectrum is white."""
    rng = np.random.default_rng(2)
    n = int(60 * FS)
    x = rng.normal(0, 5.0, n)
    write_capture("t_white", x, x)
    f, p = A.welch_psd(A.load("t_white"), 0)
    slope = A.spectral_slope(f, p)
    assert abs(slope) < 0.25, f"slope {slope:+.2f} should be ~0 for white noise"


def test_gaps_do_not_splatter_the_spectrum():
    """A capture with dropped samples must give the same PSD as one without.
    Concatenating across a discontinuity injects broadband power."""
    rng = np.random.default_rng(3)
    n = int(60 * FS)
    t = np.arange(n) / FS
    x = 20.0 * pink(n, rng, -2.0) + 10.0 * np.sin(2 * np.pi * 10 * t)
    write_capture("t_nogap", x, x)
    keep = np.ones(n, bool)
    for start in (5000, 12000, 20000):
        keep[start:start + 137] = False          # three ragged holes
    write_capture("t_gap", x[keep], x[keep], index=np.arange(n)[keep])
    f1, p1 = A.welch_psd(A.load("t_nogap"), 0)
    f2, p2 = A.welch_psd(A.load("t_gap"), 0)
    band = (f1 >= 20) & (f1 <= 45)               # away from the alpha line
    ratio = np.median(p2[band] / p1[band])
    assert 0.6 < ratio < 1.6, f"gap capture power ratio {ratio:.2f} in 20-45 Hz"


def test_gap_ledger_locates_every_hole():
    idx = np.array([0, 1, 2, 5, 6, 20])
    led = prov.gap_ledger(idx)
    assert led["n_gaps"] == 2
    assert led["samples_lost"] == 15
    assert led["gaps"][0] == {"after_sample": 2, "lost": 2}
    assert led["gaps"][1] == {"after_sample": 6, "lost": 13}


def test_integrity_detects_mutation():
    """An analysis must never silently read bytes other than the
    ones that were recorded."""
    rng = np.random.default_rng(4)
    x = rng.normal(0, 5.0, 1000)
    write_capture("t_int", x, x)
    ok, problems = prov.verify("t_int")
    assert ok and not problems, problems

    p = A.CAPTURES / "t_int.npz"
    b = bytearray(p.read_bytes())
    b[len(b) // 2] ^= 0xFF
    p.write_bytes(bytes(b))

    ok, problems = prov.verify("t_int")
    assert not ok and any("MISMATCH" in s for s in problems), problems
    try:
        A.load("t_int")
    except IOError:
        pass
    else:
        raise AssertionError("load() accepted a corrupted capture")


def test_unverified_capture_loads_with_warning():
    """Pre-P0 captures have no checksum: warn, do not fail."""
    rng = np.random.default_rng(5)
    x = rng.normal(0, 5.0, 1000)
    write_capture("t_unver", x, x, checksum=False)
    ok, problems = prov.verify("t_unver")
    assert ok and problems and "unverified" in problems[0]
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        A.load("t_unver")
    assert any("unverified" in str(x.message) for x in w)


def test_marking_legacy_is_distinct_from_finalizing():
    """The integrity phase asks that every old capture be *explicitly marked*.
    Inferring 'unverified' from a missing file is not a marking: a capture whose
    .sha256 was deleted would look identical to one recorded before the scheme
    existed, and those are very different claims."""
    rng = np.random.default_rng(6)
    x = rng.normal(0, 5.0, 800)
    write_capture("t_legacy", x, x, checksum=False)

    ok, problems = prov.verify("t_legacy")
    assert ok and "unverified" in problems[0]

    prov.mark_legacy("t_legacy")
    ok, problems = prov.verify("t_legacy")
    assert ok and problems[0].startswith("provenance incomplete")
    # a marked capture must never acquire the finalized marker
    assert not prov.checksum_path("t_legacy").exists()

    meta = json.loads((A.CAPTURES / "t_legacy.meta.json").read_text())
    assert meta["provenance"] == "incomplete"


def test_marking_never_invents_a_board_revision():
    """A guessed revision would silently pool data across the RLD rework.
    `unknown` is a fact; `an earlier revision` is a fabrication that outlives its author."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 5.0, 500)
    write_capture("t_norev", x, x, checksum=False)
    prov.mark_legacy("t_norev")
    meta = json.loads((A.CAPTURES / "t_norev.meta.json").read_text())
    assert meta["board_revision"] == "unknown"
    assert "board_revision_source" not in meta

    write_capture("t_asserted", x, x, checksum=False)
    prov.mark_legacy("t_asserted", "rev-2")
    meta = json.loads((A.CAPTURES / "t_asserted.meta.json").read_text())
    assert meta["board_revision"] == "rev-2"
    assert "asserted by a human" in meta["board_revision_source"]


def test_marked_legacy_capture_is_tamper_evident():
    """It does not become trustworthy. It becomes tamper-evident."""
    rng = np.random.default_rng(8)
    x = rng.normal(0, 5.0, 600)
    write_capture("t_tamper", x, x, checksum=False)
    prov.mark_legacy("t_tamper")
    assert prov.verify("t_tamper")[0]

    p = A.CAPTURES / "t_tamper.npz"
    p.chmod(0o644)
    with open(p, "ab") as f:
        f.write(b"x")
    ok, problems = prov.verify("t_tamper")
    assert not ok and "CHECKSUM MISMATCH" in problems[0]


def test_a_finalized_capture_cannot_be_marked_legacy():
    rng = np.random.default_rng(9)
    x = rng.normal(0, 5.0, 400)
    write_capture("t_final", x, x, checksum=True)
    try:
        prov.mark_legacy("t_final")
    except ValueError:
        return
    raise AssertionError("a finalized capture was downgraded to 'incomplete'")


def _berger_capture(label, rng, alpha_closed, alpha_open, reps=8, block=10.0):
    """Cued eyes-closed/open capture with a *specified* alpha amplitude."""
    seg = int(block * FS)
    ch, ev, t0 = [], [], 0.0
    for r in range(reps):
        for cond in ("closed", "open"):
            amp = alpha_closed if cond == "closed" else alpha_open
            t = np.arange(seg) / FS
            x = 20.0 * pink(seg, rng, -2.0) + amp * np.sin(
                2 * np.pi * 10.0 * t + rng.uniform(0, 6.28))
            ch.append(x)
            ev.append(dict(cond=cond, rep=r, host_time=t0, kind="cue"))
            t0 += block
    x = np.concatenate(ch)
    return write_capture(label, x, x, events=ev,
                         meta_extra=dict(block=block, settle=0.0, reps=reps))


def test_berger_finds_a_real_effect():
    """A genuine 2x alpha increase on eye closure must be detected."""
    rng = np.random.default_rng(6)
    _berger_capture("t_berger_pos", rng, alpha_closed=20.0, alpha_open=10.0)
    r = A.berger("t_berger_pos", ppmax=1e9)
    abs1 = r["channels"][0]["absolute"]
    assert abs1["ratio"] > 1.5, f"ratio {abs1['ratio']:.2f}"
    assert abs1["p"] < 0.05, f"p {abs1['p']:.4f}"
    assert r["pseudoreplicated"] is False


def test_berger_null_is_null():
    """No effect must produce no effect."""
    rng = np.random.default_rng(7)
    _berger_capture("t_berger_null", rng, alpha_closed=10.0, alpha_open=10.0)
    r = A.berger("t_berger_null", ppmax=1e9)
    abs1 = r["channels"][0]["absolute"]
    assert abs1["p"] > 0.05, f"false positive: p {abs1['p']:.4f}"
    assert "pooled_epochs" not in r["channels"][0], "pooling must be opt-in"


def test_pseudoreplication_is_gated_and_flagged():
    rng = np.random.default_rng(8)
    _berger_capture("t_berger_ps", rng, alpha_closed=10.0, alpha_open=10.0)
    r = A.berger("t_berger_ps", ppmax=1e9, allow_pseudoreplication=True)
    assert r["pseudoreplicated"] is True
    pooled = r["channels"][0]["pooled_epochs"]
    assert "PSEUDOREPLICATED" in pooled["warning"]


def test_false_positive_rate_of_paired_test():
    """Type-I error control: on pure null data the paired block test must
    reject at ~5%, not more."""
    from scipy import stats
    rng = np.random.default_rng(9)
    n_trials, n_blocks, rejects = 400, 8, 0
    for _ in range(n_trials):
        cl = rng.normal(0, 1, n_blocks)
        op = rng.normal(0, 1, n_blocks)
        _, p = stats.wilcoxon(cl, op, alternative="greater")
        rejects += p < 0.05
    rate = rejects / n_trials
    assert rate < 0.10, f"type-I error {rate:.3f} — test is anticonservative"


def test_p_floor_is_reported():
    """With few blocks the smallest attainable p is 2^-n. A 3-rep berger can
    never reach 0.05 and the result must say so."""
    rng = np.random.default_rng(10)
    _berger_capture("t_berger_3", rng, 20.0, 10.0, reps=3)
    r = A.berger("t_berger_3", ppmax=1e9)
    assert abs(r["channels"][0]["absolute"]["p_floor"] - 0.125) < 1e-9


def test_analysis_version_and_capture_hash_are_stamped():
    """A result that cannot name its inputs is not reproducible."""
    rng = np.random.default_rng(11)
    _berger_capture("t_stamp", rng, 20.0, 10.0, reps=4)
    r = A.berger("t_stamp", ppmax=1e9)
    assert r["analysis_version"] == A.ANALYSIS_VERSION
    assert len(r["capture_sha256"]) == 64
    assert r["params"]["alpha"] == [8.0, 13.0]


def _saccade_capture(label, rng, *, invert=True, amp=120.0, sham_amp=0.0,
                     trials=12, hold=2.0, center=2.0):
    """Synthetic saccade recording.

    invert=True   -> ch2 is the mirror of ch1 (a real dipole)
    invert=False  -> ch2 follows ch1 (a common-mode artifact, must FAIL)
    """
    seg_c, seg_h = int(center * FS), int(hold * FS)
    conds = (["left", "right"] * (trials // 2)) + ["sham", "sham"]
    ch1, ch2, ev, t = [], [], [], 0.0
    for i, cond in enumerate(conds):
        n = seg_c + seg_h
        b1 = 8.0 * pink(n, rng, -1.5)
        b2 = 8.0 * pink(n, rng, -1.5)
        step = np.zeros(n)
        a = {"left": +amp, "right": -amp, "sham": sham_amp}[cond]
        tau = 0.05 * FS
        k = np.arange(seg_h)
        step[seg_c:] = a * (1 - np.exp(-k / tau))     # fast rise, held
        ch1.append(b1 + step)
        ch2.append(b2 + (-step if invert else step))
        ev.append(dict(cond=cond, trial=i, kind="cue",
                       host_time=(t + seg_c) / FS))
        t += n
    x1, x2 = np.concatenate(ch1), np.concatenate(ch2)
    return write_capture(label, x1, x2, events=ev,
                         meta_extra=dict(trials=trials))


def test_saccade_passes_on_a_real_dipole():
    """V2: opposite-polarity deflection across L/R temples, sign reversing with
    direction, sham null. This is the acceptance criterion for the instrument."""
    rng = np.random.default_rng(20)
    _saccade_capture("t_sac_pos", rng, invert=True)
    r = A.saccade("t_sac_pos")
    assert r["verdict"] == "PASS", (r["verdict"], r.get("failed_criteria"),
                                    r["criteria"])
    assert r["criteria"]["polarity_inverted_left"]
    assert r["criteria"]["sign_reverses_with_direction"]
    assert r["criteria"]["sham_is_null"]
    L = r["conditions"]["left"]["channels"]
    assert L[0]["sign"] == -L[1]["sign"]


def test_saccade_fails_on_common_mode_artifact():
    """A deflection that does NOT invert across channels is not ocular.
    Guards: accepting a cue-locked artifact as proof the device sees a brain."""
    rng = np.random.default_rng(21)
    _saccade_capture("t_sac_cm", rng, invert=False)
    r = A.saccade("t_sac_cm")
    assert r["verdict"] == "FAIL"
    assert not r["criteria"]["polarity_inverted_left"]


def test_saccade_fails_when_sham_responds():
    """If the cue itself produces a deflection, the response is not ocular."""
    rng = np.random.default_rng(22)
    _saccade_capture("t_sac_sham", rng, invert=True, sham_amp=120.0)
    r = A.saccade("t_sac_sham")
    assert r["criteria"]["sham_is_null"] is False
    assert r["verdict"] == "FAIL"


def test_saccade_fails_when_deflection_too_small():
    rng = np.random.default_rng(23)
    _saccade_capture("t_sac_small", rng, invert=True, amp=10.0)
    r = A.saccade("t_sac_small")
    assert r["criteria"]["deflection_big_enough"] is False
    assert r["verdict"] == "FAIL"


def _berger_denominator_capture(label, rng, reps=8, block=10.0):
    """Alpha identical in both conditions; THETA halves on eye closure.

    Relative alpha (alpha / 4-30 Hz) must therefore rise even though absolute
    alpha never moves. This is the exact shape of the 2026-07-10 false positive.
    """
    seg = int(block * FS)
    ch, ev, t0 = [], [], 0.0
    for r in range(reps):
        for cond in ("closed", "open"):
            t = np.arange(seg) / FS
            theta_amp = 10.0 if cond == "closed" else 25.0
            x = (3.0 * pink(seg, rng, -1.0)
                 + 10.0 * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 6.28))
                 + theta_amp * np.sin(2 * np.pi * 6.0 * t + rng.uniform(0, 6.28)))
            ch.append(x)
            ev.append(dict(cond=cond, rep=r, host_time=t0, kind="cue"))
            t0 += block
    x = np.concatenate(ch)
    return write_capture(label, x, x, events=ev,
                         meta_extra=dict(block=block, settle=0.0, reps=reps))


def test_denominator_artifact_is_flagged():
    """Guards: the 2026-07-10 false positive. Relative alpha rose (p=0.0117)
    only because eyes-closed cut theta by 40%. Absolute alpha never moved."""
    rng = np.random.default_rng(30)
    _berger_denominator_capture("t_denom", rng)
    r = A.berger("t_denom", ppmax=1e9)
    c = r["channels"][0]
    assert c["relative"]["ratio"] > 1.1, c["relative"]["ratio"]
    assert c["relative"]["p"] < 0.05, c["relative"]["p"]
    assert c["absolute"]["p"] > 0.05, c["absolute"]["p"]
    assert c["denominator_artifact"] is True
    assert "denominator effect" in c["interpretation"]
    assert c["reference_band_ratio"] < 0.9


def test_real_effect_is_not_flagged_as_artifact():
    rng = np.random.default_rng(31)
    _berger_capture("t_real", rng, alpha_closed=20.0, alpha_open=10.0)
    r = A.berger("t_real", ppmax=1e9)
    assert r["channels"][0]["denominator_artifact"] is False


def test_aborted_capture_is_never_analysable():
    """A run killed by a link drop must never be mistaken for a
    complete capture."""
    rng = np.random.default_rng(40)
    x = rng.normal(0, 5.0, 1000)
    write_capture("t_abort", x, x, checksum=False,
                  meta_extra=dict(aborted="link_lost"))
    ok, problems = prov.verify("t_abort")
    assert not ok and "ABORTED" in problems[0], problems
    try:
        A.load("t_abort")
    except IOError as e:
        assert "ABORTED" in str(e)
    else:
        raise AssertionError("load() accepted an aborted capture")


# ---------------------------------------------------------------- runner
def main() -> int:
    use_tmp_captures()
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
    cleanup()
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("failed:", ", ".join(failed))
    return 1 if failed else 0



def test_microvolts_use_the_device_resolution_not_a_hardcoded_24():
    """A recording's resolution is a property of the DEVICE, not a constant.

    The scaling hardcoded 2^24. That is right for the first two converters
    this library met, and silently wrong for anything else. A cross-device
    comparison that rests on an assumption nobody wrote down is how you get a
    wrong answer that looks fine.
    """
    from intomind import protocol as P

    def info(bits):
        return P.DeviceInfo(protocol=(1, 0), firmware=(1, 0, 0), channels=4,
                            adc_bits=bits, tick_hz=1_000_000, vref_uv=4_500_000,
                            capabilities=0, supported_rates=7, device_id="00" * 8)

    at24 = info(24).microvolts_per_count(24)
    at16 = info(16).microvolts_per_count(24)
    assert abs(at24 - 0.022_351_741_790_771_484) < 1e-12
    assert abs(at16 / at24 - 256.0) < 1e-9, "a narrower converter has a larger step"
    # And the stream uses that, rather than a constant of its own. The
    # scaling lives in one place so it cannot be right in one path and
    # wrong in another, which is how it was wrong before.
    import inspect
    from intomind import client
    src = inspect.getsource(client.Device._on_data)
    assert "1 << 24" not in src, "the microvolt scaling hardcodes a width again"
    assert "microvolts_per_count" in src, "the stream scales by something other than the device's own answer"


def test_a_manifest_states_the_resolution_it_was_recorded_at():
    """v1 manifests do not say what resolution made them, so they carry an
    assumption. New ones state it, along with the channel count and protocol."""
    import inspect
    from intomind import provenance
    src = inspect.getsource(provenance.build_manifest)
    for field in ("adc_bits", "channels", "device_proto"):
        assert field in src, f"the manifest does not record {field}"
    assert provenance.MANIFEST_VERSION >= 2


# --- events land on the sample ruler, even when the wall clock moved ---------

def _stepped_capture(label, event):
    """10 s of samples; the wall clock jumps +30 s five seconds in.

    Sample `host_time` runs 0..10 s (write_capture uses idx/FS), and the
    anchors put the recording's frame at mono 0, so an event's aligned time is
    its monotonic reading. An event stamped AFTER the step carries a raw
    `host_time` of ~37 s, which indexes off the end of a 10 s capture.
    """
    anchors = [(0.0, 0.0), (5.0, 5.0), (36.0, 6.0), (40.0, 10.0)]
    n = int(10 * FS)
    return write_capture(label, np.zeros(n), np.zeros(n), events=[event],
                         meta_extra=dict(clock=dict(anchors=anchors,
                                                    frame_mono=0.0,
                                                    wall_clock_stepped=True)))


def test_events_are_aligned_onto_the_sample_ruler():
    """The event happened 7 s into the recording. The wall clock says 37 s.
    Indexing the samples with the raw stamp puts it past the end of the data."""
    lbl = _stepped_capture("t_step", dict(kind="marker", cond="press",
                                          host_time=37.0, mono=7.0))
    ev = A.events(lbl)[0]
    assert ev["aligned"] is True
    assert abs(ev["host_time"] - 7.0) < 1e-9, ev["host_time"]
    assert abs(ev["host_time_recorded"] - 37.0) < 1e-9

    cap = A.load(lbl)
    i = int(np.searchsorted(cap.host_time, ev["host_time"]))
    assert i == int(7.0 * FS), i
    # what the uncorrected stamp would have done
    bad = int(np.searchsorted(cap.host_time, ev["host_time_recorded"]))
    assert bad == len(cap.host_time), bad


def test_events_recorded_before_dual_stamping_are_marked_not_aligned():
    """No `mono` means no correction is possible. It must say so rather than
    present the raw stamp as though it had been checked."""
    lbl = _stepped_capture("t_step_legacy",
                           dict(kind="cue", cond="old", host_time=3.0))
    ev = A.events(lbl)[0]
    assert ev["aligned"] is False
    assert abs(ev["host_time"] - 3.0) < 1e-9


def test_a_clean_capture_needs_no_correction():
    """No step, so aligned times equal recorded times exactly. The machinery
    must be inert when there is nothing to fix."""
    n = int(4 * FS)
    lbl = write_capture("t_nostep", np.zeros(n), np.zeros(n),
                        events=[dict(kind="marker", cond="a",
                                     host_time=2.0, mono=2.0)],
                        meta_extra=dict(clock=dict(
                            anchors=[(0.0, 0.0), (4.0, 4.0)], frame_mono=0.0,
                            wall_clock_stepped=False)))
    ev = A.events(lbl)[0]
    assert ev["aligned"] is True
    assert ev["host_time"] == ev["host_time_recorded"] == 2.0


# --- controller input joins the same stream ----------------------------------

def _capture_with_inputs(label):
    n = int(10 * FS)
    anchors = [(0.0, 0.0), (5.0, 5.0), (36.0, 6.0), (40.0, 10.0)]
    write_capture(label, np.zeros(n), np.zeros(n),
                  meta_extra=dict(clock=dict(anchors=anchors, frame_mono=0.0,
                                             wall_clock_stepped=True)))
    np.savez_compressed(
        A.CAPTURES / f"{label}.inputs.npz",
        input_mono=np.array([7.0, 7.25, 7.5]),
        input_host_time=np.array([37.0, 37.25, 37.5]),
        input_type=np.array([1, 3, 1], dtype=np.int16),      # KEY, ABS, KEY
        input_code=np.array([0x130, 0x00, 0x130], dtype=np.int32),
        input_value=np.array([1, -32768, 0], dtype=np.int32))
    return label


def test_controller_inputs_land_on_the_sample_ruler():
    """Same guarantee as events(): the time you get back is the one that
    indexes the samples, wall-clock step or not."""
    lbl = _capture_with_inputs("t_inputs")
    got = A.inputs(lbl)
    assert len(got) == 3
    press = got[0]
    assert press["kind"] == "button" and press["cond"] == "BTN_SOUTH"
    assert press["value"] == 1
    assert abs(press["host_time"] - 7.0) < 1e-9
    assert abs(press["host_time_recorded"] - 37.0) < 1e-9

    cap = A.load(lbl)
    assert int(np.searchsorted(cap.host_time, press["host_time"])) == int(7 * FS)


def test_inputs_separate_buttons_from_axes():
    """A press is an event; a stick position is a signal. Treating a hundred
    axis samples a second as discrete labels makes a corpus unusable."""
    lbl = _capture_with_inputs("t_inputs_k")
    btns = A.inputs(lbl, kinds=("button",))
    axes = A.inputs(lbl, kinds=("axis",))
    assert [e["value"] for e in btns] == [1, 0]
    assert len(axes) == 1 and axes[0]["cond"] == "ABS_X"


def test_a_capture_with_no_controller_returns_nothing():
    """Not an error, and not an empty controller — simply no input stream."""
    n = int(2 * FS)
    lbl = write_capture("t_noinputs", np.zeros(n), np.zeros(n))
    assert A.inputs(lbl) == []


# ------------------------------------------- repairs made on read, not on disk
def _write_raw(label, counts, ticks, host_time, meta_extra=None):
    """A capture written straight from arrays, defects and all.

    `write_capture` derives ticks and host_time from the sample index, which
    is exactly what a healthy recorder produces -- so these tests, which are
    about recorders that were not healthy, have to lay the arrays down
    themselves.
    """
    n = counts.shape[1]
    np.savez_compressed(A.CAPTURES / f"{label}.npz", counts=counts,
                        index=np.arange(n, dtype=np.int64),
                        device_ticks=np.asarray(ticks, np.int64),
                        host_time=np.asarray(host_time, float),
                        leadoff=np.zeros(n, np.uint8),
                        gap_before=np.zeros(n, bool))
    meta = dict(label=label, n=n, fs_effective=FS, vref_uv=VREF_UV, gain=GAIN,
                adc_bits=24,
                tick_hz=1_000_000, settle=0.0, block=10.0, gaps_announced=0)
    meta.update(meta_extra or {})
    (A.CAPTURES / f"{label}.meta.json").write_text(json.dumps(meta))
    prov.write_checksums(label)
    return label


def test_duplicate_packet_delivery_is_removed():
    """Two BLE subscriptions deliver every notification twice, and both
    copies are well-formed, so the recorder stored each sample twice. One
    session came out at 2x its own sample rate with every device tick
    repeated, and the analysis read a signal that jumped five samples
    backwards every five samples."""
    n, spp = 40, 5
    base = np.vstack([np.arange(n), np.arange(n) * 2]).astype(np.int64)
    ticks = np.arange(n, dtype=np.int64) * 4000
    keep = [slice(i, i + spp) for i in range(0, n, spp)]
    dup_c = np.hstack([base[:, s] for s in keep for _ in (0, 1)])
    dup_t = np.hstack([ticks[s] for s in keep for _ in (0, 1)])
    lbl = _write_raw("t_dupe", dup_c, dup_t, dup_t / 1e6)

    cap = A.load(lbl)
    assert cap.counts.shape[1] == n, cap.counts.shape
    assert np.array_equal(cap.counts, base)
    assert np.array_equal(cap.device_ticks, ticks)
    assert any("delivered 2 times over" in r for r in cap.repairs), cap.repairs
    assert A.load(lbl, repair=False).counts.shape[1] == 2 * n


def test_repeated_ticks_with_different_counts_are_not_halved():
    """The signature of duplicate delivery is a repeated device time carrying
    IDENTICAL counts. Repeated times over different samples are some other
    fault, and dropping half the recording to make it fit this one would
    destroy the evidence of whatever it actually was."""
    c = np.vstack([np.arange(8), np.arange(8) * 3]).astype(np.int64)
    ticks = np.array([0, 0, 4000, 4000, 8000, 8000, 12000, 12000], np.int64)
    lbl = _write_raw("t_dupe_differs", c, ticks, ticks / 1e6)
    cap = A.load(lbl)
    assert cap.counts.shape[1] == 8
    assert any("DIFFERENT counts" in r for r in cap.repairs), cap.repairs


def test_an_impossible_clock_skew_is_restamped_from_the_device():
    """A slope fitted across a sub-second baseline is round-trip jitter. One
    session got 1.6986 and stamped 1945 s of recording over 3304 s."""
    n, skew = 100, 1.6986
    ticks = np.arange(n, dtype=np.int64) * 4000            # 250 SPS
    dev = ticks / 1e6
    # The exchanges as the broken fit left them: mono = skew*device + offset.
    off, t0 = 20.0, 1_700_000_000.0
    mono = np.array([0.10, 0.25, 0.40, 0.55, 0.70]) + 20.0
    anchors = [[t0 + (m - mono[0]), m] for m in mono]
    clock = dict(skew=skew, offset=off, anchors=anchors, span_s=0.35,
                 frame_mono=mono[0])
    lbl = _write_raw("t_skew", np.zeros((2, n), np.int64), ticks,
                     dev * skew + off, meta_extra=dict(clock=clock))

    cap = A.load(lbl)
    span = float(cap.host_time[-1] - cap.host_time[0])
    assert abs(span - (n - 1) * 0.004) < 1e-6, span   # the device's own time
    assert any("skew 1.6986" in r for r in cap.repairs), cap.repairs
    # And the times still sit in the wall clock the recording was written in,
    # not at the device's uptime.
    assert abs(cap.host_time[0] - t0) < 1.0
    assert not A.load(lbl, repair=False).repairs


def test_a_believable_skew_is_left_alone():
    """This overrules a past recording's own arithmetic, so it does so only
    where the arithmetic is impossible -- not wherever it is non-zero."""
    n = 50
    ticks = np.arange(n, dtype=np.int64) * 4000
    clock = dict(skew=1.00002, offset=0.0, span_s=120.0,
                 anchors=[[0.0, 0.0], [120.0, 120.0]], frame_mono=0.0)
    lbl = _write_raw("t_skew_ok", np.zeros((2, n), np.int64), ticks,
                     ticks / 1e6, meta_extra=dict(clock=clock))
    assert A.load(lbl).repairs == []


if __name__ == "__main__":
    sys.exit(main())
