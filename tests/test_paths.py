"""Where captures live, and what a count is worth.

Three defects this suite exists to hold down, each of which was silent:

1. `analysis` worked out the captures directory from where its own file sat,
   which is right for a checkout and wrong for an installed package, and
   disagreed with `provenance`, which asked the environment.
2. `experiments` bound that directory at import time, so an application that
   redirected it had its recorder writing to the old place while everything
   that read looked at the new one. The failure was an empty capture list
   rather than an error.
3. `Capture.lsb_uv` assumed twenty four bits rather than reading what the
   device said, so every microvolt an analysis produced was silently wrong on
   any device of a different width.
"""
from __future__ import annotations
import pathlib, shutil, sys, tempfile, traceback

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from intomind import analysis as A          # noqa: E402
from intomind import experiments as E       # noqa: E402
from intomind import provenance as prov     # noqa: E402

_TMP: pathlib.Path | None = None


def use_tmp() -> pathlib.Path:
    global _TMP
    _TMP = pathlib.Path(tempfile.mkdtemp(prefix="intomind-paths-"))
    return _TMP


def capture(meta: dict) -> A.Capture:
    return A.Capture(
        label="x", counts=np.zeros((2, 4), dtype=np.int64),
        index=np.arange(4), host_time=np.arange(4, dtype=float), meta=meta)


def test_the_captures_directory_is_never_inferred_from_this_files_location():
    here = pathlib.Path(A.__file__).resolve()
    assert A.CAPTURES != here.parents[2] / "captures"
    assert A.CAPTURES == prov.CAPTURES
    assert prov.captures_dir() == prov.CAPTURES


def test_redirecting_moves_every_reader_and_every_writer():
    root = use_tmp()
    tmp = root / "elsewhere"
    prov.use_captures_dir(tmp)
    try:
        assert prov.captures_dir() == tmp
        assert prov.CAPTURES == tmp
        assert A.CAPTURES == tmp, "a reader left behind reports an empty collection, not an error"
        assert prov.SESSIONS.parent == tmp
        assert prov.LEGACY.parent == tmp
        assert prov.MONTAGE_BACKFILL.parent == tmp
        # And a reader really does look there.
        (tmp / "demo.meta.json").write_text('{"label": "demo"}')
        assert any(c["label"] == "demo" for c in A.list_captures())
    finally:
        # Leave the process pointing somewhere harmless. The suite never
        # writes into a real collection, which is the reason this function
        # exists at all.
        prov.use_captures_dir(root / "after")


def test_the_recorder_asks_for_the_directory_rather_than_remembering_it():
    src = pathlib.Path(E.__file__).read_text()
    assert "from .analysis import CAPTURES" not in src, (
        "importing the name binds it at import time, which is the defect")
    assert not hasattr(E, "CAPTURES"), "a module level copy is a copy that can go stale"
    asks = src.count("prov.captures_dir()") + src.count("prov.capture_path(")
    assert asks >= 5, f"every write path asks, and only {asks} do"


def test_a_count_is_worth_what_the_device_said_it_is_worth():
    at24 = capture(dict(vref_uv=4_500_000, gain=24, adc_bits=24)).lsb_uv
    at16 = capture(dict(vref_uv=4_500_000, gain=24, adc_bits=16)).lsb_uv
    assert abs(at24 - 0.022_351_741_790_771_484) < 1e-12
    assert abs(at16 / at24 - 256.0) < 1e-9, "a narrower converter has a larger step"
    # Gain divides it, as the contract says.
    at1 = capture(dict(vref_uv=4_500_000, gain=1, adc_bits=24)).lsb_uv
    assert abs(at1 / at24 - 24.0) < 1e-9


def test_a_capture_that_never_recorded_its_width_says_so():
    try:
        capture(dict(vref_uv=4_500_000, gain=24)).lsb_uv
    except ValueError as e:
        assert "adc_bits" in str(e)
    else:
        raise AssertionError("assuming a width would scale every number silently")


def test_per_sample_marks_travel_with_the_samples_they_belong_to():
    """A repair that drops samples has to drop their marks with them.

    Lead-off and gap flags are recorded one per sample. Reading them from
    the file and pairing them by position after a repair has removed
    samples puts every mark against the wrong sample, quietly.
    """
    import numpy as np
    from intomind.analysis import _dedupe_delivery

    # Six samples where every one was delivered twice.
    counts = np.array([[1, 1, 2, 2, 3, 3]], dtype=np.int64)
    ticks = np.array([10, 10, 20, 20, 30, 30], dtype=np.int64)
    index = np.arange(6)
    host = np.arange(6, dtype=float)
    leadoff = np.array([0, 0, 4, 4, 0, 0], dtype=np.uint8)
    gaps = np.array([False, False, True, True, False, False])

    fixed, said = _dedupe_delivery(counts, index, host, ticks,
                                   dict(leadoff=leadoff, gap_before=gaps))
    assert "copies" in said
    assert fixed["counts"].shape == (1, 3)
    assert list(fixed["leadoff"]) == [0, 4, 0], "the mark stayed with its sample"
    assert list(fixed["gap_before"]) == [False, True, False]


def test_a_capture_label_cannot_walk_out_of_its_directory():
    """A label reaches this library from a command line, a web request, or
    a filename somebody typed. Joined to a path unchecked, it reads and
    writes wherever it likes, and the Command Center hands it one from a
    browser."""
    root = use_tmp()
    prov.use_captures_dir(root / "captures")
    inside = (root / "captures").resolve()

    for bad in ["../escaped", "..", "a/b", "/etc/passwd", "", ".hidden", "x" * 200,
                "with space", "semi;colon", "null\x00byte"]:
        try:
            prov.check_label(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} was accepted as a capture label")

    for good in ["ok", "run_1", "2026-09-19.baseline", "a", "A1-b_c.d"]:
        assert prov.check_label(good) == good
        p = prov.capture_path(good, ".npz").resolve()
        assert str(p).startswith(str(inside)), f"{good!r} landed at {p}"

    # And the library's own entry points refuse one, rather than reading
    # whatever the path happened to reach.
    from intomind import export
    for call in (lambda: A.load("../escaped"), lambda: A.events("../escaped"),
                 lambda: export.export_npz("../escaped")):
        try:
            call()
        except ValueError as e:
            assert "label" in str(e)
        else:
            raise AssertionError("a label that walks out of the collection was accepted")


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
    if _TMP and _TMP.exists():
        shutil.rmtree(_TMP)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("failed:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
