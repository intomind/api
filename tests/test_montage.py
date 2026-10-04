"""The montage as data: schema, selection by identity, and back-fill honesty.

    python3 tests/test_montage.py

The point of structuring the montage is that an analysis can ask what a channel
IS instead of where it sits, because index does not survive a change of
instrument. So the tests that matter here are the ones that fail if selection
silently falls back to position, and the ones that fail if a montage nobody
verified becomes indistinguishable from one the instrument recorded.
"""
from __future__ import annotations
import json, pathlib, shutil, sys, tempfile, traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from intomind import montage as M           # noqa: E402
from intomind import provenance as prov     # noqa: E402

_TMP: pathlib.Path | None = None


def use_tmp():
    global _TMP
    _TMP = pathlib.Path(tempfile.mkdtemp(prefix="intomind_mtg_"))
    prov.CAPTURES = _TMP
    prov.MONTAGE_BACKFILL = _TMP / "MONTAGE_BACKFILL.json"


def _bipolar_pair():
    """The bench as it is: two bipolar temple pairs, forehead bias."""
    return M.build(
        [M.channel(1, M.BIPOLAR,
                   M.site("L-temple-ant"), M.site("L-temple-post"),
                   label="L temporal"),
         M.channel(2, M.BIPOLAR,
                   M.site("R-temple-ant"), M.site("R-temple-post"),
                   label="R temporal")],
        electrode="dry gold comb", bias=M.site("forehead"))


def _referential_8():
    """A build this bench does not have yet, which is the point of the schema."""
    ref = M.site("M1", ten_twenty="M1")
    names = ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"]
    return M.build(
        [M.channel(i + 1, M.REFERENTIAL, M.site(n, ten_twenty=n), ref,
                   label=f"{n}-M1")
         for i, n in enumerate(names)],
        electrode="wet Ag/AgCl", reference=ref, bias=M.site("Fpz",
                                                            ten_twenty="Fpz"))


# ---------------------------------------------------------------- schema
def test_one_schema_expresses_bipolar_and_referential():
    """A two-channel bipolar layout and an eight-channel referential one are the
    same shape. If they are not, the corpus needs special cases per device and
    the two can never be compared."""
    b, r = _bipolar_pair(), _referential_8()
    assert M.channel_count(b) == 2 and M.channel_count(r) == 8
    assert M.derivations(b) == {M.BIPOLAR}
    assert M.derivations(r) == {M.REFERENTIAL}


def test_a_referential_channel_must_agree_with_the_named_reference():
    """A montage claiming C3-M1 while naming M2 as its reference describes no
    real recording. It is refused, not stored."""
    ref, other = M.site("M1", ten_twenty="M1"), M.site("M2", ten_twenty="M2")
    try:
        M.build([M.channel(1, M.REFERENTIAL, M.site("C3", ten_twenty="C3"),
                           other)],
                electrode="wet", reference=ref)
    except M.MontageError as e:
        assert "reference" in str(e), e
        return
    raise AssertionError("a contradictory reference was accepted")


def test_referential_without_a_reference_is_refused():
    try:
        M.build([M.channel(1, M.REFERENTIAL, M.site("C3", ten_twenty="C3"),
                           M.site("M1", ten_twenty="M1"))], electrode="wet")
    except M.MontageError:
        return
    raise AssertionError("a referential montage with no reference was accepted")


def test_channel_indices_must_be_dense_and_one_based():
    """Indices tie the structure to the rows of `counts`. A gap means one of
    them does not."""
    try:
        M.build([M.channel(1, M.BIPOLAR, M.site("a"), M.site("b")),
                 M.channel(3, M.BIPOLAR, M.site("c"), M.site("d"))],
                electrode="dry")
    except M.MontageError as e:
        assert "1..n" in str(e), e
        return
    raise AssertionError("a montage with a hole in its indices was accepted")


def test_a_free_text_position_cannot_pose_as_a_standard_one():
    """`ten_twenty` answers 'is this a real O1'. A typo must not be able to
    make that answer yes."""
    try:
        M.site("left temple", ten_twenty="O11")
    except M.MontageError:
        pass
    else:
        raise AssertionError("a non-standard 10-20 label was accepted")
    # and free text with no standard equivalent is perfectly legal
    s = M.site("L-temple-ant")
    assert s["ten_twenty"] is None


# ---------------------------------------------------------------- selection
def test_channels_are_found_by_what_they_are_not_by_index():
    r = _referential_8()
    o1 = M.find(r, ten_twenty="O1")
    assert len(o1) == 1 and o1[0]["index"] == 7
    assert M.indices(o1) == [6]                    # 0-based row into `counts`


def test_selection_matches_either_pole():
    """'Does this recording see M1' is a question about the montage, not about
    which side of the subtraction M1 landed on."""
    r = _referential_8()
    assert len(M.find(r, ten_twenty="M1")) == 8    # the shared reference


def test_a_montage_without_the_site_returns_nothing_rather_than_channel_one():
    """The failure this schema exists to prevent: asking for occipital on a
    temple montage must return nothing, never a plausible default."""
    b = _bipolar_pair()
    assert M.find(b, ten_twenty="O1") == []
    assert M.indices(M.find(b, ten_twenty="O1")) == []


def test_describe_regenerates_prose_from_structure():
    """Kept so prose and structure can be cross-checked; a silent disagreement
    between them is the failure mode this module exists to remove."""
    d = M.describe(_bipolar_pair())
    assert "ch1 = L-temple-ant/L-temple-post (bipolar)" in d
    assert "bias = forehead" in d
    assert M.describe(None) == "unknown"


def test_a_recorded_montage_is_not_confused_with_a_backfilled_one():
    """The distinction is the whole reason the back-fill is a sidecar."""
    m = _bipolar_pair()
    (prov.CAPTURES / "rec.meta.json").write_text(json.dumps(
        dict(label="rec", montage=m)))
    (prov.CAPTURES / "old.meta.json").write_text(json.dumps(
        dict(label="old", electrode_montage="ch1 = ...")))
    prov.MONTAGE_BACKFILL.write_text(json.dumps(
        {"old": dict(montage=m, source="site.json", marked_utc="2026-08-06")}))

    assert prov.montage_for("rec")["provenance"] == "recorded"
    b = prov.montage_for("old")
    assert b["provenance"] == "backfilled"
    assert b["source"] == "site.json"
    assert b["montage"]["channels"][0]["index"] == 1


def test_an_undescribed_capture_is_unknown_not_defaulted():
    (prov.CAPTURES / "bare.meta.json").write_text(json.dumps(dict(label="bare")))
    info = prov.montage_for("bare")
    assert info["provenance"] == "unknown"
    assert info["montage"] is None


def test_the_manifest_records_a_mismatch_rather_than_hiding_it():
    """Two channels described, eight recorded: every per-channel result from
    that capture is suspect and the capture has to say so itself."""
    site = dict(montage=_bipolar_pair())
    assert prov._montage_fields(site, 2)["montage_matches_device"] is True
    assert prov._montage_fields(site, 8)["montage_matches_device"] is False
    # a device that does not report a channel count cannot be checked against
    assert prov._montage_fields(site, None)["montage_matches_device"] is None


def test_a_malformed_montage_is_reported_not_dropped():
    """A capture whose site config is broken must say the montage was broken,
    not silently record no montage at all."""
    bad = dict(montage=dict(electrode="dry", channels=[
        dict(index=1, derivation="telepathy",
             positive=dict(name="a"), negative=dict(name="b"))]))
    f = prov._montage_fields(bad, 1)
    assert f["montage"] is None
    assert f["montage_error"] and "telepathy" in f["montage_error"]


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


if __name__ == "__main__":
    sys.exit(main())
