"""Where the electrodes went: the vocabulary, the layouts, and the refusals.

    python3 tests/test_placement.py

The point of a closed vocabulary is that a wrong answer becomes impossible
rather than unlikely, so most of these tests are refusals. The rest guard the
two properties the vocabulary is FOR: that a scalp position is comparable
across devices (one 10-10 anchor, and no two positions share one), and that a
cortical region can never be mistaken for something observed.
"""
from __future__ import annotations
import pathlib, sys, traceback

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from intomind import montage as M              # noqa: E402
from intomind import placement as P            # noqa: E402


# ----------------------------------------------------------- the vocabulary
def test_every_anchor_is_a_real_ten_ten_position():
    for key, s in P.SCALP.items():
        if s["ten_ten"] is not None:
            assert s["ten_ten"] in M.TEN_TWENTY, f"{key}: {s['ten_ten']}"


def test_no_two_positions_share_an_anchor():
    """The 10-10 code is what pooling compares, so two human names on one code
    are two placements a corpus cannot tell apart. `temple, front` and
    `forehead at the hairline` both anchored to F7 in the first draft."""
    seen = {}
    for key, s in P.SCALP.items():
        a = s["ten_ten"]
        if a is None:
            continue
        assert a not in seen, f"{key} and {seen[a]} both anchor to {a}"
        seen[a] = key


def test_every_region_a_position_names_exists():
    for key, s in P.SCALP.items():
        for r in s["over"]:
            assert r in P.REGIONS, f"{key} names region {r!r}"


def test_both_hemispheres_are_offered_for_every_lateral_position():
    zones = {}
    for s in P.SCALP.values():
        zones.setdefault(s["zone"], set()).add(s["side"])
    for zone, sides in zones.items():
        if sides == {P.MIDLINE}:
            continue
        assert {P.LEFT, P.RIGHT} <= sides, f"{zone} is offered on one side only"


def test_the_list_runs_front_to_back():
    """It is a dropdown a person reads. Alphabetical would put `above the ear`
    above `center of the forehead`, which is not how a head is shaped."""
    order = [s["zone"] for s in P.choices()]
    assert order[0] == "forehead-outer"
    assert order[-1] == "nape"
    # and each zone's entries are contiguous
    firsts = [order.index(z) for z in dict.fromkeys(order)]
    assert firsts == sorted(firsts)


def test_an_unknown_position_says_what_is_on_offer():
    try:
        P.scalp("left-eyebrow")
    except P.PlacementError as e:
        assert "not a scalp position" in str(e) and "choices()" in str(e)
    else:
        raise AssertionError("an invented position was accepted")


# -------------------------------------------------------------- the autofill
def test_the_side_comes_from_the_site_not_the_region_list():
    """Fourteen unsided regions and a sided site compose into "left dlpfc" and
    "right temporoparietal junction", which is what was asked for, without
    enumerating twenty-eight."""
    got = [r["human"] for r in P.regions_under("right-behind-above-ear")]
    assert "right inferior parietal and temporoparietal junction" in got
    got = [r["human"] for r in P.regions_under("left-hairline")]
    assert "left dorsolateral prefrontal" in got


def test_every_autofilled_region_is_marked_as_inferred():
    """Where the electrode sits was observed. What it is over was not, and a
    record that cannot tell the two apart is worse than one that omits the
    second."""
    for key in P.SCALP:
        for r in P.regions_under(key):
            assert r["derived"] is True, key


def test_the_temples_name_the_muscle_first():
    """On eight pooled sessions the walk-vs-rest decoder scored 0.608 and the
    muscle-band control scored 0.609. A temple electrode rendered as "left
    DLPFC" and nothing else would be the most misleading string in the
    record."""
    kinds = [r["kind"] for r in P.regions_under("left-temple-back")]
    assert P.MUSCLE in kinds
    assert kinds[0] == P.MUSCLE, "cortex is named before the muscle"


def test_an_override_is_not_marked_as_inferred():
    f = _headband()
    p = P.placement(f, preset="temples",
                    over={"l-front": ["dlpfc"]})
    slot = next(s for s in P.resolve(f, p)["slots"] if s["slot"] == "l-front")
    assert [r["region"] for r in slot["over"]] == ["dlpfc"]
    assert slot["over"][0]["derived"] is False


def test_an_override_naming_no_such_region_is_refused():
    f = _headband()
    try:
        P.placement(f, preset="temples", over={"l-front": ["brocas-area"]})
    except P.PlacementError as e:
        assert "not a region" in str(e)
    else:
        raise AssertionError("an invented region was accepted")


# ---------------------------------------------------------------- layouts
def _headband() -> dict:
    """A two-channel bipolar layout: four electrodes and a bias.

    Not the headband's geometry. The band is one way to wear this and the
    electrodes can go anywhere; what the device fixes is electrical.
    """
    return P.layout(
        "test-headband", "two bipolar pairs and a bias",
        slots=[P.slot("l-front", "left pair, front", name="L-temple-ant"),
               P.slot("l-back", "left pair, back", name="L-temple-post"),
               P.slot("r-front", "right pair, front", name="R-temple-ant"),
               P.slot("r-back", "right pair, back", name="R-temple-post"),
               P.slot("bias", "bias", P.BIAS, name="forehead")],
        channels=[dict(index=1, derivation=M.BIPOLAR, positive="l-front",
                       negative="l-back", label="L temporal"),
                  dict(index=2, derivation=M.BIPOLAR, positive="r-front",
                       negative="r-back", label="R temporal")],
        rotatable=True, electrode="dry gold comb",
        presets=[P.preset("temples", "across the temples", rotation_deg=0,
                          sites=dict(**{"l-front": "left-temple-front",
                                        "l-back": "left-temple-back",
                                        "r-front": "right-temple-front",
                                        "r-back": "right-temple-back",
                                        "bias": "forehead-center"}))])


def _cluster() -> dict:
    """The IntoMind One's lock: four single-ended channels around a center
    reference, placed as one unit, one unit per device.

    THE VOCABULARY IS COARSER THAN THIS DEVICE, and that is fine because the
    device is smaller than any word for a place on a head. A placement says
    where the CLUSTER is and nothing about which electrode sat where inside
    it, because there is nothing there to say.
    """
    return P.layout(
        "test-cluster", "four single-ended around a center reference",
        slots=[P.slot("n", "cluster, forward", name="ch1", angle_deg=0),
               P.slot("e", "cluster, right", name="ch2", angle_deg=90),
               P.slot("s", "cluster, back", name="ch3", angle_deg=180),
               P.slot("w", "cluster, left", name="ch4", angle_deg=270),
               P.slot("center", "cluster, center", P.REFERENCE, name="ref")],
        channels=[dict(index=i + 1, derivation=M.REFERENTIAL, positive=k,
                       negative="center", label=f"cluster {k}")
                  for i, k in enumerate(("n", "e", "s", "w"))],
        rotatable=True)


def test_a_single_ended_layout_with_no_reference_slot_is_refused():
    try:
        P.layout("bad", "no ref",
                  slots=[P.slot("a", "a"), P.slot("b", "b")],
                  channels=[dict(index=1, derivation=M.REFERENTIAL,
                                 positive="a", negative="b")])
    except P.PlacementError as e:
        assert "exactly one reference slot" in str(e)
    else:
        raise AssertionError("a single-ended layout with no reference passed")


def test_a_single_ended_channel_against_the_wrong_slot_is_refused():
    try:
        P.layout("bad", "wrong ref",
                  slots=[P.slot("a", "a"), P.slot("b", "b"),
                         P.slot("c", "c", P.REFERENCE)],
                  channels=[dict(index=1, derivation=M.REFERENTIAL,
                                 positive="a", negative="b")])
    except P.PlacementError as e:
        assert "reference slot is" in str(e)
    else:
        raise AssertionError("a channel referenced to a recording slot passed")


def test_a_channel_naming_a_slot_the_layout_lacks_is_refused():
    try:
        P.layout("bad", "ghost slot", slots=[P.slot("a", "a")],
                  channels=[dict(index=1, derivation=M.BIPOLAR, positive="a",
                                 negative="ghost")])
    except P.PlacementError as e:
        assert "does not have" in str(e)
    else:
        raise AssertionError("a channel wired to nothing passed")


def test_a_preset_that_leaves_a_recording_electrode_unplaced_is_refused():
    try:
        P.layout("bad", "half a preset",
                  slots=[P.slot("a", "a"), P.slot("b", "b")],
                  channels=[dict(index=1, derivation=M.BIPOLAR, positive="a",
                                 negative="b")],
                  presets=[P.preset("p", "p", sites={"a": "left-above-ear"})])
    except P.PlacementError as e:
        assert "no site for slot" in str(e) and "b" in str(e)
    else:
        raise AssertionError("a preset with an unplaced electrode passed")


def test_a_rotation_on_a_layout_that_does_not_turn_is_refused():
    try:
        P.layout("bad", "fixed",
                  slots=[P.slot("a", "a"), P.slot("b", "b")],
                  channels=[dict(index=1, derivation=M.BIPOLAR, positive="a",
                                 negative="b")],
                  rotatable=False,
                  presets=[P.preset("p", "p", rotation_deg=90,
                                    sites={"a": "left-above-ear",
                                           "b": "left-behind-ear"})])
    except P.PlacementError as e:
        assert "not rotatable" in str(e)
    else:
        raise AssertionError("a rotation was accepted on a fixed device")


# --------------------------------------------------------------- placement
def test_a_placement_must_put_every_recording_electrode_somewhere():
    f = _headband()
    try:
        P.placement(f, sites={"l-front": "left-temple-front",
                              "bias": "forehead-center"})
    except P.PlacementError as e:
        assert "no site for slot" in str(e)
    else:
        raise AssertionError("a placement with three electrodes nowhere passed")


def test_a_bias_that_is_not_driven_need_not_be_anywhere():
    """A layout may have "a bias elec (that doesnt
    necessarily need to be turned on)"."""
    f = _headband()
    p = P.placement(f, sites={"l-front": "left-temple-front",
                              "l-back": "left-temple-back",
                              "r-front": "right-temple-front",
                              "r-back": "right-temple-back"},
                    bias_driven=False)
    assert P.to_montage(f, p)["bias"] is None
    assert "not driven" in P.describe(f, p)


def test_a_placement_naming_an_unknown_preset_lists_the_ones_there_are():
    f = _headband()
    try:
        P.placement(f, preset="over-the-ears")
    except P.PlacementError as e:
        assert "no preset" in str(e) and "temples" in str(e)
    else:
        raise AssertionError("an invented preset was accepted")


def test_a_preset_fills_in_the_sites_and_the_rotation():
    f = _headband()
    p = P.placement(f, preset="temples")
    assert p["sites"]["l-front"] == "left-temple-front"
    assert p["rotation_deg"] == 0


# --------------------------------------------------------------- the join
def test_a_placement_becomes_a_montage_the_montage_module_accepts():
    f = _headband()
    m = P.to_montage(f, P.placement(f, preset="temples"))
    M.validate(m)
    assert M.channel_count(m) == 2
    assert M.derivations(m) == {M.BIPOLAR}
    assert M.find(m, site_name="L-temple-ant")[0]["index"] == 1
    # And it carries the anchors the hand-written montage never had.
    assert M.find(m, ten_twenty="F7")[0]["index"] == 1
    assert m["bias"]["name"] == "forehead"


def test_a_single_ended_cluster_becomes_a_valid_referential_montage():
    f = _cluster()
    here = {k: "left-crown-side" for k in ("n", "e", "s", "w", "center")}
    m = P.to_montage(f, P.placement(f, sites=here, rotation_deg=0))
    M.validate(m)
    assert M.channel_count(m) == 4
    assert M.derivations(m) == {M.REFERENTIAL}
    assert m["reference"]["name"] == "ref"
    # Every channel is at the same scalp position and they are still four
    # different measurements: the cluster is smaller than any word for it.
    assert {c["positive"]["ten_twenty"] for c in m["channels"]} == {"C3"}
    assert len({c["positive"]["name"] for c in m["channels"]}) == 4


def test_the_sentence_is_regenerated_from_the_structure():
    f = _headband()
    s = P.describe(f, P.placement(f, preset="temples"))
    assert "ch1 = left temple, front / left temple, back" in s
    assert "bias = center of the forehead" in s


def test_a_cluster_is_described_once_not_once_per_channel():
    """Listing four channels that are all at the same position gives "left
    crown / left crown" four times, which says nothing and reads as a bug."""
    lay = _cluster()
    here = {k: "left-crown-side" for k in ("n", "e", "s", "w", "center")}
    s = P.describe(lay, P.placement(lay, sites=here))
    assert s.count("left crown") == 1, s
    assert "4 channels in one cluster" in s
    assert "Referenced to the center" in s
    for within in ("forward", "right", "back", "left"):
        assert within in s, within


def test_a_rotation_and_a_note_reach_the_sentence():
    f = _cluster()
    here = {k: "left-crown-side" for k in ("n", "e", "s", "w", "center")}
    s = P.describe(f, P.placement(f, sites=here, rotation_deg=45,
                                  note="Gel, not dry."))
    assert "45" in s and "Gel, not dry." in s


def test_the_note_stays_out_of_the_generated_montage():
    """A corpus decides whether two recordings are the same instrument by
    comparing their montages. Anything in there that is not about the
    electrodes splits it, and two sessions whose notes differed by a word would
    have refused to pool."""
    lay = _headband()
    a = P.to_montage(lay, P.placement(lay, preset="temples", note="dry"))
    b = P.to_montage(lay, P.placement(lay, preset="temples", note="gel today"))
    assert a == b, "the note reached the montage"
    assert "dry" not in a["prose"] and "Rotated" not in a["prose"]
    # And it is still recorded, on the placement, where it belongs.
    assert P.resolve(lay, P.placement(lay, preset="temples",
                                      note="dry"))["note"] == "dry"


def test_moving_an_electrode_does_change_the_montage():
    """The other half of the same rule: what IS about the electrodes must
    split the corpus, or two different montages pool as one."""
    lay = _headband()
    a = P.to_montage(lay, P.placement(lay, preset="temples"))
    moved = dict(P.placement(lay, preset="temples")["sites"])
    moved["l-front"] = "left-above-ear"
    b = P.to_montage(lay, P.placement(lay, sites=moved))
    assert a != b


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
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
