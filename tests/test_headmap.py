"""The head map matches its source, and the picture reads correctly.

    python3 tests/test_headmap.py

The coordinates are the published `standard_1005` montage projected the way
MNE projects it, baked into `headmap.py` so this package has no MNE
dependency. The first test here is the one that matters: when MNE is
importable it regenerates the table and compares, so the baked copy cannot
drift from its source without somebody being told.

The rest check the picture, because a person has to be able to point at their
own head and hit the right dot: left is left, the midline is straight, and no
two positions land on each other.

WHAT IS DELIBERATELY NOT ASSERTED. That the vertex is the center, that the
inion is on the rim, or that left mirrors right. An earlier version of this
file asserted all three, they are all false of a real head, and the tests
passed anyway because the coordinates were derived from the same wrong idea.
"""
from __future__ import annotations
import math, pathlib, sys, traceback

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from intomind import headmap as H              # noqa: E402
from intomind import montage as M              # noqa: E402
from intomind import placement as P            # noqa: E402


def test_the_baked_table_still_matches_the_published_montage():
    """The whole point of baking it is that nobody has to install MNE to draw
    a head. The whole risk of baking it is that it silently rots."""
    try:
        import mne
        from mne.channels.layout import _find_topomap_coords
        from mne.defaults import HEAD_SIZE_DEFAULT
    except Exception:
        print("      (no MNE on this machine, cannot re-derive the table)")
        return
    mm = mne.channels.make_standard_montage("standard_1005")
    want = [n for n in mm.ch_names if n in M.TEN_TWENTY]
    info = mne.create_info(want, 1000., "eeg")
    info.set_montage(mm)
    xy = _find_topomap_coords(info, picks=None, sphere=None)
    fresh = {n: (round(float(x) / HEAD_SIZE_DEFAULT, 4),
                 round(float(y) / HEAD_SIZE_DEFAULT, 4))
             for n, (x, y) in zip(want, xy)}
    assert set(fresh) == set(H.XY), \
        f"positions differ: {sorted(set(fresh) ^ set(H.XY))}"
    for n, (x, y) in fresh.items():
        bx, by = H.XY[n]
        assert abs(x - bx) < 1e-4 and abs(y - by) < 1e-4, \
            f"{n}: baked {(bx, by)}, montage says {(x, y)}"


def test_odd_is_left_and_even_is_right():
    for name, (x, _y) in H.XY.items():
        tail = name[-1]
        if tail == "z":
            assert abs(x) < 0.02, f"{name} is on the midline"
        elif tail.isdigit():
            n = int(name[-2:]) if name[-2:].isdigit() else int(tail)
            if n % 2:
                assert x < 0, f"{name} is odd and belongs on the left"
            else:
                assert x > 0, f"{name} is even and belongs on the right"


def test_the_midline_runs_front_to_back_in_order():
    order = ["Fpz", "AFz", "Fz", "FCz", "Cz", "CPz", "Pz", "POz", "Oz", "Iz"]
    ys = [H.XY[n][1] for n in order]
    assert ys == sorted(ys, reverse=True), list(zip(order, ys))


def test_a_row_walks_outward_from_the_midline():
    for row in (("Fz", "F1", "F3", "F5", "F7"),
                ("Cz", "C1", "C3", "C5", "T7"),
                ("Pz", "P1", "P3", "P5", "P7")):
        xs = [H.XY[n][0] for n in row]
        assert xs == sorted(xs, reverse=True), row


def test_the_ear_positions_are_the_outermost_of_the_vocabulary():
    """A person pointing at the side of their own head must land on an ear
    position and not on something on the scalp cap."""
    anchors = {s["ten_ten"] for s in P.SCALP.values() if s["ten_ten"]}
    left = min(anchors, key=lambda n: H.XY[n][0])
    right = max(anchors, key=lambda n: H.XY[n][0])
    assert left in ("FT9", "T9", "TP9"), left
    assert right in ("FT10", "T10", "TP10"), right


def test_a_drawing_is_told_how_far_out_to_leave_room():
    """Some positions are outside the rim and it is NOT the whole 9/10 ring:
    FT9 is at 1.096 while T9 and TP9 are just inside it. Guessing which would
    have been wrong twice, so REACH is measured off the table."""
    reach = max(math.hypot(*v) for v in H.XY.values())
    assert reach > H.RIM, "nothing is outside the rim, so REACH is pointless"
    assert reach <= H.REACH, f"REACH says {H.REACH}, the table reaches {reach}"
    assert math.hypot(*H.XY["FT9"]) > H.RIM > math.hypot(*H.XY["TP9"])


def test_no_two_positions_land_on_each_other():
    # A picker cannot offer two dots a click can never tell apart.
    assert len(set(H.XY.values())) == len(H.XY)


def test_every_position_is_a_real_ten_ten_name():
    assert set(H.XY) <= M.TEN_TWENTY


def test_every_scalp_position_in_the_vocabulary_can_be_drawn():
    # The map is the picker for `placement.SCALP`. A position with nowhere to
    # go would be unreachable by mouse and reachable by nothing else.
    missing = sorted(s["ten_ten"] for s in P.SCALP.values()
                     if s["ten_ten"] and s["ten_ten"] not in H.XY)
    assert missing == []


def test_a_click_picks_the_nearest_position():
    assert H.nearest(*H.XY["Cz"]) == "Cz"
    assert H.nearest(*H.XY["T7"]) == "T7"
    assert H.nearest(-0.74, 0.66) == "F7"


def test_a_click_can_be_held_to_the_positions_on_offer():
    # A picker whose vocabulary is 32 positions must not answer with a 33rd.
    among = {"T7", "T8", "Cz"}
    assert H.nearest(-0.5, 0.4, among=among) == "T7"
    assert H.nearest(0.02, 0.9, among=among) == "Cz"


def test_a_click_always_picks_something():
    # Anywhere, including well outside the head. A picker that ignores a
    # click reads as broken.
    assert H.nearest(5.0, -5.0) is not None
    assert H.nearest(0, 0, among=set()) is None


def test_an_unplaceable_name_says_so_rather_than_guessing():
    assert H.xy("Cz") == H.XY["Cz"]
    assert H.xy("not-a-position") is None
    assert H.xy("Nz") is None, \
        "the published montage has no nasion; do not invent one"


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
