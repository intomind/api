"""Where the electrodes went, and what is under them.

`montage.py` says what a channel IS: a difference between two named sites. It
does not say where those sites are on a head, and until now nothing did. The
five site names every capture on this machine carries -- `L-temple-ant`,
`R-temple-post`, `forehead` and their pairs -- are free text somebody typed
into one bench's config file. They are not comparable across devices, they are
not checkable, and they cannot be picked from a list.

This module is that list, and the model underneath it.

**Two vocabularies, because they are two kinds of claim.** Where an electrode
sits is something you OBSERVED. What it is over is something you INFERRED. One
merged list makes a record unable to say which of the two was written down, so
they are separate fields: `SCALP` is picked, `REGIONS` is filled in from it and
may be overridden. The question this answers, as it was asked: "whether
to use common face and scalp and head anatomy or cortical anatomy or both".
Both, in that order, with only the first one typed.

**Two words, and they mean different things.** A LAYOUT is what a device gives
you: how many electrodes it has and how they are wired into channels. It does
not change when you move the thing on your head. A PLACEMENT is where you put
it, and how far you turned it.

A layout is electrical. Take two channels, both bipolar,
four recording electrodes and a bias that may or may not be driven. The
IntoMind One's is geometric: four single-ended channels around a center
reference, placed as one unit. Same object here, different contents.

**Rotation is recorded, not computed.** A headband turns, and turning it moves
every electrode to a different scalp position. This module does not derive
those positions from an angle, because doing so would require a geometric model
of a head and of each band, and inventing one would produce coordinates that
look measured. Instead a device declares its own PRESETS: complete slot-to-site
mappings, each carrying the rotation it corresponds to. The positions are
stated, the angle documents them, and a preset can be checked.

**This module names no device.** The vocabulary and the machinery live here
because every device needs them; the layouts and presets live in each device's
own repository, where the person who owns the hardware states them.
"""
from __future__ import annotations

from . import montage as M

# --------------------------------------------------------------------- sides
LEFT, RIGHT, MIDLINE = "left", "right", "midline"
SIDES = (LEFT, RIGHT, MIDLINE)


class PlacementError(ValueError):
    """A placement that cannot describe a real head. Raised, never guessed."""


# ------------------------------------------------------------------- regions
#
# WHAT IS UNDER AN ELECTRODE, at the granularity that was asked for: "not quite as
# granular as the DK atlas but almost, like for example, brocas area is too
# granular, but right temporoparietal junction, left dlpfc, temporal pole,
# temple, behind left ear".
#
# Desikan-Killiany is a CORTICAL parcellation from MRI: 34 regions per
# hemisphere, defined in brain space, with no one-to-one correspondence to any
# scalp position. Broca's area is pars opercularis plus pars triangularis,
# which are two separate DK labels -- so "Broca's is too granular" means
# COARSER than DK, not almost as fine. That is the better instinct and it is
# what this list follows: fourteen cortical regions, unsided. The side comes
# from the SITE, so "left dlpfc" and "right tpj" are composed rather than
# enumerated, and the list stays readable.
#
# Medial and deep structures are deliberately absent. Anterior cingulate and
# medial prefrontal are not attributable from surface EEG at any honest
# confidence, and offering them in a picker invites a claim the instrument
# cannot support.
CORTICAL = "cortical"
MUSCLE = "muscle"
OCULAR = "ocular"
BONE = "bone"

REGIONS = {}


def _region(key: str, human: str, kind: str = CORTICAL) -> None:
    REGIONS[key] = dict(key=key, human=human, kind=kind)


_region("frontal-pole", "frontal pole")
_region("dlpfc", "dorsolateral prefrontal")
_region("vlpfc", "ventrolateral prefrontal")     # Broca's lives in here
_region("orbitofrontal", "orbitofrontal")
_region("premotor", "premotor and supplementary motor")
_region("motor", "primary motor")
_region("somatosensory", "primary somatosensory")
_region("superior-parietal", "superior parietal")
_region("tpj", "inferior parietal and temporoparietal junction")
_region("temporal-pole", "temporal pole")
_region("superior-temporal", "superior temporal")
_region("inferior-temporal", "middle and inferior temporal")
_region("lateral-occipital", "lateral occipital")
_region("medial-occipital", "medial occipital")
# Not cortex, and named anyway, because they are what the electrode actually
# picks up. On a temple electrode the muscle is not a caveat, it is often the
# loudest thing there, and a motion decoder can score no better than its own
# muscle-band control. A field rendering that channel as "left DLPFC" and
# nothing else would be the most misleading string in the record.
_region("temporalis", "temporalis (jaw muscle)", MUSCLE)
_region("frontalis", "frontalis (brow muscle)", MUSCLE)
_region("ocular", "eye movement", OCULAR)
_region("mastoid-bone", "mastoid bone, little cortex beneath", BONE)


# --------------------------------------------------------------------- scalp
#
# WHERE A PERSON CAN PUT AN ELECTRODE, in words they can act on without a tape
# measure. Each zone carries its nearest 10-10 anchor per side, which is what
# makes two recordings comparable: the human name is for choosing, the 10-10
# code is for pooling. Hemisphere is in both -- explicitly here, and in 10-10
# by convention, where odd is left, even is right and z is midline.
#
# `over` is ordered by what dominates, not alphabetically.
_ZONES = [
    dict(key="forehead-outer", human="outer forehead, above the brow's outer end",
         sides=(LEFT, RIGHT), ten_ten=dict(left="AF7", right="AF8"),
         over=["frontal-pole", "dlpfc", "ocular", "frontalis"]),
    dict(key="forehead-inner", human="inner forehead, above the brow",
         sides=(LEFT, RIGHT), ten_ten=dict(left="Fp1", right="Fp2"),
         over=["frontal-pole", "orbitofrontal", "ocular", "frontalis"]),
    dict(key="forehead-center", human="center of the forehead",
         sides=(MIDLINE,), ten_ten=dict(midline="Fpz"),
         over=["frontal-pole", "ocular", "frontalis"]),
    dict(key="hairline", human="forehead at the hairline",
         sides=(LEFT, RIGHT, MIDLINE),
         # F5/F6 rather than F7/F8: the hairline above the outer brow sits
         # higher and more medial than the temple, and giving both the same
         # anchor would make two different placements indistinguishable to
         # anything pooling by position.
         ten_ten=dict(left="F5", right="F6", midline="Fz"),
         over=["dlpfc", "vlpfc", "frontalis"]),
    dict(key="temple-front", human="temple, front",
         sides=(LEFT, RIGHT), ten_ten=dict(left="F7", right="F8"),
         over=["vlpfc", "temporalis", "temporal-pole"]),
    dict(key="temple-back", human="temple, back",
         sides=(LEFT, RIGHT), ten_ten=dict(left="FT7", right="FT8"),
         over=["temporalis", "superior-temporal", "vlpfc"]),
    dict(key="in-front-of-ear", human="in front of the ear",
         sides=(LEFT, RIGHT), ten_ten=dict(left="FT9", right="FT10"),
         over=["temporalis", "temporal-pole"]),
    dict(key="above-ear", human="above the ear",
         sides=(LEFT, RIGHT), ten_ten=dict(left="T7", right="T8"),
         over=["superior-temporal", "temporalis"]),
    dict(key="behind-ear", human="behind the ear, on the bone",
         sides=(LEFT, RIGHT), ten_ten=dict(left="TP9", right="TP10"),
         over=["mastoid-bone", "inferior-temporal"]),
    dict(key="behind-above-ear", human="behind and above the ear",
         sides=(LEFT, RIGHT), ten_ten=dict(left="TP7", right="TP8"),
         over=["tpj", "superior-temporal"]),
    dict(key="side-of-head", human="side of the head, above and forward of the ear",
         sides=(LEFT, RIGHT), ten_ten=dict(left="C5", right="C6"),
         over=["somatosensory", "motor"]),
    dict(key="crown-side", human="crown, off to the side",
         sides=(LEFT, RIGHT), ten_ten=dict(left="C3", right="C4"),
         over=["motor", "somatosensory", "premotor"]),
    dict(key="top-of-head", human="top of the head",
         sides=(MIDLINE,), ten_ten=dict(midline="Cz"),
         over=["motor", "premotor"]),
    dict(key="back-of-head-side", human="back of the head, off to the side",
         sides=(LEFT, RIGHT), ten_ten=dict(left="P3", right="P4"),
         over=["superior-parietal", "tpj"]),
    dict(key="crown-back", human="crown, toward the back",
         sides=(MIDLINE,), ten_ten=dict(midline="Pz"),
         over=["superior-parietal"]),
    dict(key="back-of-head-low", human="back of the head, low",
         sides=(LEFT, RIGHT), ten_ten=dict(left="O1", right="O2"),
         over=["lateral-occipital", "medial-occipital"]),
    dict(key="back-of-head-center", human="back of the head, center",
         sides=(MIDLINE,), ten_ten=dict(midline="Oz"),
         over=["medial-occipital"]),
    dict(key="nape", human="the bump at the base of the skull",
         sides=(MIDLINE,), ten_ten=dict(midline="Iz"),
         over=["medial-occipital"]),
]

_SIDE_WORD = {LEFT: "left", RIGHT: "right", MIDLINE: ""}

SCALP: dict[str, dict] = {}
for _z in _ZONES:
    for _side in _z["sides"]:
        _tt = _z["ten_ten"].get(_side)
        if _tt is not None and _tt not in M.TEN_TWENTY:
            raise PlacementError(f"{_z['key']}/{_side}: {_tt!r} is not a "
                                 f"10-20/10-10 position")
        _key = (f"{_side}-{_z['key']}" if _side != MIDLINE else _z["key"])
        _human = (f"{_SIDE_WORD[_side]} {_z['human']}".strip()
                  if _side != MIDLINE else _z["human"])
        SCALP[_key] = dict(key=_key, zone=_z["key"], side=_side, human=_human,
                           ten_ten=_tt, over=list(_z["over"]))
del _z, _side, _tt, _key, _human

# NO TWO POSITIONS MAY SHARE AN ANCHOR. The 10-10 code is what pooling
# compares, so two human names on one code are two placements a corpus cannot
# tell apart. Checked at import, because a vocabulary that has to be tested to
# be safe will eventually be extended by someone who does not run the tests.
_seen: dict[str, str] = {}
for _k, _s in SCALP.items():
    _a = _s["ten_ten"]
    if _a is None:
        continue
    if _a in _seen:
        raise PlacementError(f"{_k!r} and {_seen[_a]!r} both anchor to {_a}; "
                             f"nothing could tell them apart")
    _seen[_a] = _k
del _seen, _k, _s, _a


def scalp(key: str) -> dict:
    """One scalp position by key, or an error naming what is on offer."""
    try:
        return dict(SCALP[key])
    except KeyError:
        raise PlacementError(
            f"{key!r} is not a scalp position. There are {len(SCALP)}; "
            f"`placement.choices()` lists them.") from None


def choices(side: str | None = None) -> list[dict]:
    """Every scalp position, in head order, front to back. The dropdown."""
    order = {z["key"]: i for i, z in enumerate(_ZONES)}
    rows = [s for s in SCALP.values() if side is None or s["side"] == side]
    return sorted(rows, key=lambda s: (order[s["zone"]],
                                       SIDES.index(s["side"])))


def regions_under(scalp_key: str) -> list[dict]:
    """What is under a scalp position, side carried down from the site.

    This is the autofill, as it was asked for: "selecting scalp anatomy should
    come first which autopopulates the cortical anatomy, which can be changed
    if the user wants to."

    IT IS AN INFERENCE AND SAYS SO. Every entry is marked `derived`, and the
    correspondence between a scalp position and the cortex beneath it is
    probabilistic, varies with head shape, and for a BIPOLAR channel does not
    strictly apply at all: a bipolar pair sees the field gradient between two
    sites, not the cortex under either one. The sites are still what a person
    needs in order to reason about the pair, so they are given, marked.
    """
    s = scalp(scalp_key)
    out = []
    for key in s["over"]:
        r = REGIONS[key]
        sided = (f"{s['side']} {r['human']}"
                 if s["side"] != MIDLINE and r["kind"] != OCULAR
                 else r["human"])
        out.append(dict(region=key, human=sided, kind=r["kind"],
                        side=s["side"], derived=True))
    return out


# ------------------------------------------------------------------ layouts
RECORDING, BIAS, REFERENCE = "recording", "bias", "reference"
SLOT_ROLES = (RECORDING, BIAS, REFERENCE)


def slot(key: str, human: str, role: str = RECORDING, *,
         name: str | None = None, angle_deg: float | None = None) -> dict:
    """One electrode position ON THE DEVICE, before it is put on a head.

    `name` is what this electrode is called on this device (`IN1P`,
    `L-temple-ant`). It is carried into the montage as the site name, so a
    device that has been recording under one set of names keeps them and its
    existing captures stay comparable.

    `angle_deg` documents where the slot sits on a rotatable layout, measured
    from its reference orientation. Nothing is computed from it; it is there so
    that a preset claiming a 90 degree rotation can be read against the
    geometry it claims to describe.
    """
    if role not in SLOT_ROLES:
        raise PlacementError(f"slot role must be one of {SLOT_ROLES}, "
                             f"got {role!r}")
    return dict(key=key, human=human, role=role, name=name or key,
                angle_deg=angle_deg)


def layout(key: str, human: str, slots: list[dict], channels: list[dict], *,
            rotatable: bool = False, presets: list[dict] | None = None,
            electrode: str | None = None, notes: str | None = None) -> dict:
    """What a device can offer as a montage: its slots and its wiring.

    `channels` are dicts of `index`, `derivation`, `positive`, `negative`,
    where the two poles name SLOTS rather than sites. The head is not in this
    object at all -- that is the point. A layout is the same whether it is
    worn on the temples or over the ears.
    """
    f = dict(key=key, human=human, slots=list(slots), channels=list(channels),
             rotatable=bool(rotatable), presets=list(presets or []),
             electrode=electrode, notes=notes)
    validate_layout(f)
    return f


def validate_layout(f: dict) -> None:
    """Refuse a layout that could not be wired."""
    slots = {s["key"]: s for s in (f.get("slots") or [])}
    if not slots:
        raise PlacementError("a layout with no slots describes nothing")
    chans = f.get("channels") or []
    if not chans:
        raise PlacementError(f"{f.get('key')!r} declares no channels")
    idx = [c["index"] for c in chans]
    if sorted(idx) != list(range(1, len(idx) + 1)):
        raise PlacementError(f"channel indices must be 1..n with no gaps, "
                             f"got {idx}")
    refs = [s for s in slots.values() if s["role"] == REFERENCE]
    for c in chans:
        if c["derivation"] not in M.DERIVATIONS:
            raise PlacementError(f"ch{c['index']}: bad derivation "
                                 f"{c['derivation']!r}")
        for pole in ("positive", "negative"):
            if c.get(pole) not in slots:
                raise PlacementError(f"ch{c['index']}: {pole} names slot "
                                     f"{c.get(pole)!r}, which this layout "
                                     f"does not have")
        if c["derivation"] == M.REFERENTIAL:
            if len(refs) != 1:
                raise PlacementError(
                    f"ch{c['index']} is single-ended, so this layout needs "
                    f"exactly one reference slot; it has {len(refs)}")
            if c["negative"] != refs[0]["key"]:
                raise PlacementError(
                    f"ch{c['index']} is single-ended against "
                    f"{c['negative']!r}, but the reference slot is "
                    f"{refs[0]['key']!r}")
    for p in f.get("presets") or []:
        _validate_preset(f, p, slots)


def _validate_preset(f: dict, p: dict, slots: dict) -> None:
    if not p.get("key"):
        raise PlacementError(f"{f['key']}: a preset needs a key")
    sites = p.get("sites") or {}
    missing = [k for k, s in slots.items()
               if k not in sites and s["role"] != BIAS]
    if missing:
        raise PlacementError(f"{f['key']}/{p['key']}: no site for slot(s) "
                             f"{', '.join(sorted(missing))}")
    for k, v in sites.items():
        if k not in slots:
            raise PlacementError(f"{f['key']}/{p['key']}: {k!r} is not a slot")
        scalp(v)                                  # raises if not a position
    if p.get("rotation_deg") is not None and not f.get("rotatable"):
        raise PlacementError(f"{f['key']}/{p['key']} states a rotation, but "
                             f"{f['key']} is not rotatable")


def preset(key: str, human: str, sites: dict, *,
           rotation_deg: float | None = None, note: str | None = None) -> dict:
    """A named, complete slot-to-site mapping: one row of the dropdown.

    A rotatable layout expresses its rotations AS PRESETS rather than as an
    angle the code turns into positions. Turning a band moves every electrode,
    and deriving where they land needs a geometric model of a head and of that
    band; inventing one would produce positions that look measured. So the
    positions are stated by whoever owns the hardware, and `rotation_deg`
    documents which turn they correspond to.
    """
    return dict(key=key, human=human, sites=dict(sites),
                rotation_deg=rotation_deg, note=note)


def layout_from(d: dict | None) -> dict | None:
    """A layout as declared in a device's own config, validated."""
    if not d:
        return None
    validate_layout(d)
    return d


# ----------------------------------------------------------------- placement
def placement(lay: dict, sites: dict | None = None, *, preset: str | None = None,
              rotation_deg: float | None = None, over: dict | None = None,
              note: str = "", bias_driven: bool = True) -> dict:
    """Where this layout was put, on this head, this time.

    `sites` maps slot key to scalp key and may be omitted when `preset` names
    one the layout declares. `over` overrides the autofilled cortical regions
    for named slots; the autofill was asked to be changeable, and an
    override is recorded as `derived: False` so a hand-entered region can never
    be mistaken for one the table produced.

    `note` is free text and is never parsed. It is where the thing that does
    not fit any vocabulary goes, which is most of what is worth writing down
    the first time a montage is tried.
    """
    chosen = None
    if preset is not None:
        chosen = next((p for p in lay.get("presets") or []
                       if p["key"] == preset), None)
        if chosen is None:
            have = ", ".join(p["key"] for p in lay.get("presets") or [])
            raise PlacementError(f"{lay['key']} has no preset {preset!r}"
                                 + (f"; it has {have}" if have else ""))
    sites = dict(sites or (chosen or {}).get("sites") or {})
    if rotation_deg is None and chosen is not None:
        rotation_deg = chosen.get("rotation_deg")
    p = dict(layout=lay["key"], preset=preset, sites=sites,
             rotation_deg=rotation_deg, over=dict(over or {}),
             note=note or "", bias_driven=bool(bias_driven))
    validate_placement(lay, p)
    return p


def validate_placement(lay: dict, p: dict) -> None:
    """Refuse a placement that does not put every electrode somewhere."""
    if p.get("layout") != lay.get("key"):
        raise PlacementError(f"this placement is for layout "
                             f"{p.get('layout')!r}, not {lay.get('key')!r}")
    slots = {s["key"]: s for s in lay["slots"]}
    sites = p.get("sites") or {}
    for k in sites:
        if k not in slots:
            raise PlacementError(f"{k!r} is not a slot on {lay['key']}")
        scalp(sites[k])
    for k, s in slots.items():
        if k in sites:
            continue
        # A bias that is not driven need not be anywhere. Every recording
        # electrode must be, or the record does not say where the signal came
        # from, which is the whole point of writing it down.
        if s["role"] == BIAS and not p.get("bias_driven", True):
            continue
        raise PlacementError(
            f"{lay['key']}: no site for slot {k!r} ({s['human']})")
    for k in p.get("over") or {}:
        if k not in slots:
            raise PlacementError(f"override names {k!r}, which is not a slot")
        for r in p["over"][k]:
            if r not in REGIONS:
                raise PlacementError(f"{r!r} is not a region; there are "
                                     f"{len(REGIONS)}")
    if p.get("rotation_deg") is not None and not lay.get("rotatable"):
        raise PlacementError(f"{lay['key']} is not rotatable, so a rotation "
                             f"of {p['rotation_deg']} means nothing")


def resolve(lay: dict, p: dict) -> dict:
    """The placement with every slot's site and regions filled in."""
    validate_placement(lay, p)
    slots = {s["key"]: s for s in lay["slots"]}
    out = []
    for key, s in slots.items():
        site_key = (p.get("sites") or {}).get(key)
        if site_key is None:
            continue
        sc = scalp(site_key)
        if key in (p.get("over") or {}):
            regions = [dict(region=r, human=REGIONS[r]["human"],
                            kind=REGIONS[r]["kind"], side=sc["side"],
                            derived=False) for r in p["over"][key]]
        else:
            regions = regions_under(site_key)
        out.append(dict(slot=key, role=s["role"], name=s["name"],
                        human=s["human"], scalp=sc, over=regions))
    return dict(layout=lay["key"], preset=p.get("preset"),
                rotation_deg=p.get("rotation_deg"), note=p.get("note", ""),
                bias_driven=p.get("bias_driven", True), slots=out)


def to_montage(lay: dict, p: dict, *, electrode: str | None = None,
               prose: str | None = None, source: str | None = None) -> dict:
    """The placement as a `montage.py` montage, so everything downstream works.

    This is the join. `montage` is what an analysis selects channels with and
    what every capture already records; `placement` is how a person states it.
    Producing one from the other means the picker cannot drift from the record,
    and a device that has been declaring its montage by hand can be checked
    against its placement instead of trusted.
    """
    r = resolve(lay, p)
    by_slot = {s["slot"]: s for s in r["slots"]}

    def _site(key: str) -> dict:
        s = by_slot[key]
        return M.site(s["name"], ten_twenty=s["scalp"]["ten_ten"],
                      description=s["scalp"]["human"])

    ref_slot = next((s for s in lay["slots"] if s["role"] == REFERENCE), None)
    chans = [M.channel(c["index"], c["derivation"],
                       _site(c["positive"]), _site(c["negative"]),
                       label=c.get("label"))
             for c in sorted(lay["channels"], key=lambda x: x["index"])]
    bias_slot = next((s for s in lay["slots"] if s["role"] == BIAS), None)
    bias = (_site(bias_slot["key"])
            if bias_slot and bias_slot["key"] in by_slot
            and r["bias_driven"] else None)
    return M.build(chans, electrode=electrode or lay.get("electrode") or "unknown",
                   bias=bias,
                   reference=_site(ref_slot["key"]) if ref_slot else None,
                   # WITHOUT THE NOTE AND WITHOUT THE ROTATION. A corpus
                   # decides whether two recordings are the same instrument by
                   # comparing their montages, so anything in here that is not
                   # about the electrodes splits the corpus: two sessions whose
                   # notes differed by a word would have refused to pool. The
                   # note lives on the placement, which is where it belongs.
                   prose=(prose if prose is not None
                          else describe(lay, p, extras=False)),
                   source=source)


def describe(lay: dict, p: dict, *, extras: bool = True) -> str:
    """The placement as a sentence, regenerated from the structure.

    A CLUSTER IS DESCRIBED ONCE. When every recording electrode is at the same
    scalp position -- which is what a localized array looks like through a
    vocabulary coarser than its own spacing -- listing each channel produces
    "left crown / left crown" four times over, which says nothing and reads as
    a bug. The position is stated once and the electrodes are told apart by
    where they sit within the cluster, which is the only thing that actually
    distinguishes them.
    """
    r = resolve(lay, p)
    by_slot = {s["slot"]: s for s in r["slots"]}
    rec = [s for s in r["slots"] if s["role"] == RECORDING]
    ref = next((s for s in r["slots"] if s["role"] == REFERENCE), None)
    spots = {s["scalp"]["key"] for s in rec}
    parts = []
    if len(rec) > 1 and len(spots) == 1:
        parts.append(f"{len(lay['channels'])} channels in one cluster at "
                     f"{rec[0]['scalp']['human']}: "
                     + ", ".join(s["human"].split(", ")[-1] for s in rec))
        if ref is not None:
            parts.append("referenced to the "
                         + ref["human"].split(", ")[-1])
    else:
        for c in sorted(lay["channels"], key=lambda x: x["index"]):
            a = by_slot[c["positive"]]["scalp"]["human"]
            b = by_slot[c["negative"]]["scalp"]["human"]
            parts.append(f"ch{c['index']} = {a}, referenced to {b}"
                         if c["derivation"] == M.REFERENTIAL
                         else f"ch{c['index']} = {a} / {b}")
    bias_slot = next((s for s in lay["slots"] if s["role"] == BIAS), None)
    if bias_slot:
        parts.append("bias = " + (by_slot[bias_slot["key"]]["scalp"]["human"]
                                  if bias_slot["key"] in by_slot
                                  and r["bias_driven"] else "not driven"))
    tail = ""
    if extras:
        if r["rotation_deg"]:
            tail += f" Rotated {r['rotation_deg']:g}°."
        if r["note"]:
            tail += f" {r['note']}"
    # SENTENCES, NOT CLAUSES. This string is read in a dialog and stored in
    # every capture's montage, and it is cheap to settle now and expensive
    # later: the montage prose is part of what a corpus compares when it
    # decides two recordings are the same instrument, so changing the joiner
    # after a session has been recorded would split the corpus at that date.
    # No capture carries a placement yet, checked 2026-08-20.
    # A part that labels something ("ch2 = ...", "bias = ...") stays lower
    # case after the stop, because it is a name and not the start of a
    # sentence. A part that IS a sentence gets its capital.
    said = [parts[0]] + [p if " = " in p[:14] else p[:1].upper() + p[1:]
                         for p in parts[1:]]
    return ". ".join(said) + "." + tail
