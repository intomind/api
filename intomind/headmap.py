"""Where to draw a 10-20/10-10 position on a picture of a head.

WHAT THIS IS FOR. Picking an electrode position out of a list of thirty-two
names makes a person translate a place on their own head into a word. A map
does not. This module says where each 10-10 name goes on a flat, top-down
drawing, so an application can put a dot there and turn a click back into a
name.

**These numbers are never recorded.** What a capture stores is the position's
name. The coordinates decide where a dot is drawn and which dot a click is
nearest to, and nothing else.

WHERE THEY COME FROM, exactly. The published `standard_1005` montage, as
shipped with MNE:

    montage    mne.channels.make_standard_montage("standard_1005")
    file       mne/channels/data/montages/standard_1005.elc
    projected  mne.channels.layout._find_topomap_coords(info, picks=None,
                                                        sphere=None)
    scaled     divided by mne.defaults.HEAD_SIZE_DEFAULT, 0.095 m, so the
               head outline MNE draws is a radius of 1
    version    mne 1.9.0

The table below is that output, baked in so this package has no MNE
dependency. `tests/test_headmap.py` regenerates it and compares whenever MNE
is importable, so it cannot drift from its source unnoticed.

THE FIRST VERSION OF THIS FILE DERIVED THE POSITIONS instead, from the 10-10
system's construction on a symmetric sphere. As it was put: "make sure you
are not guessing at the electrode montage dimensions or anything. no guesswork
allowed, especially for determinable features." It was determinable, the file
was on this machine, and the derivation was wrong in ways worth naming,
because they are counterintuitive enough that the next reader will be tempted
to re-derive them the same way:

  - Cz IS NOT THE CENTER. It sits about a fifth of the head radius forward,
    because a head is longer behind the vertex than in front of it.
  - Iz is INSIDE the rim, not on it.
  - SOME POSITIONS SIT OUTSIDE THE RIM and it is not the whole 9/10 ring.
    FT9 is at 1.096 and F9, A1 and M1 further still, while T9 at 0.989 and
    TP9 at 0.964 are just inside. Guessing which would have been wrong twice,
    so `REACH` is measured off the table rather than reasoned about, and a
    drawing leaves room for it.
  - Left and right are NOT mirror images. T7 is at -0.7512 and T8 at +0.7249.
    The published head is not symmetric and this table does not make it so.

Axes: x is to the RIGHT, y is toward the FRONT, MNE's head outline is a radius
of 1, and a few positions sit outside it. Screen coordinates run the other way
vertically, so a drawing flips y.
"""
from __future__ import annotations

from . import montage as M

#: MNE's head outline. NOT the largest radius here: see the note above.
RIM = 1.0

#: The furthest any position sits from the center, which is what a drawing has
#: to leave room for.
REACH = 1.16

#: Every 10-10 position this module can draw, as (x, y). Generated, not typed.
XY: dict[str, tuple[float, float]] = {
    "A1": (-1.1505, 0.034), "A2": (1.1109, 0.026), "AF3": (-0.2926, 0.9075),
    "AF4": (0.2843, 0.9119), "AF7": (-0.5236, 0.9211), "AF8": (0.504, 0.928),
    "AFz": (-0.0099, 0.9113), "C1": (-0.2593, 0.18), "C2": (0.2451, 0.18),
    "C3": (-0.4779, 0.1662), "C4": (0.4643, 0.1675), "C5": (-0.6284, 0.1477),
    "C6": (0.6247, 0.1507), "CP1": (-0.2542, -0.0735), "CP2": (0.2476, -0.074),
    "CP3": (-0.4619, -0.0839), "CP4": (0.4561, -0.0845), "CP5": (-0.6173, -0.1011),
    "CP6": (0.6161, -0.1019), "CPz": (-0.0102, -0.0706), "Cz": (-0.0093, 0.1862),
    "F1": (-0.216, 0.68), "F10": (0.8112, 0.8054), "F2": (0.2082, 0.6841),
    "F3": (-0.4, 0.6693), "F4": (0.3892, 0.6769), "F5": (-0.5449, 0.6619),
    "F6": (0.5493, 0.6751), "F7": (-0.6578, 0.669), "F8": (0.6563, 0.6845),
    "F9": (-0.8217, 0.8031), "FC1": (-0.2515, 0.4347), "FC2": (0.2332, 0.4352),
    "FC3": (-0.4539, 0.4193), "FC4": (0.4446, 0.4232), "FC5": (-0.6243, 0.4064),
    "FC6": (0.6155, 0.4119), "FCz": (-0.0088, 0.441), "FT10": (0.9439, 0.4843),
    "FT7": (-0.74, 0.4024), "FT8": (0.7178, 0.4082), "FT9": (-0.9792, 0.4917),
    "Fp1": (-0.2851, 1.0571), "Fp2": (0.2623, 1.0648), "Fpz": (-0.0121, 1.0775),
    "Fz": (-0.009, 0.6856), "Iz": (-0.0199, -0.8065), "M1": (-1.1352, -0.224),
    "M2": (1.093, -0.2346), "O1": (-0.252, -0.6431), "O2": (0.2204, -0.6408),
    "Oz": (-0.0161, -0.6453), "P1": (-0.2118, -0.3104), "P10": (0.7418, -0.4725),
    "P2": (0.2063, -0.3109), "P3": (-0.3932, -0.3158), "P4": (0.3824, -0.3161),
    "P5": (-0.5249, -0.3274), "P6": (0.4966, -0.3268), "P7": (-0.6212, -0.3514),
    "P8": (0.5895, -0.3507), "P9": (-0.774, -0.4601), "PO3": (-0.2852, -0.4974),
    "PO4": (0.2556, -0.4989), "PO7": (-0.4651, -0.5384), "PO8": (0.436, -0.5407),
    "POz": (-0.0136, -0.4864), "Pz": (-0.0117, -0.309), "T10": (0.9359, 0.131),
    "T7": (-0.7512, 0.1299), "T8": (0.7249, 0.1328), "T9": (-0.9781, 0.1434),
    "TP10": (0.9104, -0.2029), "TP7": (-0.7455, -0.1285), "TP8": (0.715, -0.129),
    "TP9": (-0.9447, -0.1902)
}

_unknown = sorted(set(XY) - M.TEN_TWENTY)
if _unknown:
    raise ValueError(f"headmap places positions that are not 10-10: {_unknown}")
del _unknown


def xy(name: str) -> tuple[float, float] | None:
    """Where to draw a position, or None if this map has no place for it."""
    return XY.get(name)


def nearest(x: float, y: float, among=None) -> str | None:
    """The position a click landed on, or None if nothing is on offer.

    `among` restricts the answer to a set of names, which is how a picker
    keeps a click from landing on a position its vocabulary does not have.
    There is no distance limit: a click anywhere picks something, because a
    picker that silently ignores a click reads as broken.
    """
    names = [n for n in (among if among is not None else XY) if n in XY]
    if not names:
        return None
    return min(names, key=lambda n: (XY[n][0] - x) ** 2 + (XY[n][1] - y) ** 2)
