"""Where the electrodes are — as data, not as a sentence.

Every capture has always recorded the montage. It recorded it like this:

    electrode_montage = "ch1 = L-temple ant./post.; ch2 = R-temple ant./post.;
                         bias = forehead. Dry gold comb electrodes."

which is true, and which no analysis can use. Nothing can ask whether ch1 is
bipolar or referential, which sites it spans, or whether this recording has an
occipital channel, without parsing English. So the montage was *present* and
*unusable*, and every analysis fell back to selecting channels by index.

Index is the one thing that does not survive a change of instrument. `ch1` on
a two-channel bipolar layout and `ch1` on an eight-channel referential one are not
the same measurement and must never be pooled as though they were. A corpus
whose whole value is being re-analysable across that change cannot be indexed
by position.

**The montage is information the analysis needs, never a constraint the
paradigm obeys.** A game, a cued protocol or a recorder must not know or care
what is on the head. This module exists so that what was on the head is written
down precisely enough to be used afterwards.

Design notes, each learned from something in this repo:

  * **One shape covers both derivations.** A channel is a pair of sites. For a
    bipolar channel both are on the scalp; for a referential one the negative
    is the shared reference. That expresses a two-channel bipolar layout and the
    4- and 8-channel referential builds without special cases.
  * **Standard labels where they apply, free text where they do not, and both
    kept.** The temple-pole positions do not map cleanly onto 10-20. Forcing
    them into it would destroy exactly the detail that distinguishes anterior
    from posterior, and refusing free text would make the montage unwritable.
  * **Unknown is written as unknown.** A capture from a bench that never stated
    its montage says so. It does not get a plausible one.
"""
from __future__ import annotations

BIPOLAR = "bipolar"
REFERENTIAL = "referential"
DERIVATIONS = (BIPOLAR, REFERENTIAL)

# The 10-20 / 10-10 names this module recognizes as standard. A site may carry
# any `name`; `ten_twenty` is only set when the position genuinely IS one of
# these, so "this recording has a real O1" is answerable and a free-text
# approximation can never masquerade as a standard position.
TEN_TWENTY = {
    "Nz", "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
    "F9", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "F10",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10",
    "T9", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "T10",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P9", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "P10",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "Iz",
    "A1", "A2", "M1", "M2",
}


class MontageError(ValueError):
    """The montage as written cannot be true. Raised rather than guessed."""


def site(name: str, *, ten_twenty: str | None = None,
         description: str | None = None) -> dict:
    """One electrode position.

    `ten_twenty` is validated against the standard set, so a typo becomes an
    error instead of an unrecognized-but-plausible position.
    """
    if not name:
        raise MontageError("a site needs a name")
    if ten_twenty is not None and ten_twenty not in TEN_TWENTY:
        raise MontageError(f"{ten_twenty!r} is not a 10-20/10-10 position")
    return dict(name=name, ten_twenty=ten_twenty, description=description)


def channel(index: int, derivation: str, positive: dict, negative: dict,
            *, label: str | None = None) -> dict:
    """One recorded channel: a difference between two sites.

    `index` is 1-based and matches the channel numbering in the capture
    (`ch1`, `ch2`, ... and the rows of `counts`). It is recorded so the
    structure can be tied back to the data; it is NOT how a channel should be
    selected.
    """
    if derivation not in DERIVATIONS:
        raise MontageError(f"derivation must be one of {DERIVATIONS}, "
                           f"got {derivation!r}")
    if int(index) < 1:
        raise MontageError("channel index is 1-based")
    return dict(index=int(index), derivation=derivation,
                positive=positive, negative=negative,
                label=label or f"{positive['name']}-{negative['name']}")


def build(channels: list[dict], *, electrode: str,
          bias: dict | None = None, reference: dict | None = None,
          prose: str | None = None, source: str | None = None) -> dict:
    """Assemble and validate a montage.

    `bias` is the driven electrode. It is not a recording site, but it is
    part of what was on the head and therefore part of the record. `reference`
    is the shared reference of a referential montage, and must be the negative
    site of every referential channel; a montage that claims otherwise is
    rejected rather than stored.
    """
    m = dict(electrode=electrode, channels=list(channels), bias=bias,
             reference=reference, prose=prose, source=source)
    validate(m)
    return m


def validate(m: dict) -> None:
    """Refuse a montage that cannot describe a real recording."""
    chans = m.get("channels") or []
    if not chans:
        raise MontageError("a montage with no channels describes nothing")
    idx = [c["index"] for c in chans]
    if sorted(idx) != list(range(1, len(idx) + 1)):
        raise MontageError(f"channel indices must be 1..n with no gaps, got {idx}")
    ref = m.get("reference")
    for c in chans:
        if c["derivation"] not in DERIVATIONS:
            raise MontageError(f"ch{c['index']}: bad derivation {c['derivation']!r}")
        for pole in ("positive", "negative"):
            s = c.get(pole)
            if not s or not s.get("name"):
                raise MontageError(f"ch{c['index']}: {pole} site is missing")
            tt = s.get("ten_twenty")
            if tt is not None and tt not in TEN_TWENTY:
                raise MontageError(f"ch{c['index']}: {tt!r} is not a "
                                   f"10-20/10-10 position")
        if c["derivation"] == REFERENTIAL:
            if not ref:
                raise MontageError(
                    f"ch{c['index']} is referential but the montage names no "
                    f"reference")
            if c["negative"]["name"] != ref["name"]:
                raise MontageError(
                    f"ch{c['index']} is referential against "
                    f"{c['negative']['name']!r}, but the montage's reference "
                    f"is {ref['name']!r}")


def channel_count(m: dict | None) -> int:
    """How many channels this montage has, 0 if it names none."""
    return len(((m or {}).get("channels")) or [])


def derivations(m: dict | None) -> set[str]:
    """Which derivations this montage uses. A recording may mix them."""
    return {c["derivation"] for c in ((m or {}).get("channels") or [])}


def find(m: dict | None, *, ten_twenty: str | None = None,
         site_name: str | None = None, label: str | None = None,
         derivation: str | None = None) -> list[dict]:
    """Channels matching what they ARE, rather than where they sit.

    This is the selector an analysis should use. Matching on any pole, because
    "does this recording see O1" is a question about the montage, not about
    which side of the subtraction O1 landed on.
    """
    out = []
    for c in ((m or {}).get("channels") or []):
        poles = (c.get("positive") or {}), (c.get("negative") or {})
        if derivation and c["derivation"] != derivation:
            continue
        if ten_twenty and not any(p.get("ten_twenty") == ten_twenty
                                  for p in poles):
            continue
        if site_name and not any(p.get("name") == site_name for p in poles):
            continue
        if label and c.get("label") != label:
            continue
        out.append(c)
    return out


def indices(channels: list[dict]) -> list[int]:
    """0-based row numbers into a capture's `counts`, from matched channels."""
    return [c["index"] - 1 for c in channels]


def describe(m: dict | None) -> str:
    """Regenerate the prose from the structure.

    Kept so the two forms can be compared: the free text is what a human wrote
    and the structure is what a machine will act on, and a silent disagreement
    between them is exactly the failure this module exists to prevent.
    """
    if not m or not m.get("channels"):
        return "unknown"
    parts = []
    for c in sorted(m["channels"], key=lambda x: x["index"]):
        p, n = c["positive"]["name"], c["negative"]["name"]
        kind = "bipolar" if c["derivation"] == BIPOLAR else "ref"
        parts.append(f"ch{c['index']} = {p}/{n} ({kind})")
    if m.get("bias"):
        parts.append(f"bias = {m['bias']['name']}")
    tail = f" {m['electrode']} electrodes." if m.get("electrode") else ""
    return "; ".join(parts) + "." + tail


def from_site(site_cfg: dict | None) -> dict | None:
    """The structured montage a bench declares, or None if it declares none.

    `config/site.json` is the only place a human states what is physically on
    the bench, so this is where the structure is read from. A bench that has
    not been described structurally yet returns None, and the capture records
    `montage: null` — unknown, written as unknown.
    """
    m = (site_cfg or {}).get("montage")
    if not m:
        return None
    validate(m)
    return m
