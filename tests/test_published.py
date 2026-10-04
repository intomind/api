"""Nothing in this repository names what it must not name.

This package is published. A part number, an unreleased project, a
person, or somebody's directory layout, left in a comment, goes out with
it and cannot be taken back.

The file list comes from git rather than from a list typed here, because
a list typed here goes stale the first time somebody adds a file, and the
gate then passes over exactly the new work it was meant to read.
"""
from __future__ import annotations
import os, pathlib, re, subprocess, sys, traceback

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Patterns that give nothing away by being written down: an address, a
#: path on somebody's machine. Every pattern was once somewhere it should
#: not have been. Matched without regard to case.
PUBLIC: list[tuple[str, str, bool, str]] = [
    (r"[\w.+-]+@gmail\.com", "a personal address", True, "write to someone@gmail.com"),
    (r"/home/[a-z]", "a path on somebody's machine", True, "/home/someone/project"),
]

#: The rest are themselves what they protect: unreleased devices and
#: projects, part numbers, register names, people, private repositories.
#: Written here, they would be published by the very check that keeps them
#: out. They live on the maintainers' machines, in the file
#: INTOMIND_PRIVATE_WORDS names or ~/.config/intomind/private-words.tsv:
#: one pattern per line, then what it is, "fold" or "exact", and an example
#: the check must catch, separated by tabs. Without the file this check
#: fails and says why, because a check that quietly does not run is not a
#: check.
def private_words() -> list[tuple[str, str, bool, str]]:
    path = pathlib.Path(os.environ.get("INTOMIND_PRIVATE_WORDS",
                                       "~/.config/intomind/private-words.tsv")).expanduser()
    if not path.is_file():
        raise AssertionError(
            f"the private word list is not on this machine ({path}), so what this repository "
            "may publish cannot be checked. Set INTOMIND_PRIVATE_WORDS to it.")
    out = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        pattern, why, case, example = line.split("\t")
        out.append((pattern, why, case == "fold", example))
    assert out, f"{path} holds no patterns"
    return out


def banned() -> list[tuple[str, str, bool, str]]:
    return PUBLIC + private_words()


#: The wire field is spelled this way in the contract every
#: implementation is held to, so it is this library's word and not a
#: borrowed one. Renaming it would change the contract and hide nothing:
#: the gain ladder and the converter width say as much to a reader who
#: is looking. Listed here so the choice is deliberate rather than an
#: oversight.
ALLOWED = [
    re.compile(r"loff_statp"),
]


def tracked() -> list[pathlib.Path]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.split()
    return [ROOT / f for f in out]


def scan(text: str, patterns=None) -> list[str]:
    hits = []
    for pattern, why, fold, _example in (patterns if patterns is not None else banned()):
        for m in re.finditer(pattern, text, re.I if fold else 0):
            if any(a.fullmatch(m.group(0)) or a.match(m.group(0)) for a in ALLOWED):
                continue
            hits.append(f"names {m.group(0)!r}, which is {why}")
    return hits


def test_no_tracked_file_names_what_it_must_not():
    files = tracked()
    assert len(files) > 20, "git listed almost nothing, so this gate read almost nothing"
    problems = []
    read = 0
    patterns, private = banned(), private_words()
    for f in files:
        # This file writes the public patterns' made-up examples, so it is
        # held to the private words alone, which it never writes.
        own = f.resolve() == pathlib.Path(__file__).resolve()
        try:
            text = f.read_text()
        except (UnicodeDecodeError, FileNotFoundError):
            continue          # a binary fixture has no prose to read
        read += 1
        for hit in set(scan(text, private if own else patterns)):
            problems.append(f"{f.relative_to(ROOT)}: {hit}")
    assert read > 20, "almost nothing was read, so this gate proves almost nothing"
    assert not problems, "\n  " + "\n  ".join(sorted(problems))


def test_the_gate_can_see_a_name():
    """A gate never seen to fail is not a gate: every pattern catches its
    own example."""
    for pattern, why, fold, example in banned():
        hits = scan(example)
        assert any(why in h for h in hits), f"{example!r} went unseen by {pattern!r}"


def test_the_contracts_own_field_is_not_a_finding():
    """`loff_statp` is the contract's word, held by four implementations."""
    assert scan("the packet carries loff_statp, one bit per channel") == []


def test_this_file_names_nothing_it_protects():
    """No file is exempt, this one included: the words it protects are read
    from the private list, never written here."""
    text = pathlib.Path(__file__).read_text()
    assert scan(text, private_words()) == [], "this check publishes what it protects"


def test_the_file_list_is_not_typed_by_hand():
    """A hand-typed list goes stale on the first new file, and the gate
    then passes over exactly the work it was meant to read."""
    files = {f.name for f in tracked()}
    assert "test_published.py" in files, "this gate does not read itself"
    assert "pyproject.toml" in files


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
