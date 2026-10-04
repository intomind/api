"""Every public item in this package carries a docstring, because this
package is what a stranger reads first, on PyPI, with no other context.

    python3 tests/test_docstrings.py

Checked: the module itself; every public class and function defined in it,
as opposed to merely imported into it; and every public method, classmethod,
staticmethod and property a public class defines in its own body, as
opposed to one it only inherits. "Public" is the one rule, applied
uniformly: a name that does not start with an underscore. That rule alone
is why a dunder is exempt, and why the methods a dataclass or an enum
generates are too: every one of them starts with "__".

One more trap, specific to a dataclass: when a dataclass has no docstring
of its own, Python writes it one, a one-liner that is just its name and
its field signature. That text is not None and not blank, so a check that
only asks "is `__doc__` empty" is fooled into calling an undocumented
dataclass documented. This gate reconstructs that placeholder and treats a
class whose `__doc__` IS the placeholder exactly as it treats one whose
`__doc__` is empty.

A module that needs an optional dependency this package declares in
pyproject.toml (scipy, h5py, pyserial) and does not have it is skipped
rather than failing this gate. Requiring every extra just to check a
docstring would defeat the point of declaring them optional.
"""
from __future__ import annotations
import dataclasses, importlib, inspect, pathlib, sys, traceback

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PKG_DIR = ROOT / "intomind"

#: Import names of this package's optional dependencies (pyproject.toml's
#: `[project.optional-dependencies]`). A module that fails to import because
#: one of these is missing is skipped, not failed.
OPTIONAL = {"scipy", "h5py", "serial"}


def _missing_optional(exc: BaseException) -> str | None:
    """The optional dependency behind an ImportError, by walking its
    `__cause__`/`__context__` chain, or None if the gate must not hide it."""
    seen, guard = exc, 0
    while isinstance(seen, BaseException) and guard < 10:
        name = getattr(seen, "name", None)
        if name and any(name == dep or name.startswith(dep + ".")
                        for dep in OPTIONAL):
            return name
        nxt = seen.__cause__ or seen.__context__
        if nxt is None:
            break
        seen, guard = nxt, guard + 1
    return None


def module_names() -> list[str]:
    """Every module in the package, read from the files on disk rather than
    a list typed here, so a new file is checked the day it is added."""
    names = []
    for f in sorted(PKG_DIR.glob("*.py")):
        if f.name == "__init__.py":
            names.append("intomind")
        elif not f.stem.startswith("_"):
            names.append(f"intomind.{f.stem}")
    return names


def load_modules() -> tuple[list, list[str]]:
    """Every module that imports cleanly, and a note for each module skipped
    because an optional dependency it needs is not installed here."""
    mods, skipped = [], []
    for name in module_names():
        try:
            mods.append(importlib.import_module(name))
        except ImportError as e:
            missing = _missing_optional(e)
            if missing is None:
                raise
            skipped.append(f"{name} (needs {missing!r}, not installed)")
    return mods, skipped


def _empty(doc) -> bool:
    return doc is None or not doc.strip()


def _dataclass_placeholder(cls) -> str | None:
    """The docstring `dataclasses` writes in by itself for a class that has
    none of its own: its name and its field signature. None for a class
    `dataclasses` never touched."""
    if not dataclasses.is_dataclass(cls):
        return None
    try:
        sig = str(inspect.signature(cls)).replace(" -> None", "")
    except (TypeError, ValueError):
        sig = ""
    return cls.__name__ + sig


def _empty_class_doc(cls) -> bool:
    """Whether a class's own `__doc__` fails to say anything: either empty,
    or the placeholder `dataclasses` wrote in because it was."""
    doc = cls.__doc__
    return _empty(doc) or doc == _dataclass_placeholder(cls)


def class_items(cls) -> list[tuple[str, object]]:
    """(name, doc) for every public method, classmethod, staticmethod and
    property this class defines in its own body. For a property, the doc is
    its getter's. A name the class only inherits is not its own, and is not
    returned."""
    out = []
    for name, obj in cls.__dict__.items():
        if name.startswith("_"):
            continue                       # dunders, and anything generated
        if isinstance(obj, (staticmethod, classmethod)):
            out.append((name, obj.__func__.__doc__))
        elif isinstance(obj, property):
            out.append((name, obj.fget.__doc__ if obj.fget else None))
        elif inspect.isfunction(obj):
            out.append((name, obj.__doc__))
    return out


def check_module(mod) -> list[str]:
    """Every item on `mod` the gate holds to a docstring that lacks one,
    named `module`, `module.Class`, `module.function` or
    `module.Class.member`."""
    bad = []
    if _empty(mod.__doc__):
        bad.append(mod.__name__)
    for name, obj in vars(mod).items():
        if name.startswith("_") or getattr(obj, "__module__", None) != mod.__name__:
            continue                       # private, or merely imported here
        if inspect.isclass(obj):
            if _empty_class_doc(obj):
                bad.append(f"{mod.__name__}.{name}")
            for mname, mdoc in class_items(obj):
                if _empty(mdoc):
                    bad.append(f"{mod.__name__}.{name}.{mname}")
        elif inspect.isfunction(obj):
            if _empty(obj.__doc__):
                bad.append(f"{mod.__name__}.{name}")
    return bad


def test_every_public_item_has_a_docstring():
    mods, skipped = load_modules()
    assert mods, "not one module of the package imported"
    bad = [item for mod in mods for item in check_module(mod)]
    if skipped:
        print("  (skipped, an optional dependency is not installed here: "
              + ", ".join(skipped) + ")")
    assert not bad, f"{len(bad)} without a docstring:\n  " + "\n  ".join(bad)


def test_the_gate_can_actually_fail():
    """A checker that has only ever seen clean modules might be checking
    nothing. Build one module-shaped object that fails in every way the
    real gate watches for, by the real function, and confirm each is named
    and nothing innocent is."""
    import types

    mod = types.ModuleType("intomind._docstring_gate_self_test")
    mod.__doc__ = None   # 1: the module itself

    class Undocumented:
        pass            # 2: the class itself

    def method(self):
        pass             # 3: a plain method

    def cm(cls):
        pass              # 4: a classmethod

    def sm():
        pass               # 5: a staticmethod

    def prop(self):
        return 1          # 6: a property's getter

    def _private(self):
        pass        # must stay unnamed: private

    def __repr__(self):
        return "x"  # must stay unnamed: dunder

    Undocumented.method = method
    Undocumented.cm = classmethod(cm)
    Undocumented.sm = staticmethod(sm)
    Undocumented.prop = property(prop)
    Undocumented._private = _private
    Undocumented.__repr__ = __repr__
    Undocumented.__module__ = mod.__name__
    mod.Undocumented = Undocumented

    def undocumented_function():
        pass
    undocumented_function.__module__ = mod.__name__
    mod.undocumented_function = undocumented_function

    mod.imported_name = int   # defined elsewhere: must never be named

    # 7: a dataclass with no docstring of its own carries a non-empty
    # `__doc__` anyway, written in by `dataclasses` -- a signature, not a
    # description. The gate must see through that, and must not punish the
    # dataclass next to it that wrote a real one.
    import dataclasses as _dc

    @_dc.dataclass
    class Placeholder:
        x: int

    Placeholder.__module__ = mod.__name__
    mod.Placeholder = Placeholder

    @_dc.dataclass
    class Documented:
        """A dataclass that said something about itself."""
        x: int

    Documented.__module__ = mod.__name__
    mod.Documented = Documented

    bad = check_module(mod)

    for want in (mod.__name__, f"{mod.__name__}.Undocumented",
                f"{mod.__name__}.Undocumented.method",
                f"{mod.__name__}.Undocumented.cm",
                f"{mod.__name__}.Undocumented.sm",
                f"{mod.__name__}.Undocumented.prop",
                f"{mod.__name__}.undocumented_function",
                f"{mod.__name__}.Placeholder"):
        assert want in bad, f"the gate missed {want}"

    for innocent in (f"{mod.__name__}.Undocumented._private",
                    f"{mod.__name__}.Undocumented.__repr__",
                    f"{mod.__name__}.imported_name",
                    f"{mod.__name__}.Documented"):
        assert innocent not in bad, f"the gate wrongly named {innocent}"


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
    if failed:
        print("failed:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
