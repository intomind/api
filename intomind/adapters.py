"""Support for hardware this library was not written for.

The rule this library lives by is that the device is the authority on what
the device is, and that a table of specific hardware here is a defect. An
adapter is how a device that needs more than the contract offers gets it,
without that knowledge entering the library.

An adapter is a small object in a separate package. It says which devices
it claims, and whatever it offers appears as `device.extra` on those
devices and on no others. Nothing in this library calls into an adapter,
so a device with no adapter is a device with `extra` set to None and
every published behavior unchanged.

    from intomind import adapters

    class MyAdapter:
        name = "acme.eeg"
        version = "1"

        def claims(self, info):
            return info.protocol < (1, 0)

        def attach(self, device):
            return MyExtras(device)

    adapters.register(MyAdapter())

What an adapter must not do is change what the contract means. It adds
operations on numbers the contract reserves for exactly this, and a device
that does not implement them answers unsupported, which is the contract
working rather than failing.

Every adapter's identity is written into each recording made through it,
so a capture made by code IntoMind did not ship is identifiable as such
forever, not only while somebody remembers.
"""
from __future__ import annotations

_REGISTERED: list = []


def register(adapter) -> None:
    """Add an adapter. Later registrations are asked first, so an
    application can override one that a library installed."""
    for required in ("name", "version", "claims", "attach"):
        if not hasattr(adapter, required):
            raise TypeError(f"an adapter needs {required}")
    _REGISTERED.insert(0, adapter)


def unregister(adapter) -> None:
    if adapter in _REGISTERED:
        _REGISTERED.remove(adapter)


def registered() -> list:
    return list(_REGISTERED)


def clear() -> None:
    """Forget every adapter. For tests, which must not inherit what another
    test registered."""
    _REGISTERED.clear()


def for_device(info):
    """The first adapter that claims this device, or None.

    An adapter that raises while deciding is passed over rather than
    allowed to stop a device connecting, because a device the library can
    already talk to must not be made unusable by a third party's mistake.
    """
    for a in _REGISTERED:
        try:
            if a.claims(info):
                return a
        except Exception:
            continue
    return None


def identity(adapter) -> dict:
    """What goes into a recording about the code that made it."""
    if adapter is None:
        return dict(name="intomind.ble", version="1", first_party=True,
                    transport="bluetooth-le")
    return dict(name=getattr(adapter, "name", "unknown"),
                version=str(getattr(adapter, "version", "unknown")),
                first_party=bool(getattr(adapter, "first_party", False)),
                transport=getattr(adapter, "transport", "unknown"))
