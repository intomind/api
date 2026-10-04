"""The IntoMind instrument API.

One library, every IntoMind device. It owns the link to a device, the
protocol spoken over it, the recording of what it streams, the provenance
of that recording, the exports it can be read out as, and the analyses
that read it back. It knows nothing about any particular device beyond
what that device tells it, and nothing about any application built on top
of it.

    from intomind.client import display_names, scan
    from intomind.session import Session

    found = await scan()
    labels = display_names(found)
    chosen = next(d for d in found if labels[d.address] == "Ada's IntoMind One")
    s = Session()
    await s.connect(chosen)
    device = s.devices[0]

Free software under the GNU Affero General Public License, version 3,
with a commercial license available. See LICENSING.md.
"""
from .client import Device, Refused, Sample, scan
from .protocol import DeviceInfo, Packet, Prediction, Status
from .session import Session

from . import adapters, montage, placement, protocol, provenance, timebase

__version__ = "1.4.0"

# `analysis`, `export`, and `model` are reached on first use rather than at
# import, because the analyses need a signal processing library that this
# package declares optional. Importing the package must not require
# something the package says you may leave out.
_LAZY = ("analysis", "export", "model")


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))

__all__ = [
    # The device, and what it produces.
    "Device", "DeviceInfo", "Packet", "Prediction", "Refused", "Sample",
    "Session", "Status", "scan",
    # The layers, for anyone who wants one of them on its own.
    "adapters", "analysis", "export", "model", "montage", "placement",
    "protocol", "provenance", "timebase",
    "__version__",
]
