"""Pairing on the host, so a device can be paired without a person at the
computer's Bluetooth settings.

On Linux, the system's Bluetooth service completes a pairing only if some
program has registered an agent to answer for the host. The desktop's
Bluetooth settings register one while they are open; a plain program does
not, and then pairing fails quietly and the first encrypted operation waits
forever.

This module registers an agent for this process and asks the system to pair
over the same connection the agent lives on. The system uses a program's own
agent for the actions that program starts, so this agent answers for this
pairing and for nothing else. It is never the system's default agent:
pairing requests that other devices send to the computer, and pairings the
user starts from the desktop's settings, are answered exactly as before.
A device paired from the desktop's settings first is simply found paired
and nothing is asked again.

The agent accepts the pairing an IntoMind device asks for, which needs no
passkey and no display, and refuses anything that would need one, because
no IntoMind device asks for that.

The module also lists the devices the system is already connected to.
A device paired a moment ago from the desktop's settings is connected by
the settings, and a connected device does not advertise, so no scan can
see it. The system knows it, and a connection made through the system
shares the link the settings hold.

On other platforms the system pairs by itself and this module does nothing.
"""
from __future__ import annotations
import asyncio
import sys

AGENT_PATH = "/com/intomind/agent"
CAPABILITY = "NoInputNoOutput"

_state = None       # (bus, agent), kept alive for the life of the process


def _agent_class():
    from dbus_fast import DBusError
    from dbus_fast.service import ServiceInterface, method

    class Agent(ServiceInterface):
        """org.bluez.Agent1, answering for the host."""

        def __init__(self):
            super().__init__("org.bluez.Agent1")

        @method()
        def Release(self):
            pass

        @method()
        def RequestAuthorization(self, device: "o"):
            pass

        @method()
        def AuthorizeService(self, device: "o", uuid: "s"):
            pass

        @method()
        def RequestConfirmation(self, device: "o", passkey: "u"):
            pass

        @method()
        def DisplayPasskey(self, device: "o", passkey: "u", entered: "q"):
            pass

        @method()
        def DisplayPinCode(self, device: "o", pincode: "s"):
            pass

        @method()
        def RequestPinCode(self, device: "o") -> "s":
            raise DBusError("org.bluez.Error.Rejected", "no IntoMind device asks for a PIN")

        @method()
        def RequestPasskey(self, device: "o") -> "u":
            raise DBusError("org.bluez.Error.Rejected", "no IntoMind device asks for a passkey")

        @method()
        def Cancel(self):
            pass

    return Agent


async def _bus():
    """The process's own connection to the system's Bluetooth service, with
    the agent registered on it. None where this module cannot act."""
    global _state
    if sys.platform != "linux":
        return None
    if _state is not None:
        return _state[0]
    try:
        from dbus_fast.aio import MessageBus
        from dbus_fast.constants import BusType
    except ImportError:
        return None
    try:
        bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
        agent = _agent_class()()
        bus.export(AGENT_PATH, agent)
        introspection = await bus.introspect("org.bluez", "/org/bluez")
        proxy = bus.get_proxy_object("org.bluez", "/org/bluez", introspection)
        manager = proxy.get_interface("org.bluez.AgentManager1")
        await manager.call_register_agent(AGENT_PATH, CAPABILITY)
        _state = (bus, agent)
        return bus
    except Exception:                                    # noqa: BLE001
        return None


async def pair(device_path: str, timeout: float = 20.0):
    """Pair the device at this system path, answering for the host.

    Returns a short note of what happened, or None where this module cannot
    act and the caller should let the system pair by itself. A pairing that
    fails is reported in the note, not raised: the caller's first encrypted
    operation is the test that matters, and it is bounded.
    """
    bus = await _bus()
    if bus is None:
        return None
    try:
        introspection = await bus.introspect("org.bluez", device_path)
        proxy = bus.get_proxy_object("org.bluez", device_path, introspection)
        device = proxy.get_interface("org.bluez.Device1")
        if await device.get_paired():
            return "already paired"
        await asyncio.wait_for(device.call_pair(), timeout)
        return "pairing completed"
    except asyncio.TimeoutError:
        return f"pairing did not complete within {timeout:g} s"
    except Exception as e:                               # noqa: BLE001
        name = getattr(e, "type", "") or type(e).__name__
        if "AlreadyExists" in name:
            return "already paired"
        return f"pairing reported {name}: {e}"


def _unwrap(props):
    return {k: (v.value if hasattr(v, "value") else v) for k, v in props.items()}


async def held_devices() -> list:
    """The devices the system is connected to right now, as device objects a
    client can connect through. Empty where this module cannot act."""
    bus = await _bus()
    if bus is None:
        return []
    try:
        from bleak.backends.device import BLEDevice
        introspection = await bus.introspect("org.bluez", "/")
        proxy = bus.get_proxy_object("org.bluez", "/", introspection)
        manager = proxy.get_interface("org.freedesktop.DBus.ObjectManager")
        objects = await manager.call_get_managed_objects()
    except Exception:                                    # noqa: BLE001
        return []
    held = []
    for path, interfaces in objects.items():
        dev = interfaces.get("org.bluez.Device1")
        if not dev:
            continue
        props = _unwrap(dev)
        if not props.get("Connected") or "Address" not in props:
            continue
        held.append(BLEDevice(props["Address"], props.get("Name"), {"path": path, "props": props}))
    return held
