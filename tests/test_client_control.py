"""The client against a stand-in link, at the level of bytes.

Every control exchange is written as the contract's bytes and answered as
the contract's bytes, so what the client sends and what it makes of an
answer are both checked without a device. This is the suite that would
have caught the client handing a decoder the whole answer, opcode and
status included, and reading the opcode as the battery's millivolts.
"""
from __future__ import annotations
import asyncio, dataclasses, pathlib, struct, sys, traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from intomind import protocol as P          # noqa: E402
from intomind.client import Device          # noqa: E402


class _BLE:
    name = "IntoMind-TEST"
    address = "00:11:22:33:44:55"


class _Link:
    """Answers each control write from a table, as the device would."""
    is_connected = True

    def __init__(self, dev, answers: dict, status: bytes):
        self.dev, self.answers, self.status = dev, answers, status
        self.writes: list[bytes] = []

    async def write_gatt_char(self, _uuid, payload, response=True):
        payload = bytes(payload)
        self.writes.append(payload)
        answer = self.answers.get(payload[0])
        if callable(answer):
            answer = answer(payload)
        if answer is None:
            answer = bytes([payload[0], 0])
        self.dev._rsp.put_nowait(bytes(answer))

    async def read_gatt_char(self, _uuid):
        return self.status

    async def start_notify(self, uuid, _callback):
        self.notified = getattr(self, "notified", []) + [uuid]


def _device(capabilities: int, answers: dict) -> tuple[Device, _Link]:
    dev = Device(_BLE())
    dev.info = P.DeviceInfo(protocol=(1, 2), firmware=(1, 2, 0), channels=4, adc_bits=24,
                            tick_hz=1_000_000, vref_uv=4_500_000, capabilities=capabilities,
                            supported_rates=7, device_id="a0a1a2a3a4a5a6a7",
                            hardware=(1, 1, 5), head_slots=4)
    # idle, electrodes, gain 24, 500 SPS, charging, 61 %, no lead-off, no flags
    status = bytes([0, 0, 6, 5, 1, 61, 0, 0]) + struct.pack("<HH", 0, 0)
    link = _Link(dev, answers, status)
    dev._client = link
    return dev, link


ALL_1_1 = sum(P.CAPABILITIES.values())
DEFAULT = P.encode_chain(P.default_chain(500))


def _answers() -> dict:
    catalog = bytes([3]) + b"".join(
        bytes([e.kind, P.STAGE_CLASSES[e.cls], e.n_params, e.max_instances]) + e.name.encode().ljust(8, b"\0")
        for e in P.CONTRACT_CATALOG)
    return {
        P.OPCODES["get_battery"]: bytes([0x42, 0]) + struct.pack("<HBB", 3712, 61, 1),
        P.OPCODES["get_boot_info"]: bytes([0x43, 0]) + struct.pack("<BBBBI", 1, 5, 1, 0, 70_000),
        P.OPCODES["get_model_info"]: bytes([0x84, 0, 2, 1, 1, 0]) + bytes.fromhex("a7c61680d9a202db") + bytes([1, 0, 0, 1, 20]),
        P.OPCODES["get_indicator"]: bytes([0x44, 0, 1]),
        P.OPCODES["get_model_interval"]: bytes([0x89, 0]) + struct.pack("<HH", 0, 4),
        P.OPCODES["get_name"]: bytes([0x47, 0]) + P.encode_name_parts("Ada", "Blue"),
        P.OPCODES["get_converter_registers"]: bytes([0x34, 0, 1, 0, 24]) + bytes.fromhex("3c95d1e00065656565" + "00" * 11 + "0f200000"),
        P.OPCODES["list_heads"]: bytes([0x82, 0, 1, 1]) + bytes([1, 1]) + struct.pack("<H", 3) + bytes(8) + b"walk".ljust(16, b"\0") + bytes(2),
        P.OPCODES["get_pipeline_catalog"]: bytes([0x90, 0]) + catalog,
        P.OPCODES["get_pipeline"]: bytes([0x91, 0, 0]) + DEFAULT,
        P.OPCODES["get_prediction_input"]: bytes([0x86, 0, 0]) + DEFAULT,
        P.OPCODES["get_bias_diagnostic"]: bytes([0x33, 0]) + struct.pack("<4h", -12, 3, -40, 25),
    }


def test_answers_are_decoded_from_their_payload_not_the_whole_response():
    dev, _ = _device(ALL_1_1, _answers())
    b = asyncio.run(dev.battery())
    assert (b.millivolts, b.percent, b.charger) == (3712, 61, "charging"), b
    bi = asyncio.run(dev.boot_info())
    assert (bi.slot, bi.reason, bi.confirmed, bi.boot_count) == (1, "update", True, 70_000), bi
    mi = asyncio.run(dev.model_info())
    assert mi.ready and mi.active_head == 1 and mi.encoder_id == "a7c61680d9a202db" and mi.takes("time_domain"), mi
    active, heads = asyncio.run(dev.heads())
    assert active == 1 and heads[0].usable and heads[0].name.rstrip("\0") == "walk", heads


def test_a_1_4_device_says_which_encoder_each_head_names_in_its_own_answer():
    answers = _answers()
    answers[P.OPCODES["list_head_encoders"]] = bytes([0x8A, 0, 1, 1]) + bytes.fromhex("a7c61680d9a202db")
    dev, link = _device(ALL_1_1, answers)
    dev.info = dataclasses.replace(dev.info, protocol=(1, 4))
    active, heads = asyncio.run(dev.heads())
    assert [w[0] for w in link.writes] == [0x82, 0x8A], link.writes
    assert active == 1 and heads[0].encoder_id == "a7c61680d9a202db", heads
    assert heads[0].trained_beside("a7c61680d9a202db") is True


def test_a_device_that_refuses_the_encoder_answer_still_lists_its_heads():
    answers = _answers()
    answers[P.OPCODES["list_head_encoders"]] = bytes([0x8A, 1])
    dev, _ = _device(ALL_1_1, answers)
    dev.info = dataclasses.replace(dev.info, protocol=(1, 4))
    active, heads = asyncio.run(dev.heads())
    assert active == 1 and heads[0].usable, heads
    assert heads[0].trained_beside("a7c61680d9a202db") is None, "a head the device did not describe is not judged"


def test_a_device_before_1_4_is_not_asked_for_encoders():
    dev, link = _device(ALL_1_1, _answers())
    asyncio.run(dev.heads())
    assert [w[0] for w in link.writes] == [0x82], link.writes


def test_the_chain_operations_send_the_contracts_bytes_and_read_its_answers():
    dev, link = _device(ALL_1_1, _answers())
    cat = asyncio.run(dev.pipeline_catalog())
    assert [e.name for e in cat] == ["highpass", "lowpass", "notch"]
    st = asyncio.run(dev.pipeline())
    assert st.is_default and [(s.kind, list(s.params)) for s in st.stages] == [(s.kind, list(s.params)) for s in P.default_chain(500)]

    chain = [P.highpass(1.0), P.notch(58, 62)]
    asyncio.run(dev.set_pipeline(chain))
    assert link.writes[-1] == bytes([0x92]) + P.encode_chain(chain)
    # A chain the device would refuse is refused here, in words, before any
    # write.
    n = len(link.writes)
    try:
        asyncio.run(dev.set_pipeline([P.lowpass(400)]))
    except P.Invalid as e:
        assert "nine tenths" in str(e)
    else:
        raise AssertionError("a corner above Nyquist was sent")
    assert len(link.writes) == n
    asyncio.run(dev.clear_pipeline())
    asyncio.run(dev.restore_pipeline_default())
    assert link.writes[-2:] == [bytes([0x93]), bytes([0x94])]

    pi = asyncio.run(dev.prediction_input())
    assert pi.source == "stream" and len(pi.stages) == len(P.default_chain(500))
    asyncio.run(dev.set_prediction_input("own_chain", [P.highpass(1.0)]))
    assert link.writes[-1] == bytes([0x85, 2]) + P.encode_chain([P.highpass(1.0)])
    asyncio.run(dev.set_prediction_input("natural"))
    assert link.writes[-1] == bytes([0x85, 1, 0])

    asyncio.run(dev.set_bias("loop_open"))
    assert link.writes[-1] == bytes([0x32, 2])
    d = asyncio.run(dev.bias_diagnostic())
    assert (d.mean_mv, d.max_mv) == (-12, 25)


def test_the_lamp_the_registers_and_embeddings_send_the_contracts_bytes():
    dev, link = _device(ALL_1_1, _answers())
    assert asyncio.run(dev.indicator()) == "reserved"
    asyncio.run(dev.set_indicator("verbose"))
    assert link.writes[-1] == bytes([0x45, 2])
    asyncio.run(dev.identify(7))
    assert link.writes[-1] == bytes([0x46, 7])
    for bad in ({"level": "loud"}, {"seconds": 31}):
        try:
            asyncio.run(dev.set_indicator(bad["level"]) if "level" in bad else dev.identify(bad["seconds"]))
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} was sent")
    regs = asyncio.run(dev.converter_registers())
    assert regs.family == "ads1299" and len(regs.values) == 24 and regs.values[0] == 0x3C
    words = regs.describe()
    assert words["data_rate_sps"] == 500 and words["test_signal"]["internal"] and words["channels"][0]["input"] == "test signal"
    assert words["bias"]["amplifier_on"] is False
    # Embeddings: the model's token count is read first, the characteristic
    # is subscribed once, and the form goes as its byte.
    asyncio.run(dev.set_embeddings("both"))
    assert link.writes[-1] == bytes([0x87, 3]) and link.notified == [P.EMBEDDINGS]
    assert dev._assembler is not None and dev._assembler.n_tokens == 80
    asyncio.run(dev.set_embeddings("window"))
    assert link.writes[-1] == bytes([0x87, 1]) and link.notified == [P.EMBEDDINGS]
    asyncio.run(dev.set_embeddings("off"))
    assert link.writes[-1] == bytes([0x87, 0]) and dev._assembler is None


def test_a_device_that_claims_nothing_new_is_refused_in_words_before_the_wire():
    caps = ALL_1_1 & ~(P.CAPABILITIES["pipeline"] | P.CAPABILITIES["bias_drive"]
                       | P.CAPABILITIES["indicator"] | P.CAPABILITIES["converter_registers"]
                       | P.CAPABILITIES["embeddings"])
    dev, link = _device(caps, _answers())
    for call in (dev.pipeline, lambda: dev.set_pipeline([P.highpass(1.0)]), lambda: dev.set_bias("on"), dev.bias_diagnostic,
                 dev.indicator, lambda: dev.set_indicator("silent"), dev.identify, dev.converter_registers,
                 lambda: dev.set_embeddings("window")):
        try:
            asyncio.run(call())
        except RuntimeError as e:
            assert "does not" in str(e) and "capability" in str(e), e
        else:
            raise AssertionError("an operation the device does not claim was sent")
    assert link.writes == []


def test_start_reads_the_chain_in_force_and_refresh_config_carries_it():
    dev, link = _device(ALL_1_1, _answers())
    asyncio.run(dev.start(samples_per_packet=10))
    assert dev.pipeline_at_start is not None and dev.pipeline_at_start.is_default
    assert [w[0] for w in link.writes] == [0x41, 0x50, 0x91, 0x01], [hex(w[0]) for w in link.writes]
    cfg = asyncio.run(dev.refresh_config())
    assert cfg["rate_sps"] == 500 and cfg["processing"]["origin"] == "default"
    assert cfg["processing"]["description"].startswith("high-pass 0.5 Hz")
    # A 1.0 device: no chain read, nothing recorded.
    dev, link = _device(ALL_1_1 & ~P.CAPABILITIES["pipeline"], _answers())
    asyncio.run(dev.start(samples_per_packet=10))
    assert dev.pipeline_at_start is None and 0x91 not in [w[0] for w in link.writes]
    assert "processing" not in asyncio.run(dev.refresh_config())


def test_the_pairing_agent_answers_for_the_host_and_refuses_a_passkey():
    """The agent accepts the pairing an IntoMind device asks for, which
    needs no passkey and no display, and refuses anything that would."""
    from intomind import pairing
    Agent = pairing._agent_class()
    a = Agent()
    assert pairing.CAPABILITY == "NoInputNoOutput"
    assert a.RequestAuthorization("/org/bluez/hci0/dev_X") is None
    assert a.AuthorizeService("/org/bluez/hci0/dev_X", "0000") is None
    assert a.RequestConfirmation("/org/bluez/hci0/dev_X", 123456) is None
    for ask in (lambda: a.RequestPinCode("/org/bluez/hci0/dev_X"), lambda: a.RequestPasskey("/org/bluez/hci0/dev_X")):
        try:
            ask()
        except Exception as e:                           # noqa: BLE001
            assert "Rejected" in str(getattr(e, "type", "")) or "no IntoMind device" in str(e), e
        else:
            raise AssertionError("a passkey request was answered")
    # The source holds the bound the hang taught: the first encrypted
    # operation never waits forever.
    src = pathlib.Path(__file__).resolve().parent.parent / "intomind" / "client.py"
    text = src.read_text()
    assert "pairing.pair(path, 20)" in text and "wait_for(self._client.start_notify(RSP" in text
    # The agent is never the system's default: it answers for this process only.
    agent_src = (src.parent / "pairing.py").read_text()
    assert "request_default_agent" not in agent_src.lower() and "call_register_agent" in agent_src


def test_the_cadence_the_name_and_the_synthetic_mode_speak_the_contract():
    import warnings
    dev, link = _device(ALL_1_1, _answers())
    dev.info = P.DeviceInfo(**{**dev.info.__dict__, "capabilities_high": sum(P.CAPABILITIES_HIGH.values())})
    mi = asyncio.run(dev.model_interval())
    assert (mi.interval_s, mi.minimum_s) == (0, 4)
    asyncio.run(dev.set_model_interval(30))
    assert link.writes[-1] == bytes([0x88, 30, 0])
    assert asyncio.run(dev.get_name()) == ("Ada", "Blue")
    asyncio.run(dev.set_name("Ada", "Blue"))
    assert link.writes[-1] == bytes([0x48]) + P.encode_name_parts("Ada", "Blue")
    try:
        asyncio.run(dev.set_name("Beatrice", "Purple"))
    except ValueError as e:
        assert "29" in str(e)
    else:
        raise AssertionError("a name that does not fit the air was sent")
    asyncio.run(dev.set_mode("synthetic"))
    assert link.writes[-1] == bytes([0x20, 3])
    asyncio.run(dev.set_mode("normal"))
    assert link.writes[-1] == bytes([0x20, 0])
    controls = dev.profile()["controls"]
    assert "synthetic" in controls["signal_source"]["options"]
    assert controls["name"]["max_bytes"] == 29 and controls["model_cadence"]["supported"]
    cfg = asyncio.run(dev.refresh_config())
    assert cfg["composed_name"] == "Ada's Blue IntoMind One" and cfg["model_interval_s"] == 0

    # Without the claims, nothing is sent.
    plain, plain_link = _device(ALL_1_1 & ~P.CAPABILITIES["model_cadence"], _answers())
    for ask in (lambda: plain.set_mode("synthetic"), lambda: plain.get_name(), lambda: plain.set_model_interval(4)):
        try:
            asyncio.run(ask())
        except RuntimeError as e:
            assert "capability" in str(e)
        else:
            raise AssertionError("a 1.3 request went to a device that never claimed it")
    assert not any(w[0] in (0x20, 0x47, 0x88) for w in plain_link.writes)


def test_a_head_trained_beside_another_encoder_is_warned_about_and_still_sent():
    import warnings
    dev, link = _device(ALL_1_1, _answers())
    dev.info = P.DeviceInfo(**{**dev.info.__dict__, "model_embed_dim": 3})
    sent = []
    async def fake_transfer(blob, target="app", slot=0, **_):
        sent.append((target, slot, blob))
    dev.transfer = fake_transfer
    other = P.build_head([[1, 2, 3]], [0], [1.0], name="walk", encoder_id="0000000000000001")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(dev.upload_head(other, slot=1))
    assert sent and sent[0][0] == "head", "the head went anyway"
    assert any("a7c61680d9a202db" in str(w.message) for w in caught), [str(w.message) for w in caught]
    same = P.build_head([[1, 2, 3]], [0], [1.0], name="walk", encoder_id="a7c61680d9a202db")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(dev.upload_head(same, slot=2))
    assert not caught, "a head trained beside this encoder warns of nothing"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            failed.append(t.__name__)
            print(f"  FAIL  {t.__name__}: {e}")
            if "-v" in sys.argv:
                traceback.print_exc()
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("failed:", ", ".join(failed))
    return 1 if failed else 0


def test_scan_lists_a_device_the_system_already_holds_connected():
    """A device paired from the system's settings is connected by them and
    does not advertise. The scan asks the system for such devices and keeps
    the ones that speak the protocol."""
    import asyncio
    from bleak.backends.device import BLEDevice
    from intomind import client, pairing, protocol as P
    ours = BLEDevice("EE:AE:00:00:00:01", "IntoMind-1-0001", {"path": "/org/bluez/hci0/dev_EE_AE_00_00_00_01",
                     "props": {"Connected": True, "UUIDs": [P.SERVICE]}})
    other = BLEDevice("11:22:33:44:55:66", "Headphones", {"path": "/org/bluez/hci0/dev_11_22_33_44_55_66",
                      "props": {"Connected": True, "UUIDs": ["0000110b-0000-1000-8000-00805f9b34fb"]}})
    class Scanner:
        def __init__(self, detection_callback=None):
            pass
        async def start(self):
            pass
        async def stop(self):
            pass
    saved = client.BleakScanner, pairing.held_devices
    client.BleakScanner = Scanner
    async def held():
        return [ours, other]
    pairing.held_devices = held
    try:
        found = asyncio.run(client.scan(0.1))
    finally:
        client.BleakScanner, pairing.held_devices = saved
    assert [d.address for d in found] == ["EE:AE:00:00:00:01"]


def test_scan_reports_each_device_the_moment_it_is_heard_and_stops_when_told():
    """Nothing that wants a device waits the timeout out: a device is
    reported when first heard, again when its name arrives with the scan
    response, and the scan ends as soon as the caller says so. A device
    the system already holds is reported before the radio listens."""
    import asyncio, time
    from types import SimpleNamespace
    from bleak.backends.device import BLEDevice
    from intomind import client, pairing, protocol as P
    held_one = BLEDevice("EE:AE:00:00:00:02", "IntoMind-1-0002", {"path": "/org/bluez/hci0/dev_EE_AE_00_00_00_02",
                         "props": {"Connected": True, "UUIDs": [P.SERVICE]}})
    ours, name = "EE:AE:00:00:00:01", "IntoMind-1-0001"
    log = []

    def adv(uuids, local_name=None):
        return SimpleNamespace(service_uuids=uuids, local_name=local_name)

    class Scanner:
        def __init__(self, detection_callback=None):
            self.report = detection_callback
        async def start(self):
            log.append("radio on")
            self.report(BLEDevice(ours, None, {}), adv([P.SERVICE]))                 # heard, no name yet
            self.report(BLEDevice(ours, None, {}), adv([P.SERVICE]))                 # heard again, nothing new
            self.report(BLEDevice("11:22:33:44:55:66", "Headphones", {}),
                        adv(["0000110b-0000-1000-8000-00805f9b34fb"], "Headphones"))  # not ours
            self.report(BLEDevice(ours, name, {}), adv([P.SERVICE], name))           # the name arrives
        async def stop(self):
            log.append("radio off")

    async def held():
        return [held_one]

    async def run():
        stop = asyncio.Event()
        def on_device(d):
            log.append((d.address, d.name))
            if d.name == name:
                stop.set()
        return await client.scan(30.0, on_device=on_device, stop=stop)

    saved = client.BleakScanner, pairing.held_devices
    client.BleakScanner, pairing.held_devices = Scanner, held
    t0 = time.monotonic()
    try:
        found = asyncio.run(run())
    finally:
        client.BleakScanner, pairing.held_devices = saved
    assert time.monotonic() - t0 < 2.0, "the scan waited its timeout out instead of stopping when told"
    assert log == [(held_one.address, held_one.name), "radio on", (ours, None), (ours, name), "radio off"], log
    assert [(d.address, d.name) for d in found] == [(held_one.address, held_one.name), (ours, name)]


def test_a_packet_delivered_twice_is_counted_once():
    """The system's Bluetooth stack has handed the same notification over
    twice in some sessions. The second copy is dropped and counted, not
    taken as samples and not flagged as a gap."""
    import types
    from test_protocol import by_kind
    from intomind import client, protocol as P
    vectors = [v for v in by_kind("eeg_data") if not P.decode_packet(bytes.fromhex(v["bytes"]), 4).discontinuity]
    assert vectors, "a plain data packet vector"
    raw = bytes.fromhex(vectors[0]["bytes"])
    dev = client.Device(types.SimpleNamespace(address="EE:AE:00:00:00:01", name="IntoMind-1-0001", details={}))
    dev.info = types.SimpleNamespace(channels=4, tick_hz=1_000_000, microvolts_per_count=lambda gain: 1.0)
    dev._on_data(None, raw)
    n, gaps = dev.samples_total, dev.gaps_announced
    assert not dev.another_listener
    import warnings
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        dev._on_data(None, raw)
        dev._on_data(None, raw)
    assert dev.samples_total == n and dev.gaps_announced == gaps and dev.duplicates == 2
    # Another program listening is said once as it starts, in words.
    assert dev.another_listener
    assert len([w for w in said if "another program" in str(w.message)]) == 1
    # A step back that is not a copy, a count restarted without the flag and
    # a clock that moved on, is never dropped: the contract calls it a break.
    back = bytearray(raw)
    back[4:8] = ((int.from_bytes(raw[4:8], "little") - 50) & 0xFFFFFFFF).to_bytes(4, "little")
    back[8:16] = (int.from_bytes(raw[8:16], "little") + 1_000_000).to_bytes(8, "little")
    before = dev.samples_total
    dev._on_data(None, bytes(back))
    assert dev.duplicates == 2 and dev.samples_total > before and dev.gaps_announced == gaps + 1
    # A restarted epoch steps the index back too, and says so; it is not a duplicate.
    restart = next((bytes.fromhex(v["bytes"]) for v in by_kind("eeg_data")
                    if P.decode_packet(bytes.fromhex(v["bytes"]), 4).discontinuity), None)
    if restart is not None:
        dev._on_data(None, restart)
        assert dev.duplicates == 2 and dev.samples_total > n


def test_a_generated_sample_says_so_and_a_measured_one_does_not():
    """1.3: every sample from a synthetic packet carries the mark, so a
    consumer of samples alone can still tell a generated signal from a
    person's."""
    import types
    from test_protocol import by_kind
    from intomind import client, protocol as P
    kinds = set()
    for v in by_kind("eeg_data"):
        raw = bytes.fromhex(v["bytes"])
        pkt = P.decode_packet(raw, 4)
        dev = client.Device(types.SimpleNamespace(address="EE:AE:00:00:00:01", name="IntoMind One", details={}))
        dev.info = types.SimpleNamespace(channels=4, tick_hz=1_000_000, microvolts_per_count=lambda gain: 1.0)
        got = []
        dev.on_sample = lambda _d, s: got.append(s)
        dev._on_data(None, raw)
        assert got and all(s.synthetic is pkt.synthetic for s in got), v["name"]
        kinds.add(pkt.synthetic)
    assert kinds == {True, False}, "vectors of both kinds"


def test_a_refusal_on_usb_power_says_to_unplug():
    """1.4: a stream refused while USB power is present is a typed refusal
    whose words say what to do, and still read `-> status 7`."""
    from intomind.client import Refused
    dev, _ = _device(ALL_1_1, {P.OPCODES["start_stream"]: bytes([P.OPCODES["start_stream"], P.STATUS_USB_POWER])})
    try:
        asyncio.run(dev.command(P.OPCODES["start_stream"]))
    except Refused as e:
        assert (e.op, e.status, e.status_name) == (P.OPCODES["start_stream"], 7, "usb power")
        assert "-> status 7" in str(e) and "Unplug it to record" in str(e), str(e)
    else:
        raise AssertionError("a start refused on USB power was taken")


def test_a_stream_the_device_ends_on_usb_power_is_noticed_and_its_samples_kept():
    """1.4: the device ends a stream of the natural signal when USB power
    appears, and its packets say so. The client stops counting the stream
    as running and says why, once, and still delivers the samples, which
    were converted before. A 1.3 device's flag ends nothing, and neither
    does the generated signal's."""
    import types
    from test_protocol import by_kind
    from intomind import client
    raws = [bytes.fromhex(v["bytes"]) for v in by_kind("eeg_data")]
    natural = next(r for r in raws if not P.decode_packet(r, 4).synthetic)
    generated = next(r for r in raws if P.decode_packet(r, 4).synthetic)

    def usb(raw, step=0):
        """The same packet with USB power present, `step` packets later."""
        b = bytearray(raw)
        b[1] |= P.DATA_FLAGS["usb_present"]
        f = list(P._DATA.unpack_from(b, 0))
        f[3] += step * f[5]
        f[4] += step * 2000 * f[5]
        P._DATA.pack_into(b, 0, *f)
        return bytes(b)

    def device(protocol):
        dev = client.Device(types.SimpleNamespace(address="EE:AE:00:00:00:01", name="IntoMind One", details={}))
        dev.info = types.SimpleNamespace(channels=4, tick_hz=1_000_000, protocol=protocol,
                                         microvolts_per_count=lambda gain: 1.0)
        dev.streaming = True
        stops, packets = [], []
        dev.on_usb_stop = stops.append
        dev.on_packet = lambda _d, p: packets.append(p)
        return dev, stops, packets

    dev, stops, packets = device((1, 4))
    dev._on_data(None, usb(natural))
    dev._on_data(None, usb(natural, 1))
    assert dev.ended_by_usb and not dev.streaming
    assert stops == [dev], "said once"
    assert len(packets) == 2, "the samples converted before are kept"

    dev, stops, _ = device((1, 3))
    dev._on_data(None, usb(natural))
    assert dev.streaming and not dev.ended_by_usb and not stops, "a 1.3 device keeps streaming"

    dev, stops, _ = device((1, 4))
    dev._on_data(None, usb(generated))
    assert dev.streaming and not dev.ended_by_usb and not stops, "the generated signal runs on USB power"


def test_collecting_enough_windows_for_a_head_waits_long_enough():
    """A head needs more windows than the embedding has numbers, so the
    documented collection is a hundred windows: nearly seven minutes of
    signal. The wait has to scale with what was asked for, or the call gives
    up before the windows can exist."""
    dev, _link = _device(ALL_1_1, {})

    async def no_form(form, on_window=None):
        pass

    dev.set_embeddings = no_form
    seen = []
    real = asyncio.wait_for

    async def wait_for(aw, timeout):
        seen.append(timeout)
        aw.close()
        raise asyncio.TimeoutError

    asyncio.wait_for = wait_for
    try:
        try:
            asyncio.run(dev.collect_embeddings(100))
        except asyncio.TimeoutError:
            pass
    finally:
        asyncio.wait_for = real
    assert seen and seen[0] >= 100 * 4, f"waited {seen} s for a hundred four-second windows"
    seen.clear()
    asyncio.wait_for = wait_for
    try:
        try:
            asyncio.run(dev.collect_embeddings(3, timeout=5.0))
        except asyncio.TimeoutError:
            pass
    finally:
        asyncio.wait_for = real
    assert seen == [5.0], "a timeout given is the timeout used"

if __name__ == "__main__":
    sys.exit(main())
