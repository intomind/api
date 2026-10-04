"""The protocol module against the contract's own vectors.

`contract/conformance.json` is emitted by the firmware from the codec the
device runs. Every message in it decodes here to the fields it states, and
every malformed one is refused for the reason it states. A drift between
this implementation and the device shows up here rather than on a bench.
"""
from __future__ import annotations
import json, math, pathlib, sys, traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from intomind import protocol as P          # noqa: E402

VECTORS = json.loads((pathlib.Path(__file__).resolve().parent.parent
                      / "contract" / "conformance.json").read_text())


def u64(field) -> int:
    """A 64-bit field, which the contract writes as a decimal string. A JSON
    number cannot hold one exactly, and a reader that parses it as a float is
    quietly wrong while still agreeing with an equally wrong expectation."""
    assert isinstance(field, str), "a 64-bit field is written as a decimal string"
    return int(field)


def by_kind(kind: str, source: str = "decode"):
    return [v for v in VECTORS[source] if v["kind"] == kind]


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_the_vectors_are_for_this_version():
    assert VECTORS["protocol"] == "{}.{}".format(*P.VERSION)


def test_every_identifier_matches():
    named = {
        "service": P.SERVICE, "device_info": P.DEVICE_INFO, "control": P.CONTROL,
        "control_response": P.CONTROL_RESPONSE, "eeg_data": P.EEG_DATA,
        "status": P.STATUS, "update_control": P.UPDATE_CONTROL,
        "update_data": P.UPDATE_DATA, "predictions": P.PREDICTIONS,
        "embeddings": P.EMBEDDINGS,
    }
    for u in VECTORS["uuids"]:
        assert named[u["name"]] == u["uuid"], u["name"]
    assert len(named) == len(VECTORS["uuids"])


def test_device_info_decodes_both_layouts():
    for v in by_kind("device_info"):
        f, info = v["fields"], P.decode_device_info(bytes.fromhex(v["bytes"]))
        assert info.protocol == (f["proto_major"], f["proto_minor"]), v["name"]
        assert info.channels == f["channel_count"]
        assert info.device_id == f["device_id"]
        if f.get("extension_present") is False:
            assert info.hardware is None, "an older device is not given fields it never reported"
            assert info.head_slots == 0
            continue
        assert info.firmware_string == f["fw_version"]
        assert info.hardware == tuple(int(x) for x in f["hw_version"].split("."))
        assert info.adc_bits == f["adc_bits"]
        assert info.tick_hz == f["time_tick_hz"]
        assert info.vref_uv == f["vref_uv"]
        assert info.rates == [250, 500, 1000]
        assert info.model_embed_dim == f["model_embed_dim"]
        assert info.head_slots == f["head_slots"]
        assert info.update_chunk_max == f["update_chunk_max"]
        for name in ("battery_voltage", "leadoff", "test_signal", "update", "model", "model_ready", "heads"):
            assert info.can(name), name
        assert not info.can("battery_low_flag")
        assert not info.can("a capability invented after this host was written")


def test_a_packet_decodes_to_its_samples():
    v = by_kind("eeg_data")[0]
    f = v["fields"]
    p = P.decode_packet(bytes.fromhex(v["bytes"]), 4)
    assert p.index == f["sample_index"]
    assert p.device_time == u64(f["device_time"])
    assert p.device_time == 81_985_529_216_486_895, "the full width, not a rounded one"
    assert p.n_samples == f["n_samples"]
    assert p.gain == f["gain"]
    assert p.sample_rate_hz == f["sps"]
    assert p.mode == "test" and p.leadoff_active and p.usb_present
    assert not p.discontinuity
    assert p.leadoff == f["loff_statp"]
    assert p.counts == f["samples"], "sign extension across the whole range"


def test_control_requests_round_trip():
    for v in by_kind("control_request"):
        f, data = v["fields"], bytes.fromhex(v["bytes"])
        name, arg = P.decode_request(data)
        assert P.OPCODES[name] == f["opcode"], v["name"]
        if name in P.TAKES_PAYLOAD:
            # The payload is the bytes after the opcode, judged by the
            # processing decoders; the vector states what they hold.
            assert arg == data[1:], v["name"]
            if name == "set_model_interval":
                assert P.decode_model_interval(arg + b"\0\0").interval_s == f["interval_s"]
                assert P.encode_model_interval(f["interval_s"]) == arg
            elif name == "set_name":
                assert P.decode_name_parts(arg) == (f["name"], f["adjective"])
                assert P.encode_name_parts(f["name"], f["adjective"]) == arg
            elif "source" in f:
                pi = P.decode_prediction_input(arg)
                assert P.INPUT_SOURCES[pi.source] == f["source"]
                assert [s.as_dict() for s in pi.stages] == [dict(kind=x["kind"], name=P.KIND_NAMES[x["kind"]], params=x["params"]) for x in f["stages"]]
            else:
                assert [(s.kind, list(s.params)) for s in P.decode_chain(arg)] == [(x["kind"], x["params"]) for x in f["stages"]]
        else:
            assert arg == f["arg"]
        assert P.encode_request(name, arg) == data, "encoding gives back the same bytes"


def test_control_responses_carry_their_payloads():
    for v in by_kind("control_response"):
        f, r = v["fields"], P.decode_response(bytes.fromhex(v["bytes"]))
        assert r.opcode == f["opcode"] and r.status == f["status"], v["name"]
        if "status_name" in f:
            assert r.status_name == f["status_name"]
        if "device_time" in f:
            assert int.from_bytes(r.payload, "little") == u64(f["device_time"])


def test_the_device_reports_itself():
    v = by_kind("battery_info")[0]
    b = P.decode_battery(bytes.fromhex(v["bytes"]))
    assert (b.millivolts, b.percent) == (v["fields"]["battery_mv"], v["fields"]["battery_percent"])
    assert b.charger == "charging"

    v = by_kind("boot_info")[0]
    bi = P.decode_boot_info(bytes.fromhex(v["bytes"]))
    assert bi.slot == v["fields"]["active_slot"]
    assert bi.reason == v["fields"]["boot_reason_name"]
    assert bi.confirmed and bi.boot_count == v["fields"]["boot_count"]

    v = by_kind("model_info")[0]
    mi = P.decode_model_info(bytes.fromhex(v["bytes"]))
    assert mi.ready and mi.predictions_on
    assert mi.active_head == v["fields"]["active_head"]
    assert mi.encoder_id == v["fields"]["encoder_id"]

    v = by_kind("list_heads")[0]
    active, heads = P.decode_heads(bytes.fromhex(v["bytes"]))
    assert active == v["fields"]["active_slot"]
    assert len(heads) == len(v["fields"]["heads"])
    assert [h.state for h in heads] == ["empty", "valid", "invalid"]
    assert heads[1].name == "focus" and heads[1].out_dim == 2
    assert heads[1].head_id == v["fields"]["heads"][1]["head_id"]
    assert heads[1].usable and not heads[2].usable


def test_status_reports_no_battery_as_no_battery():
    for v in by_kind("status"):
        f, s = v["fields"], P.decode_status(bytes.fromhex(v["bytes"]))
        if f.get("battery_percent_raw") == 255:
            assert s.battery_percent is None, "255 percent is not a measurement"
            continue
        assert s.streaming == f["streaming"]
        assert s.gain == 24 and s.sample_rate_hz == 500
        assert s.battery_percent == f["battery_percent"]
        assert s.usb_present and s.buffer_high_watermark
        assert s.dropped_total == f["dropped_total"]
        assert s.buffer_fill == f["buffer_fill"]
        assert s.charger == "charging"
        assert s.leadoff == f["loff_statp"]


def test_the_update_service():
    v = by_kind("update_request")[0]
    f = v["fields"]
    assert P.encode_update_start("app", f["slot"], f["total_len"],
                                 bytes.fromhex(f["transfer_id"])) == bytes.fromhex(v["bytes"])
    for v in by_kind("update_response"):
        f, r = v["fields"], P.decode_update_response(bytes.fromhex(v["bytes"]))
        assert r.op == f["op"] and r.status == f["status"], v["name"]
        if f["op"] == P.UPDATE_OPS["start"]:
            assert r.start() == (f["target_slot"], f["chunk_max"], f["resume_offset"])
        if f["op"] == P.UPDATE_OPS["query"]:
            state, offset, crc = r.query()
            assert (state, offset, crc) == ("receiving", f["offset"], f["crc32"])
        if "verify_result_name" in f:
            assert r.verify_result == f["verify_result_name"]
            assert not r.ok

    v = by_kind("envelope")[0]
    f, e = v["fields"], P.decode_envelope(bytes.fromhex(v["bytes"]))
    assert e.target == "weights" and e.slot_link == f["slot_link"]
    assert e.plain_len == f["plain_len"] and e.total_len == f["total_len"]


def test_a_prediction_names_the_head_that_made_it():
    v = by_kind("prediction")[0]
    f, p = v["fields"], P.decode_prediction(bytes.fromhex(v["bytes"]))
    assert p.head_slot == f["head_slot"]
    assert p.head_id == f["head_id"]
    assert p.index == f["sample_index"] and p.device_time == u64(f["device_time"])
    assert p.window_samples == f["window_samples"]
    assert p.duty_reduced and not p.gap_in_window and not p.leadoff_in_window
    assert len(p.outputs) == f["n_outputs"]
    for got, want in zip(p.outputs, f["outputs"]):
        assert close(got, want, 1e-6)


def test_a_head_is_read_checked_and_evaluated_the_way_the_device_does_it():
    v = by_kind("head")[0]
    f = v["fields"]
    blob = bytes.fromhex(v["bytes"])
    h = P.decode_head(blob)
    assert (h.in_dim, h.out_dim, h.name) == (f["in_dim"], f["out_dim"], f["name"])
    assert h.head_id == f["head_id"]
    assert h.weights == f["weights"] and h.bias == f["bias"]
    for got, want in zip(h.scale, f["scale"]):
        assert close(got, want, 1e-6)
    case = f["outputs_for_embedding"]
    for got, want in zip(P.evaluate_head(h, case["embedding"]), case["outputs"]):
        assert close(got, want, 1e-6)
    # Building one gives back exactly the same bytes, hash included, in the
    # format the vector was written in.
    assert P.build_head(h.weights, h.bias, h.scale, name=h.name, version=h.version, encoder_id=h.encoder_id) == blob
    # A head whose weights were changed after it was hashed is refused, the
    # same way the device refuses it.
    broken = bytearray(blob)
    broken[P.HEAD_HEADER_LEN] ^= 0x01
    try:
        P.decode_head(bytes(broken))
    except P.Invalid:
        pass
    else:
        raise AssertionError("a head that does not match its own hash was accepted")


def test_every_malformed_message_is_refused_for_its_stated_reason():
    kinds = {
        "device_info": P.decode_device_info,
        "eeg_data": lambda b: P.decode_packet(b, 4),
        "control_request": P.decode_request,
        "control_response": P.decode_response,
        "envelope": P.decode_envelope,
        "status": P.decode_status,
        "battery_info": P.decode_battery,
        "boot_info": P.decode_boot_info,
        "model_info": P.decode_model_info,
        "list_heads": P.decode_heads,
        "prediction": P.decode_prediction,
        "update_response": P.decode_update_response,
        "head": P.decode_head,
        "pipeline_catalog": P.decode_catalog,
        "chain": P.decode_chain,
        "pipeline_state": P.decode_pipeline_state,
        "prediction_input": P.decode_prediction_input,
        "bias_diagnostic": P.decode_bias_diagnostic,
        "converter_registers": P.decode_converter_registers,
        "embedding": P.decode_embedding,
    }
    expected = {"truncated": P.Truncated, "invalid": P.Invalid, "reserved": P.Reserved}
    seen = 0
    for v in VECTORS["refuse"]:
        decode = kinds.get(v["kind"])
        assert decode is not None, f"{v['kind']} has a refusal vector and no reader here"
        seen += 1
        try:
            decode(bytes.fromhex(v["bytes"]))
        except expected[v["reason"]]:
            pass
        except Exception as e:
            raise AssertionError(f"{v['name']}: refused as {type(e).__name__}, "
                                 f"and the contract calls it {v['reason']}") from e
        else:
            raise AssertionError(f"{v['name']}: accepted, and it is not a message")
    assert seen == len(VECTORS["refuse"]), "every refusal vector has a decoder here"


def test_the_lamp_the_registers_and_the_token_count_decode_from_the_vectors():
    lamp = next(v for v in by_kind("control_response") if "lamp" in v["name"])
    r = P.decode_response(bytes.fromhex(lamp["bytes"]))
    assert r.opcode == P.OPCODES["get_indicator"] and r.status == 0
    assert P.INDICATOR_LEVEL_NAMES[r.payload[0]] == lamp["fields"]["level_name"] == "reserved"

    regs = by_kind("converter_registers")[0]
    cr = P.decode_converter_registers(bytes.fromhex(regs["bytes"]))
    f = regs["fields"]
    assert (cr.family, cr.family_code, cr.first, cr.values.hex()) == (f["family_name"], f["family"], f["first"], f["values"])
    words = cr.describe()
    assert words["identity"]["device"] == "ADS1299" and words["identity"]["channels"] == 4
    assert words["data_rate_sps"] == 250 and words["test_signal"]["frequency"] == "fclk / 2^20"
    assert words["bias"]["amplifier_on"] is False and words["reference_buffer_on"] is True
    assert words["channels"][0]["gain"] == 24 and words["channels"][0]["input"] == "electrode"
    assert words["channels"][4]["powered_down"] is True and words["channels"][4]["input"] == "shorted"
    assert words["lead_off_sense_positive"] == [] and words["lead_off"]["current"] == "6 nA"
    assert words["common_reference_switch"] is False and words["single_shot"] is False

    infos = by_kind("model_info")
    for v in infos:
        mi = P.decode_model_info(bytes.fromhex(v["bytes"]))
        assert mi.tokens_per_channel == v["fields"]["tokens_per_channel"], v["name"]


def test_embeddings_decode_and_are_put_back_together():
    parts = by_kind("embedding")
    window = P.decode_embedding(bytes.fromhex(parts[0]["bytes"]))
    f = parts[0]["fields"]
    assert window.is_window_embedding and window.embed_dim == 96 and window.values == f["values"]
    assert window.leadoff_in_window and not window.more_parts and window.encoder_id == f["encoder_id"]
    assert window.device_time == u64(f["device_time"]) and window.index == f["sample_index"]
    one, two = (P.decode_embedding(bytes.fromhex(v["bytes"])) for v in parts[1:3])
    assert one.token == two.token == 17 and one.more_parts and not two.more_parts
    assert (one.first, len(one.values), two.first, len(two.values)) == (0, 108, 108, 20)

    # A whole window from notifications, arriving in any order: one channel,
    # three tokens of four values, and the window embedding.
    def packet(token, first, values, more=False):
        flags = P.EMBEDDING_FLAGS["more_parts"] if more else 0
        return P._EMBEDDING.pack(P.PACKET_EMBEDDING, flags, 4, first, 77, 1000, 2000,
                                 bytes.fromhex("a7c61680d9a202db"), 0, token) + \
            b"".join(int(v).to_bytes(2, "little", signed=True) for v in values)
    a = P.EmbeddingAssembler(channels=1, tokens_per_channel=3, form="both")
    pk = [packet(0, 0, [1, 2, 3, 4]), packet(2, 0, [9, 9], more=True), packet(P.WINDOW_TOKEN, 0, [5, 6, 7, 8]),
          packet(1, 0, [-1, -2, -3, -4]), packet(2, 2, [8, 8])]
    got = [a.feed(P.decode_embedding(p)) for p in pk]
    assert got[:4] == [None] * 4 and got[4] is not None
    w = got[4]
    assert w.embedding == [5, 6, 7, 8] and w.tokens == [[1, 2, 3, 4], [-1, -2, -3, -4], [9, 9, 8, 8]]
    assert w.index == 77 and w.embedding_floats()[0] == 5 / 4096
    assert w.token_array(1, 3).shape == (1, 3, 4)
    # Asking for the window form alone completes without any token.
    b = P.EmbeddingAssembler(channels=1, tokens_per_channel=3, form="window")
    assert b.feed(P.decode_embedding(packet(P.WINDOW_TOKEN, 0, [1, 1, 1, 1]))) is not None
    # A window left behind by two newer ones is dropped and counted.
    c = P.EmbeddingAssembler(channels=1, tokens_per_channel=3, form="tokens")
    for idx in (1, 2, 3, 4):
        raw = bytearray(packet(0, 0, [1, 2, 3, 4]))
        raw[4:8] = idx.to_bytes(4, "little")
        c.feed(P.decode_embedding(bytes(raw)))
    assert c.incomplete == 1


def test_a_count_is_worth_what_the_contract_says():
    for s in VECTORS["scaling"]:
        got = s["raw"] * (2.0 * s["vref_uv"]) / (s["gain"] * (1 << s["adc_bits"]))
        assert close(got, s["microvolts"], 1e-12), s
    # And the device's own numbers give the same answer.
    info = P.decode_device_info(bytes.fromhex(by_kind("device_info")[0]["bytes"]))
    assert close(info.microvolts_per_count(24), 0.022_351_741_790_771_484, 1e-12)


def test_the_timeline_is_judged_by_the_contracts_rule():
    for c in VECTORS["continuity"]:
        got = P.continuity(c["prev_index"], c["prev_n_samples"], c["new_index"],
                           c["discontinuity_flag"])
        assert got.kind == {"continuous": "continuous", "gap": "loss", "break": "rebase"}[c["verdict"]], c
        if got.kind == "loss":
            assert got.lost == c["lost"], c
        if got.kind == "rebase":
            assert got.lost is None, "a break has no count, and zero would be a different claim"
            assert c["lost"] is None, "and the contract states it as no count, not as zero"
    # The wrap is exact rather than four billion samples of imaginary loss.
    wrapped = P.continuity(0xFFFFFFFC, 5, 3, False)
    assert wrapped.kind == "loss" and wrapped.lost == 2


def test_an_embedding_quantizes_to_the_published_scale():
    for e in VECTORS["embedding"]:
        assert P.quantize_embedding(e["floats"]) == e["integers"], e


def _stages_of(field) -> list:
    return [(x["kind"], list(x["params"])) for x in field]


def _as_pairs(stages) -> list:
    return [(s.kind, list(s.params)) for s in stages]


def test_the_processing_messages_decode_to_their_fields():
    v = by_kind("pipeline_catalog")[0]
    cat = P.decode_catalog(bytes.fromhex(v["bytes"]))
    assert [(e.kind, P.STAGE_CLASSES[e.cls], e.n_params, e.max_instances, e.name) for e in cat] == \
        [(k["kind"], k["class"], k["n_params"], k["max_instances"], k["name"]) for k in v["fields"]["kinds"]]
    for v in by_kind("chain"):
        stages = P.decode_chain(bytes.fromhex(v["bytes"]))
        assert _as_pairs(stages) == _stages_of(v["fields"]["stages"]), v["name"]
        assert P.encode_chain(stages) == bytes.fromhex(v["bytes"]), "encoding gives back the same bytes"
    for v in by_kind("pipeline_state"):
        st = P.decode_pipeline_state(bytes.fromhex(v["bytes"]))
        assert P.PIPELINE_ORIGINS[st.origin] == v["fields"]["origin"], v["name"]
        assert _as_pairs(st.stages) == _stages_of(v["fields"]["stages"])
    for v in by_kind("prediction_input"):
        pi = P.decode_prediction_input(bytes.fromhex(v["bytes"]))
        assert P.INPUT_SOURCES[pi.source] == v["fields"]["source"], v["name"]
        assert _as_pairs(pi.stages) == _stages_of(v["fields"]["stages"])
    v = by_kind("bias_diagnostic")[0]
    d = P.decode_bias_diagnostic(bytes.fromhex(v["bytes"]))
    assert (d.mean_mv, d.sd_mv, d.min_mv, d.max_mv) == tuple(v["fields"][k] for k in ("mean_mv", "sd_mv", "min_mv", "max_mv"))
    # The two bytes 1.1 took from reserved.
    v = by_kind("model_info")[0]
    mi = P.decode_model_info(bytes.fromhex(v["bytes"]))
    assert mi.input_classes == v["fields"]["input_classes"] and mi.takes("time_domain")
    v = by_kind("prediction")[0]
    pr = P.decode_prediction(bytes.fromhex(v["bytes"]))
    assert P.INPUT_SOURCES[pr.input_source] == v["fields"]["input_source"]
    # The default chain the firmware emits is the one this module composes.
    v = [x for x in by_kind("chain") if "default" in x["name"]][0]
    assert _as_pairs(P.decode_chain(bytes.fromhex(v["bytes"]))) == _as_pairs(P.default_chain(500))


def test_the_rules_refuse_what_the_device_refuses_and_say_why():
    hp, lp, notch = P.highpass, P.lowpass, P.notch
    def refused(stages, rate, word):
        try:
            P.check_chain(stages, rate)
        except P.Invalid as e:
            assert word in str(e), f"{word!r} not in {e}"
        else:
            raise AssertionError(f"{P.describe_chain(stages, rate)} ran at {rate}")
    P.check_chain(P.default_chain(250), 250)
    P.check_chain(P.default_chain(1000), 1000)
    P.check_chain([], 500)
    refused([P.Stage(9, (5,))], 500, "catalog")
    refused([P.Stage(1, (5, 0))], 500, "parameter")
    refused([hp(0.5), hp(1.0)], 500, "at most 1")
    refused([notch(48, 52)] * 9, 500, "at most 8")
    refused([lp(225.0)], 500, "nine tenths")
    P.check_chain([lp(224.9)], 500)
    refused([lp(224.9)], 250, "nine tenths")
    refused([hp(0)], 500, "at least 0.1")
    refused([notch(52, 48)], 500, "high edge")
    refused([notch(0, 48)], 500, "above zero")
    refused([notch(118, 122)], 250, "nine tenths")
    refused([hp(30), lp(30)], 500, "passes nothing")
    refused([hp(200), lp()], 500, "passes nothing")
    P.check_chain([hp(200), lp()], 1000)
    # The default follows the rate: at 250 the 120 Hz band does not fit.
    assert len(P.default_chain(250)) == 1 + 3 + 1
    assert len(P.default_chain(500)) == 1 + 6 + 1
    assert _as_pairs(P.mains_bands(60)) == [(3, [580, 620]), (3, [1180, 1220]), (3, [1780, 1820])]
    assert _as_pairs(P.mains_bands(50, 250)) == [(3, [480, 520]), (3, [980, 1020])]
    assert _as_pairs(P.mains_bands(60, 250)) == [(3, [580, 620])]
    P.check_chain(P.mains_bands(50, 250), 250)
    assert P.describe_chain([]) == "natural signal"
    assert P.describe_chain([hp(0.5), notch(58, 62), lp()], 500) == "high-pass 0.5 Hz, notch 58 to 62 Hz, low-pass 200 Hz (automatic at 500 SPS)"
    assert P.output_class(P.default_chain(500)) == "time_domain"
    # A request carries a chain only for the model's own source.
    assert P.encode_prediction_input("natural") == bytes([1, 0])
    try:
        P.encode_prediction_input("stream", [hp(1.0)])
    except P.Invalid:
        pass
    else:
        raise AssertionError("a chain with the stream source was accepted")
    # Every 1.1 opcode is one a 1.0 device does not know, and none is reserved.
    for name in P.NEW_IN_1_1:
        assert P.OPCODES[name] not in P.RESERVED_OPCODES


def test_the_second_capability_word_and_the_models_appended_fields_decode():
    v = by_kind("device_info")[0]
    info = P.decode_device_info(bytes.fromhex(v["bytes"]))
    assert info.capabilities_high == v["fields"]["capabilities_high"]
    assert info.can("synthetic") and info.can("device_name") and info.can("model_cadence")
    # A 1.2 device's 76 bytes claim nothing in the second word.
    older = bytearray(bytes.fromhex(v["bytes"])[:76])
    older[27] = 76
    assert P.decode_device_info(bytes(older)).capabilities_high == 0
    assert not P.decode_device_info(bytes(older)).can("synthetic")
    for v in by_kind("model_info"):
        mi = P.decode_model_info(bytes.fromhex(v["bytes"]))
        f = v["fields"]
        assert (mi.pass_ms, mi.interval_s, mi.generator) == (f.get("pass_ms", 0), f.get("interval_s", 0), f.get("generator", False)), v["name"]


def test_the_cadence_and_the_name_encode_and_decode_as_the_vectors_say():
    for v in by_kind("control_response"):
        f, r = v["fields"], P.decode_response(bytes.fromhex(v["bytes"]))
        if r.opcode == P.OPCODES["get_model_interval"] and r.ok:
            mi = P.decode_model_interval(r.payload)
            assert (mi.interval_s, mi.minimum_s) == (f["interval_s"], f["minimum_s"])
        if r.opcode == P.OPCODES["get_name"] and r.ok:
            assert P.decode_name_parts(r.payload) == (f["name"], f["adjective"])
    assert P.compose_name("Ada", "Blue") == "Ada's Blue IntoMind One"
    assert P.compose_name("", "Blue") == "Blue IntoMind One"
    assert P.compose_name("Ada", "") == "Ada's IntoMind One"
    assert P.compose_name("", "") == "IntoMind One"
    assert P.name_fits("Beatrice", "Green"), "8 + 5 makes 29 exactly"
    assert not P.name_fits("Beatrice", "Purple"), "8 + 6 makes 30"
    assert not P.name_fits(" Ada", "") and not P.name_fits("Ada ", "") and not P.name_fits("A\x01", "")
    assert P.name_fits("José", ""), "UTF-8 is counted in bytes and allowed"
    for bad in (b"", b"\x04Ada\x04Blue", b"\x03Ada\x03Blue", b"\x03Ada"):
        try:
            P.decode_name_parts(bad)
        except P.ProtocolError:
            pass
        else:
            raise AssertionError(f"{bad!r} was read as a name")


def test_a_head_names_its_encoder_and_the_list_carries_it():
    blob = P.build_head([[1, -2, 3]], [7], [0.5], name="focus", encoder_id="a7c61680d9a202db")
    h = P.decode_head(blob)
    assert (h.version, h.encoder_id, h.in_dim, h.out_dim) == (2, "a7c61680d9a202db", 3, 1)
    assert len(blob) == P.head_blob_len(3, 1, 2) == P.head_blob_len(3, 1, 1) + 8
    plain = P.decode_head(P.build_head([[1, -2, 3]], [7], [0.5], name="focus"))
    assert plain.encoder_id == P.NO_ENCODER_ID
    v = by_kind("list_heads")[0]
    _active, heads = P.decode_heads(bytes.fromhex(v["bytes"]))
    assert heads[1].encoder_id == v["fields"]["heads"][1]["encoder_id"]
    assert heads[1].trained_beside("a7c61680d9a202db") is True
    assert heads[1].trained_beside("0000000000000001") is False
    assert heads[0].trained_beside("a7c61680d9a202db") is None, "a head that does not say is not judged"


def test_a_synthetic_packet_decodes_like_a_measured_one_and_says_so():
    for v in by_kind("eeg_data"):
        p = P.decode_packet(bytes.fromhex(v["bytes"]), 4)
        assert p.synthetic == (v["fields"]["packet_type"] == P.PACKET_SYNTHETIC), v["name"]
        if p.synthetic:
            assert p.mode == "synthetic" and p.counts == v["fields"]["samples"]
    for v in by_kind("status"):
        st = P.decode_status(bytes.fromhex(v["bytes"]))
        assert (st.mode == "synthetic") == (v["fields"].get("mode") == 3), v["name"]


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
