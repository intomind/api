"""The encoder on the host, against the same references the device is.

The weights file the device runs is the weights file this reads, and the
embeddings the training framework produced are what both are measured
against. When the two agree, a head trained on a host behaves on the
device exactly as it behaved in training, which is the whole reason a head
is a portable blob.
"""
from __future__ import annotations
import pathlib, struct, sys, traceback

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from intomind import protocol as P            # noqa: E402
from intomind.model import Encoder, Reconstructor, WeightsError, normalize, resample, train_head  # noqa: E402

DATA = pathlib.Path(__file__).resolve().parent / "data"


def vectors():
    b = DATA.joinpath("tiny_vectors.bin").read_bytes()
    n, channels, samples, d_model = struct.unpack_from("<4I", b, 0)
    at = 16
    def take(count):
        nonlocal at
        v = np.frombuffer(b, dtype="<f4", count=count, offset=at)
        at += count * 4
        return v
    windows = [take(channels * samples).reshape(channels, samples) for _ in range(n)]
    exact = [take(d_model) for _ in range(n)]
    quantized = [take(d_model) for _ in range(n)]
    return windows, exact, quantized


def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_the_host_reproduces_the_framework_exactly():
    e = Encoder.load(DATA / "tiny.imw")
    windows, _exact, quantized = vectors()
    for i, w in enumerate(windows):
        got = e.embed(w)
        assert cosine(got, quantized[i]) > 0.999_999, f"window {i}"
        worst = float(np.max(np.abs(got - quantized[i])))
        assert worst < 1e-4 * max(1.0, float(np.max(np.abs(quantized[i])))), f"window {i}: {worst}"


def test_the_price_of_the_devices_arithmetic_is_what_it_was_measured_to_be():
    e = Encoder.load(DATA / "tiny.imw")
    windows, exact, _q = vectors()
    worst = min(cosine(e.embed(w), exact[i]) for i, w in enumerate(windows))
    assert worst > 0.9975, f"holding the weights as integers cost a cosine of {worst}"


def test_a_channel_that_fell_off_is_left_out_rather_than_believed():
    e = Encoder.load(DATA / "tiny.imw")
    windows, _e, _q = vectors()
    w = windows[0].copy()
    w[2, :] = 9_000.0                      # a detached electrode reads like this
    everything = e.embed(w)
    without = e.embed(w, live=[True, True, False])
    assert cosine(everything, without) < 0.9999, "excluding a dead channel has to change the answer"
    nothing = e.embed(w, live=[False, False, False])
    assert cosine(everything, nothing) > 0.999_99, (
        "no channel live is no information, not nothing to attend to")


def test_normalizing_removes_offset_and_gain():
    base = np.arange(40, dtype=float) - 19.5
    a = normalize(base * 10.0 + 500.0)
    b = normalize(base * 1000.0 - 20.0)
    assert np.allclose(a, b, atol=1e-3), "a hundredfold gain difference is not a difference"
    flat = normalize(np.full(40, 7.0))
    assert np.allclose(flat, 0.0), "no spread is no signal, not infinity"


def test_resampling_hits_the_native_grid():
    line = np.arange(100, dtype=float) * 3.0 - 7.0
    up = resample(line, 250, 500)
    assert len(up) == 200
    assert abs(up[2] - line[1]) < 1e-6
    down = resample(line, 1000, 500)
    assert len(down) == 50
    assert abs(down[1] - line[2]) < 1e-6
    assert np.allclose(resample(line, 500, 500), line)


def test_a_damaged_weights_file_is_refused():
    blob = bytearray(DATA.joinpath("tiny.imw").read_bytes())
    Encoder.load(bytes(blob))
    for spoil in (lambda b: b.__setitem__(0, 0x58), lambda b: b.__setitem__(4, 2),
                  lambda b: b.__setitem__(10, 3)):
        bad = bytearray(blob)
        spoil(bad)
        try:
            Encoder.load(bytes(bad))
        except WeightsError:
            pass
        else:
            raise AssertionError("a weights file that cannot be run was accepted")


def test_a_head_trained_here_is_the_head_the_device_runs():
    e = Encoder.load(DATA / "tiny.imw")
    rng = np.random.default_rng(5)
    # A target that really is a function of the embedding, so the fit has
    # something to find.
    windows = [rng.normal(0, 12.0, size=(3, e.header.window_samples)) + rng.normal(0, 40.0, size=(3, 1))
               for _ in range(60)]
    embeddings = np.stack([e.embed(w) for w in windows])
    direction = rng.normal(size=e.header.d_model)
    targets = embeddings @ direction

    blob = train_head(embeddings, targets, name="demo")
    head = P.decode_head(blob)
    assert head.in_dim == e.header.d_model and head.out_dim == 1
    assert head.name == "demo"

    # What the head says here is what the device will say, because the
    # device evaluates exactly these integers.
    predicted = np.array([P.evaluate_head(head, P.quantize_embedding(v))[0] for v in embeddings])
    r = float(np.corrcoef(predicted, targets)[0, 1])
    assert r > 0.99, f"a head fitted to a linear target should recover it, and this one got {r:.3f}"


def test_a_head_cannot_be_fitted_from_fewer_windows_than_it_has_weights():
    rng = np.random.default_rng(1)
    e = rng.normal(size=(4, 8))
    try:
        train_head(e, rng.normal(size=4))
    except ValueError as err:
        assert "windows" in str(err)
    else:
        raise AssertionError("a head with more weights than windows describes the noise")


def reconstruction_blob(patch, d_model, weights, bias):
    """A reconstruction head in the published format."""
    import struct
    head = struct.pack("<4sBBHH", b"IMR1", 1, 0, d_model, patch)
    return head + np.asarray(weights, dtype="<f4").tobytes() + np.asarray(bias, dtype="<f4").tobytes()


def test_a_reconstruction_head_reads_back_what_was_written():
    w = np.arange(300, dtype=np.float32).reshape(100, 3) / 100.0
    b = np.arange(100, dtype=np.float32)
    r = Reconstructor.load(reconstruction_blob(100, 3, w, b))
    assert (r.patch, r.d_model) == (100, 3)
    assert np.allclose(r.weights, w) and np.allclose(r.bias, b)

    # One token through it is one patch, computed the obvious way.
    token = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
    out = r.patches(token)
    assert out.shape == (1, 1, 100)
    assert np.allclose(out[0, 0], w @ token[0, 0] + b, atol=1e-4)

    for spoil in (lambda x: x[:9], lambda x: b"XXXX" + x[4:], lambda x: x[:-4]):
        try:
            Reconstructor.load(spoil(reconstruction_blob(100, 3, w, b)))
        except WeightsError:
            pass
        else:
            raise AssertionError("a reconstruction head that cannot be run was accepted")
    try:
        r.patches(np.zeros((1, 1, 4), dtype=np.float32))
    except ValueError as e:
        assert "width" in str(e)
    else:
        raise AssertionError("a token of the wrong width was accepted")


def test_generating_gives_back_a_window_shaped_thing():
    e = Encoder.load(DATA / "tiny.imw")
    h = e.header
    rng = np.random.default_rng(2)
    # A head that reads only the first component, so what comes back is
    # predictable from the tokens and the arithmetic is checkable.
    w = np.zeros((h.patch, h.d_model), dtype=np.float32)
    w[:, 0] = 1.0
    r = Reconstructor.load(reconstruction_blob(h.patch, h.d_model, w, np.zeros(h.patch)))

    window = rng.normal(0, 12.0, size=(3, h.window_samples))
    tokens = e.embed_tokens(window)
    assert tokens.shape == (3, h.tokens, h.d_model)
    # The pooled embedding is the mean of the tokens, which is what makes
    # a head and a reconstruction two readouts of one forward pass.
    assert np.allclose(tokens.mean(axis=(0, 1)), e.embed(window), atol=1e-5)

    out = r.generate(e, window)
    assert out.shape == (3, h.window_samples)
    for c in range(3):
        for t in range(h.tokens):
            expected = tokens[c, t, 0]
            got = out[c, t * h.patch:(t + 1) * h.patch]
            assert np.allclose(got, expected, atol=1e-4), f"channel {c} token {t}"


def test_a_dead_channel_changes_the_tokens_it_is_left_out_of():
    e = Encoder.load(DATA / "tiny.imw")
    windows, _e, _q = vectors()
    w = windows[0].copy()
    w[2, :] = 9_000.0
    everything = e.embed_tokens(w)
    without = e.embed_tokens(w, live=[True, True, False])
    assert everything.shape == without.shape
    assert not np.allclose(everything[0], without[0]), (
        "leaving a channel out has to change what the others attend to")


def test_the_default_ridge_keeps_the_integer_row_for_the_directions_that_carry_signal():
    # Embeddings the way the encoder's last normalization leaves them: every
    # row on one hyperplane, so one direction holds nothing but the
    # integer rounding. The target leans slightly on that rounding, which
    # is what a fit finds by chance in real data and is built in here so
    # the test is deterministic. A fixed small ridge lets least squares
    # spend the row's integer range on the dead direction; the ridge chosen
    # from the embeddings removes it and leaves the informative weights
    # their resolution.
    from intomind.model import gap_ridge
    rng = np.random.default_rng(3)
    n, d = 400, 16
    z = rng.normal(size=(n, d)) * 1200.0
    normal = 1.0 / rng.uniform(0.5, 2.0, size=d)
    normal /= np.linalg.norm(normal)
    z -= np.outer(z @ normal, normal)
    e = np.rint(z).astype(np.int64)                    # already the device's integers
    direction = rng.normal(size=d)
    direction -= (direction @ normal) * normal
    direction /= np.linalg.norm(direction)
    targets = (e @ direction) / 1200.0 + 0.05 * (e @ normal) + 0.3 * rng.normal(size=n)

    ridge = gap_ridge(e)
    ev = np.linalg.eigvalsh((e - e.mean(axis=0)).T @ (e - e.mean(axis=0)))
    assert ev[0] <= ridge <= ev[1], f"the ridge {ridge:.1f} is not in the gap {ev[0]:.1f}..{ev[1]:.1f}"

    def anatomy(blob):
        head = P.decode_head(blob)
        w = np.array(head.weights[0], dtype=float) * head.scale[0]
        dead = float(w @ normal)
        informative = w - dead * normal
        levels = float(np.abs(informative).max() / head.scale[0])
        predicted = np.array([P.evaluate_head(head, [int(v) for v in row])[0] for row in e])
        return abs(dead) / np.linalg.norm(informative), levels, float(np.corrcoef(predicted, targets)[0, 1])

    dead_share, levels, r = anatomy(train_head(e, targets))
    assert dead_share < 0.02, f"the chosen ridge left {dead_share:.3f} of the row on the dead direction"
    assert levels > 60, f"the informative weights have {levels:.0f} of 127 levels"
    assert r > 0.95, f"the head the device runs recovers the target at r {r:.3f}"

    dead_share, levels, _r = anatomy(train_head(e, targets, ridge=1.0))
    assert dead_share > 1.0, f"a fixed ridge of 1.0 should let the dead direction dominate, and it has {dead_share:.3f}"
    assert levels < 10, f"under a fixed ridge the informative weights should be left few levels, and they have {levels:.0f}"


def test_the_shipped_reconstruction_head_is_the_devices_decoder():
    r = Reconstructor.shipped()
    assert (r.patch, r.d_model) == (100, 76)
    import hashlib, importlib.resources
    blob = importlib.resources.files("intomind").joinpath("data/ladder_mid_d76_l3_reconstruction.imr").read_bytes()
    assert hashlib.sha256(blob).hexdigest().startswith("636c50bedea1f084"), "the decoder the generator was made for"


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
