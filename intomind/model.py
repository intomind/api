"""The encoder, on the host, and what is built on its output.

The device turns four seconds of signal into an embedding. `Encoder` is
the same encoder in numpy, held to the reference embeddings the training
framework produced, as the firmware is: given the weights file a device
runs, it gives the embeddings that device gives. The encoder's weights are
not published. They reach a device only as a signed image, and the device
sends its embeddings and tokens instead (`Device.collect_embeddings`).

`train_head` fits a head to those embeddings and returns the bytes the
device will run. `Reconstructor` turns the device's tokens back into
signal.

    from intomind.model import train_head

    windows = await device.collect_embeddings(100)      # while streaming
    blob = train_head([w.embedding for w in windows], targets, name="focus",
                      encoder_id=windows[0].encoder_id)

Nothing here is fast on purpose: it is numpy, which is fast enough for a
recording and far simpler than anything that would be faster.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from . import protocol as P

MAGIC = b"IMW1"
FORMAT_VERSION = 1
HEADER_LEN = 32
#: The floor under a channel's spread, in microvolts, so a flat channel
#: cannot explode when it is divided by its own spread. The training
#: code's value, and it is absolute, which is why a window has to be in
#: real microvolts rather than converter counts.
MAD_FLOOR_UV = 0.1
MAD_TO_SIGMA = 1.4826


class WeightsError(Exception):
    """Bytes that are not a weights file, with the reason."""


@dataclass(frozen=True)
class Header:
    layers: int
    d_model: int
    d_ff: int
    heads: int
    patch: int
    tokens: int
    native_sps: int
    encoder_id: str

    @property
    def window_samples(self) -> int:
        return self.patch * self.tokens

    @property
    def d_head(self) -> int:
        return self.d_model // self.heads


def _linear(blob: memoryview, at: int, out_dim: int, in_dim: int):
    """One int8 matrix with a scale per output row, and its bias, brought
    back to floats. The device dequantizes one row at a time because it has
    no memory to spare. A host has memory to spare."""
    q = np.frombuffer(blob, dtype=np.int8, count=out_dim * in_dim, offset=at).reshape(out_dim, in_dim)
    at += out_dim * in_dim
    scale = np.frombuffer(blob, dtype="<f4", count=out_dim, offset=at)
    at += out_dim * 4
    bias = np.frombuffer(blob, dtype="<f4", count=out_dim, offset=at).astype(np.float32)
    at += out_dim * 4
    return (q.astype(np.float32) * scale[:, None]).astype(np.float32), bias, at


def _floats(blob: memoryview, at: int, n: int):
    return np.frombuffer(blob, dtype="<f4", count=n, offset=at).astype(np.float32), at + n * 4


def _gelu(x):
    """The exact form, which is what the model was trained with. The tanh
    approximation is a different function."""
    from math import sqrt
    try:
        from scipy.special import erf            # noqa: PLC0415
        return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))
    except ImportError:
        v = np.vectorize(__import__("math").erf, otypes=[np.float32])
        return 0.5 * x * (1.0 + v(x / np.float32(sqrt(2.0))))


def _layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def normalize(channel) -> np.ndarray:
    """One channel, against its own middle and spread.

    Per window and per channel, which makes the model blind to amplifier
    gain and to a resting offset, and is what it was trained under. The
    median is the lower one of the two middle values, which is what the
    training framework's median returns.
    """
    x = np.asarray(channel, dtype=np.float64)
    order = np.sort(x)
    median = order[(len(x) - 1) // 2]
    deviations = np.sort(np.abs(x - median))
    mad = deviations[(len(x) - 1) // 2] * MAD_TO_SIGMA
    return ((x - median) / max(mad, MAD_FLOOR_UV)).astype(np.float32)


def resample(channel, source_sps: int, native_sps: int) -> np.ndarray:
    """A channel onto the model's grid, linearly."""
    x = np.asarray(channel, dtype=np.float64)
    if source_sps == native_sps or len(x) == 0:
        return x.astype(np.float32)
    n = int(len(x) * native_sps / source_sps)
    at = np.arange(n) * (source_sps / native_sps)
    return np.interp(at, np.arange(len(x)), x).astype(np.float32)


class Encoder:
    """The encoder, loaded from the weights the device runs."""

    def __init__(self, header: Header, descriptor, time_emb, proj, layers, final_norm):
        self.header = header
        self._descriptor = descriptor
        self._time_emb = time_emb
        self._proj = proj
        self._layers = layers
        self._final_norm = final_norm

    @classmethod
    def load(cls, source) -> "Encoder":
        """From a path, or from the bytes themselves."""
        if isinstance(source, (bytes, bytearray, memoryview)):
            blob = bytes(source)
        else:
            import pathlib
            blob = pathlib.Path(source).read_bytes()
        if len(blob) < HEADER_LEN or blob[0:4] != MAGIC or blob[4] != FORMAT_VERSION:
            raise WeightsError("not a weights file this version reads")
        layers, d_model, d_ff = blob[5], *struct.unpack_from("<HH", blob, 6)
        heads = blob[10]
        patch, tokens, native = struct.unpack_from("<HHH", blob, 12)
        if not (d_model and heads and d_model % heads == 0 and patch and tokens and layers):
            raise WeightsError("a weights file whose shape cannot be run")
        header = Header(layers, d_model, d_ff, heads, patch, tokens, native,
                        blob[20:28].hex())
        mv = memoryview(blob)
        at = HEADER_LEN
        descriptor, at = _floats(mv, at, d_model)
        time_emb, at = _floats(mv, at, tokens * d_model)
        proj_w, proj_b, at = _linear(mv, at, d_model, patch)
        stack = []
        for _ in range(layers):
            n1w, at = _floats(mv, at, d_model)
            n1b, at = _floats(mv, at, d_model)
            qkv_w, qkv_b, at = _linear(mv, at, 3 * d_model, d_model)
            out_w, out_b, at = _linear(mv, at, d_model, d_model)
            n2w, at = _floats(mv, at, d_model)
            n2b, at = _floats(mv, at, d_model)
            ff1_w, ff1_b, at = _linear(mv, at, d_ff, d_model)
            ff2_w, ff2_b, at = _linear(mv, at, d_model, d_ff)
            stack.append(dict(n1w=n1w, n1b=n1b, qkv_w=qkv_w, qkv_b=qkv_b,
                              out_w=out_w, out_b=out_b, n2w=n2w, n2b=n2b,
                              ff1_w=ff1_w, ff1_b=ff1_b, ff2_w=ff2_w, ff2_b=ff2_b))
        fnw, at = _floats(mv, at, d_model)
        fnb, at = _floats(mv, at, d_model)
        if at > len(blob):
            raise WeightsError("a weights file that ends before its own tensors")
        return cls(header, descriptor, time_emb.reshape(tokens, d_model),
                   (proj_w, proj_b), stack, (fnw, fnb))

    def embed(self, window, live=None, *, source_sps: int | None = None) -> np.ndarray:
        """A window of microvolts to an embedding.

        `window` is `(channels, samples)` in microvolts. `live` says which
        channels are attached: one that is not is neither attended to nor
        pooled, which is what keeps a fallen electrode out of the answer.
        `source_sps` resamples onto the model's own grid first, for a
        recording made at another rate.
        """
        tokens, live = self._forward(window, live, source_sps)
        return tokens[live].mean(axis=(0, 1)).astype(np.float32)

    def _forward(self, window, live, source_sps):
        h = self.header
        x = np.asarray(window, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("a window is (channels, samples)")
        if source_sps and source_sps != h.native_sps:
            x = np.stack([resample(c, source_sps, h.native_sps) for c in x])
        channels, samples = x.shape
        if samples < h.window_samples:
            raise ValueError(f"this model reads {h.window_samples} samples and was given {samples}")
        x = x[:, -h.window_samples:]
        live = np.ones(channels, dtype=bool) if live is None else np.asarray(live, dtype=bool)
        if live.shape != (channels,):
            raise ValueError("one liveness flag per channel")
        if not live.any():
            live = np.ones(channels, dtype=bool)
        tokens = self._run(x, live, channels, tokens_only=True)
        return tokens, live

    def _run(self, x, live, channels, tokens_only: bool = False):
        h = self.header
        normalized = np.stack([normalize(c) for c in x])
        patches = normalized.reshape(channels, h.tokens, h.patch)
        proj_w, proj_b = self._proj
        tokens = patches @ proj_w.T + proj_b + self._descriptor + self._time_emb
        seq = tokens.reshape(channels * h.tokens, h.d_model)

        # A key that belongs to a channel that is not attached is not
        # attended to. Every query still runs, and the output of a dead
        # channel's token is simply not pooled.
        key_live = np.repeat(live, h.tokens)
        mask = np.where(key_live, 0.0, -np.inf).astype(np.float32)

        for layer in self._layers:
            normed = _layer_norm(seq, layer["n1w"], layer["n1b"])
            qkv = normed @ layer["qkv_w"].T + layer["qkv_b"]
            d, dh = h.d_model, h.d_head
            q = qkv[:, :d].reshape(-1, h.heads, dh).transpose(1, 0, 2)
            k = qkv[:, d:2 * d].reshape(-1, h.heads, dh).transpose(1, 0, 2)
            v = qkv[:, 2 * d:].reshape(-1, h.heads, dh).transpose(1, 0, 2)
            scores = q @ k.transpose(0, 2, 1) / np.sqrt(dh) + mask
            scores = scores - scores.max(axis=-1, keepdims=True)
            weights = np.exp(scores)
            weights /= weights.sum(axis=-1, keepdims=True)
            attended = (weights @ v).transpose(1, 0, 2).reshape(-1, d)
            seq = seq + attended @ layer["out_w"].T + layer["out_b"]
            normed = _layer_norm(seq, layer["n2w"], layer["n2b"])
            hidden = _gelu(normed @ layer["ff1_w"].T + layer["ff1_b"])
            seq = seq + hidden @ layer["ff2_w"].T + layer["ff2_b"]

        out = _layer_norm(seq, self._final_norm[0], self._final_norm[1])
        out = out.reshape(channels, h.tokens, h.d_model)
        if tokens_only:
            return out.astype(np.float32)
        return out[live].mean(axis=(0, 1)).astype(np.float32)

    def embed_tokens(self, window, live=None, *, source_sps: int | None = None) -> np.ndarray:
        """The per token output, before it is pooled.

        `(channels, tokens, width)`. The pooled embedding is the mean of
        this over live channels and time. Reconstruction needs the tokens
        themselves, because each one describes its own fifth of a second.
        """
        return self._forward(window, live, source_sps)[0]

    def embed_capture(self, capture, *, hop_seconds: float | None = None):
        """Every window of a recording, and where each one starts.

        Yields `(row, embedding)`. `row` is the index into the capture's
        samples, so a caller can line an embedding up against anything else
        on the same timeline.
        """
        h = self.header
        fs = float(capture.meta.get("fs_effective") or capture.meta["rate_sps"])
        window_rows = int(round(h.window_samples * fs / h.native_sps))
        hop = int(round((hop_seconds if hop_seconds else h.window_samples / h.native_sps) * fs))
        data = np.stack([capture.uv(c) for c in range(capture.nch)])
        for row in range(0, data.shape[1] - window_rows + 1, max(hop, 1)):
            yield row, self.embed(data[:, row:row + window_rows],
                                  source_sps=int(round(fs)))


RECON_MAGIC = b"IMR1"


class Reconstructor:
    """Turns a token's embedding back into the patch it came from.

    This is what makes an embedding inspectable. It is not a head: a head
    produces a few numbers for a person to act on, and this produces a
    fifth of a second of signal, which is a different object. It runs on a
    host.

    What comes back is in the normalized units the model works in, which
    are the signal with its own middle and spread taken out. To compare it
    against a recording, normalize the recording the same way.
    """

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    @property
    def d_model(self) -> int:
        return self.weights.shape[1]

    @property
    def patch(self) -> int:
        return self.weights.shape[0]

    #: The encoder the shipped reconstruction head decodes, by the id every
    #: embedding window from the device carries.
    SHIPPED_ENCODER_ID = "e2f9b60f411f0e73"

    @classmethod
    def shipped(cls) -> "Reconstructor":
        """The reconstruction head for the encoder the IntoMind One ships
        with: the open file that turns its tokens back into signal, the same
        bytes the device carries beside its encoder. Fitted on the corpus to
        the device's own tokens; it returns 0.66 of a held-out window's
        variance, and what it cannot return is what the tokens do not carry.
        """
        import importlib.resources
        blob = importlib.resources.files(__package__).joinpath("data/ladder_mid_d76_l3_reconstruction.imr").read_bytes()
        return cls.load(blob)

    @classmethod
    def load(cls, source) -> "Reconstructor":
        if isinstance(source, (bytes, bytearray, memoryview)):
            blob = bytes(source)
        else:
            import pathlib
            blob = pathlib.Path(source).read_bytes()
        if len(blob) < 10 or blob[0:4] != RECON_MAGIC or blob[4] != 1:
            raise WeightsError("not a reconstruction head this version reads")
        d_model, patch = struct.unpack_from("<HH", blob, 6)
        want = 10 + patch * d_model * 4 + patch * 4
        if len(blob) != want:
            raise WeightsError(f"a reconstruction head of {patch} by {d_model} is {want} bytes, and this is {len(blob)}")
        w = np.frombuffer(blob, dtype="<f4", count=patch * d_model, offset=10).reshape(patch, d_model)
        b = np.frombuffer(blob, dtype="<f4", count=patch, offset=10 + patch * d_model * 4)
        return cls(w.astype(np.float32), b.astype(np.float32))

    def patches(self, tokens) -> np.ndarray:
        """`(channels, tokens, width)` to `(channels, tokens, patch)`."""
        t = np.asarray(tokens, dtype=np.float32)
        if t.shape[-1] != self.d_model:
            raise ValueError(f"this head reads a width of {self.d_model} and was given {t.shape[-1]}")
        return t @ self.weights.T + self.bias

    def from_tokens(self, tokens, *, channels: int | None = None, tokens_per_channel: int | None = None) -> np.ndarray:
        """Signal from tokens the device sent.

        `tokens` is an `EmbeddingWindow.tokens` list (quantized, channel
        major) with `channels` and `tokens_per_channel` from the device, or
        any `(channels, tokens, width)` array of floats or quantized
        integers. The result is `(channels, samples)` in the model's
        normalized units, like `generate`. This is how generation works on
        a host with the weights closed: the device runs the encoder and
        sends the tokens, and this open head turns them back into signal.
        Mixing, moving, or sampling in the token space before this call is
        generating synthetic signal.
        """
        t = np.asarray(tokens)
        if t.dtype.kind in "iu":
            t = t.astype(np.float32) / P.EMBED_SCALE
        t = t.astype(np.float32)
        if t.ndim == 2:
            if channels is None or tokens_per_channel is None:
                raise ValueError("a flat token list needs channels and tokens_per_channel to take its shape")
            t = t.reshape(channels, tokens_per_channel, t.shape[-1])
        if t.ndim != 3:
            raise ValueError("tokens are (channels, tokens, width)")
        p = self.patches(t)
        return p.reshape(p.shape[0], -1)

    def generate(self, encoder: "Encoder", window, live=None, *, source_sps: int | None = None) -> np.ndarray:
        """A window, through the model, and back out as signal.

        The result is `(channels, samples)` in normalized units. It is what
        the model believes the window was, which is not the window: the
        difference between them is what the embedding could not carry, and
        looking at that difference is the point.
        """
        tokens = encoder.embed_tokens(window, live, source_sps=source_sps)
        p = self.patches(tokens)
        return p.reshape(p.shape[0], -1)


def gap_ridge(quantized) -> float:
    """The ridge `train_head` uses unless told otherwise, from the
    embeddings alone.

    The encoder ends in a layer normalization, so every embedding it makes
    lies on one hyperplane, and only the integer rounding moves it off. In
    the training set that shows as one direction with almost no variance.
    Plain least squares with a small ridge puts a large weight along that
    direction to fit the rounding noise, and the weight costs nothing in
    float. It costs a lot in the device's integers: a row is scaled to its
    largest weight, so the informative weights are left with a few levels
    each. Measured on the shipping encoder, the integer head's scores then
    correlated 0.92 with its own float fit; with this ridge, 0.9999.

    The ridge is the geometric mean of the two smallest eigenvalues of the
    centered embeddings' Gram matrix: inside the gap between the dead
    direction and the first real one, so the dead direction is removed and
    the real ones are left essentially unshrunk. No label enters the
    choice. The smallest eigenvalue is floored at one so an exactly dead
    direction still gives a finite ridge.
    """
    x = np.asarray(quantized, dtype=np.float64)
    xc = x - x.mean(axis=0)
    ev = np.linalg.eigvalsh(xc.T @ xc)
    return float(np.sqrt(max(ev[0], 1.0) * ev[1]))


def train_head(embeddings, targets, *, name: str = "", ridge: float | None = None, encoder_id: str | None = None):
    """Fit a head to embeddings, and return the blob to upload.

    `embeddings` is one row per window, either the floats this encoder
    produced or the integers a head receives. `targets` is one row per
    window with one column per output.

    The fit is ordinary least squares with a ridge term. The ridge comes
    from the embeddings themselves unless given (`gap_ridge` says why), so
    there is nothing to tune. The result is quantized the way the device
    holds it, and the returned blob is what the device will actually run,
    so what you measure here is what you get there.

    `encoder_id` names the encoder the embeddings came from, as every
    embedding window the device sends carries it (`EmbeddingWindow.encoder_id`).
    The head file records it, the device stores and reports it, and a host
    warns when the head is uploaded to a device running another encoder.
    Nothing blocks: the head is yours.
    """
    e = np.asarray(embeddings, dtype=np.float64)
    if e.ndim != 2:
        raise ValueError("embeddings are one row per window")
    if np.issubdtype(np.asarray(embeddings).dtype, np.floating):
        e = np.array([P.quantize_embedding(row) for row in e], dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if len(e) != len(y):
        raise ValueError(f"{len(e)} embeddings and {len(y)} targets")
    if len(e) <= e.shape[1]:
        raise ValueError(
            f"fitting {e.shape[1]} weights needs more than {len(e)} windows, "
            "or the head will describe the noise in this recording rather than the signal")
    if ridge is None:
        ridge = gap_ridge(e)

    # Solve with an intercept, then fold the intercept into the bias.
    design = np.hstack([e, np.ones((len(e), 1))])
    penalty = ridge * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0                       # never penalize the intercept
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y).T
    weights_f, intercepts = coefficients[:, :-1], coefficients[:, -1]

    rows, bias, scale = [], [], []
    for o in range(weights_f.shape[0]):
        peak = np.abs(weights_f[o]).max()
        s = float(peak / 127.0) if peak > 0 else 1.0
        rows.append(np.clip(np.rint(weights_f[o] / s), -127, 127).astype(int).tolist())
        bias.append(int(round(intercepts[o] / s)))
        scale.append(s)
    return P.build_head(rows, bias, scale, name=name, encoder_id=encoder_id)
