# intomind

The IntoMind instrument API. One library, every IntoMind device.

```python
from intomind.client import display_names, scan
from intomind.session import Session

found = await scan()                   # every device that answered, as heard
labels = display_names(found)          # each one's name, numbered when two share one
chosen = next(d for d in found if labels[d.address] == "Ada's IntoMind One")
s = Session()
await s.connect(chosen)                # only the one picked
device = s.devices[0]
```

It owns the link to a device, the protocol spoken over it, the recording
of what it streams, the provenance of that recording, the formats it can
be read out as, and the analyses that read it back. It knows nothing
about any particular device beyond what that device tells it, and nothing
about any application built on top of it.

## What is in here

| | |
|---|---|
| `protocol.py` | the wire, and nothing else |
| `client.py` | one device: discovery, control, streaming, the clock |
| `session.py` | discovery and the set of connected devices |
| `instrument.py` | instrument state and decoding, independent of any protocol |
| `experiments.py` | the recorder, and the cued protocols built on it |
| `provenance.py` | what a capture records about how it was produced |
| `analysis.py` | the analyses a capture supports |
| `export.py` | a capture as npz, EDF, BDF, csv, tsv, or MATLAB |
| `model.py` | the device's encoder, on the host, and head training |
| `adapters.py` | support for hardware this library was not written for |
| `montage.py` | where the electrodes are, as data rather than as a sentence |
| `placement.py` | how a device was placed, and what that implies |
| `timebase.py` | putting events and samples on one ruler |
| `controller.py` | game controllers, as a second input stream on that ruler |
| `audio.py` | spoken cues |
| `headmap.py` | scalp coordinates, for drawing |

## Capabilities are the device's answer, never this library's assumption

A device declares what it is over the protocol, and everything follows
from that declaration. The channel count, the converter's resolution and
reference, the rates it offers, whether it can detect lead-off, whether it
carries a model: all of it is asked and none of it is assumed.

```python
device.info.channels            # what the device said
device.can("model")             # a capability bit, not a version number
device.profile()                # controls composed from the above
```

A table of specific hardware in this library is a defect. Hardware that
needs one brings an adapter, which is a separate package, and whatever it
offers appears as `device.extra` on the devices it claims. See
`adapters.py`.

## The timeline is never silently repaired

Every sample carries the device's own index and the device time of its own
conversion. A forward step in that index is a loss with an exact count. A
step that is not forward is a break whose extent is not a number, and this
library says so rather than returning a count that would be a fiction.

A recording writes its gaps down. Nothing splices.

## The contract is tested, not assumed

`contract/conformance.json` is emitted by the firmware from the codec the
device itself runs. Every message this library decodes, it decodes to the
fields that file states, and every malformed one it refuses for the reason
that file states. An implementation that drifts from the device fails a
test rather than a bench session.

```
python3 tests/run_all.py
```

No device, no network, no pytest. That is the command, and there is no
other.

## The model

The device's encoder turns four seconds of signal into an embedding of 76
numbers. Its weights are not published: they reach a device only as a
signed image. The device sends its embeddings instead, so a head is
trained on a host and uploaded:

```python
from intomind.model import train_head

await device.start()
windows = await device.collect_embeddings(100)    # about seven minutes
blob = train_head([w.embedding for w in windows], labels, name="focus",
                  encoder_id=windows[0].encoder_id)
await device.upload_head(blob, slot=1, select=True)
await device.set_predictions(True)
```

A head needs more windows than the embedding has numbers. A head is
weights, never code. What it produces on the host is what it produces on
the device, because the device evaluates exactly those integers.

## Exports

```python
from intomind import export

export.export(label, "bdf")        # any name in export.FORMATS
export.export_npz(label)           # or by name, same six keys back
```

Six formats, one shape, and every one of them refuses a capture that does
not verify. `npz` and `bdf` return every sample to the count it was
recorded at, measured per file rather than assumed. `edf` is sixteen bits
and says so. `mat` is MATLAB v7.3, which needs `h5py` and says which line
to type when it is missing.

## Where captures live

`$INTOMIND_CAPTURES`, or `~/.local/share/intomind/captures`. An
application that keeps its own collection says so once:

```python
from intomind import provenance
provenance.use_captures_dir("~/my-study/captures")
```

Nothing is inferred from where a file happens to sit.

## License

Free software under the GNU Affero General Public License, version 3. We
also license it commercially, on fair terms shaped by your use case: write
to contact@intomind.com. See `LICENSING.md`, and `CONTRIBUTING.md` before a
first pull request.

IntoMind is a trademark of IntoMind, Inc. The license covers the code, not
the name.
