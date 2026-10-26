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
await device.upload_head(blob, slot=2, select=True)   # slot 1 ships with the open sex head
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
to type when it is missing. The electrode-off flags recorded with each
sample are kept in `npz` and `mat` only.

In `bdf` and `edf`, every sample is written at the time it was recorded.
Where samples were lost, the file holds one placeholder for each of them,
under a `BAD_ACQ_SKIP` annotation that covers the hole. MNE reads that label
as its own: it filters each side of a hole separately, rejects an epoch that
overlaps one, and returns the hole as NaN when asked. A reader that does not
know the label shows a hole as a flat line at the bottom of the signal's
range. These files are whole records of about a second, and the last record
is completed after the last sample with the same placeholder under the same
label, as MNE's own writer completes it, so no recorded sample is cut.
`tail_filled` says how many placeholders that took.

The placeholder is the lowest number the file can store. NaN is not
possible, because these formats store whole numbers. Zero is possible, but
each signal's range would have to stretch to reach it, and most recordings
sit thousands of microvolts from zero, so every recorded sample in a 16-bit
file would be rounded more coarsely. The lowest number costs nothing, and
no recorded sample can be it, because each range leaves a margin below the
signal's lowest sample. A build that prefers zero sets
`edf.GAP_FILL_UV = 0.0` and takes on that cost.

A recording whose sample count starts again within its first second begins
with samples from before the recording started. `bdf` and `edf` start at
the reset, and `samples_before_restart` says how many samples were left
out. A count that steps back later is a break whose length is not known,
and they refuse it rather than invent one.

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
