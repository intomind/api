# Contributing

Issues, questions, and pull requests are welcome.

## Before your first pull request

Sign the contributor license agreement. It is short, it is in `CLA.md`,
and it is signed once rather than per contribution. Comment on your pull
request with:

```
I have read CLA.md and I agree to it.
```

from the account that owns the contribution. A maintainer records the
agreement against your account and later pull requests need nothing
further.

You keep the copyright in your work. The agreement gives IntoMind the
right to use your contribution under both the Affero license and the
commercial license, which is what makes it possible to offer either one.
Without it a contribution could only ever go out under one of them, and
the project would have to refuse the contribution or drop the commercial
license.

If you are contributing for an employer, get whoever owns your work to
agree as well, and say so on the pull request.

## What makes a change easy to accept

- One change per pull request, with the reason in the message rather than
  in a comment on the diff.
- A test that fails before your change and passes after it. A change to
  behavior without a test that shows the behavior is hard to review and
  hard to keep.
- The house style: American English, plain sentences, no abbreviation a
  reader has to look up.
- Nothing that names hardware this library does not support. The device
  is the authority on what the device is, and a table of specific
  hardware here is a defect.

## What gets refused

- Code you did not write, or code whose license is not compatible.
- A dependency added for something small. Every dependency is a thing
  that has to keep working for as long as this library does.
- Anything that makes a recording less honest: a silent gap, an invented
  sample, a statistic that was not measured.

## Security

Report anything security sensitive to contact@intomind.com rather than in
a public issue.

## How this code is written

Four things are true of this library, and a change that makes any of them
false is a change to refuse.

1. **The device is the authority on what the device is.** Channel count,
   rates, gains, capabilities, what it is set to, what processing it runs:
   all of it is asked over the protocol and none of it is assumed. A table of
   specific hardware in this library is a bug, because hardware this library
   does not know about is the whole point.
2. **This library declares nothing about the world it is installed in.** It
   does not know where captures go, which installation it is part of, or
   which software is recording. An application declares those
   (`use_captures_dir`, `use_site`, `use_repo`). Anything undeclared is
   recorded as unknown, never guessed.
3. **A capture is the record, and it is immutable once written.** Checksums
   are written last, so a run that dies mid-save is never mistaken for a
   complete capture. Nothing rewrites a manifest after the fact.
4. **Real numbers only, from the code that measured them.** Never invent a
   statistic.

The dependency runs one way. Applications import this library; this library
imports nothing of theirs and names none of them. A test that reads a
particular device's firmware is a conformance test for that device and
belongs with that device, not here: this library is held to the published
contract vectors instead.
