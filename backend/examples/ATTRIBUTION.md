# Attribution: the bundled satellite recording

`intelsat37e_qpsk_excerpt.sigmf` is an excerpt of this dataset:

- **Title:** Test of gr4-packet-modem through an Intelsat 37e C-band transponder
- **Author:** Daniel Estévez
- **Source:** Zenodo, published 2024-09-01,
  [https://doi.org/10.5281/zenodo.13371136](https://doi.org/10.5281/zenodo.13371136)
- **License:** Creative Commons Attribution 4.0 International,
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (the Zenodo record
  and the recording's SigMF metadata both state it)
- **Original file:** `gr4-packet-modem-intelsat37e-test.sigmf-data`: 596,633
  complex float32 samples (`cf32_le`) at 500 samples per second, center
  frequency 3,671,055,830 Hz, recorded on 2024-08-14 with a USRP B200mini
  connected to the L-band IF of a teleport in Usingen (19 m antenna).

Using the recording here does not mean that the author endorses SpectrumSense.

## What was changed

- Kept samples 204,000 to 234,719 of the original (counting from 0): 30,720
  samples, or 61.44 seconds starting 408 seconds into the recording. The sample
  values are unchanged, byte for byte.
- Wrote new SigMF metadata (version 1.2.0): a description of the excerpt, the
  capture time of its first sample (2024-08-14T20:52:56Z), a SHA-512 of its
  data, and ten annotations that mark the packets. The annotations are
  SpectrumSense's estimates from signal power (a 64-sample moving average
  compared with a threshold halfway between the quiet and loud levels), not
  the author's labels.
- Packed the metadata and samples as an uncompressed SigMF archive.

## How it was made

On 2026-09-25, from the repository folder:

```bash
.venv/bin/python tools/make_real_example.py
```

The script downloads the record with `tools/fetch_real_iq.py`, which checks the
data file against the MD5 checksum that Zenodo publishes and the SHA-512 in the
original SigMF metadata, and then writes
`examples/intelsat37e_qpsk_excerpt.sigmf`. The archive's dates are fixed, so a
rebuild gives the same bytes. The file's SHA-256 is:

```
1963116ecbd2d25440178477056e7f531207f738e5fdbe93a3404b53181c0cf9
```

## What it contains

The modulation is not in the metadata. The author's
[blog post](https://destevez.net/2024/09/testing-gr4-packet-modem-through-a-geo-transponder/)
and the [gr4-packet-modem](https://github.com/daniestevez/gr4-packet-modem)
source code describe it: uncoded QPSK at 125 symbols per second (4 samples per
symbol at 500 samples per second) with root-raised-cosine pulses, and each
packet opens with a 64-symbol BPSK syncword (256 samples). Measured on this
excerpt: ten complete packets of about 2,200 samples, one starting about every
3,090 samples, about 6 dB stronger than the gaps between them. The excerpt
starts and ends in a gap.
