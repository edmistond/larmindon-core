# ASR regression fixtures

`processing_loop` has no unit-test seam — it owns a capture buffer, a model and
a thread. These fixtures are the only thing that catches a change in transcript
output, so run them after touching anything in `src/asr/`, `src/audio_engine.rs`,
`src/vad.rs` or `src/agc.rs`.

```sh
bash testdata/check_regression.sh
```

## What is here

| File | What it is |
| --- | --- |
| `baseline_speech.wav` | 33 s of 16 kHz mono speech, synthesized with Windows SAPI so it is reproducible and carries no third-party audio |
| `golden_baseline.txt` | Emissions + transcript for the ordinary path |
| `golden_reset.txt` | Same, with `--empty-reset 1` |
| `golden_reset_diag.txt` | Drain-invariant diagnostics rows for that run |
| `dump_diag.py` | Normalizes a diagnostics DB for diffing |
| `check_regression.sh` | Runs both fixtures and diffs all three goldens |
| `two_speaker.wav` | 75 s, 10 alternating turns across two voices, for the Soniox diarization check |
| `two_speaker_short.wav` | 29 s, first 4 turns of the same script, for fast iteration |
| `make_two_speaker.ps1` | Regenerates both. Windows-only; see below |

## The two-speaker fixtures

These exercise the Soniox path. No golden is attached to either — a cloud
model's output is not reproducible — so they are **not** part of
`check_regression.sh`.

Both use SSML voice switching, so the two speakers share one continuous stream
rather than being concatenated files. Turn boundaries then look like a
conversation to the diarizer rather than like hard cuts.

```sh
SONIOX_API_KEY=... cargo run --release --example replay_wav -- \
  testdata/two_speaker.wav --provider soniox --endpoint-detection off
```

Use `two_speaker_short.wav` while iterating: a run costs its own duration, since
anything faster than 1x stops resembling a live stream to a remote service.

**The generator is Windows-only (SAPI), but its output is committed.** That is
deliberate — the fixtures have to work on macOS and Linux, where there is no
SAPI, so a platform-specific generator must not make the testing
platform-specific. Regenerate only on Windows, and only when changing the script:

```sh
powershell -ExecutionPolicy Bypass -File testdata/make_two_speaker.ps1
powershell -ExecutionPolicy Bypass -File testdata/make_two_speaker.ps1 -Short
```

It fails loudly if the named voices are not installed. Because `two_speaker.wav`
is 2.3 MB, adding it needed jj's new-file guard raised
(`jj config set --repo snapshot.max-new-file-size 8388608`). That setting is
per-machine and lives outside the repo, so a fresh clone does not need it —
checking out already-tracked files is unaffected.

Read the `=== SEGMENTS ===` block: each line carries `speaker=`. What is being
checked is that a label arrives, stays stable within a turn, and changes at turn
boundaries.

> **This is a weak test of diarization *accuracy*.** Diarization models are
> trained on human speech, and these are synthetic voices. It is a strong test
> of the *plumbing* — that a speaker label flows wire → accumulator →
> `TranscriptUpdate` → UI. If speakers come back merged, suspect the fixture
> before the code, and confirm against real two-person audio.

Soniox's own documentation notes that endpoint detection finalizes earlier,
which raises WER, splits long speech into more endpoints **and reduces
diarization accuracy** — hence `--endpoint-detection off` above. Run it both
ways to see the difference.

The two transcript goldens are currently byte-identical: forcing the reset makes
the decoder replay buffered chunks, but those replays yield no new text. That is
the point of the second fixture — the replay path must not *add* or *corrupt*
output. Its distinctive coverage lives in `golden_reset_diag.txt`, which
contains the four `mid_speech_reset` rows and their replay statistics. Ordinary
speech almost never triggers that path (2 occurrences in 44 speech starts of
real recorded usage), hence the `--empty-reset 1` lever.

## Why this is reproducible

The transcript is invariant to drain boundaries: `asr_buffer` accumulates across
loop iterations and chunks at fixed offsets anchored at speech onset, so
scheduling jitter changes *when* work happens but not which samples land in
which chunk. A diff therefore means a real behaviour change, not flakiness.

`dump_diag.py` excludes every timing-dependent column for the same reason.

## Things that legitimately change the goldens

These are not regressions; regenerate if you change one deliberately.

- **A different Nemotron model or version.** The goldens were captured against
  the model at the `model_path` in `~/.config/larmindon/settings.json`.
- **DSP-relevant settings**, because the harness starts from your saved
  settings: `chunk_ms`, `punctuation_reset`, `empty_reset_threshold`,
  `vad_threshold_start` / `_end`, and all `agc_*` fields. The goldens were taken
  with AGC **enabled** and `chunk_ms: 560`.
- **Execution provider** (`--features directml` / `webgpu`) — different kernels,
  slightly different numerics.

Pass `--model <path>` to pin the model without editing settings.

## Regenerating

Only after confirming the change is intended:

```sh
cargo run --release --example replay_wav -- testdata/baseline_speech.wav \
  | sed -n '/=== EMISSIONS/,$p' > testdata/golden_baseline.txt

cargo run --release --example replay_wav -- testdata/baseline_speech.wav \
  --empty-reset 1 --diag /tmp/r.sqlite | sed -n '/=== EMISSIONS/,$p' \
  > testdata/golden_reset.txt
python testdata/dump_diag.py /tmp/r.sqlite > testdata/golden_reset_diag.txt
```

## Two traps worth knowing

`check_regression.sh` pins `--provider nemotron`. The harness otherwise starts
from your saved settings, so once `asr_provider` is set to anything else — which
it will be, the moment you use the app with a cloud backend — the gate silently
runs that backend and diffs its output against Nemotron goldens. The failure
looks like a catastrophic transcript regression (254 emissions against 88) and
is nothing of the sort. Do not remove that flag.


`replay_wav` runs a warm-up session with its audio feeder gated off, purely to
populate the engine's model cache. Without it, a cold load of the ~2.4 GB
encoder takes long enough to overrun the 10-second capture buffer, silently
dropping the head of the file and producing a *different transcript that looks
exactly like a regression*. Do not remove that pass.
