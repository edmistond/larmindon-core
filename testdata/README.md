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

## A trap worth knowing

`replay_wav` runs a warm-up session with its audio feeder gated off, purely to
populate the engine's model cache. Without it, a cold load of the ~2.4 GB
encoder takes long enough to overrun the 10-second capture buffer, silently
dropping the head of the file and producing a *different transcript that looks
exactly like a regression*. Do not remove that pass.
