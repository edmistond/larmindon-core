# Evaluation plan — corpus, metrics, and the cross-project harness

Status: plan, not yet built. Owns the pieces shared by all three Larmindon
implementations (`larmindon-core`, the Tauri app, `larmindon-swift`). The Swift
app has capture-fidelity problems that don't apply here; those live in
`larmindon-swift/eval-plan.md`.

## What this is not

`testdata/check_regression.sh` is a **behaviour-identity gate**: it replays one
fixed WAV and byte-diffs the result. Everything below is **quality
measurement**, which is a different question with different failure modes — it
is statistical, it has confidence intervals, and it must never be turned into an
exact-match diff. Keep the two separate. Tier 0 below is the existing gate,
listed only so the tiers make sense as a set.

## 1. Corpus

### Get the small splits, not the big ones

The 23G LibriSpeech download is `train-clean-360`. There is no reason to touch
any train split — evaluation lives in the dev/test sets:

| split | size | notes |
| --- | --- | --- |
| `dev-clean` | 337M | 40 speakers, ~5.4h — **primary working set** |
| `dev-other` | 314M | harder acoustics; also the babble source |
| `test-clean` | 346M | hold out |
| `test-other` | 328M | hold out |

Pull `dev-clean` + `dev-other` (651M) and stop. `test-*` stays undownloaded
until there is a release to check, because tuning `vad_threshold_start/_end`,
`chunk_ms`, and the `agc_*` constants against a set is what stops it being a
held-out set. When test-* does get used, it gets used once per release and the
result gets recorded — not iterated against.

Room impulse responses: **openslr.org/26 `sim_rir_16k.zip` (178M)**, not the
1.3G `rirs_noises.zip` from openslr/28. Skip MUSAN entirely (11G); see §6.

LibriSpeech is CC BY 4.0, so derived clips are redistributable with attribution.
That does not make committing them to git a good idea.

### Cache layout

Nothing generated or downloaded goes in any repo. One shared cache, resolved as
`$LARMINDON_EVAL_DATA`, defaulting to `~/.cache/larmindon-eval/`:

```
$LARMINDON_EVAL_DATA/
  corpora/librispeech/{dev-clean,dev-other}/...   # as extracted
  corpora/rirs/sim_rir_16k/...
  corpora/noise/<self-recorded>/...
  derived/<recipe-hash>/...                       # generated panels & streams
  runs/<run-id>/emissions.jsonl
```

`derived/` is keyed by a hash of the recipe TOML plus the seed, so a recipe
change produces a new directory instead of silently mutating an existing one and
a stale cache can never masquerade as a fresh generation. `derived/` is
disposable: deleting it must cost only regeneration time.

The repos commit **recipes and manifests only**.

## 2. Manifests

Panels are frozen by committed ID lists, not by "sample 150 at runtime". A
panel that resamples is a panel whose numbers can't be compared across commits.

```toml
# eval/panels/tier1-dev-clean.toml
[panel]
id         = "tier1-dev-clean"
corpus     = "librispeech/dev-clean"
normalizer = "en-v1"
# sha256 over the concatenated normalized refs, in listed order.
# Mismatch means the wrong corpus was extracted, before any run happens.
refs_sha256 = "..."

[[utterance]]
id       = "1272-128104-0000"
speaker  = "1272"
sex      = "M"
dur_ms   = 5855
rate_wps = 2.9      # words / sec, from the ref — the fast-talker stratum
```

Paths and reference text resolve from the corpus at load time; duplicating refs
into the manifest just creates two things to keep in sync. Everything needed to
*stratify* is in the manifest, so panel composition is reviewable in a diff.

### Tier 1 composition

`dev-clean` has 40 speakers, 20M/20F. Take 3–5 utterances per speaker, chosen to
spread across three strata:

- **duration**: `<4s` / `4–12s` / `>12s`. The short bucket is where endpointing
  over-truncates and where a whole utterance can be swallowed by VAD hangover;
  it is the most informative bucket and the easiest to under-sample by accident.
- **speaking rate**: bottom/middle/top tercile of words-per-second. Fast talkers
  are what break fixed `chunk_ms` boundaries.
- **sex**: balanced, because it's free to balance and expensive to discover
  later that it wasn't.

Target ~150 utterances / ~20 min. Mirror the whole construction on `dev-other`
as `tier1-dev-other`; the clean↔other gap is a diagnostic in its own right and
often moves before either absolute number does.

## 3. Tier 2 — long-form streams (the tier that finds real bugs)

LibriSpeech utterances average ~7s. That barely touches `vad.rs`, doesn't
exercise `asr/accumulator.rs` at all, and can't produce a reconnect. Tier 2
fixes this by *constructing* continuous audio with a known timeline.

```toml
# eval/streams/tier2-single-1272.toml
[stream]
id           = "tier2-single-1272"
seed         = 0x5EED0001
corpus       = "librispeech/dev-clean"
speakers     = ["1272"]
target_dur_s = 480

[gaps]
dist   = "lognormal"   # inter-utterance silence
p50_ms = 900
p95_ms = 2400
min_ms = 300

[long_pauses]
every_n = 12
ms      = [5000, 10000]
```

The generator writes a WAV **and a sidecar timeline JSON** giving the exact
sample offset of every utterance boundary and its reference text. That ground
truth is what makes the following measurable, none of which WER can see:

- **boundary error** — emitted segment boundaries vs known ones, as median
  absolute offset plus a boundary F1 at a fixed tolerance (250ms).
- **boundary deletions** — words lost specifically in the ±500ms window around a
  known boundary, reported separately from the global deletion count. This is
  the VAD-hangover signature and it disappears into the noise if pooled.
- **join duplication** — text emitted twice across a segment seam. Direct test
  of `accumulator.rs`, currently untested outside the golden diff.
- **latency drift** — regress emission latency against stream position. A
  positive slope means the pipeline is falling behind and will eventually drop
  buffers; a flat line with a high intercept is a different bug entirely.

Also build `tier2-two-speaker-*` by interleaving two speakers' utterances. That
gives diarization a fixture with **real** speaker ground truth, retiring the
caveat in `testdata/README.md` about the SAPI voices testing plumbing rather
than accuracy. Keep the SAPI fixtures — they stay valuable precisely because
they're synthetic and reproducible — but stop treating them as an accuracy
signal once this exists.

## 4. Text normalization

LibriSpeech references are uppercase, unpunctuated, with numbers spelled out.
Every backend here emits mixed case with punctuation and digits. Scoring raw
gives ~30% WER composed almost entirely of formatting, so:

- Implement one normalizer, ID it (`en-v1`), and record the ID in every run
  header. Whisper's English normalizer semantics are the sane default: lowercase,
  strip punctuation, expand contractions consistently, spell out numbers, drop
  fillers.
- **Treat a normalizer change as a baseline-regeneration event**, on the same
  footing as the model changes already listed in `testdata/README.md`. A silent
  normalizer edit shifts every number in every panel at once and looks exactly
  like a model regression.
- Normalize both sides identically, and store normalized refs' hash in the
  manifest (§2) so a corpus/normalizer mismatch fails loudly at load.

## 5. Metrics

WER alone under-describes a live captioner. Report, per panel and per condition:

| metric | why |
| --- | --- |
| WER, split into I/D/S | insertions spiking = hallucination under noise; deletions spiking = VAD too aggressive. Same WER, opposite fixes. |
| first-token latency, p50/p95 | perceived responsiveness |
| finalization latency, p50/p95 | how long text stays provisional |
| **revision rate** | edit distance summed across consecutive partial hypotheses ÷ final word count. Flicker is often more damaging to a live caption than 1% WER, and nothing else measures it. |
| RTF + `CaptureBuffer::dropped_samples` | already tracked; just surface it. Any nonzero drop invalidates the run's other numbers. |
| boundary F1 / boundary deletions / join duplication | Tier 2 only (§3) |
| DER or speaker-label purity | Tier 2 two-speaker only |

**Report deltas against a committed baseline with a paired bootstrap CI over
utterances.** On 150 utterances a 0.5% WER move is noise. Without the interval
this harness will generate weeks of chasing nothing, which is a worse outcome
than not having it.

## 6. Degradation matrix

Sweep, don't spot-check. Every condition runs at SNR ∈ {20, 15, 10, 5, 0} dB so
the output is a curve; a regression is a shift in the curve, especially at the
knee, which is far more robust than any single point.

Ordered by return on effort:

1. **Transport fuzzing — needs no corpus at all.** Variable chunk sizes, jitter
   in delivery timing, 50–500ms stalls, dropped buffers, device swap mid-stream,
   wrong sample rate. `replay_wav`'s feeder is already 90% of this; it needs a
   fault-injection layer between the pacer and `push_sample`. Highest yield per
   hour of work in the whole document.
2. **Level / AGC stress.** Scale to −6 / −20 / −35 / −50 dBFS, add a slow ramp
   (speaker turning away from the mic), add clipping at 0 dBFS. A separate axis
   from SNR: AGC pumping shows up as boundary deletions and will never appear in
   a noise sweep. `agc.rs` currently has no dedicated quality signal.
3. **Babble.** The most realistic degradation for this app and free — sum 4 / 8 /
   16 speakers drawn from `dev-other` into `dev-clean` targets. Never let a
   speaker appear in their own noise.
4. **Reverb.** Convolve with `sim_rir_16k`, bucketed short/medium/long RT60.
   This is the "laptop mic across the room" case.
5. **Environmental noise.** Skip MUSAN. Record five minutes each of the actual
   deployment environments — the office fan, the car, the cafe. MUSAN's only
   real advantage is comparability with published papers, which is not a goal
   here; self-recorded noise is strictly more relevant.
6. **Channel simulation.** Band-limit to 300–3400 Hz (Bluetooth HFP), 8k↔16k
   resample round-trip, Opus round-trip, DC offset. Cheap, and it catches
   `rubato` plumbing bugs that nothing else will.

Two implementation details that are easy to get wrong and expensive to discover:

- **Compute SNR over speech-active regions only.** Including leading and
  trailing silence puts the effective SNR off by several dB and makes it
  inconsistent between short and long clips, so the sweep stops being a sweep.
  Gate on the Tier 2 timeline where available, energy otherwise.
- **Seed per (utterance_id, condition, param)**, never from a global RNG.
  Regeneration must be bit-identical on another machine or the derived cache is
  worthless.

Generate offline into `derived/`. Prep in Python (numpy/soundfile — much faster
to write, and it isn't shipped code); runner and scorer in Rust so all three
projects share them.

## 7. Per-backend allocation

The backends have different weak points, so running the full matrix against all
of them wastes time and money on questions already answered.

**Nemotron (local, free, deterministic-ish)** — carries the full matrix. Local
inference means the panel can run at `--speed` > 1 for throughput. This is where
DSP work gets evaluated.

**Soniox (cloud, $0.12/hr, real-time only)** — already strong in high noise, so
the SNR sweep buys little. Its actual failure modes are flaky network and
overlapping speakers, which splits cleanly:

- **Flaky network costs nothing to test.** `Tests/CoreTests/FakeWebSocketTransport.swift`
  on the Swift side already establishes the seam; the Rust `soniox/client.rs`
  wants the equivalent. Mid-utterance close, slow response, out-of-order finals,
  and reconnect-mid-segment are all unit tests with no audio and no API spend.
  Do this first and do it thoroughly — it's the documented weak spot and it's
  free.
- **Overlap needs real audio and real API time.** So the paid budget goes almost
  entirely here: Tier 2 two-speaker streams with deliberately overlapped turns
  (negative gaps in the recipe), which the generator can produce exactly because
  it controls the timeline.
- Everything else Soniox gets is one 5-minute Tier 2 stream as a "did token
  handling regress" check, run at 1x.

Use a **separate Soniox API key for evaluation** so test spend is visible
independently of real use. At $0.12/hr a 5-minute check is under two cents and a
full 20-minute panel is four; the risk isn't a single run, it's an unattended
loop, so the runner should refuse to start a cloud panel over a configured
minutes budget without an explicit `--yes-spend` flag.

## 8. The cross-project contract

The thing that makes three codebases comparable is **not shared code** — it's a
shared emission log and one scorer. `replay_wav` already prints
`=== EMISSIONS ===` / `=== SEGMENTS ===`; formalize that as `--emit-jsonl`.

First line is a header, remaining lines are events:

```json
{"schema":"larmindon-emit/1","run":{"provider":"nemotron","model":"...","panel":"tier1-dev-clean","condition":"babble8@10db","normalizer":"en-v1","speed":1.0,"seed":123,"git":"abc1234","host":"macos-15.1-m3"}}
{"t_ms":1240,"kind":"partial","seg":3,"text":"mister quilter is the"}
{"t_ms":1610,"kind":"final","seg":3,"text":"mister quilter is the apostle","audio_ms":[1020,1580],"speaker":"A"}
{"t_ms":9000,"kind":"error","message":"websocket closed"}
```

**`t_ms` and `audio_ms` are different clocks and both are required.** `t_ms` is
wall-clock since the first sample was fed; `audio_ms` is position within the
stream. Latency is `t_ms − audio_ms[1]`, and it is unrecoverable if only one is
logged — which is the single most likely way to have to re-run every panel.

Latency metrics are only meaningful at `--speed 1`; the header records `speed`
so the scorer can refuse to report them otherwise.

Consumers:

- **larmindon-core** — `replay_wav --emit-jsonl`
- **Tauri app** — same core, free
- **larmindon-swift** — see that repo's `eval-plan.md`; harder, and the reason it
  gets its own document

Then one `score` binary takes (manifest | timeline, emissions.jsonl) → metrics
JSON, and Apple SpeechAnalyzer, Nemotron, and Soniox become directly comparable
on byte-identical audio. That comparison is the actual payoff for doing this
across three implementations and it should be designed toward from the start,
not retrofitted.

## 9. Scheduling

| when | what | runtime |
| --- | --- | --- |
| per commit | Tier 0 gate + 20-utterance Tier 1 slice at max speed | ~1 min |
| per commit | transport-fuzz and Soniox network-fault unit tests | seconds |
| nightly | full Tier 1 clean + other, full degradation matrix, Nemotron | hours |
| nightly | one 5-min Soniox Tier 2 stream at 1x | 5 min, ~$0.01 |
| manual / pre-release | Tier 2 suite, two-speaker overlap panels, `test-*` | — |

A 20-minute panel at 1x is 20 minutes *per condition*; the matrix is a nightly
job by construction, and anything claiming to run it per-commit is running it at
a speed where the latency numbers are meaningless.

## 10. Build order

1. `--emit-jsonl` on `replay_wav` + the `score` binary + `en-v1` normalizer.
   Nothing else is useful until output is machine-readable.
2. Fault injection in the `replay_wav` feeder (§6.1) and Soniox WS fault tests
   (§7). Free, no corpus, highest yield.
3. Corpus fetch script + Tier 1 manifest generator. First real WER number.
4. Tier 2 stream generator + timeline sidecar + the boundary/join metrics.
5. Degradation generators, in the §6 order.
6. Paired bootstrap CI in the scorer — before, not after, anyone starts making
   decisions from the numbers.

Personal-usage set, worth starting in parallel with all of the above: 20 minutes
of real recorded usage with hand-typed ground truth will say more about whether
the app works than five hours of LibriSpeech. LibriSpeech is a *regression
substrate* — stable, clean, and unlike the target domain in every way that
matters. Don't confuse a good regression substrate with a validity argument.
