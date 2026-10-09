//! Sortformer v3 (Nemotron-3 Diarization) alongside Nemotron.
//!
//! The diarizer is fed exactly the live slices the ASR consumes, so its
//! absolute segment offsets and the ASR's consumed-sample count are one clock.
//! Two rules keep them aligned: replayed chunks are never fed, and a decoder
//! reset never resets the diarizer (that would renumber the speakers). Its
//! state is reset only at the start of a session.
//!
//! Segments go to [`super::attribution`], which labels Nemotron's text with
//! them once [`Diarizer::horizon`] has passed it.
//!
//! The model runs on its own thread behind [`DiarizerThread`]. Inline on the
//! processing thread, a diarizer slower than real time (a debug build, a busy
//! machine) stalled Nemotron until the capture buffer dropped seconds of
//! audio. Off-thread, the same slowness only delays labels, and
//! [`DiarizerThread::effective_horizon`] bounds that delay.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};

use parakeet_rs::sortformer::{
    DiarizationConfig, Sortformer, SpeakerSegment, StreamingProfile, SUBSAMPLING,
};
use parakeet_rs::ExecutionConfig;

use super::attribution::Span;

/// Samples per 80 ms model frame: `SUBSAMPLING` mel frames of 10 ms (160
/// samples at 16 kHz) each.
const FRAME_SAMPLES: u64 = (SUBSAMPLING * 160) as u64;

/// The live preset named by `diarization_latency`. Validated in settings, so
/// an unknown name is a programming error; it falls back to the default.
pub(super) fn profile_for(latency: &str) -> StreamingProfile {
    match latency {
        "ultra" => StreamingProfile::ultra_low_latency(),
        "very_low" => StreamingProfile::very_low_latency(),
        _ => StreamingProfile::low_latency(),
    }
}

/// What one `feed` or `flush` produced.
pub(super) struct Step {
    pub segments: Vec<SpeakerSegment>,
    pub windows: u64,
    pub inference_ms: i64,
}

impl Step {
    pub fn spans(&self) -> impl Iterator<Item = Span> + '_ {
        self.segments.iter().map(|s| Span {
            start: s.start,
            end: s.end,
            speaker: s.speaker_id,
        })
    }
}

pub(super) struct Diarizer {
    model: Sortformer,
    path: PathBuf,
    profile: StreamingProfile,
    samples_fed: u64,
    /// Mirrors Sortformer's private audio buffer and `elapsed_samples`: fed
    /// samples not yet diarized, and samples diarized.
    buffered: u64,
    diarized: u64,
    speakers_seen: BTreeSet<usize>,
    segment_count: u64,
    inference_ms: i64,
}

impl Diarizer {
    pub fn load(path: &Path, config: ExecutionConfig) -> Result<Self, String> {
        let model = Sortformer::with_config(path, Some(config), DiarizationConfig::default())
            .map_err(|e| format!("Failed to load diarization model {}: {e}", path.display()))?;
        let profile = model.profile();
        Ok(Self {
            model,
            path: path.to_path_buf(),
            profile,
            samples_fed: 0,
            buffered: 0,
            diarized: 0,
            speakers_seen: BTreeSet::new(),
            segment_count: 0,
            inference_ms: 0,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Starts a session: applies the preset and clears all streaming state,
    /// so speaker ids start again from 0.
    pub fn begin(&mut self, profile: StreamingProfile) -> Result<(), String> {
        // set_profile resets state itself; reset explicitly as well so a
        // session never inherits the last one's clock even if that changes.
        self.model
            .set_profile(profile)
            .map_err(|e| format!("Invalid diarization profile: {e}"))?;
        self.model.reset_state();
        self.profile = profile;
        self.samples_fed = 0;
        self.buffered = 0;
        self.diarized = 0;
        self.speakers_seen.clear();
        self.segment_count = 0;
        self.inference_ms = 0;
        Ok(())
    }

    pub fn feed(&mut self, samples: &[f32]) -> Result<Step, String> {
        let start = Instant::now();
        let result = self.model.feed(samples);
        // Counted even on error: Sortformer buffered the audio before failing,
        // and the clock must keep matching the ASR's.
        self.samples_fed += samples.len() as u64;
        let (stride, window) = (self.stride_samples(), self.window_samples());
        let windows = advance(
            &mut self.buffered,
            &mut self.diarized,
            samples.len() as u64,
            stride,
            window,
        );
        let segments = result.map_err(|e| format!("Diarizer feed failed: {e}"))?;
        Ok(self.record(segments, windows, start))
    }

    /// Runs one zero-padded window over whatever is still buffered, so every
    /// sample fed so far is diarized. Streaming state and speaker ids carry
    /// on: feeding may continue afterwards on the same clock.
    pub fn flush(&mut self) -> Result<Step, String> {
        let windows = u64::from(self.buffered > 0);
        let start = Instant::now();
        let result = self.model.flush();
        // Sortformer takes its buffer before it can fail.
        self.diarized += self.buffered;
        self.buffered = 0;
        let segments = result.map_err(|e| format!("Diarizer flush failed: {e}"))?;
        Ok(self.record(segments, windows, start))
    }

    fn record(&mut self, segments: Vec<SpeakerSegment>, windows: u64, start: Instant) -> Step {
        let inference_ms = start.elapsed().as_millis() as i64;
        self.inference_ms += inference_ms;
        self.segment_count += segments.len() as u64;
        self.speakers_seen
            .extend(segments.iter().map(|s| s.speaker_id));
        Step {
            segments,
            windows,
            inference_ms,
        }
    }

    pub fn samples_fed(&self) -> u64 {
        self.samples_fed
    }

    pub fn speakers_seen(&self) -> usize {
        self.speakers_seen.len()
    }

    /// Samples up to which the diarizer has emitted every segment it will
    /// emit: labels for audio before this point are settled.
    pub fn horizon(&self) -> u64 {
        self.diarized
    }

    pub fn summary(&self) -> String {
        format!(
            "{} segment(s), {} speaker(s), {} ms inference over {:.1} s of audio",
            self.segment_count,
            self.speakers_seen.len(),
            self.inference_ms,
            self.samples_fed as f64 / 16_000.0
        )
    }

    fn stride_samples(&self) -> u64 {
        self.profile.chunk_len as u64 * FRAME_SAMPLES
    }

    fn window_samples(&self) -> u64 {
        (self.profile.chunk_len + self.profile.right_context) as u64 * FRAME_SAMPLES
    }
}

/// Mirrors `Sortformer::feed` for `fed` new samples: it runs a window whenever
/// a full `chunk_len + right_context` window is buffered, then drops `stride`
/// (`chunk_len`) samples and keeps the right context. Returns windows run.
fn advance(buffered: &mut u64, diarized: &mut u64, fed: u64, stride: u64, window: u64) -> u64 {
    *buffered += fed;
    let mut windows = 0;
    while stride > 0 && *buffered >= window {
        *buffered -= stride;
        *diarized += stride;
        windows += 1;
    }
    windows
}

/// Audio the labels may trail the ASR by before text is finalized with the
/// best label available, so a diarizer that cannot keep up costs accuracy
/// rather than leaving captions provisional indefinitely.
const MAX_LABEL_LAG_SAMPLES: u64 = 10 * 16_000;

enum Request {
    Begin(StreamingProfile),
    Feed(Vec<f32>),
    Flush(&'static str),
}

enum Reply {
    Begun(Result<(), String>),
    Stepped(Stepped),
}

/// A `feed` or `flush` the worker has finished, with the counters the
/// diagnostics row reads as they stood right after it.
pub(super) struct Stepped {
    /// `"feed"`, or the name the flush was requested under.
    pub step: &'static str,
    pub result: Result<Step, String>,
    pub samples_fed: u64,
    pub speakers_seen: usize,
    /// Requests this reply answers: consecutive feeds are coalesced.
    requests: usize,
    horizon: u64,
    summary: String,
}

/// A [`Diarizer`] on its own thread. Calls never block on inference except
/// [`DiarizerThread::begin`] and [`DiarizerThread::wait`].
pub(super) struct DiarizerThread {
    requests: Sender<Request>,
    replies: Receiver<Reply>,
    path: PathBuf,
    /// Samples sent this session: the ASR's clock.
    sent: u64,
    /// The worker's horizon as of its latest reply.
    horizon: u64,
    in_flight: usize,
    summary: String,
    stopped: bool,
}

impl DiarizerThread {
    pub fn spawn(diarizer: Diarizer) -> Result<Self, String> {
        let path = diarizer.path().to_path_buf();
        let (requests, worker_requests) = mpsc::channel();
        let (worker_replies, replies) = mpsc::channel();
        thread::Builder::new()
            .name("diarizer".to_string())
            .spawn(move || run_worker(diarizer, worker_requests, worker_replies))
            .map_err(|e| format!("Failed to start diarization thread: {e}"))?;
        Ok(Self {
            requests,
            replies,
            path,
            sent: 0,
            horizon: 0,
            in_flight: 0,
            summary: String::new(),
            stopped: false,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// [`Diarizer::begin`] on the worker. Blocks for the answer, discarding
    /// anything the previous session left in flight.
    pub fn begin(&mut self, profile: StreamingProfile) -> Result<(), String> {
        self.send(Request::Begin(profile))?;
        loop {
            match self.replies.recv() {
                Ok(Reply::Begun(result)) => break result?,
                Ok(Reply::Stepped(_)) => {}
                Err(_) => return Err(self.stop()),
            }
        }
        self.sent = 0;
        self.horizon = 0;
        self.in_flight = 0;
        self.summary.clear();
        Ok(())
    }

    pub fn feed(&mut self, samples: &[f32]) -> Result<(), String> {
        self.sent += samples.len() as u64;
        self.send(Request::Feed(samples.to_vec()))?;
        self.in_flight += 1;
        Ok(())
    }

    pub fn flush(&mut self, step: &'static str) -> Result<(), String> {
        self.send(Request::Flush(step))?;
        self.in_flight += 1;
        Ok(())
    }

    /// Steps finished since the last call. Never blocks.
    pub fn poll(&mut self) -> Result<Vec<Stepped>, String> {
        let mut done = Vec::new();
        loop {
            match self.replies.try_recv() {
                Ok(reply) => self.accept(reply, &mut done),
                Err(TryRecvError::Empty) => return Ok(done),
                Err(TryRecvError::Disconnected) => return Err(self.stop()),
            }
        }
    }

    /// Like [`DiarizerThread::poll`], then waits up to `timeout` for every
    /// request sent so far. For session end, where a short wait beats
    /// finalizing text without a label.
    pub fn wait(&mut self, timeout: Duration) -> Result<Vec<Stepped>, String> {
        let mut done = self.poll()?;
        let deadline = Instant::now() + timeout;
        while self.in_flight > 0 {
            let remaining = deadline.saturating_duration_since(Instant::now());
            match self.replies.recv_timeout(remaining) {
                Ok(reply) => self.accept(reply, &mut done),
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => return Err(self.stop()),
            }
        }
        Ok(done)
    }

    /// How far text may be finalized: the diarizer's horizon, or the point
    /// [`MAX_LABEL_LAG_SAMPLES`] behind the ASR if labels trail further than
    /// that. Unbounded once the worker has stopped.
    pub fn effective_horizon(&self) -> u64 {
        if self.stopped {
            return u64::MAX;
        }
        self.horizon
            .max(self.sent.saturating_sub(MAX_LABEL_LAG_SAMPLES))
    }

    pub fn summary(&self) -> &str {
        &self.summary
    }

    fn accept(&mut self, reply: Reply, done: &mut Vec<Stepped>) {
        if let Reply::Stepped(stepped) = reply {
            self.in_flight = self.in_flight.saturating_sub(stepped.requests);
            self.horizon = stepped.horizon;
            self.summary.clone_from(&stepped.summary);
            done.push(stepped);
        }
    }

    fn send(&mut self, request: Request) -> Result<(), String> {
        if self.requests.send(request).is_err() {
            return Err(self.stop());
        }
        Ok(())
    }

    fn stop(&mut self) -> String {
        self.stopped = true;
        "Diarization thread stopped".to_string()
    }
}

/// Owns the model. Consecutive queued feeds are coalesced into one call, so a
/// worker that falls behind catches up in fewer, larger passes; Sortformer
/// buffers internally, so the segments are the same either way. Exits when
/// the [`DiarizerThread`] is dropped.
fn run_worker(mut diarizer: Diarizer, requests: Receiver<Request>, replies: Sender<Reply>) {
    let mut next = None;
    loop {
        let request = match next.take() {
            Some(request) => request,
            None => match requests.recv() {
                Ok(request) => request,
                Err(_) => return,
            },
        };
        let reply = match request {
            Request::Begin(profile) => Reply::Begun(diarizer.begin(profile)),
            Request::Feed(mut audio) => {
                let mut coalesced = 1;
                loop {
                    match requests.try_recv() {
                        Ok(Request::Feed(more)) => {
                            audio.extend_from_slice(&more);
                            coalesced += 1;
                        }
                        Ok(other) => {
                            next = Some(other);
                            break;
                        }
                        Err(_) => break,
                    }
                }
                let result = diarizer.feed(&audio);
                Reply::Stepped(stepped(&diarizer, "feed", result, coalesced))
            }
            Request::Flush(step) => {
                let result = diarizer.flush();
                Reply::Stepped(stepped(&diarizer, step, result, 1))
            }
        };
        if replies.send(reply).is_err() {
            return;
        }
    }
}

fn stepped(
    diarizer: &Diarizer,
    step: &'static str,
    result: Result<Step, String>,
    requests: usize,
) -> Stepped {
    Stepped {
        step,
        result,
        samples_fed: diarizer.samples_fed(),
        speakers_seen: diarizer.speakers_seen(),
        requests,
        horizon: diarizer.horizon(),
        summary: diarizer.summary(),
    }
}

/// `speaker@start-end` per segment, space-separated, for the diagnostics row.
pub(super) fn format_segments(segments: &[SpeakerSegment]) -> String {
    let mut out = String::new();
    for (i, s) in segments.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        let _ = write!(out, "{}@{}-{}", s.speaker_id, s.start, s.end);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_is_80_ms() {
        assert_eq!(FRAME_SAMPLES, 1280);
    }

    #[test]
    fn no_window_runs_until_one_is_full() {
        // low_latency: 9 + 4 frames = 16 640 samples, stride 11 520.
        let (stride, window) = (9 * 1280, 13 * 1280);
        let (mut buffered, mut diarized) = (0, 0);
        assert_eq!(
            advance(&mut buffered, &mut diarized, window - 1, stride, window),
            0
        );
        assert_eq!(diarized, 0);
        assert_eq!(advance(&mut buffered, &mut diarized, 1, stride, window), 1);
        assert_eq!((buffered, diarized), (window - stride, stride));
        assert_eq!(
            advance(&mut buffered, &mut diarized, stride - 1, stride, window),
            0
        );
        assert_eq!(advance(&mut buffered, &mut diarized, 1, stride, window), 1);
        assert_eq!(diarized, 2 * stride);
    }

    #[test]
    fn one_feed_can_run_several_windows() {
        let (stride, window) = (9 * 1280, 13 * 1280);
        let (mut buffered, mut diarized) = (0, 0);
        assert_eq!(
            advance(
                &mut buffered,
                &mut diarized,
                window + 2 * stride,
                stride,
                window
            ),
            3
        );
        assert_eq!(diarized, 3 * stride);
        assert_eq!(buffered, window - stride);
    }

    #[test]
    fn presets_map_by_name() {
        assert_eq!(profile_for("low"), StreamingProfile::low_latency());
        assert_eq!(
            profile_for("very_low"),
            StreamingProfile::very_low_latency()
        );
        assert_eq!(profile_for("ultra"), StreamingProfile::ultra_low_latency());
    }

    #[test]
    fn segments_format_compactly() {
        let segs = [
            SpeakerSegment {
                start: 0,
                end: 1280,
                speaker_id: 0,
            },
            SpeakerSegment {
                start: 1280,
                end: 2560,
                speaker_id: 1,
            },
        ];
        assert_eq!(format_segments(&segs), "0@0-1280 1@1280-2560");
        assert_eq!(format_segments(&[]), "");
    }
}
