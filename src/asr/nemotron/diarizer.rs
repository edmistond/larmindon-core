//! Sortformer v3 (Nemotron-3 Diarization) alongside Nemotron.
//!
//! The diarizer is fed exactly the live slices the ASR consumes, so its
//! absolute segment offsets and the ASR's consumed-sample count are one clock.
//! Two rules keep them aligned: replayed chunks are never fed, and a decoder
//! reset never resets the diarizer (that would renumber the speakers). Its
//! state is reset only at the start of a session.
//!
//! At this stage segments are only logged, to check the plumbing and the
//! inference cost before any speaker reaches the transcript.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use parakeet_rs::sortformer::{
    DiarizationConfig, Sortformer, SpeakerSegment, StreamingProfile, SUBSAMPLING,
};
use parakeet_rs::ExecutionConfig;

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

pub(super) struct Diarizer {
    model: Sortformer,
    path: PathBuf,
    profile: StreamingProfile,
    samples_fed: u64,
    /// Set by `flush`, after which every fed sample has been diarized.
    flushed: bool,
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
            flushed: false,
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
        self.flushed = false;
        self.speakers_seen.clear();
        self.segment_count = 0;
        self.inference_ms = 0;
        Ok(())
    }

    pub fn feed(&mut self, samples: &[f32]) -> Result<Step, String> {
        let before = self.windows_completed();
        let start = Instant::now();
        let result = self.model.feed(samples);
        // Counted even on error: Sortformer buffered the audio before failing,
        // and the clock must keep matching the ASR's.
        self.samples_fed += samples.len() as u64;
        let segments = result.map_err(|e| format!("Diarizer feed failed: {e}"))?;
        Ok(self.record(segments, self.windows_completed() - before, start))
    }

    /// Runs whatever is still buffered, zero-padded. End of session only.
    pub fn flush(&mut self) -> Result<Step, String> {
        // Sortformer runs one padded window over anything still buffered.
        let windows = u64::from(self.samples_fed > self.horizon());
        let start = Instant::now();
        let segments = self
            .model
            .flush()
            .map_err(|e| format!("Diarizer flush failed: {e}"))?;
        self.flushed = true;
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
        if self.flushed {
            self.samples_fed
        } else {
            self.windows_completed() * self.stride_samples()
        }
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

    /// Windows `feed` has run so far. It runs one once a full
    /// `chunk_len + right_context` window is buffered, then strides by
    /// `chunk_len`. Mirrors Sortformer's private `elapsed_samples`.
    fn windows_completed(&self) -> u64 {
        windows_completed(
            self.samples_fed,
            self.stride_samples(),
            (self.profile.chunk_len + self.profile.right_context) as u64 * FRAME_SAMPLES,
        )
    }
}

fn windows_completed(samples_fed: u64, stride: u64, window: u64) -> u64 {
    if samples_fed < window || stride == 0 {
        0
    } else {
        (samples_fed - window) / stride + 1
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
        assert_eq!(windows_completed(0, stride, window), 0);
        assert_eq!(windows_completed(window - 1, stride, window), 0);
        assert_eq!(windows_completed(window, stride, window), 1);
        assert_eq!(windows_completed(window + stride - 1, stride, window), 1);
        assert_eq!(windows_completed(window + stride, stride, window), 2);
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
