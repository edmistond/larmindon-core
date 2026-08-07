//! The [`AsrBackend`] implementation on top of [`SonioxClient`].
//!
//! This is a thin adapter: the socket lifetime lives in `client.rs` and the
//! token bookkeeping lives in `accumulator.rs`. What is here is the mapping
//! between the pipeline's two-part audio path and the socket's frames, plus the
//! rule that decides when text and when status wins a return slot.

use std::time::{Duration, Instant};

use super::accumulator::Accumulator;
use super::client::{Inbound, SonioxClient, SonioxConfig};
use super::protocol;
use crate::asr::{
    ActivitySource, AsrBackend, AsrCapabilities, AsrContext, AsrError, AudioGating, SegmentIds,
    SessionContext, TranscriptUpdate,
};
use crate::settings::Settings;

/// How long the meter stays lit after the last speech token.
///
/// Must exceed the service's idle cadence — it answers roughly once a second
/// even in total silence — or the indicator strobes. Much longer and it lies
/// about silence.
const SPEECH_HOLD: Duration = Duration::from_millis(1500);

/// Backstop for teardown. Slightly longer than the socket thread's own
/// end-of-stream watchdog so that watchdog is what normally fires; this only
/// covers a socket thread that has stopped answering altogether.
const FINISH_WAIT: Duration = Duration::from_millis(2500);

pub struct SonioxBackend {
    client: Option<SonioxClient>,
    accumulator: Accumulator,
    /// PCM staged by `push_audio`, flushed as one frame by `process`.
    staged: Vec<u8>,
    staged_samples: usize,
    last_speech_at: Option<Instant>,
    frames_sent: u64,
    /// Status that lost a return slot to text and is owed on the next call.
    pending_transient: Option<String>,
    pending_fatal: Option<String>,
}

impl Default for SonioxBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl SonioxBackend {
    pub fn new() -> Self {
        Self {
            client: None,
            accumulator: Accumulator::new(),
            staged: Vec::new(),
            staged_samples: 0,
            last_speech_at: None,
            frames_sent: 0,
            pending_transient: None,
            pending_fatal: None,
        }
    }

    /// Folds everything the socket thread has produced into the accumulator.
    fn drain_inbound(&mut self, ids: &SegmentIds) -> Vec<TranscriptUpdate> {
        let messages = match self.client.as_ref() {
            Some(client) => client.drain(),
            None => return Vec::new(),
        };
        self.fold(messages, ids)
    }

    /// The socket-independent half of draining, so the folding rules are
    /// testable without a live connection.
    fn fold(&mut self, messages: Vec<Inbound>, ids: &SegmentIds) -> Vec<TranscriptUpdate> {
        let mut updates = Vec::new();
        for message in messages {
            match message {
                Inbound::Response(response) => {
                    // Message arrival is not evidence of speech: the service
                    // answers during silence too. Only token content counts.
                    if response.tokens.iter().any(Accumulator::is_speech_token) {
                        self.last_speech_at = Some(Instant::now());
                    }
                    updates.extend(self.accumulator.ingest(&response, ids));
                }
                Inbound::Reconnecting(reason) => {
                    // The reconnected stream restarts its timeline from zero,
                    // so the open segment's current hypothesis is the best text
                    // it will ever have. Already-finalized transcript is
                    // untouched.
                    updates.extend(self.accumulator.finish(ids));
                    self.accumulator.reset();
                    self.pending_transient = Some(format!("Reconnecting to Soniox: {reason}"));
                }
                Inbound::Finished => {}
                Inbound::Fatal(message) => {
                    // Flush before reporting: the engine discards whatever
                    // `end_session` returns on the fatal path, so this is the
                    // last chance to keep the text that was already recognized.
                    updates.extend(self.accumulator.finish(ids));
                    self.accumulator.reset();
                    self.pending_fatal = Some(message);
                }
            }
        }
        updates
    }

    /// Drains, then decides between returning text and returning status.
    ///
    /// The trait returns `Result`, so one call cannot carry both. Text wins,
    /// and the status is owed on the next call — at most ~10 ms later, and it
    /// cannot be stranded, because the engine calls `poll` even on iterations
    /// where capture drained empty.
    fn results(&mut self, ids: &SegmentIds) -> Result<Vec<TranscriptUpdate>, AsrError> {
        let updates = self.drain_inbound(ids);
        self.deliver(updates, None)
    }

    /// `fallback` is used only when the socket thread reported nothing better.
    /// A send failure means the thread has gone, and the reason it went is
    /// almost always already queued and far more useful than "the channel is
    /// closed" — a rejected API key being the obvious case.
    fn deliver(
        &mut self,
        updates: Vec<TranscriptUpdate>,
        fallback: Option<AsrError>,
    ) -> Result<Vec<TranscriptUpdate>, AsrError> {
        if !updates.is_empty() {
            return Ok(updates);
        }
        if let Some(message) = self.pending_fatal.take() {
            return Err(AsrError::Fatal(message));
        }
        if let Some(message) = self.pending_transient.take() {
            return Err(AsrError::Transient(message));
        }
        if let Some(error) = fallback {
            return Err(error);
        }
        Ok(updates)
    }
}

/// Splits the comma-separated setting into the wire form. An empty setting
/// means "no hints", which is how the service auto-detects.
fn parse_language_hints(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|hint| !hint.is_empty())
        .map(str::to_string)
        .collect()
}

impl AsrBackend for SonioxBackend {
    fn id(&self) -> &'static str {
        "soniox"
    }

    fn capabilities(&self) -> AsrCapabilities {
        AsrCapabilities {
            revises_output: true,
            diarization: true,
            // The service does its own semantic endpointing and diarization, so
            // it needs the whole stream, silence included.
            audio_gating: AudioGating::Continuous,
            activity_source: ActivitySource::Backend { hold: SPEECH_HOLD },
        }
    }

    /// A live socket cannot be handed to the next session.
    fn is_cacheable(&self) -> bool {
        false
    }

    fn begin_session(&mut self, ctx: &SessionContext) -> Result<(), AsrError> {
        let settings = ctx.settings;
        let config = SonioxConfig {
            api_key: settings.soniox_api_key.clone(),
            model: settings.soniox_model.clone(),
            language_hints: parse_language_hints(&settings.soniox_language_hints),
            diarization: settings.soniox_diarization,
            endpoint_detection: settings.soniox_endpoint_detection,
        };
        println!("Connecting to Soniox: {config:?}");

        self.accumulator.reset();
        self.staged.clear();
        self.staged_samples = 0;
        self.frames_sent = 0;
        self.last_speech_at = None;
        self.pending_transient = None;
        self.pending_fatal = None;
        self.client = Some(SonioxClient::connect(config)?);
        Ok(())
    }

    fn push_audio(&mut self, samples: &[f32]) {
        // Buffer only — no socket traffic inside the VAD frame loop. The
        // conversion clamps, so full-scale overshoot cannot wrap sign.
        self.staged
            .extend_from_slice(&protocol::f32_to_pcm_s16le(samples));
        self.staged_samples += samples.len();
    }

    fn process(&mut self, ctx: &AsrContext) -> Result<Vec<TranscriptUpdate>, AsrError> {
        let mut send_error = None;

        if !self.staged.is_empty() {
            let pcm = std::mem::take(&mut self.staged);
            let samples = std::mem::replace(&mut self.staged_samples, 0);

            let Some(client) = self.client.as_ref() else {
                return Err(AsrError::Fatal(
                    "The Soniox session is not open".to_string(),
                ));
            };

            // On backpressure the staged buffer is already gone, which is
            // exactly the relief the ceiling is there to provide.
            match client.send_audio(pcm, samples) {
                Ok(()) => self.frames_sent += 1,
                // Deliberately not `?`. Returning here would skip the drain
                // below, and a send failure means the socket thread has already
                // exited — leaving the reason it exited unread. That reason is
                // the one worth showing.
                Err(e) => send_error = Some(e),
            }
        }

        let updates = self.drain_inbound(ctx.session.ids);
        self.deliver(updates, send_error)
    }

    fn poll(&mut self, ctx: &SessionContext) -> Result<Vec<TranscriptUpdate>, AsrError> {
        self.results(ctx.ids)
    }

    fn last_speech_at(&self) -> Option<Instant> {
        self.last_speech_at
    }

    fn chunks_processed(&self) -> u64 {
        self.frames_sent
    }

    /// Every Soniox parameter is fixed by the start frame, so none of them can
    /// change without a new socket. They take effect on the next Start.
    fn update_settings(&mut self, _settings: &Settings) {}

    fn end_session(&mut self, ctx: &SessionContext) -> Result<Vec<TranscriptUpdate>, AsrError> {
        // Third layer of re-entrancy safety: taking the client means a second
        // stop finds nothing to cancel and cannot discard finals still in
        // flight. (The other two are `finish`'s idempotence and the socket
        // task's own `finishing` flag.)
        let Some(mut client) = self.client.take() else {
            return Ok(self.accumulator.finish(ctx.ids));
        };

        let mut updates = Vec::new();

        // Audio staged below a full iteration is still part of the stream.
        if !self.staged.is_empty() {
            let pcm = std::mem::take(&mut self.staged);
            let samples = std::mem::replace(&mut self.staged_samples, 0);
            let _ = client.send_audio(pcm, samples);
        }

        client.finish();

        // After end-of-stream the server flushes whatever is left — possibly
        // several messages, carrying newly-final tokens — and only then
        // acknowledges. Those finals are the tail of the transcript, so they
        // are folded in rather than dropped.
        let deadline = Instant::now() + FINISH_WAIT;
        while let Some(remaining) = deadline.checked_duration_since(Instant::now()) {
            let Some(message) = client.recv_timeout(remaining) else {
                break;
            };
            match message {
                Inbound::Response(response) => {
                    updates.extend(self.accumulator.ingest(&response, ctx.ids));
                }
                Inbound::Reconnecting(_) => {
                    updates.extend(self.accumulator.finish(ctx.ids));
                    self.accumulator.reset();
                }
                Inbound::Finished | Inbound::Fatal(_) => break,
            }
        }

        updates.extend(self.accumulator.finish(ctx.ids));
        self.accumulator.reset();
        self.last_speech_at = None;
        self.pending_transient = None;
        self.pending_fatal = None;
        // Sends Shutdown and joins the socket thread.
        drop(client);

        Ok(updates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asr::soniox::protocol::{SonioxResponse, SonioxToken};

    fn token(text: &str, is_final: bool) -> SonioxToken {
        SonioxToken {
            text: text.to_string(),
            is_final,
            ..Default::default()
        }
    }

    #[test]
    fn language_hints_split_and_trim() {
        assert_eq!(parse_language_hints("en"), vec!["en".to_string()]);
        assert_eq!(
            parse_language_hints(" en , es "),
            vec!["en".to_string(), "es".to_string()]
        );
    }

    #[test]
    fn blank_language_hints_mean_no_hints() {
        assert!(parse_language_hints("").is_empty());
        assert!(parse_language_hints("  ").is_empty());
        assert!(parse_language_hints(",,").is_empty());
    }

    #[test]
    fn capabilities_match_what_the_pipeline_needs() {
        let caps = SonioxBackend::new().capabilities();
        assert!(caps.revises_output);
        assert!(caps.diarization);
        assert_eq!(caps.audio_gating, AudioGating::Continuous);
        assert_eq!(
            caps.activity_source,
            ActivitySource::Backend { hold: SPEECH_HOLD }
        );
        // The hold has to outlast the service's ~1 Hz idle chatter, or the
        // meter strobes through silence.
        assert!(SPEECH_HOLD > Duration::from_secs(1));
    }

    #[test]
    fn a_live_socket_is_never_reused_across_sessions() {
        assert!(!SonioxBackend::new().is_cacheable());
    }

    #[test]
    fn push_audio_stages_little_endian_pcm_without_sending() {
        let mut backend = SonioxBackend::new();
        backend.push_audio(&[0.0, 1.0]);
        assert_eq!(backend.staged, vec![0x00, 0x00, 0xFF, 0x7F]);
        assert_eq!(backend.staged_samples, 2);
        // No socket was opened, so nothing could have gone out.
        assert!(backend.client.is_none());
    }

    #[test]
    fn a_reconnect_finalizes_the_open_segment_and_owes_a_status() {
        let ids = SegmentIds::new();
        let mut backend = SonioxBackend::new();

        let updates = backend.fold(
            vec![
                Inbound::Response(Box::new(SonioxResponse {
                    tokens: vec![token("Half a sentence", true)],
                    ..Default::default()
                })),
                Inbound::Reconnecting("the connection dropped".to_string()),
            ],
            &ids,
        );

        // The best hypothesis available is finalized rather than lost, because
        // the new stream restarts its timeline from zero.
        let finals: Vec<_> = updates.iter().filter(|u| u.is_final).collect();
        assert_eq!(finals.len(), 1);
        assert_eq!(finals[0].text, "Half a sentence");
        assert!(backend.pending_transient.is_some());
    }

    #[test]
    fn speech_evidence_comes_from_token_content_not_message_arrival() {
        let ids = SegmentIds::new();
        let mut backend = SonioxBackend::new();

        // The service answers about once a second through total silence.
        backend.fold(
            vec![Inbound::Response(Box::new(SonioxResponse {
                tokens: vec![token("<end>", true)],
                ..Default::default()
            }))],
            &ids,
        );
        assert!(
            backend.last_speech_at().is_none(),
            "a control token is not speech"
        );

        backend.fold(
            vec![Inbound::Response(Box::new(SonioxResponse {
                tokens: vec![token(" hello", false)],
                ..Default::default()
            }))],
            &ids,
        );
        assert!(backend.last_speech_at().is_some());
    }

    #[test]
    fn text_wins_the_return_slot_and_status_is_owed_next_call() {
        let ids = SegmentIds::new();
        let mut backend = SonioxBackend::new();
        backend.pending_transient = Some("Reconnecting to Soniox: dropped".to_string());

        let updates = backend.fold(
            vec![Inbound::Response(Box::new(SonioxResponse {
                tokens: vec![token("Hello there", true)],
                ..Default::default()
            }))],
            &ids,
        );
        assert!(!updates.is_empty());

        let delivered = backend
            .deliver(updates, None)
            .expect("text is delivered first");
        assert_eq!(delivered.len(), 1);
        assert!(
            backend.pending_transient.is_some(),
            "the status is owed, not dropped"
        );

        // With no text left to deliver, the status surfaces.
        match backend.deliver(Vec::new(), None) {
            Err(AsrError::Transient(message)) => assert!(message.contains("Reconnecting")),
            other => panic!("expected the owed transient status, got {other:?}"),
        }
        assert!(backend.pending_transient.is_none());
    }

    #[test]
    fn a_fatal_flushes_its_text_first_and_is_reported_once() {
        let ids = SegmentIds::new();
        let mut backend = SonioxBackend::new();

        let updates = backend.fold(
            vec![
                Inbound::Response(Box::new(SonioxResponse {
                    tokens: vec![token("Recognized before the failure", true)],
                    ..Default::default()
                })),
                Inbound::Fatal("Soniox rejected the session".to_string()),
            ],
            &ids,
        );

        // The engine discards whatever end_session returns on the fatal path,
        // so this flush is the only thing keeping that text.
        let delivered = backend.deliver(updates, None).expect("text first");
        assert!(delivered
            .iter()
            .any(|u| u.is_final && u.text == "Recognized before the failure"));

        match backend.deliver(Vec::new(), None) {
            Err(AsrError::Fatal(message)) => assert!(message.contains("rejected")),
            other => panic!("expected a fatal, got {other:?}"),
        }
        // Taken, not repeated.
        assert!(backend.deliver(Vec::new(), None).is_ok());
    }

    #[test]
    fn the_socket_threads_reason_beats_the_send_failure_it_caused() {
        // Regression: a rejected API key made the socket thread exit, which made
        // the next send fail, and `?` on that send returned "the connection has
        // ended" without ever draining the real reason. An invalid key is the
        // likeliest first-run failure, so it is the one that has to read well.
        let ids = SegmentIds::new();
        let mut backend = SonioxBackend::new();
        backend.fold(
            vec![Inbound::Fatal(
                "Soniox rejected the session: bad key (unauthenticated)".to_string(),
            )],
            &ids,
        );

        let generic = Some(AsrError::Fatal(
            "The Soniox connection has ended".to_string(),
        ));
        match backend.deliver(Vec::new(), generic) {
            Err(AsrError::Fatal(message)) => assert!(
                message.contains("unauthenticated"),
                "expected the socket thread's reason, got {message:?}"
            ),
            other => panic!("expected a fatal, got {other:?}"),
        }
    }

    #[test]
    fn the_send_failure_is_still_reported_when_nothing_better_exists() {
        let mut backend = SonioxBackend::new();
        let generic = Some(AsrError::Fatal(
            "The Soniox connection has ended".to_string(),
        ));
        match backend.deliver(Vec::new(), generic) {
            Err(AsrError::Fatal(message)) => assert!(message.contains("connection has ended")),
            other => panic!("expected the fallback fatal, got {other:?}"),
        }
    }

    #[test]
    fn end_session_without_a_client_still_flushes_the_open_segment() {
        let ids = SegmentIds::new();
        let settings = Settings::default();
        let mut backend = SonioxBackend::new();
        backend.accumulator.ingest(
            &SonioxResponse {
                tokens: vec![token("An unfinished thought", true)],
                ..Default::default()
            },
            &ids,
        );

        // A None path yields an inert sink that writes nothing.
        let diag = crate::diag::DiagSink::open(
            None,
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            16_000,
            8_960,
            560,
            false,
        )
        .expect("an inert diagnostics sink always opens");
        let ctx = SessionContext {
            settings: &settings,
            diag: &diag,
            ids: &ids,
            loop_start: Instant::now(),
            sample_rate: 16_000,
            input_rate: 16_000,
            needs_resample: false,
        };

        let updates = backend.end_session(&ctx).expect("teardown succeeds");
        assert_eq!(updates.len(), 1);
        assert!(updates[0].is_final);
        assert_eq!(updates[0].text, "An unfinished thought");
    }
}
