use rubato::{FftFixedIn, Resampler};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::agc::AgcProcessor;
use crate::asr::{
    self, ActivitySource, AsrBackend, AsrContext, AsrError, AudioGating, IterationStats,
    SegmentIds, SessionContext, TranscriptUpdate,
};
use crate::audio_capture::{
    self, ActiveSessionInfo, AudioCapture, AudioDevice, AudioStream, CaptureBuffer,
};
use crate::diag::DiagSink;
use crate::settings::{self, Settings};
use crate::vad::{VadDecision, VadProcessor, VadState};
use crate::{EngineEventSink, StatusLevel};

/// Forwards a batch of updates to the UI. Backends return updates rather than
/// holding a sink, because the sink is a generic parameter and a backend that
/// held one could not be `dyn AsrBackend`.
fn emit_updates<E: EngineEventSink>(event_sink: &E, updates: Vec<TranscriptUpdate>) {
    for update in updates {
        event_sink.on_transcript_update(update);
    }
}

/// Routes a backend error to diagnostics and the UI. Returns true when the
/// session must end.
///
/// `Fatal` reuses the existing `transcription-error` path, which already stops
/// the UI. `Transient` must not: that path clears the running state, and a
/// recoverable hiccup should leave the session alive.
fn report_backend_error<E: EngineEventSink>(
    event_sink: &E,
    diag: &DiagSink,
    error: &AsrError,
    loop_start: Instant,
) -> bool {
    let uptime = loop_start.elapsed().as_millis() as i64;
    match error {
        AsrError::Transient(message) => {
            diag.backend_error(uptime, "transient", message);
            eprintln!("[asr] Transient backend error: {message}");
            event_sink.on_status(StatusLevel::Warn, message.clone());
            false
        }
        AsrError::Fatal(message) => {
            diag.backend_error(uptime, "fatal", message);
            eprintln!("[asr] Fatal backend error: {message}");
            event_sink.on_error(format!("Transcription error: {message}"));
            true
        }
    }
}

const VAD_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/silero_vad.onnx");
const ASR_SAMPLE_RATE: usize = 16000;
const VAD_FRAME_SIZE: usize = 512;
const AUDIO_LEVEL_EMIT_INTERVAL: Duration = Duration::from_millis(50);
const AUDIO_LEVEL_FLOOR_DB: f32 = -60.0;

/// Accumulates RMS across processing iterations and emits no faster than the
/// UI needs. This keeps metering off the capture callback and limits bridge
/// traffic to 20 events per second.
struct AudioLevelMeter {
    sum_squares: f64,
    sample_count: usize,
    last_emit: Instant,
}

impl AudioLevelMeter {
    fn new(now: Instant) -> Self {
        Self {
            sum_squares: 0.0,
            sample_count: 0,
            last_emit: now,
        }
    }

    fn observe(&mut self, samples: &[f32]) {
        self.sum_squares += samples
            .iter()
            .map(|&sample| {
                let sample = sample as f64;
                sample * sample
            })
            .sum::<f64>();
        self.sample_count += samples.len();
    }

    fn take_level_if_due(&mut self, now: Instant) -> Option<f32> {
        if now.duration_since(self.last_emit) < AUDIO_LEVEL_EMIT_INTERVAL {
            return None;
        }

        let rms = if self.sample_count == 0 {
            0.0
        } else {
            (self.sum_squares / self.sample_count as f64).sqrt() as f32
        };
        self.sum_squares = 0.0;
        self.sample_count = 0;
        self.last_emit = now;

        Some(normalize_audio_level(rms))
    }
}

fn normalize_audio_level(rms: f32) -> f32 {
    if !rms.is_finite() || rms <= 0.0 {
        return 0.0;
    }

    let db = 20.0 * rms.log10();
    ((db - AUDIO_LEVEL_FLOOR_DB) / -AUDIO_LEVEL_FLOOR_DB).clamp(0.0, 1.0)
}

/// What the processing thread hands back so the engine can reuse the loaded
/// models on the next session.
type SessionModels = (Box<dyn AsrBackend>, VadProcessor);

pub enum Command {
    ListDevices {
        reply: mpsc::Sender<Vec<AudioDevice>>,
    },
    Start {
        device_id: Option<String>,
        settings: Settings,
    },
    Stop,
    /// Swap the audio stream to a new device without restarting the processing thread.
    /// Used by the PipeWire watcher when an app stream reappears.
    Reconnect {
        device_id: String,
    },
    /// Push updated settings to the active processing thread (hot-reload).
    UpdateSettings {
        settings: Settings,
    },
}

pub struct AudioEngine<E: EngineEventSink> {
    event_sink: E,
    cmd_rx: mpsc::Receiver<Command>,
    capture_backend: Box<dyn AudioCapture>,
    // Active session state
    active_stream: Option<Box<dyn AudioStream>>,
    processing_thread: Option<JoinHandle<Option<SessionModels>>>,
    stop_flag: Option<Arc<AtomicBool>>,
    capture_stop_flag: Option<Arc<AtomicBool>>,
    active_buffer: Option<Arc<Mutex<CaptureBuffer>>>,
    active_session_info: Arc<Mutex<ActiveSessionInfo>>,
    settings_tx: Option<mpsc::Sender<Settings>>,
    // Cached models for reuse across sessions
    cached_backend: Option<Box<dyn AsrBackend>>,
    cached_backend_id: Option<String>,
    cached_vad: Option<VadProcessor>,
    cached_model_path: Option<String>,
    cached_model_config: Option<(usize, usize)>,
    /// Segment ids are allocated from here for the whole process lifetime, so
    /// an id is never reused across sessions or provider switches while a
    /// stale segment carrying it could still be on screen.
    segment_ids: SegmentIds,
    /// Runtime toggle for diagnostics logging. Shared with the active
    /// processing thread so flipping it off takes effect mid-session.
    diag_enabled: Arc<AtomicBool>,
}

impl<E: EngineEventSink> AudioEngine<E> {
    pub fn new(
        event_sink: E,
        cmd_rx: mpsc::Receiver<Command>,
        capture_backend: Box<dyn AudioCapture>,
        active_session_info: Arc<Mutex<ActiveSessionInfo>>,
        diag_enabled: Arc<AtomicBool>,
    ) -> Self {
        println!(
            "AudioEngine initialized with {} backend",
            capture_backend.name()
        );
        Self {
            event_sink,
            cmd_rx,
            capture_backend,
            active_stream: None,
            processing_thread: None,
            stop_flag: None,
            capture_stop_flag: None,
            active_buffer: None,
            active_session_info,
            settings_tx: None,
            cached_backend: None,
            cached_backend_id: None,
            cached_vad: None,
            cached_model_path: None,
            cached_model_config: None,
            segment_ids: SegmentIds::new(),
            diag_enabled,
        }
    }

    pub fn run(mut self) {
        loop {
            let cmd = match self.cmd_rx.recv() {
                Ok(cmd) => cmd,
                Err(_) => break, // Channel closed
            };

            match cmd {
                Command::ListDevices { reply } => {
                    let devices = match self.capture_backend.enumerate_devices() {
                        Ok(devices) => {
                            // Sort by priority: apps first, then inputs, then monitors
                            audio_capture::sort_devices_by_priority(devices)
                        }
                        Err(e) => {
                            eprintln!("Failed to enumerate devices: {}", e);
                            Vec::new()
                        }
                    };
                    let _ = reply.send(devices);
                }
                Command::Start {
                    device_id,
                    settings,
                } => {
                    self.stop_active_session();
                    if let Err(e) = self.start_session(device_id, settings) {
                        eprintln!("Failed to start transcription: {}", e);
                        self.event_sink.on_error(format!("Error: {}", e));
                    }
                }
                Command::Stop => {
                    self.stop_active_session();
                }
                Command::Reconnect { device_id } => {
                    self.reconnect_stream(device_id);
                }
                Command::UpdateSettings { settings } => {
                    self.diag_enabled
                        .store(settings.diagnostics_enabled, Ordering::Relaxed);
                    if let Some(ref tx) = self.settings_tx {
                        let _ = tx.send(settings);
                    }
                }
            }
        }
    }

    fn start_session(
        &mut self,
        device_id: Option<String>,
        settings: Settings,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let chunk_size = settings::chunk_ms_to_samples(settings.chunk_ms);
        println!(
            "Session starting with chunk_ms={}ms ({} samples), intra={}, inter={}, punctuation_reset={}, empty_reset_threshold={}",
            settings.chunk_ms, chunk_size, settings.intra_threads, settings.inter_threads,
            settings.punctuation_reset, settings.empty_reset_threshold
        );

        // If no device specified, try to select default
        let device_id = match device_id {
            Some(id) => Some(id),
            None => {
                let devices = self.capture_backend.enumerate_devices()?;
                audio_capture::select_default_device(&devices)
            }
        };

        if device_id.is_none() {
            return Err("No device available for capture".into());
        }

        let buffer: Arc<Mutex<CaptureBuffer>> =
            Arc::new(Mutex::new(CaptureBuffer::for_sample_rate(48000)));
        let stop_flag = Arc::new(AtomicBool::new(false));
        let capture_stop_flag = Arc::new(AtomicBool::new(false));
        let stop_flag_thread = Arc::clone(&stop_flag);
        let event_sink = self.event_sink.clone();

        // Look up the device info for session tracking (used by watcher for reconnect)
        let device_info = device_id.as_ref().and_then(|id| {
            self.capture_backend
                .enumerate_devices()
                .ok()
                .and_then(|devs| devs.into_iter().find(|d| d.id == *id))
        });

        // Start the capture backend
        let stream = self.capture_backend.start(
            device_id.clone(),
            Arc::clone(&buffer),
            Arc::clone(&capture_stop_flag),
        )?;

        let input_rate = stream.metadata.sample_rate;
        if let Ok(mut guard) = buffer.lock() {
            guard.set_capacity(input_rate * 10);
        }
        let needs_resample = input_rate != ASR_SAMPLE_RATE;

        println!(
            "Audio config: {} Hz, {} channel(s), {} (resample: {})",
            input_rate, stream.metadata.channels, stream.metadata.sample_format, needs_resample
        );

        // Check if cached models are compatible with current settings
        let model_path_str = settings::expand_tilde(&settings.model_path)
            .to_string_lossy()
            .to_string();
        let model_config = (settings.intra_threads, settings.inter_threads);
        let cached_compatible = self.cached_model_path.as_deref() == Some(&model_path_str)
            && self.cached_model_config == Some(model_config)
            && self.cached_backend_id.as_deref() == Some(settings.asr_provider.as_str());

        let cached_backend = if cached_compatible {
            self.cached_backend.take()
        } else {
            if self.cached_backend.is_some() {
                println!("Model config changed — discarding cached models");
            }
            self.cached_backend.take(); // drop old
            None
        };
        let cached_vad = if cached_compatible {
            self.cached_vad.take()
        } else {
            self.cached_vad.take(); // drop old
            None
        };

        let (settings_tx, settings_rx) = mpsc::channel();
        self.settings_tx = Some(settings_tx);

        // Snapshot live diagnostics toggle to match the new session's settings,
        // then compute the DB path only if enabled at session start.
        self.diag_enabled
            .store(settings.diagnostics_enabled, Ordering::Relaxed);
        let diag_db_path = if settings.diagnostics_enabled {
            Some(settings::expand_tilde(&settings.diagnostics_db_path))
        } else {
            None
        };
        let diag_enabled_for_thread = Arc::clone(&self.diag_enabled);
        let buffer_for_thread = Arc::clone(&buffer);
        let segment_ids = self.segment_ids.clone();
        let provider_id = settings.asr_provider.clone();
        // A crash inside the loop used to end transcription silently; keep a
        // clone so it can be reported to the UI.
        let sink_for_crash = event_sink.clone();
        let processing_thread = thread::spawn(move || {
            println!("[diag] Processing thread started");
            match Self::processing_loop(
                event_sink,
                buffer_for_thread,
                stop_flag_thread,
                input_rate,
                needs_resample,
                settings,
                cached_backend,
                cached_vad,
                settings_rx,
                diag_db_path,
                diag_enabled_for_thread,
                segment_ids,
            ) {
                Ok(models) => {
                    println!("[diag] Processing loop exited normally");
                    Some(models)
                }
                Err(e) => {
                    eprintln!("[diag] Processing loop CRASHED: {}", e);
                    sink_for_crash.on_error(format!("Transcription stopped: {}", e));
                    None
                }
            }
        });

        self.active_stream = Some(stream.stream);
        self.processing_thread = Some(processing_thread);
        self.stop_flag = Some(stop_flag);
        self.capture_stop_flag = Some(capture_stop_flag);
        self.active_buffer = Some(buffer);
        self.cached_model_path = Some(model_path_str);
        self.cached_model_config = Some(model_config);
        self.cached_backend_id = Some(provider_id);

        // Update shared session info for the watcher
        if let Ok(mut info) = self.active_session_info.lock() {
            info.device_id = device_id;
            info.application_name = device_info
                .as_ref()
                .and_then(|d| d.application_name.clone());
            info.device_type = device_info.map(|d| d.device_type);
        }

        Ok(())
    }

    fn stop_active_session(&mut self) {
        if let Some(flag) = self.stop_flag.take() {
            flag.store(true, Ordering::Relaxed);
        }
        if let Some(flag) = self.capture_stop_flag.take() {
            flag.store(true, Ordering::Relaxed);
        }
        // Drop the settings sender so the processing thread's try_recv sees disconnect
        self.settings_tx = None;
        // Stop and drop the stream
        if let Some(stream) = self.active_stream.take() {
            stream.stop();
        }
        if let Some(handle) = self.processing_thread.take() {
            match handle.join() {
                Ok(Some((backend, vad))) => {
                    if backend.is_cacheable() {
                        println!("[diag] Processing thread joined — caching models for reuse");
                        self.cached_backend = Some(backend);
                    } else {
                        println!("[diag] Processing thread joined — backend is not cacheable");
                        self.cached_backend_id = None;
                    }
                    self.cached_vad = Some(vad);
                }
                Ok(None) => {
                    println!("[diag] Processing thread joined — no models to cache (error path)");
                    self.cached_backend_id = None;
                }
                Err(e) => {
                    eprintln!("[diag] Processing thread PANICKED: {:?}", e);
                    self.cached_backend_id = None;
                    self.event_sink
                        .on_error("Transcription thread crashed.".to_string());
                }
            }
        }
        self.active_buffer = None;

        // Clear shared session info
        if let Ok(mut info) = self.active_session_info.lock() {
            *info = ActiveSessionInfo::default();
        }
    }

    /// Swap the audio stream to a new device without restarting the processing thread.
    /// The processing loop keeps running and reading from the same shared buffer.
    fn reconnect_stream(&mut self, device_id: String) {
        // Only reconnect if we have an active session
        let Some(buffer) = self.active_buffer.as_ref() else {
            println!("[Engine] Reconnect ignored — no active session");
            return;
        };

        println!("[Engine] Reconnecting to device {}", device_id);

        // Stop only the audio stream, NOT the processing thread
        if let Some(stream) = self.active_stream.take() {
            stream.stop();
        }
        if let Some(flag) = self.capture_stop_flag.take() {
            flag.store(true, Ordering::Relaxed);
        }
        let capture_stop_flag = Arc::new(AtomicBool::new(false));

        // Start a new stream with the same buffer and stop_flag
        match self.capture_backend.start(
            Some(device_id.clone()),
            Arc::clone(buffer),
            Arc::clone(&capture_stop_flag),
        ) {
            Ok(stream) => {
                self.active_stream = Some(stream.stream);
                self.capture_stop_flag = Some(capture_stop_flag);

                // Update session info for the watcher
                let device_info = self
                    .capture_backend
                    .enumerate_devices()
                    .ok()
                    .and_then(|devs| devs.into_iter().find(|d| d.id == device_id));

                if let Ok(mut info) = self.active_session_info.lock() {
                    info.device_id = Some(device_id.clone());
                    info.application_name = device_info
                        .as_ref()
                        .and_then(|d| d.application_name.clone());
                    info.device_type = device_info.map(|d| d.device_type);
                }

                // Notify frontend of the source change
                self.event_sink.on_source_switched(device_id.clone());
                println!("[Engine] Reconnected to device {}", device_id);
            }
            Err(e) => {
                eprintln!("[Engine] Reconnect failed: {}", e);
                self.event_sink.on_error(format!("Reconnect failed: {}", e));
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn processing_loop(
        event_sink: E,
        buffer: Arc<Mutex<CaptureBuffer>>,
        stop_flag: Arc<AtomicBool>,
        input_rate: usize,
        needs_resample: bool,
        settings: Settings,
        cached_backend: Option<Box<dyn AsrBackend>>,
        cached_vad: Option<VadProcessor>,
        settings_rx: mpsc::Receiver<Settings>,
        diag_db_path: Option<PathBuf>,
        diag_enabled: Arc<AtomicBool>,
        segment_ids: SegmentIds,
    ) -> Result<SessionModels, Box<dyn std::error::Error>> {
        let chunk_size = settings::chunk_ms_to_samples(settings.chunk_ms);

        // Every write through this sink is gated on the live `diag_enabled`
        // toggle. If the user flips diagnostics off mid-session, writes are
        // skipped while the connection stays open, so re-enabling resumes
        // writes to the same session row.
        let diag = DiagSink::open(
            diag_db_path.as_deref(),
            Arc::clone(&diag_enabled),
            input_rate,
            chunk_size,
            settings.chunk_ms,
            needs_resample,
        )?;

        let mut backend = asr::create_backend(&settings, cached_backend)
            .map_err(|e| Box::<dyn std::error::Error>::from(e.to_string()))?;
        let caps = backend.capabilities();

        let mut vad = if let Some(mut v) = cached_vad {
            println!("Using cached VAD model (skipping reload)");
            v.reset();
            v
        } else {
            println!("Loading Silero VAD model from {}...", VAD_MODEL_PATH);
            let v = VadProcessor::new(
                Path::new(VAD_MODEL_PATH),
                settings.vad_threshold_start,
                settings.vad_threshold_end,
                500, // min_silence_duration_ms
                250, // min_speech_duration_ms
                500, // pre_speech_ms (ring buffer = 500ms)
            )?;
            println!("VAD model loaded.");
            v
        };

        let mut agc = AgcProcessor::new(
            settings.agc_enabled,
            settings.agc_target_rms_dbfs,
            settings.agc_max_gain_db,
            settings.agc_attack_ms,
            settings.agc_release_ms,
            ASR_SAMPLE_RATE as f32,
        );

        let mut resampler: Option<FftFixedIn<f32>> = if needs_resample {
            Some(FftFixedIn::<f32>::new(
                input_rate,
                ASR_SAMPLE_RATE,
                1024,
                1,
                1,
            )?)
        } else {
            None
        };

        let mut resample_leftover: Vec<f32> = Vec::new();
        let mut vad_leftover: Vec<f32> = Vec::new();
        let loop_start = Instant::now();
        let mut speech_start_uptime_ms: Option<i64> = None;
        let mut audio_level_meter = AudioLevelMeter::new(Instant::now());

        // `loop_start` is fixed for the session and everything else is a
        // borrow, so this is rebuilt per call rather than stored.
        macro_rules! session_ctx {
            () => {
                SessionContext {
                    settings: &settings,
                    diag: &diag,
                    ids: &segment_ids,
                    loop_start,
                    sample_rate: ASR_SAMPLE_RATE,
                    input_rate,
                    needs_resample,
                }
            };
        }

        if let Err(e) = backend.begin_session(&session_ctx!()) {
            diag.backend_error(
                loop_start.elapsed().as_millis() as i64,
                "fatal",
                &e.to_string(),
            );
            return Err(Box::<dyn std::error::Error>::from(e.to_string()));
        }

        loop {
            if stop_flag.load(Ordering::Relaxed) {
                match backend.end_session(&session_ctx!()) {
                    Ok(updates) => emit_updates(&event_sink, updates),
                    Err(e) => diag.backend_error(
                        loop_start.elapsed().as_millis() as i64,
                        "fatal",
                        &e.to_string(),
                    ),
                }
                diag.shutdown(
                    loop_start.elapsed().as_millis() as i64,
                    backend.chunks_processed(),
                );
                return Ok((backend, vad));
            }

            // Check for hot-reloaded settings
            if let Ok(new_settings) = settings_rx.try_recv() {
                println!(
                    "[diag] Hot-reloading settings: punctuation_reset={}, empty_reset_threshold={}, vad_threshold_start={}, vad_threshold_end={}",
                    new_settings.punctuation_reset, new_settings.empty_reset_threshold,
                    new_settings.vad_threshold_start, new_settings.vad_threshold_end
                );
                backend.update_settings(&new_settings);
                vad.update_params(
                    new_settings.vad_threshold_start,
                    new_settings.vad_threshold_end,
                    500,
                    250,
                );
                agc.update_params(
                    new_settings.agc_enabled,
                    new_settings.agc_target_rms_dbfs,
                    new_settings.agc_max_gain_db,
                    new_settings.agc_attack_ms,
                    new_settings.agc_release_ms,
                );
            }

            let (drained, dropped_samples) = {
                let mut guard = buffer.lock().unwrap();
                guard.drain_all()
            };

            if dropped_samples > 0 {
                eprintln!(
                    "[diag] Capture buffer dropped {} stale sample(s)",
                    dropped_samples
                );
            }

            if drained.is_empty() {
                // Service socket-driven backends while capture is starved, so
                // they can surface results that arrived during silence.
                match backend.poll(&session_ctx!()) {
                    Ok(updates) => emit_updates(&event_sink, updates),
                    Err(e) => {
                        if report_backend_error(&event_sink, &diag, &e, loop_start) {
                            let _ = backend.end_session(&session_ctx!());
                            return Ok((backend, vad));
                        }
                    }
                }
                thread::sleep(std::time::Duration::from_millis(10));
                continue;
            }

            let iter_start = Instant::now();
            let drain_count = drained.len();
            let drain_audio_ms = drain_count as f64 / input_rate as f64 * 1000.0;

            let resample_start = Instant::now();
            let (mut samples_16k, _resample_in, _resample_out, _resample_leftover) =
                if let Some(ref mut resampler) = resampler {
                    let rs_chunk = resampler.input_frames_next();
                    let resample_input =
                        take_complete_frames(&mut resample_leftover, &drained, rs_chunk);
                    let mut resampled = Vec::new();
                    let mut offset = 0;

                    while offset + rs_chunk <= resample_input.len() {
                        let input_chunk = &resample_input[offset..offset + rs_chunk];
                        match resampler.process(&[input_chunk], None) {
                            Ok(output) => {
                                if !output.is_empty() {
                                    resampled.extend_from_slice(&output[0]);
                                }
                            }
                            Err(e) => {
                                diag.resample_error(
                                    loop_start.elapsed().as_millis() as i64,
                                    &e.to_string(),
                                );
                            }
                        }
                        offset += rs_chunk;
                    }

                    let leftover = resample_leftover.len();
                    let rs_in = resample_input.len();
                    let rs_out = resampled.len();
                    (resampled, rs_in, rs_out, leftover)
                } else {
                    let len = drained.len();
                    (drained, len, len, 0usize)
                };

            let resample_ms = resample_start.elapsed().as_millis() as i64;

            // Meter the captured signal before AGC so the UI reflects the
            // source's real input level rather than the configured gain.
            audio_level_meter.observe(&samples_16k);

            // --- AGC ---
            agc.process(&mut samples_16k);

            // --- VAD gating ---
            // Prepend any leftover from last iteration
            let mut vad_input = std::mem::take(&mut vad_leftover);
            vad_input.extend_from_slice(&samples_16k);

            let vad_start = Instant::now();
            let mut offset = 0;
            while offset + VAD_FRAME_SIZE <= vad_input.len() {
                let frame = &vad_input[offset..offset + VAD_FRAME_SIZE];
                offset += VAD_FRAME_SIZE;

                let (decision, _prob) = match vad.process_frame(frame) {
                    Ok(result) => result,
                    Err(e) => {
                        diag.vad_error(loop_start.elapsed().as_millis() as i64, &e.to_string());
                        continue;
                    }
                };

                // The only capability-dependent branch in the pipeline.
                match caps.audio_gating {
                    AudioGating::VadGated => match decision {
                        VadDecision::Silence => {
                            // Audio is in the ring buffer; nothing to do
                        }
                        VadDecision::SpeechStarted { pre_speech_samples } => {
                            let uptime = loop_start.elapsed().as_millis() as i64;
                            speech_start_uptime_ms = Some(uptime);
                            backend.mark_speech_start();

                            diag.speech_start(uptime, pre_speech_samples.len());

                            // Prepend ring buffer contents then this frame
                            backend.push_audio(&pre_speech_samples);
                            backend.push_audio(frame);
                        }
                        VadDecision::SpeechContinues => {
                            backend.push_audio(frame);
                        }
                        VadDecision::SpeechEnded => {
                            backend.push_audio(frame);

                            let uptime = loop_start.elapsed().as_millis() as i64;
                            let duration_ms = speech_start_uptime_ms
                                .map(|start| (uptime - start) as f64)
                                .unwrap_or(0.0);

                            backend.mark_speech_end(uptime, duration_ms);

                            speech_start_uptime_ms = None;
                        }
                    },
                    AudioGating::Continuous => {
                        // Every frame exactly once, in order. Never the
                        // pre-speech ring buffer: those samples already went
                        // out as ordinary silence frames, and re-sending them
                        // would duplicate ~500 ms of audio at each onset.
                        backend.push_audio(frame);

                        match decision {
                            VadDecision::SpeechStarted { pre_speech_samples } => {
                                let uptime = loop_start.elapsed().as_millis() as i64;
                                speech_start_uptime_ms = Some(uptime);
                                backend.mark_speech_start();
                                diag.speech_start(uptime, pre_speech_samples.len());
                            }
                            VadDecision::SpeechEnded => {
                                let uptime = loop_start.elapsed().as_millis() as i64;
                                let duration_ms = speech_start_uptime_ms
                                    .map(|start| (uptime - start) as f64)
                                    .unwrap_or(0.0);
                                backend.mark_speech_end(uptime, duration_ms);
                                speech_start_uptime_ms = None;
                            }
                            VadDecision::Silence | VadDecision::SpeechContinues => {}
                        }
                    }
                }
            }

            let vad_ms = vad_start.elapsed().as_millis() as i64;

            // Save leftover sub-frame samples for next iteration
            if offset < vad_input.len() {
                vad_leftover = vad_input[offset..].to_vec();
            }

            let vad_state = vad.state();

            let now = Instant::now();
            if let Some(level) = audio_level_meter.take_level_if_due(now) {
                // Grey vs coloured is load-bearing: it means speech is being
                // heard right now. A VAD-gated backend uses the VAD; a backend
                // that does its own endpointing reports its own evidence.
                let speech_active = match caps.activity_source {
                    ActivitySource::Vad => vad_state == VadState::Speech,
                    ActivitySource::Backend { hold } => backend
                        .last_speech_at()
                        .is_some_and(|at| now.duration_since(at) < hold),
                };
                event_sink.on_audio_level(level, speech_active);
            }

            let stats = IterationStats {
                iter_start,
                drain_samples: drain_count,
                drain_audio_ms,
                resample_ms,
                vad_ms,
                vad_state,
            };
            let asr_result = backend.process(&AsrContext {
                session: &session_ctx!(),
                stats: &stats,
            });
            match asr_result {
                Ok(updates) => emit_updates(&event_sink, updates),
                Err(e) => {
                    if report_backend_error(&event_sink, &diag, &e, loop_start) {
                        let _ = backend.end_session(&session_ctx!());
                        return Ok((backend, vad));
                    }
                }
            }
        }
    }
}

fn take_complete_frames(leftover: &mut Vec<f32>, drained: &[f32], frame_size: usize) -> Vec<f32> {
    let mut input = std::mem::take(leftover);
    input.extend_from_slice(drained);
    let complete_len = input.len() / frame_size * frame_size;
    if complete_len < input.len() {
        leftover.extend_from_slice(&input[complete_len..]);
        input.truncate(complete_len);
    }
    input
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_level_maps_floor_and_full_scale() {
        assert_eq!(normalize_audio_level(0.0), 0.0);
        assert!((normalize_audio_level(0.001) - 0.0).abs() < f32::EPSILON);
        assert!((normalize_audio_level(1.0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn audio_level_meter_accumulates_until_interval() {
        let start = Instant::now();
        let mut meter = AudioLevelMeter::new(start);
        meter.observe(&[0.1, -0.1, 0.1, -0.1]);

        assert!(meter
            .take_level_if_due(start + Duration::from_millis(49))
            .is_none());
        let level = meter
            .take_level_if_due(start + AUDIO_LEVEL_EMIT_INTERVAL)
            .expect("meter should emit at its configured interval");
        assert!(level > 0.0 && level < 1.0);
    }

    #[test]
    fn take_complete_frames_preserves_leftover_order() {
        let mut leftover = vec![1.0, 2.0];

        let complete = take_complete_frames(&mut leftover, &[3.0, 4.0, 5.0], 4);

        assert_eq!(complete, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(leftover, vec![5.0]);
    }

    #[test]
    fn take_complete_frames_keeps_all_incomplete_input_as_leftover() {
        let mut leftover = vec![1.0];

        let complete = take_complete_frames(&mut leftover, &[2.0], 4);

        assert!(complete.is_empty());
        assert_eq!(leftover, vec![1.0, 2.0]);
    }
}
