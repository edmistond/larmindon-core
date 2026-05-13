#[cfg(any(feature = "webgpu", feature = "directml", feature = "migraphx"))]
use parakeet_rs::ExecutionProvider;
use parakeet_rs::{ExecutionConfig, Nemotron};
use rubato::{FftFixedIn, Resampler};
use rusqlite::Connection;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use crate::agc::AgcProcessor;
use crate::audio_capture::{
    self, ActiveSessionInfo, AudioCapture, AudioDevice, AudioStream, CaptureBuffer,
};
use crate::settings::{self, Settings};
use crate::vad::{VadDecision, VadProcessor, VadState};
use crate::EngineEventSink;

const VAD_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/silero_vad.onnx");
const ASR_SAMPLE_RATE: usize = 16000;
const VAD_FRAME_SIZE: usize = 512;

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
    processing_thread: Option<JoinHandle<Option<(Nemotron, VadProcessor)>>>,
    stop_flag: Option<Arc<AtomicBool>>,
    capture_stop_flag: Option<Arc<AtomicBool>>,
    active_buffer: Option<Arc<Mutex<CaptureBuffer>>>,
    active_session_info: Arc<Mutex<ActiveSessionInfo>>,
    settings_tx: Option<mpsc::Sender<Settings>>,
    // Cached models for reuse across sessions
    cached_nemotron: Option<Nemotron>,
    cached_vad: Option<VadProcessor>,
    cached_model_path: Option<String>,
    cached_model_config: Option<(usize, usize)>,
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
            cached_nemotron: None,
            cached_vad: None,
            cached_model_path: None,
            cached_model_config: None,
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
            && self.cached_model_config == Some(model_config);

        let cached_nemotron = if cached_compatible {
            self.cached_nemotron.take()
        } else {
            if self.cached_nemotron.is_some() {
                println!("Model config changed — discarding cached models");
            }
            self.cached_nemotron.take(); // drop old
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
        let processing_thread = thread::spawn(move || {
            println!("[diag] Processing thread started");
            match Self::processing_loop(
                event_sink,
                buffer_for_thread,
                stop_flag_thread,
                input_rate,
                needs_resample,
                settings,
                cached_nemotron,
                cached_vad,
                settings_rx,
                diag_db_path,
                diag_enabled_for_thread,
            ) {
                Ok(models) => {
                    println!("[diag] Processing loop exited normally");
                    Some(models)
                }
                Err(e) => {
                    eprintln!("[diag] Processing loop CRASHED: {}", e);
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
                Ok(Some((nemotron, vad))) => {
                    println!("[diag] Processing thread joined — caching models for reuse");
                    self.cached_nemotron = Some(nemotron);
                    self.cached_vad = Some(vad);
                }
                Ok(None) => {
                    println!("[diag] Processing thread joined — no models to cache (error path)");
                }
                Err(e) => eprintln!("[diag] Processing thread PANICKED: {:?}", e),
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

    fn init_diag_db(
        db_path: Option<&Path>,
    ) -> Result<Option<Connection>, Box<dyn std::error::Error>> {
        let Some(db_path) = db_path else {
            return Ok(None);
        };
        println!("[diag] Diagnostics DB: {}", db_path.display());
        let conn = Connection::open(db_path)?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             CREATE TABLE IF NOT EXISTS sessions (
                 id INTEGER PRIMARY KEY,
                 started_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now', 'localtime')),
                 input_rate INTEGER,
                 chunk_size INTEGER,
                 needs_resample INTEGER
             );
             CREATE TABLE IF NOT EXISTS events (
                 id INTEGER PRIMARY KEY,
                 session_id INTEGER,
                 ts TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now', 'localtime')),
                 uptime_ms INTEGER,
                 event_type TEXT,
                 chunk_num INTEGER,
                 inference_ms INTEGER,
                 drain_samples INTEGER,
                 drain_audio_ms REAL,
                 resample_in INTEGER,
                 resample_out INTEGER,
                 resample_leftover INTEGER,
                 asr_buf_len INTEGER,
                 text_empty INTEGER,
                 text_preview TEXT,
                 error_msg TEXT,
                 vad_state TEXT
             );
             CREATE TABLE IF NOT EXISTS vad_events (
                 id INTEGER PRIMARY KEY,
                 session_id INTEGER,
                 ts TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now', 'localtime')),
                 uptime_ms INTEGER,
                 event_type TEXT,
                 pre_speech_samples INTEGER,
                 speech_duration_ms REAL,
                 consecutive_empty INTEGER,
                 probability REAL,
                 chunks_since_decoder_reset INTEGER,
                 audio_ms_since_decoder_reset INTEGER
             );",
        )?;
        // Migrate: add columns if they don't exist (ALTER TABLE has no IF NOT EXISTS).
        let _ = conn.execute_batch("ALTER TABLE events ADD COLUMN vad_state TEXT;");
        let _ = conn.execute_batch("ALTER TABLE events ADD COLUMN vad_ms INTEGER;");
        let _ = conn.execute_batch("ALTER TABLE events ADD COLUMN resample_ms INTEGER;");
        let _ = conn.execute_batch("ALTER TABLE events ADD COLUMN iteration_ms INTEGER;");
        let _ = conn
            .execute_batch("ALTER TABLE vad_events ADD COLUMN chunks_since_decoder_reset INTEGER;");
        let _ = conn.execute_batch(
            "ALTER TABLE vad_events ADD COLUMN audio_ms_since_decoder_reset INTEGER;",
        );
        Ok(Some(conn))
    }

    #[allow(clippy::too_many_arguments)]
    fn processing_loop(
        event_sink: E,
        buffer: Arc<Mutex<CaptureBuffer>>,
        stop_flag: Arc<AtomicBool>,
        input_rate: usize,
        needs_resample: bool,
        settings: Settings,
        cached_nemotron: Option<Nemotron>,
        cached_vad: Option<VadProcessor>,
        settings_rx: mpsc::Receiver<Settings>,
        diag_db_path: Option<PathBuf>,
        diag_enabled: Arc<AtomicBool>,
    ) -> Result<(Nemotron, VadProcessor), Box<dyn std::error::Error>> {
        let chunk_size = settings::chunk_ms_to_samples(settings.chunk_ms);
        let db = Self::init_diag_db(diag_db_path.as_deref())?;

        // Every `db.as_ref().filter(|_| diag_enabled.load(...))` site below
        // gates a write on the live toggle. If the user flips diagnostics off
        // mid-session, writes are skipped while the connection stays open, so
        // re-enabling resumes writes to the same session row.
        let session_id =
            if let Some(db) = db.as_ref().filter(|_| diag_enabled.load(Ordering::Relaxed)) {
                db.execute(
                "INSERT INTO sessions (input_rate, chunk_size, needs_resample) VALUES (?1, ?2, ?3)",
                rusqlite::params![input_rate as i64, chunk_size as i64, needs_resample as i64],
            )?;
                db.last_insert_rowid()
            } else {
                0
            };

        let mut punctuation_reset_enabled = settings.punctuation_reset;
        let mut empty_reset_threshold = settings.empty_reset_threshold;

        let mut model = if let Some(mut m) = cached_nemotron {
            println!("Using cached Nemotron model (skipping reload)");
            m.reset();
            m
        } else {
            let model_path = settings::expand_tilde(&settings.model_path);
            println!(
                "Loading Nemotron model from {} (intra_threads={}, inter_threads={})...",
                model_path.display(),
                settings.intra_threads,
                settings.inter_threads
            );
            #[allow(unused_mut)]
            let mut model_config = ExecutionConfig::new()
                .with_intra_threads(settings.intra_threads)
                .with_inter_threads(settings.inter_threads);

            #[cfg(feature = "webgpu")]
            {
                println!("WebGPU feature enabled — using WebGPU (Metal) execution provider");
                model_config = model_config.with_execution_provider(ExecutionProvider::WebGPU);
            }

            let m = Nemotron::from_pretrained(&model_path, Some(model_config))?;
            println!("Model loaded.");
            m
        };

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

        let mut asr_buffer: Vec<f32> = Vec::with_capacity(chunk_size * 2);
        let mut resample_leftover: Vec<f32> = Vec::new();
        let mut vad_leftover: Vec<f32> = Vec::new();
        let loop_start = Instant::now();
        let mut chunk_num: u64 = 0;
        let mut consecutive_empty: u32 = 0;
        let mut chunks_since_decoder_reset: u64 = 0;
        let mut speech_start_uptime_ms: Option<i64> = None;

        loop {
            if stop_flag.load(Ordering::Relaxed) {
                if let Some(db) = db.as_ref().filter(|_| diag_enabled.load(Ordering::Relaxed)) {
                    let _ = db.execute(
                        "INSERT INTO events (session_id, uptime_ms, event_type, chunk_num)
                         VALUES (?1, ?2, 'shutdown', ?3)",
                        rusqlite::params![
                            session_id,
                            loop_start.elapsed().as_millis() as i64,
                            chunk_num as i64
                        ],
                    );
                }
                return Ok((model, vad));
            }

            // Check for hot-reloaded settings
            if let Ok(new_settings) = settings_rx.try_recv() {
                println!(
                    "[diag] Hot-reloading settings: punctuation_reset={}, empty_reset_threshold={}, vad_threshold_start={}, vad_threshold_end={}",
                    new_settings.punctuation_reset, new_settings.empty_reset_threshold,
                    new_settings.vad_threshold_start, new_settings.vad_threshold_end
                );
                punctuation_reset_enabled = new_settings.punctuation_reset;
                empty_reset_threshold = new_settings.empty_reset_threshold;
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
                thread::sleep(std::time::Duration::from_millis(10));
                continue;
            }

            let iter_start = Instant::now();
            let drain_count = drained.len();
            let drain_audio_ms = drain_count as f64 / input_rate as f64 * 1000.0;

            let resample_start = Instant::now();
            let (mut samples_16k, _resample_in, _resample_out, _resample_leftover) = if let Some(
                ref mut resampler,
            ) =
                resampler
            {
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
                            if let Some(db) =
                                db.as_ref().filter(|_| diag_enabled.load(Ordering::Relaxed))
                            {
                                let _ = db.execute(
                                    "INSERT INTO events (session_id, uptime_ms, event_type, error_msg)
                                         VALUES (?1, ?2, 'resample_error', ?3)",
                                    rusqlite::params![
                                        session_id,
                                        loop_start.elapsed().as_millis() as i64,
                                        e.to_string()
                                    ],
                                );
                            }
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
                        if let Some(db) =
                            db.as_ref().filter(|_| diag_enabled.load(Ordering::Relaxed))
                        {
                            let _ = db.execute(
                                "INSERT INTO events (session_id, uptime_ms, event_type, error_msg)
                                 VALUES (?1, ?2, 'vad_error', ?3)",
                                rusqlite::params![
                                    session_id,
                                    loop_start.elapsed().as_millis() as i64,
                                    e.to_string()
                                ],
                            );
                        }
                        continue;
                    }
                };

                match decision {
                    VadDecision::Silence => {
                        // Audio is in the ring buffer; nothing to do
                    }
                    VadDecision::SpeechStarted { pre_speech_samples } => {
                        let uptime = loop_start.elapsed().as_millis() as i64;
                        speech_start_uptime_ms = Some(uptime);
                        consecutive_empty = 0;

                        if let Some(db) =
                            db.as_ref().filter(|_| diag_enabled.load(Ordering::Relaxed))
                        {
                            let _ = db.execute(
                                "INSERT INTO vad_events (session_id, uptime_ms, event_type, pre_speech_samples)
                                 VALUES (?1, ?2, 'speech_start', ?3)",
                                rusqlite::params![session_id, uptime, pre_speech_samples.len() as i64],
                            );
                        }

                        // Prepend ring buffer contents then this frame
                        asr_buffer.extend_from_slice(&pre_speech_samples);
                        asr_buffer.extend_from_slice(frame);
                    }
                    VadDecision::SpeechContinues => {
                        asr_buffer.extend_from_slice(frame);
                    }
                    VadDecision::SpeechEnded => {
                        asr_buffer.extend_from_slice(frame);

                        let uptime = loop_start.elapsed().as_millis() as i64;
                        let duration_ms = speech_start_uptime_ms
                            .map(|start| (uptime - start) as f64)
                            .unwrap_or(0.0);

                        if let Some(db) =
                            db.as_ref().filter(|_| diag_enabled.load(Ordering::Relaxed))
                        {
                            let chunks_at_reset = chunks_since_decoder_reset as i64;
                            let _ = db.execute(
                                "INSERT INTO vad_events (
                                    session_id, uptime_ms, event_type, speech_duration_ms,
                                    consecutive_empty, chunks_since_decoder_reset,
                                    audio_ms_since_decoder_reset
                                 )
                                 VALUES (?1, ?2, 'speech_end', ?3, ?4, ?5, ?6)",
                                rusqlite::params![
                                    session_id,
                                    uptime,
                                    duration_ms,
                                    consecutive_empty as i64,
                                    chunks_at_reset,
                                    chunks_to_audio_ms(
                                        chunks_since_decoder_reset,
                                        settings.chunk_ms
                                    ),
                                ],
                            );
                        }

                        // Flush remaining asr_buffer: pad final sub-chunk if needed
                        if !asr_buffer.is_empty() && asr_buffer.len() < chunk_size {
                            asr_buffer.resize(chunk_size, 0.0);
                        }

                        speech_start_uptime_ms = None;
                        consecutive_empty = 0;
                        model.reset();
                        chunks_since_decoder_reset = 0;
                    }
                }
            }

            let vad_ms = vad_start.elapsed().as_millis() as i64;

            // Save leftover sub-frame samples for next iteration
            if offset < vad_input.len() {
                vad_leftover = vad_input[offset..].to_vec();
            }

            // --- ASR transcription (only runs when asr_buffer has data, i.e., during speech) ---
            let vad_state_str = match vad.state() {
                VadState::Silence => "silence",
                VadState::Speech => "speech",
            };

            let mut asr_consumed = 0;
            while asr_buffer.len().saturating_sub(asr_consumed) >= chunk_size {
                let chunk = &asr_buffer[asr_consumed..asr_consumed + chunk_size];
                asr_consumed += chunk_size;
                chunks_since_decoder_reset += 1;
                let infer_start = Instant::now();
                match model.transcribe_chunk(chunk) {
                    Ok(text) => {
                        let infer_ms = infer_start.elapsed().as_millis() as i64;
                        chunk_num += 1;
                        let is_empty = text.is_empty();
                        let preview = text_preview(&text, 200);

                        if is_empty {
                            consecutive_empty += 1;
                        } else {
                            consecutive_empty = 0;
                        }

                        let iteration_ms = iter_start.elapsed().as_millis() as i64;
                        if let Some(db) =
                            db.as_ref().filter(|_| diag_enabled.load(Ordering::Relaxed))
                        {
                            let _ = db.execute(
                                "INSERT INTO events (session_id, uptime_ms, event_type, chunk_num,
                                 inference_ms, drain_samples, drain_audio_ms,
                                 asr_buf_len, text_empty, text_preview, vad_state,
                                 vad_ms, resample_ms, iteration_ms)
                                 VALUES (?1, ?2, 'transcribe', ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
                                rusqlite::params![
                                    session_id,
                                    loop_start.elapsed().as_millis() as i64,
                                    chunk_num as i64,
                                    infer_ms,
                                    drain_count as i64,
                                    drain_audio_ms,
                                    asr_buffer.len().saturating_sub(asr_consumed) as i64,
                                    is_empty as i64,
                                    preview,
                                    vad_state_str,
                                    vad_ms,
                                    resample_ms,
                                    iteration_ms,
                                ],
                            );
                        }

                        // Punctuation-based decoder reset
                        if punctuation_reset_enabled
                            && !is_empty
                            && ends_with_sentence_punctuation(&text)
                        {
                            if let Some(db) =
                                db.as_ref().filter(|_| diag_enabled.load(Ordering::Relaxed))
                            {
                                let uptime = loop_start.elapsed().as_millis() as i64;
                                let chunks_at_reset = chunks_since_decoder_reset as i64;
                                let _ = db.execute(
                                    "INSERT INTO vad_events (
                                        session_id, uptime_ms, event_type, consecutive_empty,
                                        chunks_since_decoder_reset, audio_ms_since_decoder_reset
                                     )
                                     VALUES (?1, ?2, 'punctuation_reset', ?3, ?4, ?5)",
                                    rusqlite::params![
                                        session_id,
                                        uptime,
                                        consecutive_empty as i64,
                                        chunks_at_reset,
                                        chunks_to_audio_ms(
                                            chunks_since_decoder_reset,
                                            settings.chunk_ms
                                        ),
                                    ],
                                );
                            }
                            model.reset();
                            chunks_since_decoder_reset = 0;
                            consecutive_empty = 0;
                        }

                        if !is_empty {
                            event_sink.on_transcription(text);
                        }

                        // Mid-speech reset heuristic
                        if consecutive_empty >= empty_reset_threshold
                            && vad.state() == VadState::Speech
                        {
                            if let Some(db) =
                                db.as_ref().filter(|_| diag_enabled.load(Ordering::Relaxed))
                            {
                                let uptime = loop_start.elapsed().as_millis() as i64;
                                let chunks_at_reset = chunks_since_decoder_reset as i64;
                                let _ = db.execute(
                                    "INSERT INTO vad_events (
                                        session_id, uptime_ms, event_type, consecutive_empty,
                                        chunks_since_decoder_reset, audio_ms_since_decoder_reset
                                     )
                                     VALUES (?1, ?2, 'mid_speech_reset', ?3, ?4, ?5)",
                                    rusqlite::params![
                                        session_id,
                                        uptime,
                                        consecutive_empty as i64,
                                        chunks_at_reset,
                                        chunks_to_audio_ms(
                                            chunks_since_decoder_reset,
                                            settings.chunk_ms
                                        ),
                                    ],
                                );
                            }
                            model.reset();
                            chunks_since_decoder_reset = 0;
                            consecutive_empty = 0;
                        }
                    }
                    Err(e) => {
                        if let Some(db) =
                            db.as_ref().filter(|_| diag_enabled.load(Ordering::Relaxed))
                        {
                            let _ = db.execute(
                                "INSERT INTO events (session_id, uptime_ms, event_type, chunk_num,
                                 inference_ms, error_msg, vad_state)
                                 VALUES (?1, ?2, 'asr_error', ?3, ?4, ?5, ?6)",
                                rusqlite::params![
                                    session_id,
                                    loop_start.elapsed().as_millis() as i64,
                                    chunk_num as i64,
                                    infer_start.elapsed().as_millis() as i64,
                                    e.to_string(),
                                    vad_state_str,
                                ],
                            );
                        }
                    }
                }
            }
            if asr_consumed > 0 {
                asr_buffer.drain(..asr_consumed);
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

fn chunks_to_audio_ms(chunks: u64, chunk_ms: usize) -> i64 {
    chunks.saturating_mul(chunk_ms as u64).min(i64::MAX as u64) as i64
}

fn text_preview(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }

    let end = text
        .char_indices()
        .map(|(idx, _)| idx)
        .take_while(|&idx| idx <= max_bytes)
        .last()
        .unwrap_or(0);
    text[..end].to_string()
}

/// Check if text ends with sentence-ending punctuation (`.`, `?`, `!`),
/// filtering out ellipsis and decimal-looking patterns.
pub fn ends_with_sentence_punctuation(text: &str) -> bool {
    let trimmed = text.trim_end();
    if trimmed.is_empty() {
        return false;
    }
    match trimmed.as_bytes()[trimmed.len() - 1] {
        b'?' | b'!' => true,
        b'.' => {
            // Filter out ellipsis ("...")
            if trimmed.ends_with("...") {
                return false;
            }
            // Filter out decimal-looking patterns (digit before ".")
            let before_dot = &trimmed[..trimmed.len() - 1];
            let last_char = before_dot.trim_end().bytes().last();
            !matches!(last_char, Some(b'0'..=b'9'))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn period_is_sentence_punctuation() {
        assert!(ends_with_sentence_punctuation("Hello."));
    }

    #[test]
    fn question_mark_is_sentence_punctuation() {
        assert!(ends_with_sentence_punctuation("Hello?"));
    }

    #[test]
    fn exclamation_is_sentence_punctuation() {
        assert!(ends_with_sentence_punctuation("Hello!"));
    }

    #[test]
    fn ellipsis_is_not_sentence_punctuation() {
        assert!(!ends_with_sentence_punctuation("Hello..."));
    }

    #[test]
    fn digit_before_period_is_not_sentence_punctuation() {
        assert!(!ends_with_sentence_punctuation("3."));
        assert!(!ends_with_sentence_punctuation("The value is 3.14."));
    }

    #[test]
    fn word_before_period_is_sentence_punctuation() {
        assert!(ends_with_sentence_punctuation("end."));
        assert!(ends_with_sentence_punctuation("The end."));
    }

    #[test]
    fn empty_string_is_not_sentence_punctuation() {
        assert!(!ends_with_sentence_punctuation(""));
    }

    #[test]
    fn whitespace_only_is_not_sentence_punctuation() {
        assert!(!ends_with_sentence_punctuation("   "));
    }

    #[test]
    fn trailing_whitespace_is_trimmed() {
        assert!(ends_with_sentence_punctuation("Hello.  "));
        assert!(ends_with_sentence_punctuation("Hello?  "));
    }

    #[test]
    fn no_punctuation_is_not_sentence_ending() {
        assert!(!ends_with_sentence_punctuation("Hello"));
        assert!(!ends_with_sentence_punctuation("Hello,"));
        assert!(!ends_with_sentence_punctuation("Hello;"));
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

    #[test]
    fn text_preview_does_not_split_utf8() {
        assert_eq!(text_preview("abcédef", 4), "abc");
        assert_eq!(text_preview("abcédef", 5), "abcé");
    }

    #[test]
    fn full_text_punctuation_can_differ_from_preview() {
        let long_text = format!("{}.", "word ".repeat(60));
        let preview = text_preview(&long_text, 200);

        assert!(!ends_with_sentence_punctuation(&preview));
        assert!(ends_with_sentence_punctuation(&long_text));
    }

    #[test]
    fn chunks_to_audio_ms_uses_chunk_duration() {
        assert_eq!(chunks_to_audio_ms(6, 560), 3360);
    }
}
