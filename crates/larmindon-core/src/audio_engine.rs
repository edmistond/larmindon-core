use rubato::{FftFixedIn, Resampler};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use crate::agc::AgcProcessor;
use crate::audio_capture::{
    self, ActiveSessionInfo, AudioCapture, AudioDevice, AudioStream, CaptureBuffer,
};
use crate::diagnostics::DiagSink;
use crate::engine::registry::EngineRegistry;
use crate::engine::tracker::SegmentTracker;
use crate::engine::{EngineError, SegmentUpdate, SessionContext, SpeechEngine};
use crate::settings::{self, Settings};
use crate::vad::{VadDecision, VadProcessor};
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

/// An engine instance kept alive across sessions so its loaded model can be
/// reused. Reused only when the engine id and the factory's cache key (model
/// path, thread counts, ...) still match.
struct CachedEngine {
    engine_id: String,
    cache_key: u64,
    engine: Box<dyn SpeechEngine>,
}

pub struct AudioEngine<E: EngineEventSink> {
    event_sink: E,
    cmd_rx: mpsc::Receiver<Command>,
    capture_backend: Box<dyn AudioCapture>,
    registry: Arc<EngineRegistry>,
    // Active session state
    active_stream: Option<Box<dyn AudioStream>>,
    #[allow(clippy::type_complexity)]
    processing_thread: Option<JoinHandle<Option<(Box<dyn SpeechEngine>, VadProcessor)>>>,
    stop_flag: Option<Arc<AtomicBool>>,
    capture_stop_flag: Option<Arc<AtomicBool>>,
    active_buffer: Option<Arc<Mutex<CaptureBuffer>>>,
    active_session_info: Arc<Mutex<ActiveSessionInfo>>,
    settings_tx: Option<mpsc::Sender<Settings>>,
    // Cached for reuse across sessions
    cached_engine: Option<CachedEngine>,
    cached_vad: Option<VadProcessor>,
    /// Identity (engine id, cache key) of the engine running in the active
    /// session, used to label the engine box returned when the thread joins.
    pending_cache_identity: Option<(String, u64)>,
    /// Global segment id allocator. Survives sessions and engine switches so
    /// the persistent frontend transcript never sees an id collide.
    next_segment_id: Arc<AtomicU64>,
    /// Runtime toggle for diagnostics logging. Shared with the active
    /// processing thread so flipping it off takes effect mid-session.
    diag_enabled: Arc<AtomicBool>,
}

impl<E: EngineEventSink> AudioEngine<E> {
    pub fn new(
        event_sink: E,
        cmd_rx: mpsc::Receiver<Command>,
        capture_backend: Box<dyn AudioCapture>,
        registry: Arc<EngineRegistry>,
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
            registry,
            active_stream: None,
            processing_thread: None,
            stop_flag: None,
            capture_stop_flag: None,
            active_buffer: None,
            active_session_info,
            settings_tx: None,
            cached_engine: None,
            cached_vad: None,
            pending_cache_identity: None,
            next_segment_id: Arc::new(AtomicU64::new(0)),
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
        let engine_id = settings.active_engine.clone();
        let factory = self.registry.get(&engine_id).ok_or_else(|| {
            let available: Vec<&str> = self.registry.descriptors().iter().map(|d| d.id).collect();
            format!(
                "Speech engine '{}' is not available in this build (available: {})",
                engine_id,
                available.join(", ")
            )
        })?;
        let engine_config = settings
            .engine_config(&engine_id)
            .cloned()
            .unwrap_or_else(|| factory.default_config());
        factory.validate_config(&engine_config)?;
        let cache_key = factory.cache_key(&engine_config);

        println!("Session starting with engine '{}'", engine_id);

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
        let event_sink_for_errors = self.event_sink.clone();

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

        // Reuse the cached engine instance (and its loaded model) when the
        // engine id and creation-relevant config still match.
        let engine = match self.cached_engine.take() {
            Some(cached) if cached.engine_id == engine_id && cached.cache_key == cache_key => {
                println!("Reusing cached '{}' engine (model stays loaded)", engine_id);
                cached.engine
            }
            Some(_) => {
                println!("Engine config changed — discarding cached engine");
                self.cached_vad = None;
                factory.create(&engine_config)?
            }
            None => factory.create(&engine_config)?,
        };
        let cached_vad = self.cached_vad.take();

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
        let engine_id_for_thread = engine_id.clone();
        let next_segment_id = Arc::clone(&self.next_segment_id);
        let processing_thread = thread::spawn(move || {
            println!("[diag] Processing thread started");
            match Self::processing_loop(
                event_sink,
                buffer_for_thread,
                stop_flag_thread,
                input_rate,
                needs_resample,
                settings,
                engine_id_for_thread,
                engine,
                engine_config,
                cached_vad,
                settings_rx,
                diag_db_path,
                diag_enabled_for_thread,
                next_segment_id,
            ) {
                Ok(models) => {
                    println!("[diag] Processing loop exited normally");
                    Some(models)
                }
                Err(e) => {
                    eprintln!("[diag] Processing loop CRASHED: {}", e);
                    event_sink_for_errors.on_error(format!("Error: {}", e));
                    None
                }
            }
        });

        self.active_stream = Some(stream.stream);
        self.processing_thread = Some(processing_thread);
        self.stop_flag = Some(stop_flag);
        self.capture_stop_flag = Some(capture_stop_flag);
        self.active_buffer = Some(buffer);

        // Update shared session info for the watcher
        if let Ok(mut info) = self.active_session_info.lock() {
            info.device_id = device_id;
            info.application_name = device_info
                .as_ref()
                .and_then(|d| d.application_name.clone());
            info.device_type = device_info.map(|d| d.device_type);
        }

        // Remember what the spawned session was created with so the engine box
        // it returns on stop can be matched against the next Start.
        self.cached_engine = None;
        self.pending_cache_identity = Some((engine_id, cache_key));

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
                Ok(Some((engine, vad))) => {
                    println!("[diag] Processing thread joined — caching engine for reuse");
                    if let Some((engine_id, cache_key)) = self.pending_cache_identity.take() {
                        self.cached_engine = Some(CachedEngine {
                            engine_id,
                            cache_key,
                            engine,
                        });
                    }
                    self.cached_vad = Some(vad);
                }
                Ok(None) => {
                    println!("[diag] Processing thread joined — nothing to cache (error path)");
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

    #[allow(clippy::too_many_arguments)]
    fn processing_loop(
        event_sink: E,
        buffer: Arc<Mutex<CaptureBuffer>>,
        stop_flag: Arc<AtomicBool>,
        input_rate: usize,
        needs_resample: bool,
        settings: Settings,
        engine_id: String,
        mut engine: Box<dyn SpeechEngine>,
        engine_config: serde_json::Value,
        cached_vad: Option<VadProcessor>,
        settings_rx: mpsc::Receiver<Settings>,
        diag_db_path: Option<std::path::PathBuf>,
        diag_enabled: Arc<AtomicBool>,
        next_segment_id: Arc<AtomicU64>,
    ) -> Result<(Box<dyn SpeechEngine>, VadProcessor), Box<dyn std::error::Error>> {
        let diag = match diag_db_path.as_deref() {
            Some(path) => DiagSink::open(
                path,
                Arc::clone(&diag_enabled),
                &engine_id,
                input_rate,
                needs_resample,
            )?,
            None => DiagSink::disabled(),
        };

        engine.begin_session(SessionContext { diag: diag.clone() }, &engine_config)?;

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

        // Forward engine results to the UI, remapping engine-local segment
        // ids to global ones. Mid-session engine errors are logged and the
        // session continues; only `begin_session` failures (and
        // infrastructure errors) abort the loop.
        let mut tracker = SegmentTracker::new(next_segment_id);
        let mut handle_engine_result =
            |result: Result<Vec<SegmentUpdate>, EngineError>| match result {
                Ok(updates) => {
                    for update in updates {
                        event_sink.on_segment_update(tracker.remap(update));
                    }
                }
                Err(e) => {
                    eprintln!("[diag] Engine error: {}", e);
                    diag.log_error("engine_error", &e.to_string());
                }
            };

        loop {
            if stop_flag.load(Ordering::Relaxed) {
                handle_engine_result(engine.end_session());
                diag.log_shutdown();
                return Ok((engine, vad));
            }

            // Check for hot-reloaded settings
            if let Ok(new_settings) = settings_rx.try_recv() {
                println!(
                    "[diag] Hot-reloading settings: vad_threshold_start={}, vad_threshold_end={}",
                    new_settings.vad_threshold_start, new_settings.vad_threshold_end
                );
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
                if let Some(config) = new_settings.engine_config(&engine_id) {
                    engine.update_config(config);
                }
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
                // Engines with asynchronous result delivery (callback threads,
                // sockets) surface results during silence via poll().
                handle_engine_result(engine.poll());
                thread::sleep(std::time::Duration::from_millis(10));
                continue;
            }

            let iter_start = Instant::now();
            let drain_count = drained.len();
            let drain_audio_ms = drain_count as f64 / input_rate as f64 * 1000.0;

            let resample_start = Instant::now();
            let mut samples_16k = if let Some(ref mut resampler) = resampler {
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
                            diag.log_error("resample_error", &e.to_string());
                        }
                    }
                    offset += rs_chunk;
                }
                resampled
            } else {
                drained
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
                        diag.log_error("vad_error", &e.to_string());
                        continue;
                    }
                };

                match decision {
                    VadDecision::Silence => {
                        // Audio is in the ring buffer; nothing to do
                    }
                    VadDecision::SpeechStarted { pre_speech_samples } => {
                        speech_start_uptime_ms = Some(loop_start.elapsed().as_millis() as i64);
                        diag.log_speech_start(pre_speech_samples.len());

                        engine.on_speech_start();
                        handle_engine_result(engine.feed(&pre_speech_samples));
                        handle_engine_result(engine.feed(frame));
                    }
                    VadDecision::SpeechContinues => {
                        handle_engine_result(engine.feed(frame));
                    }
                    VadDecision::SpeechEnded => {
                        handle_engine_result(engine.feed(frame));
                        handle_engine_result(engine.on_speech_end());

                        let uptime = loop_start.elapsed().as_millis() as i64;
                        let duration_ms = speech_start_uptime_ms
                            .map(|start| (uptime - start) as f64)
                            .unwrap_or(0.0);
                        diag.log_speech_end(duration_ms);
                        speech_start_uptime_ms = None;
                    }
                }
            }

            let vad_ms = vad_start.elapsed().as_millis() as i64;

            // Save leftover sub-frame samples for next iteration
            if offset < vad_input.len() {
                vad_leftover = vad_input[offset..].to_vec();
            }

            handle_engine_result(engine.poll());

            diag.log_feed(
                drain_count,
                drain_audio_ms,
                resample_ms,
                vad_ms,
                iter_start.elapsed().as_millis() as i64,
            );
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
