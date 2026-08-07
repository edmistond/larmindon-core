//! Replay a WAV file through the real audio engine and print the transcript.
//!
//! This is the regression harness for the ASR pipeline: it drives the actual
//! `AudioEngine`, resampler, AGC, VAD and ASR backend via a fake capture
//! backend, so the whole shipped code path runs without audio hardware.
//!
//! ```sh
//! cargo run --release --example replay_wav -- fixture.wav
//! cargo run --release --example replay_wav -- fixture.wav --speed 2 --diag out.sqlite
//! ```
//!
//! Output is written to stdout as `[NN] <text>` lines (one per emission) plus a
//! `=== TRANSCRIPT ===` block holding the concatenation the UI would build.
//! Diff those between runs to detect a behaviour change.

use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;

use larmindon_core::asr::TranscriptUpdate;
use larmindon_core::audio_capture::{
    ActiveSessionInfo, AudioCapture, AudioDevice, AudioStream, AudioStreamMetadata, CaptureBuffer,
    DeviceType, StartedAudioStream,
};
use larmindon_core::audio_engine::{AudioEngine, Command};
use larmindon_core::settings::Settings;
use larmindon_core::EngineEventSink;

// ---------------------------------------------------------------------------
// Minimal 16-bit PCM WAV reader (avoids adding a dependency for one fixture).
// ---------------------------------------------------------------------------

struct Wav {
    samples: Vec<f32>,
    sample_rate: usize,
}

fn read_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

fn read_u16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn parse_wav(path: &Path) -> Result<Wav, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }

    let mut channels = 0usize;
    let mut sample_rate = 0usize;
    let mut bits = 0u16;
    let mut format = 0u16;
    let mut data: Option<&[u8]> = None;

    // Walk the chunk list. Chunks are 8-byte header + payload, word aligned.
    let mut pos = 12usize;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = read_u32(&bytes, pos + 4) as usize;
        let body = pos + 8;
        let end = (body + size).min(bytes.len());

        match id {
            b"fmt " if size >= 16 => {
                format = read_u16(&bytes, body);
                channels = read_u16(&bytes, body + 2) as usize;
                sample_rate = read_u32(&bytes, body + 4) as usize;
                bits = read_u16(&bytes, body + 14);
            }
            b"data" => data = Some(&bytes[body..end]),
            _ => {}
        }

        pos = body + size + (size & 1);
    }

    let data = data.ok_or("no data chunk")?;
    if format != 1 || bits != 16 {
        return Err(format!(
            "expected 16-bit PCM (format=1, bits=16), got format={format}, bits={bits}"
        )
        .into());
    }
    if channels == 0 || sample_rate == 0 {
        return Err("missing or invalid fmt chunk".into());
    }

    // Downmix to mono; the engine's pipeline is mono throughout.
    let frames = data.len() / 2 / channels;
    let mut samples = Vec::with_capacity(frames);
    for f in 0..frames {
        let mut acc = 0.0f32;
        for c in 0..channels {
            let at = (f * channels + c) * 2;
            acc += i16::from_le_bytes([data[at], data[at + 1]]) as f32 / 32768.0;
        }
        samples.push(acc / channels as f32);
    }

    Ok(Wav {
        samples,
        sample_rate,
    })
}

// ---------------------------------------------------------------------------
// Fake capture backend: feeds the WAV into the engine's capture buffer at a
// controlled rate, mimicking a real device callback.
// ---------------------------------------------------------------------------

struct FakeCapture {
    wav: Arc<Wav>,
    speed: f32,
    done: Arc<AtomicBool>,
    /// Gate so the warm-up session feeds nothing. A cold model load takes long
    /// enough to overflow the 10 s capture buffer and silently drop the head of
    /// the file, which changes the transcript; the warm-up pass populates the
    /// engine's model cache so the real pass starts instantly.
    release: Arc<AtomicBool>,
}

struct FakeStream {
    handle: Option<thread::JoinHandle<()>>,
}

impl AudioStream for FakeStream {
    fn stop(mut self: Box<Self>) {
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl AudioCapture for FakeCapture {
    fn enumerate_devices(&self) -> Result<Vec<AudioDevice>, Box<dyn Error>> {
        Ok(vec![AudioDevice {
            id: "fake".to_string(),
            name: "Replay WAV".to_string(),
            device_type: DeviceType::Input,
            is_default: true,
            application_name: None,
        }])
    }

    fn start(
        &self,
        _device_id: Option<String>,
        buffer: Arc<Mutex<CaptureBuffer>>,
        stop_flag: Arc<AtomicBool>,
    ) -> Result<StartedAudioStream, Box<dyn Error>> {
        let wav = Arc::clone(&self.wav);
        let done = Arc::clone(&self.done);
        let release = Arc::clone(&self.release);
        let speed = self.speed;
        let rate = wav.sample_rate;

        // 20 ms of audio per push, paced to `speed` x realtime.
        let block = (rate / 50).max(1);
        let interval = Duration::from_secs_f64(block as f64 / rate as f64 / speed as f64);

        let handle = thread::spawn(move || {
            if !release.load(Ordering::Relaxed) {
                done.store(true, Ordering::Relaxed);
                return;
            }
            let mut at = 0usize;
            while at < wav.samples.len() {
                if stop_flag.load(Ordering::Relaxed) {
                    break;
                }
                let end = (at + block).min(wav.samples.len());
                if let Ok(mut guard) = buffer.lock() {
                    guard.extend_samples(wav.samples[at..end].iter().copied());
                }
                at = end;
                thread::sleep(interval);
            }
            done.store(true, Ordering::Relaxed);
        });

        Ok(StartedAudioStream {
            stream: Box::new(FakeStream {
                handle: Some(handle),
            }),
            metadata: AudioStreamMetadata {
                sample_rate: rate,
                channels: 1,
                sample_format: "f32".to_string(),
            },
        })
    }

    fn name(&self) -> &'static str {
        "replay-wav"
    }
}

// ---------------------------------------------------------------------------
// Collecting sink
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct CollectSink {
    lines: Arc<Mutex<Vec<String>>>,
    updates: Arc<Mutex<Vec<String>>>,
    errors: Arc<Mutex<Vec<String>>>,
}

impl EngineEventSink for CollectSink {
    fn on_transcript_update(&self, update: TranscriptUpdate) {
        let speaker = update.speaker.as_ref().map(|s| s.0.as_str()).unwrap_or("-");
        self.lines.lock().unwrap().push(update.text.clone());
        self.updates.lock().unwrap().push(format!(
            "id={} final={} speaker={} text={:?}",
            update.segment_id, update.is_final, speaker, update.text
        ));
    }

    fn on_error(&self, message: String) {
        self.errors.lock().unwrap().push(message);
    }

    fn on_source_switched(&self, _device_id: String) {}

    fn on_devices_changed(&self, _devices: Vec<AudioDevice>) {}
}

// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args[0].starts_with("--") {
        eprintln!("usage: replay_wav <file.wav> [--speed N] [--model PATH] [--diag PATH]");
        std::process::exit(2);
    }

    let wav_path = args[0].clone();
    let mut speed = 1.0f32;
    let mut model_override: Option<String> = None;
    let mut diag_path: Option<String> = None;
    let mut empty_reset: Option<u32> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--speed" => {
                speed = args.get(i + 1).ok_or("--speed needs a value")?.parse()?;
                i += 2;
            }
            "--model" => {
                model_override = Some(args.get(i + 1).ok_or("--model needs a value")?.clone());
                i += 2;
            }
            "--diag" => {
                diag_path = Some(args.get(i + 1).ok_or("--diag needs a value")?.clone());
                i += 2;
            }
            // Lowering this makes the mid-speech stuck-decoder reset (and its
            // replay path) fire on ordinary speech, which is otherwise rare.
            "--empty-reset" => {
                empty_reset = Some(
                    args.get(i + 1)
                        .ok_or("--empty-reset needs a value")?
                        .parse()?,
                );
                i += 2;
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    let wav = parse_wav(Path::new(&wav_path))?;
    eprintln!(
        "[replay] {} — {} samples @ {} Hz ({:.1}s), feeding at {}x",
        wav_path,
        wav.samples.len(),
        wav.sample_rate,
        wav.samples.len() as f64 / wav.sample_rate as f64,
        speed
    );

    // Start from saved settings so the model path matches the real app, then
    // pin everything that affects DSP so runs are comparable.
    let mut settings = Settings::load();
    if let Some(m) = model_override {
        settings.model_path = m;
    }
    if let Some(t) = empty_reset {
        settings.empty_reset_threshold = t;
    }
    settings.diagnostics_enabled = diag_path.is_some();
    if let Some(p) = diag_path {
        settings.diagnostics_db_path = p;
    }

    let done = Arc::new(AtomicBool::new(false));
    let release = Arc::new(AtomicBool::new(false));
    let capture = FakeCapture {
        wav: Arc::new(wav),
        speed,
        done: Arc::clone(&done),
        release: Arc::clone(&release),
    };

    let sink = CollectSink {
        lines: Arc::new(Mutex::new(Vec::new())),
        updates: Arc::new(Mutex::new(Vec::new())),
        errors: Arc::new(Mutex::new(Vec::new())),
    };

    let (cmd_tx, cmd_rx) = mpsc::channel();
    let diag_enabled = Arc::new(AtomicBool::new(settings.diagnostics_enabled));
    let engine = AudioEngine::new(
        sink.clone(),
        cmd_rx,
        Box::new(capture),
        Arc::new(Mutex::new(ActiveSessionInfo::default())),
        Arc::clone(&diag_enabled),
    );
    let engine_thread = thread::spawn(move || engine.run());

    // Warm-up pass: loads the model into the engine's cache while the feeder is
    // gated off. Without this, a cold load overflows the capture buffer.
    eprintln!("[replay] warm-up pass (loading model)...");
    cmd_tx.send(Command::Start {
        device_id: Some("fake".to_string()),
        settings: settings.clone(),
    })?;
    while !done.load(Ordering::Relaxed) {
        thread::sleep(Duration::from_millis(50));
    }
    cmd_tx.send(Command::Stop)?;

    // Real pass: model comes from cache, so capture starts being consumed
    // immediately and no samples are dropped.
    done.store(false, Ordering::Relaxed);
    release.store(true, Ordering::Relaxed);
    eprintln!("[replay] replaying...");
    cmd_tx.send(Command::Start {
        device_id: Some("fake".to_string()),
        settings,
    })?;

    // Wait for the feeder to exhaust the file, then let the loop drain.
    while !done.load(Ordering::Relaxed) {
        thread::sleep(Duration::from_millis(50));
    }
    thread::sleep(Duration::from_millis(1500));

    cmd_tx.send(Command::Stop)?;
    drop(cmd_tx);
    let _ = engine_thread.join();

    let lines = sink.lines.lock().unwrap().clone();
    let errors = sink.errors.lock().unwrap().clone();

    println!("=== EMISSIONS ({}) ===", lines.len());
    for (n, line) in lines.iter().enumerate() {
        println!("[{n:03}] {line}");
    }
    println!("=== TRANSCRIPT ===");
    println!("{}", lines.concat());
    let updates = sink.updates.lock().unwrap().clone();
    println!("=== SEGMENTS ({}) ===", updates.len());
    for u in &updates {
        println!("{u}");
    }
    if !errors.is_empty() {
        println!("=== ERRORS ({}) ===", errors.len());
        for e in &errors {
            println!("{e}");
        }
    }

    Ok(())
}
