//! Windows per-application audio capture via WASAPI process loopback.
//!
//! Enumerates processes that currently have an active audio render session
//! (apps that are playing audio) and captures a chosen process's render stream
//! using `wasapi-rs`'s process-loopback client (Windows 10 build 19041+).
//!
//! Device IDs produced here use the `proc:<pid>` prefix so the composite backend
//! ([`super::windows_composite`]) can route `start()` calls to this backend while
//! cpal handles `input:` / `output:` devices.

use crate::audio_capture::{
    AudioCapture, AudioDevice, AudioStream, AudioStreamMetadata, CaptureBuffer, DeviceType,
    StartedAudioStream,
};
use std::collections::{HashSet, VecDeque};
use std::error::Error;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;

use wasapi::{Direction, SampleType, StreamMode, WaveFormat};
use windows::core::Interface;
use windows::core::PWSTR;
use windows::Win32::Foundation::CloseHandle;
use windows::Win32::Media::Audio::{
    eConsole, eRender, AudioSessionState, AudioSessionStateActive, IAudioSessionControl2,
    IAudioSessionManager2, IMMDeviceEnumerator, MMDeviceEnumerator,
};
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoUninitialize, CLSCTX_ALL, COINIT_MULTITHREADED,
};
use windows::Win32::System::Threading::{
    OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION,
};

pub const PROCESS_DEVICE_ID_PREFIX: &str = "proc:";

/// Capture format we request from WASAPI. With `autoconvert` enabled the audio
/// engine resamples/remixes the process stream into this format for us, so the
/// frames we read are always 48 kHz stereo f32 regardless of the app's output.
const CAPTURE_SAMPLE_RATE: usize = 48000;
const CAPTURE_CHANNELS: usize = 2;
const BYTES_PER_FRAME: usize = CAPTURE_CHANNELS * 4; // f32 stereo

pub struct WasapiProcessBackend;

pub fn create_backend() -> Box<dyn AudioCapture> {
    Box::new(WasapiProcessBackend)
}

impl AudioCapture for WasapiProcessBackend {
    fn enumerate_devices(&self) -> Result<Vec<AudioDevice>, Box<dyn Error>> {
        // COM objects are not `Send`, so do all the work on a dedicated thread
        // (with its own COM apartment) and only return the plain `AudioDevice`
        // values. Bound it with a timeout so a wedged audio service can't hang
        // device enumeration (mirrors the PipeWire backend).
        // `Box<dyn Error>` isn't `Send`, so stringify on the worker thread and
        // rebuild the error on this side.
        let (tx, rx) = mpsc::channel::<Result<Vec<AudioDevice>, String>>();
        thread::spawn(move || {
            let result = enumerate_sessions().map_err(|e| e.to_string());
            let _ = tx.send(result);
        });

        match rx.recv_timeout(Duration::from_secs(2)) {
            Ok(Ok(devices)) => Ok(devices),
            Ok(Err(e)) => Err(e.into()),
            Err(_) => {
                eprintln!("WASAPI: process enumeration timed out");
                Ok(Vec::new())
            }
        }
    }

    fn start(
        &self,
        device_id: Option<String>,
        buffer: Arc<Mutex<CaptureBuffer>>,
        stop_flag: Arc<AtomicBool>,
    ) -> Result<StartedAudioStream, Box<dyn Error>> {
        let id = device_id.ok_or("WASAPI process backend requires a device ID")?;
        let pid: u32 = id
            .strip_prefix(PROCESS_DEVICE_ID_PREFIX)
            .ok_or_else(|| format!("Not a process device ID: {}", id))?
            .parse()
            .map_err(|_| format!("Invalid process ID in device ID: {}", id))?;

        // The loopback client must be created and driven on a single thread with
        // an initialized COM apartment. Set up inside the thread and report the
        // setup result back so `start()` can surface failures (e.g. unsupported
        // Windows build, process already gone) to the caller.
        let (init_tx, init_rx) = mpsc::channel::<Result<(), String>>();
        let thread_buffer = Arc::clone(&buffer);
        let thread_stop = Arc::clone(&stop_flag);

        let handle = thread::spawn(move || {
            capture_thread(pid, thread_buffer, thread_stop, init_tx);
        });

        match init_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(())) => {}
            Ok(Err(msg)) => {
                let _ = handle.join();
                return Err(msg.into());
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                stop_flag.store(true, Ordering::Relaxed);
                return Err("WASAPI capture initialization timed out".into());
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let _ = handle.join();
                return Err("WASAPI capture thread exited during initialization".into());
            }
        }

        Ok(StartedAudioStream {
            stream: Box::new(WasapiProcessStream {
                stop_flag,
                thread: Some(handle),
            }),
            metadata: AudioStreamMetadata {
                sample_rate: CAPTURE_SAMPLE_RATE,
                channels: 1, // we downmix to mono before pushing into the buffer
                sample_format: "F32".to_string(),
            },
        })
    }

    fn name(&self) -> &'static str {
        "wasapi-process"
    }
}

/// Capture-thread entry point: initialize COM + the loopback client, report the
/// setup result, then pump audio into `buffer` until `stop_flag` is set.
fn capture_thread(
    pid: u32,
    buffer: Arc<Mutex<CaptureBuffer>>,
    stop_flag: Arc<AtomicBool>,
    init_tx: mpsc::Sender<Result<(), String>>,
) {
    // MTA init for this thread; ignore "already initialized" outcomes.
    let _ = wasapi::initialize_mta();

    let setup = setup_capture(pid);
    let (client, capture_client, event_handle) = match setup {
        Ok(parts) => {
            let _ = init_tx.send(Ok(()));
            parts
        }
        Err(e) => {
            let _ = init_tx.send(Err(e.to_string()));
            return;
        }
    };

    if let Err(e) = pump_audio(&capture_client, &event_handle, &buffer, &stop_flag) {
        eprintln!("WASAPI: capture loop ended: {}", e);
    }

    let _ = client.stop_stream();
}

type CaptureParts = (
    wasapi::AudioClient,
    wasapi::AudioCaptureClient,
    wasapi::Handle,
);

fn setup_capture(pid: u32) -> Result<CaptureParts, Box<dyn Error>> {
    let mut client = wasapi::AudioClient::new_application_loopback_client(pid, true)?;
    let format = WaveFormat::new(
        32,
        32,
        &SampleType::Float,
        CAPTURE_SAMPLE_RATE,
        CAPTURE_CHANNELS,
        None,
    );
    let mode = StreamMode::EventsShared {
        autoconvert: true,
        buffer_duration_hns: 0,
    };
    client.initialize_client(&format, &Direction::Capture, &mode)?;
    let event_handle = client.set_get_eventhandle()?;
    let capture_client = client.get_audiocaptureclient()?;
    client.start_stream()?;
    Ok((client, capture_client, event_handle))
}

fn pump_audio(
    capture_client: &wasapi::AudioCaptureClient,
    event_handle: &wasapi::Handle,
    buffer: &Arc<Mutex<CaptureBuffer>>,
    stop_flag: &Arc<AtomicBool>,
) -> Result<(), Box<dyn Error>> {
    let mut queue: VecDeque<u8> = VecDeque::new();

    while !stop_flag.load(Ordering::Relaxed) {
        // Short timeout so we re-check the stop flag promptly even when the
        // process is silent (no events firing).
        if event_handle.wait_for_event(200).is_err() {
            continue;
        }

        // Drain all packets currently available, then downmix to mono.
        loop {
            match capture_client.get_next_packet_size()? {
                Some(frames) if frames > 0 => {
                    capture_client.read_from_device_to_deque(&mut queue)?;
                }
                _ => break,
            }
        }

        push_downmixed(&mut queue, buffer);
    }

    Ok(())
}

/// Drain whole stereo f32 frames from `queue`, downmix to mono, and push into the
/// shared capture buffer. Any trailing partial frame stays in `queue`.
fn push_downmixed(queue: &mut VecDeque<u8>, buffer: &Arc<Mutex<CaptureBuffer>>) {
    let full = (queue.len() / BYTES_PER_FRAME) * BYTES_PER_FRAME;
    if full == 0 {
        return;
    }
    let bytes: Vec<u8> = queue.drain(..full).collect();
    if let Ok(mut guard) = buffer.lock() {
        for frame in bytes.chunks_exact(BYTES_PER_FRAME) {
            let left = f32::from_le_bytes([frame[0], frame[1], frame[2], frame[3]]);
            let right = f32::from_le_bytes([frame[4], frame[5], frame[6], frame[7]]);
            guard.push_sample((left + right) * 0.5);
        }
    }
}

struct WasapiProcessStream {
    stop_flag: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl AudioStream for WasapiProcessStream {
    fn stop(mut self: Box<Self>) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

/// Enumerate processes with an active audio render session on the default render
/// endpoint. Returns one [`AudioDevice`] per distinct producing process.
fn enumerate_sessions() -> Result<Vec<AudioDevice>, Box<dyn Error>> {
    unsafe {
        // Fresh thread => fresh apartment. If this fails, the COM calls below
        // will surface the real error; only uninitialize when this call balanced
        // the apartment initialization count.
        let com_initialized = CoInitializeEx(None, COINIT_MULTITHREADED).is_ok();

        let result = (|| -> Result<Vec<AudioDevice>, Box<dyn Error>> {
            let enumerator: IMMDeviceEnumerator =
                CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;
            let device = enumerator.GetDefaultAudioEndpoint(eRender, eConsole)?;
            let manager: IAudioSessionManager2 = device.Activate(CLSCTX_ALL, None)?;
            let sessions = manager.GetSessionEnumerator()?;
            let count = sessions.GetCount()?;

            let mut seen: HashSet<u32> = HashSet::new();
            let mut devices = Vec::new();

            for i in 0..count {
                let control = match sessions.GetSession(i) {
                    Ok(c) => c,
                    Err(_) => continue,
                };

                // Only list sessions that are actively producing audio. WASAPI
                // keeps inactive sessions around, and offering them as capture
                // targets produces confusing duplicate/silent app entries.
                match control.GetState() {
                    Ok(state) if is_active_session(state) => {}
                    _ => continue,
                }

                let control2: IAudioSessionControl2 = match control.cast() {
                    Ok(c) => c,
                    Err(_) => continue,
                };

                // The system-sounds session reports pid 0, so the guard below
                // filters it out. (We deliberately don't call
                // IsSystemSoundsSession: it returns S_FALSE for normal sessions,
                // which windows-rs maps to Ok(()) just like S_OK — so it can't
                // distinguish the two without inspecting the raw HRESULT.)
                let pid = match control2.GetProcessId() {
                    Ok(p) if p != 0 => p,
                    _ => continue,
                };
                if !seen.insert(pid) {
                    continue;
                }

                let name = process_name(pid).unwrap_or_else(|| format!("PID {}", pid));
                let label = application_label(&name, pid);
                devices.push(AudioDevice {
                    id: format!("{}{}", PROCESS_DEVICE_ID_PREFIX, pid),
                    name: format!("[app] {}", label),
                    device_type: DeviceType::Application,
                    is_default: false,
                    application_name: Some(label),
                });
            }

            println!(
                "WASAPI: {} audio session(s) on default render endpoint, {} app source(s) listed",
                count,
                devices.len()
            );
            Ok(devices)
        })();

        if com_initialized {
            CoUninitialize();
        }
        result
    }
}

fn is_active_session(state: AudioSessionState) -> bool {
    state == AudioSessionStateActive
}

fn application_label(name: &str, pid: u32) -> String {
    if name == format!("PID {}", pid) {
        name.to_string()
    } else {
        format!("{} (PID {})", name, pid)
    }
}

/// Resolve a process's executable base name (e.g. `firefox.exe`) from its PID.
fn process_name(pid: u32) -> Option<String> {
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid).ok()?;
        let result = [260usize, 1024, 32768].into_iter().find_map(|capacity| {
            let mut buf = vec![0u16; capacity];
            let mut size = buf.len() as u32;
            QueryFullProcessImageNameW(
                handle,
                PROCESS_NAME_WIN32,
                PWSTR(buf.as_mut_ptr()),
                &mut size,
            )
            .ok()?;

            let full = String::from_utf16_lossy(&buf[..size as usize]);
            let base = full
                .rsplit(|c| c == '\\' || c == '/')
                .next()
                .unwrap_or(&full);
            if base.is_empty() {
                None
            } else {
                Some(base.to_string())
            }
        });
        let _ = CloseHandle(handle);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Media::Audio::{AudioSessionStateExpired, AudioSessionStateInactive};

    #[test]
    fn only_active_sessions_are_listed() {
        assert!(is_active_session(AudioSessionStateActive));
        assert!(!is_active_session(AudioSessionStateInactive));
        assert!(!is_active_session(AudioSessionStateExpired));
    }

    #[test]
    fn application_label_includes_pid_for_duplicate_process_names() {
        assert_eq!(
            application_label("firefox.exe", 1234),
            "firefox.exe (PID 1234)"
        );
    }

    #[test]
    fn application_label_does_not_repeat_pid_fallback() {
        assert_eq!(application_label("PID 1234", 1234), "PID 1234");
    }
}
