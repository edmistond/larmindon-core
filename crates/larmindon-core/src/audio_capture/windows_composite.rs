//! Composite Windows backend: presents cpal devices (inputs / system-audio
//! monitors) together with per-application WASAPI process-loopback sources in a
//! single device list, and routes `start()` to the right backend by device-ID
//! prefix (`proc:` => WASAPI process loopback, everything else => cpal).
//!
//! Only one source is captured at a time, so this keeps the engine's existing
//! single-stream model unchanged — the composite is transparent to it.

use crate::audio_capture::wasapi_process::{self, PROCESS_DEVICE_ID_PREFIX};
use crate::audio_capture::{
    cpal, sort_devices_by_priority, AudioCapture, AudioDevice, CaptureBuffer, StartedAudioStream,
};
use std::error::Error;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

pub struct WindowsCompositeBackend {
    cpal: Box<dyn AudioCapture>,
    wasapi: Box<dyn AudioCapture>,
}

pub fn create_backend() -> Box<dyn AudioCapture> {
    Box::new(WindowsCompositeBackend {
        cpal: cpal::create_backend(),
        wasapi: wasapi_process::create_backend(),
    })
}

impl AudioCapture for WindowsCompositeBackend {
    fn enumerate_devices(&self) -> Result<Vec<AudioDevice>, Box<dyn Error>> {
        // cpal devices are the baseline and must always be available; treat
        // process enumeration as best-effort so a WASAPI hiccup never hides the
        // physical inputs / monitors.
        let mut devices = self.cpal.enumerate_devices()?;

        match self.wasapi.enumerate_devices() {
            Ok(apps) => devices.extend(apps),
            Err(e) => eprintln!("WASAPI: process enumeration failed: {}", e),
        }

        Ok(sort_devices_by_priority(devices))
    }

    fn start(
        &self,
        device_id: Option<String>,
        buffer: Arc<Mutex<CaptureBuffer>>,
        stop_flag: Arc<AtomicBool>,
    ) -> Result<StartedAudioStream, Box<dyn Error>> {
        let is_process = device_id
            .as_deref()
            .is_some_and(|id| id.starts_with(PROCESS_DEVICE_ID_PREFIX));

        if is_process {
            self.wasapi.start(device_id, buffer, stop_flag)
        } else {
            self.cpal.start(device_id, buffer, stop_flag)
        }
    }

    fn name(&self) -> &'static str {
        "windows-composite"
    }
}
