use crate::audio_capture::{
    AudioCapture, AudioDevice, AudioStream, AudioStreamMetadata, CaptureBuffer, DeviceType,
    StartedAudioStream,
};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig};
use std::error::Error;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

const INPUT_DEVICE_ID_PREFIX: &str = "input:";
const OUTPUT_DEVICE_ID_PREFIX: &str = "output:";

pub struct CpalBackend;

pub fn create_backend() -> Box<dyn AudioCapture> {
    Box::new(CpalBackend)
}

impl AudioCapture for CpalBackend {
    fn enumerate_devices(&self) -> Result<Vec<AudioDevice>, Box<dyn Error>> {
        let host = cpal::default_host();
        let mut devices = Vec::new();

        let default_input_name = host
            .default_input_device()
            .and_then(|d| d.description().ok().map(|desc| desc.name().to_string()));

        // Input devices
        if let Ok(input_devices) = host.input_devices() {
            for device in input_devices {
                let raw_name = device
                    .description()
                    .map(|d| d.name().to_string())
                    .map_err(|e| format!("Failed to get device name: {}", e))?;

                if !raw_name.is_empty() {
                    devices.push(AudioDevice {
                        id: cpal_device_id(DeviceType::Input, &raw_name),
                        name: format!("[in] {}", raw_name),
                        device_type: DeviceType::Input,
                        is_default: default_input_name.as_deref() == Some(&raw_name),
                        application_name: None,
                    });
                }
            }
        }

        // On Windows, WASAPI allows monitoring output devices as loopback inputs.
        // On macOS, CPAL's CoreAudio backend uses process taps for output capture.
        #[cfg(any(target_os = "windows", target_os = "macos"))]
        if let Ok(output_devices) = host.output_devices() {
            let default_output_name = host
                .default_output_device()
                .and_then(|d| d.description().ok().map(|desc| desc.name().to_string()));

            for device in output_devices {
                let raw_name = device
                    .description()
                    .map(|d| d.name().to_string())
                    .map_err(|e| format!("Failed to get device name: {}", e))?;

                if !raw_name.is_empty() {
                    devices.push(AudioDevice {
                        id: cpal_device_id(DeviceType::Monitor, &raw_name),
                        name: format!("[out] {}", raw_name),
                        device_type: DeviceType::Monitor,
                        is_default: cfg!(target_os = "macos")
                            && default_output_name.as_deref() == Some(&raw_name),
                        application_name: None,
                    });
                }
            }
        }

        Ok(devices)
    }

    fn start(
        &self,
        device_id: Option<String>,
        buffer: Arc<Mutex<CaptureBuffer>>,
        stop_flag: Arc<AtomicBool>,
    ) -> Result<StartedAudioStream, Box<dyn Error>> {
        let host = cpal::default_host();

        let (device, is_output_device) = if let Some(ref id) = device_id {
            match parse_cpal_device_id(id) {
                Some((DeviceType::Input, raw_name)) => {
                    let device = find_input_device(&host, raw_name)
                        .ok_or_else(|| format!("No input device found with ID: {}", id))?;
                    (device, false)
                }
                Some((DeviceType::Monitor, raw_name)) => {
                    let device = find_output_device(&host, raw_name)
                        .ok_or_else(|| format!("No output device found with ID: {}", id))?;
                    (device, true)
                }
                Some((DeviceType::Application, _)) => {
                    return Err(
                        format!("CPAL does not support application device ID: {}", id).into(),
                    );
                }
                None => {
                    // Backward compatibility for persisted selections from before CPAL
                    // device IDs included their direction.
                    if let Some(device) = find_input_device(&host, id) {
                        (device, false)
                    } else if let Some(device) = find_output_device(&host, id) {
                        (device, true)
                    } else {
                        return Err(format!("No device found with ID: {}", id).into());
                    }
                }
            }
        } else {
            (
                host.default_input_device()
                    .ok_or("No default input device found")?,
                false,
            )
        };
        #[cfg(not(target_os = "macos"))]
        let _ = is_output_device;

        let device_name = device
            .description()
            .map(|d| d.name().to_string())
            .map_err(|e| format!("Failed to get device name: {}", e))?;
        println!("CPAL: Using device: {}", device_name);

        let config = device
            .default_input_config()
            .or_else(|_| device.default_output_config())
            .map_err(|e| format!("Failed to get default config: {}", e))?;
        let sample_format = config.sample_format();
        let stream_config: StreamConfig = config.into();
        let channels = stream_config.channels as usize;

        println!(
            "CPAL: {} channels, {} Hz, {:?}",
            channels, stream_config.sample_rate, sample_format
        );

        let metadata = AudioStreamMetadata {
            sample_rate: stream_config.sample_rate as usize,
            channels,
            sample_format: format!("{:?}", sample_format),
        };
        let stream = match build_stream(&device, &stream_config, sample_format, channels, buffer) {
            Ok(stream) => stream,
            Err(err) => {
                #[cfg(target_os = "macos")]
                if is_output_device {
                    return Err(macos_system_audio_error(&device_name, err));
                }

                return Err(err);
            }
        };
        if let Err(err) = stream.play() {
            #[cfg(target_os = "macos")]
            if is_output_device {
                return Err(macos_system_audio_error(&device_name, err));
            }

            return Err(err.into());
        }

        Ok(StartedAudioStream {
            stream: Box::new(CpalStream { stream, stop_flag }),
            metadata,
        })
    }

    fn name(&self) -> &'static str {
        "CPAL"
    }
}

fn cpal_device_id(device_type: DeviceType, raw_name: &str) -> String {
    match device_type {
        DeviceType::Input => format!("{}{}", INPUT_DEVICE_ID_PREFIX, raw_name),
        DeviceType::Monitor => format!("{}{}", OUTPUT_DEVICE_ID_PREFIX, raw_name),
        DeviceType::Application => raw_name.to_string(),
    }
}

fn parse_cpal_device_id(id: &str) -> Option<(DeviceType, &str)> {
    if let Some(raw_name) = id.strip_prefix(INPUT_DEVICE_ID_PREFIX) {
        Some((DeviceType::Input, raw_name))
    } else if let Some(raw_name) = id.strip_prefix(OUTPUT_DEVICE_ID_PREFIX) {
        Some((DeviceType::Monitor, raw_name))
    } else {
        None
    }
}

fn find_input_device(host: &cpal::Host, raw_name: &str) -> Option<Device> {
    host.input_devices()
        .ok()?
        .find(|d| device_name_matches(d, raw_name))
}

fn find_output_device(host: &cpal::Host, raw_name: &str) -> Option<Device> {
    host.output_devices()
        .ok()?
        .find(|d| device_name_matches(d, raw_name))
}

fn device_name_matches(device: &Device, raw_name: &str) -> bool {
    device
        .description()
        .map(|desc| desc.name() == raw_name)
        .unwrap_or(false)
}

#[cfg(target_os = "macos")]
fn macos_system_audio_error(device_name: &str, err: impl std::fmt::Display) -> Box<dyn Error> {
    format!(
        "Failed to start macOS system audio capture for '{}': {}. Native output capture requires macOS 14.6 or newer and System Audio Recording permission.",
        device_name, err
    )
    .into()
}

struct CpalStream {
    #[allow(dead_code)] // kept alive to prevent stream drop
    stream: Stream,
    stop_flag: Arc<AtomicBool>,
}

impl AudioStream for CpalStream {
    fn stop(self: Box<Self>) {
        self.stop_flag.store(true, Ordering::Relaxed);
        // Stream is dropped when self is dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpal_device_ids_include_device_direction() {
        let raw_name = "L-Phonak hearing aid";

        assert_eq!(
            cpal_device_id(DeviceType::Input, raw_name),
            "input:L-Phonak hearing aid"
        );
        assert_eq!(
            cpal_device_id(DeviceType::Monitor, raw_name),
            "output:L-Phonak hearing aid"
        );
    }

    #[test]
    fn parse_cpal_device_id_preserves_raw_device_name() {
        assert_eq!(
            parse_cpal_device_id("input:L-Phonak hearing aid"),
            Some((DeviceType::Input, "L-Phonak hearing aid"))
        );
        assert_eq!(
            parse_cpal_device_id("output:L-Phonak hearing aid"),
            Some((DeviceType::Monitor, "L-Phonak hearing aid"))
        );
        assert_eq!(parse_cpal_device_id("L-Phonak hearing aid"), None);
    }
}

fn build_stream(
    device: &Device,
    config: &StreamConfig,
    sample_format: SampleFormat,
    channels: usize,
    buffer: Arc<Mutex<CaptureBuffer>>,
) -> Result<Stream, Box<dyn Error>> {
    let err_fn = |err| eprintln!("CPAL stream error: {}", err);

    let stream = match sample_format {
        SampleFormat::F32 => {
            let buf = Arc::clone(&buffer);
            device.build_input_stream(
                config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    push_mono_f32(data, channels, &buf);
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let buf = Arc::clone(&buffer);
            device.build_input_stream(
                config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    push_mono_convert(data, channels, &buf, |s| s as f32 / 32768.0);
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::U8 => {
            let buf = Arc::clone(&buffer);
            device.build_input_stream(
                config,
                move |data: &[u8], _: &cpal::InputCallbackInfo| {
                    push_mono_convert(data, channels, &buf, |s| (s as f32 - 128.0) / 128.0);
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I32 => {
            let buf = Arc::clone(&buffer);
            device.build_input_stream(
                config,
                move |data: &[i32], _: &cpal::InputCallbackInfo| {
                    push_mono_convert(data, channels, &buf, |s| s as f32 / 2147483648.0);
                },
                err_fn,
                None,
            )?
        }
        _ => return Err(format!("Unsupported sample format: {:?}", sample_format).into()),
    };

    Ok(stream)
}

/// Downmix interleaved multi-channel audio to mono and push into the shared buffer.
fn push_mono_f32(data: &[f32], channels: usize, buffer: &Arc<Mutex<CaptureBuffer>>) {
    if let Ok(mut guard) = buffer.lock() {
        if channels == 1 {
            guard.extend_samples(data.iter().copied());
        } else {
            for frame in data.chunks_exact(channels) {
                guard.push_sample(frame.iter().sum::<f32>() / channels as f32);
            }
        }
    }
}

fn push_mono_convert<T, F>(
    data: &[T],
    channels: usize,
    buffer: &Arc<Mutex<CaptureBuffer>>,
    convert: F,
) where
    T: Copy,
    F: Fn(T) -> f32,
{
    if let Ok(mut guard) = buffer.lock() {
        if channels == 1 {
            guard.extend_samples(data.iter().copied().map(convert));
        } else {
            for frame in data.chunks_exact(channels) {
                let mono = frame.iter().copied().map(&convert).sum::<f32>() / channels as f32;
                guard.push_sample(mono);
            }
        }
    }
}
