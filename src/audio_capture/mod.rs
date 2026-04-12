use serde::Serialize;
use std::any::Any;
use std::collections::VecDeque;
use std::error::Error;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

/// Device type for UI organization
#[derive(Serialize, Clone, Debug, PartialEq)]
pub enum DeviceType {
    /// Per-application audio capture (highest priority, PipeWire only)
    #[allow(dead_code)]
    Application,
    /// Physical input devices (microphones)
    Input,
    /// System audio monitors
    Monitor,
}

/// Audio device information
#[derive(Serialize, Clone, Debug)]
pub struct AudioDevice {
    pub id: String,
    pub name: String,
    pub device_type: DeviceType,
    pub is_default: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub application_name: Option<String>,
}

/// Shared state describing the currently active capture session.
/// Written by AudioEngine, read by the PipeWire watcher for reconnect logic.
#[derive(Default)]
pub struct ActiveSessionInfo {
    pub device_id: Option<String>,
    pub application_name: Option<String>,
    pub device_type: Option<DeviceType>,
}

/// Trait for audio capture backends
pub trait AudioCapture: Send {
    /// Enumerate available audio devices
    fn enumerate_devices(&self) -> Result<Vec<AudioDevice>, Box<dyn Error>>;

    /// Start capturing audio from a device
    ///
    /// # Arguments
    /// * `device_id` - ID of the device to capture from, or None for default
    /// * `buffer` - Shared buffer to push audio samples into
    /// * `stop_flag` - Atomic flag to signal capture should stop
    fn start(
        &self,
        device_id: Option<String>,
        buffer: Arc<Mutex<VecDeque<f32>>>,
        stop_flag: Arc<AtomicBool>,
    ) -> Result<Box<dyn AudioStream>, Box<dyn Error>>;

    /// Get the name of this backend
    fn name(&self) -> &'static str;

    /// Downcast support for backend-specific features (e.g., PipeWire watcher)
    #[allow(dead_code)]
    fn as_any(&self) -> Option<&dyn Any> {
        None
    }
}

/// Trait for active audio streams
pub trait AudioStream: Send {
    /// Stop the stream and clean up resources
    fn stop(self: Box<Self>);
}

/// Select a default device from the list
/// Priority: Monitor > Input > Application
pub fn select_default_device(devices: &[AudioDevice]) -> Option<String> {
    // Priority 1: The system default monitor
    devices
        .iter()
        .find(|d| d.device_type == DeviceType::Monitor && d.is_default)
        .map(|d| d.id.clone())
        // Priority 2: Any monitor
        .or_else(|| {
            devices
                .iter()
                .find(|d| d.device_type == DeviceType::Monitor)
                .map(|d| d.id.clone())
        })
        // Priority 3: Any input device
        .or_else(|| {
            devices
                .iter()
                .find(|d| d.device_type == DeviceType::Input)
                .map(|d| d.id.clone())
        })
}

/// Sort devices by priority: Applications first, then Inputs, then Monitors
pub fn sort_devices_by_priority(mut devices: Vec<AudioDevice>) -> Vec<AudioDevice> {
    devices.sort_by(|a, b| {
        use DeviceType::*;
        let priority = |t: &DeviceType| match t {
            Application => 0,
            Input => 1,
            Monitor => 2,
        };
        priority(&a.device_type).cmp(&priority(&b.device_type))
    });
    devices
}

#[cfg(all(target_os = "linux", feature = "pipewire"))]
pub mod pipewire;

#[cfg(feature = "cpal")]
pub mod cpal;

#[cfg(test)]
mod tests {
    use super::*;

    fn make_device(id: &str, device_type: DeviceType, is_default: bool) -> AudioDevice {
        AudioDevice {
            id: id.to_string(),
            name: id.to_string(),
            device_type,
            is_default,
            application_name: None,
        }
    }

    // -----------------------------------------------------------------------
    // select_default_device tests
    // -----------------------------------------------------------------------

    #[test]
    fn select_default_prefers_default_monitor() {
        let devices = vec![
            make_device("input1", DeviceType::Input, true),
            make_device("monitor1", DeviceType::Monitor, false),
            make_device("monitor2", DeviceType::Monitor, true),
        ];
        assert_eq!(select_default_device(&devices), Some("monitor2".into()));
    }

    #[test]
    fn select_default_falls_back_to_any_monitor() {
        let devices = vec![
            make_device("input1", DeviceType::Input, true),
            make_device("monitor1", DeviceType::Monitor, false),
        ];
        assert_eq!(select_default_device(&devices), Some("monitor1".into()));
    }

    #[test]
    fn select_default_falls_back_to_input() {
        let devices = vec![
            make_device("input1", DeviceType::Input, false),
            make_device("app1", DeviceType::Application, false),
        ];
        assert_eq!(select_default_device(&devices), Some("input1".into()));
    }

    #[test]
    fn select_default_returns_none_for_only_apps() {
        let devices = vec![make_device("app1", DeviceType::Application, false)];
        assert_eq!(select_default_device(&devices), None);
    }

    #[test]
    fn select_default_returns_none_for_empty_list() {
        assert_eq!(select_default_device(&[]), None);
    }

    // -----------------------------------------------------------------------
    // sort_devices_by_priority tests
    // -----------------------------------------------------------------------

    #[test]
    fn sort_orders_application_input_monitor() {
        let devices = vec![
            make_device("monitor1", DeviceType::Monitor, false),
            make_device("input1", DeviceType::Input, false),
            make_device("app1", DeviceType::Application, false),
        ];
        let sorted = sort_devices_by_priority(devices);
        assert_eq!(sorted[0].id, "app1");
        assert_eq!(sorted[1].id, "input1");
        assert_eq!(sorted[2].id, "monitor1");
    }

    #[test]
    fn sort_preserves_order_within_same_type() {
        let devices = vec![
            make_device("input2", DeviceType::Input, false),
            make_device("input1", DeviceType::Input, true),
        ];
        let sorted = sort_devices_by_priority(devices);
        // Both are Input, original order preserved (stable sort)
        assert_eq!(sorted[0].id, "input2");
        assert_eq!(sorted[1].id, "input1");
    }

    #[test]
    fn sort_empty_list() {
        let sorted = sort_devices_by_priority(vec![]);
        assert!(sorted.is_empty());
    }
}
