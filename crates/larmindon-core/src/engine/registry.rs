//! Engine discovery and construction.
//!
//! Engine crates provide an [`EngineFactory`]; the application shell registers
//! the factories it was compiled with into an [`EngineRegistry`]. The
//! [`EngineDescriptor`] carries enough metadata (including a typed config
//! field list) for the frontend to render a settings UI for any engine without
//! hardcoding it.

use std::sync::Arc;

use super::{EngineError, SpeechEngine};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineKind {
    Local,
    Cloud,
}

#[derive(Clone, serde::Serialize)]
pub struct EngineDescriptor {
    pub id: &'static str,
    /// Human-readable name shown in the engine selector.
    pub name: &'static str,
    pub kind: EngineKind,
    /// Whether this engine emits transient (revisable) segments. Engines with
    /// `false` only ever produce finalized segments.
    pub emits_partials: bool,
    pub config_fields: Vec<ConfigField>,
}

#[derive(Clone, serde::Serialize)]
pub struct ConfigField {
    /// Key in the engine's config blob.
    pub key: &'static str,
    pub label: &'static str,
    #[serde(flatten)]
    pub field: FieldType,
    pub default: serde_json::Value,
    /// Environment variable that overrides this field at startup.
    pub env_var: Option<&'static str>,
    pub help: Option<&'static str>,
}

#[derive(Clone, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FieldType {
    Bool,
    Int {
        min: i64,
        max: i64,
    },
    Float {
        min: f64,
        max: f64,
        step: f64,
    },
    Enum {
        options: Vec<EnumOption>,
    },
    Path {
        directory: bool,
    },
    Text,
    /// Sensitive value (API key); rendered as a password input.
    Secret,
}

#[derive(Clone, serde::Serialize)]
pub struct EnumOption {
    pub value: serde_json::Value,
    pub label: String,
}

pub trait EngineFactory: Send + Sync {
    fn descriptor(&self) -> EngineDescriptor;

    fn default_config(&self) -> serde_json::Value;

    fn validate_config(&self, config: &serde_json::Value) -> Result<(), String>;

    /// Hash of the config fields that require constructing a fresh engine
    /// (model path, thread counts) — NOT fields a reused instance can apply at
    /// `begin_session` or hot-reload. Drives the cross-session engine cache.
    fn cache_key(&self, config: &serde_json::Value) -> u64;

    /// Construct the engine. Must be cheap; heavy resources (models, network
    /// connections) load in `begin_session` on the processing thread.
    fn create(&self, config: &serde_json::Value) -> Result<Box<dyn SpeechEngine>, EngineError>;
}

#[derive(Default)]
pub struct EngineRegistry {
    factories: Vec<Arc<dyn EngineFactory>>,
}

impl EngineRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, factory: Arc<dyn EngineFactory>) {
        self.factories.push(factory);
    }

    pub fn get(&self, id: &str) -> Option<Arc<dyn EngineFactory>> {
        self.factories
            .iter()
            .find(|f| f.descriptor().id == id)
            .cloned()
    }

    pub fn descriptors(&self) -> Vec<EngineDescriptor> {
        self.factories.iter().map(|f| f.descriptor()).collect()
    }

    /// The first registered engine, used when no explicit selection exists.
    pub fn default_engine_id(&self) -> Option<&'static str> {
        self.factories.first().map(|f| f.descriptor().id)
    }
}
