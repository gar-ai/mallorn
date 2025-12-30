//! TensorRT build configuration types
//!
//! These types capture the TensorRT engine build settings needed to
//! rebuild an engine after applying an ONNX patch.

use serde::{Deserialize, Serialize};

/// TensorRT compute precision mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    /// Full 32-bit floating point
    #[default]
    FP32,
    /// Half precision (16-bit floating point)
    FP16,
    /// 8-bit integer quantization (requires calibration)
    INT8,
    /// TensorFloat-32 (NVIDIA Ampere+)
    TF32,
    /// Brain floating point (16-bit)
    BF16,
}

impl Precision {
    /// Convert to trtexec command-line flag
    pub fn to_trtexec_flag(&self) -> &'static str {
        match self {
            Precision::FP32 => "--best",
            Precision::FP16 => "--fp16",
            Precision::INT8 => "--int8",
            Precision::TF32 => "--tf32",
            Precision::BF16 => "--bf16",
        }
    }
}

/// TensorRT engine build configuration
///
/// Captures all settings needed to rebuild a TensorRT engine from ONNX.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorRTConfig {
    /// Compute precision mode
    pub precision: Precision,

    /// GPU workspace size in megabytes for layer algorithms
    pub workspace_size_mb: u32,

    /// Maximum batch size the engine should support
    pub max_batch_size: u32,

    /// DLA (Deep Learning Accelerator) core index for Jetson devices
    /// None means use GPU only
    pub dla_core: Option<i32>,

    /// Additional trtexec builder flags
    pub builder_flags: Vec<String>,

    /// Path to INT8 calibration cache file (required for INT8 precision)
    pub calibration_cache: Option<String>,

    /// Minimum TensorRT version required (e.g., "8.6.0")
    pub min_tensorrt_version: Option<String>,

    /// Target GPU compute capability (e.g., "8.6" for RTX 3080)
    pub compute_capability: Option<String>,

    /// Enable strict type constraints
    pub strict_types: bool,

    /// Enable sparsity for structured pruned models
    pub sparsity: bool,

    /// Optional description/notes
    pub description: Option<String>,
}

impl Default for TensorRTConfig {
    fn default() -> Self {
        Self {
            precision: Precision::FP32,
            workspace_size_mb: 1024,
            max_batch_size: 1,
            dla_core: None,
            builder_flags: Vec::new(),
            calibration_cache: None,
            min_tensorrt_version: None,
            compute_capability: None,
            strict_types: false,
            sparsity: false,
            description: None,
        }
    }
}

impl TensorRTConfig {
    /// Create a new config with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set compute precision
    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.precision = precision;
        self
    }

    /// Set workspace size in MB
    pub fn with_workspace_mb(mut self, size_mb: u32) -> Self {
        self.workspace_size_mb = size_mb;
        self
    }

    /// Set maximum batch size
    pub fn with_max_batch_size(mut self, batch_size: u32) -> Self {
        self.max_batch_size = batch_size;
        self
    }

    /// Set DLA core for Jetson devices
    pub fn with_dla_core(mut self, core: i32) -> Self {
        self.dla_core = Some(core);
        self
    }

    /// Add a builder flag
    pub fn with_builder_flag(mut self, flag: impl Into<String>) -> Self {
        self.builder_flags.push(flag.into());
        self
    }

    /// Set calibration cache path for INT8
    pub fn with_calibration_cache(mut self, path: impl Into<String>) -> Self {
        self.calibration_cache = Some(path.into());
        self
    }

    /// Set minimum TensorRT version requirement
    pub fn with_min_version(mut self, version: impl Into<String>) -> Self {
        self.min_tensorrt_version = Some(version.into());
        self
    }

    /// Set target compute capability
    pub fn with_compute_capability(mut self, capability: impl Into<String>) -> Self {
        self.compute_capability = Some(capability.into());
        self
    }

    /// Enable strict type constraints
    pub fn with_strict_types(mut self, enabled: bool) -> Self {
        self.strict_types = enabled;
        self
    }

    /// Enable sparsity support
    pub fn with_sparsity(mut self, enabled: bool) -> Self {
        self.sparsity = enabled;
        self
    }

    /// Set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Generate trtexec command for rebuilding the engine
    pub fn to_trtexec_command(&self, onnx_path: &str, engine_path: &str) -> String {
        let mut cmd = format!(
            "trtexec --onnx={} --saveEngine={} {}",
            onnx_path,
            engine_path,
            self.precision.to_trtexec_flag()
        );

        cmd.push_str(&format!(" --workspace={}", self.workspace_size_mb));

        if self.max_batch_size > 1 {
            cmd.push_str(&format!(
                " --minShapes=input:1x... --optShapes=input:{}x... --maxShapes=input:{}x...",
                self.max_batch_size, self.max_batch_size
            ));
        }

        if let Some(dla) = self.dla_core {
            cmd.push_str(&format!(" --useDLACore={} --allowGPUFallback", dla));
        }

        if let Some(ref cache) = self.calibration_cache {
            cmd.push_str(&format!(" --calib={}", cache));
        }

        if self.strict_types {
            cmd.push_str(" --strictTypes");
        }

        if self.sparsity {
            cmd.push_str(" --sparsity=enable");
        }

        for flag in &self.builder_flags {
            cmd.push_str(&format!(" {}", flag));
        }

        cmd
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TensorRTConfig::default();
        assert_eq!(config.precision, Precision::FP32);
        assert_eq!(config.workspace_size_mb, 1024);
        assert_eq!(config.max_batch_size, 1);
        assert!(config.dla_core.is_none());
    }

    #[test]
    fn test_config_builder() {
        let config = TensorRTConfig::new()
            .with_precision(Precision::FP16)
            .with_workspace_mb(2048)
            .with_max_batch_size(8)
            .with_dla_core(0)
            .with_strict_types(true);

        assert_eq!(config.precision, Precision::FP16);
        assert_eq!(config.workspace_size_mb, 2048);
        assert_eq!(config.max_batch_size, 8);
        assert_eq!(config.dla_core, Some(0));
        assert!(config.strict_types);
    }

    #[test]
    fn test_precision_flags() {
        assert_eq!(Precision::FP32.to_trtexec_flag(), "--best");
        assert_eq!(Precision::FP16.to_trtexec_flag(), "--fp16");
        assert_eq!(Precision::INT8.to_trtexec_flag(), "--int8");
        assert_eq!(Precision::TF32.to_trtexec_flag(), "--tf32");
        assert_eq!(Precision::BF16.to_trtexec_flag(), "--bf16");
    }

    #[test]
    fn test_trtexec_command_basic() {
        let config = TensorRTConfig::new().with_precision(Precision::FP16);
        let cmd = config.to_trtexec_command("model.onnx", "model.engine");

        assert!(cmd.contains("--onnx=model.onnx"));
        assert!(cmd.contains("--saveEngine=model.engine"));
        assert!(cmd.contains("--fp16"));
        assert!(cmd.contains("--workspace=1024"));
    }

    #[test]
    fn test_trtexec_command_dla() {
        let config = TensorRTConfig::new()
            .with_precision(Precision::FP16)
            .with_dla_core(0);
        let cmd = config.to_trtexec_command("model.onnx", "model.engine");

        assert!(cmd.contains("--useDLACore=0"));
        assert!(cmd.contains("--allowGPUFallback"));
    }

    #[test]
    fn test_config_serialization() {
        let config = TensorRTConfig::new()
            .with_precision(Precision::INT8)
            .with_calibration_cache("calib.cache")
            .with_description("Test config");

        let json = serde_json::to_string(&config).unwrap();
        let restored: TensorRTConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.precision, Precision::INT8);
        assert_eq!(restored.calibration_cache, Some("calib.cache".to_string()));
        assert_eq!(restored.description, Some("Test config".to_string()));
    }
}
