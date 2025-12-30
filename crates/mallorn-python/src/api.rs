//! Python API implementations

use mallorn_core::{CompressionMethod, DiffOptions as CoreDiffOptions, ModelFingerprint};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fs;
use std::path::Path;

/// Statistics from patch creation
#[pyclass]
#[derive(Clone)]
pub struct PatchStats {
    #[pyo3(get)]
    pub source_size: usize,
    #[pyo3(get)]
    pub target_size: usize,
    #[pyo3(get)]
    pub patch_size: usize,
    #[pyo3(get)]
    pub compression_ratio: f64,
    #[pyo3(get)]
    pub tensors_modified: usize,
    #[pyo3(get)]
    pub tensors_unchanged: usize,
}

#[pymethods]
impl PatchStats {
    fn __repr__(&self) -> String {
        format!(
            "PatchStats(source_size={}, target_size={}, patch_size={}, compression_ratio={:.1}x, tensors_modified={}, tensors_unchanged={})",
            self.source_size, self.target_size, self.patch_size,
            self.compression_ratio, self.tensors_modified, self.tensors_unchanged
        )
    }
}

/// Result of patch verification
#[pyclass]
#[derive(Clone)]
pub struct PatchVerification {
    #[pyo3(get)]
    pub source_valid: bool,
    #[pyo3(get)]
    pub patch_valid: bool,
    #[pyo3(get)]
    pub expected_target_hash: String,
    #[pyo3(get)]
    pub actual_target_hash: Option<String>,
    #[pyo3(get)]
    pub stats: PatchStats,
}

#[pymethods]
impl PatchVerification {
    fn __repr__(&self) -> String {
        format!(
            "PatchVerification(source_valid={}, patch_valid={}, expected_target_hash='{}')",
            self.source_valid, self.patch_valid, self.expected_target_hash
        )
    }

    /// Check if verification passed
    fn is_valid(&self) -> bool {
        self.source_valid && self.patch_valid
    }
}

/// Information about a patch file
#[pyclass]
#[derive(Clone)]
pub struct PatchInfo {
    #[pyo3(get)]
    pub format: String,
    #[pyo3(get)]
    pub version: u32,
    #[pyo3(get)]
    pub source_hash: String,
    #[pyo3(get)]
    pub target_hash: String,
    #[pyo3(get)]
    pub compression: String,
    #[pyo3(get)]
    pub operation_count: usize,
    #[pyo3(get)]
    pub tensors_modified: usize,
    #[pyo3(get)]
    pub tensors_unchanged: usize,
}

#[pymethods]
impl PatchInfo {
    fn __repr__(&self) -> String {
        format!(
            "PatchInfo(format='{}', version={}, operations={}, modified={}, unchanged={})",
            self.format,
            self.version,
            self.operation_count,
            self.tensors_modified,
            self.tensors_unchanged
        )
    }
}

/// Options for diff creation
#[pyclass]
#[derive(Clone)]
pub struct DiffOptions {
    #[pyo3(get, set)]
    pub compression: String,
    #[pyo3(get, set)]
    pub compression_level: i32,
    #[pyo3(get, set)]
    pub neural_compression: bool,
    #[pyo3(get, set)]
    pub parallel: bool,
}

#[pymethods]
impl DiffOptions {
    #[new]
    #[pyo3(signature = (compression="zstd", compression_level=3, neural_compression=false, parallel=false))]
    fn new(
        compression: &str,
        compression_level: i32,
        neural_compression: bool,
        parallel: bool,
    ) -> Self {
        Self {
            compression: compression.to_string(),
            compression_level,
            neural_compression,
            parallel,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DiffOptions(compression='{}', compression_level={}, neural_compression={}, parallel={})",
            self.compression, self.compression_level, self.neural_compression, self.parallel
        )
    }
}

/// Model fingerprint for quick version detection
#[pyclass]
#[derive(Clone)]
pub struct Fingerprint {
    #[pyo3(get)]
    pub format: String,
    #[pyo3(get)]
    pub file_size: u64,
    #[pyo3(get)]
    pub header_hash: String,
    #[pyo3(get)]
    pub tail_hash: String,
    #[pyo3(get)]
    pub combined_hash: String,
}

#[pymethods]
impl Fingerprint {
    fn __repr__(&self) -> String {
        format!(
            "Fingerprint(format='{}', file_size={}, combined_hash='{}')",
            self.format,
            self.file_size,
            &self.combined_hash[..16.min(self.combined_hash.len())]
        )
    }

    /// Check if two fingerprints match
    fn matches(&self, other: &Fingerprint) -> bool {
        self.combined_hash == other.combined_hash
    }

    /// Get a short version of the fingerprint for display
    fn short(&self) -> String {
        self.combined_hash[..16.min(self.combined_hash.len())].to_string()
    }
}

/// Detect model format from file path
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
enum ModelFormat {
    TFLite,
    GGUF,
    ONNX,
    SafeTensors,
    OpenVINO,
    CoreML,
}

fn detect_model_format(path: &Path, data: &[u8]) -> PyResult<ModelFormat> {
    // Get extension first for ONNX (protobuf has no fixed magic)
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase());

    // ONNX files are protobuf - check by extension
    if ext.as_deref() == Some("onnx") {
        return Ok(ModelFormat::ONNX);
    }

    // Check by magic bytes for TFLite, GGUF, and SafeTensors
    if data.len() >= 8 {
        // TFLite magic check
        let offset = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        if offset < data.len() && data.get(offset..offset + 4) == Some(b"TFL3") {
            return Ok(ModelFormat::TFLite);
        }

        // SafeTensors: JSON header starting with '{' after 8-byte size prefix
        let header_size = u64::from_le_bytes(data[0..8].try_into().unwrap_or([0; 8])) as usize;
        if header_size > 0 && header_size < data.len() && data.get(8) == Some(&b'{') {
            return Ok(ModelFormat::SafeTensors);
        }
    }

    if data.len() >= 5 {
        // OpenVINO XML check
        if &data[0..5] == b"<?xml" {
            return Ok(ModelFormat::OpenVINO);
        }
    }

    if data.len() >= 4 {
        // GGUF magic
        if &data[0..4] == b"GGUF"
            || u32::from_le_bytes([data[0], data[1], data[2], data[3]]) == 0x46554747
        {
            return Ok(ModelFormat::GGUF);
        }
    }

    // Fall back to extension
    match ext.as_deref() {
        Some("tflite") => Ok(ModelFormat::TFLite),
        Some("gguf") => Ok(ModelFormat::GGUF),
        Some("onnx") => Ok(ModelFormat::ONNX),
        Some("safetensors") => Ok(ModelFormat::SafeTensors),
        Some("xml") => Ok(ModelFormat::OpenVINO),
        Some("mlpackage") | Some("mlmodelc") => Ok(ModelFormat::CoreML),
        _ => Err(PyValueError::new_err(
            "Unknown model format. Supported: .tflite, .gguf, .onnx, .safetensors, .xml (OpenVINO), .mlpackage/.mlmodelc (CoreML)",
        )),
    }
}

/// Detect patch format from magic bytes
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
enum PatchFormat {
    TFLite,
    GGUF,
    ONNX,
    SafeTensors,
    OpenVINO,
    CoreML,
    TensorRT,
}

fn detect_patch_format(data: &[u8]) -> PyResult<PatchFormat> {
    if data.len() < 4 {
        return Err(PyValueError::new_err("File too small to be a valid patch"));
    }

    if mallorn_tflite::is_tflp(data) {
        return Ok(PatchFormat::TFLite);
    }

    if mallorn_gguf::is_ggup(data) {
        return Ok(PatchFormat::GGUF);
    }

    if mallorn_onnx::is_onxp(data) {
        return Ok(PatchFormat::ONNX);
    }

    if mallorn_safetensors::is_sftp(data) {
        return Ok(PatchFormat::SafeTensors);
    }

    if mallorn_openvino::is_ovinp(data) {
        return Ok(PatchFormat::OpenVINO);
    }

    if mallorn_coreml::is_cmlp(data) {
        return Ok(PatchFormat::CoreML);
    }

    if mallorn_tensorrt::format::is_trtp(data) {
        return Ok(PatchFormat::TensorRT);
    }

    Err(PyValueError::new_err(
        "Unknown patch format. Expected .tflp, .ggup, .onxp, .sftp, .ovinp, .cmlp, or .trtp file.",
    ))
}

/// Create a patch between two models
///
/// Args:
///     old_model: Path to the source/old model file
///     new_model: Path to the target/new model file
///     output: Path for output patch file (optional, auto-generated if not provided)
///     compression_level: Zstd compression level (1-22, default 3)
///     neural: Enable neural-optimized compression (default False)
///     parallel: Enable parallel compression (default False)
///
/// Returns:
///     PatchStats with compression statistics
#[pyfunction]
#[pyo3(signature = (old_model, new_model, output=None, compression_level=3, neural=false, parallel=false))]
pub fn create_patch(
    _py: Python<'_>,
    old_model: &str,
    new_model: &str,
    output: Option<&str>,
    compression_level: i32,
    neural: bool,
    parallel: bool,
) -> PyResult<PatchStats> {
    let old_path = Path::new(old_model);
    let new_path = Path::new(new_model);

    // Read model files
    let old_data = fs::read(old_path)
        .map_err(|e| PyValueError::new_err(format!("Failed to read source model: {}", e)))?;
    let new_data = fs::read(new_path)
        .map_err(|e| PyValueError::new_err(format!("Failed to read target model: {}", e)))?;

    // Detect format
    let old_format = detect_model_format(old_path, &old_data)?;
    let new_format = detect_model_format(new_path, &new_data)?;

    if old_format != new_format {
        return Err(PyValueError::new_err(
            "Source and target models must be the same format",
        ));
    }

    // Determine output path
    let output_path = match output {
        Some(p) => p.to_string(),
        None => {
            let ext = match old_format {
                ModelFormat::TFLite => "tflp",
                ModelFormat::GGUF => "ggup",
                ModelFormat::ONNX => "onxp",
                ModelFormat::SafeTensors => "sftp",
                ModelFormat::OpenVINO => "ovinp",
                ModelFormat::CoreML => "cmlp",
            };
            format!(
                "{}.{}",
                new_path.file_stem().unwrap_or_default().to_string_lossy(),
                ext
            )
        }
    };

    // Create diff options
    let options = CoreDiffOptions {
        compression: CompressionMethod::Zstd {
            level: compression_level,
        },
        min_tensor_size: 1024,
        neural_compression: neural,
        target_size_hint: None,
        dictionary: None,
    };

    // For parallel compression, we use rayon internally in the differ
    // The parallel flag is a hint that enables parallel tensor processing
    let _ = parallel; // Used by differs that support it

    // Create patch based on format
    let (patch, patch_bytes) = match old_format {
        ModelFormat::TFLite => {
            let differ = mallorn_tflite::TFLiteDiffer::with_options(options);
            let patch = differ
                .diff_from_bytes(&old_data, &new_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to compute diff: {}", e)))?;
            let bytes = mallorn_tflite::serialize_patch(&patch)
                .map_err(|e| PyValueError::new_err(format!("Failed to serialize patch: {}", e)))?;
            (patch, bytes)
        }
        ModelFormat::GGUF => {
            let differ = mallorn_gguf::GGUFDiffer::with_options(options);
            let patch = differ
                .diff_from_bytes(&old_data, &new_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to compute diff: {}", e)))?;
            let bytes = mallorn_gguf::serialize_patch(&patch)
                .map_err(|e| PyValueError::new_err(format!("Failed to serialize patch: {}", e)))?;
            (patch, bytes)
        }
        ModelFormat::ONNX => {
            let differ = mallorn_onnx::ONNXDiffer::with_options(options);
            let patch = differ
                .diff_from_bytes(&old_data, &new_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to compute diff: {}", e)))?;
            let bytes = mallorn_onnx::serialize_patch(&patch)
                .map_err(|e| PyValueError::new_err(format!("Failed to serialize patch: {}", e)))?;
            (patch, bytes)
        }
        ModelFormat::SafeTensors => {
            let differ = mallorn_safetensors::SafeTensorsDiffer::with_options(options);
            let patch = differ
                .diff_from_bytes(&old_data, &new_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to compute diff: {}", e)))?;
            let bytes = mallorn_safetensors::serialize_patch(&patch)
                .map_err(|e| PyValueError::new_err(format!("Failed to serialize patch: {}", e)))?;
            (patch, bytes)
        }
        ModelFormat::OpenVINO => {
            let differ = mallorn_openvino::OpenVINODiffer::with_options(options);
            let patch = differ
                .diff_from_bytes(&old_data, &new_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to compute diff: {}", e)))?;
            let bytes = mallorn_openvino::serialize_patch(&patch)
                .map_err(|e| PyValueError::new_err(format!("Failed to serialize patch: {}", e)))?;
            (patch, bytes)
        }
        ModelFormat::CoreML => {
            let differ = mallorn_coreml::CoreMLDiffer::with_options(options);
            let patch = differ
                .diff_from_bytes(&old_data, &new_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to compute diff: {}", e)))?;
            let bytes = mallorn_coreml::serialize_patch(&patch)
                .map_err(|e| PyValueError::new_err(format!("Failed to serialize patch: {}", e)))?;
            (patch, bytes)
        }
    };

    // Write patch file
    fs::write(&output_path, &patch_bytes)
        .map_err(|e| PyValueError::new_err(format!("Failed to write patch file: {}", e)))?;

    let source_size = old_data.len();
    let patch_size = patch_bytes.len();

    Ok(PatchStats {
        source_size,
        target_size: new_data.len(),
        patch_size,
        compression_ratio: source_size as f64 / patch_size as f64,
        tensors_modified: patch.modified_count(),
        tensors_unchanged: patch.unchanged_count(),
    })
}

/// Apply a patch to a model
///
/// Args:
///     model: Path to the source model file
///     patch: Path to the patch file
///     output: Path for the output model file
///
/// Returns:
///     PatchVerification with verification results
#[pyfunction]
pub fn apply_patch(model: &str, patch: &str, output: &str) -> PyResult<PatchVerification> {
    let model_path = Path::new(model);
    let patch_path = Path::new(patch);
    let output_path = Path::new(output);

    // Read files
    let model_data = fs::read(model_path)
        .map_err(|e| PyValueError::new_err(format!("Failed to read model file: {}", e)))?;
    let patch_data = fs::read(patch_path)
        .map_err(|e| PyValueError::new_err(format!("Failed to read patch file: {}", e)))?;

    // Detect patch format
    let format = detect_patch_format(&patch_data)?;

    // Apply patch based on format
    let (new_model_data, verification) = match format {
        PatchFormat::TFLite => {
            let patch = mallorn_tflite::deserialize_patch(&patch_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to parse patch: {}", e)))?;
            let patcher = mallorn_tflite::TFLitePatcher::new();
            patcher
                .apply_and_verify(&model_data, &patch)
                .map_err(|e| PyValueError::new_err(format!("Failed to apply patch: {}", e)))?
        }
        PatchFormat::GGUF => {
            let patch = mallorn_gguf::deserialize_patch(&patch_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to parse patch: {}", e)))?;
            let patcher = mallorn_gguf::GGUFPatcher::new();
            patcher
                .apply_and_verify(&model_data, &patch)
                .map_err(|e| PyValueError::new_err(format!("Failed to apply patch: {}", e)))?
        }
        PatchFormat::ONNX => {
            let patch = mallorn_onnx::deserialize_patch(&patch_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to parse patch: {}", e)))?;
            let patcher = mallorn_onnx::ONNXPatcher::new();
            patcher
                .apply_and_verify(&model_data, &patch)
                .map_err(|e| PyValueError::new_err(format!("Failed to apply patch: {}", e)))?
        }
        PatchFormat::SafeTensors => {
            let patch = mallorn_safetensors::deserialize_patch(&patch_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to parse patch: {}", e)))?;
            let patcher = mallorn_safetensors::SafeTensorsPatcher::new();
            patcher
                .apply_and_verify(&model_data, &patch)
                .map_err(|e| PyValueError::new_err(format!("Failed to apply patch: {}", e)))?
        }
        PatchFormat::OpenVINO => {
            let patch = mallorn_openvino::deserialize_patch(&patch_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to parse patch: {}", e)))?;
            let patcher = mallorn_openvino::OpenVINOPatcher::new();
            patcher
                .apply_and_verify(&model_data, &patch)
                .map_err(|e| PyValueError::new_err(format!("Failed to apply patch: {}", e)))?
        }
        PatchFormat::CoreML => {
            let patch = mallorn_coreml::deserialize_patch(&patch_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to parse patch: {}", e)))?;
            let patcher = mallorn_coreml::CoreMLPatcher::new();
            patcher
                .apply_and_verify(&model_data, &patch)
                .map_err(|e| PyValueError::new_err(format!("Failed to apply patch: {}", e)))?
        }
        PatchFormat::TensorRT => {
            // TensorRT patches produce ONNX output (user must rebuild engine)
            let patch = mallorn_tensorrt::deserialize_patch(&patch_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to parse patch: {}", e)))?;
            let patcher = mallorn_tensorrt::TensorRTPatcher::new();
            let (result, verification) = patcher
                .apply_and_verify(&model_data, &patch)
                .map_err(|e| PyValueError::new_err(format!("Failed to apply patch: {}", e)))?;

            // Write output ONNX model
            fs::write(output_path, &result.onnx_data).map_err(|e| {
                PyValueError::new_err(format!("Failed to write output model: {}", e))
            })?;

            return Ok(PatchVerification {
                source_valid: verification.source_valid,
                patch_valid: verification.patch_valid,
                expected_target_hash: hex::encode(verification.expected_target),
                actual_target_hash: verification.actual_target.map(hex::encode),
                stats: PatchStats {
                    source_size: verification.stats.source_size,
                    target_size: verification.stats.target_size,
                    patch_size: verification.stats.patch_size,
                    compression_ratio: verification.stats.compression_ratio,
                    tensors_modified: verification.stats.tensors_modified,
                    tensors_unchanged: verification.stats.tensors_unchanged,
                },
            });
        }
    };

    // Write output model
    fs::write(output_path, &new_model_data)
        .map_err(|e| PyValueError::new_err(format!("Failed to write output model: {}", e)))?;

    Ok(PatchVerification {
        source_valid: verification.source_valid,
        patch_valid: verification.patch_valid,
        expected_target_hash: hex::encode(verification.expected_target),
        actual_target_hash: verification.actual_target.map(hex::encode),
        stats: PatchStats {
            source_size: verification.stats.source_size,
            target_size: verification.stats.target_size,
            patch_size: verification.stats.patch_size,
            compression_ratio: verification.stats.compression_ratio,
            tensors_modified: verification.stats.tensors_modified,
            tensors_unchanged: verification.stats.tensors_unchanged,
        },
    })
}

/// Verify a patch without applying
///
/// Args:
///     model: Path to the source model file
///     patch: Path to the patch file
///
/// Returns:
///     PatchVerification with verification results
#[pyfunction]
pub fn verify_patch(model: &str, patch: &str) -> PyResult<PatchVerification> {
    let model_path = Path::new(model);
    let patch_path = Path::new(patch);

    // Read files
    let model_data = fs::read(model_path)
        .map_err(|e| PyValueError::new_err(format!("Failed to read model file: {}", e)))?;
    let patch_data = fs::read(patch_path)
        .map_err(|e| PyValueError::new_err(format!("Failed to read patch file: {}", e)))?;

    // Detect patch format
    let format = detect_patch_format(&patch_data)?;

    // Verify patch based on format
    let verification = match format {
        PatchFormat::TFLite => {
            let patch = mallorn_tflite::deserialize_patch(&patch_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to parse patch: {}", e)))?;
            let patcher = mallorn_tflite::TFLitePatcher::new();
            patcher
                .verify(&model_data, &patch)
                .map_err(|e| PyValueError::new_err(format!("Failed to verify patch: {}", e)))?
        }
        PatchFormat::GGUF => {
            let patch = mallorn_gguf::deserialize_patch(&patch_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to parse patch: {}", e)))?;
            let patcher = mallorn_gguf::GGUFPatcher::new();
            patcher
                .verify(&model_data, &patch)
                .map_err(|e| PyValueError::new_err(format!("Failed to verify patch: {}", e)))?
        }
        PatchFormat::ONNX => {
            let patch = mallorn_onnx::deserialize_patch(&patch_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to parse patch: {}", e)))?;
            let patcher = mallorn_onnx::ONNXPatcher::new();
            patcher
                .verify(&model_data, &patch)
                .map_err(|e| PyValueError::new_err(format!("Failed to verify patch: {}", e)))?
        }
        PatchFormat::SafeTensors => {
            let patch = mallorn_safetensors::deserialize_patch(&patch_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to parse patch: {}", e)))?;
            let patcher = mallorn_safetensors::SafeTensorsPatcher::new();
            patcher
                .verify(&model_data, &patch)
                .map_err(|e| PyValueError::new_err(format!("Failed to verify patch: {}", e)))?
        }
        PatchFormat::OpenVINO => {
            let patch = mallorn_openvino::deserialize_patch(&patch_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to parse patch: {}", e)))?;
            let patcher = mallorn_openvino::OpenVINOPatcher::new();
            patcher
                .verify(&model_data, &patch)
                .map_err(|e| PyValueError::new_err(format!("Failed to verify patch: {}", e)))?
        }
        PatchFormat::CoreML => {
            let patch = mallorn_coreml::deserialize_patch(&patch_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to parse patch: {}", e)))?;
            let patcher = mallorn_coreml::CoreMLPatcher::new();
            patcher
                .verify(&model_data, &patch)
                .map_err(|e| PyValueError::new_err(format!("Failed to verify patch: {}", e)))?
        }
        PatchFormat::TensorRT => {
            let patch = mallorn_tensorrt::deserialize_patch(&patch_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to parse patch: {}", e)))?;
            let patcher = mallorn_tensorrt::TensorRTPatcher::new();
            patcher
                .verify(&model_data, &patch)
                .map_err(|e| PyValueError::new_err(format!("Failed to verify patch: {}", e)))?
        }
    };

    Ok(PatchVerification {
        source_valid: verification.source_valid,
        patch_valid: verification.patch_valid,
        expected_target_hash: hex::encode(verification.expected_target),
        actual_target_hash: verification.actual_target.map(hex::encode),
        stats: PatchStats {
            source_size: verification.stats.source_size,
            target_size: verification.stats.target_size,
            patch_size: verification.stats.patch_size,
            compression_ratio: verification.stats.compression_ratio,
            tensors_modified: verification.stats.tensors_modified,
            tensors_unchanged: verification.stats.tensors_unchanged,
        },
    })
}

/// Get information about a patch file
///
/// Args:
///     patch_path: Path to the patch file
///
/// Returns:
///     PatchInfo with patch details
#[pyfunction]
pub fn patch_info(patch_path: &str) -> PyResult<PatchInfo> {
    let path = Path::new(patch_path);

    // Read patch file
    let patch_data = fs::read(path)
        .map_err(|e| PyValueError::new_err(format!("Failed to read patch file: {}", e)))?;

    // Detect patch format
    let format = detect_patch_format(&patch_data)?;

    // Parse patch based on format
    let (format_name, patch) = match format {
        PatchFormat::TFLite => {
            let patch = mallorn_tflite::deserialize_patch(&patch_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to parse patch: {}", e)))?;
            ("tflite", patch)
        }
        PatchFormat::GGUF => {
            let patch = mallorn_gguf::deserialize_patch(&patch_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to parse patch: {}", e)))?;
            ("gguf", patch)
        }
        PatchFormat::ONNX => {
            let patch = mallorn_onnx::deserialize_patch(&patch_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to parse patch: {}", e)))?;
            ("onnx", patch)
        }
        PatchFormat::SafeTensors => {
            let patch = mallorn_safetensors::deserialize_patch(&patch_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to parse patch: {}", e)))?;
            ("safetensors", patch)
        }
        PatchFormat::OpenVINO => {
            let patch = mallorn_openvino::deserialize_patch(&patch_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to parse patch: {}", e)))?;
            ("openvino", patch)
        }
        PatchFormat::CoreML => {
            let patch = mallorn_coreml::deserialize_patch(&patch_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to parse patch: {}", e)))?;
            ("coreml", patch)
        }
        PatchFormat::TensorRT => {
            // TensorRT patches contain an inner ONNX patch
            let trtp = mallorn_tensorrt::deserialize_patch(&patch_data)
                .map_err(|e| PyValueError::new_err(format!("Failed to parse patch: {}", e)))?;

            let compression_name = match trtp.onnx_patch.compression {
                CompressionMethod::None => "none".to_string(),
                CompressionMethod::Lz4 => "lz4".to_string(),
                CompressionMethod::Zstd { level } => format!("zstd({})", level),
                CompressionMethod::Neural { .. } => "neural".to_string(),
                CompressionMethod::Adaptive { strategy } => format!("adaptive({:?})", strategy),
                CompressionMethod::ZstdDict { level, dict_id } => {
                    format!("zstd+dict({},{})", level, dict_id)
                }
            };

            return Ok(PatchInfo {
                format: "tensorrt".to_string(),
                version: trtp.onnx_patch.version,
                source_hash: hex::encode(trtp.source_onnx_hash),
                target_hash: hex::encode(trtp.target_onnx_hash),
                compression: compression_name,
                operation_count: trtp.onnx_patch.operations.len(),
                tensors_modified: trtp.onnx_patch.modified_count(),
                tensors_unchanged: trtp.onnx_patch.unchanged_count(),
            });
        }
    };

    let compression_name = match patch.compression {
        CompressionMethod::None => "none".to_string(),
        CompressionMethod::Lz4 => "lz4".to_string(),
        CompressionMethod::Zstd { level } => format!("zstd({})", level),
        CompressionMethod::Neural { .. } => "neural".to_string(),
        CompressionMethod::Adaptive { strategy } => format!("adaptive({:?})", strategy),
        CompressionMethod::ZstdDict { level, dict_id } => {
            format!("zstd+dict({},{})", level, dict_id)
        }
    };

    Ok(PatchInfo {
        format: format_name.to_string(),
        version: patch.version,
        source_hash: hex::encode(patch.source_hash),
        target_hash: hex::encode(patch.target_hash),
        compression: compression_name,
        operation_count: patch.operations.len(),
        tensors_modified: patch.modified_count(),
        tensors_unchanged: patch.unchanged_count(),
    })
}

/// Generate a fingerprint for a model file
///
/// Fingerprints enable quick version detection without hashing the entire file.
/// They sample the first 64KB (header) and last 4KB (tail) of the file.
///
/// Args:
///     model_path: Path to the model file
///
/// Returns:
///     Fingerprint with header_hash, tail_hash, and combined_hash
#[pyfunction]
pub fn fingerprint(model_path: &str) -> PyResult<Fingerprint> {
    let path = Path::new(model_path);

    // Compute fingerprint
    let fp = ModelFingerprint::from_file(path)
        .map_err(|e| PyValueError::new_err(format!("Failed to compute fingerprint: {}", e)))?;

    // Create combined hash by XOR-ing header and tail hashes
    let mut combined = [0u8; 16];
    for (i, byte) in combined.iter_mut().enumerate() {
        *byte = fp.header_hash[i] ^ fp.tail_hash[i];
    }

    Ok(Fingerprint {
        format: fp.format.clone(),
        file_size: fp.file_size,
        header_hash: hex::encode(fp.header_hash),
        tail_hash: hex::encode(fp.tail_hash),
        combined_hash: hex::encode(combined),
    })
}

/// Compare two model files by fingerprint
///
/// This is much faster than comparing full file hashes for large models.
///
/// Args:
///     model1: Path to the first model file
///     model2: Path to the second model file
///
/// Returns:
///     True if the fingerprints match (models are likely identical)
#[pyfunction]
pub fn compare_fingerprints(model1: &str, model2: &str) -> PyResult<bool> {
    let fp1 = ModelFingerprint::from_file(Path::new(model1))
        .map_err(|e| PyValueError::new_err(format!("Failed to fingerprint first model: {}", e)))?;

    let fp2 = ModelFingerprint::from_file(Path::new(model2))
        .map_err(|e| PyValueError::new_err(format!("Failed to fingerprint second model: {}", e)))?;

    Ok(fp1.matches(&fp2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff_options_default() {
        let opts = DiffOptions::new("zstd", 3, false, false);
        assert_eq!(opts.compression, "zstd");
        assert_eq!(opts.compression_level, 3);
        assert!(!opts.neural_compression);
        assert!(!opts.parallel);
    }

    #[test]
    fn test_diff_options_repr() {
        let opts = DiffOptions::new("lz4", 5, true, true);
        let repr = opts.__repr__();
        assert!(repr.contains("lz4"));
        assert!(repr.contains("5"));
        assert!(repr.contains("true"));
    }

    #[test]
    fn test_patch_verification_is_valid() {
        let stats = PatchStats {
            source_size: 1000,
            target_size: 1000,
            patch_size: 100,
            compression_ratio: 10.0,
            tensors_modified: 5,
            tensors_unchanged: 10,
        };

        let verification = PatchVerification {
            source_valid: true,
            patch_valid: true,
            expected_target_hash: "abc".into(),
            actual_target_hash: Some("abc".into()),
            stats,
        };

        assert!(verification.is_valid());
    }

    #[test]
    fn test_fingerprint_matches() {
        let fp1 = Fingerprint {
            format: "tflite".to_string(),
            file_size: 1000,
            header_hash: "abc123".to_string(),
            tail_hash: "def456".to_string(),
            combined_hash: "combined789".to_string(),
        };

        let fp2 = Fingerprint {
            format: "tflite".to_string(),
            file_size: 1000,
            header_hash: "abc123".to_string(),
            tail_hash: "def456".to_string(),
            combined_hash: "combined789".to_string(),
        };

        let fp3 = Fingerprint {
            format: "tflite".to_string(),
            file_size: 1000,
            header_hash: "abc123".to_string(),
            tail_hash: "def456".to_string(),
            combined_hash: "different".to_string(),
        };

        assert!(fp1.matches(&fp2));
        assert!(!fp1.matches(&fp3));
    }
}
