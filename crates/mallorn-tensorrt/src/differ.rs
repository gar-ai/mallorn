//! TensorRT differ - creates patches from ONNX models with TensorRT config
//!
//! This differ wraps `mallorn-onnx` to compute tensor-aware deltas,
//! then attaches TensorRT build configuration for the rebuild workflow.

use crate::config::TensorRTConfig;
use crate::format::TensorRTPatch;
use mallorn_core::{sha256, CompressionMethod, DiffError, DiffOptions};
use mallorn_onnx::ONNXDiffer;

/// TensorRT differ
///
/// Creates patches from ONNX source models with attached TensorRT
/// build configuration for engine reconstruction.
pub struct TensorRTDiffer {
    options: DiffOptions,
    onnx_differ: ONNXDiffer,
}

impl TensorRTDiffer {
    /// Create a new differ with default options
    pub fn new() -> Self {
        Self::with_options(DiffOptions::default())
    }

    /// Create a new differ with custom options
    pub fn with_options(options: DiffOptions) -> Self {
        Self {
            onnx_differ: ONNXDiffer::with_options(options.clone()),
            options,
        }
    }

    /// Create a patch between two ONNX models with TensorRT build config
    ///
    /// The patch contains:
    /// 1. ONNX tensor delta (computed by mallorn-onnx)
    /// 2. TensorRT build configuration
    /// 3. Source/target ONNX hashes for verification
    ///
    /// # Arguments
    /// * `old_onnx` - Original ONNX model bytes
    /// * `new_onnx` - Updated ONNX model bytes
    /// * `config` - TensorRT build configuration for engine reconstruction
    pub fn diff(
        &self,
        old_onnx: &[u8],
        new_onnx: &[u8],
        config: TensorRTConfig,
    ) -> Result<TensorRTPatch, DiffError> {
        // Delegate to ONNX differ for actual tensor diffing
        let onnx_patch = self.onnx_differ.diff_from_bytes(old_onnx, new_onnx)?;

        Ok(TensorRTPatch {
            onnx_patch,
            config,
            source_onnx_hash: sha256(old_onnx),
            target_onnx_hash: sha256(new_onnx),
        })
    }

    /// Create a patch with default TensorRT configuration
    pub fn diff_default(
        &self,
        old_onnx: &[u8],
        new_onnx: &[u8],
    ) -> Result<TensorRTPatch, DiffError> {
        self.diff(old_onnx, new_onnx, TensorRTConfig::default())
    }

    /// Get the compression method being used
    pub fn compression_method(&self) -> &CompressionMethod {
        &self.options.compression
    }

    /// Get current diff options
    pub fn options(&self) -> &DiffOptions {
        &self.options
    }
}

impl Default for TensorRTDiffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics from a TensorRT diff operation
#[derive(Debug, Clone)]
pub struct TensorRTDiffStats {
    /// Size of the source ONNX model
    pub source_size: usize,
    /// Size of the target ONNX model
    pub target_size: usize,
    /// Size of the serialized patch
    pub patch_size: usize,
    /// Number of tensors modified
    pub tensors_modified: usize,
    /// Number of tensors unchanged
    pub tensors_unchanged: usize,
    /// Compression ratio achieved
    pub compression_ratio: f64,
    /// TensorRT precision mode
    pub precision: crate::config::Precision,
}

impl TensorRTDiffStats {
    /// Create stats from a patch operation
    pub fn from_patch(
        source_size: usize,
        target_size: usize,
        patch: &TensorRTPatch,
        serialized_size: usize,
    ) -> Self {
        Self {
            source_size,
            target_size,
            patch_size: serialized_size,
            tensors_modified: patch.onnx_patch.modified_count(),
            tensors_unchanged: patch.onnx_patch.unchanged_count(),
            compression_ratio: source_size as f64 / serialized_size as f64,
            precision: patch.config.precision,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mallorn_core::Patch;

    #[test]
    fn test_differ_creation() {
        let differ = TensorRTDiffer::new();
        assert_eq!(differ.options().min_tensor_size, 1024);
    }

    #[test]
    fn test_differ_with_options() {
        let options = DiffOptions {
            compression: CompressionMethod::Zstd { level: 5 },
            min_tensor_size: 512,
            neural_compression: false,
            target_size_hint: None,
            dictionary: None,
        };
        let differ = TensorRTDiffer::with_options(options);
        assert_eq!(differ.options().min_tensor_size, 512);
    }

    #[test]
    fn test_differ_compression_method() {
        let options = DiffOptions {
            compression: CompressionMethod::Lz4,
            min_tensor_size: 1024,
            neural_compression: false,
            target_size_hint: None,
            dictionary: None,
        };
        let differ = TensorRTDiffer::with_options(options);
        assert!(matches!(
            differ.compression_method(),
            CompressionMethod::Lz4
        ));
    }

    #[test]
    fn test_diff_stats() {
        let patch = TensorRTPatch {
            onnx_patch: Patch {
                version: 1,
                source_hash: [0u8; 32],
                target_hash: [1u8; 32],
                operations: vec![
                    mallorn_core::PatchOperation::CopyTensor {
                        name: "tensor1".into(),
                    },
                    mallorn_core::PatchOperation::ReplaceTensor {
                        name: "tensor2".into(),
                        data: vec![1, 2, 3, 4],
                        compression: None,
                    },
                ],
                compression: CompressionMethod::Zstd { level: 3 },
                metadata: mallorn_core::PatchMetadata::default(),
            },
            config: TensorRTConfig::new().with_precision(crate::config::Precision::FP16),
            source_onnx_hash: [0u8; 32],
            target_onnx_hash: [1u8; 32],
        };

        let stats = TensorRTDiffStats::from_patch(10000, 10000, &patch, 500);
        assert_eq!(stats.tensors_modified, 1);
        assert_eq!(stats.tensors_unchanged, 1);
        assert_eq!(stats.precision, crate::config::Precision::FP16);
        assert!((stats.compression_ratio - 20.0).abs() < 0.01);
    }
}
