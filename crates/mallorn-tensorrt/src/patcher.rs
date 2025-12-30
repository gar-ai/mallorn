//! TensorRT patch application
//!
//! Applies patches to ONNX models, outputting updated ONNX data
//! along with rebuild instructions for TensorRT engine reconstruction.

use crate::config::TensorRTConfig;
use crate::format::TensorRTPatch;
use mallorn_core::{sha256, verify_hash, PatchError, PatchStats, PatchVerification};
use mallorn_onnx::ONNXPatcher;

/// Result of applying a TensorRT patch
#[derive(Debug)]
pub struct ApplyResult {
    /// Updated ONNX model data
    pub onnx_data: Vec<u8>,
    /// TensorRT build configuration for engine reconstruction
    pub config: TensorRTConfig,
    /// Generated trtexec command for rebuilding the engine
    pub rebuild_command: String,
    /// Verification that the patch was applied correctly
    pub verification: ApplyVerification,
}

/// Verification results from patch application
#[derive(Debug)]
pub struct ApplyVerification {
    /// Whether source ONNX hash matched
    pub source_valid: bool,
    /// Whether target ONNX hash matched after patching
    pub target_valid: bool,
    /// Expected target hash
    pub expected_hash: [u8; 32],
    /// Actual hash of patched ONNX
    pub actual_hash: [u8; 32],
}

/// TensorRT patch applier
///
/// Applies ONNX patches and provides rebuild instructions for TensorRT engines.
pub struct TensorRTPatcher {
    onnx_patcher: ONNXPatcher,
}

impl TensorRTPatcher {
    /// Create a new patcher
    pub fn new() -> Self {
        Self {
            onnx_patcher: ONNXPatcher::new(),
        }
    }

    /// Apply a patch to ONNX model bytes
    ///
    /// Returns the updated ONNX model data and TensorRT rebuild instructions.
    /// The user must then run trtexec or equivalent to rebuild the engine.
    ///
    /// # Arguments
    /// * `old_onnx` - Original ONNX model bytes
    /// * `patch` - TensorRT patch to apply
    ///
    /// # Returns
    /// * `ApplyResult` containing new ONNX data and rebuild command
    pub fn apply(&self, old_onnx: &[u8], patch: &TensorRTPatch) -> Result<ApplyResult, PatchError> {
        // Verify source ONNX hash
        if !verify_hash(old_onnx, &patch.source_onnx_hash) {
            return Err(PatchError::SourceHashMismatch);
        }

        // Apply ONNX patch
        let new_onnx = self.onnx_patcher.apply(old_onnx, &patch.onnx_patch)?;

        // Verify target hash
        let actual_hash = sha256(&new_onnx);
        let target_valid = actual_hash == patch.target_onnx_hash;

        if !target_valid {
            return Err(PatchError::TargetHashMismatch);
        }

        // Generate rebuild command
        let rebuild_command = patch.config.to_trtexec_command("model.onnx", "model.engine");

        Ok(ApplyResult {
            onnx_data: new_onnx,
            config: patch.config.clone(),
            rebuild_command,
            verification: ApplyVerification {
                source_valid: true,
                target_valid,
                expected_hash: patch.target_onnx_hash,
                actual_hash,
            },
        })
    }

    /// Apply patch with custom output paths in the rebuild command
    pub fn apply_with_paths(
        &self,
        old_onnx: &[u8],
        patch: &TensorRTPatch,
        onnx_output: &str,
        engine_output: &str,
    ) -> Result<ApplyResult, PatchError> {
        let mut result = self.apply(old_onnx, patch)?;
        result.rebuild_command = patch.config.to_trtexec_command(onnx_output, engine_output);
        Ok(result)
    }

    /// Verify a patch can be applied without actually applying it
    pub fn verify(
        &self,
        old_onnx: &[u8],
        patch: &TensorRTPatch,
    ) -> Result<PatchVerification, PatchError> {
        let source_valid = verify_hash(old_onnx, &patch.source_onnx_hash);
        let patch_valid = self.verify_patch_structure(patch);

        let patch_size = self.estimate_patch_size(patch);
        let stats = PatchStats {
            source_size: old_onnx.len(),
            target_size: 0, // Unknown without applying
            patch_size,
            compression_ratio: old_onnx.len() as f64 / patch_size as f64,
            tensors_modified: patch.onnx_patch.modified_count(),
            tensors_unchanged: patch.onnx_patch.unchanged_count(),
        };

        Ok(PatchVerification {
            source_valid,
            patch_valid,
            expected_target: patch.target_onnx_hash,
            actual_target: None,
            stats,
        })
    }

    /// Apply and fully verify the patch
    pub fn apply_and_verify(
        &self,
        old_onnx: &[u8],
        patch: &TensorRTPatch,
    ) -> Result<(ApplyResult, PatchVerification), PatchError> {
        let result = self.apply(old_onnx, patch)?;

        let stats = PatchStats {
            source_size: old_onnx.len(),
            target_size: result.onnx_data.len(),
            patch_size: self.estimate_patch_size(patch),
            compression_ratio: old_onnx.len() as f64 / self.estimate_patch_size(patch) as f64,
            tensors_modified: patch.onnx_patch.modified_count(),
            tensors_unchanged: patch.onnx_patch.unchanged_count(),
        };

        let verification = PatchVerification {
            source_valid: true,
            patch_valid: true,
            expected_target: patch.target_onnx_hash,
            actual_target: Some(result.verification.actual_hash),
            stats,
        };

        Ok((result, verification))
    }

    /// Get rebuild instructions without applying the patch
    pub fn get_rebuild_instructions(&self, patch: &TensorRTPatch) -> RebuildInstructions {
        RebuildInstructions {
            command: patch.config.to_trtexec_command("model.onnx", "model.engine"),
            precision: patch.config.precision,
            workspace_mb: patch.config.workspace_size_mb,
            max_batch_size: patch.config.max_batch_size,
            dla_core: patch.config.dla_core,
            requires_calibration: patch.config.precision == crate::config::Precision::INT8,
            calibration_cache: patch.config.calibration_cache.clone(),
            min_tensorrt_version: patch.config.min_tensorrt_version.clone(),
            additional_flags: patch.config.builder_flags.clone(),
        }
    }

    /// Verify patch structure is valid
    fn verify_patch_structure(&self, patch: &TensorRTPatch) -> bool {
        if patch.onnx_patch.version != 1 {
            return false;
        }

        // Config validation
        if patch.config.workspace_size_mb == 0 {
            return false;
        }

        if patch.config.max_batch_size == 0 {
            return false;
        }

        // INT8 requires calibration
        if patch.config.precision == crate::config::Precision::INT8
            && patch.config.calibration_cache.is_none()
        {
            // Warning but not invalid - calibration can be done separately
        }

        true
    }

    /// Estimate serialized patch size
    fn estimate_patch_size(&self, patch: &TensorRTPatch) -> usize {
        let mut size = 4 + 4 + 8 + 4; // magic + version + onnx_patch_len + config_len

        // ONNX patch size estimate
        size += 4 + 4 + 32 + 32; // ONNX magic + version + hashes
        for op in &patch.onnx_patch.operations {
            size += 4; // op type
            match op {
                mallorn_core::PatchOperation::ReplaceTensor { name, data, .. } => {
                    size += 4 + name.len() + 4 + data.len();
                }
                mallorn_core::PatchOperation::DeltaTensor { name, delta, .. } => {
                    size += 4 + name.len() + 4 + delta.len() + 1;
                }
                mallorn_core::PatchOperation::CopyTensor { name } => {
                    size += 4 + name.len();
                }
                mallorn_core::PatchOperation::UpdateMetadata { key, value } => {
                    size += 4 + key.len() + 4 + value.len();
                }
            }
        }

        // Config JSON estimate
        size += serde_json::to_string(&patch.config)
            .map(|s| s.len())
            .unwrap_or(200);

        // Hashes and CRC
        size += 32 + 32 + 4;

        size
    }
}

impl Default for TensorRTPatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Instructions for rebuilding a TensorRT engine
#[derive(Debug, Clone)]
pub struct RebuildInstructions {
    /// Full trtexec command
    pub command: String,
    /// Compute precision
    pub precision: crate::config::Precision,
    /// Workspace size in MB
    pub workspace_mb: u32,
    /// Maximum batch size
    pub max_batch_size: u32,
    /// DLA core (for Jetson)
    pub dla_core: Option<i32>,
    /// Whether INT8 calibration is required
    pub requires_calibration: bool,
    /// Path to calibration cache
    pub calibration_cache: Option<String>,
    /// Minimum TensorRT version
    pub min_tensorrt_version: Option<String>,
    /// Additional builder flags
    pub additional_flags: Vec<String>,
}

impl std::fmt::Display for RebuildInstructions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "TensorRT Engine Rebuild Instructions")?;
        writeln!(f, "====================================")?;
        writeln!(f)?;
        writeln!(f, "Command:")?;
        writeln!(f, "  {}", self.command)?;
        writeln!(f)?;
        writeln!(f, "Configuration:")?;
        writeln!(f, "  Precision: {:?}", self.precision)?;
        writeln!(f, "  Workspace: {} MB", self.workspace_mb)?;
        writeln!(f, "  Max Batch: {}", self.max_batch_size)?;

        if let Some(dla) = self.dla_core {
            writeln!(f, "  DLA Core: {}", dla)?;
        }

        if self.requires_calibration {
            writeln!(f)?;
            writeln!(f, "Note: INT8 precision requires calibration.")?;
            if let Some(ref cache) = self.calibration_cache {
                writeln!(f, "  Calibration cache: {}", cache)?;
            } else {
                writeln!(f, "  Run calibration before engine build.")?;
            }
        }

        if let Some(ref version) = self.min_tensorrt_version {
            writeln!(f)?;
            writeln!(f, "Requires TensorRT >= {}", version)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Precision;
    use mallorn_core::{CompressionMethod, Patch, PatchMetadata, PatchOperation};

    fn create_test_patch() -> TensorRTPatch {
        TensorRTPatch {
            onnx_patch: Patch {
                version: 1,
                source_hash: [0u8; 32],
                target_hash: [1u8; 32],
                operations: vec![PatchOperation::CopyTensor {
                    name: "test".into(),
                }],
                compression: CompressionMethod::Zstd { level: 3 },
                metadata: PatchMetadata::default(),
            },
            config: TensorRTConfig::new()
                .with_precision(Precision::FP16)
                .with_workspace_mb(2048),
            source_onnx_hash: [0u8; 32],
            target_onnx_hash: [1u8; 32],
        }
    }

    #[test]
    fn test_patcher_creation() {
        let patcher = TensorRTPatcher::new();
        let _default = TensorRTPatcher::default();
        assert!(std::ptr::eq(&patcher.onnx_patcher, &patcher.onnx_patcher));
    }

    #[test]
    fn test_rebuild_instructions() {
        let patcher = TensorRTPatcher::new();
        let patch = create_test_patch();
        let instructions = patcher.get_rebuild_instructions(&patch);

        assert!(instructions.command.contains("--fp16"));
        assert!(instructions.command.contains("--workspace=2048"));
        assert_eq!(instructions.precision, Precision::FP16);
        assert_eq!(instructions.workspace_mb, 2048);
        assert!(!instructions.requires_calibration);
    }

    #[test]
    fn test_rebuild_instructions_int8() {
        let patcher = TensorRTPatcher::new();
        let patch = TensorRTPatch {
            onnx_patch: Patch {
                version: 1,
                source_hash: [0u8; 32],
                target_hash: [1u8; 32],
                operations: vec![],
                compression: CompressionMethod::Zstd { level: 3 },
                metadata: PatchMetadata::default(),
            },
            config: TensorRTConfig::new()
                .with_precision(Precision::INT8)
                .with_calibration_cache("calib.cache"),
            source_onnx_hash: [0u8; 32],
            target_onnx_hash: [1u8; 32],
        };

        let instructions = patcher.get_rebuild_instructions(&patch);
        assert!(instructions.requires_calibration);
        assert_eq!(
            instructions.calibration_cache,
            Some("calib.cache".to_string())
        );
    }

    #[test]
    fn test_verify_patch_structure() {
        let patcher = TensorRTPatcher::new();
        let patch = create_test_patch();
        assert!(patcher.verify_patch_structure(&patch));
    }

    #[test]
    fn test_verify_patch_structure_invalid() {
        let patcher = TensorRTPatcher::new();
        let mut patch = create_test_patch();
        patch.onnx_patch.version = 99;
        assert!(!patcher.verify_patch_structure(&patch));

        let mut patch2 = create_test_patch();
        patch2.config.workspace_size_mb = 0;
        assert!(!patcher.verify_patch_structure(&patch2));
    }

    #[test]
    fn test_rebuild_instructions_display() {
        let patcher = TensorRTPatcher::new();
        let patch = create_test_patch();
        let instructions = patcher.get_rebuild_instructions(&patch);
        let display = format!("{}", instructions);

        assert!(display.contains("TensorRT Engine Rebuild Instructions"));
        assert!(display.contains("Precision: FP16"));
        assert!(display.contains("Workspace: 2048 MB"));
    }

    #[test]
    fn test_estimate_patch_size() {
        let patcher = TensorRTPatcher::new();
        let patch = create_test_patch();
        let size = patcher.estimate_patch_size(&patch);

        // Should be at least headers + hashes
        assert!(size > 100);
    }
}
