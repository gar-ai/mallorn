//! ONNX patch application
//!
//! Applies delta patches to ONNX models, reconstructing new model
//! versions from old models and patch files.

use crate::parser::{ONNXParser, ONNXTensor};
use mallorn_core::{
    apply_xor_delta, sha256, verify_hash, Compressor, CompressionMethod, DataType, DeltaFormat,
    Lz4Compressor, NeuralCompressor, Patch, PatchError, PatchOperation, PatchStats,
    PatchVerification, ZstdCompressor,
};
use std::collections::HashMap;

/// ONNX patch applier
pub struct ONNXPatcher {
    parser: ONNXParser,
}

impl ONNXPatcher {
    /// Create a new patcher
    pub fn new() -> Self {
        Self {
            parser: ONNXParser::new(),
        }
    }

    /// Apply a patch to ONNX model bytes, returning new model bytes
    pub fn apply(&self, old_data: &[u8], patch: &Patch) -> Result<Vec<u8>, PatchError> {
        // Verify source hash
        if !verify_hash(old_data, &patch.source_hash) {
            return Err(PatchError::SourceHashMismatch);
        }

        // Parse old model
        let old_model = self
            .parser
            .parse(old_data)
            .map_err(|_e| PatchError::Corrupted)?;

        // Build tensor lookup
        let old_tensor_map: HashMap<&str, &ONNXTensor> = old_model
            .tensors
            .iter()
            .map(|t| (t.name.as_str(), t))
            .collect();

        // Get appropriate decompressor
        let decompressor: Box<dyn Compressor> = match patch.compression {
            CompressionMethod::Zstd { level } => Box::new(ZstdCompressor::new(level)),
            CompressionMethod::Lz4 => Box::new(Lz4Compressor::new()),
            CompressionMethod::Neural { .. } => Box::new(NeuralCompressor::new(3)),
            CompressionMethod::None => Box::new(ZstdCompressor::new(3)),
            CompressionMethod::Adaptive { .. } => Box::new(ZstdCompressor::new(3)),
            // ZstdDict requires dictionary for decompression - fall back to Zstd
            CompressionMethod::ZstdDict { level, .. } => Box::new(ZstdCompressor::new(level)),
        };

        // Apply operations to build new tensor data
        let mut new_tensors: Vec<ONNXTensor> = Vec::new();

        for op in &patch.operations {
            let new_tensor = match op {
                PatchOperation::CopyTensor { name } => {
                    let old_tensor = old_tensor_map
                        .get(name.as_str())
                        .ok_or_else(|| PatchError::MissingTensor(name.clone()))?;
                    (*old_tensor).clone()
                }

                PatchOperation::ReplaceTensor { name, data, .. } => {
                    let dtype: DataType = old_tensor_map
                        .get(name.as_str())
                        .map(|t| t.data_type.into())
                        .unwrap_or(DataType::Float32);

                    let decompressed = decompressor
                        .decompress(data, dtype)
                        .map_err(|e| PatchError::DecompressionFailed(e.to_string()))?;

                    if let Some(old_tensor) = old_tensor_map.get(name.as_str()) {
                        ONNXTensor {
                            name: name.clone(),
                            dims: old_tensor.dims.clone(),
                            data_type: old_tensor.data_type,
                            data: decompressed,
                            initializer_index: old_tensor.initializer_index,
                        }
                    } else {
                        // New tensor being added
                        ONNXTensor {
                            name: name.clone(),
                            dims: vec![decompressed.len() as i64],
                            data_type: crate::parser::ONNXDataType::Float,
                            data: decompressed,
                            initializer_index: new_tensors.len(),
                        }
                    }
                }

                PatchOperation::DeltaTensor {
                    name,
                    delta,
                    delta_format,
                    ..
                } => {
                    let old_tensor = old_tensor_map
                        .get(name.as_str())
                        .ok_or_else(|| PatchError::MissingTensor(name.clone()))?;

                    let decompressed_delta = decompressor
                        .decompress(delta, old_tensor.data_type.into())
                        .map_err(|e| PatchError::DecompressionFailed(e.to_string()))?;

                    let new_data = match delta_format {
                        DeltaFormat::Xor => apply_xor_delta(&old_tensor.data, &decompressed_delta),
                        _ => apply_xor_delta(&old_tensor.data, &decompressed_delta),
                    };

                    ONNXTensor {
                        name: name.clone(),
                        dims: old_tensor.dims.clone(),
                        data_type: old_tensor.data_type,
                        data: new_data,
                        initializer_index: old_tensor.initializer_index,
                    }
                }

                PatchOperation::UpdateMetadata { .. } => {
                    continue;
                }
            };

            new_tensors.push(new_tensor);
        }

        // Rebuild ONNX model binary (re-encode protobuf)
        let new_model_data = self.parser.reconstruct(&old_model, &new_tensors)
            .map_err(|_e| PatchError::Corrupted)?;

        // Verify target hash
        if !verify_hash(&new_model_data, &patch.target_hash) {
            return Err(PatchError::TargetHashMismatch);
        }

        Ok(new_model_data)
    }

    /// Verify a patch can be applied without actually applying it
    pub fn verify(&self, old_data: &[u8], patch: &Patch) -> Result<PatchVerification, PatchError> {
        let source_valid = verify_hash(old_data, &patch.source_hash);
        let patch_valid = self.verify_patch_structure(patch);

        let patch_size = self.estimate_patch_size(patch);
        let stats = PatchStats {
            source_size: old_data.len(),
            target_size: 0,
            patch_size,
            compression_ratio: old_data.len() as f64 / patch_size as f64,
            tensors_modified: patch.modified_count(),
            tensors_unchanged: patch.unchanged_count(),
        };

        Ok(PatchVerification {
            source_valid,
            patch_valid,
            expected_target: patch.target_hash,
            actual_target: None,
            stats,
        })
    }

    /// Apply patch and verify, returning full verification results
    pub fn apply_and_verify(
        &self,
        old_data: &[u8],
        patch: &Patch,
    ) -> Result<(Vec<u8>, PatchVerification), PatchError> {
        let source_valid = verify_hash(old_data, &patch.source_hash);

        if !source_valid {
            return Err(PatchError::SourceHashMismatch);
        }

        let new_data = self.apply(old_data, patch)?;
        let actual_hash = sha256(&new_data);

        let stats = PatchStats {
            source_size: old_data.len(),
            target_size: new_data.len(),
            patch_size: self.estimate_patch_size(patch),
            compression_ratio: old_data.len() as f64 / self.estimate_patch_size(patch) as f64,
            tensors_modified: patch.modified_count(),
            tensors_unchanged: patch.unchanged_count(),
        };

        let verification = PatchVerification {
            source_valid: true,
            patch_valid: true,
            expected_target: patch.target_hash,
            actual_target: Some(actual_hash),
            stats,
        };

        Ok((new_data, verification))
    }

    /// Verify patch structure is valid
    fn verify_patch_structure(&self, patch: &Patch) -> bool {
        if patch.version != 1 {
            return false;
        }

        for op in &patch.operations {
            match op {
                PatchOperation::ReplaceTensor { name, data, .. } => {
                    if name.is_empty() || data.is_empty() {
                        return false;
                    }
                }
                PatchOperation::DeltaTensor { name, delta, .. } => {
                    if name.is_empty() || delta.is_empty() {
                        return false;
                    }
                }
                PatchOperation::CopyTensor { name } => {
                    if name.is_empty() {
                        return false;
                    }
                }
                PatchOperation::UpdateMetadata { key, .. } => {
                    if key.is_empty() {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Estimate serialized patch size
    fn estimate_patch_size(&self, patch: &Patch) -> usize {
        let mut size = 4 + 4 + 32 + 32; // magic + version + hashes

        for op in &patch.operations {
            size += 4;
            match op {
                PatchOperation::ReplaceTensor { name, data, .. } => {
                    size += 4 + name.len() + 4 + data.len();
                }
                PatchOperation::DeltaTensor { name, delta, .. } => {
                    size += 4 + name.len() + 4 + delta.len() + 1;
                }
                PatchOperation::CopyTensor { name } => {
                    size += 4 + name.len();
                }
                PatchOperation::UpdateMetadata { key, value } => {
                    size += 4 + key.len() + 4 + value.len();
                }
            }
        }

        size += 4; // CRC
        size
    }
}

impl Default for ONNXPatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patcher_creation() {
        let patcher = ONNXPatcher::new();
        assert!(patcher.parser.parse(&[]).is_err());
    }

    #[test]
    fn test_verify_patch_structure_valid() {
        let patcher = ONNXPatcher::new();
        let patch = Patch {
            version: 1,
            source_hash: [0u8; 32],
            target_hash: [0u8; 32],
            operations: vec![PatchOperation::CopyTensor {
                name: "test".into(),
            }],
            compression: CompressionMethod::Zstd { level: 3 },
            metadata: mallorn_core::PatchMetadata::default(),
        };
        assert!(patcher.verify_patch_structure(&patch));
    }

    #[test]
    fn test_verify_patch_structure_invalid_version() {
        let patcher = ONNXPatcher::new();
        let patch = Patch {
            version: 99,
            source_hash: [0u8; 32],
            target_hash: [0u8; 32],
            operations: vec![],
            compression: CompressionMethod::Zstd { level: 3 },
            metadata: mallorn_core::PatchMetadata::default(),
        };
        assert!(!patcher.verify_patch_structure(&patch));
    }

    #[test]
    fn test_verify_patch_structure_empty_tensor_name() {
        let patcher = ONNXPatcher::new();
        let patch = Patch {
            version: 1,
            source_hash: [0u8; 32],
            target_hash: [0u8; 32],
            operations: vec![PatchOperation::CopyTensor { name: "".into() }],
            compression: CompressionMethod::Zstd { level: 3 },
            metadata: mallorn_core::PatchMetadata::default(),
        };
        assert!(!patcher.verify_patch_structure(&patch));
    }

    #[test]
    fn test_estimate_patch_size() {
        let patcher = ONNXPatcher::new();
        let patch = Patch {
            version: 1,
            source_hash: [0u8; 32],
            target_hash: [0u8; 32],
            operations: vec![
                PatchOperation::CopyTensor {
                    name: "tensor1".into(),
                },
                PatchOperation::ReplaceTensor {
                    name: "tensor2".into(),
                    data: vec![1, 2, 3, 4],
                    compression: None,
                },
            ],
            compression: CompressionMethod::Zstd { level: 3 },
            metadata: mallorn_core::PatchMetadata::default(),
        };

        let size = patcher.estimate_patch_size(&patch);
        assert!(size > 72);
    }
}
