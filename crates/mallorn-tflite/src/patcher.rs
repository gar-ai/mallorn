//! TFLite patch application
//!
//! Applies delta patches to TFLite models, reconstructing new model
//! versions from old models and patch files.

use crate::parser::{TFLiteDataType, TFLiteModel, TFLiteParser, TFLiteTensor};
use mallorn_core::{
    apply_xor_delta, sha256, verify_hash, Compressor, CompressionMethod, DataType, DeltaFormat,
    Lz4Compressor, NeuralCompressor, Patch, PatchError, PatchOperation, PatchStats,
    PatchVerification, ZstdCompressor,
};
use std::collections::HashMap;

/// TFLite patch applier
pub struct TFLitePatcher {
    parser: TFLiteParser,
}

impl TFLitePatcher {
    /// Create a new patcher
    pub fn new() -> Self {
        Self {
            parser: TFLiteParser::new(),
        }
    }

    /// Apply a patch to TFLite model bytes, returning new model bytes
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
        let old_tensor_map: HashMap<&str, &TFLiteTensor> = old_model
            .tensors
            .iter()
            .map(|t| (t.name.as_str(), t))
            .collect();

        // Get appropriate decompressor
        let decompressor: Box<dyn Compressor> = match patch.compression {
            CompressionMethod::Zstd { level } => Box::new(ZstdCompressor::new(level)),
            CompressionMethod::Lz4 => Box::new(Lz4Compressor::new()),
            CompressionMethod::Neural { .. } => Box::new(NeuralCompressor::new(3)),
            CompressionMethod::None => Box::new(ZstdCompressor::new(1)),
            CompressionMethod::Adaptive { .. } => Box::new(ZstdCompressor::new(3)),
            // ZstdDict requires dictionary for decompression - fall back to Zstd
            CompressionMethod::ZstdDict { level, .. } => Box::new(ZstdCompressor::new(level)),
        };

        // Apply operations to build new tensor data
        let mut new_tensors: Vec<TFLiteTensor> = Vec::new();

        for op in &patch.operations {
            let new_tensor = match op {
                PatchOperation::CopyTensor { name } => {
                    // Copy unchanged tensor from old model
                    let old_tensor = old_tensor_map
                        .get(name.as_str())
                        .ok_or_else(|| PatchError::MissingTensor(name.clone()))?;
                    (*old_tensor).clone()
                }

                PatchOperation::ReplaceTensor { name, data, .. } => {
                    // Get tensor metadata from old model if exists
                    let dtype = old_tensor_map
                        .get(name.as_str())
                        .map(|t| t.dtype.into())
                        .unwrap_or(DataType::UInt8);

                    // Decompress and use new data entirely
                    let decompressed = decompressor
                        .decompress(data, dtype)
                        .map_err(|e| PatchError::DecompressionFailed(e.to_string()))?;

                    // Get tensor metadata from old model if exists, or create new
                    if let Some(old_tensor) = old_tensor_map.get(name.as_str()) {
                        TFLiteTensor {
                            name: name.clone(),
                            shape: old_tensor.shape.clone(),
                            dtype: old_tensor.dtype,
                            buffer_index: old_tensor.buffer_index,
                            data: decompressed,
                            quantization: old_tensor.quantization.clone(),
                        }
                    } else {
                        // New tensor - infer what we can
                        TFLiteTensor {
                            name: name.clone(),
                            shape: vec![decompressed.len() as i32],
                            dtype: TFLiteDataType::UInt8,
                            buffer_index: 0,
                            data: decompressed,
                            quantization: None,
                        }
                    }
                }

                PatchOperation::DeltaTensor {
                    name,
                    delta,
                    delta_format,
                    ..
                } => {
                    // Get old tensor
                    let old_tensor = old_tensor_map
                        .get(name.as_str())
                        .ok_or_else(|| PatchError::MissingTensor(name.clone()))?;

                    // Decompress delta
                    let decompressed_delta = decompressor
                        .decompress(delta, old_tensor.dtype.into())
                        .map_err(|e| PatchError::DecompressionFailed(e.to_string()))?;

                    // Apply delta based on format
                    let new_data = match delta_format {
                        DeltaFormat::Xor => apply_xor_delta(&old_tensor.data, &decompressed_delta),
                        _ => {
                            // For other formats, fall back to XOR for now
                            apply_xor_delta(&old_tensor.data, &decompressed_delta)
                        }
                    };

                    TFLiteTensor {
                        name: name.clone(),
                        shape: old_tensor.shape.clone(),
                        dtype: old_tensor.dtype,
                        buffer_index: old_tensor.buffer_index,
                        data: new_data,
                        quantization: old_tensor.quantization.clone(),
                    }
                }

                PatchOperation::UpdateMetadata { .. } => {
                    // Metadata updates don't produce tensors
                    continue;
                }
            };

            new_tensors.push(new_tensor);
        }

        // Rebuild TFLite model binary
        let new_model_data = self.rebuild_model(&old_model, &new_tensors)?;

        // Verify target hash
        if !verify_hash(&new_model_data, &patch.target_hash) {
            return Err(PatchError::TargetHashMismatch);
        }

        Ok(new_model_data)
    }

    /// Verify a patch can be applied without actually applying it
    pub fn verify(&self, old_data: &[u8], patch: &Patch) -> Result<PatchVerification, PatchError> {
        // Check source hash
        let source_valid = verify_hash(old_data, &patch.source_hash);

        // Verify patch structure
        let patch_valid = self.verify_patch_structure(patch);

        // Calculate stats
        let patch_size = self.estimate_patch_size(patch);
        let stats = PatchStats {
            source_size: old_data.len(),
            target_size: 0, // Unknown without applying
            patch_size,
            compression_ratio: old_data.len() as f64 / patch_size as f64,
            tensors_modified: patch.modified_count(),
            tensors_unchanged: patch.unchanged_count(),
        };

        Ok(PatchVerification {
            source_valid,
            patch_valid,
            expected_target: patch.target_hash,
            actual_target: None, // Only known after apply
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

        // Apply patch
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

    /// Rebuild TFLite model with new tensor data
    ///
    /// This is a simplified approach that replaces buffer data in the original
    /// FlatBuffer structure. For full FlatBuffer rebuilding, we'd need
    /// the flatbuffers builder.
    fn rebuild_model(
        &self,
        old_model: &TFLiteModel,
        new_tensors: &[TFLiteTensor],
    ) -> Result<Vec<u8>, PatchError> {
        // Build a map of buffer_index -> new data
        let mut buffer_updates: HashMap<u32, &[u8]> = HashMap::new();
        for tensor in new_tensors {
            buffer_updates.insert(tensor.buffer_index, &tensor.data);
        }

        // Clone the original raw data
        let mut new_data = old_model.raw_data.clone();

        // For each tensor, find and update its buffer in the raw data
        // This is a simplified approach - we patch buffers in place when sizes match
        for old_tensor in &old_model.tensors {
            if buffer_updates.contains_key(&old_tensor.buffer_index) {
                // Find this tensor's matching new version
                if let Some(new_tensor) = new_tensors.iter().find(|t| t.name == old_tensor.name) {
                    if old_tensor.data.len() == new_tensor.data.len() {
                        // Same size - can patch in place
                        if let Some(offset) = find_buffer_offset(&old_model.raw_data, &old_tensor.data) {
                            new_data[offset..offset + new_tensor.data.len()]
                                .copy_from_slice(&new_tensor.data);
                        }
                    } else {
                        // Different sizes - would need full FlatBuffer rebuild
                        // For MVP, we only support same-size updates
                        return Err(PatchError::Corrupted);
                    }
                }
            }
        }

        Ok(new_data)
    }

    /// Verify patch structure is valid
    fn verify_patch_structure(&self, patch: &Patch) -> bool {
        // Check version
        if patch.version != 1 {
            return false;
        }

        // Check all operations have valid data
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
        let mut size = 0;

        // Header: magic + version + hashes
        size += 4 + 4 + 32 + 32;

        // Operations
        for op in &patch.operations {
            size += 4; // op type tag
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

        // CRC
        size += 4;

        size
    }
}

impl Default for TFLitePatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Find the offset of a buffer in raw data
fn find_buffer_offset(raw_data: &[u8], buffer: &[u8]) -> Option<usize> {
    if buffer.is_empty() || buffer.len() > raw_data.len() {
        return None;
    }

    // Simple linear search for buffer content
    (0..=(raw_data.len() - buffer.len())).find(|&i| &raw_data[i..i + buffer.len()] == buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patcher_creation() {
        let patcher = TFLitePatcher::new();
        assert!(patcher.parser.parse(&[]).is_err());
    }

    #[test]
    fn test_find_buffer_offset() {
        let data = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let buffer = vec![3, 4, 5];
        assert_eq!(find_buffer_offset(&data, &buffer), Some(3));

        let not_found = vec![10, 11, 12];
        assert_eq!(find_buffer_offset(&data, &not_found), None);
    }

    #[test]
    fn test_verify_patch_structure_valid() {
        let patcher = TFLitePatcher::new();
        let patch = Patch {
            version: 1,
            source_hash: [0u8; 32],
            target_hash: [0u8; 32],
            operations: vec![
                PatchOperation::CopyTensor {
                    name: "test".into(),
                },
            ],
            compression: CompressionMethod::Zstd { level: 3 },
            metadata: mallorn_core::PatchMetadata::default(),
        };
        assert!(patcher.verify_patch_structure(&patch));
    }

    #[test]
    fn test_verify_patch_structure_invalid_version() {
        let patcher = TFLitePatcher::new();
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
        let patcher = TFLitePatcher::new();
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
        let patcher = TFLitePatcher::new();
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
        assert!(size > 72); // At least header + CRC
    }
}
