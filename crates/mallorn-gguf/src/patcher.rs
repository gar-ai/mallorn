//! GGUF patch application
//!
//! Applies delta patches to GGUF models, reconstructing new model
//! versions from old models and patch files.

use crate::parser::{GGMLType, GGUFModel, GGUFParser, GGUFTensor};
use mallorn_core::{
    apply_xor_delta, sha256, verify_hash, Compressor, CompressionMethod, DataType, DeltaFormat,
    Lz4Compressor, NeuralCompressor, Patch, PatchError, PatchOperation, PatchStats,
    PatchVerification, ZstdCompressor,
};
use std::collections::HashMap;

/// GGUF patch applier
pub struct GGUFPatcher {
    parser: GGUFParser,
}

impl GGUFPatcher {
    /// Create a new patcher
    pub fn new() -> Self {
        Self {
            parser: GGUFParser::new(),
        }
    }

    /// Apply a patch to GGUF model bytes, returning new model bytes
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
        let old_tensor_map: HashMap<&str, &GGUFTensor> = old_model
            .tensors
            .iter()
            .map(|t| (t.name.as_str(), t))
            .collect();

        // Get appropriate decompressor - GGUF prefers LZ4
        let decompressor: Box<dyn Compressor> = match patch.compression {
            CompressionMethod::Lz4 => Box::new(Lz4Compressor::new()),
            CompressionMethod::Zstd { level } => Box::new(ZstdCompressor::new(level)),
            CompressionMethod::Neural { .. } => Box::new(NeuralCompressor::new(3)),
            CompressionMethod::None => Box::new(Lz4Compressor::new()),
            CompressionMethod::Adaptive { .. } => Box::new(Lz4Compressor::new()),
            // ZstdDict requires dictionary for decompression - fall back to Zstd
            CompressionMethod::ZstdDict { level, .. } => Box::new(ZstdCompressor::new(level)),
        };

        // Apply operations to build new tensor data
        let mut new_tensors: Vec<GGUFTensor> = Vec::new();

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
                        .map(|t| t.ggml_type.into())
                        .unwrap_or(DataType::UInt8);

                    let decompressed = decompressor
                        .decompress(data, dtype)
                        .map_err(|e| PatchError::DecompressionFailed(e.to_string()))?;

                    if let Some(old_tensor) = old_tensor_map.get(name.as_str()) {
                        GGUFTensor {
                            name: name.clone(),
                            n_dimensions: old_tensor.n_dimensions,
                            dimensions: old_tensor.dimensions.clone(),
                            ggml_type: old_tensor.ggml_type,
                            offset: old_tensor.offset,
                            data: decompressed,
                        }
                    } else {
                        GGUFTensor {
                            name: name.clone(),
                            n_dimensions: 1,
                            dimensions: vec![decompressed.len() as u64],
                            ggml_type: GGMLType::I8,
                            offset: 0,
                            data: decompressed,
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
                        .decompress(delta, old_tensor.ggml_type.into())
                        .map_err(|e| PatchError::DecompressionFailed(e.to_string()))?;

                    let new_data = match delta_format {
                        DeltaFormat::Xor => apply_xor_delta(&old_tensor.data, &decompressed_delta),
                        _ => apply_xor_delta(&old_tensor.data, &decompressed_delta),
                    };

                    GGUFTensor {
                        name: name.clone(),
                        n_dimensions: old_tensor.n_dimensions,
                        dimensions: old_tensor.dimensions.clone(),
                        ggml_type: old_tensor.ggml_type,
                        offset: old_tensor.offset,
                        data: new_data,
                    }
                }

                PatchOperation::UpdateMetadata { .. } => {
                    continue;
                }
            };

            new_tensors.push(new_tensor);
        }

        // Rebuild GGUF model binary
        let new_model_data = self.rebuild_model(&old_model, &new_tensors)?;

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

    /// Rebuild GGUF model with new tensor data
    fn rebuild_model(
        &self,
        old_model: &GGUFModel,
        new_tensors: &[GGUFTensor],
    ) -> Result<Vec<u8>, PatchError> {
        // Build tensor name -> new data map
        let tensor_updates: HashMap<&str, &[u8]> = new_tensors
            .iter()
            .map(|t| (t.name.as_str(), t.data.as_slice()))
            .collect();

        // Clone raw data and patch in place
        let mut new_data = old_model.raw_data.clone();

        // Calculate data section start
        let alignment = old_model.alignment;
        let data_start = find_data_section_start(&old_model.raw_data, alignment);

        // Update each tensor's data in place
        for old_tensor in &old_model.tensors {
            if let Some(&new_tensor_data) = tensor_updates.get(old_tensor.name.as_str()) {
                if old_tensor.data.len() == new_tensor_data.len() {
                    let offset = data_start + old_tensor.offset as usize;
                    if offset + new_tensor_data.len() <= new_data.len() {
                        new_data[offset..offset + new_tensor_data.len()]
                            .copy_from_slice(new_tensor_data);
                    }
                } else {
                    // Different sizes not supported in-place
                    return Err(PatchError::Corrupted);
                }
            }
        }

        Ok(new_data)
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

impl Default for GGUFPatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Find the start of the data section in a GGUF file
fn find_data_section_start(data: &[u8], alignment: u64) -> usize {
    // GGUF header is at least 24 bytes, data section is aligned
    // This is a simplified heuristic - real implementation would parse header
    let min_header = 24;
    let aligned = ((min_header + alignment as usize - 1) / alignment as usize) * alignment as usize;
    aligned.min(data.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patcher_creation() {
        let patcher = GGUFPatcher::new();
        assert!(patcher.parser.parse(&[]).is_err());
    }

    #[test]
    fn test_verify_patch_structure_valid() {
        let patcher = GGUFPatcher::new();
        let patch = Patch {
            version: 1,
            source_hash: [0u8; 32],
            target_hash: [0u8; 32],
            operations: vec![PatchOperation::CopyTensor {
                name: "test".into(),
            }],
            compression: CompressionMethod::Lz4,
            metadata: mallorn_core::PatchMetadata::default(),
        };
        assert!(patcher.verify_patch_structure(&patch));
    }

    #[test]
    fn test_verify_patch_structure_invalid_version() {
        let patcher = GGUFPatcher::new();
        let patch = Patch {
            version: 99,
            source_hash: [0u8; 32],
            target_hash: [0u8; 32],
            operations: vec![],
            compression: CompressionMethod::Lz4,
            metadata: mallorn_core::PatchMetadata::default(),
        };
        assert!(!patcher.verify_patch_structure(&patch));
    }

    #[test]
    fn test_estimate_patch_size() {
        let patcher = GGUFPatcher::new();
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
            compression: CompressionMethod::Lz4,
            metadata: mallorn_core::PatchMetadata::default(),
        };

        let size = patcher.estimate_patch_size(&patch);
        assert!(size > 72);
    }

    #[test]
    fn test_find_data_section_start() {
        assert_eq!(find_data_section_start(&[0u8; 100], 32), 32);
        assert_eq!(find_data_section_start(&[0u8; 100], 64), 64);
        assert_eq!(find_data_section_start(&[0u8; 20], 32), 20); // Truncated
    }
}
