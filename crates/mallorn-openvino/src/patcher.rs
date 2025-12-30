//! OpenVINO patch application
//!
//! Applies delta patches to OpenVINO models, reconstructing new model
//! versions from old models and patch files.

use crate::parser::{OpenVINOModel, OpenVINOParser, OpenVINOTensor};
use mallorn_core::{
    apply_xor_delta, sha256, verify_hash, CompressionMethod, Compressor, DataType, DeltaFormat,
    Lz4Compressor, NeuralCompressor, Patch, PatchError, PatchOperation, PatchStats,
    PatchVerification, ZstdCompressor,
};
use std::collections::HashMap;

/// OpenVINO patch applier
pub struct OpenVINOPatcher {
    parser: OpenVINOParser,
}

impl OpenVINOPatcher {
    /// Create a new patcher
    pub fn new() -> Self {
        Self {
            parser: OpenVINOParser::new(),
        }
    }

    /// Apply a patch to OpenVINO model bytes, returning new model bytes
    ///
    /// Input is the combined XML + bin data (concatenated).
    /// Returns combined new XML + bin data.
    pub fn apply(&self, old_data: &[u8], patch: &Patch) -> Result<Vec<u8>, PatchError> {
        // Verify source hash
        if !verify_hash(old_data, &patch.source_hash) {
            return Err(PatchError::SourceHashMismatch);
        }

        // Parse old model (assumes XML only for simplicity - full impl would separate)
        let old_model = self
            .parser
            .parse(old_data)
            .map_err(|_e| PatchError::Corrupted)?;

        // Build tensor lookup
        let old_tensor_map: HashMap<&str, &OpenVINOTensor> = old_model
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
        let mut new_tensors: Vec<OpenVINOTensor> = Vec::new();

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
                        .map(|t| t.dtype)
                        .unwrap_or(DataType::Float32);

                    // Decompress and use new data entirely
                    let decompressed = decompressor
                        .decompress(data, dtype)
                        .map_err(|e| PatchError::DecompressionFailed(e.to_string()))?;

                    // Get tensor metadata from old model if exists, or create new
                    if let Some(old_tensor) = old_tensor_map.get(name.as_str()) {
                        OpenVINOTensor {
                            name: name.clone(),
                            shape: old_tensor.shape.clone(),
                            dtype: old_tensor.dtype,
                            offset: old_tensor.offset,
                            size: decompressed.len(),
                            data: decompressed,
                        }
                    } else {
                        // New tensor - infer what we can
                        OpenVINOTensor {
                            name: name.clone(),
                            shape: vec![decompressed.len() as i64],
                            dtype: DataType::Float32,
                            offset: 0,
                            size: decompressed.len(),
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
                    // Get old tensor
                    let old_tensor = old_tensor_map
                        .get(name.as_str())
                        .ok_or_else(|| PatchError::MissingTensor(name.clone()))?;

                    // Decompress delta
                    let decompressed_delta = decompressor
                        .decompress(delta, old_tensor.dtype)
                        .map_err(|e| PatchError::DecompressionFailed(e.to_string()))?;

                    // Apply delta based on format
                    let new_data = match delta_format {
                        DeltaFormat::Xor => apply_xor_delta(&old_tensor.data, &decompressed_delta),
                        _ => {
                            // For other formats, fall back to XOR for now
                            apply_xor_delta(&old_tensor.data, &decompressed_delta)
                        }
                    };

                    OpenVINOTensor {
                        name: name.clone(),
                        shape: old_tensor.shape.clone(),
                        dtype: old_tensor.dtype,
                        offset: old_tensor.offset,
                        size: new_data.len(),
                        data: new_data,
                    }
                }

                PatchOperation::UpdateMetadata { .. } => {
                    // Metadata updates don't produce tensors
                    continue;
                }
            };

            new_tensors.push(new_tensor);
        }

        // Rebuild OpenVINO model binary
        let new_model_data = self.rebuild_model(&old_model, &new_tensors)?;

        // Verify target hash
        if !verify_hash(&new_model_data, &patch.target_hash) {
            return Err(PatchError::TargetHashMismatch);
        }

        Ok(new_model_data)
    }

    /// Apply a patch with separate XML and binary inputs
    pub fn apply_with_weights(
        &self,
        old_xml: &[u8],
        old_bin: &[u8],
        patch: &Patch,
    ) -> Result<(Vec<u8>, Vec<u8>), PatchError> {
        // Combine for hash verification
        let old_combined = [old_xml, old_bin].concat();

        if !verify_hash(&old_combined, &patch.source_hash) {
            return Err(PatchError::SourceHashMismatch);
        }

        // Parse old model
        let old_model = self
            .parser
            .parse_with_weights(old_xml, old_bin)
            .map_err(|_e| PatchError::Corrupted)?;

        // Build tensor lookup
        let old_tensor_map: HashMap<&str, &OpenVINOTensor> = old_model
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

        // Apply operations
        let mut new_tensors: Vec<OpenVINOTensor> = Vec::new();

        for op in &patch.operations {
            let new_tensor = match op {
                PatchOperation::CopyTensor { name } => {
                    let old_tensor = old_tensor_map
                        .get(name.as_str())
                        .ok_or_else(|| PatchError::MissingTensor(name.clone()))?;
                    (*old_tensor).clone()
                }

                PatchOperation::ReplaceTensor { name, data, .. } => {
                    let dtype = old_tensor_map
                        .get(name.as_str())
                        .map(|t| t.dtype)
                        .unwrap_or(DataType::Float32);

                    let decompressed = decompressor
                        .decompress(data, dtype)
                        .map_err(|e| PatchError::DecompressionFailed(e.to_string()))?;

                    if let Some(old_tensor) = old_tensor_map.get(name.as_str()) {
                        OpenVINOTensor {
                            name: name.clone(),
                            shape: old_tensor.shape.clone(),
                            dtype: old_tensor.dtype,
                            offset: old_tensor.offset,
                            size: decompressed.len(),
                            data: decompressed,
                        }
                    } else {
                        OpenVINOTensor {
                            name: name.clone(),
                            shape: vec![decompressed.len() as i64],
                            dtype: DataType::Float32,
                            offset: 0,
                            size: decompressed.len(),
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
                        .decompress(delta, old_tensor.dtype)
                        .map_err(|e| PatchError::DecompressionFailed(e.to_string()))?;

                    let new_data = match delta_format {
                        DeltaFormat::Xor => apply_xor_delta(&old_tensor.data, &decompressed_delta),
                        _ => apply_xor_delta(&old_tensor.data, &decompressed_delta),
                    };

                    OpenVINOTensor {
                        name: name.clone(),
                        shape: old_tensor.shape.clone(),
                        dtype: old_tensor.dtype,
                        offset: old_tensor.offset,
                        size: new_data.len(),
                        data: new_data,
                    }
                }

                PatchOperation::UpdateMetadata { .. } => {
                    continue;
                }
            };

            new_tensors.push(new_tensor);
        }

        // Rebuild model
        let (new_xml, new_bin) = self.rebuild_model_split(&old_model, &new_tensors)?;

        // Verify target hash
        let new_combined = [&new_xml[..], &new_bin[..]].concat();
        if !verify_hash(&new_combined, &patch.target_hash) {
            return Err(PatchError::TargetHashMismatch);
        }

        Ok((new_xml, new_bin))
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

    /// Rebuild OpenVINO model with new tensor data (combined output)
    fn rebuild_model(
        &self,
        old_model: &OpenVINOModel,
        new_tensors: &[OpenVINOTensor],
    ) -> Result<Vec<u8>, PatchError> {
        // Build a map of tensor name -> new data
        let tensor_updates: HashMap<&str, &[u8]> = new_tensors
            .iter()
            .map(|t| (t.name.as_str(), t.data.as_slice()))
            .collect();

        // Clone original bin data and patch in place
        let mut new_bin = old_model.bin_data.clone();

        for old_tensor in &old_model.tensors {
            if let Some(&new_data) = tensor_updates.get(old_tensor.name.as_str()) {
                if old_tensor.data.len() == new_data.len() {
                    let offset = old_tensor.offset;
                    if offset + new_data.len() <= new_bin.len() {
                        new_bin[offset..offset + new_data.len()].copy_from_slice(new_data);
                    }
                } else {
                    // Different sizes not supported in-place
                    return Err(PatchError::Corrupted);
                }
            }
        }

        // Return combined XML + bin
        let mut result = old_model.xml_data.clone();
        result.extend_from_slice(&new_bin);
        Ok(result)
    }

    /// Rebuild OpenVINO model with new tensor data (split output)
    fn rebuild_model_split(
        &self,
        old_model: &OpenVINOModel,
        new_tensors: &[OpenVINOTensor],
    ) -> Result<(Vec<u8>, Vec<u8>), PatchError> {
        let tensor_updates: HashMap<&str, &[u8]> = new_tensors
            .iter()
            .map(|t| (t.name.as_str(), t.data.as_slice()))
            .collect();

        let mut new_bin = old_model.bin_data.clone();

        for old_tensor in &old_model.tensors {
            if let Some(&new_data) = tensor_updates.get(old_tensor.name.as_str()) {
                if old_tensor.data.len() == new_data.len() {
                    let offset = old_tensor.offset;
                    if offset + new_data.len() <= new_bin.len() {
                        new_bin[offset..offset + new_data.len()].copy_from_slice(new_data);
                    }
                } else {
                    return Err(PatchError::Corrupted);
                }
            }
        }

        Ok((old_model.xml_data.clone(), new_bin))
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

impl Default for OpenVINOPatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patcher_creation() {
        let patcher = OpenVINOPatcher::new();
        assert!(patcher.parser.parse(&[]).is_err());
    }

    #[test]
    fn test_verify_patch_structure_valid() {
        let patcher = OpenVINOPatcher::new();
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
        let patcher = OpenVINOPatcher::new();
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
        let patcher = OpenVINOPatcher::new();
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
        let patcher = OpenVINOPatcher::new();
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
