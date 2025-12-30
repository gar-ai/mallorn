//! SafeTensors patch application
//!
//! Applies delta patches to SafeTensors models, reconstructing new model
//! versions from old models and patch files.

use crate::parser::{serialize_safetensors, SafeTensorsParser};
use mallorn_core::{
    apply_xor_delta, sha256, verify_hash, CompressionMethod, Compressor, DataType, DeltaFormat,
    Lz4Compressor, ModelMetadata, NeuralCompressor, ParsedModel, Patch, PatchError, PatchOperation,
    PatchStats, PatchVerification, Tensor, ZstdCompressor,
};
use std::collections::HashMap;

/// SafeTensors patch applier
pub struct SafeTensorsPatcher {
    parser: SafeTensorsParser,
}

impl SafeTensorsPatcher {
    /// Create a new patcher
    pub fn new() -> Self {
        Self {
            parser: SafeTensorsParser::new(),
        }
    }

    /// Apply a patch to SafeTensors model bytes, returning new model bytes
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

        // Build tensor lookup: name -> (metadata, data)
        let old_tensor_map: HashMap<&str, (&crate::parser::SafeTensorMeta, Vec<u8>)> = old_model
            .tensors
            .iter()
            .map(|(name, meta)| {
                let data = old_model.data[meta.data_offsets[0]..meta.data_offsets[1]].to_vec();
                (name.as_str(), (meta, data))
            })
            .collect();

        // Get appropriate decompressor
        let decompressor: Box<dyn Compressor> = match patch.compression {
            CompressionMethod::Zstd { level } => Box::new(ZstdCompressor::new(level)),
            CompressionMethod::Lz4 => Box::new(Lz4Compressor::new()),
            CompressionMethod::Neural { .. } => Box::new(NeuralCompressor::new(3)),
            CompressionMethod::None => Box::new(ZstdCompressor::new(1)),
            CompressionMethod::Adaptive { .. } => Box::new(ZstdCompressor::new(3)),
            // ZstdDict requires dictionary for decompression - fall back to Zstd
            // In production, dictionary should be provided to patcher
            CompressionMethod::ZstdDict { level, .. } => Box::new(ZstdCompressor::new(level)),
        };

        // Apply operations to build new tensors
        let mut new_tensors: Vec<Tensor> = Vec::new();

        for op in &patch.operations {
            let new_tensor = match op {
                PatchOperation::CopyTensor { name } => {
                    // Copy unchanged tensor from old model
                    let (old_meta, old_data) = old_tensor_map
                        .get(name.as_str())
                        .ok_or_else(|| PatchError::MissingTensor(name.clone()))?;

                    Tensor {
                        name: name.clone(),
                        shape: old_meta.shape.clone(),
                        dtype: infer_core_dtype(&old_meta.dtype),
                        data: old_data.clone(),
                        quantization: None,
                    }
                }

                PatchOperation::ReplaceTensor { name, data, .. } => {
                    // Get tensor metadata from old model if exists
                    let (dtype, shape) = old_tensor_map
                        .get(name.as_str())
                        .map(|(meta, _)| (infer_core_dtype(&meta.dtype), meta.shape.clone()))
                        .unwrap_or((DataType::UInt8, vec![]));

                    // Decompress and use new data entirely
                    let decompressed = decompressor
                        .decompress(data, dtype)
                        .map_err(|e| PatchError::DecompressionFailed(e.to_string()))?;

                    // Infer shape if not known
                    let final_shape = if shape.is_empty() {
                        vec![decompressed.len()]
                    } else {
                        shape
                    };

                    Tensor {
                        name: name.clone(),
                        shape: final_shape,
                        dtype,
                        data: decompressed,
                        quantization: None,
                    }
                }

                PatchOperation::DeltaTensor {
                    name,
                    delta,
                    delta_format,
                    ..
                } => {
                    // Get old tensor
                    let (old_meta, old_data) = old_tensor_map
                        .get(name.as_str())
                        .ok_or_else(|| PatchError::MissingTensor(name.clone()))?;

                    let dtype = infer_core_dtype(&old_meta.dtype);

                    // Decompress delta
                    let decompressed_delta = decompressor
                        .decompress(delta, dtype)
                        .map_err(|e| PatchError::DecompressionFailed(e.to_string()))?;

                    // Apply delta based on format
                    let new_data = match delta_format {
                        DeltaFormat::Xor => apply_xor_delta(old_data, &decompressed_delta),
                        _ => apply_xor_delta(old_data, &decompressed_delta),
                    };

                    Tensor {
                        name: name.clone(),
                        shape: old_meta.shape.clone(),
                        dtype,
                        data: new_data,
                        quantization: None,
                    }
                }

                PatchOperation::UpdateMetadata { .. } => {
                    // Metadata updates don't produce tensors
                    continue;
                }
            };

            new_tensors.push(new_tensor);
        }

        // Rebuild SafeTensors model
        let new_parsed = ParsedModel {
            format: "safetensors".to_string(),
            metadata: ModelMetadata {
                custom: old_model.metadata.clone(),
                ..Default::default()
            },
            tensors: new_tensors,
            graph: None,
        };

        let new_model_data = serialize_safetensors(&new_parsed)
            .map_err(|e| PatchError::DecompressionFailed(e.to_string()))?;

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

impl Default for SafeTensorsPatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert SafeTensors dtype string to mallorn-core DataType
fn infer_core_dtype(dtype: &str) -> DataType {
    match dtype.to_uppercase().as_str() {
        "F32" | "FLOAT32" => DataType::Float32,
        "F16" | "FLOAT16" => DataType::Float16,
        "BF16" | "BFLOAT16" => DataType::BFloat16,
        "F64" | "FLOAT64" => DataType::Float64,
        "I8" | "INT8" => DataType::Int8,
        "U8" | "UINT8" => DataType::UInt8,
        "I16" | "INT16" => DataType::Int16,
        "U16" | "UINT16" => DataType::UInt16,
        "I32" | "INT32" => DataType::Int32,
        "U32" | "UINT32" => DataType::UInt32,
        "I64" | "INT64" => DataType::Int64,
        "U64" | "UINT64" => DataType::UInt64,
        _ => DataType::UInt8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::serialize_safetensors;

    #[test]
    fn test_patcher_creation() {
        let patcher = SafeTensorsPatcher::new();
        assert!(patcher.parser.parse(&[]).is_err());
    }

    #[test]
    fn test_verify_patch_structure_valid() {
        let patcher = SafeTensorsPatcher::new();
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
        let patcher = SafeTensorsPatcher::new();
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
    fn test_roundtrip_patch() {
        use crate::differ::SafeTensorsDiffer;

        // Create two simple models
        let model1 = ParsedModel {
            format: "safetensors".to_string(),
            metadata: ModelMetadata::default(),
            tensors: vec![Tensor {
                name: "weight".to_string(),
                shape: vec![4],
                dtype: DataType::Float32,
                data: vec![0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64], // [1.0, 2.0, 3.0, 4.0]
                quantization: None,
            }],
            graph: None,
        };

        let model2 = ParsedModel {
            format: "safetensors".to_string(),
            metadata: ModelMetadata::default(),
            tensors: vec![Tensor {
                name: "weight".to_string(),
                shape: vec![4],
                dtype: DataType::Float32,
                data: vec![0, 0, 160, 64, 0, 0, 192, 64, 0, 0, 224, 64, 0, 0, 0, 65], // [5.0, 6.0, 7.0, 8.0]
                quantization: None,
            }],
            graph: None,
        };

        // Serialize both
        let bytes1 = serialize_safetensors(&model1).unwrap();
        let bytes2 = serialize_safetensors(&model2).unwrap();

        // Create diff
        let differ = SafeTensorsDiffer::new();
        let patch = differ.diff_from_bytes(&bytes1, &bytes2).unwrap();

        // Apply patch
        let patcher = SafeTensorsPatcher::new();
        let result = patcher.apply(&bytes1, &patch).unwrap();

        // Verify result matches model2
        assert_eq!(sha256(&result), sha256(&bytes2));
    }
}
