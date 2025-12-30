//! SafeTensors tensor-aware differ
//!
//! Computes minimal delta patches between SafeTensors models by aligning
//! tensors and computing XOR deltas on their data.

use crate::parser::{SafeTensorsModel, SafeTensorsParser};
use mallorn_core::{
    sha256, xor_delta, CompressionMethod, Compressor, DeltaFormat, DiffError, DiffOptions,
    Lz4Compressor, NeuralCompressor, Patch, PatchMetadata, PatchOperation, ZstdCompressor,
    ZstdDictCompressor,
};
use std::collections::HashMap;

/// SafeTensors tensor-aware differ
pub struct SafeTensorsDiffer {
    compressor: Box<dyn Compressor>,
    options: DiffOptions,
}

impl SafeTensorsDiffer {
    /// Create a new differ with default options
    pub fn new() -> Self {
        Self::with_options(DiffOptions::default())
    }

    /// Create a new differ with custom options
    pub fn with_options(options: DiffOptions) -> Self {
        let compressor: Box<dyn Compressor> = match &options.compression {
            CompressionMethod::Zstd { level } => {
                if options.neural_compression {
                    Box::new(NeuralCompressor::new(*level))
                } else {
                    Box::new(ZstdCompressor::new(*level))
                }
            }
            CompressionMethod::ZstdDict { level, .. } => {
                if let Some(ref dict) = options.dictionary {
                    Box::new(
                        ZstdDictCompressor::new(*level, dict)
                            .unwrap_or_else(|_| panic!("Failed to create ZstdDictCompressor")),
                    )
                } else {
                    Box::new(ZstdCompressor::new(*level))
                }
            }
            CompressionMethod::Lz4 => Box::new(Lz4Compressor::new()),
            CompressionMethod::Neural { .. } => Box::new(NeuralCompressor::new(3)),
            CompressionMethod::None => Box::new(ZstdCompressor::new(1)),
            CompressionMethod::Adaptive { .. } => Box::new(ZstdCompressor::new(3)),
        };

        Self {
            compressor,
            options,
        }
    }

    /// Diff two SafeTensors models from raw bytes
    pub fn diff_from_bytes(&self, old_data: &[u8], new_data: &[u8]) -> Result<Patch, DiffError> {
        let parser = SafeTensorsParser::new();

        let old_model = parser
            .parse(old_data)
            .map_err(|e| DiffError::ParseFailed(e.to_string()))?;
        let new_model = parser
            .parse(new_data)
            .map_err(|e| DiffError::ParseFailed(e.to_string()))?;

        self.diff(&old_model, &new_model, old_data, new_data)
    }

    /// Diff two parsed SafeTensors models
    pub fn diff(
        &self,
        old: &SafeTensorsModel,
        new: &SafeTensorsModel,
        old_raw: &[u8],
        new_raw: &[u8],
    ) -> Result<Patch, DiffError> {
        let mut operations = Vec::new();

        // Build lookup map for old tensors by name
        let old_tensor_map: HashMap<&str, (&crate::parser::SafeTensorMeta, &[u8])> = old
            .tensors
            .iter()
            .map(|(name, meta)| {
                let data = &old.data[meta.data_offsets[0]..meta.data_offsets[1]];
                (name.as_str(), (meta, data))
            })
            .collect();

        // Process each tensor in the new model
        for (name, new_meta) in &new.tensors {
            let new_data = &new.data[new_meta.data_offsets[0]..new_meta.data_offsets[1]];

            let op = if let Some((old_meta, old_data)) = old_tensor_map.get(name.as_str()) {
                // Tensor exists in both models
                self.diff_tensor(name, old_meta, old_data, new_meta, new_data)?
            } else {
                // New tensor - store entirely
                self.add_tensor(name, new_meta, new_data)?
            };
            operations.push(op);
        }

        // Compute hashes
        let source_hash = sha256(old_raw);
        let target_hash = sha256(new_raw);

        // Extract version info from metadata if available
        let source_version = old.metadata.get("version").cloned();
        let target_version = new.metadata.get("version").cloned();

        Ok(Patch {
            version: 1,
            source_hash,
            target_hash,
            operations,
            compression: self.compressor.method(),
            metadata: PatchMetadata {
                source_version,
                target_version,
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
                description: Some("SafeTensors model patch".into()),
            },
        })
    }

    /// Diff a single tensor
    fn diff_tensor(
        &self,
        name: &str,
        _old_meta: &crate::parser::SafeTensorMeta,
        old_data: &[u8],
        new_meta: &crate::parser::SafeTensorMeta,
        new_data: &[u8],
    ) -> Result<PatchOperation, DiffError> {
        // If data is identical, just copy
        if old_data == new_data {
            return Ok(PatchOperation::CopyTensor {
                name: name.to_string(),
            });
        }

        // Infer data type for compression hints
        let dtype = infer_core_dtype(&new_meta.dtype);

        // If tensor is too small, just replace entirely
        if new_data.len() < self.options.min_tensor_size {
            let compressed = self
                .compressor
                .compress(new_data, dtype)
                .map_err(|e| DiffError::CompressionFailed(e.to_string()))?;

            return Ok(PatchOperation::ReplaceTensor {
                name: name.to_string(),
                data: compressed,
                compression: None,
            });
        }

        // Compute XOR delta
        let delta = if old_data.len() == new_data.len() {
            xor_delta(old_data, new_data)
        } else {
            // Different sizes - XOR on overlapping portion + append remainder
            let mut delta = xor_delta(old_data, new_data);

            // If new is longer, append the extra bytes
            if new_data.len() > old_data.len() {
                delta.extend_from_slice(&new_data[old_data.len()..]);
            }

            delta
        };

        // Compress the delta
        let compressed_delta = self
            .compressor
            .compress(&delta, dtype)
            .map_err(|e| DiffError::CompressionFailed(e.to_string()))?;

        // Check if delta is actually smaller than full replacement
        let compressed_full = self
            .compressor
            .compress(new_data, dtype)
            .map_err(|e| DiffError::CompressionFailed(e.to_string()))?;

        if compressed_delta.len() < compressed_full.len() {
            Ok(PatchOperation::DeltaTensor {
                name: name.to_string(),
                delta: compressed_delta,
                delta_format: DeltaFormat::Xor,
                compression: None,
            })
        } else {
            Ok(PatchOperation::ReplaceTensor {
                name: name.to_string(),
                data: compressed_full,
                compression: None,
            })
        }
    }

    /// Add a new tensor (full replacement)
    fn add_tensor(
        &self,
        name: &str,
        meta: &crate::parser::SafeTensorMeta,
        data: &[u8],
    ) -> Result<PatchOperation, DiffError> {
        let dtype = infer_core_dtype(&meta.dtype);
        let compressed = self
            .compressor
            .compress(data, dtype)
            .map_err(|e| DiffError::CompressionFailed(e.to_string()))?;

        Ok(PatchOperation::ReplaceTensor {
            name: name.to_string(),
            data: compressed,
            compression: None,
        })
    }

    /// Get the compressor being used
    pub fn compressor(&self) -> &dyn Compressor {
        self.compressor.as_ref()
    }
}

impl Default for SafeTensorsDiffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert SafeTensors dtype string to mallorn-core DataType
fn infer_core_dtype(dtype: &str) -> mallorn_core::DataType {
    match dtype.to_uppercase().as_str() {
        "F32" | "FLOAT32" => mallorn_core::DataType::Float32,
        "F16" | "FLOAT16" => mallorn_core::DataType::Float16,
        "BF16" | "BFLOAT16" => mallorn_core::DataType::BFloat16,
        "F64" | "FLOAT64" => mallorn_core::DataType::Float64,
        "I8" | "INT8" => mallorn_core::DataType::Int8,
        "U8" | "UINT8" => mallorn_core::DataType::UInt8,
        "I16" | "INT16" => mallorn_core::DataType::Int16,
        "U16" | "UINT16" => mallorn_core::DataType::UInt16,
        "I32" | "INT32" => mallorn_core::DataType::Int32,
        "U32" | "UINT32" => mallorn_core::DataType::UInt32,
        "I64" | "INT64" => mallorn_core::DataType::Int64,
        "U64" | "UINT64" => mallorn_core::DataType::UInt64,
        _ => mallorn_core::DataType::UInt8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_differ_creation() {
        let differ = SafeTensorsDiffer::new();
        assert!(matches!(
            differ.compressor.method(),
            CompressionMethod::Zstd { .. }
        ));
    }

    #[test]
    fn test_differ_with_options() {
        let options = DiffOptions {
            compression: CompressionMethod::Lz4,
            ..Default::default()
        };
        let differ = SafeTensorsDiffer::with_options(options);
        assert!(matches!(differ.compressor.method(), CompressionMethod::Lz4));
    }
}
