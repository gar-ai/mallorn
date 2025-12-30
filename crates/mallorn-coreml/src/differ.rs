//! CoreML tensor-aware differ
//!
//! Computes minimal delta patches between CoreML models by aligning
//! tensors and computing XOR deltas on their data.

use crate::parser::{CoreMLModel, CoreMLParser, CoreMLTensor};
use mallorn_core::{
    sha256, xor_delta, CompressionMethod, Compressor, DeltaFormat, DiffError, DiffOptions,
    Lz4Compressor, NeuralCompressor, Patch, PatchMetadata, PatchOperation, ZstdCompressor,
    ZstdDictCompressor,
};
use std::collections::HashMap;

/// CoreML tensor-aware differ
pub struct CoreMLDiffer {
    compressor: Box<dyn Compressor>,
    options: DiffOptions,
}

impl CoreMLDiffer {
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
                    Box::new(ZstdDictCompressor::new(*level, dict).unwrap_or_else(|_| {
                        panic!("Failed to create ZstdDictCompressor")
                    }))
                } else {
                    Box::new(ZstdCompressor::new(*level))
                }
            }
            CompressionMethod::Lz4 => Box::new(Lz4Compressor::new()),
            CompressionMethod::Neural { .. } => Box::new(NeuralCompressor::new(3)),
            CompressionMethod::None => Box::new(ZstdCompressor::new(1)),
            CompressionMethod::Adaptive { .. } => Box::new(ZstdCompressor::new(3)),
        };

        Self { compressor, options }
    }

    /// Diff two CoreML models from raw weight bytes
    pub fn diff_from_bytes(&self, old_data: &[u8], new_data: &[u8]) -> Result<Patch, DiffError> {
        let parser = CoreMLParser::new();

        let old_model = parser
            .parse(old_data)
            .map_err(|e| DiffError::ParseFailed(e.to_string()))?;
        let new_model = parser
            .parse(new_data)
            .map_err(|e| DiffError::ParseFailed(e.to_string()))?;

        self.diff(&old_model, &new_model, old_data, new_data)
    }

    /// Diff two parsed CoreML models
    pub fn diff(
        &self,
        old: &CoreMLModel,
        new: &CoreMLModel,
        old_raw: &[u8],
        new_raw: &[u8],
    ) -> Result<Patch, DiffError> {
        let mut operations = Vec::new();

        // Build lookup map for old tensors by name
        let old_tensor_map: HashMap<&str, &CoreMLTensor> = old
            .tensors
            .iter()
            .map(|t| (t.name.as_str(), t))
            .collect();

        // Process each tensor in the new model
        for new_tensor in &new.tensors {
            let op = if let Some(old_tensor) = old_tensor_map.get(new_tensor.name.as_str()) {
                // Tensor exists in both models
                self.diff_tensor(&new_tensor.name, old_tensor, new_tensor)?
            } else {
                // New tensor - store entirely
                self.add_tensor(new_tensor)?
            };
            operations.push(op);
        }

        // Compute hashes
        let source_hash = sha256(old_raw);
        let target_hash = sha256(new_raw);

        Ok(Patch {
            version: 1,
            source_hash,
            target_hash,
            operations,
            compression: self.compressor.method(),
            metadata: PatchMetadata {
                source_version: Some(old.version.clone()),
                target_version: Some(new.version.clone()),
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
                description: Some("CoreML model patch".into()),
            },
        })
    }

    /// Diff a single tensor
    fn diff_tensor(
        &self,
        name: &str,
        old_tensor: &CoreMLTensor,
        new_tensor: &CoreMLTensor,
    ) -> Result<PatchOperation, DiffError> {
        // If data is identical, just copy
        if old_tensor.data == new_tensor.data {
            return Ok(PatchOperation::CopyTensor {
                name: name.to_string(),
            });
        }

        let dtype = new_tensor.dtype;

        // If tensor is too small, just replace entirely
        if new_tensor.data.len() < self.options.min_tensor_size {
            let compressed = self
                .compressor
                .compress(&new_tensor.data, dtype)
                .map_err(|e| DiffError::CompressionFailed(e.to_string()))?;

            return Ok(PatchOperation::ReplaceTensor {
                name: name.to_string(),
                data: compressed,
                compression: None,
            });
        }

        // Compute XOR delta
        let delta = if old_tensor.data.len() == new_tensor.data.len() {
            xor_delta(&old_tensor.data, &new_tensor.data)
        } else {
            // Different sizes - XOR on overlapping portion + append remainder
            let mut delta = xor_delta(&old_tensor.data, &new_tensor.data);

            if new_tensor.data.len() > old_tensor.data.len() {
                delta.extend_from_slice(&new_tensor.data[old_tensor.data.len()..]);
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
            .compress(&new_tensor.data, dtype)
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
    fn add_tensor(&self, tensor: &CoreMLTensor) -> Result<PatchOperation, DiffError> {
        let compressed = self
            .compressor
            .compress(&tensor.data, tensor.dtype)
            .map_err(|e| DiffError::CompressionFailed(e.to_string()))?;

        Ok(PatchOperation::ReplaceTensor {
            name: tensor.name.clone(),
            data: compressed,
            compression: None,
        })
    }

    /// Get the compressor being used
    pub fn compressor(&self) -> &dyn Compressor {
        self.compressor.as_ref()
    }
}

impl Default for CoreMLDiffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_differ_creation() {
        let differ = CoreMLDiffer::new();
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
        let differ = CoreMLDiffer::with_options(options);
        assert!(matches!(differ.compressor.method(), CompressionMethod::Lz4));
    }

    #[test]
    fn test_diff_identical_tensors() {
        let differ = CoreMLDiffer::new();

        let old_tensor = CoreMLTensor {
            name: "test".into(),
            shape: vec![4],
            dtype: mallorn_core::DataType::Float32,
            offset: 0,
            size: 16,
            data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        };

        let new_tensor = old_tensor.clone();

        let op = differ.diff_tensor("test", &old_tensor, &new_tensor).unwrap();
        assert!(matches!(op, PatchOperation::CopyTensor { .. }));
    }

    #[test]
    fn test_diff_small_tensor_replace() {
        let differ = CoreMLDiffer::new();

        let old_tensor = CoreMLTensor {
            name: "small".into(),
            shape: vec![2],
            dtype: mallorn_core::DataType::Float32,
            offset: 0,
            size: 8,
            data: vec![1, 2, 3, 4, 5, 6, 7, 8],
        };

        let new_tensor = CoreMLTensor {
            name: "small".into(),
            shape: vec![2],
            dtype: mallorn_core::DataType::Float32,
            offset: 0,
            size: 8,
            data: vec![8, 7, 6, 5, 4, 3, 2, 1],
        };

        let op = differ.diff_tensor("small", &old_tensor, &new_tensor).unwrap();
        assert!(matches!(op, PatchOperation::ReplaceTensor { .. }));
    }
}
