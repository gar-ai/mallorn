//! ONNX tensor differ
//!
//! Computes tensor-aware diffs between ONNX models, using Zstd
//! compression for floating-point tensor data.

use crate::parser::{ONNXModel, ONNXParser, ONNXTensor};
use mallorn_core::{
    sha256, xor_delta, CompressionMethod, Compressor, DeltaFormat, DiffError, DiffOptions,
    Lz4Compressor, NeuralCompressor, Patch, PatchMetadata, PatchOperation, ZstdCompressor,
    ZstdDictCompressor,
};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// ONNX tensor differ
pub struct ONNXDiffer {
    options: DiffOptions,
    parser: ONNXParser,
    compressor: Arc<dyn Compressor + Send + Sync>,
    /// Enable parallel tensor processing
    parallel: bool,
}

impl ONNXDiffer {
    /// Create a new differ with default options
    pub fn new() -> Self {
        Self::with_options(DiffOptions::default())
    }

    /// Create a new differ with custom options
    pub fn with_options(options: DiffOptions) -> Self {
        let compressor: Arc<dyn Compressor + Send + Sync> = match &options.compression {
            CompressionMethod::Zstd { level } => {
                if options.neural_compression {
                    Arc::new(NeuralCompressor::new(*level))
                } else {
                    Arc::new(ZstdCompressor::new(*level))
                }
            }
            CompressionMethod::ZstdDict { level, .. } => {
                if let Some(ref dict) = options.dictionary {
                    Arc::new(ZstdDictCompressor::new(*level, dict).unwrap_or_else(|_| {
                        panic!("Failed to create ZstdDictCompressor")
                    }))
                } else {
                    Arc::new(ZstdCompressor::new(*level))
                }
            }
            CompressionMethod::Lz4 => Arc::new(Lz4Compressor::new()),
            CompressionMethod::Neural { .. } => Arc::new(NeuralCompressor::new(3)),
            CompressionMethod::None => Arc::new(ZstdCompressor::new(3)), // Default to Zstd for ONNX
            CompressionMethod::Adaptive { .. } => Arc::new(ZstdCompressor::new(3)),
        };

        Self {
            options,
            parser: ONNXParser::new(),
            compressor,
            parallel: true, // Enable parallel by default
        }
    }

    /// Enable or disable parallel tensor processing
    pub fn with_parallel(mut self, enabled: bool) -> Self {
        self.parallel = enabled;
        self
    }

    /// Diff two ONNX models
    pub fn diff(&self, old: &ONNXModel, new: &ONNXModel) -> Result<Patch, DiffError> {
        // Build lookup maps
        let old_tensors: HashMap<&str, &ONNXTensor> =
            old.tensors.iter().map(|t| (t.name.as_str(), t)).collect();

        // Process tensors - parallel or sequential based on config
        let operations: Vec<PatchOperation> = if self.parallel {
            new.tensors
                .par_iter()
                .map(|new_tensor| {
                    let old_tensor = old_tensors.get(new_tensor.name.as_str()).copied();
                    self.diff_tensor(old_tensor, new_tensor)
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            new.tensors
                .iter()
                .map(|new_tensor| {
                    let old_tensor = old_tensors.get(new_tensor.name.as_str()).copied();
                    self.diff_tensor(old_tensor, new_tensor)
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        // Get timestamps
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Ok(Patch {
            version: 1,
            source_hash: sha256(&old.raw_data),
            target_hash: sha256(&new.raw_data),
            operations,
            compression: self.compressor.method(),
            metadata: PatchMetadata {
                source_version: old.producer_version.clone(),
                target_version: new.producer_version.clone(),
                created_at,
                description: Some(format!(
                    "ONNX model patch ({} tensors)",
                    new.tensors.len()
                )),
            },
        })
    }

    /// Diff a single tensor
    fn diff_tensor(
        &self,
        old_tensor: Option<&ONNXTensor>,
        new_tensor: &ONNXTensor,
    ) -> Result<PatchOperation, DiffError> {
        if let Some(old_tensor) = old_tensor {
            // Tensor exists in both models
            if old_tensor.data == new_tensor.data {
                // Unchanged - emit copy operation
                Ok(PatchOperation::CopyTensor {
                    name: new_tensor.name.clone(),
                })
            } else if new_tensor.data.len() < self.options.min_tensor_size {
                // Small tensor - replace entirely
                let compressed = self
                    .compressor
                    .compress(&new_tensor.data, new_tensor.data_type.into())
                    .map_err(|e| DiffError::CompressionFailed(e.to_string()))?;
                Ok(PatchOperation::ReplaceTensor {
                    name: new_tensor.name.clone(),
                    data: compressed,
                    compression: None,
                })
            } else if old_tensor.data.len() == new_tensor.data.len() {
                // Same size - try XOR delta
                let delta = xor_delta(&old_tensor.data, &new_tensor.data);
                let sparsity = calculate_sparsity(&delta);

                // If delta is mostly zeros, compress it
                if sparsity > 0.5 {
                    let compressed = self
                        .compressor
                        .compress(&delta, new_tensor.data_type.into())
                        .map_err(|e| DiffError::CompressionFailed(e.to_string()))?;

                    // Only use delta if it's smaller than replacement
                    if compressed.len() < new_tensor.data.len() {
                        Ok(PatchOperation::DeltaTensor {
                            name: new_tensor.name.clone(),
                            delta: compressed,
                            delta_format: DeltaFormat::Xor,
                            compression: None,
                        })
                    } else {
                        let compressed_full = self
                            .compressor
                            .compress(&new_tensor.data, new_tensor.data_type.into())
                            .map_err(|e| DiffError::CompressionFailed(e.to_string()))?;
                        Ok(PatchOperation::ReplaceTensor {
                            name: new_tensor.name.clone(),
                            data: compressed_full,
                            compression: None,
                        })
                    }
                } else {
                    // Low sparsity - just replace
                    let compressed = self
                        .compressor
                        .compress(&new_tensor.data, new_tensor.data_type.into())
                        .map_err(|e| DiffError::CompressionFailed(e.to_string()))?;
                    Ok(PatchOperation::ReplaceTensor {
                        name: new_tensor.name.clone(),
                        data: compressed,
                        compression: None,
                    })
                }
            } else {
                // Different sizes - must replace
                let compressed = self
                    .compressor
                    .compress(&new_tensor.data, new_tensor.data_type.into())
                    .map_err(|e| DiffError::CompressionFailed(e.to_string()))?;
                Ok(PatchOperation::ReplaceTensor {
                    name: new_tensor.name.clone(),
                    data: compressed,
                    compression: None,
                })
            }
        } else {
            // New tensor - add it
            let compressed = self
                .compressor
                .compress(&new_tensor.data, new_tensor.data_type.into())
                .map_err(|e| DiffError::CompressionFailed(e.to_string()))?;
            Ok(PatchOperation::ReplaceTensor {
                name: new_tensor.name.clone(),
                data: compressed,
                compression: None,
            })
        }
    }

    /// Diff two ONNX models from raw bytes
    pub fn diff_from_bytes(&self, old: &[u8], new: &[u8]) -> Result<Patch, DiffError> {
        let old_model = self
            .parser
            .parse(old)
            .map_err(|e| DiffError::ParseFailed(e.to_string()))?;
        let new_model = self
            .parser
            .parse(new)
            .map_err(|e| DiffError::ParseFailed(e.to_string()))?;
        self.diff(&old_model, &new_model)
    }
}

impl Default for ONNXDiffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate sparsity (fraction of zero bytes) in delta
fn calculate_sparsity(delta: &[u8]) -> f64 {
    if delta.is_empty() {
        return 1.0;
    }
    let zeros = delta.iter().filter(|&&b| b == 0).count();
    zeros as f64 / delta.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_differ_creation() {
        let differ = ONNXDiffer::new();
        assert!(differ.parser.parse(&[]).is_err());
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
        let _differ = ONNXDiffer::with_options(options);
    }

    #[test]
    fn test_calculate_sparsity() {
        assert_eq!(calculate_sparsity(&[]), 1.0);
        assert_eq!(calculate_sparsity(&[0, 0, 0, 0]), 1.0);
        assert_eq!(calculate_sparsity(&[1, 1, 1, 1]), 0.0);
        assert_eq!(calculate_sparsity(&[0, 0, 1, 1]), 0.5);
    }

    #[test]
    fn test_parallel_mode() {
        let differ_parallel = ONNXDiffer::new().with_parallel(true);
        let differ_sequential = ONNXDiffer::new().with_parallel(false);

        assert!(differ_parallel.parallel);
        assert!(!differ_sequential.parallel);
    }
}
