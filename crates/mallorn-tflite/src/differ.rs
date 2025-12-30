//! TFLite tensor-aware differ
//!
//! Computes minimal delta patches between TFLite models by aligning
//! tensors and computing XOR deltas on their data.

use crate::parser::{TFLiteModel, TFLiteParser, TFLiteTensor};
use mallorn_core::{
    sha256, xor_delta, CompressionMethod, Compressor, DeltaFormat, DiffError, DiffOptions,
    Lz4Compressor, NeuralCompressor, Patch, PatchMetadata, PatchOperation, ZstdCompressor,
    ZstdDictCompressor,
};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// TFLite tensor-aware differ
pub struct TFLiteDiffer {
    compressor: Arc<dyn Compressor + Send + Sync>,
    options: DiffOptions,
    /// Enable parallel tensor processing
    parallel: bool,
}

impl TFLiteDiffer {
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
                // Use dictionary if available in options, otherwise fall back to regular zstd
                if let Some(ref dict) = options.dictionary {
                    Arc::new(
                        ZstdDictCompressor::new(*level, dict)
                            .unwrap_or_else(|_| panic!("Failed to create ZstdDictCompressor")),
                    )
                } else {
                    Arc::new(ZstdCompressor::new(*level))
                }
            }
            CompressionMethod::Lz4 => Arc::new(Lz4Compressor::new()),
            CompressionMethod::Neural { .. } => Arc::new(NeuralCompressor::new(3)),
            CompressionMethod::None => Arc::new(ZstdCompressor::new(1)),
            CompressionMethod::Adaptive { .. } => Arc::new(ZstdCompressor::new(3)),
        };

        Self {
            compressor,
            options,
            parallel: true, // Enable parallel by default
        }
    }

    /// Enable or disable parallel tensor processing
    pub fn with_parallel(mut self, enabled: bool) -> Self {
        self.parallel = enabled;
        self
    }

    /// Diff two TFLite models from raw bytes
    pub fn diff_from_bytes(&self, old_data: &[u8], new_data: &[u8]) -> Result<Patch, DiffError> {
        let parser = TFLiteParser::new();

        let old_model = parser
            .parse(old_data)
            .map_err(|e| DiffError::ParseFailed(e.to_string()))?;
        let new_model = parser
            .parse(new_data)
            .map_err(|e| DiffError::ParseFailed(e.to_string()))?;

        self.diff(&old_model, &new_model)
    }

    /// Diff two parsed TFLite models
    pub fn diff(&self, old: &TFLiteModel, new: &TFLiteModel) -> Result<Patch, DiffError> {
        // Build lookup map for old tensors by name
        let old_tensor_map: HashMap<&str, &TFLiteTensor> =
            old.tensors.iter().map(|t| (t.name.as_str(), t)).collect();

        // Process tensors - parallel or sequential based on config
        let operations: Vec<PatchOperation> = if self.parallel {
            // Parallel processing using rayon
            new.tensors
                .par_iter()
                .map(|new_tensor| {
                    if let Some(old_tensor) = old_tensor_map.get(new_tensor.name.as_str()) {
                        self.diff_tensor(old_tensor, new_tensor)
                    } else {
                        self.add_tensor(new_tensor)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            // Sequential processing
            new.tensors
                .iter()
                .map(|new_tensor| {
                    if let Some(old_tensor) = old_tensor_map.get(new_tensor.name.as_str()) {
                        self.diff_tensor(old_tensor, new_tensor)
                    } else {
                        self.add_tensor(new_tensor)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        // Compute hashes
        let source_hash = sha256(&old.raw_data);
        let target_hash = sha256(&new.raw_data);

        Ok(Patch {
            version: 1,
            source_hash,
            target_hash,
            operations,
            compression: self.compressor.method(),
            metadata: PatchMetadata {
                source_version: old.description.clone(),
                target_version: new.description.clone(),
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
                description: Some("TFLite model patch".into()),
            },
        })
    }

    /// Diff a single tensor
    fn diff_tensor(
        &self,
        old: &TFLiteTensor,
        new: &TFLiteTensor,
    ) -> Result<PatchOperation, DiffError> {
        // If data is identical, just copy
        if old.data == new.data {
            return Ok(PatchOperation::CopyTensor {
                name: new.name.clone(),
            });
        }

        // If tensor is too small, just replace entirely
        if new.data.len() < self.options.min_tensor_size {
            let compressed = self
                .compressor
                .compress(&new.data, new.dtype.into())
                .map_err(|e| DiffError::CompressionFailed(e.to_string()))?;

            return Ok(PatchOperation::ReplaceTensor {
                name: new.name.clone(),
                data: compressed,
                compression: None,
            });
        }

        // Compute XOR delta
        let delta = if old.data.len() == new.data.len() {
            xor_delta(&old.data, &new.data)
        } else {
            // Different sizes - need to handle carefully
            // For now, use XOR on overlapping portion + append remainder
            let mut delta = xor_delta(&old.data, &new.data);

            // If new is longer, append the extra bytes
            if new.data.len() > old.data.len() {
                delta.extend_from_slice(&new.data[old.data.len()..]);
            }

            delta
        };

        // Compress the delta
        let compressed_delta = self
            .compressor
            .compress(&delta, new.dtype.into())
            .map_err(|e| DiffError::CompressionFailed(e.to_string()))?;

        // Check if delta is actually smaller than full replacement
        let compressed_full = self
            .compressor
            .compress(&new.data, new.dtype.into())
            .map_err(|e| DiffError::CompressionFailed(e.to_string()))?;

        if compressed_delta.len() < compressed_full.len() {
            Ok(PatchOperation::DeltaTensor {
                name: new.name.clone(),
                delta: compressed_delta,
                delta_format: DeltaFormat::Xor,
                compression: None,
            })
        } else {
            Ok(PatchOperation::ReplaceTensor {
                name: new.name.clone(),
                data: compressed_full,
                compression: None,
            })
        }
    }

    /// Add a new tensor (full replacement)
    fn add_tensor(&self, tensor: &TFLiteTensor) -> Result<PatchOperation, DiffError> {
        let compressed = self
            .compressor
            .compress(&tensor.data, tensor.dtype.into())
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

impl Default for TFLiteDiffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_differ_creation() {
        let differ = TFLiteDiffer::new();
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
        let differ = TFLiteDiffer::with_options(options);
        assert!(matches!(differ.compressor.method(), CompressionMethod::Lz4));
    }

    #[test]
    fn test_diff_identical_tensors() {
        let differ = TFLiteDiffer::new();

        let tensor = TFLiteTensor {
            name: "test".into(),
            shape: vec![1, 2, 3],
            dtype: crate::parser::TFLiteDataType::Float32,
            buffer_index: 0,
            data: vec![1, 2, 3, 4, 5, 6, 7, 8],
            quantization: None,
        };

        let op = differ.diff_tensor(&tensor, &tensor).unwrap();
        assert!(matches!(op, PatchOperation::CopyTensor { .. }));
    }

    #[test]
    fn test_diff_small_tensor_replaced() {
        let differ = TFLiteDiffer::with_options(DiffOptions {
            min_tensor_size: 100, // Set high threshold
            ..Default::default()
        });

        let old_tensor = TFLiteTensor {
            name: "test".into(),
            shape: vec![4],
            dtype: crate::parser::TFLiteDataType::Float32,
            buffer_index: 0,
            data: vec![1, 2, 3, 4],
            quantization: None,
        };

        let new_tensor = TFLiteTensor {
            name: "test".into(),
            shape: vec![4],
            dtype: crate::parser::TFLiteDataType::Float32,
            buffer_index: 0,
            data: vec![5, 6, 7, 8],
            quantization: None,
        };

        let op = differ.diff_tensor(&old_tensor, &new_tensor).unwrap();
        assert!(matches!(op, PatchOperation::ReplaceTensor { .. }));
    }
}
