//! GGUF tensor differ
//!
//! Computes tensor-aware diffs between GGUF models, using LZ4 for
//! quantized data (faster, already compressed by quantization).

use crate::parser::{GGUFModel, GGUFParser, GGUFTensor};
use mallorn_core::{
    sha256, xor_delta, CompressionMethod, Compressor, DeltaFormat, DiffError, DiffOptions,
    Lz4Compressor, NeuralCompressor, Patch, PatchMetadata, PatchOperation, ZstdCompressor,
    ZstdDictCompressor,
};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// GGUF tensor differ
pub struct GGUFDiffer {
    options: DiffOptions,
    parser: GGUFParser,
    compressor: Arc<dyn Compressor + Send + Sync>,
    /// Enable parallel tensor processing
    parallel: bool,
}

impl GGUFDiffer {
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
            CompressionMethod::None => Arc::new(Lz4Compressor::new()), // Default to LZ4 for GGUF
            CompressionMethod::Adaptive { .. } => Arc::new(Lz4Compressor::new()),
        };

        Self {
            options,
            parser: GGUFParser::new(),
            compressor,
            parallel: true, // Enable parallel by default
        }
    }

    /// Enable or disable parallel tensor processing
    pub fn with_parallel(mut self, enabled: bool) -> Self {
        self.parallel = enabled;
        self
    }

    /// Diff two GGUF models
    pub fn diff(&self, old: &GGUFModel, new: &GGUFModel) -> Result<Patch, DiffError> {
        // Build lookup maps
        let old_tensors: HashMap<&str, &GGUFTensor> =
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
            compression: CompressionMethod::Lz4, // GGUF prefers LZ4
            metadata: PatchMetadata {
                source_version: old
                    .metadata
                    .get("general.name")
                    .and_then(|v| v.as_string())
                    .map(|s| s.to_string()),
                target_version: new
                    .metadata
                    .get("general.name")
                    .and_then(|v| v.as_string())
                    .map(|s| s.to_string()),
                created_at,
                description: Some(format!("GGUF model patch ({} tensors)", new.tensor_count)),
            },
        })
    }

    /// Diff a single tensor
    fn diff_tensor(
        &self,
        old_tensor: Option<&GGUFTensor>,
        new_tensor: &GGUFTensor,
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
                    .compress(&new_tensor.data, new_tensor.ggml_type.into())
                    .map_err(|e| DiffError::CompressionFailed(e.to_string()))?;
                Ok(PatchOperation::ReplaceTensor {
                    name: new_tensor.name.clone(),
                    data: compressed,
                    compression: None,
                })
            } else if old_tensor.data.len() == new_tensor.data.len() {
                // Same size - use XOR delta
                let delta = xor_delta(&old_tensor.data, &new_tensor.data);
                let sparsity = calculate_sparsity(&delta);

                // If delta is mostly zeros, compress it
                if sparsity > 0.5 {
                    let compressed = self
                        .compressor
                        .compress(&delta, new_tensor.ggml_type.into())
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
                            .compress(&new_tensor.data, new_tensor.ggml_type.into())
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
                        .compress(&new_tensor.data, new_tensor.ggml_type.into())
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
                    .compress(&new_tensor.data, new_tensor.ggml_type.into())
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
                .compress(&new_tensor.data, new_tensor.ggml_type.into())
                .map_err(|e| DiffError::CompressionFailed(e.to_string()))?;
            Ok(PatchOperation::ReplaceTensor {
                name: new_tensor.name.clone(),
                data: compressed,
                compression: None,
            })
        }
    }

    /// Diff two GGUF models from raw bytes
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

impl Default for GGUFDiffer {
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
    use crate::parser::GGMLType;

    #[test]
    fn test_differ_creation() {
        let differ = GGUFDiffer::new();
        assert!(differ.parser.parse(&[]).is_err());
    }

    #[test]
    fn test_differ_with_options() {
        let options = DiffOptions {
            compression: CompressionMethod::Lz4,
            min_tensor_size: 512,
            neural_compression: false,
            target_size_hint: None,
            dictionary: None,
        };
        let _differ = GGUFDiffer::with_options(options);
    }

    #[test]
    fn test_calculate_sparsity() {
        assert_eq!(calculate_sparsity(&[]), 1.0);
        assert_eq!(calculate_sparsity(&[0, 0, 0, 0]), 1.0);
        assert_eq!(calculate_sparsity(&[1, 1, 1, 1]), 0.0);
        assert_eq!(calculate_sparsity(&[0, 0, 1, 1]), 0.5);
    }

    #[test]
    fn test_diff_identical_tensors() {
        // Create minimal GGUF-like structure
        let model = GGUFModel {
            version: 3,
            tensor_count: 1,
            metadata: HashMap::new(),
            tensors: vec![GGUFTensor {
                name: "test.weight".into(),
                n_dimensions: 2,
                dimensions: vec![4, 4],
                ggml_type: GGMLType::F32,
                offset: 0,
                data: vec![0u8; 64],
            }],
            raw_data: vec![0u8; 100],
            alignment: 32,
        };

        let differ = GGUFDiffer::new();
        let patch = differ.diff(&model, &model).unwrap();

        // Identical models should produce only copy operations
        assert_eq!(patch.operations.len(), 1);
        assert!(matches!(
            &patch.operations[0],
            PatchOperation::CopyTensor { name } if name == "test.weight"
        ));
    }

    #[test]
    fn test_parallel_mode() {
        let differ_parallel = GGUFDiffer::new().with_parallel(true);
        let differ_sequential = GGUFDiffer::new().with_parallel(false);

        assert!(differ_parallel.parallel);
        assert!(!differ_sequential.parallel);
    }
}
