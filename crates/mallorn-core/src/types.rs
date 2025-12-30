//! Core data types for Mallorn

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Data types supported by neural network models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataType {
    // Float types
    Float32,
    Float16,
    BFloat16,
    Float64,
    // Integer types
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
    // GGUF quantized types
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
}

impl DataType {
    /// Size of one element in bytes (for non-quantized types)
    pub fn element_size(&self) -> Option<usize> {
        match self {
            DataType::Float32 => Some(4),
            DataType::Float16 | DataType::BFloat16 => Some(2),
            DataType::Float64 => Some(8),
            DataType::Int8 | DataType::UInt8 => Some(1),
            DataType::Int16 | DataType::UInt16 => Some(2),
            DataType::Int32 | DataType::UInt32 => Some(4),
            DataType::Int64 | DataType::UInt64 => Some(8),
            // Quantized types have variable size per element
            _ => None,
        }
    }

    /// Check if this is a float type
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16 | DataType::Float64
        )
    }

    /// Check if this is a quantized type
    pub fn is_quantized(&self) -> bool {
        matches!(
            self,
            DataType::Q4_0
                | DataType::Q4_1
                | DataType::Q5_0
                | DataType::Q5_1
                | DataType::Q8_0
                | DataType::Q8_1
                | DataType::Q2K
                | DataType::Q3K
                | DataType::Q4K
                | DataType::Q5K
                | DataType::Q6K
                | DataType::Q8K
        )
    }
}

/// Quantization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationParams {
    /// Scale factor
    pub scale: f32,
    /// Zero point
    pub zero_point: i32,
}

/// Model metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name/description
    pub name: Option<String>,
    /// Version info
    pub version: Option<String>,
    /// Custom metadata key-value pairs
    pub custom: HashMap<String, String>,
}

/// A tensor in a model
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Tensor name/identifier
    pub name: String,
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: DataType,
    /// Raw tensor data
    pub data: Vec<u8>,
    /// Quantization parameters (if quantized)
    pub quantization: Option<QuantizationParams>,
}

/// Lightweight tensor metadata (no data)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DataType,
    pub offset: usize,
    pub size: usize,
}

/// Operator graph (for models that expose it)
#[derive(Debug, Clone, Default)]
pub struct OperatorGraph {
    pub operators: Vec<Operator>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

/// An operator in the graph
#[derive(Debug, Clone)]
pub struct Operator {
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

/// A parsed model representation
#[derive(Debug, Clone)]
pub struct ParsedModel {
    /// Original format identifier
    pub format: String,
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Tensors in the model
    pub tensors: Vec<Tensor>,
    /// Operator graph (optional)
    pub graph: Option<OperatorGraph>,
}

/// Compression method
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionMethod {
    /// No compression
    None,
    /// Zstd compression with level
    Zstd { level: i32 },
    /// Zstd compression with pre-trained dictionary
    ZstdDict {
        level: i32,
        /// Dictionary ID for verification during decompression
        dict_id: u32,
    },
    /// LZ4 compression (fast)
    Lz4,
    /// Neural-aware compression
    Neural { variant: NeuralCompressionVariant },
    /// Adaptive compression - auto-selects best method per tensor
    Adaptive { strategy: AdaptiveStrategy },
}

impl Default for CompressionMethod {
    fn default() -> Self {
        CompressionMethod::Zstd { level: 3 }
    }
}

/// Neural compression variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeuralCompressionVariant {
    /// ZipNN-style exponent/mantissa separation
    ExponentGrouping,
    /// Byte-plane separation
    BytePlane,
}

/// Strategy for adaptive compression selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AdaptiveStrategy {
    /// Use heuristics based on dtype, size, sparsity (fast)
    #[default]
    Heuristic,
    /// Benchmark all compressors, pick best ratio (slower, better results)
    Benchmark,
}

/// Delta format for tensor patches
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeltaFormat {
    /// XOR delta
    Xor,
    /// BSDiff-style
    BsDiff,
    /// Custom tensor-aware
    TensorAware,
}

/// A patch operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatchOperation {
    /// Replace tensor entirely
    ReplaceTensor {
        name: String,
        data: Vec<u8>,
        /// Compression method used for this tensor (None = use patch default)
        #[serde(default, skip_serializing_if = "Option::is_none")]
        compression: Option<CompressionMethod>,
    },
    /// Delta update to tensor
    DeltaTensor {
        name: String,
        delta: Vec<u8>,
        delta_format: DeltaFormat,
        /// Compression method used for this delta (None = use patch default)
        #[serde(default, skip_serializing_if = "Option::is_none")]
        compression: Option<CompressionMethod>,
    },
    /// Update metadata only
    UpdateMetadata { key: String, value: String },
    /// Copy tensor unchanged (for reference)
    CopyTensor { name: String },
}

/// Patch metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PatchMetadata {
    /// Source model version
    pub source_version: Option<String>,
    /// Target model version
    pub target_version: Option<String>,
    /// Creation timestamp (Unix epoch)
    pub created_at: u64,
    /// Patch description
    pub description: Option<String>,
}

/// A model patch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Patch {
    /// Patch format version
    pub version: u32,
    /// Source model hash (SHA256)
    pub source_hash: [u8; 32],
    /// Target model hash (SHA256)
    pub target_hash: [u8; 32],
    /// Patch operations
    pub operations: Vec<PatchOperation>,
    /// Compression method used
    pub compression: CompressionMethod,
    /// Patch metadata
    pub metadata: PatchMetadata,
}

impl Patch {
    /// Count of modified tensors
    pub fn modified_count(&self) -> usize {
        self.operations
            .iter()
            .filter(|op| {
                matches!(
                    op,
                    PatchOperation::ReplaceTensor { .. } | PatchOperation::DeltaTensor { .. }
                )
            })
            .count()
    }

    /// Count of unchanged tensors
    pub fn unchanged_count(&self) -> usize {
        self.operations
            .iter()
            .filter(|op| matches!(op, PatchOperation::CopyTensor { .. }))
            .count()
    }
}

/// Patch statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PatchStats {
    /// Original model size
    pub source_size: usize,
    /// Target model size
    pub target_size: usize,
    /// Patch size
    pub patch_size: usize,
    /// Compression ratio (source_size / patch_size)
    pub compression_ratio: f64,
    /// Number of tensors modified
    pub tensors_modified: usize,
    /// Number of tensors unchanged
    pub tensors_unchanged: usize,
}

/// Result of patch verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchVerification {
    /// Source hash matches?
    pub source_valid: bool,
    /// Patch integrity valid?
    pub patch_valid: bool,
    /// Expected target hash
    pub expected_target: [u8; 32],
    /// Actual target hash (after apply)
    pub actual_target: Option<[u8; 32]>,
    /// Patch statistics
    pub stats: PatchStats,
}

/// Options for diff computation
#[derive(Debug, Clone)]
pub struct DiffOptions {
    /// Compression method for delta data
    pub compression: CompressionMethod,
    /// Minimum tensor size to diff (smaller tensors stored whole)
    pub min_tensor_size: usize,
    /// Enable neural-aware compression (ZipNN-style)
    pub neural_compression: bool,
    /// Target patch size (best-effort)
    pub target_size_hint: Option<usize>,
    /// Pre-trained dictionary for improved compression (optional)
    pub dictionary: Option<CompressionDictionary>,
}

impl Default for DiffOptions {
    fn default() -> Self {
        Self {
            compression: CompressionMethod::Zstd { level: 3 },
            min_tensor_size: 1024,
            neural_compression: false,
            target_size_hint: None,
            dictionary: None,
        }
    }
}

impl DiffOptions {
    /// Create default options
    pub fn new() -> Self {
        Self::default()
    }

    /// Set compression level
    pub fn with_compression(mut self, method: CompressionMethod) -> Self {
        self.compression = method;
        self
    }

    /// Set minimum tensor size for diffing
    pub fn with_min_tensor_size(mut self, size: usize) -> Self {
        self.min_tensor_size = size;
        self
    }

    /// Enable neural-aware compression
    pub fn with_neural_compression(mut self, enabled: bool) -> Self {
        self.neural_compression = enabled;
        self
    }

    /// Set compression dictionary
    pub fn with_dictionary(mut self, dict: CompressionDictionary) -> Self {
        self.dictionary = Some(dict);
        self
    }
}

// =============================================================================
// Delta Chain Types
// =============================================================================

/// A chain of patches for incremental model updates
///
/// Delta chains allow applying multiple sequential patches:
/// v1.0 -> v1.1 -> v1.2 -> v2.0
///
/// This is useful for:
/// - Incremental OTA updates
/// - Reducing storage (keep base + chain instead of all versions)
/// - Fast rollback to intermediate versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchChain {
    /// Chain format version
    pub version: u32,
    /// Chain identifier
    pub chain_id: String,
    /// Links in the chain (ordered from oldest to newest)
    pub links: Vec<ChainLink>,
    /// Chain metadata
    pub metadata: ChainMetadata,
}

/// A single link in a patch chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainLink {
    /// Link index in chain (0 = first patch)
    pub index: u32,
    /// Source model hash for this link
    pub source_hash: [u8; 32],
    /// Target model hash for this link
    pub target_hash: [u8; 32],
    /// The patch data (serialized)
    pub patch_data: Vec<u8>,
    /// Patch format identifier (e.g., "tflp", "ggup", "sftp")
    pub format: String,
    /// Link metadata
    pub link_metadata: LinkMetadata,
}

/// Metadata for a chain link
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LinkMetadata {
    /// Source version string
    pub source_version: Option<String>,
    /// Target version string
    pub target_version: Option<String>,
    /// Uncompressed patch size
    pub patch_size: usize,
    /// Creation timestamp
    pub created_at: u64,
}

/// Metadata for the entire chain
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChainMetadata {
    /// Base model version (chain start)
    pub base_version: Option<String>,
    /// Final model version (chain end)
    pub head_version: Option<String>,
    /// Total chain length
    pub total_links: usize,
    /// Creation timestamp
    pub created_at: u64,
    /// Description
    pub description: Option<String>,
}

impl PatchChain {
    /// Create a new empty chain
    pub fn new(chain_id: impl Into<String>) -> Self {
        Self {
            version: 1,
            chain_id: chain_id.into(),
            links: Vec::new(),
            metadata: ChainMetadata::default(),
        }
    }

    /// Add a link to the chain
    ///
    /// Returns an error if the link doesn't connect (source_hash must match
    /// previous link's target_hash, or be the first link)
    pub fn add_link(&mut self, link: ChainLink) -> Result<(), ChainError> {
        if !self.links.is_empty() {
            let last = self.links.last().unwrap();
            if last.target_hash != link.source_hash {
                return Err(ChainError::HashMismatch {
                    expected: last.target_hash,
                    got: link.source_hash,
                });
            }
        }

        self.links.push(link);
        self.update_metadata();
        Ok(())
    }

    /// Get the base (starting) hash of the chain
    pub fn base_hash(&self) -> Option<[u8; 32]> {
        self.links.first().map(|l| l.source_hash)
    }

    /// Get the head (ending) hash of the chain
    pub fn head_hash(&self) -> Option<[u8; 32]> {
        self.links.last().map(|l| l.target_hash)
    }

    /// Check if a model hash is in this chain
    pub fn contains_hash(&self, hash: &[u8; 32]) -> bool {
        for link in &self.links {
            if &link.source_hash == hash || &link.target_hash == hash {
                return true;
            }
        }
        false
    }

    /// Get the subset of links needed to go from one hash to another
    pub fn links_between(&self, from: &[u8; 32], to: &[u8; 32]) -> Option<Vec<&ChainLink>> {
        let mut start_idx = None;
        let mut end_idx = None;

        for (i, link) in self.links.iter().enumerate() {
            if &link.source_hash == from && start_idx.is_none() {
                start_idx = Some(i);
            }
            if &link.target_hash == to {
                end_idx = Some(i);
                break;
            }
        }

        match (start_idx, end_idx) {
            (Some(s), Some(e)) if s <= e => Some(self.links[s..=e].iter().collect()),
            _ => None,
        }
    }

    /// Get total size of all patch data in the chain
    pub fn total_patch_size(&self) -> usize {
        self.links.iter().map(|l| l.patch_data.len()).sum()
    }

    /// Update chain metadata based on links
    fn update_metadata(&mut self) {
        self.metadata.total_links = self.links.len();

        if let Some(first) = self.links.first() {
            self.metadata.base_version = first.link_metadata.source_version.clone();
        }

        if let Some(last) = self.links.last() {
            self.metadata.head_version = last.link_metadata.target_version.clone();
        }
    }
}

/// Errors related to patch chains
#[derive(Debug, Clone)]
pub enum ChainError {
    /// Chain link hashes don't connect
    HashMismatch { expected: [u8; 32], got: [u8; 32] },
    /// Chain is empty
    EmptyChain,
    /// Link not found
    LinkNotFound,
    /// Invalid chain structure
    InvalidChain(String),
}

impl std::fmt::Display for ChainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChainError::HashMismatch { expected, got } => {
                write!(
                    f,
                    "Chain hash mismatch: expected {:02x?}..., got {:02x?}...",
                    &expected[..4],
                    &got[..4]
                )
            }
            ChainError::EmptyChain => write!(f, "Chain is empty"),
            ChainError::LinkNotFound => write!(f, "Link not found in chain"),
            ChainError::InvalidChain(msg) => write!(f, "Invalid chain: {}", msg),
        }
    }
}

impl std::error::Error for ChainError {}

// =============================================================================
// Compression Dictionary Types
// =============================================================================

/// A pre-trained compression dictionary for improved compression ratios
///
/// Dictionaries capture common byte patterns from a family of similar models
/// (e.g., all BERT variants). Using a dictionary can improve compression
/// by 20-50% for models in the same family.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionDictionary {
    /// Dictionary version
    pub version: u32,
    /// Dictionary ID (CRC32 of data, for verification)
    pub id: u32,
    /// Raw dictionary data
    pub data: Vec<u8>,
    /// Training metadata
    pub metadata: DictionaryMetadata,
}

/// Metadata about how a dictionary was trained
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DictionaryMetadata {
    /// Number of samples used for training
    pub num_samples: usize,
    /// Total bytes of training data
    pub total_sample_bytes: usize,
    /// Optional description (e.g., "BERT family models")
    pub description: Option<String>,
    /// Creation timestamp (Unix epoch)
    pub created_at: u64,
}

impl CompressionDictionary {
    /// Create a new dictionary from raw data
    pub fn new(data: Vec<u8>) -> Self {
        let id = crc32fast::hash(&data);
        Self {
            version: 1,
            id,
            data,
            metadata: DictionaryMetadata::default(),
        }
    }

    /// Create a dictionary with metadata
    pub fn with_metadata(data: Vec<u8>, metadata: DictionaryMetadata) -> Self {
        let id = crc32fast::hash(&data);
        Self {
            version: 1,
            id,
            data,
            metadata,
        }
    }

    /// Get dictionary size in bytes
    pub fn size(&self) -> usize {
        self.data.len()
    }
}
