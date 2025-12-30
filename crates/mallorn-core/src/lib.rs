//! Mallorn Core - Shared primitives for edge model delta updates
//!
//! This crate provides the foundational types, traits, and utilities
//! used by format-specific crates (mallorn-tflite, mallorn-gguf).

pub mod chain;
pub mod compression;
pub mod diff;
pub mod error;
pub mod fingerprint;
pub mod hash;
pub mod network;
pub mod signature;
pub mod streaming;
pub mod traits;
pub mod types;

#[cfg(feature = "async")]
pub mod network_async;

// Re-export commonly used types
pub use chain::{
    chain_extension, chain_stats, deserialize_chain, find_update_path, is_chain,
    merge_chain_metadata, path_size, serialize_chain, subchain, ChainStats, CHAIN_MAGIC,
    CHAIN_VERSION,
};
pub use compression::{
    calculate_sparsity, AdaptiveCompressor, CompressedTensor, Compressor, CompressorFactory,
    DefaultCompressorFactory, DictCompressorFactory, DictionaryTrainer, Lz4Compressor,
    NeuralCompressor, ParallelCompressor, SparseCSR, SparseCompressor, SparseEncoder,
    TensorCompressionHint, TensorData, ZstdCompressor, ZstdDictCompressor,
};
pub use diff::{apply_xor_delta, xor_delta, QuantizationBlockInfo, QuantizedDelta};
pub use error::{CompressionError, DiffError, MallornError, ParseError, PatchError};
pub use fingerprint::{FingerprintDB, FingerprintError, ModelFingerprint, ModelVersion};
pub use hash::{crc32, sha256, verify_hash};
pub use network::{
    ChainInfo, DownloadConfig, DownloadState, NetworkError, PatchHeader, PatchInfo, PatchManifest,
};
pub use signature::{
    generate_keypair, is_signed_patch, load_signing_key, load_verifying_key, SignatureError,
    SignedPatch,
};
pub use streaming::{
    apply_patch_streaming, ChunkedReader, ChunkedWriter, MemoryEstimator, NoProgress, StreamConfig,
    StreamProgress, StreamingPatcher, TensorIndex, TensorLocation, DEFAULT_BUFFER_SIZE,
    MIN_BUFFER_SIZE,
};
pub use traits::{Differ, ModelFormat, Patcher};
pub use types::{
    AdaptiveStrategy, ChainError, ChainLink, ChainMetadata, CompressionDictionary,
    CompressionMethod, DataType, DeltaFormat, DictionaryMetadata, DiffOptions, LinkMetadata,
    ModelMetadata, NeuralCompressionVariant, ParsedModel, Patch, PatchChain, PatchMetadata,
    PatchOperation, PatchStats, PatchVerification, QuantizationParams, Tensor, TensorInfo,
};
