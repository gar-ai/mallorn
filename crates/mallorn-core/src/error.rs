//! Error types for Mallorn

use thiserror::Error;

/// Top-level error type for Mallorn operations
#[derive(Debug, Error)]
pub enum MallornError {
    #[error("Parse error: {0}")]
    Parse(#[from] ParseError),

    #[error("Diff error: {0}")]
    Diff(#[from] DiffError),

    #[error("Patch error: {0}")]
    Patch(#[from] PatchError),

    #[error("Compression error: {0}")]
    Compression(#[from] CompressionError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Unknown format: {0}")]
    UnknownFormat(String),
}

/// Errors during model parsing
#[derive(Debug, Error)]
pub enum ParseError {
    #[error("Invalid magic bytes")]
    InvalidMagic,

    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u32),

    #[error("Malformed data: {0}")]
    Malformed(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Errors during diff computation
#[derive(Debug, Error)]
pub enum DiffError {
    #[error("Parse failed: {0}")]
    ParseFailed(String),

    #[error("Tensor alignment failed: {0}")]
    AlignmentFailed(String),

    #[error("Compression failed: {0}")]
    CompressionFailed(String),

    #[error("Unsupported operation: {0}")]
    Unsupported(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Errors during patch application
#[derive(Debug, Error)]
pub enum PatchError {
    #[error("Source hash mismatch")]
    SourceHashMismatch,

    #[error("Target hash mismatch")]
    TargetHashMismatch,

    #[error("Patch corrupted")]
    Corrupted,

    #[error("Incompatible version")]
    IncompatibleVersion,

    #[error("Missing tensor: {0}")]
    MissingTensor(String),

    #[error("Decompression failed: {0}")]
    DecompressionFailed(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Errors during compression/decompression
#[derive(Debug, Error)]
pub enum CompressionError {
    #[error("Compression failed: {0}")]
    CompressionFailed(String),

    #[error("Decompression failed: {0}")]
    DecompressionFailed(String),

    #[error("Invalid data")]
    InvalidData,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Errors during serialization
#[derive(Debug, Error)]
pub enum SerializeError {
    #[error("Serialization failed: {0}")]
    Failed(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
