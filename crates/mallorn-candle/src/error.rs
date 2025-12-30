//! Error types for GPU operations

use thiserror::Error;

/// Errors that can occur during GPU operations
#[derive(Error, Debug)]
pub enum GpuError {
    /// Device initialization failed
    #[error("GPU device error: {0}")]
    DeviceError(String),

    /// Tensor operation failed
    #[error("Tensor operation error: {0}")]
    TensorError(String),

    /// Data size mismatch
    #[error("Size mismatch: expected {expected}, got {got}")]
    SizeMismatch { expected: usize, got: usize },

    /// Invalid input data
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Candle error
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
}
