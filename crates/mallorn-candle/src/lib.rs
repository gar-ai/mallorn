//! Mallorn Candle - GPU-accelerated compression using Candle
//!
//! This crate provides GPU-accelerated tensor operations for improved
//! compression performance when creating model patches.
//!
//! # Features
//!
//! - `cuda` - Enable CUDA GPU acceleration
//! - `metal` - Enable Metal GPU acceleration (macOS)
//!
//! # Example
//!
//! ```ignore
//! use mallorn_candle::{GpuCompressor, GpuDevice};
//!
//! let device = GpuDevice::best_available();
//! let compressor = GpuCompressor::new(device)?;
//!
//! // Compute delta with GPU acceleration
//! let delta = compressor.compute_delta(&old_weights, &new_weights)?;
//! ```

mod delta;
mod device;
mod error;
mod neural;
mod sparse;

pub use delta::GpuDelta;
pub use device::GpuDevice;
pub use error::GpuError;
pub use neural::NeuralCompressor;
pub use sparse::SparseEncoder;

/// GPU-accelerated compressor for tensor data
pub struct GpuCompressor {
    device: GpuDevice,
}

impl GpuCompressor {
    /// Create a new GPU compressor with the specified device
    pub fn new(device: GpuDevice) -> Result<Self, GpuError> {
        Ok(Self { device })
    }

    /// Create a GPU compressor using the best available device
    pub fn best_available() -> Result<Self, GpuError> {
        let device = GpuDevice::best_available();
        Self::new(device)
    }

    /// Get the device being used
    pub fn device(&self) -> &GpuDevice {
        &self.device
    }

    /// Compute delta between two tensors with GPU acceleration
    ///
    /// Returns the XOR delta between old and new tensor data
    pub fn compute_delta(&self, old: &[u8], new: &[u8]) -> Result<Vec<u8>, GpuError> {
        GpuDelta::compute(&self.device, old, new)
    }

    /// Apply neural compression to tensor data
    ///
    /// Uses byte-plane separation for improved compression of float weights
    pub fn neural_compress(&self, data: &[u8]) -> Result<Vec<u8>, GpuError> {
        NeuralCompressor::compress(&self.device, data)
    }

    /// Decompress neural-compressed data
    pub fn neural_decompress(&self, data: &[u8], original_len: usize) -> Result<Vec<u8>, GpuError> {
        NeuralCompressor::decompress(&self.device, data, original_len)
    }

    /// Encode sparse tensor data (mostly zeros)
    pub fn sparse_encode(&self, data: &[u8], threshold: u8) -> Result<Vec<u8>, GpuError> {
        SparseEncoder::encode(&self.device, data, threshold)
    }

    /// Decode sparse-encoded data
    pub fn sparse_decode(&self, data: &[u8], original_len: usize) -> Result<Vec<u8>, GpuError> {
        SparseEncoder::decode(&self.device, data, original_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressor_creation() {
        let compressor = GpuCompressor::best_available();
        assert!(compressor.is_ok());
    }

    #[test]
    fn test_delta_computation() {
        let compressor = GpuCompressor::best_available().unwrap();

        let old = vec![0u8; 1024];
        let mut new = vec![0u8; 1024];
        new[100] = 42;
        new[500] = 123;

        let delta = compressor.compute_delta(&old, &new).unwrap();
        assert_eq!(delta.len(), 1024);
        assert_eq!(delta[100], 42);
        assert_eq!(delta[500], 123);
    }
}
