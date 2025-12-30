//! Neural-aware compression using byte-plane separation

use crate::{GpuDevice, GpuError};

/// Neural compressor using byte-plane separation
///
/// Float weights have structure: exponent bits are highly compressible,
/// mantissa bits less so. Separating into byte planes improves compression.
pub struct NeuralCompressor;

impl NeuralCompressor {
    /// Compress tensor data using byte-plane separation
    ///
    /// Splits 32-bit floats into 4 byte planes, which compress better
    /// because similar exponents cluster together.
    pub fn compress(device: &GpuDevice, data: &[u8]) -> Result<Vec<u8>, GpuError> {
        if data.len() % 4 != 0 {
            return Err(GpuError::InvalidInput(
                "Data length must be multiple of 4 for neural compression".to_string(),
            ));
        }

        // For CPU or small data, use CPU implementation
        if !device.is_gpu() || data.len() < 1024 * 1024 {
            return Ok(Self::compress_cpu(data));
        }

        // GPU path would use custom kernels here
        // For now, fall back to CPU
        Ok(Self::compress_cpu(data))
    }

    /// CPU implementation of byte-plane separation
    fn compress_cpu(data: &[u8]) -> Vec<u8> {
        let num_floats = data.len() / 4;

        // Separate into 4 byte planes
        let mut plane0 = Vec::with_capacity(num_floats);
        let mut plane1 = Vec::with_capacity(num_floats);
        let mut plane2 = Vec::with_capacity(num_floats);
        let mut plane3 = Vec::with_capacity(num_floats);

        for chunk in data.chunks_exact(4) {
            plane0.push(chunk[0]);
            plane1.push(chunk[1]);
            plane2.push(chunk[2]);
            plane3.push(chunk[3]);
        }

        // Concatenate planes (they'll compress better separately)
        let mut result = Vec::with_capacity(data.len());
        result.extend_from_slice(&plane0);
        result.extend_from_slice(&plane1);
        result.extend_from_slice(&plane2);
        result.extend_from_slice(&plane3);

        result
    }

    /// Decompress byte-plane separated data
    pub fn decompress(
        device: &GpuDevice,
        data: &[u8],
        original_len: usize,
    ) -> Result<Vec<u8>, GpuError> {
        if data.len() != original_len {
            return Err(GpuError::SizeMismatch {
                expected: original_len,
                got: data.len(),
            });
        }

        if original_len % 4 != 0 {
            return Err(GpuError::InvalidInput(
                "Original length must be multiple of 4".to_string(),
            ));
        }

        // For CPU or small data, use CPU implementation
        if !device.is_gpu() || data.len() < 1024 * 1024 {
            return Ok(Self::decompress_cpu(data));
        }

        // GPU path would use custom kernels here
        Ok(Self::decompress_cpu(data))
    }

    /// CPU implementation of byte-plane reconstruction
    fn decompress_cpu(data: &[u8]) -> Vec<u8> {
        let num_floats = data.len() / 4;
        let plane_size = num_floats;

        let plane0 = &data[0..plane_size];
        let plane1 = &data[plane_size..2 * plane_size];
        let plane2 = &data[2 * plane_size..3 * plane_size];
        let plane3 = &data[3 * plane_size..];

        let mut result = Vec::with_capacity(data.len());

        for i in 0..num_floats {
            result.push(plane0[i]);
            result.push(plane1[i]);
            result.push(plane2[i]);
            result.push(plane3[i]);
        }

        result
    }

    /// Delta-compress with neural awareness
    ///
    /// Combines byte-plane separation with delta encoding for
    /// optimal fine-tuning patch compression.
    pub fn delta_compress(
        device: &GpuDevice,
        old: &[u8],
        new: &[u8],
    ) -> Result<Vec<u8>, GpuError> {
        if old.len() != new.len() {
            return Err(GpuError::SizeMismatch {
                expected: old.len(),
                got: new.len(),
            });
        }

        // Compute delta first
        let delta: Vec<u8> = old.iter().zip(new.iter()).map(|(a, b)| a ^ b).collect();

        // Then apply byte-plane separation
        Self::compress(device, &delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_roundtrip() {
        let device = GpuDevice::cpu();

        // Create test data (simulated float weights)
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();

        let compressed = NeuralCompressor::compress(&device, &data).unwrap();
        let decompressed = NeuralCompressor::decompress(&device, &compressed, data.len()).unwrap();

        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_plane_separation() {
        let device = GpuDevice::cpu();

        // 4 floats = 16 bytes
        let data = vec![
            0x00, 0x01, 0x02, 0x03, // float 0
            0x10, 0x11, 0x12, 0x13, // float 1
            0x20, 0x21, 0x22, 0x23, // float 2
            0x30, 0x31, 0x32, 0x33, // float 3
        ];

        let compressed = NeuralCompressor::compress(&device, &data).unwrap();

        // Should be separated into planes
        assert_eq!(
            compressed,
            vec![
                0x00, 0x10, 0x20, 0x30, // plane 0 (byte 0 of each float)
                0x01, 0x11, 0x21, 0x31, // plane 1
                0x02, 0x12, 0x22, 0x32, // plane 2
                0x03, 0x13, 0x23, 0x33, // plane 3
            ]
        );
    }
}
