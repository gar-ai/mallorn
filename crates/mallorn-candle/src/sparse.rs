//! Sparse tensor encoding for mostly-zero data

use crate::{GpuDevice, GpuError};

/// Sparse encoder for tensors with many zero values
///
/// Uses a simple run-length encoding for zero runs,
/// which is common in pruned or sparse models.
pub struct SparseEncoder;

impl SparseEncoder {
    /// Encode sparse data (values below threshold become zero)
    ///
    /// Format:
    /// - If more than 50% zeros, use sparse encoding
    /// - Otherwise, return original data
    pub fn encode(_device: &GpuDevice, data: &[u8], threshold: u8) -> Result<Vec<u8>, GpuError> {
        // Count values below threshold
        let zero_count = data.iter().filter(|&&b| b <= threshold).count();
        let sparsity = zero_count as f64 / data.len() as f64;

        // Only encode if sufficiently sparse
        if sparsity < 0.5 {
            // Not sparse enough, return original with marker
            let mut result = vec![0x00]; // Marker: not sparse
            result.extend_from_slice(data);
            return Ok(result);
        }

        // Use CPU implementation (GPU would be faster for large tensors)
        Ok(Self::encode_cpu(data, threshold))
    }

    /// CPU sparse encoding
    fn encode_cpu(data: &[u8], threshold: u8) -> Vec<u8> {
        let mut result = vec![0x01]; // Marker: sparse encoded

        let mut i = 0;
        while i < data.len() {
            if data[i] <= threshold {
                // Count run of zeros
                let mut run_len = 0u32;
                while i < data.len() && data[i] <= threshold && run_len < u32::MAX {
                    run_len += 1;
                    i += 1;
                }
                // Encode run: 0x00 followed by 4-byte length
                result.push(0x00);
                result.extend_from_slice(&run_len.to_le_bytes());
            } else {
                // Non-zero value
                result.push(data[i]);
                i += 1;
            }
        }

        result
    }

    /// Decode sparse-encoded data
    pub fn decode(
        _device: &GpuDevice,
        data: &[u8],
        original_len: usize,
    ) -> Result<Vec<u8>, GpuError> {
        if data.is_empty() {
            return Err(GpuError::InvalidInput("Empty data".to_string()));
        }

        let marker = data[0];

        if marker == 0x00 {
            // Not sparse encoded, just strip marker
            if data.len() - 1 != original_len {
                return Err(GpuError::SizeMismatch {
                    expected: original_len,
                    got: data.len() - 1,
                });
            }
            return Ok(data[1..].to_vec());
        }

        if marker != 0x01 {
            return Err(GpuError::InvalidInput(format!(
                "Invalid sparse marker: 0x{:02x}",
                marker
            )));
        }

        // Decode sparse format
        Self::decode_cpu(&data[1..], original_len)
    }

    /// CPU sparse decoding
    fn decode_cpu(data: &[u8], original_len: usize) -> Result<Vec<u8>, GpuError> {
        let mut result = Vec::with_capacity(original_len);
        let mut i = 0;

        while i < data.len() && result.len() < original_len {
            if data[i] == 0x00 {
                // Zero run
                if i + 5 > data.len() {
                    return Err(GpuError::InvalidInput("Truncated run length".to_string()));
                }
                let run_len =
                    u32::from_le_bytes([data[i + 1], data[i + 2], data[i + 3], data[i + 4]])
                        as usize;
                result.resize(result.len() + run_len, 0);
                i += 5;
            } else {
                result.push(data[i]);
                i += 1;
            }
        }

        if result.len() != original_len {
            return Err(GpuError::SizeMismatch {
                expected: original_len,
                got: result.len(),
            });
        }

        Ok(result)
    }

    /// Calculate sparsity of data
    pub fn sparsity(data: &[u8], threshold: u8) -> f64 {
        let zero_count = data.iter().filter(|&&b| b <= threshold).count();
        zero_count as f64 / data.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_roundtrip() {
        let device = GpuDevice::cpu();

        // Create sparse data (80% zeros)
        let mut data = vec![0u8; 1000];
        for i in (0..1000).step_by(5) {
            data[i] = 42;
        }

        let encoded = SparseEncoder::encode(&device, &data, 0).unwrap();
        let decoded = SparseEncoder::decode(&device, &encoded, data.len()).unwrap();

        assert_eq!(decoded, data);
        // Note: Encoding may be larger for data with short zero runs.
        // The important test is roundtrip correctness above.
    }

    #[test]
    fn test_non_sparse_passthrough() {
        let device = GpuDevice::cpu();

        // Create non-sparse data
        let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();

        let encoded = SparseEncoder::encode(&device, &data, 0).unwrap();
        let decoded = SparseEncoder::decode(&device, &encoded, data.len()).unwrap();

        assert_eq!(decoded, data);
        // Not sparse, so encoded size is original + 1 (marker)
        assert_eq!(encoded.len(), data.len() + 1);
    }

    #[test]
    fn test_sparsity_calculation() {
        let data = vec![0, 0, 0, 0, 42, 0, 0, 0, 42, 0];
        assert!((SparseEncoder::sparsity(&data, 0) - 0.8).abs() < 0.01);
    }
}
