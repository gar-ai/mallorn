//! Binary diffing primitives

/// Compute XOR delta between two byte slices
///
/// The XOR delta is the most efficient way to represent changes between
/// two binary blobs of the same size. XOR has the property that:
/// - `old XOR delta = new`
/// - `new XOR delta = old`
///
/// This makes it ideal for neural network weight deltas where most values
/// change slightly between versions.
pub fn xor_delta(old: &[u8], new: &[u8]) -> Vec<u8> {
    let min_len = old.len().min(new.len());
    let mut delta = Vec::with_capacity(new.len());

    // XOR the overlapping portion
    for i in 0..min_len {
        delta.push(old[i] ^ new[i]);
    }

    // If new is longer, append the extra bytes as-is
    if new.len() > old.len() {
        delta.extend_from_slice(&new[old.len()..]);
    }

    delta
}

/// Apply XOR delta to reconstruct new data from old
///
/// Given old data and an XOR delta, produces the new data.
pub fn apply_xor_delta(old: &[u8], delta: &[u8]) -> Vec<u8> {
    let min_len = old.len().min(delta.len());
    let mut new = Vec::with_capacity(delta.len());

    // XOR the overlapping portion
    for i in 0..min_len {
        new.push(old[i] ^ delta[i]);
    }

    // If delta is longer (new data was longer than old), append extra bytes
    if delta.len() > old.len() {
        new.extend_from_slice(&delta[old.len()..]);
    }

    new
}

/// Count the number of differing bytes between two slices
pub fn diff_count(a: &[u8], b: &[u8]) -> usize {
    let min_len = a.len().min(b.len());
    let mut count = 0;

    for i in 0..min_len {
        if a[i] != b[i] {
            count += 1;
        }
    }

    // Account for length difference
    count += a.len().abs_diff(b.len());

    count
}

/// Calculate the similarity ratio between two byte slices (0.0 to 1.0)
pub fn similarity_ratio(a: &[u8], b: &[u8]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }

    let max_len = a.len().max(b.len());
    if max_len == 0 {
        return 1.0;
    }

    let diff = diff_count(a, b);
    1.0 - (diff as f64 / max_len as f64)
}

/// Check if XOR delta is worth compressing
///
/// If most of the delta is zeros (unchanged bytes), compression will be very effective.
/// Returns the ratio of non-zero bytes.
pub fn delta_density(delta: &[u8]) -> f64 {
    if delta.is_empty() {
        return 0.0;
    }

    let non_zero = delta.iter().filter(|&&b| b != 0).count();
    non_zero as f64 / delta.len() as f64
}

// =============================================================================
// Quantization-Aware Delta Computation
// =============================================================================

use crate::types::DataType;

/// Block sizes for common quantization formats
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantizationBlockInfo {
    /// Number of values per block
    pub block_size: usize,
    /// Bytes per block (including scale factors)
    pub bytes_per_block: usize,
    /// Whether this is a K-quant format (256-value superblocks)
    pub is_k_quant: bool,
}

impl QuantizationBlockInfo {
    /// Get block info for a data type
    pub fn for_dtype(dtype: DataType) -> Option<Self> {
        match dtype {
            // Legacy GGUF quantization (32 values per block)
            DataType::Q4_0 => Some(Self {
                block_size: 32,
                bytes_per_block: 18, // 2 bytes scale + 16 bytes data
                is_k_quant: false,
            }),
            DataType::Q4_1 => Some(Self {
                block_size: 32,
                bytes_per_block: 20, // 2 bytes scale + 2 bytes min + 16 bytes data
                is_k_quant: false,
            }),
            DataType::Q5_0 => Some(Self {
                block_size: 32,
                bytes_per_block: 22, // 2 bytes scale + 4 bytes high bits + 16 bytes data
                is_k_quant: false,
            }),
            DataType::Q5_1 => Some(Self {
                block_size: 32,
                bytes_per_block: 24,
                is_k_quant: false,
            }),
            DataType::Q8_0 => Some(Self {
                block_size: 32,
                bytes_per_block: 34, // 2 bytes scale + 32 bytes data
                is_k_quant: false,
            }),
            DataType::Q8_1 => Some(Self {
                block_size: 32,
                bytes_per_block: 36, // scale + sum + data
                is_k_quant: false,
            }),
            // K-quant formats (256 values per superblock)
            DataType::Q2K => Some(Self {
                block_size: 256,
                bytes_per_block: 84,
                is_k_quant: true,
            }),
            DataType::Q3K => Some(Self {
                block_size: 256,
                bytes_per_block: 110,
                is_k_quant: true,
            }),
            DataType::Q4K => Some(Self {
                block_size: 256,
                bytes_per_block: 144,
                is_k_quant: true,
            }),
            DataType::Q5K => Some(Self {
                block_size: 256,
                bytes_per_block: 176,
                is_k_quant: true,
            }),
            DataType::Q6K => Some(Self {
                block_size: 256,
                bytes_per_block: 210,
                is_k_quant: true,
            }),
            DataType::Q8K => Some(Self {
                block_size: 256,
                bytes_per_block: 292,
                is_k_quant: true,
            }),
            // Non-quantized types don't have block structure
            _ => None,
        }
    }
}

/// Quantization-aware delta computation
///
/// For quantized tensors, respecting block boundaries during XOR delta
/// computation can produce better compression. Scale factors at the start
/// of each block tend to change together and compress better when aligned.
pub struct QuantizedDelta;

impl QuantizedDelta {
    /// Compute delta respecting quantization block boundaries
    ///
    /// This produces the same result as regular XOR but organizes the
    /// computation to be block-aligned, which can help downstream compression.
    ///
    /// # Arguments
    /// * `old` - Original tensor data
    /// * `new` - New tensor data
    /// * `block_info` - Quantization block information
    pub fn compute_block_aligned(
        old: &[u8],
        new: &[u8],
        block_info: &QuantizationBlockInfo,
    ) -> Vec<u8> {
        let bytes_per_block = block_info.bytes_per_block;
        let min_len = old.len().min(new.len());
        let mut delta = Vec::with_capacity(new.len());

        // Process complete blocks
        let num_complete_blocks = min_len / bytes_per_block;

        for block_idx in 0..num_complete_blocks {
            let start = block_idx * bytes_per_block;
            let end = start + bytes_per_block;

            // XOR the entire block
            for i in start..end {
                delta.push(old[i] ^ new[i]);
            }
        }

        // Handle partial block at the end
        let remaining_start = num_complete_blocks * bytes_per_block;
        for i in remaining_start..min_len {
            delta.push(old[i] ^ new[i]);
        }

        // If new is longer, append extra bytes
        if new.len() > old.len() {
            delta.extend_from_slice(&new[old.len()..]);
        }

        delta
    }

    /// Apply block-aligned delta
    ///
    /// Inverse of `compute_block_aligned`. Mathematically equivalent to
    /// `apply_xor_delta` but processes block by block.
    pub fn apply_block_aligned(
        old: &[u8],
        delta: &[u8],
        block_info: &QuantizationBlockInfo,
    ) -> Vec<u8> {
        let bytes_per_block = block_info.bytes_per_block;
        let min_len = old.len().min(delta.len());
        let mut new = Vec::with_capacity(delta.len());

        // Process complete blocks
        let num_complete_blocks = min_len / bytes_per_block;

        for block_idx in 0..num_complete_blocks {
            let start = block_idx * bytes_per_block;
            let end = start + bytes_per_block;

            for i in start..end {
                new.push(old[i] ^ delta[i]);
            }
        }

        // Handle partial block
        let remaining_start = num_complete_blocks * bytes_per_block;
        for i in remaining_start..min_len {
            new.push(old[i] ^ delta[i]);
        }

        // If delta is longer, append extra bytes
        if delta.len() > old.len() {
            new.extend_from_slice(&delta[old.len()..]);
        }

        new
    }

    /// Smart delta computation that auto-detects quantization
    ///
    /// If the data type is quantized, uses block-aligned computation.
    /// Otherwise, falls back to regular XOR delta.
    pub fn compute(old: &[u8], new: &[u8], dtype: DataType) -> Vec<u8> {
        if let Some(block_info) = QuantizationBlockInfo::for_dtype(dtype) {
            Self::compute_block_aligned(old, new, &block_info)
        } else {
            xor_delta(old, new)
        }
    }

    /// Smart delta application that auto-detects quantization
    pub fn apply(old: &[u8], delta: &[u8], dtype: DataType) -> Vec<u8> {
        if let Some(block_info) = QuantizationBlockInfo::for_dtype(dtype) {
            Self::apply_block_aligned(old, delta, &block_info)
        } else {
            apply_xor_delta(old, delta)
        }
    }

    /// Compute delta optimized for scale factor clustering
    ///
    /// This variant separates scale factors from quantized data, allowing
    /// better compression of scale factors (which cluster tightly) separately
    /// from the quantized weights.
    ///
    /// Returns (scale_delta, data_delta) for potential separate compression.
    pub fn compute_separated(
        old: &[u8],
        new: &[u8],
        block_info: &QuantizationBlockInfo,
        scale_bytes: usize,
    ) -> (Vec<u8>, Vec<u8>) {
        let bytes_per_block = block_info.bytes_per_block;
        let data_bytes = bytes_per_block - scale_bytes;

        let min_len = old.len().min(new.len());
        let num_blocks = min_len / bytes_per_block;

        let mut scale_delta = Vec::with_capacity(num_blocks * scale_bytes);
        let mut data_delta = Vec::with_capacity(num_blocks * data_bytes);

        for block_idx in 0..num_blocks {
            let start = block_idx * bytes_per_block;

            // Scale factor delta
            for i in 0..scale_bytes {
                let idx = start + i;
                scale_delta.push(old[idx] ^ new[idx]);
            }

            // Data delta
            for i in scale_bytes..bytes_per_block {
                let idx = start + i;
                data_delta.push(old[idx] ^ new[idx]);
            }
        }

        // Handle remaining bytes as data
        let remaining_start = num_blocks * bytes_per_block;
        for i in remaining_start..min_len {
            data_delta.push(old[i] ^ new[i]);
        }

        if new.len() > old.len() {
            data_delta.extend_from_slice(&new[old.len()..]);
        }

        (scale_delta, data_delta)
    }

    /// Apply separated delta (inverse of compute_separated)
    pub fn apply_separated(
        old: &[u8],
        scale_delta: &[u8],
        data_delta: &[u8],
        block_info: &QuantizationBlockInfo,
        scale_bytes: usize,
    ) -> Vec<u8> {
        let bytes_per_block = block_info.bytes_per_block;
        let data_bytes = bytes_per_block - scale_bytes;

        let num_blocks = scale_delta.len() / scale_bytes;
        let mut new = Vec::with_capacity(
            num_blocks * bytes_per_block + data_delta.len() - num_blocks * data_bytes,
        );

        let mut scale_idx = 0;
        let mut data_idx = 0;

        for block_idx in 0..num_blocks {
            let old_start = block_idx * bytes_per_block;

            // Reconstruct scale factor
            for i in 0..scale_bytes {
                let old_val = old.get(old_start + i).copied().unwrap_or(0);
                new.push(old_val ^ scale_delta[scale_idx]);
                scale_idx += 1;
            }

            // Reconstruct data
            for i in 0..data_bytes {
                let old_val = old.get(old_start + scale_bytes + i).copied().unwrap_or(0);
                new.push(old_val ^ data_delta[data_idx]);
                data_idx += 1;
            }
        }

        // Handle remaining data
        let remaining_old_start = num_blocks * bytes_per_block;
        while data_idx < data_delta.len() {
            let old_val = old
                .get(remaining_old_start + data_idx - num_blocks * data_bytes)
                .copied()
                .unwrap_or(0);
            new.push(old_val ^ data_delta[data_idx]);
            data_idx += 1;
        }

        new
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xor_delta_identical() {
        let data = vec![1, 2, 3, 4, 5];
        let delta = xor_delta(&data, &data);
        assert!(delta.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_xor_delta_roundtrip() {
        let old = vec![1, 2, 3, 4, 5];
        let new = vec![1, 2, 9, 4, 5];
        let delta = xor_delta(&old, &new);
        let reconstructed = apply_xor_delta(&old, &delta);
        assert_eq!(reconstructed, new);
    }

    #[test]
    fn test_xor_delta_different_lengths() {
        let old = vec![1, 2, 3];
        let new = vec![1, 2, 3, 4, 5];
        let delta = xor_delta(&old, &new);
        let reconstructed = apply_xor_delta(&old, &delta);
        assert_eq!(reconstructed, new);
    }

    #[test]
    fn test_similarity_identical() {
        let data = vec![1, 2, 3, 4, 5];
        assert_eq!(similarity_ratio(&data, &data), 1.0);
    }

    #[test]
    fn test_similarity_completely_different() {
        let a = vec![0, 0, 0, 0];
        let b = vec![255, 255, 255, 255];
        assert_eq!(similarity_ratio(&a, &b), 0.0);
    }

    #[test]
    fn test_delta_density() {
        let sparse_delta = vec![0, 0, 1, 0, 0, 2, 0, 0, 0, 0];
        assert!((delta_density(&sparse_delta) - 0.2).abs() < 0.001);

        let dense_delta = vec![1, 2, 3, 4, 5];
        assert_eq!(delta_density(&dense_delta), 1.0);
    }

    // =========================================================================
    // Quantization-Aware Delta Tests
    // =========================================================================

    #[test]
    fn test_quantization_block_info_q4_0() {
        let info = QuantizationBlockInfo::for_dtype(DataType::Q4_0).unwrap();
        assert_eq!(info.block_size, 32);
        assert_eq!(info.bytes_per_block, 18);
        assert!(!info.is_k_quant);
    }

    #[test]
    fn test_quantization_block_info_q4k() {
        let info = QuantizationBlockInfo::for_dtype(DataType::Q4K).unwrap();
        assert_eq!(info.block_size, 256);
        assert_eq!(info.bytes_per_block, 144);
        assert!(info.is_k_quant);
    }

    #[test]
    fn test_quantization_block_info_non_quantized() {
        assert!(QuantizationBlockInfo::for_dtype(DataType::Float32).is_none());
        assert!(QuantizationBlockInfo::for_dtype(DataType::Float16).is_none());
        assert!(QuantizationBlockInfo::for_dtype(DataType::Int8).is_none());
    }

    #[test]
    fn test_quantized_delta_block_aligned_roundtrip() {
        // Create Q4_0-like data (18 bytes per block)
        let block_info = QuantizationBlockInfo::for_dtype(DataType::Q4_0).unwrap();
        let num_blocks = 10;
        let data_len = num_blocks * block_info.bytes_per_block;

        // Create old and new data
        let old: Vec<u8> = (0..data_len).map(|i| (i % 256) as u8).collect();
        let mut new = old.clone();
        // Modify some bytes
        new[2] = 0xFF;
        new[20] = 0xAA;
        new[50] = 0xBB;

        // Compute and apply delta
        let delta = QuantizedDelta::compute_block_aligned(&old, &new, &block_info);
        let reconstructed = QuantizedDelta::apply_block_aligned(&old, &delta, &block_info);

        assert_eq!(reconstructed, new);
    }

    #[test]
    fn test_quantized_delta_smart_compute_roundtrip() {
        // Test with quantized type
        let old: Vec<u8> = (0..180).map(|i| (i % 256) as u8).collect(); // 10 Q4_0 blocks
        let mut new = old.clone();
        new[5] = 0xFF;
        new[25] = 0xAA;

        let delta = QuantizedDelta::compute(&old, &new, DataType::Q4_0);
        let reconstructed = QuantizedDelta::apply(&old, &delta, DataType::Q4_0);
        assert_eq!(reconstructed, new);

        // Test with non-quantized type (should behave like regular xor_delta)
        let delta_f32 = QuantizedDelta::compute(&old, &new, DataType::Float32);
        let reconstructed_f32 = QuantizedDelta::apply(&old, &delta_f32, DataType::Float32);
        assert_eq!(reconstructed_f32, new);
    }

    #[test]
    fn test_quantized_delta_matches_regular_xor() {
        // Block-aligned delta should produce same result as regular XOR
        let old: Vec<u8> = (0..180).map(|i| (i % 256) as u8).collect();
        let mut new = old.clone();
        new[10] = 0xFF;
        new[100] = 0xAA;

        let regular_delta = xor_delta(&old, &new);
        let block_info = QuantizationBlockInfo::for_dtype(DataType::Q4_0).unwrap();
        let block_delta = QuantizedDelta::compute_block_aligned(&old, &new, &block_info);

        // The deltas should be identical
        assert_eq!(regular_delta, block_delta);
    }

    #[test]
    fn test_quantized_delta_separated_roundtrip() {
        // Create Q4_0-like data
        let block_info = QuantizationBlockInfo::for_dtype(DataType::Q4_0).unwrap();
        let num_blocks = 5;
        let data_len = num_blocks * block_info.bytes_per_block;
        let scale_bytes = 2; // Q4_0 has 2-byte scale

        let old: Vec<u8> = (0..data_len).map(|i| (i % 256) as u8).collect();
        let mut new = old.clone();
        // Modify scale factor (first 2 bytes of each block)
        new[0] = 0xFF;
        new[1] = 0xFE;
        // Modify data
        new[10] = 0xAA;

        let (scale_delta, data_delta) =
            QuantizedDelta::compute_separated(&old, &new, &block_info, scale_bytes);
        let reconstructed = QuantizedDelta::apply_separated(
            &old,
            &scale_delta,
            &data_delta,
            &block_info,
            scale_bytes,
        );

        assert_eq!(reconstructed, new);
    }

    #[test]
    fn test_quantized_delta_k_quant_roundtrip() {
        // Test with K-quant format (larger blocks)
        let block_info = QuantizationBlockInfo::for_dtype(DataType::Q4K).unwrap();
        let num_blocks = 3;
        let data_len = num_blocks * block_info.bytes_per_block;

        let old: Vec<u8> = (0..data_len).map(|i| (i % 256) as u8).collect();
        let mut new = old.clone();
        new[50] = 0xFF;
        new[200] = 0xAA;

        let delta = QuantizedDelta::compute_block_aligned(&old, &new, &block_info);
        let reconstructed = QuantizedDelta::apply_block_aligned(&old, &delta, &block_info);

        assert_eq!(reconstructed, new);
    }
}
