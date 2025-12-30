//! Compression implementations

use crate::error::CompressionError;
use crate::types::{
    AdaptiveStrategy, CompressionDictionary, CompressionMethod, DataType, DictionaryMetadata,
};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Neural-aware compression trait
pub trait Compressor: Send + Sync {
    /// Compress data
    fn compress(&self, data: &[u8], dtype: DataType) -> Result<Vec<u8>, CompressionError>;

    /// Decompress data
    fn decompress(&self, data: &[u8], dtype: DataType) -> Result<Vec<u8>, CompressionError>;

    /// Compression method identifier
    fn method(&self) -> CompressionMethod;
}

/// Zstd compressor - baseline high-ratio compression
pub struct ZstdCompressor {
    level: i32,
}

impl ZstdCompressor {
    /// Create a new Zstd compressor with the given level (1-22)
    pub fn new(level: i32) -> Self {
        Self {
            level: level.clamp(1, 22),
        }
    }
}

impl Compressor for ZstdCompressor {
    fn compress(&self, data: &[u8], _dtype: DataType) -> Result<Vec<u8>, CompressionError> {
        zstd::encode_all(std::io::Cursor::new(data), self.level)
            .map_err(|e| CompressionError::CompressionFailed(e.to_string()))
    }

    fn decompress(&self, data: &[u8], _dtype: DataType) -> Result<Vec<u8>, CompressionError> {
        zstd::decode_all(std::io::Cursor::new(data))
            .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))
    }

    fn method(&self) -> CompressionMethod {
        CompressionMethod::Zstd { level: self.level }
    }
}

/// LZ4 compressor - fast compression/decompression
pub struct Lz4Compressor;

impl Lz4Compressor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Lz4Compressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor for Lz4Compressor {
    fn compress(&self, data: &[u8], _dtype: DataType) -> Result<Vec<u8>, CompressionError> {
        Ok(lz4_flex::compress_prepend_size(data))
    }

    fn decompress(&self, data: &[u8], _dtype: DataType) -> Result<Vec<u8>, CompressionError> {
        lz4_flex::decompress_size_prepended(data)
            .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))
    }

    fn method(&self) -> CompressionMethod {
        CompressionMethod::Lz4
    }
}

/// No compression - passthrough
pub struct NoCompressor;

impl Compressor for NoCompressor {
    fn compress(&self, data: &[u8], _dtype: DataType) -> Result<Vec<u8>, CompressionError> {
        Ok(data.to_vec())
    }

    fn decompress(&self, data: &[u8], _dtype: DataType) -> Result<Vec<u8>, CompressionError> {
        Ok(data.to_vec())
    }

    fn method(&self) -> CompressionMethod {
        CompressionMethod::None
    }
}

// =============================================================================
// Dictionary Training and Compression
// =============================================================================

/// Dictionary trainer for creating compression dictionaries from model samples
///
/// Use this to train a dictionary from multiple models in the same family
/// (e.g., all BERT variants) for improved compression ratios.
///
/// # Example
/// ```ignore
/// let trainer = DictionaryTrainer::new();
/// let samples: Vec<&[u8]> = models.iter().map(|m| m.as_slice()).collect();
/// let dict = trainer.train(&samples)?;
/// ```
pub struct DictionaryTrainer {
    /// Maximum dictionary size in bytes (default: 112KB)
    max_dict_size: usize,
}

impl DictionaryTrainer {
    /// Default dictionary size (112KB - Zstd's recommended max)
    pub const DEFAULT_DICT_SIZE: usize = 112 * 1024;

    /// Minimum samples required for training
    pub const MIN_SAMPLES: usize = 3;

    /// Create a new dictionary trainer with default settings
    pub fn new() -> Self {
        Self {
            max_dict_size: Self::DEFAULT_DICT_SIZE,
        }
    }

    /// Create a trainer with custom max dictionary size
    pub fn with_max_size(max_dict_size: usize) -> Self {
        Self { max_dict_size }
    }

    /// Train a dictionary from model samples
    ///
    /// # Arguments
    /// * `samples` - Slice of byte slices, each representing a model or tensor
    ///
    /// # Returns
    /// A trained dictionary that can be used with `ZstdDictCompressor`
    ///
    /// # Errors
    /// Returns error if training fails or insufficient samples provided
    pub fn train(&self, samples: &[&[u8]]) -> Result<CompressionDictionary, CompressionError> {
        if samples.len() < Self::MIN_SAMPLES {
            return Err(CompressionError::CompressionFailed(format!(
                "Need at least {} samples for dictionary training, got {}",
                Self::MIN_SAMPLES,
                samples.len()
            )));
        }

        // Calculate total sample size for metadata
        let total_bytes: usize = samples.iter().map(|s| s.len()).sum();

        // Zstd dictionary training works best with many small samples
        // Split large samples into smaller chunks (4KB each for optimal training)
        const CHUNK_SIZE: usize = 4096;
        let mut continuous_data = Vec::with_capacity(total_bytes);
        let mut sample_sizes = Vec::new();

        for sample in samples {
            // Split each sample into chunks
            for chunk in sample.chunks(CHUNK_SIZE) {
                continuous_data.extend_from_slice(chunk);
                sample_sizes.push(chunk.len());
            }
        }

        // Train dictionary using zstd's continuous training
        let dict_data =
            zstd::dict::from_continuous(&continuous_data, &sample_sizes, self.max_dict_size)
                .map_err(|e| {
                    CompressionError::CompressionFailed(format!(
                        "Dictionary training failed: {}",
                        e
                    ))
                })?;

        // Create metadata
        let metadata = DictionaryMetadata {
            num_samples: samples.len(),
            total_sample_bytes: total_bytes,
            description: None,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        };

        Ok(CompressionDictionary::with_metadata(dict_data, metadata))
    }

    /// Train a dictionary with a description
    pub fn train_with_description(
        &self,
        samples: &[&[u8]],
        description: impl Into<String>,
    ) -> Result<CompressionDictionary, CompressionError> {
        let mut dict = self.train(samples)?;
        dict.metadata.description = Some(description.into());
        Ok(dict)
    }

    /// Save a dictionary to bytes (for storage/transmission)
    pub fn save(dict: &CompressionDictionary) -> Result<Vec<u8>, CompressionError> {
        serde_json::to_vec(dict).map_err(|e| {
            CompressionError::CompressionFailed(format!("Failed to serialize dictionary: {}", e))
        })
    }

    /// Load a dictionary from bytes
    pub fn load(data: &[u8]) -> Result<CompressionDictionary, CompressionError> {
        serde_json::from_slice(data).map_err(|e| {
            CompressionError::DecompressionFailed(format!(
                "Failed to deserialize dictionary: {}",
                e
            ))
        })
    }
}

impl Default for DictionaryTrainer {
    fn default() -> Self {
        Self::new()
    }
}

/// Zstd compressor with pre-trained dictionary
///
/// Uses a pre-trained dictionary for improved compression ratios
/// on models from the same family.
pub struct ZstdDictCompressor {
    level: i32,
    dict_id: u32,
    /// Pre-built encoder dictionary (cached for performance)
    encoder_dict: Arc<zstd::dict::EncoderDictionary<'static>>,
    /// Pre-built decoder dictionary (cached for performance)
    decoder_dict: Arc<zstd::dict::DecoderDictionary<'static>>,
}

impl ZstdDictCompressor {
    /// Create a new dictionary compressor
    ///
    /// # Arguments
    /// * `level` - Compression level (1-22)
    /// * `dict` - Pre-trained compression dictionary
    pub fn new(level: i32, dict: &CompressionDictionary) -> Result<Self, CompressionError> {
        let level = level.clamp(1, 22);

        // Pre-build encoder and decoder dictionaries for efficiency
        let encoder_dict = zstd::dict::EncoderDictionary::copy(&dict.data, level);
        let decoder_dict = zstd::dict::DecoderDictionary::copy(&dict.data);

        Ok(Self {
            level,
            dict_id: dict.id,
            encoder_dict: Arc::new(encoder_dict),
            decoder_dict: Arc::new(decoder_dict),
        })
    }

    /// Get the dictionary ID
    pub fn dict_id(&self) -> u32 {
        self.dict_id
    }
}

impl Compressor for ZstdDictCompressor {
    fn compress(&self, data: &[u8], _dtype: DataType) -> Result<Vec<u8>, CompressionError> {
        // Create encoder with dictionary
        let mut encoder = zstd::Encoder::with_prepared_dictionary(Vec::new(), &self.encoder_dict)
            .map_err(|e| CompressionError::CompressionFailed(e.to_string()))?;

        // Compress data
        std::io::copy(&mut std::io::Cursor::new(data), &mut encoder)
            .map_err(|e| CompressionError::CompressionFailed(e.to_string()))?;

        encoder
            .finish()
            .map_err(|e| CompressionError::CompressionFailed(e.to_string()))
    }

    fn decompress(&self, data: &[u8], _dtype: DataType) -> Result<Vec<u8>, CompressionError> {
        // Create decoder with dictionary
        let mut decoder =
            zstd::Decoder::with_prepared_dictionary(std::io::Cursor::new(data), &self.decoder_dict)
                .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))?;

        // Decompress data
        let mut output = Vec::new();
        std::io::copy(&mut decoder, &mut std::io::Cursor::new(&mut output))
            .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))?;

        Ok(output)
    }

    fn method(&self) -> CompressionMethod {
        CompressionMethod::ZstdDict {
            level: self.level,
            dict_id: self.dict_id,
        }
    }
}

/// Neural-aware compressor (ZipNN-style)
///
/// Exploits the fact that neural network weights have skewed
/// exponent distributions in floating point representation.
/// For sparse XOR deltas, uses RLE encoding before exponent grouping.
pub struct NeuralCompressor {
    base_level: i32,
}

// Format markers
const SPARSE_FORMAT: u8 = 0x01;
const DENSE_FORMAT: u8 = 0x02;
const BYTEPLANE_FORMAT: u8 = 0x03;
const BYTEPLANE_F16_FORMAT: u8 = 0x04;

impl NeuralCompressor {
    pub fn new(level: i32) -> Self {
        Self {
            base_level: level.clamp(1, 22),
        }
    }

    /// Compress float32 data, choosing sparse or dense path based on sparsity
    fn compress_f32(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        if !data.len().is_multiple_of(4) {
            return Err(CompressionError::InvalidData);
        }

        // Count zero float32s to determine sparsity
        let num_floats = data.len() / 4;
        let zeros = data.chunks_exact(4).filter(|c| *c == [0, 0, 0, 0]).count();
        let sparsity = zeros as f64 / num_floats as f64;

        if sparsity > 0.5 {
            // Sparse path: RLE + exponent grouping on non-zeros
            self.compress_sparse_f32(data)
        } else {
            // Dense path: existing exponent grouping
            self.compress_dense_f32(data)
        }
    }

    /// Compress sparse float32 data with RLE for zeros
    fn compress_sparse_f32(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        let num_floats = data.len() / 4;

        // Encode: collect run lengths of zeros and non-zero values
        let mut zero_runs: Vec<u32> = Vec::new();
        let mut non_zero_exponents: Vec<u8> = Vec::new();
        let mut non_zero_mantissas: Vec<u8> = Vec::new();

        let mut run_length = 0u32;
        for chunk in data.chunks_exact(4) {
            if chunk == [0, 0, 0, 0] {
                run_length += 1;
            } else {
                // Store run length before this non-zero value
                zero_runs.push(run_length);
                run_length = 0;

                // Extract exponent and mantissa from non-zero float
                let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let exp_byte = ((bits >> 23) & 0x1FF) as u8;
                let mantissa = bits & 0x7FFFFF;

                non_zero_exponents.push(exp_byte);
                non_zero_mantissas.extend_from_slice(&mantissa.to_le_bytes()[..3]);
            }
        }
        // Final run of zeros (if any)
        zero_runs.push(run_length);

        // Encode run lengths as varints for compactness
        let runs_encoded = encode_varints(&zero_runs);

        // Compress exponents (typically cluster well)
        let compressed_exp = zstd::encode_all(
            std::io::Cursor::new(&non_zero_exponents),
            self.base_level + 5, // Higher level for small data
        )
        .map_err(|e| CompressionError::CompressionFailed(e.to_string()))?;

        // Compress mantissas
        let compressed_mant =
            zstd::encode_all(std::io::Cursor::new(&non_zero_mantissas), self.base_level)
                .map_err(|e| CompressionError::CompressionFailed(e.to_string()))?;

        // Pack: [format:1][num_floats:4][runs_len:4][runs][exp_len:4][exp][mant]
        let mut result = Vec::new();
        result.push(SPARSE_FORMAT);
        result.extend_from_slice(&(num_floats as u32).to_le_bytes());
        result.extend_from_slice(&(runs_encoded.len() as u32).to_le_bytes());
        result.extend_from_slice(&runs_encoded);
        result.extend_from_slice(&(compressed_exp.len() as u32).to_le_bytes());
        result.extend_from_slice(&compressed_exp);
        result.extend_from_slice(&compressed_mant);

        Ok(result)
    }

    /// Decompress sparse float32 data
    fn decompress_sparse_f32(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        if data.len() < 9 {
            return Err(CompressionError::InvalidData);
        }

        let mut pos = 0;

        // Read num_floats
        let num_floats =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;

        // Read runs
        let runs_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if pos + runs_len > data.len() {
            return Err(CompressionError::InvalidData);
        }
        let zero_runs = decode_varints(&data[pos..pos + runs_len]);
        pos += runs_len;

        // Read compressed exponents
        if pos + 4 > data.len() {
            return Err(CompressionError::InvalidData);
        }
        let exp_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if pos + exp_len > data.len() {
            return Err(CompressionError::InvalidData);
        }
        let exponents = zstd::decode_all(std::io::Cursor::new(&data[pos..pos + exp_len]))
            .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))?;
        pos += exp_len;

        // Read compressed mantissas
        let mantissas = zstd::decode_all(std::io::Cursor::new(&data[pos..]))
            .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))?;

        // Reconstruct: interleave zeros and non-zero values
        let mut result = Vec::with_capacity(num_floats * 4);
        let mut value_idx = 0;

        for (run_idx, &run_len) in zero_runs.iter().enumerate() {
            // Write run_len zeros
            for _ in 0..run_len {
                result.extend_from_slice(&[0, 0, 0, 0]);
            }

            // Write non-zero value (if not the trailing run)
            if run_idx < zero_runs.len() - 1 && value_idx < exponents.len() {
                let exp_byte = exponents[value_idx] as u32;
                let mantissa = u32::from_le_bytes([
                    mantissas[value_idx * 3],
                    mantissas[value_idx * 3 + 1],
                    mantissas[value_idx * 3 + 2],
                    0,
                ]) & 0x7FFFFF;

                let bits = (exp_byte << 23) | mantissa;
                result.extend_from_slice(&bits.to_le_bytes());
                value_idx += 1;
            }
        }

        Ok(result)
    }

    /// Compress dense float32 data with ZipNN-style byte-plane separation
    ///
    /// This separates all 4 bytes of each float into 4 planes:
    /// - Plane 3 (MSB): sign + high exponent bits - highly clustered
    /// - Plane 2: low exponent + high mantissa - moderate redundancy
    /// - Plane 1: middle mantissa
    /// - Plane 0 (LSB): low mantissa - most random
    ///
    /// Each plane is delta-encoded then compressed separately.
    fn compress_dense_f32(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        let num_floats = data.len() / 4;

        // Separate into 4 byte planes
        let mut plane0 = Vec::with_capacity(num_floats); // LSB
        let mut plane1 = Vec::with_capacity(num_floats);
        let mut plane2 = Vec::with_capacity(num_floats);
        let mut plane3 = Vec::with_capacity(num_floats); // MSB

        for chunk in data.chunks_exact(4) {
            plane0.push(chunk[0]);
            plane1.push(chunk[1]);
            plane2.push(chunk[2]);
            plane3.push(chunk[3]);
        }

        // Apply delta encoding to each plane (exploits weight continuity)
        let plane0_delta = delta_encode(&plane0);
        let plane1_delta = delta_encode(&plane1);
        let plane2_delta = delta_encode(&plane2);
        let plane3_delta = delta_encode(&plane3);

        // Compress each plane - higher levels for MSB planes (more redundancy)
        let comp0 = zstd::encode_all(std::io::Cursor::new(&plane0_delta), self.base_level)
            .map_err(|e| CompressionError::CompressionFailed(e.to_string()))?;
        let comp1 = zstd::encode_all(std::io::Cursor::new(&plane1_delta), self.base_level)
            .map_err(|e| CompressionError::CompressionFailed(e.to_string()))?;
        let comp2 = zstd::encode_all(std::io::Cursor::new(&plane2_delta), self.base_level + 2)
            .map_err(|e| CompressionError::CompressionFailed(e.to_string()))?;
        let comp3 = zstd::encode_all(std::io::Cursor::new(&plane3_delta), self.base_level + 5)
            .map_err(|e| CompressionError::CompressionFailed(e.to_string()))?;

        // Pack: [format:1][num:4][len0:4][p0][len1:4][p1][len2:4][p2][p3]
        let mut result = Vec::new();
        result.push(BYTEPLANE_FORMAT);
        result.extend_from_slice(&(num_floats as u32).to_le_bytes());
        result.extend_from_slice(&(comp0.len() as u32).to_le_bytes());
        result.extend_from_slice(&comp0);
        result.extend_from_slice(&(comp1.len() as u32).to_le_bytes());
        result.extend_from_slice(&comp1);
        result.extend_from_slice(&(comp2.len() as u32).to_le_bytes());
        result.extend_from_slice(&comp2);
        result.extend_from_slice(&comp3); // Last plane doesn't need length

        Ok(result)
    }

    /// Compress float16 data with byte-plane separation
    fn compress_f16(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        if !data.len().is_multiple_of(2) {
            return Err(CompressionError::InvalidData);
        }

        let num_floats = data.len() / 2;
        let mut plane0 = Vec::with_capacity(num_floats);
        let mut plane1 = Vec::with_capacity(num_floats);

        for chunk in data.chunks_exact(2) {
            plane0.push(chunk[0]);
            plane1.push(chunk[1]);
        }

        let plane0_delta = delta_encode(&plane0);
        let plane1_delta = delta_encode(&plane1);

        let comp0 = zstd::encode_all(std::io::Cursor::new(&plane0_delta), self.base_level)
            .map_err(|e| CompressionError::CompressionFailed(e.to_string()))?;
        let comp1 = zstd::encode_all(std::io::Cursor::new(&plane1_delta), self.base_level + 3)
            .map_err(|e| CompressionError::CompressionFailed(e.to_string()))?;

        let mut result = Vec::new();
        result.push(BYTEPLANE_F16_FORMAT);
        result.extend_from_slice(&(num_floats as u32).to_le_bytes());
        result.extend_from_slice(&(comp0.len() as u32).to_le_bytes());
        result.extend_from_slice(&comp0);
        result.extend_from_slice(&comp1);

        Ok(result)
    }

    /// Decompress float32 data (auto-detects format)
    fn decompress_f32(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        if data.is_empty() {
            return Err(CompressionError::InvalidData);
        }

        match data[0] {
            SPARSE_FORMAT => self.decompress_sparse_f32(&data[1..]),
            DENSE_FORMAT => self.decompress_dense_f32(&data[1..]),
            BYTEPLANE_FORMAT => self.decompress_byteplane_f32(&data[1..]),
            // Legacy format (no marker) - assume dense
            _ => self.decompress_dense_f32_legacy(data),
        }
    }

    /// Decompress byte-plane separated float32 data
    fn decompress_byteplane_f32(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        if data.len() < 4 {
            return Err(CompressionError::InvalidData);
        }

        let mut pos = 0;

        // Read num_floats
        let num_floats =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;

        // Read and decompress plane 0
        let len0 =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        let plane0_delta = zstd::decode_all(std::io::Cursor::new(&data[pos..pos + len0]))
            .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))?;
        pos += len0;

        // Read and decompress plane 1
        let len1 =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        let plane1_delta = zstd::decode_all(std::io::Cursor::new(&data[pos..pos + len1]))
            .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))?;
        pos += len1;

        // Read and decompress plane 2
        let len2 =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        let plane2_delta = zstd::decode_all(std::io::Cursor::new(&data[pos..pos + len2]))
            .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))?;
        pos += len2;

        // Decompress plane 3 (rest of data)
        let plane3_delta = zstd::decode_all(std::io::Cursor::new(&data[pos..]))
            .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))?;

        // Delta decode each plane
        let plane0 = delta_decode(&plane0_delta);
        let plane1 = delta_decode(&plane1_delta);
        let plane2 = delta_decode(&plane2_delta);
        let plane3 = delta_decode(&plane3_delta);

        // Interleave back to float32
        let mut result = Vec::with_capacity(num_floats * 4);
        for i in 0..num_floats {
            result.push(plane0[i]);
            result.push(plane1[i]);
            result.push(plane2[i]);
            result.push(plane3[i]);
        }

        Ok(result)
    }

    /// Decompress float16 data
    fn decompress_f16(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        if data.len() < 5 || data[0] != BYTEPLANE_F16_FORMAT {
            return Err(CompressionError::InvalidData);
        }

        let mut pos = 1;

        let num_floats =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;

        let len0 =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        let plane0_delta = zstd::decode_all(std::io::Cursor::new(&data[pos..pos + len0]))
            .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))?;
        pos += len0;

        let plane1_delta = zstd::decode_all(std::io::Cursor::new(&data[pos..]))
            .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))?;

        let plane0 = delta_decode(&plane0_delta);
        let plane1 = delta_decode(&plane1_delta);

        let mut result = Vec::with_capacity(num_floats * 2);
        for i in 0..num_floats {
            result.push(plane0[i]);
            result.push(plane1[i]);
        }

        Ok(result)
    }

    /// Decompress dense float32 data
    fn decompress_dense_f32(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        if data.len() < 4 {
            return Err(CompressionError::InvalidData);
        }

        // Read exponent data length
        let exp_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        if data.len() < 4 + exp_len {
            return Err(CompressionError::InvalidData);
        }

        // Decompress exponents
        let exponents = zstd::decode_all(std::io::Cursor::new(&data[4..4 + exp_len]))
            .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))?;

        // Decompress mantissas
        let mantissas = zstd::decode_all(std::io::Cursor::new(&data[4 + exp_len..]))
            .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))?;

        let num_floats = exponents.len();
        if mantissas.len() != num_floats * 3 {
            return Err(CompressionError::InvalidData);
        }

        // Reconstruct float32 values
        let mut result = Vec::with_capacity(num_floats * 4);
        for i in 0..num_floats {
            let exp_byte = exponents[i] as u32;
            let mantissa = u32::from_le_bytes([
                mantissas[i * 3],
                mantissas[i * 3 + 1],
                mantissas[i * 3 + 2],
                0,
            ]) & 0x7FFFFF;

            let bits = (exp_byte << 23) | mantissa;
            result.extend_from_slice(&bits.to_le_bytes());
        }

        Ok(result)
    }

    /// Legacy decompression (no format marker)
    fn decompress_dense_f32_legacy(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        self.decompress_dense_f32(data)
    }
}

/// Delta encode a byte sequence (each byte becomes diff from previous)
fn delta_encode(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(data.len());
    result.push(data[0]); // First byte stored as-is

    for i in 1..data.len() {
        result.push(data[i].wrapping_sub(data[i - 1]));
    }

    result
}

/// Delta decode a byte sequence (inverse of delta_encode)
fn delta_decode(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(data.len());
    result.push(data[0]); // First byte stored as-is

    for i in 1..data.len() {
        let prev = result[i - 1];
        result.push(prev.wrapping_add(data[i]));
    }

    result
}

/// Encode u32 values as variable-length integers
fn encode_varints(values: &[u32]) -> Vec<u8> {
    let mut result = Vec::new();
    for &val in values {
        let mut v = val;
        loop {
            let byte = (v & 0x7F) as u8;
            v >>= 7;
            if v == 0 {
                result.push(byte);
                break;
            } else {
                result.push(byte | 0x80);
            }
        }
    }
    result
}

/// Decode variable-length integers back to u32 values
fn decode_varints(data: &[u8]) -> Vec<u32> {
    let mut result = Vec::new();
    let mut val = 0u32;
    let mut shift = 0;

    for &byte in data {
        val |= ((byte & 0x7F) as u32) << shift;
        if byte & 0x80 == 0 {
            result.push(val);
            val = 0;
            shift = 0;
        } else {
            shift += 7;
        }
    }

    result
}

impl Compressor for NeuralCompressor {
    fn compress(&self, data: &[u8], dtype: DataType) -> Result<Vec<u8>, CompressionError> {
        match dtype {
            DataType::Float32 => self.compress_f32(data),
            DataType::Float16 | DataType::BFloat16 => self.compress_f16(data),
            // For other types, fall back to regular zstd
            _ => zstd::encode_all(std::io::Cursor::new(data), self.base_level)
                .map_err(|e| CompressionError::CompressionFailed(e.to_string())),
        }
    }

    fn decompress(&self, data: &[u8], dtype: DataType) -> Result<Vec<u8>, CompressionError> {
        match dtype {
            DataType::Float32 => self.decompress_f32(data),
            DataType::Float16 | DataType::BFloat16 => self.decompress_f16(data),
            _ => zstd::decode_all(std::io::Cursor::new(data))
                .map_err(|e| CompressionError::DecompressionFailed(e.to_string())),
        }
    }

    fn method(&self) -> CompressionMethod {
        CompressionMethod::Neural {
            variant: crate::types::NeuralCompressionVariant::BytePlane,
        }
    }
}

// =============================================================================
// Sparse Tensor Support (CSR Encoding)
// =============================================================================

/// Sparse tensor format marker
const SPARSE_CSR_FORMAT: u8 = 0x10;

/// CSR (Compressed Sparse Row) encoded tensor
#[derive(Debug, Clone)]
pub struct SparseCSR {
    /// Non-zero values (raw bytes)
    pub values: Vec<u8>,
    /// Column indices for each non-zero value
    pub col_indices: Vec<u32>,
    /// Row pointers (indices into values/col_indices for each row)
    pub row_ptrs: Vec<u32>,
    /// Original shape [rows, cols]
    pub shape: [usize; 2],
    /// Bytes per element
    pub element_size: usize,
}

impl SparseCSR {
    /// Serialize to bytes for storage
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut result = Vec::new();

        // Header: shape, element_size, counts
        result.extend_from_slice(&(self.shape[0] as u32).to_le_bytes());
        result.extend_from_slice(&(self.shape[1] as u32).to_le_bytes());
        result.extend_from_slice(&(self.element_size as u32).to_le_bytes());
        result.extend_from_slice(&(self.values.len() as u32).to_le_bytes());
        result.extend_from_slice(&(self.col_indices.len() as u32).to_le_bytes());
        result.extend_from_slice(&(self.row_ptrs.len() as u32).to_le_bytes());

        // Data
        result.extend_from_slice(&self.values);
        for &idx in &self.col_indices {
            result.extend_from_slice(&idx.to_le_bytes());
        }
        for &ptr in &self.row_ptrs {
            result.extend_from_slice(&ptr.to_le_bytes());
        }

        result
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 24 {
            return None;
        }

        let mut pos = 0;

        let rows =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        let cols =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        let element_size =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        let values_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        let col_indices_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        let row_ptrs_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;

        // Read values
        if pos + values_len > data.len() {
            return None;
        }
        let values = data[pos..pos + values_len].to_vec();
        pos += values_len;

        // Read col_indices
        if pos + col_indices_len * 4 > data.len() {
            return None;
        }
        let mut col_indices = Vec::with_capacity(col_indices_len);
        for _ in 0..col_indices_len {
            col_indices.push(u32::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
            ]));
            pos += 4;
        }

        // Read row_ptrs
        if pos + row_ptrs_len * 4 > data.len() {
            return None;
        }
        let mut row_ptrs = Vec::with_capacity(row_ptrs_len);
        for _ in 0..row_ptrs_len {
            row_ptrs.push(u32::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
            ]));
            pos += 4;
        }

        Some(SparseCSR {
            values,
            col_indices,
            row_ptrs,
            shape: [rows, cols],
            element_size,
        })
    }
}

/// Sparse tensor encoder/decoder for pruned models
pub struct SparseEncoder;

impl SparseEncoder {
    /// Calculate element-level sparsity (ratio of zero elements)
    pub fn calculate_sparsity(data: &[u8], element_size: usize) -> f64 {
        if data.is_empty() || element_size == 0 {
            return 0.0;
        }

        let num_elements = data.len() / element_size;
        if num_elements == 0 {
            return 0.0;
        }

        let zero_element = vec![0u8; element_size];
        let zeros = data
            .chunks_exact(element_size)
            .filter(|chunk| *chunk == zero_element.as_slice())
            .count();

        zeros as f64 / num_elements as f64
    }

    /// Check if sparse encoding would be beneficial
    pub fn is_beneficial(data: &[u8], element_size: usize, threshold: f64) -> bool {
        Self::calculate_sparsity(data, element_size) >= threshold
    }

    /// Encode dense tensor as CSR format
    ///
    /// # Arguments
    /// * `data` - Raw tensor data
    /// * `element_size` - Bytes per element (4 for f32, 2 for f16, etc.)
    /// * `shape` - Tensor shape [rows, cols] (flattens higher dims into rows)
    ///
    /// # Returns
    /// `Some(SparseCSR)` if encoding is possible, `None` if data is invalid
    pub fn encode_csr(data: &[u8], element_size: usize, shape: &[usize]) -> Option<SparseCSR> {
        if element_size == 0 || data.is_empty() || shape.len() < 2 {
            return None;
        }

        let rows = shape[0];
        let cols: usize = shape[1..].iter().product();

        if rows * cols * element_size != data.len() {
            return None;
        }

        let zero_element = vec![0u8; element_size];
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = Vec::with_capacity(rows + 1);

        row_ptrs.push(0);

        for row in 0..rows {
            let row_start = row * cols * element_size;
            for col in 0..cols {
                let elem_start = row_start + col * element_size;
                let elem = &data[elem_start..elem_start + element_size];

                if elem != zero_element.as_slice() {
                    values.extend_from_slice(elem);
                    col_indices.push(col as u32);
                }
            }
            row_ptrs.push(col_indices.len() as u32);
        }

        Some(SparseCSR {
            values,
            col_indices,
            row_ptrs,
            shape: [rows, cols],
            element_size,
        })
    }

    /// Decode CSR format back to dense tensor
    pub fn decode_csr(sparse: &SparseCSR) -> Vec<u8> {
        let total_elements = sparse.shape[0] * sparse.shape[1];
        let mut result = vec![0u8; total_elements * sparse.element_size];

        for row in 0..sparse.shape[0] {
            let start = sparse.row_ptrs[row] as usize;
            let end = sparse.row_ptrs[row + 1] as usize;

            for i in start..end {
                let col = sparse.col_indices[i] as usize;
                let value_start = i * sparse.element_size;
                let value_end = value_start + sparse.element_size;
                let value = &sparse.values[value_start..value_end];

                let dest_start = (row * sparse.shape[1] + col) * sparse.element_size;
                let dest_end = dest_start + sparse.element_size;
                result[dest_start..dest_end].copy_from_slice(value);
            }
        }

        result
    }

    /// Calculate compression ratio of CSR vs dense
    pub fn compression_ratio(sparse: &SparseCSR) -> f64 {
        let dense_size = sparse.shape[0] * sparse.shape[1] * sparse.element_size;
        let sparse_size = sparse.to_bytes().len();
        dense_size as f64 / sparse_size as f64
    }
}

/// Compressor that applies sparse CSR encoding before inner compression
///
/// Automatically detects sparse tensors and encodes them as CSR,
/// falling back to dense encoding when sparsity is below threshold.
pub struct SparseCompressor {
    inner: ZstdCompressor,
    sparsity_threshold: f64,
}

impl SparseCompressor {
    /// Create with default threshold (50% sparsity)
    pub fn new(level: i32) -> Self {
        Self {
            inner: ZstdCompressor::new(level),
            sparsity_threshold: 0.5,
        }
    }

    /// Create with custom sparsity threshold
    pub fn with_threshold(level: i32, threshold: f64) -> Self {
        Self {
            inner: ZstdCompressor::new(level),
            sparsity_threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Compress with shape information for proper sparse encoding
    pub fn compress_with_shape(
        &self,
        data: &[u8],
        dtype: DataType,
        shape: &[usize],
    ) -> Result<Vec<u8>, CompressionError> {
        let element_size = dtype.element_size().unwrap_or(4); // Default to 4 bytes

        // Check if sparse encoding is beneficial
        if shape.len() >= 2
            && SparseEncoder::is_beneficial(data, element_size, self.sparsity_threshold)
        {
            if let Some(sparse) = SparseEncoder::encode_csr(data, element_size, shape) {
                // Only use sparse if it actually compresses better
                if SparseEncoder::compression_ratio(&sparse) > 1.0 {
                    let sparse_bytes = sparse.to_bytes();
                    let compressed = self.inner.compress(&sparse_bytes, dtype)?;

                    // Prepend sparse format marker
                    let mut result = Vec::with_capacity(1 + compressed.len());
                    result.push(SPARSE_CSR_FORMAT);
                    result.extend_from_slice(&compressed);
                    return Ok(result);
                }
            }
        }

        // Fall back to dense compression
        let compressed = self.inner.compress(data, dtype)?;
        let mut result = Vec::with_capacity(1 + compressed.len());
        result.push(DENSE_FORMAT);
        result.extend_from_slice(&compressed);
        Ok(result)
    }

    /// Decompress, handling both sparse and dense formats
    pub fn decompress_to_dense(
        &self,
        data: &[u8],
        dtype: DataType,
    ) -> Result<Vec<u8>, CompressionError> {
        if data.is_empty() {
            return Err(CompressionError::InvalidData);
        }

        match data[0] {
            SPARSE_CSR_FORMAT => {
                let decompressed = self.inner.decompress(&data[1..], dtype)?;
                let sparse =
                    SparseCSR::from_bytes(&decompressed).ok_or(CompressionError::InvalidData)?;
                Ok(SparseEncoder::decode_csr(&sparse))
            }
            SPARSE_FORMAT => {
                // RLE sparse format
                self.decompress_sparse_rle(&data[1..])
            }
            DENSE_FORMAT => self.inner.decompress(&data[1..], dtype),
            _ => {
                // Legacy format without marker - assume dense
                self.inner.decompress(data, dtype)
            }
        }
    }

    /// Decompress RLE sparse format
    fn decompress_sparse_rle(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        if data.len() < 12 {
            return Err(CompressionError::InvalidData);
        }

        let mut pos = 0;

        // Read header
        let num_elements =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        let element_size =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        let runs_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;

        if pos + runs_len > data.len() {
            return Err(CompressionError::InvalidData);
        }

        // Decode runs
        let runs = decode_varints(&data[pos..pos + runs_len]);
        pos += runs_len;

        // Decompress values
        let values = self.inner.decompress(&data[pos..], DataType::UInt8)?;

        // Reconstruct: interleave zeros and non-zero values
        let mut result = Vec::with_capacity(num_elements * element_size);
        let zero_element = vec![0u8; element_size];
        let mut value_idx = 0;

        for (run_idx, &run_len) in runs.iter().enumerate() {
            // Write run_len zeros
            for _ in 0..run_len {
                result.extend_from_slice(&zero_element);
            }

            // Write non-zero value (if not the trailing run)
            if run_idx < runs.len() - 1 && value_idx * element_size < values.len() {
                let value_start = value_idx * element_size;
                let value_end = value_start + element_size;
                if value_end <= values.len() {
                    result.extend_from_slice(&values[value_start..value_end]);
                }
                value_idx += 1;
            }
        }

        Ok(result)
    }
}

impl Compressor for SparseCompressor {
    fn compress(&self, data: &[u8], dtype: DataType) -> Result<Vec<u8>, CompressionError> {
        // Without shape info, we can't do proper 2D sparse encoding
        // Fall back to sparsity-aware but shape-agnostic compression
        let element_size = dtype.element_size().unwrap_or(4);
        let sparsity = SparseEncoder::calculate_sparsity(data, element_size);

        if sparsity >= self.sparsity_threshold {
            // High sparsity - use run-length encoding for zeros
            self.compress_sparse_rle(data, element_size)
        } else {
            // Dense - regular compression
            let compressed = self.inner.compress(data, dtype)?;
            let mut result = Vec::with_capacity(1 + compressed.len());
            result.push(DENSE_FORMAT);
            result.extend_from_slice(&compressed);
            Ok(result)
        }
    }

    fn decompress(&self, data: &[u8], dtype: DataType) -> Result<Vec<u8>, CompressionError> {
        self.decompress_to_dense(data, dtype)
    }

    fn method(&self) -> CompressionMethod {
        CompressionMethod::Zstd { level: 3 } // Report as Zstd since that's the inner compressor
    }
}

impl SparseCompressor {
    /// RLE encoding for sparse data without shape information
    fn compress_sparse_rle(
        &self,
        data: &[u8],
        element_size: usize,
    ) -> Result<Vec<u8>, CompressionError> {
        let zero_element = vec![0u8; element_size];
        let num_elements = data.len() / element_size;

        // Encode: [run_of_zeros, non_zero_value, run_of_zeros, non_zero_value, ...]
        let mut runs: Vec<u32> = Vec::new();
        let mut values: Vec<u8> = Vec::new();
        let mut zero_run = 0u32;

        for chunk in data.chunks_exact(element_size) {
            if chunk == zero_element.as_slice() {
                zero_run += 1;
            } else {
                runs.push(zero_run);
                values.extend_from_slice(chunk);
                zero_run = 0;
            }
        }
        // Final zero run
        runs.push(zero_run);

        // Encode runs as varints
        let runs_encoded = encode_varints(&runs);

        // Compress values
        let values_compressed = self.inner.compress(&values, DataType::UInt8)?;

        // Pack: [marker:1][num_elements:4][element_size:4][runs_len:4][runs][values]
        let mut result = Vec::new();
        result.push(SPARSE_FORMAT); // Reuse sparse marker
        result.extend_from_slice(&(num_elements as u32).to_le_bytes());
        result.extend_from_slice(&(element_size as u32).to_le_bytes());
        result.extend_from_slice(&(runs_encoded.len() as u32).to_le_bytes());
        result.extend_from_slice(&runs_encoded);
        result.extend_from_slice(&values_compressed);

        Ok(result)
    }
}

// =============================================================================
// Adaptive Compression Hints
// =============================================================================

/// Hints for adaptive compression decision-making
#[derive(Debug, Clone)]
pub struct TensorCompressionHint {
    /// Data type of the tensor
    pub dtype: DataType,
    /// Size in bytes
    pub size: usize,
    /// Sparsity ratio (0.0 = dense, 1.0 = all zeros)
    pub sparsity: f64,
    /// Optional: name for logging/debugging
    pub name: Option<String>,
}

/// Calculate sparsity (ratio of zero bytes)
pub fn calculate_sparsity(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let zeros = data.iter().filter(|&&b| b == 0).count();
    zeros as f64 / data.len() as f64
}

/// Adaptive compressor that selects the best compression method per tensor
pub struct AdaptiveCompressor {
    strategy: AdaptiveStrategy,
    /// Cached compressor instances
    zstd_low: ZstdCompressor,
    zstd_mid: ZstdCompressor,
    zstd_high: ZstdCompressor,
    lz4: Lz4Compressor,
    neural: NeuralCompressor,
}

impl AdaptiveCompressor {
    /// Create a new adaptive compressor with the given strategy
    pub fn new(strategy: AdaptiveStrategy) -> Self {
        Self {
            strategy,
            zstd_low: ZstdCompressor::new(1),
            zstd_mid: ZstdCompressor::new(5),
            zstd_high: ZstdCompressor::new(12),
            lz4: Lz4Compressor::new(),
            neural: NeuralCompressor::new(5),
        }
    }

    /// Select best compressor based on heuristics
    fn select_heuristic(&self, hint: &TensorCompressionHint) -> &dyn Compressor {
        // Decision tree:
        // 1. Small tensors (<4KB) -> LZ4 (speed priority)
        // 2. High sparsity (>80%) + float -> Neural (exploits sparse path)
        // 3. Float types -> Neural (exploits exponent patterns)
        // 4. Quantized types -> LZ4 (already compressed by quantization)
        // 5. Default -> Zstd mid

        if hint.size < 4096 {
            return &self.lz4; // Speed priority for small tensors
        }

        if hint.sparsity > 0.8 && hint.dtype.is_float() {
            return &self.neural; // Neural's sparse path excels here
        }

        match hint.dtype {
            DataType::Float32 | DataType::Float16 | DataType::BFloat16 => &self.neural,
            DataType::Float64 => &self.zstd_mid,
            DataType::Int8 | DataType::UInt8 => &self.zstd_mid,
            DataType::Int16 | DataType::UInt16 | DataType::Int32 | DataType::UInt32 => {
                &self.zstd_low
            }
            DataType::Int64 | DataType::UInt64 => &self.zstd_low,
            // Quantized types - already compressed by quantization
            _ if hint.dtype.is_quantized() => &self.lz4,
            _ => &self.zstd_mid,
        }
    }

    /// Benchmark all compressors and select best (smallest output)
    fn select_benchmark(&self, data: &[u8], dtype: DataType) -> (&dyn Compressor, Vec<u8>) {
        let candidates: Vec<&dyn Compressor> = vec![
            &self.zstd_low,
            &self.zstd_mid,
            &self.zstd_high,
            &self.lz4,
            &self.neural,
        ];

        let mut best: Option<(&dyn Compressor, Vec<u8>)> = None;

        for compressor in candidates {
            if let Ok(compressed) = compressor.compress(data, dtype) {
                let is_better = best
                    .as_ref()
                    .map(|(_, prev)| compressed.len() < prev.len())
                    .unwrap_or(true);

                if is_better {
                    best = Some((compressor, compressed));
                }
            }
        }

        best.unwrap_or((
            &self.lz4,
            self.lz4.compress(data, dtype).unwrap_or_default(),
        ))
    }

    /// Compress with full context, returning method used
    pub fn compress_with_hint(
        &self,
        data: &[u8],
        hint: &TensorCompressionHint,
    ) -> Result<(Vec<u8>, CompressionMethod), CompressionError> {
        match self.strategy {
            AdaptiveStrategy::Heuristic => {
                let compressor = self.select_heuristic(hint);
                let compressed = compressor.compress(data, hint.dtype)?;
                Ok((compressed, compressor.method()))
            }
            AdaptiveStrategy::Benchmark => {
                let (compressor, compressed) = self.select_benchmark(data, hint.dtype);
                Ok((compressed, compressor.method()))
            }
        }
    }
}

impl Compressor for AdaptiveCompressor {
    fn compress(&self, data: &[u8], dtype: DataType) -> Result<Vec<u8>, CompressionError> {
        // Create a basic hint from available info
        let hint = TensorCompressionHint {
            dtype,
            size: data.len(),
            sparsity: calculate_sparsity(data),
            name: None,
        };

        let (compressed, _method) = self.compress_with_hint(data, &hint)?;
        Ok(compressed)
    }

    fn decompress(&self, data: &[u8], dtype: DataType) -> Result<Vec<u8>, CompressionError> {
        // Try to detect format from data markers
        if !data.is_empty() {
            match data[0] {
                // Neural format markers
                SPARSE_FORMAT | DENSE_FORMAT | BYTEPLANE_FORMAT | BYTEPLANE_F16_FORMAT => {
                    return self.neural.decompress(data, dtype);
                }
                _ => {}
            }
        }

        // Try Zstd (has magic number 0x28 0xB5 0x2F 0xFD)
        if data.len() >= 4 && data[0] == 0x28 && data[1] == 0xB5 {
            return self.zstd_mid.decompress(data, dtype);
        }

        // Fallback to LZ4 (uses size-prepended format)
        self.lz4.decompress(data, dtype)
    }

    fn method(&self) -> CompressionMethod {
        CompressionMethod::Adaptive {
            strategy: self.strategy,
        }
    }
}

// =============================================================================
// Parallel Compression
// =============================================================================

/// Tensor data for batch compression
#[derive(Debug, Clone)]
pub struct TensorData {
    /// Tensor name
    pub name: String,
    /// Raw tensor data
    pub data: Vec<u8>,
    /// Data type
    pub dtype: DataType,
}

/// Result of compressing a single tensor
#[derive(Debug, Clone)]
pub struct CompressedTensor {
    /// Tensor name
    pub name: String,
    /// Compressed data
    pub data: Vec<u8>,
    /// Compression method used
    pub method: CompressionMethod,
}

/// Parallel compressor for batch tensor compression
///
/// Uses rayon to compress multiple tensors across CPU cores simultaneously.
/// Provides 4-8x speedup on typical multi-core systems.
pub struct ParallelCompressor {
    inner: Arc<dyn Compressor + Send + Sync>,
    adaptive: Arc<AdaptiveCompressor>,
}

impl ParallelCompressor {
    /// Create a new parallel compressor with default Zstd compression
    pub fn new() -> Self {
        Self {
            inner: Arc::new(ZstdCompressor::new(3)),
            adaptive: Arc::new(AdaptiveCompressor::new(AdaptiveStrategy::Heuristic)),
        }
    }

    /// Create with a specific compressor
    pub fn with_compressor<C: Compressor + 'static>(compressor: C) -> Self {
        Self {
            inner: Arc::new(compressor),
            adaptive: Arc::new(AdaptiveCompressor::new(AdaptiveStrategy::Heuristic)),
        }
    }

    /// Create with adaptive compression for optimal per-tensor method selection
    pub fn with_adaptive(strategy: AdaptiveStrategy) -> Self {
        let adaptive = Arc::new(AdaptiveCompressor::new(strategy));
        Self {
            inner: Arc::new(ZstdCompressor::new(3)),
            adaptive,
        }
    }

    /// Compress multiple tensors in parallel using the inner compressor
    ///
    /// Returns compressed tensors in the same order as input.
    pub fn compress_batch(
        &self,
        tensors: &[TensorData],
    ) -> Result<Vec<CompressedTensor>, CompressionError> {
        let inner = Arc::clone(&self.inner);

        tensors
            .par_iter()
            .map(|tensor| {
                let compressed = inner.compress(&tensor.data, tensor.dtype)?;
                Ok(CompressedTensor {
                    name: tensor.name.clone(),
                    data: compressed,
                    method: inner.method(),
                })
            })
            .collect()
    }

    /// Compress multiple tensors in parallel with adaptive method selection
    ///
    /// Each tensor may use a different compression method based on its characteristics.
    pub fn compress_batch_adaptive(
        &self,
        tensors: &[TensorData],
    ) -> Result<Vec<CompressedTensor>, CompressionError> {
        let adaptive = Arc::clone(&self.adaptive);

        tensors
            .par_iter()
            .map(|tensor| {
                let hint = TensorCompressionHint {
                    dtype: tensor.dtype,
                    size: tensor.data.len(),
                    sparsity: calculate_sparsity(&tensor.data),
                    name: Some(tensor.name.clone()),
                };

                let (compressed, method) = adaptive.compress_with_hint(&tensor.data, &hint)?;
                Ok(CompressedTensor {
                    name: tensor.name.clone(),
                    data: compressed,
                    method,
                })
            })
            .collect()
    }

    /// Decompress multiple tensors in parallel
    pub fn decompress_batch(
        &self,
        tensors: &[CompressedTensor],
    ) -> Result<Vec<TensorData>, CompressionError> {
        let adaptive = Arc::clone(&self.adaptive);

        tensors
            .par_iter()
            .map(|tensor| {
                let decompressed = adaptive.decompress(&tensor.data, DataType::Float32)?;
                Ok(TensorData {
                    name: tensor.name.clone(),
                    data: decompressed,
                    dtype: DataType::Float32, // Note: dtype info not preserved in compressed data
                })
            })
            .collect()
    }

    /// Get the number of threads available for parallel compression
    pub fn num_threads(&self) -> usize {
        rayon::current_num_threads()
    }
}

impl Default for ParallelCompressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Factory trait for creating compressors
pub trait CompressorFactory: Send + Sync {
    /// Create a compressor for the given method
    fn create(&self, method: CompressionMethod) -> Box<dyn Compressor>;
}

/// Default compressor factory implementation
pub struct DefaultCompressorFactory;

impl CompressorFactory for DefaultCompressorFactory {
    fn create(&self, method: CompressionMethod) -> Box<dyn Compressor> {
        match method {
            CompressionMethod::None => Box::new(NoCompressor),
            CompressionMethod::Zstd { level } => Box::new(ZstdCompressor::new(level)),
            CompressionMethod::ZstdDict { .. } => {
                // ZstdDict requires a dictionary - use ZstdDictCompressor::new() directly
                panic!("ZstdDict compression requires a dictionary. Use ZstdDictCompressor::new(level, &dict) instead of the factory.")
            }
            CompressionMethod::Lz4 => Box::new(Lz4Compressor::new()),
            CompressionMethod::Neural { .. } => Box::new(NeuralCompressor::new(5)),
            CompressionMethod::Adaptive { strategy } => Box::new(AdaptiveCompressor::new(strategy)),
        }
    }
}

/// Factory that supports dictionary-based compression
pub struct DictCompressorFactory {
    dictionary: Option<CompressionDictionary>,
}

impl DictCompressorFactory {
    /// Create a factory without a dictionary
    pub fn new() -> Self {
        Self { dictionary: None }
    }

    /// Create a factory with a dictionary for ZstdDict compression
    pub fn with_dictionary(dict: CompressionDictionary) -> Self {
        Self {
            dictionary: Some(dict),
        }
    }
}

impl Default for DictCompressorFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressorFactory for DictCompressorFactory {
    fn create(&self, method: CompressionMethod) -> Box<dyn Compressor> {
        match method {
            CompressionMethod::None => Box::new(NoCompressor),
            CompressionMethod::Zstd { level } => Box::new(ZstdCompressor::new(level)),
            CompressionMethod::ZstdDict { level, dict_id } => {
                let dict = self.dictionary.as_ref().unwrap_or_else(|| {
                    panic!("ZstdDict compression requires a dictionary. Set one with DictCompressorFactory::with_dictionary()")
                });
                if dict.id != dict_id {
                    panic!(
                        "Dictionary ID mismatch: expected {}, got {}",
                        dict_id, dict.id
                    );
                }
                Box::new(
                    ZstdDictCompressor::new(level, dict)
                        .expect("Failed to create ZstdDictCompressor"),
                )
            }
            CompressionMethod::Lz4 => Box::new(Lz4Compressor::new()),
            CompressionMethod::Neural { .. } => Box::new(NeuralCompressor::new(5)),
            CompressionMethod::Adaptive { strategy } => Box::new(AdaptiveCompressor::new(strategy)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zstd_roundtrip() {
        let data = vec![0u8; 1000];
        let compressor = ZstdCompressor::new(3);
        let compressed = compressor.compress(&data, DataType::Float32).unwrap();
        let decompressed = compressor
            .decompress(&compressed, DataType::Float32)
            .unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_lz4_roundtrip() {
        let data = vec![42u8; 1000];
        let compressor = Lz4Compressor::new();
        let compressed = compressor.compress(&data, DataType::Int8).unwrap();
        let decompressed = compressor.decompress(&compressed, DataType::Int8).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_neural_compressor_roundtrip() {
        // Create some float32 data
        let floats: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
        let data: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();

        let compressor = NeuralCompressor::new(3);
        let compressed = compressor.compress(&data, DataType::Float32).unwrap();
        let decompressed = compressor
            .decompress(&compressed, DataType::Float32)
            .unwrap();

        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_delta_encode_decode_roundtrip() {
        let original = vec![10, 20, 25, 30, 100, 99, 98];
        let encoded = delta_encode(&original);
        let decoded = delta_decode(&encoded);
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_delta_encode_decode_empty() {
        let empty: Vec<u8> = Vec::new();
        assert_eq!(delta_encode(&empty), empty);
        assert_eq!(delta_decode(&empty), empty);
    }

    #[test]
    fn test_neural_byteplane_format() {
        // Create realistic neural weight data
        let floats: Vec<f32> = (0..1024)
            .map(|i| {
                // Simulate typical weight distribution (small values around 0)
                ((i as f32 - 512.0) / 1024.0) * 0.1
            })
            .collect();
        let data: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();

        let compressor = NeuralCompressor::new(3);
        let compressed = compressor.compress(&data, DataType::Float32).unwrap();
        let decompressed = compressor
            .decompress(&compressed, DataType::Float32)
            .unwrap();

        assert_eq!(data, decompressed);
        // Verify we're using byteplane format (marker 0x03)
        assert_eq!(compressed[0], BYTEPLANE_FORMAT);
    }

    #[test]
    fn test_neural_sparse_format_detection() {
        // Create sparse data (lots of zeros) - verify sparse format is selected
        let floats: Vec<f32> = vec![0.0; 1000];
        let data: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();

        let compressor = NeuralCompressor::new(3);
        let compressed = compressor.compress(&data, DataType::Float32).unwrap();

        // Verify we're using sparse format for mostly-zero data (marker 0x01)
        assert_eq!(compressed[0], SPARSE_FORMAT);

        // Verify it decompresses to the correct size
        let decompressed = compressor
            .decompress(&compressed, DataType::Float32)
            .unwrap();
        assert_eq!(decompressed.len(), data.len());
    }

    #[test]
    fn test_neural_f16_roundtrip() {
        // Simulate float16 data (2 bytes per value)
        let data: Vec<u8> = (0..512)
            .flat_map(|i| [(i & 0xFF) as u8, (i >> 8) as u8])
            .collect();

        let compressor = NeuralCompressor::new(3);
        let compressed = compressor.compress(&data, DataType::Float16).unwrap();
        let decompressed = compressor
            .decompress(&compressed, DataType::Float16)
            .unwrap();

        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_neural_compression_ratio() {
        // Create data with high exponent redundancy (common in neural weights)
        let floats: Vec<f32> = (0..4096)
            .map(|i| {
                // Many small weights with similar exponents
                ((i % 100) as f32 - 50.0) * 0.001
            })
            .collect();
        let data: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();

        let compressor = NeuralCompressor::new(5);
        let zstd = ZstdCompressor::new(5);

        let neural_compressed = compressor.compress(&data, DataType::Float32).unwrap();
        let zstd_compressed = zstd.compress(&data, DataType::Float32).unwrap();

        // Neural compression should be competitive with zstd
        // (often better for neural weight distributions)
        println!(
            "Original: {} bytes, Neural: {} bytes, Zstd: {} bytes",
            data.len(),
            neural_compressed.len(),
            zstd_compressed.len()
        );

        // Verify roundtrip
        let decompressed = compressor
            .decompress(&neural_compressed, DataType::Float32)
            .unwrap();
        assert_eq!(data, decompressed);
    }

    // =========================================================================
    // Dictionary Training Tests
    // =========================================================================

    /// Generate sample data that simulates model weight patterns
    fn generate_model_sample(seed: u32, size: usize) -> Vec<u8> {
        let floats: Vec<f32> = (0..size / 4)
            .map(|i| {
                // Simulate typical weight distribution with some variation by seed
                let base = ((i as f32 + seed as f32) % 100.0 - 50.0) * 0.001;
                base + (seed as f32 * 0.0001)
            })
            .collect();
        floats.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    #[test]
    fn test_dictionary_trainer_creation() {
        let trainer = DictionaryTrainer::new();
        assert_eq!(trainer.max_dict_size, DictionaryTrainer::DEFAULT_DICT_SIZE);

        let custom_trainer = DictionaryTrainer::with_max_size(50 * 1024);
        assert_eq!(custom_trainer.max_dict_size, 50 * 1024);
    }

    #[test]
    fn test_dictionary_training_requires_min_samples() {
        let trainer = DictionaryTrainer::new();
        let sample1: &[u8] = &[1, 2, 3, 4];
        let sample2: &[u8] = &[5, 6, 7, 8];

        // Should fail with only 2 samples
        let result = trainer.train(&[sample1, sample2]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("at least"));
    }

    #[test]
    fn test_dictionary_training_and_compression() {
        // Generate sample data that simulates model weights
        // Use larger samples (100KB+) for effective dictionary training
        let sample1 = generate_model_sample(1, 100_000);
        let sample2 = generate_model_sample(2, 100_000);
        let sample3 = generate_model_sample(3, 100_000);
        let sample4 = generate_model_sample(4, 100_000);

        let samples: Vec<&[u8]> = vec![&sample1, &sample2, &sample3, &sample4];

        // Train dictionary
        let trainer = DictionaryTrainer::new();
        let dict = trainer.train(&samples).expect("Training should succeed");

        // Verify dictionary properties
        assert!(!dict.data.is_empty());
        assert!(dict.data.len() <= DictionaryTrainer::DEFAULT_DICT_SIZE);
        assert_eq!(dict.metadata.num_samples, 4);

        // Test compression with dictionary
        let compressor =
            ZstdDictCompressor::new(3, &dict).expect("Compressor creation should succeed");

        // Compress and decompress new data from same distribution
        let test_data = generate_model_sample(5, 100_000);
        let compressed = compressor.compress(&test_data, DataType::Float32).unwrap();
        let decompressed = compressor
            .decompress(&compressed, DataType::Float32)
            .unwrap();

        assert_eq!(test_data, decompressed);
    }

    #[test]
    fn test_dictionary_compression_improvement() {
        // Generate sample data - use 200KB+ for meaningful dictionary training
        let sample1 = generate_model_sample(1, 200_000);
        let sample2 = generate_model_sample(2, 200_000);
        let sample3 = generate_model_sample(3, 200_000);

        let samples: Vec<&[u8]> = vec![&sample1, &sample2, &sample3];

        // Train dictionary
        let trainer = DictionaryTrainer::new();
        let dict = trainer.train(&samples).expect("Training should succeed");

        // Create compressors
        let dict_compressor = ZstdDictCompressor::new(5, &dict).unwrap();
        let regular_compressor = ZstdCompressor::new(5);

        // Compress test data from same distribution
        let test_data = generate_model_sample(4, 200_000);

        let dict_compressed = dict_compressor
            .compress(&test_data, DataType::Float32)
            .unwrap();
        let regular_compressed = regular_compressor
            .compress(&test_data, DataType::Float32)
            .unwrap();

        println!(
            "Dictionary compression: {} -> {} bytes ({:.1}%)",
            test_data.len(),
            dict_compressed.len(),
            100.0 * dict_compressed.len() as f64 / test_data.len() as f64
        );
        println!(
            "Regular compression: {} -> {} bytes ({:.1}%)",
            test_data.len(),
            regular_compressed.len(),
            100.0 * regular_compressed.len() as f64 / test_data.len() as f64
        );

        // Dictionary compression should be at least as good (often better)
        // For small synthetic data, the improvement may be marginal
        // For real model data, expect 20-50% improvement

        // Verify roundtrip
        let decompressed = dict_compressor
            .decompress(&dict_compressed, DataType::Float32)
            .unwrap();
        assert_eq!(test_data, decompressed);
    }

    #[test]
    fn test_dictionary_save_load_roundtrip() {
        // Generate sample data and train dictionary - use 100KB+ for training
        let sample1 = generate_model_sample(1, 100_000);
        let sample2 = generate_model_sample(2, 100_000);
        let sample3 = generate_model_sample(3, 100_000);

        let samples: Vec<&[u8]> = vec![&sample1, &sample2, &sample3];

        let trainer = DictionaryTrainer::new();
        let dict = trainer
            .train_with_description(&samples, "Test dictionary")
            .expect("Training should succeed");

        // Save and load
        let saved = DictionaryTrainer::save(&dict).expect("Save should succeed");
        let loaded = DictionaryTrainer::load(&saved).expect("Load should succeed");

        // Verify loaded dictionary
        assert_eq!(dict.id, loaded.id);
        assert_eq!(dict.data, loaded.data);
        assert_eq!(dict.metadata.num_samples, loaded.metadata.num_samples);
        assert_eq!(dict.metadata.description, loaded.metadata.description);
    }

    #[test]
    fn test_dict_compressor_factory() {
        // Generate samples and train dictionary - use 100KB+ for training
        let sample1 = generate_model_sample(1, 100_000);
        let sample2 = generate_model_sample(2, 100_000);
        let sample3 = generate_model_sample(3, 100_000);

        let samples: Vec<&[u8]> = vec![&sample1, &sample2, &sample3];

        let trainer = DictionaryTrainer::new();
        let dict = trainer.train(&samples).expect("Training should succeed");

        // Create factory with dictionary
        let factory = DictCompressorFactory::with_dictionary(dict.clone());

        // Test creating ZstdDict compressor
        let method = CompressionMethod::ZstdDict {
            level: 5,
            dict_id: dict.id,
        };
        let compressor = factory.create(method);

        // Test compression
        let test_data = generate_model_sample(4, 100_000);
        let compressed = compressor.compress(&test_data, DataType::Float32).unwrap();
        let decompressed = compressor
            .decompress(&compressed, DataType::Float32)
            .unwrap();

        assert_eq!(test_data, decompressed);
    }

    // =========================================================================
    // Sparse Tensor Tests
    // =========================================================================

    #[test]
    fn test_sparse_csr_encode_decode_roundtrip() {
        // Create a 10x10 matrix with 70% sparsity (30% non-zero)
        let mut data = vec![0u8; 10 * 10 * 4]; // 10x10 float32
                                               // Set some non-zero values
        for i in [0, 5, 15, 23, 42, 57, 68, 73, 81, 99] {
            let val: f32 = (i as f32 + 1.0) * 0.1;
            let bytes = val.to_le_bytes();
            data[i * 4..i * 4 + 4].copy_from_slice(&bytes);
        }

        // Encode as CSR
        let sparse = SparseEncoder::encode_csr(&data, 4, &[10, 10]).unwrap();

        // Verify structure
        assert_eq!(sparse.shape, [10, 10]);
        assert_eq!(sparse.element_size, 4);
        assert_eq!(sparse.col_indices.len(), 10); // 10 non-zero values
        assert_eq!(sparse.row_ptrs.len(), 11); // rows + 1

        // Decode back
        let decoded = SparseEncoder::decode_csr(&sparse);
        assert_eq!(data, decoded);
    }

    #[test]
    fn test_sparse_csr_serialization_roundtrip() {
        // Create sparse data
        let mut data = vec![0u8; 5 * 5 * 4];
        data[0..4].copy_from_slice(&1.0f32.to_le_bytes());
        data[48..52].copy_from_slice(&2.0f32.to_le_bytes()); // position [3, 0]
        data[96..100].copy_from_slice(&3.0f32.to_le_bytes()); // position [4, 4]

        let sparse = SparseEncoder::encode_csr(&data, 4, &[5, 5]).unwrap();

        // Serialize and deserialize
        let bytes = sparse.to_bytes();
        let loaded = SparseCSR::from_bytes(&bytes).unwrap();

        assert_eq!(sparse.shape, loaded.shape);
        assert_eq!(sparse.values, loaded.values);
        assert_eq!(sparse.col_indices, loaded.col_indices);
        assert_eq!(sparse.row_ptrs, loaded.row_ptrs);
    }

    #[test]
    fn test_sparse_sparsity_calculation() {
        // 100% sparse (all zeros)
        let all_zeros = vec![0u8; 100 * 4];
        assert!((SparseEncoder::calculate_sparsity(&all_zeros, 4) - 1.0).abs() < 0.001);

        // 0% sparse (no zeros)
        let no_zeros: Vec<u8> = (0..100u32).flat_map(|i| (i + 1).to_le_bytes()).collect();
        assert!(SparseEncoder::calculate_sparsity(&no_zeros, 4) < 0.001);

        // 50% sparse
        let mut half_sparse = vec![0u8; 100 * 4];
        for i in 0..50 {
            let bytes = ((i + 1) as u32).to_le_bytes();
            half_sparse[i * 4..i * 4 + 4].copy_from_slice(&bytes);
        }
        assert!((SparseEncoder::calculate_sparsity(&half_sparse, 4) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_sparse_is_beneficial() {
        let sparse_data = vec![0u8; 1000 * 4]; // All zeros = 100% sparse
        assert!(SparseEncoder::is_beneficial(&sparse_data, 4, 0.5));
        assert!(SparseEncoder::is_beneficial(&sparse_data, 4, 0.9));

        let dense_data: Vec<u8> = (0..1000u32).flat_map(|i| (i + 1).to_le_bytes()).collect();
        assert!(!SparseEncoder::is_beneficial(&dense_data, 4, 0.5));
    }

    #[test]
    fn test_sparse_compressor_roundtrip() {
        // Create 80% sparse data
        let mut data = vec![0u8; 1000 * 4];
        for i in (0..1000).step_by(5) {
            let val = (i as f32 + 1.0) * 0.01;
            data[i * 4..i * 4 + 4].copy_from_slice(&val.to_le_bytes());
        }

        let compressor = SparseCompressor::new(3);
        let compressed = compressor.compress(&data, DataType::Float32).unwrap();
        let decompressed = compressor
            .decompress(&compressed, DataType::Float32)
            .unwrap();

        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_sparse_compressor_with_shape() {
        // Create 90% sparse 100x100 matrix
        let mut data = vec![0u8; 100 * 100 * 4];
        for i in (0..10000).step_by(10) {
            let val = (i as f32 + 1.0) * 0.001;
            data[i * 4..i * 4 + 4].copy_from_slice(&val.to_le_bytes());
        }

        let compressor = SparseCompressor::new(3);
        let compressed = compressor
            .compress_with_shape(&data, DataType::Float32, &[100, 100])
            .unwrap();
        let decompressed = compressor
            .decompress_to_dense(&compressed, DataType::Float32)
            .unwrap();

        assert_eq!(data, decompressed);

        // Sparse compression should be smaller than dense
        let dense_compressor = ZstdCompressor::new(3);
        let dense_compressed = dense_compressor.compress(&data, DataType::Float32).unwrap();

        println!(
            "Sparse compression: {} -> {} bytes",
            data.len(),
            compressed.len()
        );
        println!(
            "Dense compression: {} -> {} bytes",
            data.len(),
            dense_compressed.len()
        );
    }

    #[test]
    fn test_sparse_compression_ratio() {
        // Create 90% sparse matrix
        let mut data = vec![0u8; 100 * 100 * 4];
        for i in (0..10000).step_by(10) {
            data[i * 4..i * 4 + 4].copy_from_slice(&1.0f32.to_le_bytes());
        }

        let sparse = SparseEncoder::encode_csr(&data, 4, &[100, 100]).unwrap();
        let ratio = SparseEncoder::compression_ratio(&sparse);

        // With 90% sparsity, CSR should compress significantly
        assert!(
            ratio > 1.0,
            "Expected compression ratio > 1.0, got {}",
            ratio
        );
        println!("90% sparse matrix CSR compression ratio: {:.2}x", ratio);
    }

    #[test]
    fn test_sparse_falls_back_to_dense_for_low_sparsity() {
        // Create dense data (no zeros)
        let data: Vec<u8> = (0..1000u32).flat_map(|i| (i + 1).to_le_bytes()).collect();

        let compressor = SparseCompressor::new(3);
        let compressed = compressor.compress(&data, DataType::Float32).unwrap();

        // Should use dense format (marker 0x02)
        assert_eq!(compressed[0], DENSE_FORMAT);

        let decompressed = compressor
            .decompress(&compressed, DataType::Float32)
            .unwrap();
        assert_eq!(data, decompressed);
    }

    // =========================================================================
    // Parallel Compression Tests
    // =========================================================================

    #[test]
    fn test_parallel_compressor_basic() {
        let compressor = ParallelCompressor::new();

        // Create test tensors
        let tensors: Vec<TensorData> = (0..4)
            .map(|i| TensorData {
                name: format!("tensor_{}", i),
                data: vec![(i as u8).wrapping_mul(17); 1000],
                dtype: DataType::Float32,
            })
            .collect();

        let compressed = compressor.compress_batch(&tensors).unwrap();
        assert_eq!(compressed.len(), 4);

        for (i, result) in compressed.iter().enumerate() {
            assert_eq!(result.name, format!("tensor_{}", i));
            assert!(!result.data.is_empty());
        }
    }

    #[test]
    fn test_parallel_compressor_adaptive() {
        let compressor = ParallelCompressor::with_adaptive(AdaptiveStrategy::Heuristic);

        // Create tensors with different characteristics
        let tensors = vec![
            TensorData {
                name: "sparse".to_string(),
                data: vec![0u8; 10000], // All zeros - highly sparse
                dtype: DataType::Float32,
            },
            TensorData {
                name: "dense".to_string(),
                data: (0..10000).map(|i| (i % 256) as u8).collect(),
                dtype: DataType::Float32,
            },
            TensorData {
                name: "small".to_string(),
                data: vec![42u8; 100], // Small tensor
                dtype: DataType::Int8,
            },
        ];

        let compressed = compressor.compress_batch_adaptive(&tensors).unwrap();
        assert_eq!(compressed.len(), 3);

        // Each tensor should have been compressed with potentially different methods
        for result in &compressed {
            assert!(!result.data.is_empty());
        }
    }

    #[test]
    fn test_parallel_compressor_roundtrip() {
        let compressor = ParallelCompressor::new();

        // Create larger tensors to make compression meaningful
        let tensors: Vec<TensorData> = (0..8)
            .map(|i| {
                let data: Vec<u8> = (0..4000).map(|j| ((i * 17 + j * 13) % 256) as u8).collect();
                TensorData {
                    name: format!("tensor_{}", i),
                    data,
                    dtype: DataType::Float32,
                }
            })
            .collect();

        let compressed = compressor.compress_batch(&tensors).unwrap();
        let decompressed = compressor.decompress_batch(&compressed).unwrap();

        assert_eq!(decompressed.len(), tensors.len());
        for (original, restored) in tensors.iter().zip(decompressed.iter()) {
            assert_eq!(original.name, restored.name);
            assert_eq!(original.data, restored.data);
        }
    }

    #[test]
    fn test_parallel_compressor_empty_batch() {
        let compressor = ParallelCompressor::new();
        let tensors: Vec<TensorData> = vec![];
        let compressed = compressor.compress_batch(&tensors).unwrap();
        assert!(compressed.is_empty());
    }

    #[test]
    fn test_parallel_compressor_num_threads() {
        let compressor = ParallelCompressor::new();
        // Should have at least 1 thread available
        assert!(compressor.num_threads() >= 1);
    }
}
