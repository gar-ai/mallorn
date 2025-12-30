//! GGML quantization type utilities

use crate::parser::GGMLType;

/// Information about a quantization type
#[derive(Debug, Clone, Copy)]
pub struct QuantizationInfo {
    /// Block size (number of values per block)
    pub block_size: usize,
    /// Bytes per block
    pub bytes_per_block: usize,
    /// Whether this is a K-quant type
    pub is_k_quant: bool,
}

impl GGMLType {
    /// Get quantization info for this type
    pub fn quant_info(&self) -> QuantizationInfo {
        match self {
            // Float types - no quantization
            GGMLType::F32 => QuantizationInfo {
                block_size: 1,
                bytes_per_block: 4,
                is_k_quant: false,
            },
            GGMLType::F16 => QuantizationInfo {
                block_size: 1,
                bytes_per_block: 2,
                is_k_quant: false,
            },
            GGMLType::BF16 => QuantizationInfo {
                block_size: 1,
                bytes_per_block: 2,
                is_k_quant: false,
            },
            GGMLType::F64 => QuantizationInfo {
                block_size: 1,
                bytes_per_block: 8,
                is_k_quant: false,
            },

            // Integer types
            GGMLType::I8 => QuantizationInfo {
                block_size: 1,
                bytes_per_block: 1,
                is_k_quant: false,
            },
            GGMLType::I16 => QuantizationInfo {
                block_size: 1,
                bytes_per_block: 2,
                is_k_quant: false,
            },
            GGMLType::I32 => QuantizationInfo {
                block_size: 1,
                bytes_per_block: 4,
                is_k_quant: false,
            },
            GGMLType::I64 => QuantizationInfo {
                block_size: 1,
                bytes_per_block: 8,
                is_k_quant: false,
            },

            // Legacy quantization (32 values per block)
            GGMLType::Q4_0 => QuantizationInfo {
                block_size: 32,
                bytes_per_block: 18, // 16 bytes data + 2 bytes scale
                is_k_quant: false,
            },
            GGMLType::Q4_1 => QuantizationInfo {
                block_size: 32,
                bytes_per_block: 20, // 16 bytes data + 2 scale + 2 min
                is_k_quant: false,
            },
            GGMLType::Q5_0 => QuantizationInfo {
                block_size: 32,
                bytes_per_block: 22,
                is_k_quant: false,
            },
            GGMLType::Q5_1 => QuantizationInfo {
                block_size: 32,
                bytes_per_block: 24,
                is_k_quant: false,
            },
            GGMLType::Q8_0 => QuantizationInfo {
                block_size: 32,
                bytes_per_block: 34, // 32 bytes data + 2 bytes scale
                is_k_quant: false,
            },
            GGMLType::Q8_1 => QuantizationInfo {
                block_size: 32,
                bytes_per_block: 36,
                is_k_quant: false,
            },

            // K-quant types (256 values per superblock)
            GGMLType::Q2K => QuantizationInfo {
                block_size: 256,
                bytes_per_block: 84,
                is_k_quant: true,
            },
            GGMLType::Q3K => QuantizationInfo {
                block_size: 256,
                bytes_per_block: 110,
                is_k_quant: true,
            },
            GGMLType::Q4K => QuantizationInfo {
                block_size: 256,
                bytes_per_block: 144,
                is_k_quant: true,
            },
            GGMLType::Q5K => QuantizationInfo {
                block_size: 256,
                bytes_per_block: 176,
                is_k_quant: true,
            },
            GGMLType::Q6K => QuantizationInfo {
                block_size: 256,
                bytes_per_block: 210,
                is_k_quant: true,
            },
            GGMLType::Q8K => QuantizationInfo {
                block_size: 256,
                bytes_per_block: 292,
                is_k_quant: true,
            },

            // IQ types (various)
            _ => QuantizationInfo {
                block_size: 32,
                bytes_per_block: 18, // Default estimate
                is_k_quant: false,
            },
        }
    }

    /// Check if this is a float type
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            GGMLType::F32 | GGMLType::F16 | GGMLType::BF16 | GGMLType::F64
        )
    }

    /// Check if this is a quantized type
    pub fn is_quantized(&self) -> bool {
        !self.is_float()
            && !matches!(
                self,
                GGMLType::I8 | GGMLType::I16 | GGMLType::I32 | GGMLType::I64
            )
    }
}
