//! Streaming patch format for embedded systems
//!
//! A simplified, sequential patch format designed for minimal RAM usage.

/// Magic number for streaming patch format: "MLLP" (Mallorn Lite Patch)
pub const PATCH_MAGIC: u32 = 0x504C4C4D; // "MLLP" in little-endian

/// Streaming patch format version
pub const PATCH_VERSION: u32 = 2;

/// Operation types
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpType {
    /// Copy tensor unchanged from source
    Copy = 0,
    /// Apply XOR delta to source tensor
    Delta = 1,
    /// Replace tensor entirely
    Replace = 2,
}

/// Compression types for payloads
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    /// No compression
    None = 0,
    /// LZ4 compression (recommended for embedded)
    Lz4 = 1,
}

/// Patch header (48 bytes, fixed size)
///
/// This is the first thing read from a streaming patch.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct PatchHeader {
    /// Magic number (PATCH_MAGIC)
    pub magic: u32,
    /// Format version (PATCH_VERSION)
    pub version: u32,
    /// SHA256 hash of source model
    pub source_hash: [u8; 32],
    /// Number of tensor operations
    pub tensor_count: u32,
    /// Total uncompressed size (for progress)
    pub total_size: u32,
}

impl PatchHeader {
    /// Create a new empty header
    pub const fn new() -> Self {
        Self {
            magic: 0,
            version: 0,
            source_hash: [0u8; 32],
            tensor_count: 0,
            total_size: 0,
        }
    }

    /// Check if this is a valid patch header
    pub fn is_valid(&self) -> bool {
        self.magic == PATCH_MAGIC && self.version == PATCH_VERSION
    }
}

impl Default for PatchHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// Tensor operation header (16 bytes, fixed size)
///
/// Describes a single operation to apply to a tensor.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct TensorOp {
    /// Operation type (OpType)
    pub op_type: u8,
    /// Compression type for payload (CompressionType)
    pub compression: u8,
    /// Reserved for alignment
    pub reserved: u16,
    /// Offset in source model
    pub offset: u32,
    /// Uncompressed size in bytes
    pub size: u32,
    /// Compressed payload size (0 for Copy ops)
    pub payload_size: u32,
}

impl TensorOp {
    /// Create a new empty operation
    pub const fn new() -> Self {
        Self {
            op_type: 0,
            compression: 0,
            reserved: 0,
            offset: 0,
            size: 0,
            payload_size: 0,
        }
    }

    /// Create a copy operation
    pub fn copy(offset: u32, size: u32) -> Self {
        Self {
            op_type: OpType::Copy as u8,
            compression: CompressionType::None as u8,
            reserved: 0,
            offset,
            size,
            payload_size: 0,
        }
    }

    /// Create a delta operation
    pub fn delta(offset: u32, size: u32, payload_size: u32, compression: CompressionType) -> Self {
        Self {
            op_type: OpType::Delta as u8,
            compression: compression as u8,
            reserved: 0,
            offset,
            size,
            payload_size,
        }
    }

    /// Create a replace operation
    pub fn replace(offset: u32, size: u32, payload_size: u32, compression: CompressionType) -> Self {
        Self {
            op_type: OpType::Replace as u8,
            compression: compression as u8,
            reserved: 0,
            offset,
            size,
            payload_size,
        }
    }
}

impl Default for TensorOp {
    fn default() -> Self {
        Self::new()
    }
}

/// Patch footer (36 bytes, fixed size)
///
/// This is read last to verify the patch was applied correctly.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct PatchFooter {
    /// SHA256 hash of target model
    pub target_hash: [u8; 32],
    /// CRC32 of entire patch file (excluding this field)
    pub crc32: u32,
}

impl PatchFooter {
    /// Create a new empty footer
    pub const fn new() -> Self {
        Self {
            target_hash: [0u8; 32],
            crc32: 0,
        }
    }
}

impl Default for PatchFooter {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate sizes for format structures
#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    #[test]
    fn test_header_size() {
        // Header should be exactly 48 bytes
        assert_eq!(size_of::<PatchHeader>(), 48);
    }

    #[test]
    fn test_tensor_op_size() {
        // TensorOp should be exactly 16 bytes
        assert_eq!(size_of::<TensorOp>(), 16);
    }

    #[test]
    fn test_footer_size() {
        // Footer should be exactly 36 bytes
        assert_eq!(size_of::<PatchFooter>(), 36);
    }

    #[test]
    fn test_header_validation() {
        let mut header = PatchHeader::new();
        assert!(!header.is_valid());

        header.magic = PATCH_MAGIC;
        header.version = PATCH_VERSION;
        assert!(header.is_valid());
    }

    #[test]
    fn test_op_creation() {
        let copy = TensorOp::copy(0, 1024);
        assert_eq!(copy.op_type, OpType::Copy as u8);
        // Use copy to avoid unaligned access on packed struct
        let payload_size = { copy.payload_size };
        assert_eq!(payload_size, 0);

        let delta = TensorOp::delta(1024, 512, 256, CompressionType::Lz4);
        assert_eq!(delta.op_type, OpType::Delta as u8);
        assert_eq!(delta.compression, CompressionType::Lz4 as u8);

        let replace = TensorOp::replace(2048, 256, 128, CompressionType::None);
        assert_eq!(replace.op_type, OpType::Replace as u8);
    }
}
