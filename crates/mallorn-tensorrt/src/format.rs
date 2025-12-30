//! TensorRT patch format (.trtp)
//!
//! Binary format for TensorRT workflow patches.
//!
//! ## Format Layout
//!
//! ```text
//! [TRTP magic: 4 bytes]
//! [Version: u32 LE]
//! [Source ONNX hash: 32 bytes]
//! [Target ONNX hash: 32 bytes]
//! [ONNX patch length: u64 LE]
//! [ONNX patch data: variable]
//! [Config length: u32 LE]
//! [Config JSON: variable]
//! [CRC32 checksum: u32 LE]
//! ```
//!
//! The ONNX patch data is the raw serialized mallorn-onnx patch.
//! The config is the TensorRT build configuration as JSON.

use crate::config::TensorRTConfig;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use mallorn_core::{crc32, ParseError, Patch};
use std::io::{Cursor, Read, Write};

/// Magic bytes for TensorRT patch files
pub const TENSORRT_MAGIC: &[u8; 4] = b"TRTP";

/// Current patch format version
pub const TENSORRT_VERSION: u32 = 1;

/// TensorRT patch containing ONNX delta and build config
#[derive(Debug, Clone)]
pub struct TensorRTPatch {
    /// The underlying ONNX patch (delta)
    pub onnx_patch: Patch,
    /// TensorRT build configuration
    pub config: TensorRTConfig,
    /// SHA256 hash of source ONNX model
    pub source_onnx_hash: [u8; 32],
    /// SHA256 hash of target ONNX model
    pub target_onnx_hash: [u8; 32],
}

/// Serialize a TensorRT patch to .trtp format
pub fn serialize_patch(patch: &TensorRTPatch) -> Result<Vec<u8>, std::io::Error> {
    let mut buf = Vec::new();

    // Magic
    buf.write_all(TENSORRT_MAGIC)?;

    // Version
    buf.write_u32::<LittleEndian>(TENSORRT_VERSION)?;

    // ONNX hashes
    buf.write_all(&patch.source_onnx_hash)?;
    buf.write_all(&patch.target_onnx_hash)?;

    // Serialize ONNX patch
    let onnx_patch_bytes = mallorn_onnx::serialize_patch(&patch.onnx_patch)?;
    buf.write_u64::<LittleEndian>(onnx_patch_bytes.len() as u64)?;
    buf.write_all(&onnx_patch_bytes)?;

    // Config as JSON
    let config_json = serde_json::to_string(&patch.config)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let config_bytes = config_json.as_bytes();
    buf.write_u32::<LittleEndian>(config_bytes.len() as u32)?;
    buf.write_all(config_bytes)?;

    // CRC32 checksum
    let checksum = crc32(&buf);
    buf.write_u32::<LittleEndian>(checksum)?;

    Ok(buf)
}

/// Deserialize a .trtp patch file
pub fn deserialize_patch(data: &[u8]) -> Result<TensorRTPatch, ParseError> {
    if data.len() < 4 {
        return Err(ParseError::Malformed("File too small".into()));
    }

    // Verify magic
    if &data[0..4] != TENSORRT_MAGIC {
        return Err(ParseError::InvalidMagic);
    }

    // Verify CRC32 (last 4 bytes)
    if data.len() < 8 {
        return Err(ParseError::Malformed("File too small for checksum".into()));
    }
    let payload = &data[..data.len() - 4];
    let stored_crc = u32::from_le_bytes([
        data[data.len() - 4],
        data[data.len() - 3],
        data[data.len() - 2],
        data[data.len() - 1],
    ]);
    let computed_crc = crc32(payload);
    if stored_crc != computed_crc {
        return Err(ParseError::Malformed("CRC32 checksum mismatch".into()));
    }

    let mut cursor = Cursor::new(payload);
    cursor.set_position(4); // Skip magic

    // Version
    let version = cursor
        .read_u32::<LittleEndian>()
        .map_err(|e| ParseError::Malformed(e.to_string()))?;

    if version != TENSORRT_VERSION {
        return Err(ParseError::UnsupportedVersion(version));
    }

    // ONNX hashes
    let mut source_onnx_hash = [0u8; 32];
    let mut target_onnx_hash = [0u8; 32];
    cursor
        .read_exact(&mut source_onnx_hash)
        .map_err(|e| ParseError::Malformed(e.to_string()))?;
    cursor
        .read_exact(&mut target_onnx_hash)
        .map_err(|e| ParseError::Malformed(e.to_string()))?;

    // ONNX patch
    let onnx_patch_len = cursor
        .read_u64::<LittleEndian>()
        .map_err(|e| ParseError::Malformed(e.to_string()))? as usize;
    let mut onnx_patch_bytes = vec![0u8; onnx_patch_len];
    cursor
        .read_exact(&mut onnx_patch_bytes)
        .map_err(|e| ParseError::Malformed(e.to_string()))?;
    let onnx_patch = mallorn_onnx::deserialize_patch(&onnx_patch_bytes)?;

    // Config JSON
    let config_len = cursor
        .read_u32::<LittleEndian>()
        .map_err(|e| ParseError::Malformed(e.to_string()))? as usize;
    let mut config_bytes = vec![0u8; config_len];
    cursor
        .read_exact(&mut config_bytes)
        .map_err(|e| ParseError::Malformed(e.to_string()))?;
    let config: TensorRTConfig = serde_json::from_slice(&config_bytes)
        .map_err(|e| ParseError::Malformed(format!("Invalid config JSON: {}", e)))?;

    Ok(TensorRTPatch {
        onnx_patch,
        config,
        source_onnx_hash,
        target_onnx_hash,
    })
}

/// Get file extension for TensorRT patches
pub fn extension() -> &'static str {
    "trtp"
}

/// Check if data looks like a TensorRT patch
pub fn is_trtp(data: &[u8]) -> bool {
    data.len() >= 4 && &data[0..4] == TENSORRT_MAGIC
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Precision;
    use mallorn_core::{CompressionMethod, PatchMetadata, PatchOperation};

    fn make_test_patch() -> TensorRTPatch {
        TensorRTPatch {
            onnx_patch: Patch {
                version: 1,
                source_hash: [0xAA; 32],
                target_hash: [0xBB; 32],
                operations: vec![
                    PatchOperation::CopyTensor {
                        name: "encoder.weight".into(),
                    },
                    PatchOperation::ReplaceTensor {
                        name: "decoder.weight".into(),
                        data: vec![1, 2, 3, 4, 5],
                        compression: None,
                    },
                ],
                compression: CompressionMethod::Zstd { level: 3 },
                metadata: PatchMetadata {
                    source_version: Some("model-v1.0".into()),
                    target_version: Some("model-v1.1".into()),
                    created_at: 1234567890,
                    description: Some("Test TensorRT patch".into()),
                },
            },
            config: TensorRTConfig::new()
                .with_precision(Precision::FP16)
                .with_workspace_mb(2048)
                .with_max_batch_size(8)
                .with_description("Test config"),
            source_onnx_hash: [0xCC; 32],
            target_onnx_hash: [0xDD; 32],
        }
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let patch = make_test_patch();
        let serialized = serialize_patch(&patch).unwrap();

        assert_eq!(&serialized[0..4], TENSORRT_MAGIC);

        let deserialized = deserialize_patch(&serialized).unwrap();

        assert_eq!(deserialized.source_onnx_hash, patch.source_onnx_hash);
        assert_eq!(deserialized.target_onnx_hash, patch.target_onnx_hash);
        assert_eq!(deserialized.config.precision, Precision::FP16);
        assert_eq!(deserialized.config.workspace_size_mb, 2048);
        assert_eq!(deserialized.config.max_batch_size, 8);
        assert_eq!(
            deserialized.onnx_patch.operations.len(),
            patch.onnx_patch.operations.len()
        );
    }

    #[test]
    fn test_invalid_magic() {
        let data = b"BAAD1234567890123456789012345678901234567890";
        let result = deserialize_patch(data);
        assert!(matches!(result, Err(ParseError::InvalidMagic)));
    }

    #[test]
    fn test_is_trtp() {
        assert!(is_trtp(b"TRTP1234"));
        assert!(!is_trtp(b"ONXP1234"));
        assert!(!is_trtp(b"TRT")); // Too short
    }

    #[test]
    fn test_extension() {
        assert_eq!(extension(), "trtp");
    }

    #[test]
    fn test_crc_mismatch() {
        let patch = make_test_patch();
        let mut serialized = serialize_patch(&patch).unwrap();

        // Corrupt some data (not the CRC itself)
        if serialized.len() > 20 {
            serialized[20] ^= 0xFF;
        }

        let result = deserialize_patch(&serialized);
        assert!(matches!(
            result,
            Err(ParseError::Malformed(msg)) if msg.contains("CRC32")
        ));
    }

    #[test]
    fn test_config_roundtrip() {
        let patch = TensorRTPatch {
            onnx_patch: Patch {
                version: 1,
                source_hash: [0; 32],
                target_hash: [0; 32],
                operations: vec![],
                compression: CompressionMethod::None,
                metadata: PatchMetadata::default(),
            },
            config: TensorRTConfig::new()
                .with_precision(Precision::INT8)
                .with_calibration_cache("calib.cache")
                .with_dla_core(1)
                .with_strict_types(true)
                .with_sparsity(true)
                .with_min_version("10.0.0")
                .with_compute_capability("8.6"),
            source_onnx_hash: [0; 32],
            target_onnx_hash: [0; 32],
        };

        let serialized = serialize_patch(&patch).unwrap();
        let deserialized = deserialize_patch(&serialized).unwrap();

        assert_eq!(deserialized.config.precision, Precision::INT8);
        assert_eq!(
            deserialized.config.calibration_cache,
            Some("calib.cache".to_string())
        );
        assert_eq!(deserialized.config.dla_core, Some(1));
        assert!(deserialized.config.strict_types);
        assert!(deserialized.config.sparsity);
        assert_eq!(
            deserialized.config.min_tensorrt_version,
            Some("10.0.0".to_string())
        );
        assert_eq!(
            deserialized.config.compute_capability,
            Some("8.6".to_string())
        );
    }

    #[test]
    fn test_all_precisions() {
        for precision in [
            Precision::FP32,
            Precision::FP16,
            Precision::INT8,
            Precision::TF32,
            Precision::BF16,
        ] {
            let patch = TensorRTPatch {
                onnx_patch: Patch {
                    version: 1,
                    source_hash: [0; 32],
                    target_hash: [0; 32],
                    operations: vec![],
                    compression: CompressionMethod::None,
                    metadata: PatchMetadata::default(),
                },
                config: TensorRTConfig::new().with_precision(precision),
                source_onnx_hash: [0; 32],
                target_onnx_hash: [0; 32],
            };

            let serialized = serialize_patch(&patch).unwrap();
            let deserialized = deserialize_patch(&serialized).unwrap();

            assert_eq!(deserialized.config.precision, precision);
        }
    }
}
