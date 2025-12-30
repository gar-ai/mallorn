//! GGUF patch format (.ggup)
//!
//! Binary format for serializing GGUF model patches. Uses LZ4 compression
//! by default for compatibility with quantized tensor data.
//!
//! ## Format Layout
//!
//! ```text
//! [GGUP magic: 4 bytes]
//! [Version: u32 LE]
//! [Source hash: 32 bytes]
//! [Target hash: 32 bytes]
//! [Compression method: u8]
//! [Metadata length: u32 LE]
//! [Metadata: JSON]
//! [Num operations: u32 LE]
//! [Operations: variable]
//! [CRC32 checksum: u32 LE]
//! ```

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use mallorn_core::{
    crc32, CompressionMethod, DeltaFormat, NeuralCompressionVariant, ParseError, Patch,
    PatchMetadata, PatchOperation,
};
use std::io::{Cursor, Read, Write};

/// Magic bytes for GGUF patch files
pub const GGUP_MAGIC: &[u8; 4] = b"GGUP";

/// Current patch format version
pub const GGUP_VERSION: u32 = 1;

/// Operation type tags
mod op_tags {
    pub const REPLACE_TENSOR: u8 = 1;
    pub const DELTA_TENSOR: u8 = 2;
    pub const COPY_TENSOR: u8 = 3;
    pub const UPDATE_METADATA: u8 = 4;
}

/// Delta format tags
mod delta_tags {
    pub const XOR: u8 = 1;
    pub const BSDIFF: u8 = 2;
    pub const TENSOR_AWARE: u8 = 3;
}

/// Compression method tags
mod compression_tags {
    pub const NONE: u8 = 0;
    pub const ZSTD: u8 = 1;
    pub const LZ4: u8 = 2;
    pub const NEURAL: u8 = 3;
    pub const ZSTD_DICT: u8 = 5;
}

/// Serialize a patch to .ggup format
pub fn serialize_patch(patch: &Patch) -> Result<Vec<u8>, std::io::Error> {
    let mut buf = Vec::new();

    // Magic
    buf.write_all(GGUP_MAGIC)?;

    // Version
    buf.write_u32::<LittleEndian>(patch.version)?;

    // Hashes
    buf.write_all(&patch.source_hash)?;
    buf.write_all(&patch.target_hash)?;

    // Compression method
    match patch.compression {
        CompressionMethod::None => buf.write_u8(compression_tags::NONE)?,
        CompressionMethod::Zstd { level } => {
            buf.write_u8(compression_tags::ZSTD)?;
            buf.write_i8(level as i8)?;
        }
        CompressionMethod::Lz4 => buf.write_u8(compression_tags::LZ4)?,
        CompressionMethod::Neural { .. } => buf.write_u8(compression_tags::NEURAL)?,
        CompressionMethod::Adaptive { .. } => {
            // Adaptive falls back to LZ4 for GGUF
            buf.write_u8(compression_tags::LZ4)?;
        }
        CompressionMethod::ZstdDict { level, dict_id } => {
            buf.write_u8(compression_tags::ZSTD_DICT)?;
            buf.write_i8(level as i8)?;
            buf.write_u32::<LittleEndian>(dict_id)?;
        }
    }

    // Metadata as JSON
    let metadata_json = serde_json::to_string(&patch.metadata)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let metadata_bytes = metadata_json.as_bytes();
    buf.write_u32::<LittleEndian>(metadata_bytes.len() as u32)?;
    buf.write_all(metadata_bytes)?;

    // Number of operations
    buf.write_u32::<LittleEndian>(patch.operations.len() as u32)?;

    // Operations
    for op in &patch.operations {
        serialize_operation(&mut buf, op)?;
    }

    // CRC32 checksum
    let checksum = crc32(&buf);
    buf.write_u32::<LittleEndian>(checksum)?;

    Ok(buf)
}

/// Serialize a single operation
fn serialize_operation<W: Write>(writer: &mut W, op: &PatchOperation) -> Result<(), std::io::Error> {
    match op {
        PatchOperation::ReplaceTensor { name, data, .. } => {
            writer.write_u8(op_tags::REPLACE_TENSOR)?;
            write_string(writer, name)?;
            write_bytes(writer, data)?;
        }
        PatchOperation::DeltaTensor {
            name,
            delta,
            delta_format,
            ..
        } => {
            writer.write_u8(op_tags::DELTA_TENSOR)?;
            write_string(writer, name)?;
            write_bytes(writer, delta)?;
            let format_byte = match delta_format {
                DeltaFormat::Xor => delta_tags::XOR,
                DeltaFormat::BsDiff => delta_tags::BSDIFF,
                DeltaFormat::TensorAware => delta_tags::TENSOR_AWARE,
            };
            writer.write_u8(format_byte)?;
        }
        PatchOperation::CopyTensor { name } => {
            writer.write_u8(op_tags::COPY_TENSOR)?;
            write_string(writer, name)?;
        }
        PatchOperation::UpdateMetadata { key, value } => {
            writer.write_u8(op_tags::UPDATE_METADATA)?;
            write_string(writer, key)?;
            write_string(writer, value)?;
        }
    }
    Ok(())
}

/// Write a length-prefixed string
fn write_string<W: Write>(writer: &mut W, s: &str) -> Result<(), std::io::Error> {
    let bytes = s.as_bytes();
    writer.write_u32::<LittleEndian>(bytes.len() as u32)?;
    writer.write_all(bytes)?;
    Ok(())
}

/// Write a length-prefixed byte array
fn write_bytes<W: Write>(writer: &mut W, data: &[u8]) -> Result<(), std::io::Error> {
    writer.write_u32::<LittleEndian>(data.len() as u32)?;
    writer.write_all(data)?;
    Ok(())
}

/// Deserialize a .ggup patch file
pub fn deserialize_patch(data: &[u8]) -> Result<Patch, ParseError> {
    if data.len() < 4 {
        return Err(ParseError::Malformed("File too small".into()));
    }

    // Verify magic
    if &data[0..4] != GGUP_MAGIC {
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

    if version != GGUP_VERSION {
        return Err(ParseError::UnsupportedVersion(version));
    }

    // Hashes
    let mut source_hash = [0u8; 32];
    let mut target_hash = [0u8; 32];
    cursor
        .read_exact(&mut source_hash)
        .map_err(|e| ParseError::Malformed(e.to_string()))?;
    cursor
        .read_exact(&mut target_hash)
        .map_err(|e| ParseError::Malformed(e.to_string()))?;

    // Compression method
    let compression = read_compression_method(&mut cursor)?;

    // Metadata
    let metadata_len = cursor
        .read_u32::<LittleEndian>()
        .map_err(|e| ParseError::Malformed(e.to_string()))? as usize;
    let mut metadata_bytes = vec![0u8; metadata_len];
    cursor
        .read_exact(&mut metadata_bytes)
        .map_err(|e| ParseError::Malformed(e.to_string()))?;
    let metadata: PatchMetadata = serde_json::from_slice(&metadata_bytes)
        .map_err(|e| ParseError::Malformed(format!("Invalid metadata JSON: {}", e)))?;

    // Operations
    let num_ops = cursor
        .read_u32::<LittleEndian>()
        .map_err(|e| ParseError::Malformed(e.to_string()))? as usize;
    let mut operations = Vec::with_capacity(num_ops);
    for _ in 0..num_ops {
        let op = deserialize_operation(&mut cursor)?;
        operations.push(op);
    }

    Ok(Patch {
        version,
        source_hash,
        target_hash,
        operations,
        compression,
        metadata,
    })
}

/// Read compression method from stream
fn read_compression_method(cursor: &mut Cursor<&[u8]>) -> Result<CompressionMethod, ParseError> {
    let tag = cursor
        .read_u8()
        .map_err(|e| ParseError::Malformed(e.to_string()))?;

    match tag {
        compression_tags::NONE => Ok(CompressionMethod::None),
        compression_tags::ZSTD => {
            let level = cursor
                .read_i8()
                .map_err(|e| ParseError::Malformed(e.to_string()))? as i32;
            Ok(CompressionMethod::Zstd { level })
        }
        compression_tags::LZ4 => Ok(CompressionMethod::Lz4),
        compression_tags::NEURAL => Ok(CompressionMethod::Neural {
            variant: NeuralCompressionVariant::ExponentGrouping,
        }),
        compression_tags::ZSTD_DICT => {
            let level = cursor
                .read_i8()
                .map_err(|e| ParseError::Malformed(e.to_string()))? as i32;
            let dict_id = cursor
                .read_u32::<LittleEndian>()
                .map_err(|e| ParseError::Malformed(e.to_string()))?;
            Ok(CompressionMethod::ZstdDict { level, dict_id })
        }
        _ => Err(ParseError::Malformed(format!(
            "Unknown compression method: {}",
            tag
        ))),
    }
}

/// Deserialize a single operation
fn deserialize_operation(cursor: &mut Cursor<&[u8]>) -> Result<PatchOperation, ParseError> {
    let tag = cursor
        .read_u8()
        .map_err(|e| ParseError::Malformed(e.to_string()))?;

    match tag {
        op_tags::REPLACE_TENSOR => {
            let name = read_string(cursor)?;
            let data = read_bytes(cursor)?;
            Ok(PatchOperation::ReplaceTensor { name, data, compression: None })
        }
        op_tags::DELTA_TENSOR => {
            let name = read_string(cursor)?;
            let delta = read_bytes(cursor)?;
            let format_tag = cursor
                .read_u8()
                .map_err(|e| ParseError::Malformed(e.to_string()))?;
            let delta_format = match format_tag {
                delta_tags::XOR => DeltaFormat::Xor,
                delta_tags::BSDIFF => DeltaFormat::BsDiff,
                delta_tags::TENSOR_AWARE => DeltaFormat::TensorAware,
                _ => {
                    return Err(ParseError::Malformed(format!(
                        "Unknown delta format: {}",
                        format_tag
                    )))
                }
            };
            Ok(PatchOperation::DeltaTensor {
                name,
                delta,
                delta_format,
                compression: None,
            })
        }
        op_tags::COPY_TENSOR => {
            let name = read_string(cursor)?;
            Ok(PatchOperation::CopyTensor { name })
        }
        op_tags::UPDATE_METADATA => {
            let key = read_string(cursor)?;
            let value = read_string(cursor)?;
            Ok(PatchOperation::UpdateMetadata { key, value })
        }
        _ => Err(ParseError::Malformed(format!(
            "Unknown operation type: {}",
            tag
        ))),
    }
}

/// Read a length-prefixed string
fn read_string(cursor: &mut Cursor<&[u8]>) -> Result<String, ParseError> {
    let len = cursor
        .read_u32::<LittleEndian>()
        .map_err(|e| ParseError::Malformed(e.to_string()))? as usize;
    let mut bytes = vec![0u8; len];
    cursor
        .read_exact(&mut bytes)
        .map_err(|e| ParseError::Malformed(e.to_string()))?;
    String::from_utf8(bytes).map_err(|e| ParseError::Malformed(format!("Invalid UTF-8: {}", e)))
}

/// Read a length-prefixed byte array
fn read_bytes(cursor: &mut Cursor<&[u8]>) -> Result<Vec<u8>, ParseError> {
    let len = cursor
        .read_u32::<LittleEndian>()
        .map_err(|e| ParseError::Malformed(e.to_string()))? as usize;
    let mut bytes = vec![0u8; len];
    cursor
        .read_exact(&mut bytes)
        .map_err(|e| ParseError::Malformed(e.to_string()))?;
    Ok(bytes)
}

/// Get file extension for GGUF patches
pub fn extension() -> &'static str {
    "ggup"
}

/// Check if data looks like a GGUF patch
pub fn is_ggup(data: &[u8]) -> bool {
    data.len() >= 4 && &data[0..4] == GGUP_MAGIC
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_patch() -> Patch {
        Patch {
            version: GGUP_VERSION,
            source_hash: [0xAA; 32],
            target_hash: [0xBB; 32],
            operations: vec![
                PatchOperation::CopyTensor {
                    name: "model.layers.0.weight".into(),
                },
                PatchOperation::ReplaceTensor {
                    name: "model.layers.1.weight".into(),
                    data: vec![1, 2, 3, 4, 5],
                    compression: None,
                },
                PatchOperation::DeltaTensor {
                    name: "model.layers.2.weight".into(),
                    delta: vec![0, 0, 1, 0, 0],
                    delta_format: DeltaFormat::Xor,
                    compression: None,
                },
            ],
            compression: CompressionMethod::Lz4,
            metadata: PatchMetadata {
                source_version: Some("tinyllama-v1.0".into()),
                target_version: Some("tinyllama-v1.1".into()),
                created_at: 1234567890,
                description: Some("Test GGUF patch".into()),
            },
        }
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let patch = make_test_patch();
        let serialized = serialize_patch(&patch).unwrap();

        assert_eq!(&serialized[0..4], GGUP_MAGIC);

        let deserialized = deserialize_patch(&serialized).unwrap();

        assert_eq!(deserialized.version, patch.version);
        assert_eq!(deserialized.source_hash, patch.source_hash);
        assert_eq!(deserialized.target_hash, patch.target_hash);
        assert_eq!(deserialized.operations.len(), patch.operations.len());
    }

    #[test]
    fn test_invalid_magic() {
        let data = b"BAAD1234567890";
        let result = deserialize_patch(data);
        assert!(matches!(result, Err(ParseError::InvalidMagic)));
    }

    #[test]
    fn test_is_ggup() {
        assert!(is_ggup(b"GGUP1234"));
        assert!(!is_ggup(b"TFLP1234"));
        assert!(!is_ggup(b"GGU")); // Too short
    }

    #[test]
    fn test_extension() {
        assert_eq!(extension(), "ggup");
    }

    #[test]
    fn test_crc_mismatch() {
        let patch = make_test_patch();
        let mut serialized = serialize_patch(&patch).unwrap();

        if serialized.len() > 10 {
            serialized[10] ^= 0xFF;
        }

        let result = deserialize_patch(&serialized);
        assert!(matches!(
            result,
            Err(ParseError::Malformed(msg)) if msg.contains("CRC32")
        ));
    }

    #[test]
    fn test_all_compression_methods() {
        for compression in [
            CompressionMethod::None,
            CompressionMethod::Lz4,
            CompressionMethod::Zstd { level: 5 },
        ] {
            let patch = Patch {
                version: GGUP_VERSION,
                source_hash: [0; 32],
                target_hash: [0; 32],
                operations: vec![],
                compression,
                metadata: PatchMetadata::default(),
            };

            let serialized = serialize_patch(&patch).unwrap();
            let deserialized = deserialize_patch(&serialized).unwrap();

            match (&patch.compression, &deserialized.compression) {
                (CompressionMethod::Lz4, CompressionMethod::Lz4) => {}
                (CompressionMethod::Zstd { level: l1 }, CompressionMethod::Zstd { level: l2 }) => {
                    assert_eq!(l1, l2);
                }
                (CompressionMethod::None, CompressionMethod::None) => {}
                _ => panic!("Compression method mismatch"),
            }
        }
    }
}
