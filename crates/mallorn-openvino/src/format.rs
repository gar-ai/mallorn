//! OpenVINO format definitions and patch serialization
//!
//! OpenVINO IR model format structure:
//! - `.xml` - Model graph structure (layers, connections, metadata)
//! - `.bin` - Binary weights data (tensors)
//!
//! OpenVINO patch format (.ovinp):
//! ┌─────────────────────────────────┐
//! │ OVIN magic (4 bytes)            │
//! │ Version (u32 LE)                │
//! │ Source hash (32 bytes)          │
//! │ Target hash (32 bytes)          │
//! │ Compression method (1-2 bytes)  │
//! │ Metadata length (u32 LE)        │
//! │ Metadata (JSON)                 │
//! │ Num operations (u32 LE)         │
//! │ Operations (variable)           │
//! │ CRC32 checksum (u32 LE)         │
//! └─────────────────────────────────┘

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use mallorn_core::error::SerializeError;
use mallorn_core::traits::ModelFormat;
use mallorn_core::{
    crc32, CompressionMethod, DeltaFormat, NeuralCompressionVariant, ParseError, ParsedModel,
    Patch, PatchMetadata, PatchOperation, TensorInfo,
};
use std::io::{Cursor, Read, Write};

use crate::parser::OpenVINOParser;

/// File extensions for OpenVINO format
pub const OPENVINO_EXTENSIONS: &[&str] = &["xml"];

/// Patch file extension for OpenVINO
pub const OPENVINO_PATCH_EXTENSION: &str = "ovinp";

/// Magic bytes for OpenVINO patch files
pub const OVIN_MAGIC: &[u8; 4] = b"OVIN";

/// Current patch format version
pub const OVIN_VERSION: u32 = 1;

/// Operation type tags for serialization
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

/// OpenVINO format handler implementing ModelFormat trait
#[derive(Debug, Clone, Default)]
pub struct OpenVINOFormat {
    parser: OpenVINOParser,
}

impl OpenVINOFormat {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ModelFormat for OpenVINOFormat {
    fn format_id(&self) -> &'static str {
        "openvino"
    }

    fn extensions(&self) -> &[&'static str] {
        OPENVINO_EXTENSIONS
    }

    fn parse(&self, data: &[u8]) -> Result<ParsedModel, ParseError> {
        let model = self.parser.parse(data)?;
        Ok(model.into_parsed_model())
    }

    fn serialize(&self, _model: &ParsedModel) -> Result<Vec<u8>, SerializeError> {
        // OpenVINO serialization would require rebuilding XML structure
        // This is not currently supported - use the patcher for model updates
        Err(SerializeError::Failed(
            "OpenVINO model serialization not supported - use patcher for updates".into(),
        ))
    }

    fn extract_tensor_info(&self, data: &[u8]) -> Result<Vec<TensorInfo>, ParseError> {
        self.parser.extract_tensor_info(data)
    }
}

/// Serialize a patch to .ovinp format
pub fn serialize_patch(patch: &Patch) -> Result<Vec<u8>, std::io::Error> {
    let mut buf = Vec::new();

    // Magic
    buf.write_all(OVIN_MAGIC)?;

    // Version
    buf.write_u32::<LittleEndian>(patch.version)?;

    // Hashes
    buf.write_all(&patch.source_hash)?;
    buf.write_all(&patch.target_hash)?;

    // Compression method
    match patch.compression {
        CompressionMethod::None => {
            buf.write_u8(compression_tags::NONE)?;
        }
        CompressionMethod::Zstd { level } => {
            buf.write_u8(compression_tags::ZSTD)?;
            buf.write_i8(level as i8)?;
        }
        CompressionMethod::Lz4 => {
            buf.write_u8(compression_tags::LZ4)?;
        }
        CompressionMethod::Neural { .. } => {
            buf.write_u8(compression_tags::NEURAL)?;
        }
        CompressionMethod::Adaptive { .. } => {
            // Adaptive falls back to Zstd for serialization
            buf.write_u8(compression_tags::ZSTD)?;
            buf.write_i8(3)?;
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

    // Calculate CRC32 of everything so far
    let checksum = crc32(&buf);
    buf.write_u32::<LittleEndian>(checksum)?;

    Ok(buf)
}

/// Serialize a single operation
fn serialize_operation<W: Write>(
    writer: &mut W,
    op: &PatchOperation,
) -> Result<(), std::io::Error> {
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

/// Deserialize a .ovinp patch file
pub fn deserialize_patch(data: &[u8]) -> Result<Patch, ParseError> {
    if data.len() < 4 {
        return Err(ParseError::Malformed("File too small".into()));
    }

    // Verify magic
    if &data[0..4] != OVIN_MAGIC {
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

    if version != OVIN_VERSION {
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
            Ok(PatchOperation::ReplaceTensor {
                name,
                data,
                compression: None,
            })
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

/// Get file extension for OpenVINO patches
pub fn extension() -> &'static str {
    OPENVINO_PATCH_EXTENSION
}

/// Check if data looks like an OpenVINO patch
pub fn is_ovinp(data: &[u8]) -> bool {
    data.len() >= 4 && &data[0..4] == OVIN_MAGIC
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_patch() -> Patch {
        Patch {
            version: OVIN_VERSION,
            source_hash: [0xAA; 32],
            target_hash: [0xBB; 32],
            operations: vec![
                PatchOperation::CopyTensor {
                    name: "tensor1".into(),
                },
                PatchOperation::ReplaceTensor {
                    name: "tensor2".into(),
                    data: vec![1, 2, 3, 4, 5],
                    compression: None,
                },
                PatchOperation::DeltaTensor {
                    name: "tensor3".into(),
                    delta: vec![0, 0, 1, 0, 0],
                    delta_format: DeltaFormat::Xor,
                    compression: None,
                },
            ],
            compression: CompressionMethod::Zstd { level: 3 },
            metadata: PatchMetadata {
                source_version: Some("v1.0".into()),
                target_version: Some("v1.1".into()),
                created_at: 1234567890,
                description: Some("Test patch".into()),
            },
        }
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let patch = make_test_patch();
        let serialized = serialize_patch(&patch).unwrap();

        // Check magic
        assert_eq!(&serialized[0..4], OVIN_MAGIC);

        // Deserialize
        let deserialized = deserialize_patch(&serialized).unwrap();

        // Verify fields match
        assert_eq!(deserialized.version, patch.version);
        assert_eq!(deserialized.source_hash, patch.source_hash);
        assert_eq!(deserialized.target_hash, patch.target_hash);
        assert_eq!(deserialized.operations.len(), patch.operations.len());
    }

    #[test]
    fn test_is_ovinp() {
        assert!(is_ovinp(b"OVIN1234"));
        assert!(!is_ovinp(b"SFTP1234"));
        assert!(!is_ovinp(b"OVI")); // Too short
    }

    #[test]
    fn test_extension() {
        assert_eq!(extension(), "ovinp");
    }

    #[test]
    fn test_openvino_format_trait() {
        let format = OpenVINOFormat::new();
        assert_eq!(format.format_id(), "openvino");
        assert_eq!(format.extensions(), &["xml"]);
    }
}
