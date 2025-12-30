//! GGUF format parser
//!
//! Parses GGUF model files used by llama.cpp and compatible tools.
//! Supports GGUF v2 and v3 formats.

use byteorder::{LittleEndian, ReadBytesExt};
use mallorn_core::{DataType, ParseError};
use std::collections::HashMap;
use std::io::{Cursor, Read};

/// GGUF file magic bytes (little-endian "GGUF")
pub const GGUF_MAGIC: u32 = 0x46554747;

/// GGUF metadata value type identifiers
mod value_types {
    pub const UINT8: u32 = 0;
    pub const INT8: u32 = 1;
    pub const UINT16: u32 = 2;
    pub const INT16: u32 = 3;
    pub const UINT32: u32 = 4;
    pub const INT32: u32 = 5;
    pub const FLOAT32: u32 = 6;
    pub const BOOL: u32 = 7;
    pub const STRING: u32 = 8;
    pub const ARRAY: u32 = 9;
    pub const UINT64: u32 = 10;
    pub const INT64: u32 = 11;
    pub const FLOAT64: u32 = 12;
}

/// Parsed GGUF model
#[derive(Debug, Clone)]
pub struct GGUFModel {
    /// GGUF format version (2 or 3)
    pub version: u32,
    /// Number of tensors
    pub tensor_count: u64,
    /// Metadata key-value pairs
    pub metadata: HashMap<String, GGUFValue>,
    /// Tensors in the model
    pub tensors: Vec<GGUFTensor>,
    /// Raw model bytes (for hashing)
    pub raw_data: Vec<u8>,
    /// Alignment (default 32)
    pub alignment: u64,
}

/// A tensor in a GGUF model
#[derive(Debug, Clone)]
pub struct GGUFTensor {
    /// Tensor name
    pub name: String,
    /// Number of dimensions
    pub n_dimensions: u32,
    /// Shape dimensions (stored in reverse order in GGUF)
    pub dimensions: Vec<u64>,
    /// GGML quantization type
    pub ggml_type: GGMLType,
    /// Offset in file (relative to data section start)
    pub offset: u64,
    /// Tensor data
    pub data: Vec<u8>,
}

/// GGML tensor types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
}

impl GGMLType {
    /// Convert from raw value
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            16 => Some(Self::IQ2XXS),
            17 => Some(Self::IQ2XS),
            18 => Some(Self::IQ3XXS),
            19 => Some(Self::IQ1S),
            20 => Some(Self::IQ4NL),
            21 => Some(Self::IQ3S),
            22 => Some(Self::IQ2S),
            23 => Some(Self::IQ4XS),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            27 => Some(Self::I64),
            28 => Some(Self::F64),
            29 => Some(Self::IQ1M),
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    /// Get the block size for this type
    pub fn block_size(&self) -> usize {
        match self {
            GGMLType::F32 | GGMLType::F16 | GGMLType::BF16 | GGMLType::F64 => 1,
            GGMLType::I8 | GGMLType::I16 | GGMLType::I32 | GGMLType::I64 => 1,
            GGMLType::Q4_0 | GGMLType::Q4_1 | GGMLType::Q5_0 | GGMLType::Q5_1 => 32,
            GGMLType::Q8_0 | GGMLType::Q8_1 => 32,
            GGMLType::Q2K | GGMLType::Q3K | GGMLType::Q4K => 256,
            GGMLType::Q5K | GGMLType::Q6K | GGMLType::Q8K => 256,
            _ => 32,
        }
    }

    /// Get bytes per block for this type
    pub fn type_size(&self) -> usize {
        match self {
            GGMLType::F32 => 4,
            GGMLType::F16 | GGMLType::BF16 => 2,
            GGMLType::F64 => 8,
            GGMLType::I8 => 1,
            GGMLType::I16 => 2,
            GGMLType::I32 => 4,
            GGMLType::I64 => 8,
            GGMLType::Q4_0 => 18,
            GGMLType::Q4_1 => 20,
            GGMLType::Q5_0 => 22,
            GGMLType::Q5_1 => 24,
            GGMLType::Q8_0 => 34,
            GGMLType::Q8_1 => 36,
            GGMLType::Q2K => 84,
            GGMLType::Q3K => 110,
            GGMLType::Q4K => 144,
            GGMLType::Q5K => 176,
            GGMLType::Q6K => 210,
            GGMLType::Q8K => 292,
            _ => 18,
        }
    }
}

impl From<GGMLType> for DataType {
    fn from(t: GGMLType) -> Self {
        match t {
            GGMLType::F32 => DataType::Float32,
            GGMLType::F16 => DataType::Float16,
            GGMLType::BF16 => DataType::BFloat16,
            GGMLType::F64 => DataType::Float64,
            GGMLType::I8 => DataType::Int8,
            GGMLType::I16 => DataType::Int16,
            GGMLType::I32 => DataType::Int32,
            GGMLType::I64 => DataType::Int64,
            GGMLType::Q4_0 | GGMLType::Q4_1 => DataType::Q4_0,
            GGMLType::Q5_0 | GGMLType::Q5_1 => DataType::Q5_0,
            GGMLType::Q8_0 | GGMLType::Q8_1 => DataType::Q8_0,
            GGMLType::Q2K => DataType::Q2K,
            GGMLType::Q3K => DataType::Q3K,
            GGMLType::Q4K => DataType::Q4K,
            GGMLType::Q5K => DataType::Q5K,
            GGMLType::Q6K => DataType::Q6K,
            GGMLType::Q8K => DataType::Q8K,
            _ => DataType::UInt8,
        }
    }
}

/// GGUF metadata value types
#[derive(Debug, Clone)]
pub enum GGUFValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
}

impl GGUFValue {
    /// Get string value if this is a string
    pub fn as_string(&self) -> Option<&str> {
        if let GGUFValue::String(s) = self {
            Some(s)
        } else {
            None
        }
    }

    /// Get u64 value
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GGUFValue::UInt64(v) => Some(*v),
            GGUFValue::UInt32(v) => Some(*v as u64),
            GGUFValue::UInt16(v) => Some(*v as u64),
            GGUFValue::UInt8(v) => Some(*v as u64),
            _ => None,
        }
    }
}

/// GGUF model parser
pub struct GGUFParser;

impl GGUFParser {
    /// Create a new parser
    pub fn new() -> Self {
        Self
    }

    /// Parse a GGUF model from bytes
    pub fn parse(&self, data: &[u8]) -> Result<GGUFModel, ParseError> {
        if data.len() < 24 {
            return Err(ParseError::Malformed("File too small".into()));
        }

        let mut cursor = Cursor::new(data);

        // Read and verify magic
        let magic = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| ParseError::Malformed(e.to_string()))?;
        if magic != GGUF_MAGIC {
            return Err(ParseError::InvalidMagic);
        }

        // Read version
        let version = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| ParseError::Malformed(e.to_string()))?;
        if version < 2 || version > 3 {
            return Err(ParseError::UnsupportedVersion(version));
        }

        // Read tensor count and metadata count
        let tensor_count = cursor
            .read_u64::<LittleEndian>()
            .map_err(|e| ParseError::Malformed(e.to_string()))?;
        let metadata_kv_count = cursor
            .read_u64::<LittleEndian>()
            .map_err(|e| ParseError::Malformed(e.to_string()))?;

        // Parse metadata
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let (key, value) = self.read_metadata_kv(&mut cursor)?;
            metadata.insert(key, value);
        }

        // Get alignment from metadata (default 32)
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32);

        // Parse tensor info
        let mut tensor_infos = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let info = self.read_tensor_info(&mut cursor)?;
            tensor_infos.push(info);
        }

        // Calculate data section start (aligned)
        let header_end = cursor.position();
        let data_start = align_offset(header_end, alignment);

        // Read tensor data
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for info in tensor_infos {
            let tensor_offset = data_start + info.offset;
            let tensor_size = calculate_tensor_size(&info);

            let tensor_data = if (tensor_offset as usize) + tensor_size <= data.len() {
                data[tensor_offset as usize..(tensor_offset as usize) + tensor_size].to_vec()
            } else {
                Vec::new()
            };

            tensors.push(GGUFTensor {
                name: info.name,
                n_dimensions: info.n_dimensions,
                dimensions: info.dimensions,
                ggml_type: info.ggml_type,
                offset: info.offset,
                data: tensor_data,
            });
        }

        Ok(GGUFModel {
            version,
            tensor_count,
            metadata,
            tensors,
            raw_data: data.to_vec(),
            alignment,
        })
    }

    /// Read a metadata key-value pair
    fn read_metadata_kv(
        &self,
        cursor: &mut Cursor<&[u8]>,
    ) -> Result<(String, GGUFValue), ParseError> {
        let key = self.read_string(cursor)?;
        let value_type = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| ParseError::Malformed(e.to_string()))?;
        let value = self.read_value(cursor, value_type)?;
        Ok((key, value))
    }

    /// Read a GGUF value of the given type
    fn read_value(
        &self,
        cursor: &mut Cursor<&[u8]>,
        value_type: u32,
    ) -> Result<GGUFValue, ParseError> {
        match value_type {
            value_types::UINT8 => Ok(GGUFValue::UInt8(
                cursor
                    .read_u8()
                    .map_err(|e| ParseError::Malformed(e.to_string()))?,
            )),
            value_types::INT8 => Ok(GGUFValue::Int8(
                cursor
                    .read_i8()
                    .map_err(|e| ParseError::Malformed(e.to_string()))?,
            )),
            value_types::UINT16 => Ok(GGUFValue::UInt16(
                cursor
                    .read_u16::<LittleEndian>()
                    .map_err(|e| ParseError::Malformed(e.to_string()))?,
            )),
            value_types::INT16 => Ok(GGUFValue::Int16(
                cursor
                    .read_i16::<LittleEndian>()
                    .map_err(|e| ParseError::Malformed(e.to_string()))?,
            )),
            value_types::UINT32 => Ok(GGUFValue::UInt32(
                cursor
                    .read_u32::<LittleEndian>()
                    .map_err(|e| ParseError::Malformed(e.to_string()))?,
            )),
            value_types::INT32 => Ok(GGUFValue::Int32(
                cursor
                    .read_i32::<LittleEndian>()
                    .map_err(|e| ParseError::Malformed(e.to_string()))?,
            )),
            value_types::FLOAT32 => Ok(GGUFValue::Float32(
                cursor
                    .read_f32::<LittleEndian>()
                    .map_err(|e| ParseError::Malformed(e.to_string()))?,
            )),
            value_types::BOOL => Ok(GGUFValue::Bool(
                cursor
                    .read_u8()
                    .map_err(|e| ParseError::Malformed(e.to_string()))?
                    != 0,
            )),
            value_types::STRING => Ok(GGUFValue::String(self.read_string(cursor)?)),
            value_types::ARRAY => {
                let element_type = cursor
                    .read_u32::<LittleEndian>()
                    .map_err(|e| ParseError::Malformed(e.to_string()))?;
                let len = cursor
                    .read_u64::<LittleEndian>()
                    .map_err(|e| ParseError::Malformed(e.to_string()))? as usize;
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(self.read_value(cursor, element_type)?);
                }
                Ok(GGUFValue::Array(arr))
            }
            value_types::UINT64 => Ok(GGUFValue::UInt64(
                cursor
                    .read_u64::<LittleEndian>()
                    .map_err(|e| ParseError::Malformed(e.to_string()))?,
            )),
            value_types::INT64 => Ok(GGUFValue::Int64(
                cursor
                    .read_i64::<LittleEndian>()
                    .map_err(|e| ParseError::Malformed(e.to_string()))?,
            )),
            value_types::FLOAT64 => Ok(GGUFValue::Float64(
                cursor
                    .read_f64::<LittleEndian>()
                    .map_err(|e| ParseError::Malformed(e.to_string()))?,
            )),
            _ => Err(ParseError::Malformed(format!(
                "Unknown value type: {}",
                value_type
            ))),
        }
    }

    /// Read a length-prefixed string
    fn read_string(&self, cursor: &mut Cursor<&[u8]>) -> Result<String, ParseError> {
        let len = cursor
            .read_u64::<LittleEndian>()
            .map_err(|e| ParseError::Malformed(e.to_string()))? as usize;
        let mut buf = vec![0u8; len];
        cursor
            .read_exact(&mut buf)
            .map_err(|e| ParseError::Malformed(e.to_string()))?;
        String::from_utf8(buf).map_err(|e| ParseError::Malformed(format!("Invalid UTF-8: {}", e)))
    }

    /// Read tensor info (metadata, not data)
    fn read_tensor_info(&self, cursor: &mut Cursor<&[u8]>) -> Result<TensorInfo, ParseError> {
        let name = self.read_string(cursor)?;
        let n_dimensions = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| ParseError::Malformed(e.to_string()))?;

        let mut dimensions = Vec::with_capacity(n_dimensions as usize);
        for _ in 0..n_dimensions {
            dimensions.push(
                cursor
                    .read_u64::<LittleEndian>()
                    .map_err(|e| ParseError::Malformed(e.to_string()))?,
            );
        }

        let ggml_type_raw = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| ParseError::Malformed(e.to_string()))?;
        let ggml_type = GGMLType::from_u32(ggml_type_raw).ok_or_else(|| {
            ParseError::Malformed(format!("Unknown GGML type: {}", ggml_type_raw))
        })?;

        let offset = cursor
            .read_u64::<LittleEndian>()
            .map_err(|e| ParseError::Malformed(e.to_string()))?;

        Ok(TensorInfo {
            name,
            n_dimensions,
            dimensions,
            ggml_type,
            offset,
        })
    }
}

/// Temporary struct for tensor info during parsing
struct TensorInfo {
    name: String,
    n_dimensions: u32,
    dimensions: Vec<u64>,
    ggml_type: GGMLType,
    offset: u64,
}

/// Calculate the size of tensor data in bytes
fn calculate_tensor_size(info: &TensorInfo) -> usize {
    let n_elements: u64 = info.dimensions.iter().product();
    if n_elements == 0 {
        return 0;
    }
    let block_size = info.ggml_type.block_size() as u64;
    let type_size = info.ggml_type.type_size() as u64;
    let n_blocks = (n_elements + block_size - 1) / block_size;
    (n_blocks * type_size) as usize
}

impl Default for GGUFParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Align offset to boundary
fn align_offset(offset: u64, alignment: u64) -> u64 {
    (offset + alignment - 1) / alignment * alignment
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_creation() {
        let _parser = GGUFParser::new();
    }

    #[test]
    fn test_parse_empty_fails() {
        let parser = GGUFParser::new();
        assert!(parser.parse(&[]).is_err());
    }

    #[test]
    fn test_parse_too_small_fails() {
        let parser = GGUFParser::new();
        assert!(parser.parse(&[1, 2, 3, 4]).is_err());
    }

    #[test]
    fn test_invalid_magic() {
        let parser = GGUFParser::new();
        let data = [0u8; 24];
        let result = parser.parse(&data);
        assert!(matches!(result, Err(ParseError::InvalidMagic)));
    }

    #[test]
    fn test_ggml_type_conversion() {
        assert_eq!(GGMLType::from_u32(0), Some(GGMLType::F32));
        assert_eq!(GGMLType::from_u32(1), Some(GGMLType::F16));
        assert_eq!(GGMLType::from_u32(12), Some(GGMLType::Q4K));
        assert_eq!(GGMLType::from_u32(255), None);
    }

    #[test]
    fn test_ggml_type_sizes() {
        assert_eq!(GGMLType::F32.type_size(), 4);
        assert_eq!(GGMLType::F16.type_size(), 2);
        assert_eq!(GGMLType::Q4_0.type_size(), 18);
        assert_eq!(GGMLType::Q4K.type_size(), 144);
    }

    #[test]
    fn test_align_offset() {
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(33, 32), 64);
    }

    #[test]
    fn test_gguf_value_accessors() {
        let s = GGUFValue::String("test".into());
        assert_eq!(s.as_string(), Some("test"));

        let n = GGUFValue::UInt32(42);
        assert_eq!(n.as_u64(), Some(42));
    }

    #[test]
    fn test_datatype_conversion() {
        assert_eq!(DataType::from(GGMLType::F32), DataType::Float32);
        assert_eq!(DataType::from(GGMLType::Q4K), DataType::Q4K);
    }
}
