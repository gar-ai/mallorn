//! TFLite FlatBuffer parser
//!
//! Parses TensorFlow Lite model files (.tflite) to extract tensor data
//! for diff/patch operations.

use byteorder::{LittleEndian, ReadBytesExt};
use mallorn_core::{DataType, ParseError, QuantizationParams};
use std::io::Cursor;

/// Parsed TFLite model
#[derive(Debug, Clone)]
pub struct TFLiteModel {
    /// TFLite schema version
    pub version: u32,
    /// Model description
    pub description: Option<String>,
    /// All tensors in the model (flattened from subgraphs)
    pub tensors: Vec<TFLiteTensor>,
    /// Raw model bytes (for hashing)
    pub raw_data: Vec<u8>,
}

/// A tensor in a TFLite model
#[derive(Debug, Clone)]
pub struct TFLiteTensor {
    /// Tensor name
    pub name: String,
    /// Shape dimensions
    pub shape: Vec<i32>,
    /// Data type
    pub dtype: TFLiteDataType,
    /// Buffer index in the model
    pub buffer_index: u32,
    /// Actual tensor data
    pub data: Vec<u8>,
    /// Quantization parameters (if quantized)
    pub quantization: Option<QuantizationParams>,
}

/// TFLite data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TFLiteDataType {
    Float32 = 0,
    Float16 = 1,
    Int32 = 2,
    UInt8 = 3,
    Int64 = 4,
    String = 5,
    Bool = 6,
    Int16 = 7,
    Complex64 = 8,
    Int8 = 9,
    Float64 = 10,
    Complex128 = 11,
    UInt64 = 12,
    Resource = 13,
    Variant = 14,
    UInt32 = 15,
    UInt16 = 16,
    Int4 = 17,
}

impl TFLiteDataType {
    fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Float32),
            1 => Some(Self::Float16),
            2 => Some(Self::Int32),
            3 => Some(Self::UInt8),
            4 => Some(Self::Int64),
            5 => Some(Self::String),
            6 => Some(Self::Bool),
            7 => Some(Self::Int16),
            8 => Some(Self::Complex64),
            9 => Some(Self::Int8),
            10 => Some(Self::Float64),
            11 => Some(Self::Complex128),
            12 => Some(Self::UInt64),
            13 => Some(Self::Resource),
            14 => Some(Self::Variant),
            15 => Some(Self::UInt32),
            16 => Some(Self::UInt16),
            17 => Some(Self::Int4),
            _ => None,
        }
    }

    /// Size of one element in bytes
    pub fn element_size(&self) -> usize {
        match self {
            TFLiteDataType::Float32 => 4,
            TFLiteDataType::Float16 => 2,
            TFLiteDataType::Int32 => 4,
            TFLiteDataType::UInt8 => 1,
            TFLiteDataType::Int64 => 8,
            TFLiteDataType::Bool => 1,
            TFLiteDataType::Int16 => 2,
            TFLiteDataType::Complex64 => 8,
            TFLiteDataType::Int8 => 1,
            TFLiteDataType::Float64 => 8,
            TFLiteDataType::Complex128 => 16,
            TFLiteDataType::UInt64 => 8,
            TFLiteDataType::UInt32 => 4,
            TFLiteDataType::UInt16 => 2,
            TFLiteDataType::Int4 => 1, // Packed, so this is approximate
            _ => 1,
        }
    }
}

impl From<TFLiteDataType> for DataType {
    fn from(dtype: TFLiteDataType) -> Self {
        match dtype {
            TFLiteDataType::Float32 => DataType::Float32,
            TFLiteDataType::Float16 => DataType::Float16,
            TFLiteDataType::Int8 => DataType::Int8,
            TFLiteDataType::UInt8 => DataType::UInt8,
            TFLiteDataType::Int16 => DataType::Int16,
            TFLiteDataType::Int32 => DataType::Int32,
            TFLiteDataType::Int64 => DataType::Int64,
            TFLiteDataType::UInt16 => DataType::UInt16,
            TFLiteDataType::UInt32 => DataType::UInt32,
            TFLiteDataType::UInt64 => DataType::UInt64,
            TFLiteDataType::Float64 => DataType::Float64,
            _ => DataType::Float32, // Default fallback
        }
    }
}

/// TFLite model parser
pub struct TFLiteParser;

impl TFLiteParser {
    /// Create a new parser
    pub fn new() -> Self {
        Self
    }

    /// Parse a TFLite model from bytes
    pub fn parse(&self, data: &[u8]) -> Result<TFLiteModel, ParseError> {
        if data.len() < 8 {
            return Err(ParseError::Malformed("File too small".into()));
        }

        // FlatBuffer structure:
        // Offset 0: i32 - offset to root table
        // Offset 4: optional file identifier (4 bytes, may be "TFL3" or empty)

        let mut cursor = Cursor::new(data);

        // Read root table offset
        let root_offset = cursor
            .read_i32::<LittleEndian>()
            .map_err(|e| ParseError::Malformed(e.to_string()))?;

        if root_offset < 0 || root_offset as usize >= data.len() {
            return Err(ParseError::Malformed("Invalid root offset".into()));
        }

        // Parse the Model table
        let model_pos = root_offset as usize;
        let (version, description, subgraph_offsets, buffer_offsets) =
            self.parse_model_table(data, model_pos)?;

        // Parse buffers first (we need them for tensor data)
        let buffers = self.parse_buffers(data, &buffer_offsets)?;

        // Parse tensors from all subgraphs
        let mut tensors = Vec::new();
        for subgraph_offset in subgraph_offsets {
            let subgraph_tensors = self.parse_subgraph_tensors(data, subgraph_offset, &buffers)?;
            tensors.extend(subgraph_tensors);
        }

        Ok(TFLiteModel {
            version,
            description,
            tensors,
            raw_data: data.to_vec(),
        })
    }

    /// Parse the root Model table
    fn parse_model_table(
        &self,
        data: &[u8],
        model_pos: usize,
    ) -> Result<(u32, Option<String>, Vec<usize>, Vec<usize>), ParseError> {
        // Read vtable offset (negative offset from table start)
        let vtable_offset = read_i32(data, model_pos)?;
        let vtable_pos = (model_pos as i32 - vtable_offset) as usize;

        // Read vtable
        let vtable_size = read_u16(data, vtable_pos)?;
        let _table_size = read_u16(data, vtable_pos + 2)?;

        // Model table fields (offsets in vtable):
        // 0: version (u32)
        // 1: operator_codes (vector)
        // 2: subgraphs (vector)
        // 3: description (string)
        // 4: buffers (vector)

        let num_fields = (vtable_size as usize - 4) / 2;

        // Read field offsets from vtable
        let mut field_offsets = Vec::with_capacity(num_fields);
        for i in 0..num_fields {
            let offset = read_u16(data, vtable_pos + 4 + i * 2)?;
            field_offsets.push(offset);
        }

        // Parse version (field 0)
        let version = if !field_offsets.is_empty() && field_offsets[0] != 0 {
            read_u32(data, model_pos + field_offsets[0] as usize)?
        } else {
            3 // Default TFLite version
        };

        // Parse description (field 3)
        let description = if field_offsets.len() > 3 && field_offsets[3] != 0 {
            let desc_offset_pos = model_pos + field_offsets[3] as usize;
            let desc_offset = read_i32(data, desc_offset_pos)?;
            let desc_pos = (desc_offset_pos as i32 + desc_offset) as usize;
            Some(self.parse_string(data, desc_pos)?)
        } else {
            None
        };

        // Parse subgraphs vector (field 2)
        let subgraph_offsets = if field_offsets.len() > 2 && field_offsets[2] != 0 {
            let vec_offset_pos = model_pos + field_offsets[2] as usize;
            let vec_offset = read_i32(data, vec_offset_pos)?;
            let vec_pos = (vec_offset_pos as i32 + vec_offset) as usize;
            self.parse_offset_vector(data, vec_pos)?
        } else {
            Vec::new()
        };

        // Parse buffers vector (field 4)
        let buffer_offsets = if field_offsets.len() > 4 && field_offsets[4] != 0 {
            let vec_offset_pos = model_pos + field_offsets[4] as usize;
            let vec_offset = read_i32(data, vec_offset_pos)?;
            let vec_pos = (vec_offset_pos as i32 + vec_offset) as usize;
            self.parse_offset_vector(data, vec_pos)?
        } else {
            Vec::new()
        };

        Ok((version, description, subgraph_offsets, buffer_offsets))
    }

    /// Parse a vector of offsets
    fn parse_offset_vector(&self, data: &[u8], vec_pos: usize) -> Result<Vec<usize>, ParseError> {
        let count = read_u32(data, vec_pos)? as usize;
        let mut offsets = Vec::with_capacity(count);

        for i in 0..count {
            let elem_pos = vec_pos + 4 + i * 4;
            let offset = read_i32(data, elem_pos)?;
            let absolute_pos = (elem_pos as i32 + offset) as usize;
            offsets.push(absolute_pos);
        }

        Ok(offsets)
    }

    /// Parse buffers from the model
    fn parse_buffers(
        &self,
        data: &[u8],
        buffer_offsets: &[usize],
    ) -> Result<Vec<Vec<u8>>, ParseError> {
        let mut buffers = Vec::with_capacity(buffer_offsets.len());

        for &buffer_pos in buffer_offsets {
            // Buffer table has one field: data (vector of bytes)
            let vtable_offset = read_i32(data, buffer_pos)?;
            let vtable_pos = (buffer_pos as i32 - vtable_offset) as usize;

            let vtable_size = read_u16(data, vtable_pos)?;
            let num_fields = (vtable_size as usize - 4) / 2;

            if num_fields > 0 {
                let data_field_offset = read_u16(data, vtable_pos + 4)?;
                if data_field_offset != 0 {
                    let data_offset_pos = buffer_pos + data_field_offset as usize;
                    let data_offset = read_i32(data, data_offset_pos)?;
                    let data_pos = (data_offset_pos as i32 + data_offset) as usize;

                    // Read byte vector
                    let byte_count = read_u32(data, data_pos)? as usize;
                    let bytes_start = data_pos + 4;
                    let bytes_end = bytes_start + byte_count;

                    if bytes_end <= data.len() {
                        buffers.push(data[bytes_start..bytes_end].to_vec());
                    } else {
                        buffers.push(Vec::new());
                    }
                } else {
                    buffers.push(Vec::new());
                }
            } else {
                buffers.push(Vec::new());
            }
        }

        Ok(buffers)
    }

    /// Parse tensors from a subgraph
    fn parse_subgraph_tensors(
        &self,
        data: &[u8],
        subgraph_pos: usize,
        buffers: &[Vec<u8>],
    ) -> Result<Vec<TFLiteTensor>, ParseError> {
        // Subgraph table fields:
        // 0: tensors (vector)
        // 1: inputs (vector of int32)
        // 2: outputs (vector of int32)
        // 3: operators (vector)
        // 4: name (string)

        let vtable_offset = read_i32(data, subgraph_pos)?;
        let vtable_pos = (subgraph_pos as i32 - vtable_offset) as usize;

        let vtable_size = read_u16(data, vtable_pos)?;
        let num_fields = (vtable_size as usize - 4) / 2;

        if num_fields == 0 {
            return Ok(Vec::new());
        }

        // Read tensors field offset
        let tensors_field_offset = read_u16(data, vtable_pos + 4)?;
        if tensors_field_offset == 0 {
            return Ok(Vec::new());
        }

        let tensors_offset_pos = subgraph_pos + tensors_field_offset as usize;
        let tensors_offset = read_i32(data, tensors_offset_pos)?;
        let tensors_vec_pos = (tensors_offset_pos as i32 + tensors_offset) as usize;

        let tensor_offsets = self.parse_offset_vector(data, tensors_vec_pos)?;

        let mut tensors = Vec::with_capacity(tensor_offsets.len());
        for (idx, &tensor_pos) in tensor_offsets.iter().enumerate() {
            match self.parse_tensor(data, tensor_pos, buffers, idx) {
                Ok(tensor) => tensors.push(tensor),
                Err(_) => continue, // Skip malformed tensors
            }
        }

        Ok(tensors)
    }

    /// Parse a single tensor
    fn parse_tensor(
        &self,
        data: &[u8],
        tensor_pos: usize,
        buffers: &[Vec<u8>],
        tensor_idx: usize,
    ) -> Result<TFLiteTensor, ParseError> {
        // Tensor table fields:
        // 0: shape (vector of int32)
        // 1: type (TensorType enum, u8)
        // 2: buffer (uint32)
        // 3: name (string)
        // 4: quantization (QuantizationParameters)
        // ... more fields

        let vtable_offset = read_i32(data, tensor_pos)?;
        let vtable_pos = (tensor_pos as i32 - vtable_offset) as usize;

        let vtable_size = read_u16(data, vtable_pos)?;
        let num_fields = (vtable_size as usize - 4) / 2;

        let mut field_offsets = Vec::with_capacity(num_fields);
        for i in 0..num_fields {
            field_offsets.push(read_u16(data, vtable_pos + 4 + i * 2)?);
        }

        // Parse shape (field 0)
        let shape = if !field_offsets.is_empty() && field_offsets[0] != 0 {
            let shape_offset_pos = tensor_pos + field_offsets[0] as usize;
            let shape_offset = read_i32(data, shape_offset_pos)?;
            let shape_pos = (shape_offset_pos as i32 + shape_offset) as usize;
            self.parse_i32_vector(data, shape_pos)?
        } else {
            Vec::new()
        };

        // Parse type (field 1)
        let dtype = if field_offsets.len() > 1 && field_offsets[1] != 0 {
            let type_pos = tensor_pos + field_offsets[1] as usize;
            let type_val = data.get(type_pos).copied().unwrap_or(0);
            TFLiteDataType::from_u8(type_val).unwrap_or(TFLiteDataType::Float32)
        } else {
            TFLiteDataType::Float32
        };

        // Parse buffer index (field 2)
        let buffer_index = if field_offsets.len() > 2 && field_offsets[2] != 0 {
            read_u32(data, tensor_pos + field_offsets[2] as usize)?
        } else {
            0
        };

        // Parse name (field 3)
        let name = if field_offsets.len() > 3 && field_offsets[3] != 0 {
            let name_offset_pos = tensor_pos + field_offsets[3] as usize;
            let name_offset = read_i32(data, name_offset_pos)?;
            let name_pos = (name_offset_pos as i32 + name_offset) as usize;
            self.parse_string(data, name_pos)?
        } else {
            format!("tensor_{}", tensor_idx)
        };

        // Parse quantization (field 4) - simplified
        let quantization = if field_offsets.len() > 4 && field_offsets[4] != 0 {
            // TODO: Parse full quantization parameters
            None
        } else {
            None
        };

        // Get buffer data
        let tensor_data = if (buffer_index as usize) < buffers.len() {
            buffers[buffer_index as usize].clone()
        } else {
            Vec::new()
        };

        Ok(TFLiteTensor {
            name,
            shape,
            dtype,
            buffer_index,
            data: tensor_data,
            quantization,
        })
    }

    /// Parse a string from FlatBuffer
    fn parse_string(&self, data: &[u8], pos: usize) -> Result<String, ParseError> {
        let len = read_u32(data, pos)? as usize;
        let start = pos + 4;
        let end = start + len;

        if end > data.len() {
            return Err(ParseError::Malformed("String out of bounds".into()));
        }

        String::from_utf8(data[start..end].to_vec())
            .map_err(|e| ParseError::Malformed(format!("Invalid UTF-8: {}", e)))
    }

    /// Parse a vector of i32
    fn parse_i32_vector(&self, data: &[u8], pos: usize) -> Result<Vec<i32>, ParseError> {
        let count = read_u32(data, pos)? as usize;
        let mut values = Vec::with_capacity(count);

        for i in 0..count {
            values.push(read_i32(data, pos + 4 + i * 4)?);
        }

        Ok(values)
    }
}

impl Default for TFLiteParser {
    fn default() -> Self {
        Self::new()
    }
}

// Helper functions for reading little-endian values
fn read_u16(data: &[u8], pos: usize) -> Result<u16, ParseError> {
    if pos + 2 > data.len() {
        return Err(ParseError::Malformed("Read out of bounds".into()));
    }
    Ok(u16::from_le_bytes([data[pos], data[pos + 1]]))
}

fn read_i32(data: &[u8], pos: usize) -> Result<i32, ParseError> {
    if pos + 4 > data.len() {
        return Err(ParseError::Malformed("Read out of bounds".into()));
    }
    Ok(i32::from_le_bytes([
        data[pos],
        data[pos + 1],
        data[pos + 2],
        data[pos + 3],
    ]))
}

fn read_u32(data: &[u8], pos: usize) -> Result<u32, ParseError> {
    if pos + 4 > data.len() {
        return Err(ParseError::Malformed("Read out of bounds".into()));
    }
    Ok(u32::from_le_bytes([
        data[pos],
        data[pos + 1],
        data[pos + 2],
        data[pos + 3],
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_creation() {
        let _parser = TFLiteParser::new();
        // Just verify it can be created
        assert!(true);
    }

    #[test]
    fn test_parse_empty_fails() {
        let parser = TFLiteParser::new();
        let result = parser.parse(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_too_small_fails() {
        let parser = TFLiteParser::new();
        let result = parser.parse(&[0, 0, 0, 0]);
        assert!(result.is_err());
    }
}
