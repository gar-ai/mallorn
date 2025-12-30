//! ONNX protobuf parser
//!
//! Parses ONNX model files (.onnx) to extract tensor data (initializers)
//! for diff/patch operations.

use crate::onnx_proto;
use mallorn_core::{DataType, ParseError};
use prost::Message;

/// Parsed ONNX model representation
#[derive(Debug, Clone)]
pub struct ONNXModel {
    /// ONNX IR version
    pub ir_version: i64,
    /// ONNX opset version
    pub opset_version: i64,
    /// Model producer name
    pub producer_name: Option<String>,
    /// Model producer version
    pub producer_version: Option<String>,
    /// Model domain
    pub domain: Option<String>,
    /// Model description
    pub doc_string: Option<String>,
    /// All tensors (initializers) in the model
    pub tensors: Vec<ONNXTensor>,
    /// Graph input names
    pub inputs: Vec<String>,
    /// Graph output names
    pub outputs: Vec<String>,
    /// Raw model bytes (for hashing)
    pub raw_data: Vec<u8>,
    /// Original protobuf model (for reconstruction)
    pub(crate) proto: onnx_proto::ModelProto,
}

/// A tensor (initializer) in an ONNX model
#[derive(Debug, Clone)]
pub struct ONNXTensor {
    /// Tensor name
    pub name: String,
    /// Shape dimensions
    pub dims: Vec<i64>,
    /// Data type
    pub data_type: ONNXDataType,
    /// Tensor data (raw bytes)
    pub data: Vec<u8>,
    /// Index in the initializer list (for reconstruction)
    pub(crate) initializer_index: usize,
}

/// ONNX tensor element types (from TensorProto.DataType)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ONNXDataType {
    Undefined = 0,
    Float = 1,
    UInt8 = 2,
    Int8 = 3,
    UInt16 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    String = 8,
    Bool = 9,
    Float16 = 10,
    Double = 11,
    UInt32 = 12,
    UInt64 = 13,
    Complex64 = 14,
    Complex128 = 15,
    BFloat16 = 16,
    Float8E4M3FN = 17,
    Float8E4M3FNUZ = 18,
    Float8E5M2 = 19,
    Float8E5M2FNUZ = 20,
    UInt4 = 21,
    Int4 = 22,
}

impl ONNXDataType {
    /// Convert from protobuf i32 value
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::Undefined),
            1 => Some(Self::Float),
            2 => Some(Self::UInt8),
            3 => Some(Self::Int8),
            4 => Some(Self::UInt16),
            5 => Some(Self::Int16),
            6 => Some(Self::Int32),
            7 => Some(Self::Int64),
            8 => Some(Self::String),
            9 => Some(Self::Bool),
            10 => Some(Self::Float16),
            11 => Some(Self::Double),
            12 => Some(Self::UInt32),
            13 => Some(Self::UInt64),
            14 => Some(Self::Complex64),
            15 => Some(Self::Complex128),
            16 => Some(Self::BFloat16),
            17 => Some(Self::Float8E4M3FN),
            18 => Some(Self::Float8E4M3FNUZ),
            19 => Some(Self::Float8E5M2),
            20 => Some(Self::Float8E5M2FNUZ),
            21 => Some(Self::UInt4),
            22 => Some(Self::Int4),
            _ => None,
        }
    }

    /// Size of one element in bytes (None for variable-size or packed types)
    pub fn element_size(&self) -> Option<usize> {
        match self {
            Self::Float | Self::Int32 | Self::UInt32 => Some(4),
            Self::Double | Self::Int64 | Self::UInt64 | Self::Complex64 => Some(8),
            Self::Float16 | Self::BFloat16 | Self::Int16 | Self::UInt16 => Some(2),
            Self::Int8 | Self::UInt8 | Self::Bool => Some(1),
            Self::Complex128 => Some(16),
            Self::Float8E4M3FN | Self::Float8E4M3FNUZ | Self::Float8E5M2 | Self::Float8E5M2FNUZ => {
                Some(1)
            }
            Self::UInt4 | Self::Int4 => None, // Packed nibbles
            Self::String | Self::Undefined => None,
        }
    }
}

impl From<ONNXDataType> for DataType {
    fn from(dtype: ONNXDataType) -> Self {
        match dtype {
            ONNXDataType::Float => DataType::Float32,
            ONNXDataType::Float16 => DataType::Float16,
            ONNXDataType::BFloat16 => DataType::BFloat16,
            ONNXDataType::Double => DataType::Float64,
            ONNXDataType::Int8 => DataType::Int8,
            ONNXDataType::UInt8 => DataType::UInt8,
            ONNXDataType::Int16 => DataType::Int16,
            ONNXDataType::UInt16 => DataType::UInt16,
            ONNXDataType::Int32 => DataType::Int32,
            ONNXDataType::UInt32 => DataType::UInt32,
            ONNXDataType::Int64 => DataType::Int64,
            ONNXDataType::UInt64 => DataType::UInt64,
            _ => DataType::Float32, // Default fallback
        }
    }
}

/// ONNX model parser
pub struct ONNXParser;

impl ONNXParser {
    /// Create a new ONNX parser
    pub fn new() -> Self {
        Self
    }

    /// Parse an ONNX model from bytes
    pub fn parse(&self, data: &[u8]) -> Result<ONNXModel, ParseError> {
        // Decode protobuf
        let model_proto: onnx_proto::ModelProto = Message::decode(data)
            .map_err(|e| ParseError::Malformed(format!("Failed to decode protobuf: {}", e)))?;

        let graph = model_proto
            .graph
            .as_ref()
            .ok_or_else(|| ParseError::Malformed("Model has no graph".into()))?;

        // Extract initializers (weights) as tensors
        let mut tensors = Vec::new();
        for (idx, initializer) in graph.initializer.iter().enumerate() {
            let tensor = self.parse_tensor_proto(initializer, idx)?;
            tensors.push(tensor);
        }

        // Extract input/output names
        let inputs: Vec<String> = graph.input.iter().map(|vi| vi.name.clone()).collect();
        let outputs: Vec<String> = graph.output.iter().map(|vi| vi.name.clone()).collect();

        // Get opset version
        let opset_version = model_proto
            .opset_import
            .first()
            .map(|o| o.version)
            .unwrap_or(0);

        Ok(ONNXModel {
            ir_version: model_proto.ir_version,
            opset_version,
            producer_name: if model_proto.producer_name.is_empty() {
                None
            } else {
                Some(model_proto.producer_name.clone())
            },
            producer_version: if model_proto.producer_version.is_empty() {
                None
            } else {
                Some(model_proto.producer_version.clone())
            },
            domain: if model_proto.domain.is_empty() {
                None
            } else {
                Some(model_proto.domain.clone())
            },
            doc_string: if model_proto.doc_string.is_empty() {
                None
            } else {
                Some(model_proto.doc_string.clone())
            },
            tensors,
            inputs,
            outputs,
            raw_data: data.to_vec(),
            proto: model_proto,
        })
    }

    /// Parse a TensorProto into ONNXTensor
    fn parse_tensor_proto(
        &self,
        tensor: &onnx_proto::TensorProto,
        index: usize,
    ) -> Result<ONNXTensor, ParseError> {
        let data_type = ONNXDataType::from_i32(tensor.data_type).ok_or_else(|| {
            ParseError::Malformed(format!("Unknown data type: {}", tensor.data_type))
        })?;

        // Extract raw data
        // ONNX stores data in either:
        // 1. raw_data field (preferred, most efficient)
        // 2. Type-specific fields (float_data, int32_data, etc.)
        let data = if !tensor.raw_data.is_empty() {
            tensor.raw_data.clone()
        } else {
            self.extract_typed_data(tensor, data_type)?
        };

        Ok(ONNXTensor {
            name: tensor.name.clone(),
            dims: tensor.dims.clone(),
            data_type,
            data,
            initializer_index: index,
        })
    }

    /// Extract data from type-specific fields
    fn extract_typed_data(
        &self,
        tensor: &onnx_proto::TensorProto,
        dtype: ONNXDataType,
    ) -> Result<Vec<u8>, ParseError> {
        match dtype {
            ONNXDataType::Float => {
                let mut bytes = Vec::with_capacity(tensor.float_data.len() * 4);
                for &f in &tensor.float_data {
                    bytes.extend_from_slice(&f.to_le_bytes());
                }
                Ok(bytes)
            }
            ONNXDataType::Int32 => {
                let mut bytes = Vec::with_capacity(tensor.int32_data.len() * 4);
                for &i in &tensor.int32_data {
                    bytes.extend_from_slice(&i.to_le_bytes());
                }
                Ok(bytes)
            }
            ONNXDataType::Int64 => {
                let mut bytes = Vec::with_capacity(tensor.int64_data.len() * 8);
                for &i in &tensor.int64_data {
                    bytes.extend_from_slice(&i.to_le_bytes());
                }
                Ok(bytes)
            }
            ONNXDataType::Double => {
                let mut bytes = Vec::with_capacity(tensor.double_data.len() * 8);
                for &d in &tensor.double_data {
                    bytes.extend_from_slice(&d.to_le_bytes());
                }
                Ok(bytes)
            }
            ONNXDataType::UInt64 => {
                let mut bytes = Vec::with_capacity(tensor.uint64_data.len() * 8);
                for &u in &tensor.uint64_data {
                    bytes.extend_from_slice(&u.to_le_bytes());
                }
                Ok(bytes)
            }
            _ => Ok(Vec::new()), // Empty for unsupported types
        }
    }

    /// Reconstruct model bytes with updated tensors
    pub fn reconstruct(
        &self,
        model: &ONNXModel,
        new_tensors: &[ONNXTensor],
    ) -> Result<Vec<u8>, ParseError> {
        let mut proto = model.proto.clone();

        // Update graph initializers with new tensor data
        if let Some(ref mut graph) = proto.graph {
            for new_tensor in new_tensors {
                if new_tensor.initializer_index < graph.initializer.len() {
                    let init = &mut graph.initializer[new_tensor.initializer_index];
                    // Update raw_data and clear typed fields
                    init.raw_data = new_tensor.data.clone();
                    init.float_data.clear();
                    init.int32_data.clear();
                    init.int64_data.clear();
                    init.double_data.clear();
                    init.uint64_data.clear();
                    init.string_data.clear();
                }
            }
        }

        // Encode back to protobuf
        let mut buf = Vec::new();
        proto
            .encode(&mut buf)
            .map_err(|e| ParseError::Malformed(format!("Failed to encode protobuf: {}", e)))?;

        Ok(buf)
    }
}

impl Default for ONNXParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_creation() {
        let _parser = ONNXParser::new();
    }

    #[test]
    fn test_data_type_from_i32() {
        assert_eq!(ONNXDataType::from_i32(1), Some(ONNXDataType::Float));
        assert_eq!(ONNXDataType::from_i32(6), Some(ONNXDataType::Int32));
        assert_eq!(ONNXDataType::from_i32(100), None);
    }

    #[test]
    fn test_element_size() {
        assert_eq!(ONNXDataType::Float.element_size(), Some(4));
        assert_eq!(ONNXDataType::Double.element_size(), Some(8));
        assert_eq!(ONNXDataType::Int8.element_size(), Some(1));
        assert_eq!(ONNXDataType::Float16.element_size(), Some(2));
        assert_eq!(ONNXDataType::String.element_size(), None);
    }

    #[test]
    fn test_parse_empty_fails() {
        let parser = ONNXParser::new();
        assert!(parser.parse(&[]).is_err());
    }

    #[test]
    fn test_parse_invalid_fails() {
        let parser = ONNXParser::new();
        assert!(parser.parse(&[0xFF, 0xFF, 0xFF, 0xFF]).is_err());
    }
}
