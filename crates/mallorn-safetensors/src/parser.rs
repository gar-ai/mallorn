//! SafeTensors parser
//!
//! Parses HuggingFace SafeTensors format files.

use mallorn_core::error::SerializeError;
use mallorn_core::{DataType, ModelMetadata, ParseError, ParsedModel, Tensor, TensorInfo};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// SafeTensors tensor metadata from JSON header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeTensorMeta {
    /// Data type string (e.g., "F32", "F16", "BF16", "I8", etc.)
    pub dtype: String,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Data offsets [start, end] relative to data section
    pub data_offsets: [usize; 2],
}

/// Parsed SafeTensors model
#[derive(Debug, Clone)]
pub struct SafeTensorsModel {
    /// Tensor metadata from header
    pub tensors: HashMap<String, SafeTensorMeta>,
    /// Optional metadata (__metadata__ key)
    pub metadata: HashMap<String, String>,
    /// Raw tensor data section
    pub data: Vec<u8>,
    /// Header size (for serialization)
    pub header_size: usize,
}

impl SafeTensorsModel {
    /// Convert to generic ParsedModel
    pub fn into_parsed_model(self) -> ParsedModel {
        let model_metadata = ModelMetadata {
            custom: self.metadata.clone(),
            ..Default::default()
        };

        let tensors: Vec<Tensor> = self
            .tensors
            .iter()
            .map(|(name, meta)| {
                let start = meta.data_offsets[0];
                let end = meta.data_offsets[1];
                let data = self.data[start..end].to_vec();
                let dtype = parse_dtype(&meta.dtype);

                Tensor {
                    name: name.clone(),
                    shape: meta.shape.clone(),
                    dtype,
                    data,
                    quantization: None,
                }
            })
            .collect();

        ParsedModel {
            format: "safetensors".to_string(),
            metadata: model_metadata,
            tensors,
            graph: None,
        }
    }
}

/// Parse dtype string to DataType enum
fn parse_dtype(dtype: &str) -> DataType {
    match dtype.to_uppercase().as_str() {
        "F32" | "FLOAT32" => DataType::Float32,
        "F16" | "FLOAT16" => DataType::Float16,
        "BF16" | "BFLOAT16" => DataType::BFloat16,
        "F64" | "FLOAT64" => DataType::Float64,
        "I8" | "INT8" => DataType::Int8,
        "U8" | "UINT8" => DataType::UInt8,
        "I16" | "INT16" => DataType::Int16,
        "U16" | "UINT16" => DataType::UInt16,
        "I32" | "INT32" => DataType::Int32,
        "U32" | "UINT32" => DataType::UInt32,
        "I64" | "INT64" => DataType::Int64,
        "U64" | "UINT64" => DataType::UInt64,
        _ => DataType::UInt8, // Fallback
    }
}

/// Convert DataType back to SafeTensors dtype string
fn dtype_to_string(dtype: DataType) -> &'static str {
    match dtype {
        DataType::Float32 => "F32",
        DataType::Float16 => "F16",
        DataType::BFloat16 => "BF16",
        DataType::Float64 => "F64",
        DataType::Int8 => "I8",
        DataType::UInt8 => "U8",
        DataType::Int16 => "I16",
        DataType::UInt16 => "U16",
        DataType::Int32 => "I32",
        DataType::UInt32 => "U32",
        DataType::Int64 => "I64",
        DataType::UInt64 => "U64",
        // Quantized types not supported in SafeTensors
        _ => "U8",
    }
}

/// SafeTensors parser
#[derive(Debug, Clone, Default)]
pub struct SafeTensorsParser;

impl SafeTensorsParser {
    pub fn new() -> Self {
        Self
    }

    /// Parse a SafeTensors file
    pub fn parse(&self, data: &[u8]) -> Result<SafeTensorsModel, ParseError> {
        if data.len() < 8 {
            return Err(ParseError::Malformed("File too small".to_string()));
        }

        // Read header size (8 bytes, little-endian u64)
        let header_size = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;

        if data.len() < 8 + header_size {
            return Err(ParseError::Malformed(format!(
                "File too small for header: expected {}, got {}",
                8 + header_size,
                data.len()
            )));
        }

        // Parse JSON header
        let header_bytes = &data[8..8 + header_size];
        let header_str = std::str::from_utf8(header_bytes)
            .map_err(|e| ParseError::Malformed(format!("Invalid UTF-8 in header: {}", e)))?;

        let header: HashMap<String, serde_json::Value> = serde_json::from_str(header_str)
            .map_err(|e| ParseError::Malformed(format!("Invalid JSON header: {}", e)))?;

        // Extract tensors and metadata
        let mut tensors = HashMap::new();
        let mut metadata = HashMap::new();

        for (key, value) in header {
            if key == "__metadata__" {
                // Parse metadata
                if let serde_json::Value::Object(meta) = value {
                    for (k, v) in meta {
                        if let serde_json::Value::String(s) = v {
                            metadata.insert(k, s);
                        }
                    }
                }
            } else {
                // Parse tensor metadata
                let tensor_meta: SafeTensorMeta = serde_json::from_value(value).map_err(|e| {
                    ParseError::Malformed(format!("Invalid tensor metadata for {}: {}", key, e))
                })?;
                tensors.insert(key, tensor_meta);
            }
        }

        // Extract data section
        let data_start = 8 + header_size;
        let tensor_data = data[data_start..].to_vec();

        Ok(SafeTensorsModel {
            tensors,
            metadata,
            data: tensor_data,
            header_size,
        })
    }

    /// Extract tensor info without loading full data
    pub fn extract_tensor_info(&self, data: &[u8]) -> Result<Vec<TensorInfo>, ParseError> {
        if data.len() < 8 {
            return Err(ParseError::Malformed("File too small".to_string()));
        }

        let header_size = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;

        if data.len() < 8 + header_size {
            return Err(ParseError::Malformed(
                "File too small for header".to_string(),
            ));
        }

        let header_bytes = &data[8..8 + header_size];
        let header_str = std::str::from_utf8(header_bytes)
            .map_err(|e| ParseError::Malformed(format!("Invalid UTF-8: {}", e)))?;

        let header: HashMap<String, serde_json::Value> = serde_json::from_str(header_str)
            .map_err(|e| ParseError::Malformed(format!("Invalid JSON: {}", e)))?;

        let data_offset = 8 + header_size;
        let mut infos = Vec::new();

        for (key, value) in header {
            if key == "__metadata__" {
                continue;
            }

            let tensor_meta: SafeTensorMeta = serde_json::from_value(value)
                .map_err(|e| ParseError::Malformed(format!("Invalid tensor metadata: {}", e)))?;

            let offset = data_offset + tensor_meta.data_offsets[0];
            let size = tensor_meta.data_offsets[1] - tensor_meta.data_offsets[0];

            infos.push(TensorInfo {
                name: key,
                shape: tensor_meta.shape,
                dtype: parse_dtype(&tensor_meta.dtype),
                offset,
                size,
            });
        }

        Ok(infos)
    }
}

/// Serialize a ParsedModel to SafeTensors format
pub fn serialize_safetensors(model: &ParsedModel) -> Result<Vec<u8>, SerializeError> {
    // Build header
    let mut header: HashMap<String, serde_json::Value> = HashMap::new();

    // Add metadata if present
    if !model.metadata.custom.is_empty() {
        let meta_obj: HashMap<String, String> = model.metadata.custom.clone();
        header.insert(
            "__metadata__".to_string(),
            serde_json::to_value(meta_obj).map_err(|e| {
                SerializeError::Failed(format!("Failed to serialize metadata: {}", e))
            })?,
        );
    }

    // Calculate tensor offsets and add to header
    let mut current_offset = 0usize;
    let mut tensor_data = Vec::new();

    // Sort tensors by name for deterministic output
    let mut tensors: Vec<_> = model.tensors.iter().collect();
    tensors.sort_by(|a, b| a.name.cmp(&b.name));

    for tensor in &tensors {
        let start = current_offset;
        let end = start + tensor.data.len();

        let meta = SafeTensorMeta {
            dtype: dtype_to_string(tensor.dtype).to_string(),
            shape: tensor.shape.clone(),
            data_offsets: [start, end],
        };

        header.insert(
            tensor.name.clone(),
            serde_json::to_value(&meta).map_err(|e| {
                SerializeError::Failed(format!("Failed to serialize tensor metadata: {}", e))
            })?,
        );

        tensor_data.extend_from_slice(&tensor.data);
        current_offset = end;
    }

    // Serialize header to JSON
    let header_json = serde_json::to_string(&header)
        .map_err(|e| SerializeError::Failed(format!("Failed to serialize header: {}", e)))?;
    let header_bytes = header_json.as_bytes();
    let header_size = header_bytes.len() as u64;

    // Build final output
    let mut output = Vec::with_capacity(8 + header_bytes.len() + tensor_data.len());
    output.extend_from_slice(&header_size.to_le_bytes());
    output.extend_from_slice(header_bytes);
    output.extend_from_slice(&tensor_data);

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_dtype() {
        assert_eq!(parse_dtype("F32"), DataType::Float32);
        assert_eq!(parse_dtype("f16"), DataType::Float16);
        assert_eq!(parse_dtype("BF16"), DataType::BFloat16);
        assert_eq!(parse_dtype("I8"), DataType::Int8);
    }

    #[test]
    fn test_roundtrip() {
        // Create a simple model
        let model = ParsedModel {
            format: "safetensors".to_string(),
            metadata: ModelMetadata::default(),
            tensors: vec![Tensor {
                name: "weight".to_string(),
                shape: vec![2, 3],
                dtype: DataType::Float32,
                data: vec![0; 24], // 2*3*4 bytes
                quantization: None,
            }],
            graph: None,
        };

        // Serialize
        let bytes = serialize_safetensors(&model).unwrap();

        // Parse back
        let parser = SafeTensorsParser::new();
        let parsed = parser.parse(&bytes).unwrap();
        let roundtrip = parsed.into_parsed_model();

        assert_eq!(roundtrip.tensors.len(), 1);
        assert_eq!(roundtrip.tensors[0].name, "weight");
        assert_eq!(roundtrip.tensors[0].shape, vec![2, 3]);
        assert_eq!(roundtrip.tensors[0].data.len(), 24);
    }
}
