//! CoreML format parser
//!
//! Parses Apple CoreML models including:
//! - `.mlpackage` - Directory with manifest and weights
//! - `.mlmodelc` - Compiled model bundle
//!
//! Weight files use a simple binary format with offset tables.

use byteorder::{LittleEndian, ReadBytesExt};
use mallorn_core::{DataType, ModelMetadata, ParseError, ParsedModel, Tensor, TensorInfo};
use std::collections::HashMap;
use std::io::Cursor;

/// Parsed CoreML model
#[derive(Debug, Clone)]
pub struct CoreMLModel {
    /// Model identifier
    pub identifier: String,
    /// Model version
    pub version: String,
    /// Model metadata from manifest
    pub metadata: HashMap<String, String>,
    /// Tensors (weights) in the model
    pub tensors: Vec<CoreMLTensor>,
    /// Raw weight data
    pub weight_data: Vec<u8>,
}

impl CoreMLModel {
    /// Convert to generic ParsedModel
    pub fn into_parsed_model(self) -> ParsedModel {
        let tensors = self
            .tensors
            .iter()
            .map(|t| Tensor {
                name: t.name.clone(),
                shape: t.shape.iter().map(|&x| x as usize).collect(),
                dtype: t.dtype,
                data: t.data.clone(),
                quantization: None,
            })
            .collect();

        ParsedModel {
            format: "coreml".to_string(),
            tensors,
            metadata: ModelMetadata {
                name: Some(self.identifier.clone()),
                version: Some(self.version.clone()),
                custom: self.metadata.clone(),
            },
            graph: None,
        }
    }
}

/// A tensor (weight) in a CoreML model
#[derive(Debug, Clone)]
pub struct CoreMLTensor {
    /// Tensor name/identifier
    pub name: String,
    /// Tensor shape
    pub shape: Vec<i64>,
    /// Data type
    pub dtype: DataType,
    /// Offset in weight file
    pub offset: usize,
    /// Size in bytes
    pub size: usize,
    /// Tensor data
    pub data: Vec<u8>,
}

/// CoreML weight file header
#[derive(Debug, Clone)]
struct WeightHeader {
    /// Number of weight blobs
    num_blobs: u32,
    /// Blob metadata entries
    blobs: Vec<BlobInfo>,
}

/// Individual blob metadata
#[derive(Debug, Clone)]
struct BlobInfo {
    /// Blob offset in file
    offset: u64,
    /// Blob size in bytes
    size: u64,
    /// Blob name/identifier (if available)
    name: Option<String>,
}

/// CoreML data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreMLDType {
    Float32,
    Float16,
    Int32,
    Int16,
    Int8,
    UInt8,
}

impl CoreMLDType {
    /// Get byte size of this type
    pub fn byte_size(&self) -> usize {
        match self {
            Self::Float32 | Self::Int32 => 4,
            Self::Float16 | Self::Int16 => 2,
            Self::Int8 | Self::UInt8 => 1,
        }
    }
}

impl From<CoreMLDType> for DataType {
    fn from(t: CoreMLDType) -> Self {
        match t {
            CoreMLDType::Float32 => DataType::Float32,
            CoreMLDType::Float16 => DataType::Float16,
            CoreMLDType::Int32 => DataType::Int32,
            CoreMLDType::Int16 => DataType::Int16,
            CoreMLDType::Int8 => DataType::Int8,
            CoreMLDType::UInt8 => DataType::UInt8,
        }
    }
}

/// CoreML model parser
#[derive(Debug, Clone, Default)]
pub struct CoreMLParser;

impl CoreMLParser {
    /// Create a new parser
    pub fn new() -> Self {
        Self
    }

    /// Parse CoreML weight data
    ///
    /// This parses the raw weight.bin file format used in .mlpackage
    pub fn parse(&self, data: &[u8]) -> Result<CoreMLModel, ParseError> {
        if data.is_empty() {
            return Err(ParseError::Malformed("Empty weight data".into()));
        }

        // CoreML weight files can have different formats
        // Try to detect and parse appropriately
        if data.len() < 8 {
            return Err(ParseError::Malformed("Weight file too small".into()));
        }

        // Parse as a simple blob format
        // CoreML weights are typically stored as contiguous float arrays
        let tensors = self.parse_weight_blobs(data)?;

        Ok(CoreMLModel {
            identifier: "coreml_model".into(),
            version: "1.0".into(),
            metadata: HashMap::new(),
            tensors,
            weight_data: data.to_vec(),
        })
    }

    /// Parse CoreML model with manifest metadata
    pub fn parse_with_manifest(
        &self,
        weight_data: &[u8],
        manifest_json: &str,
    ) -> Result<CoreMLModel, ParseError> {
        if weight_data.is_empty() {
            return Err(ParseError::Malformed("Empty weight data".into()));
        }

        // Parse manifest for metadata
        let manifest: serde_json::Value = serde_json::from_str(manifest_json)
            .map_err(|e| ParseError::Malformed(format!("Invalid manifest JSON: {}", e)))?;

        let identifier = manifest
            .get("itemIdentifier")
            .and_then(|v| v.as_str())
            .unwrap_or("coreml_model")
            .to_string();

        let version = manifest
            .get("version")
            .and_then(|v| v.as_str())
            .unwrap_or("1.0")
            .to_string();

        let mut metadata = HashMap::new();
        if let Some(obj) = manifest.as_object() {
            for (key, value) in obj {
                if let Some(s) = value.as_str() {
                    metadata.insert(key.clone(), s.to_string());
                }
            }
        }

        let tensors = self.parse_weight_blobs(weight_data)?;

        Ok(CoreMLModel {
            identifier,
            version,
            metadata,
            tensors,
            weight_data: weight_data.to_vec(),
        })
    }

    /// Parse weight blobs from raw data
    fn parse_weight_blobs(&self, data: &[u8]) -> Result<Vec<CoreMLTensor>, ParseError> {
        let mut tensors = Vec::new();

        // CoreML weight.bin format detection
        // Try to parse as milc format (CoreML internal format)
        if data.len() >= 4 {
            let magic = &data[0..4];

            // Check for known CoreML weight formats
            if magic == b"milc" {
                return self.parse_milc_format(data);
            }
        }

        // Fallback: treat as raw weight blob
        // This handles simple weight files that are just concatenated tensors
        if !data.is_empty() {
            tensors.push(CoreMLTensor {
                name: "weights".into(),
                shape: vec![data.len() as i64],
                dtype: DataType::UInt8,
                offset: 0,
                size: data.len(),
                data: data.to_vec(),
            });
        }

        Ok(tensors)
    }

    /// Parse milc (CoreML internal) format
    fn parse_milc_format(&self, data: &[u8]) -> Result<Vec<CoreMLTensor>, ParseError> {
        if data.len() < 16 {
            return Err(ParseError::Malformed("milc file too small".into()));
        }

        let mut cursor = Cursor::new(data);

        // Skip magic
        cursor.set_position(4);

        // Read version
        let _version = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| ParseError::Malformed(e.to_string()))?;

        // Read number of blobs
        let num_blobs = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| ParseError::Malformed(e.to_string()))?;

        // Read header size
        let header_size = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| ParseError::Malformed(e.to_string()))? as usize;

        let mut tensors = Vec::new();

        // Parse blob table
        for i in 0..num_blobs {
            if cursor.position() as usize + 16 > header_size {
                break;
            }

            let offset = cursor
                .read_u64::<LittleEndian>()
                .map_err(|e| ParseError::Malformed(e.to_string()))? as usize;

            let size = cursor
                .read_u64::<LittleEndian>()
                .map_err(|e| ParseError::Malformed(e.to_string()))? as usize;

            // Extract blob data
            let blob_data = if offset + size <= data.len() {
                data[offset..offset + size].to_vec()
            } else {
                Vec::new()
            };

            tensors.push(CoreMLTensor {
                name: format!("blob_{}", i),
                shape: vec![size as i64],
                dtype: DataType::UInt8,
                offset,
                size,
                data: blob_data,
            });
        }

        Ok(tensors)
    }

    /// Extract tensor info without full parsing
    pub fn extract_tensor_info(&self, data: &[u8]) -> Result<Vec<TensorInfo>, ParseError> {
        let model = self.parse(data)?;
        Ok(model
            .tensors
            .iter()
            .map(|t| TensorInfo {
                name: t.name.clone(),
                shape: t.shape.iter().map(|&x| x as usize).collect(),
                dtype: t.dtype,
                offset: t.offset,
                size: t.size,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_creation() {
        let _parser = CoreMLParser::new();
    }

    #[test]
    fn test_parse_empty_fails() {
        let parser = CoreMLParser::new();
        assert!(parser.parse(&[]).is_err());
    }

    #[test]
    fn test_parse_raw_weights() {
        let parser = CoreMLParser::new();
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let result = parser.parse(&data);
        assert!(result.is_ok());

        let model = result.unwrap();
        assert_eq!(model.tensors.len(), 1);
        assert_eq!(model.tensors[0].name, "weights");
        assert_eq!(model.tensors[0].size, 8);
    }

    #[test]
    fn test_parse_with_manifest() {
        let parser = CoreMLParser::new();
        let data = vec![1u8, 2, 3, 4];
        let manifest = r#"{"itemIdentifier": "test_model", "version": "2.0"}"#;

        let result = parser.parse_with_manifest(&data, manifest);
        assert!(result.is_ok());

        let model = result.unwrap();
        assert_eq!(model.identifier, "test_model");
        assert_eq!(model.version, "2.0");
    }

    #[test]
    fn test_coreml_dtype_byte_size() {
        assert_eq!(CoreMLDType::Float32.byte_size(), 4);
        assert_eq!(CoreMLDType::Float16.byte_size(), 2);
        assert_eq!(CoreMLDType::Int8.byte_size(), 1);
    }

    #[test]
    fn test_datatype_conversion() {
        assert_eq!(DataType::from(CoreMLDType::Float32), DataType::Float32);
        assert_eq!(DataType::from(CoreMLDType::Float16), DataType::Float16);
        assert_eq!(DataType::from(CoreMLDType::Int8), DataType::Int8);
    }

    #[test]
    fn test_into_parsed_model() {
        let model = CoreMLModel {
            identifier: "test".into(),
            version: "1.0".into(),
            metadata: HashMap::new(),
            tensors: vec![CoreMLTensor {
                name: "w1".into(),
                shape: vec![4],
                dtype: DataType::Float32,
                offset: 0,
                size: 16,
                data: vec![0; 16],
            }],
            weight_data: vec![0; 16],
        };

        let parsed = model.into_parsed_model();
        assert_eq!(parsed.format, "coreml");
        assert_eq!(parsed.tensors.len(), 1);
    }
}
