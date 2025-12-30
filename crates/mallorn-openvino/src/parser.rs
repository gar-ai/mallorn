//! OpenVINO IR format parser
//!
//! Parses OpenVINO Intermediate Representation models consisting of:
//! - `.xml` - Model graph structure (layers, connections, metadata)
//! - `.bin` - Binary weights data (tensors)

use mallorn_core::{DataType, ModelMetadata, ParseError, ParsedModel, Tensor, TensorInfo};
use quick_xml::events::Event;
use quick_xml::Reader;
use std::collections::HashMap;

/// Parsed OpenVINO model
#[derive(Debug, Clone)]
pub struct OpenVINOModel {
    /// Model name
    pub name: String,
    /// OpenVINO IR version
    pub ir_version: u32,
    /// Model metadata (from XML attributes)
    pub metadata: HashMap<String, String>,
    /// Layers/operations in the model
    pub layers: Vec<OpenVINOLayer>,
    /// Tensors (weights) in the model
    pub tensors: Vec<OpenVINOTensor>,
    /// Raw XML data
    pub xml_data: Vec<u8>,
    /// Raw binary weights data
    pub bin_data: Vec<u8>,
}

impl OpenVINOModel {
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
            format: "openvino".to_string(),
            tensors,
            metadata: ModelMetadata {
                name: Some(self.name.clone()),
                version: self.metadata.get("version").cloned(),
                custom: self.metadata.clone(),
            },
            graph: None,
        }
    }
}

/// A layer in an OpenVINO model
#[derive(Debug, Clone)]
pub struct OpenVINOLayer {
    /// Layer ID
    pub id: u32,
    /// Layer name
    pub name: String,
    /// Layer type (e.g., "Convolution", "MatMul", "Add")
    pub layer_type: String,
    /// Layer attributes
    pub attributes: HashMap<String, String>,
    /// Input port IDs
    pub inputs: Vec<u32>,
    /// Output port IDs
    pub outputs: Vec<u32>,
}

/// A tensor (weight) in an OpenVINO model
#[derive(Debug, Clone)]
pub struct OpenVINOTensor {
    /// Tensor name (typically layer_id/port_id)
    pub name: String,
    /// Tensor shape
    pub shape: Vec<i64>,
    /// Data type
    pub dtype: DataType,
    /// Offset in .bin file
    pub offset: usize,
    /// Size in bytes
    pub size: usize,
    /// Tensor data
    pub data: Vec<u8>,
}

/// OpenVINO data type from XML element_type attribute
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenVINODType {
    F32,
    F16,
    BF16,
    F64,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    Bool,
}

impl OpenVINODType {
    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "f32" | "fp32" | "float" => Some(Self::F32),
            "f16" | "fp16" => Some(Self::F16),
            "bf16" => Some(Self::BF16),
            "f64" | "fp64" | "double" => Some(Self::F64),
            "i8" => Some(Self::I8),
            "u8" => Some(Self::U8),
            "i16" => Some(Self::I16),
            "u16" => Some(Self::U16),
            "i32" | "int" => Some(Self::I32),
            "u32" => Some(Self::U32),
            "i64" => Some(Self::I64),
            "u64" => Some(Self::U64),
            "bool" | "boolean" => Some(Self::Bool),
            _ => None,
        }
    }

    /// Get byte size of this type
    pub fn byte_size(&self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => 2,
            Self::F64 | Self::I64 | Self::U64 => 8,
            Self::I8 | Self::U8 | Self::Bool => 1,
        }
    }
}

impl From<OpenVINODType> for DataType {
    fn from(t: OpenVINODType) -> Self {
        match t {
            OpenVINODType::F32 => DataType::Float32,
            OpenVINODType::F16 => DataType::Float16,
            OpenVINODType::BF16 => DataType::BFloat16,
            OpenVINODType::F64 => DataType::Float64,
            OpenVINODType::I8 => DataType::Int8,
            OpenVINODType::U8 => DataType::UInt8,
            OpenVINODType::I16 => DataType::Int16,
            OpenVINODType::U16 => DataType::UInt16,
            OpenVINODType::I32 => DataType::Int32,
            OpenVINODType::U32 => DataType::UInt32,
            OpenVINODType::I64 => DataType::Int64,
            OpenVINODType::U64 => DataType::UInt64,
            OpenVINODType::Bool => DataType::UInt8,
        }
    }
}

/// OpenVINO model parser
#[derive(Debug, Clone, Default)]
pub struct OpenVINOParser;

impl OpenVINOParser {
    /// Create a new parser
    pub fn new() -> Self {
        Self
    }

    /// Parse OpenVINO model from XML bytes only (weights will be empty)
    pub fn parse(&self, xml_data: &[u8]) -> Result<OpenVINOModel, ParseError> {
        self.parse_with_weights(xml_data, &[])
    }

    /// Parse OpenVINO model from XML and binary weights
    pub fn parse_with_weights(
        &self,
        xml_data: &[u8],
        bin_data: &[u8],
    ) -> Result<OpenVINOModel, ParseError> {
        if xml_data.is_empty() {
            return Err(ParseError::Malformed("Empty XML data".into()));
        }

        let mut reader = Reader::from_reader(xml_data);
        reader.trim_text(true);

        let mut model_name = String::new();
        let mut ir_version = 0u32;
        let mut metadata = HashMap::new();
        let mut layers = Vec::new();
        let mut const_layers: Vec<ConstLayerInfo> = Vec::new();

        let mut buf = Vec::new();
        let mut current_layer: Option<LayerBuilder> = None;
        let mut in_layers = false;

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    let tag_name = String::from_utf8_lossy(e.name().as_ref()).to_string();

                    match tag_name.as_str() {
                        "net" => {
                            // Parse model-level attributes
                            for attr in e.attributes().flatten() {
                                let key = String::from_utf8_lossy(attr.key.as_ref()).to_string();
                                let value = String::from_utf8_lossy(&attr.value).to_string();
                                match key.as_str() {
                                    "name" => model_name = value,
                                    "version" => {
                                        ir_version = value.parse().unwrap_or(0);
                                    }
                                    _ => {
                                        metadata.insert(key, value);
                                    }
                                }
                            }
                        }
                        "layers" => {
                            in_layers = true;
                        }
                        "layer" if in_layers => {
                            let mut builder = LayerBuilder::default();
                            for attr in e.attributes().flatten() {
                                let key = String::from_utf8_lossy(attr.key.as_ref()).to_string();
                                let value = String::from_utf8_lossy(&attr.value).to_string();
                                match key.as_str() {
                                    "id" => builder.id = value.parse().unwrap_or(0),
                                    "name" => builder.name = value,
                                    "type" => builder.layer_type = value,
                                    _ => {
                                        builder.attributes.insert(key, value);
                                    }
                                }
                            }
                            current_layer = Some(builder);
                        }
                        "data" if current_layer.is_some() => {
                            // Const layer data reference
                            if let Some(ref mut layer) = current_layer {
                                for attr in e.attributes().flatten() {
                                    let key =
                                        String::from_utf8_lossy(attr.key.as_ref()).to_string();
                                    let value = String::from_utf8_lossy(&attr.value).to_string();
                                    match key.as_str() {
                                        "offset" => {
                                            layer.data_offset = value.parse().ok();
                                        }
                                        "size" => {
                                            layer.data_size = value.parse().ok();
                                        }
                                        "element_type" => {
                                            layer.element_type = Some(value);
                                        }
                                        "shape" => {
                                            layer.shape = Some(value);
                                        }
                                        _ => {
                                            layer.attributes.insert(key, value);
                                        }
                                    }
                                }
                            }
                        }
                        "input" if current_layer.is_some() => {
                            // Input ports
                        }
                        "output" if current_layer.is_some() => {
                            // Output ports
                        }
                        "port" if current_layer.is_some() => {
                            if let Some(ref mut layer) = current_layer {
                                for attr in e.attributes().flatten() {
                                    let key =
                                        String::from_utf8_lossy(attr.key.as_ref()).to_string();
                                    if key == "id" {
                                        let port_id: u32 = String::from_utf8_lossy(&attr.value)
                                            .parse()
                                            .unwrap_or(0);
                                        layer.port_ids.push(port_id);
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Ok(Event::Empty(ref e)) => {
                    let tag_name = String::from_utf8_lossy(e.name().as_ref()).to_string();

                    if tag_name == "layer" && in_layers {
                        // Self-closing layer tag
                        let mut builder = LayerBuilder::default();
                        for attr in e.attributes().flatten() {
                            let key = String::from_utf8_lossy(attr.key.as_ref()).to_string();
                            let value = String::from_utf8_lossy(&attr.value).to_string();
                            match key.as_str() {
                                "id" => builder.id = value.parse().unwrap_or(0),
                                "name" => builder.name = value,
                                "type" => builder.layer_type = value,
                                _ => {
                                    builder.attributes.insert(key, value);
                                }
                            }
                        }
                        layers.push(builder.build());
                    } else if tag_name == "data" && current_layer.is_some() {
                        // Self-closing data tag
                        if let Some(ref mut layer) = current_layer {
                            for attr in e.attributes().flatten() {
                                let key = String::from_utf8_lossy(attr.key.as_ref()).to_string();
                                let value = String::from_utf8_lossy(&attr.value).to_string();
                                match key.as_str() {
                                    "offset" => {
                                        layer.data_offset = value.parse().ok();
                                    }
                                    "size" => {
                                        layer.data_size = value.parse().ok();
                                    }
                                    "element_type" => {
                                        layer.element_type = Some(value);
                                    }
                                    "shape" => {
                                        layer.shape = Some(value);
                                    }
                                    _ => {
                                        layer.attributes.insert(key, value);
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(Event::End(ref e)) => {
                    let tag_name = String::from_utf8_lossy(e.name().as_ref()).to_string();

                    match tag_name.as_str() {
                        "layers" => {
                            in_layers = false;
                        }
                        "layer" => {
                            if let Some(builder) = current_layer.take() {
                                // Check if this is a Const layer with data
                                if builder.layer_type == "Const"
                                    && builder.data_offset.is_some()
                                    && builder.data_size.is_some()
                                {
                                    const_layers.push(ConstLayerInfo {
                                        name: builder.name.clone(),
                                        offset: builder.data_offset.unwrap(),
                                        size: builder.data_size.unwrap(),
                                        element_type: builder.element_type.clone(),
                                        shape: builder.shape.clone(),
                                    });
                                }
                                layers.push(builder.build());
                            }
                        }
                        _ => {}
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => {
                    return Err(ParseError::Malformed(format!("XML parse error: {}", e)));
                }
                _ => {}
            }
            buf.clear();
        }

        // Extract tensors from const layers
        let tensors = self.extract_tensors(&const_layers, bin_data);

        Ok(OpenVINOModel {
            name: model_name,
            ir_version,
            metadata,
            layers,
            tensors,
            xml_data: xml_data.to_vec(),
            bin_data: bin_data.to_vec(),
        })
    }

    /// Extract tensor info from model bytes
    pub fn extract_tensor_info(&self, xml_data: &[u8]) -> Result<Vec<TensorInfo>, ParseError> {
        let model = self.parse(xml_data)?;
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

    /// Extract tensors from const layers and binary data
    fn extract_tensors(
        &self,
        const_layers: &[ConstLayerInfo],
        bin_data: &[u8],
    ) -> Vec<OpenVINOTensor> {
        let mut tensors = Vec::new();

        for layer in const_layers {
            let dtype = layer
                .element_type
                .as_ref()
                .and_then(|s| OpenVINODType::from_str(s))
                .unwrap_or(OpenVINODType::F32);

            let shape: Vec<i64> = layer
                .shape
                .as_ref()
                .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
                .unwrap_or_default();

            let data = if layer.offset + layer.size <= bin_data.len() {
                bin_data[layer.offset..layer.offset + layer.size].to_vec()
            } else {
                Vec::new()
            };

            tensors.push(OpenVINOTensor {
                name: layer.name.clone(),
                shape,
                dtype: dtype.into(),
                offset: layer.offset,
                size: layer.size,
                data,
            });
        }

        tensors
    }
}

/// Builder for layers during parsing
#[derive(Default)]
struct LayerBuilder {
    id: u32,
    name: String,
    layer_type: String,
    attributes: HashMap<String, String>,
    port_ids: Vec<u32>,
    data_offset: Option<usize>,
    data_size: Option<usize>,
    element_type: Option<String>,
    shape: Option<String>,
}

impl LayerBuilder {
    fn build(self) -> OpenVINOLayer {
        OpenVINOLayer {
            id: self.id,
            name: self.name,
            layer_type: self.layer_type,
            attributes: self.attributes,
            inputs: Vec::new(), // Would need edge parsing for full connectivity
            outputs: self.port_ids,
        }
    }
}

/// Info about a Const layer's data
struct ConstLayerInfo {
    name: String,
    offset: usize,
    size: usize,
    element_type: Option<String>,
    shape: Option<String>,
}

/// Serialize an OpenVINO model back to XML and binary
pub fn serialize_openvino(model: &OpenVINOModel) -> Result<(Vec<u8>, Vec<u8>), ParseError> {
    // For now, return the original data - full serialization would require
    // rebuilding the XML structure
    Ok((model.xml_data.clone(), model.bin_data.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_creation() {
        let _parser = OpenVINOParser::new();
    }

    #[test]
    fn test_parse_empty_fails() {
        let parser = OpenVINOParser::new();
        assert!(parser.parse(&[]).is_err());
    }

    #[test]
    fn test_parse_minimal_xml() {
        let parser = OpenVINOParser::new();
        let xml = br#"<?xml version="1.0"?>
<net name="test_model" version="11">
    <layers></layers>
    <edges></edges>
</net>"#;
        let result = parser.parse(xml);
        assert!(result.is_ok());
        let model = result.unwrap();
        assert_eq!(model.name, "test_model");
        assert_eq!(model.ir_version, 11);
    }

    #[test]
    fn test_openvino_dtype_from_str() {
        assert_eq!(OpenVINODType::from_str("f32"), Some(OpenVINODType::F32));
        assert_eq!(OpenVINODType::from_str("FP16"), Some(OpenVINODType::F16));
        assert_eq!(OpenVINODType::from_str("i8"), Some(OpenVINODType::I8));
        assert_eq!(OpenVINODType::from_str("unknown"), None);
    }

    #[test]
    fn test_openvino_dtype_byte_size() {
        assert_eq!(OpenVINODType::F32.byte_size(), 4);
        assert_eq!(OpenVINODType::F16.byte_size(), 2);
        assert_eq!(OpenVINODType::I8.byte_size(), 1);
        assert_eq!(OpenVINODType::F64.byte_size(), 8);
    }

    #[test]
    fn test_datatype_conversion() {
        assert_eq!(DataType::from(OpenVINODType::F32), DataType::Float32);
        assert_eq!(DataType::from(OpenVINODType::F16), DataType::Float16);
        assert_eq!(DataType::from(OpenVINODType::I8), DataType::Int8);
    }

    #[test]
    fn test_parse_with_const_layer() {
        let parser = OpenVINOParser::new();
        let xml = br#"<?xml version="1.0"?>
<net name="test" version="11">
    <layers>
        <layer id="0" name="weights" type="Const">
            <data element_type="f32" offset="0" shape="2,3" size="24"/>
            <output>
                <port id="0"/>
            </output>
        </layer>
    </layers>
    <edges></edges>
</net>"#;

        let bin_data = vec![0u8; 24];
        let result = parser.parse_with_weights(xml, &bin_data);
        assert!(result.is_ok());

        let model = result.unwrap();
        assert_eq!(model.tensors.len(), 1);
        assert_eq!(model.tensors[0].name, "weights");
        assert_eq!(model.tensors[0].shape, vec![2, 3]);
        assert_eq!(model.tensors[0].dtype, DataType::Float32);
    }
}
