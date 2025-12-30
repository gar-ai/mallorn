//! Mallorn GGUF - GGUF format support for llama.cpp models
//!
//! This crate provides parsing, diffing, and patching for GGUF models.
//! It produces and consumes `.ggup` patch files.

pub mod differ;
pub mod format;
pub mod parser;
pub mod patcher;
pub mod quantization;

// Re-export main types
pub use differ::GGUFDiffer;
pub use format::{deserialize_patch, extension, is_ggup, serialize_patch, GGUP_MAGIC, GGUP_VERSION};
pub use parser::{GGMLType, GGUFModel, GGUFParser, GGUFTensor, GGUFValue};
pub use patcher::GGUFPatcher;
pub use quantization::QuantizationInfo;
