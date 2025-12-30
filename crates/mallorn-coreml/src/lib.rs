// Allow dead code in this partially-implemented format crate
#![allow(dead_code)]

//! Apple CoreML format support for Mallorn
//!
//! This crate provides parsing, diffing, patching, and serialization
//! for Apple CoreML models (.mlpackage and .mlmodelc formats).
//!
//! ## CoreML Format
//!
//! CoreML models come in two primary formats:
//! - `.mlpackage` - Directory containing model spec and weights
//! - `.mlmodelc` - Compiled model bundle
//!
//! The `.mlpackage` structure:
//! ```text
//! model.mlpackage/
//! ├── Data/
//! │   └── com.apple.CoreML/
//! │       └── weights/
//! │           └── weight.bin
//! └── Manifest.json
//! ```
//!
//! ## Patch Format (.cmlp)
//!
//! The patch format stores delta updates between model versions:
//! ```text
//! [CMLP magic: 4 bytes]
//! [Version: u32 LE]
//! [Source hash: 32 bytes]
//! [Target hash: 32 bytes]
//! [Compression method: u8 + optional level]
//! [Metadata length: u32 LE]
//! [Metadata: JSON]
//! [Num operations: u32 LE]
//! [Operations: variable]
//! [CRC32 checksum: u32 LE]
//! ```

mod differ;
mod format;
mod parser;
mod patcher;

pub use differ::CoreMLDiffer;
pub use format::{
    deserialize_patch, extension, is_cmlp, serialize_patch, CMLP_MAGIC, CMLP_VERSION,
};
pub use parser::{CoreMLModel, CoreMLParser, CoreMLTensor};
pub use patcher::CoreMLPatcher;

/// File extensions for CoreML format
pub const COREML_EXTENSIONS: &[&str] = &["mlpackage", "mlmodelc"];

/// Patch file extension for CoreML
pub const COREML_PATCH_EXTENSION: &str = "cmlp";
