// Allow dead code in this partially-implemented format crate
#![allow(dead_code)]

//! Intel OpenVINO IR format support for Mallorn
//!
//! This crate provides parsing, diffing, patching, and serialization
//! for Intel OpenVINO Intermediate Representation (IR) models.
//!
//! ## OpenVINO IR Format
//!
//! OpenVINO IR consists of two files:
//! - `.xml` - Model graph structure (layers, connections, metadata)
//! - `.bin` - Binary weights data (tensors)
//!
//! ## Patch Format (.ovinp)
//!
//! The patch format stores delta updates between model versions:
//! ```text
//! [OVIN magic: 4 bytes]
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

pub use differ::OpenVINODiffer;
pub use format::{
    deserialize_patch, extension, is_ovinp, serialize_patch, OVIN_MAGIC, OVIN_VERSION,
};
pub use parser::{OpenVINOModel, OpenVINOParser, OpenVINOTensor};
pub use patcher::OpenVINOPatcher;

/// File extensions for OpenVINO format
pub const OPENVINO_EXTENSIONS: &[&str] = &["xml"];

/// Patch file extension for OpenVINO
pub const OPENVINO_PATCH_EXTENSION: &str = "ovinp";
