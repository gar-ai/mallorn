//! Mallorn SafeTensors - HuggingFace SafeTensors format support
//!
//! This crate provides parser, differ, and patcher for SafeTensors files.
//! SafeTensors is a simple, safe format for storing tensors used by HuggingFace.

pub mod differ;
pub mod format;
pub mod parser;
pub mod patcher;

pub use differ::SafeTensorsDiffer;
pub use format::{
    deserialize_patch, extension, is_sftp, serialize_patch, SafeTensorsFormat,
    SAFETENSORS_EXTENSIONS, SAFETENSORS_PATCH_EXTENSION, SFTP_MAGIC, SFTP_VERSION,
};
pub use parser::{SafeTensorsModel, SafeTensorsParser};
pub use patcher::SafeTensorsPatcher;
