//! Mallorn ONNX - ONNX format support for neural network model delta updates
//!
//! This crate provides parsing, diffing, and patching for ONNX models,
//! producing and consuming `.onxp` patch files.
//!
//! # Example
//!
//! ```ignore
//! use mallorn_onnx::{ONNXDiffer, ONNXPatcher};
//!
//! // Create a diff between two models
//! let differ = ONNXDiffer::new();
//! let patch = differ.diff_from_bytes(&old_model, &new_model)?;
//!
//! // Apply the patch
//! let patcher = ONNXPatcher::new();
//! let new_model = patcher.apply(&old_model, &patch)?;
//! ```

pub mod differ;
pub mod format;
pub mod parser;
pub mod patcher;

// Generated protobuf types
mod onnx_proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub use differ::ONNXDiffer;
pub use format::{deserialize_patch, extension, is_onxp, serialize_patch, ONXP_MAGIC, ONXP_VERSION};
pub use parser::{ONNXDataType, ONNXModel, ONNXParser, ONNXTensor};
pub use patcher::ONNXPatcher;
