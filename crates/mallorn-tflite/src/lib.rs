//! Mallorn TFLite - TensorFlow Lite format support
//!
//! This crate provides parsing, diffing, and patching for TFLite models.
//! It produces and consumes `.tflp` patch files.

pub mod differ;
pub mod format;
pub mod parser;
pub mod patcher;

// Re-export main types
pub use differ::TFLiteDiffer;
pub use format::{deserialize_patch, extension, is_tflp, serialize_patch, TFLP_MAGIC, TFLP_VERSION};
pub use parser::{TFLiteModel, TFLiteParser, TFLiteTensor};
pub use patcher::TFLitePatcher;
