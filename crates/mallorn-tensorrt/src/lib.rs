//! Mallorn TensorRT - ONNX-based delta updates for TensorRT workflows
//!
//! TensorRT engines are GPU-specific and version-locked, making direct
//! patching impossible. This crate provides a hybrid workflow:
//!
//! 1. Store ONNX model deltas using proven `mallorn-onnx`
//! 2. Include TensorRT build configuration (precision, workspace, DLA)
//! 3. User applies patch to ONNX, then rebuilds TensorRT engine
//!
//! # Example
//!
//! ```ignore
//! use mallorn_tensorrt::{TensorRTDiffer, TensorRTConfig, Precision};
//!
//! // Create patch from ONNX models with TensorRT build config
//! let config = TensorRTConfig::default()
//!     .with_precision(Precision::FP16)
//!     .with_workspace_mb(1024);
//!
//! let differ = TensorRTDiffer::new();
//! let patch = differ.diff(&old_onnx, &new_onnx, config)?;
//!
//! // Serialize to .trtp file
//! let bytes = serialize_patch(&patch)?;
//!
//! // On device: apply patch to get new ONNX, then rebuild
//! let result = patcher.apply(&old_onnx, &patch)?;
//! println!("Rebuild command: {}", result.rebuild_command);
//! ```

pub mod config;
pub mod differ;
pub mod format;
pub mod patcher;

// Re-export main types
pub use config::{Precision, TensorRTConfig};
pub use differ::TensorRTDiffer;
pub use format::{deserialize_patch, serialize_patch, TensorRTPatch, TENSORRT_MAGIC};
pub use patcher::{ApplyResult, TensorRTPatcher};
