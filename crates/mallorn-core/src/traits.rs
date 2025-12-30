//! Core traits for Mallorn

use crate::error::{DiffError, ParseError, PatchError, SerializeError};
use crate::types::{DiffOptions, ParsedModel, Patch, PatchVerification, TensorInfo};

/// A parseable model format (TFLite, GGUF, ONNX, etc.)
pub trait ModelFormat: Send + Sync {
    /// Format identifier (e.g., "tflite", "gguf")
    fn format_id(&self) -> &'static str;

    /// File extensions this format handles
    fn extensions(&self) -> &[&'static str];

    /// Parse a model file into structured representation
    fn parse(&self, data: &[u8]) -> Result<ParsedModel, ParseError>;

    /// Serialize a parsed model back to bytes
    fn serialize(&self, model: &ParsedModel) -> Result<Vec<u8>, SerializeError>;

    /// Extract tensor metadata without full parse (for large models)
    fn extract_tensor_info(&self, data: &[u8]) -> Result<Vec<TensorInfo>, ParseError>;
}

/// Computes differences between two models
pub trait Differ: Send + Sync {
    /// Compute a patch that transforms `old` into `new`
    fn diff(&self, old: &ParsedModel, new: &ParsedModel) -> Result<Patch, DiffError>;

    /// Compute diff with custom options
    fn diff_with_options(
        &self,
        old: &ParsedModel,
        new: &ParsedModel,
        options: &DiffOptions,
    ) -> Result<Patch, DiffError>;
}

/// Applies patches to transform models
pub trait Patcher: Send + Sync {
    /// Apply a patch to transform `old` into `new`
    fn patch(&self, old: &[u8], patch: &Patch) -> Result<Vec<u8>, PatchError>;

    /// Verify a patch can be applied (without applying)
    fn verify(&self, old: &[u8], patch: &Patch) -> Result<PatchVerification, PatchError>;
}

/// Step result for streaming patcher
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepResult {
    /// More data to process
    Continue,
    /// Patch application complete
    Complete,
}

/// Streaming patcher for memory-constrained devices
pub trait StreamingPatcher {
    /// Initialize streaming patch application
    fn init(
        &mut self,
        old_reader: Box<dyn std::io::Read>,
        patch_reader: Box<dyn std::io::Read>,
    ) -> Result<(), PatchError>;

    /// Process one chunk, write output
    fn step(&mut self, output: &mut dyn std::io::Write) -> Result<StepResult, PatchError>;

    /// Finalize and verify
    fn finalize(&mut self) -> Result<PatchVerification, PatchError>;

    /// Abort and cleanup
    fn abort(&mut self);

    /// Required buffer size in bytes
    fn buffer_size(&self) -> usize;
}
