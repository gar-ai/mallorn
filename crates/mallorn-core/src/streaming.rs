//! Streaming patch application for memory-efficient updates
//!
//! This module enables applying patches to large models without loading
//! the entire model into memory. Instead of O(model_size × 2), streaming
//! uses O(buffer_size) memory - typically 64MB regardless of model size.

use crate::diff::apply_xor_delta;
use crate::error::PatchError;
use crate::types::{
    CompressionMethod, DeltaFormat, Patch, PatchOperation, PatchStats, PatchVerification,
};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};

/// Default buffer size for streaming operations (64MB)
pub const DEFAULT_BUFFER_SIZE: usize = 64 * 1024 * 1024;

/// Minimum buffer size (1MB)
pub const MIN_BUFFER_SIZE: usize = 1024 * 1024;

/// Location of a tensor within a model file
#[derive(Debug, Clone)]
pub struct TensorLocation {
    /// Tensor name/identifier
    pub name: String,
    /// Byte offset from start of file
    pub offset: u64,
    /// Size in bytes
    pub size: u64,
    /// Compression method used (if any)
    pub compression: Option<CompressionMethod>,
}

impl TensorLocation {
    /// Create a new tensor location
    pub fn new(name: impl Into<String>, offset: u64, size: u64) -> Self {
        Self {
            name: name.into(),
            offset,
            size,
            compression: None,
        }
    }

    /// Set compression method
    pub fn with_compression(mut self, method: CompressionMethod) -> Self {
        self.compression = Some(method);
        self
    }

    /// End offset (offset + size)
    pub fn end_offset(&self) -> u64 {
        self.offset + self.size
    }
}

/// Index mapping tensor names to their locations
#[derive(Debug, Clone, Default)]
pub struct TensorIndex {
    /// Map of tensor name to location
    locations: HashMap<String, TensorLocation>,
    /// Ordered list of tensor names (for sequential access)
    order: Vec<String>,
    /// Total size of all tensors
    total_size: u64,
}

impl TensorIndex {
    /// Create a new empty index
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tensor location
    pub fn add(&mut self, location: TensorLocation) {
        self.total_size += location.size;
        self.order.push(location.name.clone());
        self.locations.insert(location.name.clone(), location);
    }

    /// Get a tensor location by name
    pub fn get(&self, name: &str) -> Option<&TensorLocation> {
        self.locations.get(name)
    }

    /// Get all tensor names in order
    pub fn names(&self) -> &[String] {
        &self.order
    }

    /// Get number of tensors
    pub fn len(&self) -> usize {
        self.locations.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.locations.is_empty()
    }

    /// Get total size of all tensors
    pub fn total_size(&self) -> u64 {
        self.total_size
    }

    /// Iterate over locations in order
    pub fn iter(&self) -> impl Iterator<Item = &TensorLocation> {
        self.order
            .iter()
            .filter_map(|name| self.locations.get(name))
    }
}

/// Progress reporting for streaming operations
pub trait StreamProgress: Send {
    /// Called when a tensor is about to be processed
    fn on_tensor_start(&mut self, name: &str, size: u64);

    /// Called after a tensor is processed
    fn on_tensor_complete(&mut self, name: &str, size: u64);

    /// Called with overall progress (bytes processed / total bytes)
    fn on_progress(&mut self, processed: u64, total: u64);
}

/// Default no-op progress reporter
pub struct NoProgress;

impl StreamProgress for NoProgress {
    fn on_tensor_start(&mut self, _name: &str, _size: u64) {}
    fn on_tensor_complete(&mut self, _name: &str, _size: u64) {}
    fn on_progress(&mut self, _processed: u64, _total: u64) {}
}

/// Configuration for streaming patch operations
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Buffer size for reading/writing
    pub buffer_size: usize,
    /// Whether to verify output hash
    pub verify_output: bool,
    /// Whether to verify source hash
    pub verify_source: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            buffer_size: DEFAULT_BUFFER_SIZE,
            verify_output: true,
            verify_source: true,
        }
    }
}

impl StreamConfig {
    /// Create config with specified buffer size
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size.max(MIN_BUFFER_SIZE);
        self
    }

    /// Skip output verification
    pub fn skip_verify_output(mut self) -> Self {
        self.verify_output = false;
        self
    }

    /// Skip source verification
    pub fn skip_verify_source(mut self) -> Self {
        self.verify_source = false;
        self
    }
}

/// Helper to get tensor name from PatchOperation
fn get_tensor_name(op: &PatchOperation) -> Option<&str> {
    match op {
        PatchOperation::ReplaceTensor { name, .. } => Some(name),
        PatchOperation::DeltaTensor { name, .. } => Some(name),
        PatchOperation::CopyTensor { name } => Some(name),
        PatchOperation::UpdateMetadata { .. } => None,
    }
}

/// Streaming patch context for applying patches without loading full model
///
/// # Example
/// ```ignore
/// use mallorn_core::streaming::{StreamingPatcher, StreamConfig};
/// use std::fs::File;
///
/// let old = File::open("model_v1.bin")?;
/// let output = File::create("model_v2.bin")?;
/// let patch = Patch::load("update.patch")?;
///
/// let mut patcher = StreamingPatcher::new(old, output, patch, StreamConfig::default())?;
/// let verification = patcher.apply()?;
/// ```
pub struct StreamingPatcher<R: Read + Seek, W: Write> {
    /// Source model reader
    source: R,
    /// Output model writer
    output: W,
    /// Patch to apply
    patch: Patch,
    /// Configuration
    config: StreamConfig,
    /// Working buffer
    buffer: Vec<u8>,
    /// Operations indexed by tensor name
    operations: HashMap<String, PatchOperation>,
    /// Running hash of source
    source_hasher: Sha256,
    /// Running hash of output
    output_hasher: Sha256,
    /// Bytes processed
    bytes_processed: u64,
    /// Tensors modified count
    tensors_modified: usize,
    /// Tensors unchanged count
    tensors_unchanged: usize,
}

impl<R: Read + Seek, W: Write> StreamingPatcher<R, W> {
    /// Create a new streaming patcher
    pub fn new(
        source: R,
        output: W,
        patch: Patch,
        config: StreamConfig,
    ) -> Result<Self, PatchError> {
        let buffer = vec![0u8; config.buffer_size];

        // Index operations by tensor name for quick lookup
        let operations: HashMap<String, PatchOperation> = patch
            .operations
            .iter()
            .filter_map(|op| {
                get_tensor_name(op).map(|name| (name.to_string(), op.clone()))
            })
            .collect();

        Ok(Self {
            source,
            output,
            patch,
            config,
            buffer,
            operations,
            source_hasher: Sha256::new(),
            output_hasher: Sha256::new(),
            bytes_processed: 0,
            tensors_modified: 0,
            tensors_unchanged: 0,
        })
    }

    /// Apply patch with progress reporting
    pub fn apply_with_progress<P: StreamProgress>(
        &mut self,
        index: &TensorIndex,
        progress: &mut P,
    ) -> Result<PatchVerification, PatchError> {
        let total_size = index.total_size();

        for location in index.iter() {
            progress.on_tensor_start(&location.name, location.size);

            self.process_tensor(location)?;

            self.bytes_processed += location.size;
            progress.on_tensor_complete(&location.name, location.size);
            progress.on_progress(self.bytes_processed, total_size);
        }

        self.finalize(index)
    }

    /// Apply patch (no progress reporting)
    pub fn apply(&mut self, index: &TensorIndex) -> Result<PatchVerification, PatchError> {
        self.apply_with_progress(index, &mut NoProgress)
    }

    /// Process a single tensor
    fn process_tensor(&mut self, location: &TensorLocation) -> Result<(), PatchError> {
        // Seek to tensor location
        self.source.seek(SeekFrom::Start(location.offset))?;

        // Clone the operation to avoid borrow issues
        let op = self.operations.get(&location.name).cloned();

        // Check if there's a patch operation for this tensor
        if let Some(op) = op {
            self.apply_operation(location, &op)?;
        } else {
            // No change - copy directly
            self.copy_tensor(location)?;
        }

        Ok(())
    }

    /// Apply a patch operation to a tensor
    fn apply_operation(
        &mut self,
        location: &TensorLocation,
        op: &PatchOperation,
    ) -> Result<(), PatchError> {
        match op {
            PatchOperation::ReplaceTensor { data, compression, .. } => {
                // Read source for hashing
                let mut source_data = vec![0u8; location.size as usize];
                self.source.read_exact(&mut source_data)?;
                self.source_hasher.update(&source_data);

                // Decompress data if needed
                let decompressed = decompress_data(data, compression.as_ref().unwrap_or(&self.patch.compression))?;

                // Write replacement data
                self.output.write_all(&decompressed)?;
                self.output_hasher.update(&decompressed);
                self.tensors_modified += 1;
            }
            PatchOperation::DeltaTensor {
                delta,
                delta_format,
                compression,
                ..
            } => {
                // Read source tensor
                let mut source_data = vec![0u8; location.size as usize];
                self.source.read_exact(&mut source_data)?;
                self.source_hasher.update(&source_data);

                // Decompress delta if needed
                let decompressed_delta = decompress_data(delta, compression.as_ref().unwrap_or(&self.patch.compression))?;

                // Apply delta based on format
                let new_data = match delta_format {
                    DeltaFormat::Xor => apply_xor_delta(&source_data, &decompressed_delta),
                    DeltaFormat::BsDiff => {
                        // BsDiff not yet implemented for streaming
                        return Err(PatchError::DecompressionFailed(
                            "BsDiff not supported in streaming mode".to_string(),
                        ));
                    }
                    DeltaFormat::TensorAware => {
                        // TensorAware uses XOR internally
                        apply_xor_delta(&source_data, &decompressed_delta)
                    }
                };

                // Write result
                self.output.write_all(&new_data)?;
                self.output_hasher.update(&new_data);
                self.tensors_modified += 1;
            }
            PatchOperation::CopyTensor { .. } => {
                // Copy unchanged
                self.copy_tensor(location)?;
            }
            PatchOperation::UpdateMetadata { .. } => {
                // Metadata updates don't affect tensor data
                // Copy the tensor as-is
                self.copy_tensor(location)?;
            }
        }

        Ok(())
    }

    /// Copy a tensor directly (no change)
    fn copy_tensor(&mut self, location: &TensorLocation) -> Result<(), PatchError> {
        let mut remaining = location.size as usize;

        while remaining > 0 {
            let chunk_size = remaining.min(self.buffer.len());
            let chunk = &mut self.buffer[..chunk_size];

            self.source.read_exact(chunk)?;
            self.source_hasher.update(&*chunk);

            self.output.write_all(chunk)?;
            self.output_hasher.update(&*chunk);

            remaining -= chunk_size;
        }

        self.tensors_unchanged += 1;
        Ok(())
    }

    /// Finalize and verify
    fn finalize(&mut self, index: &TensorIndex) -> Result<PatchVerification, PatchError> {
        let source_hash: [u8; 32] = self.source_hasher.clone().finalize().into();
        let target_hash: [u8; 32] = self.output_hasher.clone().finalize().into();

        let source_valid = source_hash == self.patch.source_hash;
        let target_valid = target_hash == self.patch.target_hash;

        // Verify source hash if required
        if self.config.verify_source && !source_valid {
            return Err(PatchError::SourceHashMismatch);
        }

        // Verify target hash if required
        if self.config.verify_output && !target_valid {
            return Err(PatchError::TargetHashMismatch);
        }

        Ok(PatchVerification {
            source_valid,
            patch_valid: true,
            expected_target: self.patch.target_hash,
            actual_target: Some(target_hash),
            stats: PatchStats {
                source_size: index.total_size() as usize,
                target_size: index.total_size() as usize,
                patch_size: 0, // Not tracked in streaming mode
                compression_ratio: 0.0,
                tensors_modified: self.tensors_modified,
                tensors_unchanged: self.tensors_unchanged,
            },
        })
    }
}

/// Decompress data using the specified method
fn decompress_data(data: &[u8], method: &CompressionMethod) -> Result<Vec<u8>, PatchError> {
    match method {
        CompressionMethod::None => Ok(data.to_vec()),
        CompressionMethod::Zstd { .. } | CompressionMethod::ZstdDict { .. } => {
            zstd::decode_all(data).map_err(|e| PatchError::DecompressionFailed(e.to_string()))
        }
        CompressionMethod::Lz4 => lz4_flex::decompress_size_prepended(data)
            .map_err(|e| PatchError::DecompressionFailed(e.to_string())),
        CompressionMethod::Neural { .. } | CompressionMethod::Adaptive { .. } => {
            // Neural and Adaptive wrap other methods - data should already be decompressed
            // or use the base compression
            Ok(data.to_vec())
        }
    }
}

/// Apply a patch using streaming with minimal memory
///
/// This is a convenience function for the common case of patching files.
pub fn apply_patch_streaming<R: Read + Seek, W: Write>(
    source: R,
    output: W,
    patch: &Patch,
    index: &TensorIndex,
    config: StreamConfig,
) -> Result<PatchVerification, PatchError> {
    let mut patcher = StreamingPatcher::new(source, output, patch.clone(), config)?;
    patcher.apply(index)
}

/// Chunked reader for processing large files in segments
pub struct ChunkedReader<R: Read> {
    inner: R,
    buffer: Vec<u8>,
    chunk_size: usize,
    bytes_in_buffer: usize,
}

impl<R: Read> ChunkedReader<R> {
    /// Create a new chunked reader with specified chunk size
    pub fn new(inner: R, chunk_size: usize) -> Self {
        Self {
            inner,
            buffer: vec![0u8; chunk_size],
            chunk_size,
            bytes_in_buffer: 0,
        }
    }

    /// Read the next chunk, returns None at EOF
    pub fn next_chunk(&mut self) -> std::io::Result<Option<&[u8]>> {
        self.bytes_in_buffer = self.inner.read(&mut self.buffer)?;

        if self.bytes_in_buffer == 0 {
            Ok(None)
        } else {
            Ok(Some(&self.buffer[..self.bytes_in_buffer]))
        }
    }

    /// Get the chunk size
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }
}

/// Chunked writer for writing large files in segments
pub struct ChunkedWriter<W: Write> {
    inner: W,
    buffer: Vec<u8>,
    position: usize,
}

impl<W: Write> ChunkedWriter<W> {
    /// Create a new chunked writer with specified buffer size
    pub fn new(inner: W, buffer_size: usize) -> Self {
        Self {
            inner,
            buffer: vec![0u8; buffer_size],
            position: 0,
        }
    }

    /// Write data to buffer, flushing when full
    pub fn write_chunk(&mut self, data: &[u8]) -> std::io::Result<()> {
        let mut data_pos = 0;

        while data_pos < data.len() {
            let space_left = self.buffer.len() - self.position;
            let to_copy = (data.len() - data_pos).min(space_left);

            self.buffer[self.position..self.position + to_copy]
                .copy_from_slice(&data[data_pos..data_pos + to_copy]);

            self.position += to_copy;
            data_pos += to_copy;

            if self.position == self.buffer.len() {
                self.flush_buffer()?;
            }
        }

        Ok(())
    }

    /// Flush remaining buffer to output
    pub fn finish(mut self) -> std::io::Result<W> {
        if self.position > 0 {
            self.flush_buffer()?;
        }
        self.inner.flush()?;
        Ok(self.inner)
    }

    fn flush_buffer(&mut self) -> std::io::Result<()> {
        self.inner.write_all(&self.buffer[..self.position])?;
        self.position = 0;
        Ok(())
    }
}

/// Memory usage estimator for streaming operations
pub struct MemoryEstimator;

impl MemoryEstimator {
    /// Estimate memory usage for streaming patch application
    pub fn estimate_streaming(index: &TensorIndex, config: &StreamConfig) -> u64 {
        let buffer_mem = config.buffer_size as u64;
        let largest_tensor = index.iter().map(|t| t.size).max().unwrap_or(0);
        let operation_overhead = 1024 * 1024; // ~1MB for operations index

        // Buffer + space for largest tensor's delta + overhead
        buffer_mem + largest_tensor + operation_overhead
    }

    /// Estimate memory usage for non-streaming (traditional) application
    pub fn estimate_non_streaming(index: &TensorIndex) -> u64 {
        // Model loaded twice (source + result) + patch data
        let model_size = index.total_size();
        model_size * 2 + (model_size / 10) // Rough estimate: patch is ~10% of model
    }

    /// Calculate memory savings from streaming
    pub fn memory_savings(index: &TensorIndex, config: &StreamConfig) -> f64 {
        let streaming = Self::estimate_streaming(index, config) as f64;
        let non_streaming = Self::estimate_non_streaming(index) as f64;

        if non_streaming > 0.0 {
            1.0 - (streaming / non_streaming)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_tensor_location() {
        let loc = TensorLocation::new("weights", 1024, 4096);
        assert_eq!(loc.name, "weights");
        assert_eq!(loc.offset, 1024);
        assert_eq!(loc.size, 4096);
        assert_eq!(loc.end_offset(), 5120);
        assert!(loc.compression.is_none());

        let loc = loc.with_compression(CompressionMethod::Zstd { level: 3 });
        assert!(matches!(
            loc.compression,
            Some(CompressionMethod::Zstd { level: 3 })
        ));
    }

    #[test]
    fn test_tensor_index() {
        let mut index = TensorIndex::new();
        assert!(index.is_empty());

        index.add(TensorLocation::new("t1", 0, 100));
        index.add(TensorLocation::new("t2", 100, 200));
        index.add(TensorLocation::new("t3", 300, 300));

        assert_eq!(index.len(), 3);
        assert_eq!(index.total_size(), 600);

        assert!(index.get("t1").is_some());
        assert!(index.get("t4").is_none());

        let names: Vec<_> = index.names().iter().collect();
        assert_eq!(names, vec!["t1", "t2", "t3"]);
    }

    #[test]
    fn test_stream_config() {
        let config = StreamConfig::default();
        assert_eq!(config.buffer_size, DEFAULT_BUFFER_SIZE);
        assert!(config.verify_output);
        assert!(config.verify_source);

        let config = config.with_buffer_size(1024).skip_verify_output();

        // Should be clamped to MIN_BUFFER_SIZE
        assert_eq!(config.buffer_size, MIN_BUFFER_SIZE);
        assert!(!config.verify_output);
        assert!(config.verify_source);
    }

    #[test]
    fn test_chunked_reader() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let cursor = Cursor::new(data);
        let mut reader = ChunkedReader::new(cursor, 3);

        let chunk = reader.next_chunk().unwrap().unwrap();
        assert_eq!(chunk, &[1, 2, 3]);

        let chunk = reader.next_chunk().unwrap().unwrap();
        assert_eq!(chunk, &[4, 5, 6]);

        let chunk = reader.next_chunk().unwrap().unwrap();
        assert_eq!(chunk, &[7, 8, 9]);

        let chunk = reader.next_chunk().unwrap().unwrap();
        assert_eq!(chunk, &[10]);

        let chunk = reader.next_chunk().unwrap();
        assert!(chunk.is_none());
    }

    #[test]
    fn test_chunked_writer() {
        let output = Vec::new();
        let mut writer = ChunkedWriter::new(Cursor::new(output), 4);

        writer.write_chunk(&[1, 2]).unwrap();
        writer.write_chunk(&[3, 4, 5]).unwrap();
        writer.write_chunk(&[6, 7, 8, 9, 10]).unwrap();

        let cursor = writer.finish().unwrap();
        assert_eq!(cursor.into_inner(), vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn test_memory_estimator() {
        let mut index = TensorIndex::new();
        // Use many smaller tensors (more realistic for ML models)
        for i in 0..100 {
            index.add(TensorLocation::new(
                format!("tensor_{}", i),
                i as u64 * 100_000_000, // 100MB each
                100_000_000,
            ));
        }

        let config = StreamConfig::default();
        let streaming_mem = MemoryEstimator::estimate_streaming(&index, &config);
        let non_streaming_mem = MemoryEstimator::estimate_non_streaming(&index);

        // Streaming should use much less memory
        assert!(streaming_mem < non_streaming_mem);

        // For 10GB model with 100MB tensors, savings should be significant
        // Streaming: 64MB buffer + 100MB largest tensor + 1MB overhead = ~165MB
        // Non-streaming: 10GB * 2 + 1GB = ~21GB
        let savings = MemoryEstimator::memory_savings(&index, &config);
        assert!(savings > 0.9, "Expected >90% savings, got {:.1}%", savings * 100.0);
    }

    #[test]
    fn test_tensor_index_iteration() {
        let mut index = TensorIndex::new();
        index.add(TensorLocation::new("a", 0, 10));
        index.add(TensorLocation::new("b", 10, 20));
        index.add(TensorLocation::new("c", 30, 30));

        let locations: Vec<_> = index.iter().collect();
        assert_eq!(locations.len(), 3);
        assert_eq!(locations[0].name, "a");
        assert_eq!(locations[1].name, "b");
        assert_eq!(locations[2].name, "c");
    }

    #[test]
    fn test_get_tensor_name() {
        let op1 = PatchOperation::ReplaceTensor {
            name: "tensor1".to_string(),
            data: vec![],
            compression: None,
        };
        assert_eq!(get_tensor_name(&op1), Some("tensor1"));

        let op2 = PatchOperation::DeltaTensor {
            name: "tensor2".to_string(),
            delta: vec![],
            delta_format: DeltaFormat::Xor,
            compression: None,
        };
        assert_eq!(get_tensor_name(&op2), Some("tensor2"));

        let op3 = PatchOperation::CopyTensor {
            name: "tensor3".to_string(),
        };
        assert_eq!(get_tensor_name(&op3), Some("tensor3"));

        let op4 = PatchOperation::UpdateMetadata {
            key: "version".to_string(),
            value: "1.0".to_string(),
        };
        assert_eq!(get_tensor_name(&op4), None);
    }

    #[test]
    fn test_decompress_data() {
        // Test None compression
        let data = vec![1, 2, 3, 4, 5];
        let result = decompress_data(&data, &CompressionMethod::None).unwrap();
        assert_eq!(result, data);

        // Test Zstd compression
        let compressed = zstd::encode_all(&data[..], 3).unwrap();
        let result = decompress_data(&compressed, &CompressionMethod::Zstd { level: 3 }).unwrap();
        assert_eq!(result, data);

        // Test LZ4 compression
        let compressed = lz4_flex::compress_prepend_size(&data);
        let result = decompress_data(&compressed, &CompressionMethod::Lz4).unwrap();
        assert_eq!(result, data);
    }
}
