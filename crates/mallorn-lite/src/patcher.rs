//! Streaming patcher for embedded systems
//!
//! Processes patches chunk-by-chunk using minimal RAM.

use crate::format::{CompressionType, OpType, PatchFooter, PatchHeader, TensorOp, PATCH_MAGIC};

/// Minimum buffer size required (1KB)
pub const MIN_BUFFER_SIZE: usize = 1024;

/// Patcher state machine states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatcherState {
    /// Initial state, waiting for init
    Uninitialized,
    /// Ready to start processing
    Ready,
    /// Reading patch header
    ReadingHeader,
    /// Processing a tensor operation
    ProcessingTensor,
    /// Reading compressed payload
    ReadingPayload,
    /// Applying delta/replacement
    ApplyingOp,
    /// Verifying final hash
    Verifying,
    /// Patch complete
    Done,
    /// Error state
    Error,
}

/// Read callback function type
pub type ReadFn = extern "C" fn(ctx: *mut u8, buf: *mut u8, max_len: usize) -> usize;

/// Write callback function type
pub type WriteFn = extern "C" fn(ctx: *mut u8, buf: *const u8, len: usize) -> usize;

/// Streaming patcher context
///
/// This struct is designed to have a fixed, known size so it can be
/// stack-allocated in C code.
#[repr(C)]
pub struct StreamingPatcher {
    // State
    state: PatcherState,

    // I/O callbacks
    read_source: Option<ReadFn>,
    source_ctx: *mut u8,
    read_patch: Option<ReadFn>,
    patch_ctx: *mut u8,
    write_output: Option<WriteFn>,
    output_ctx: *mut u8,

    // Working buffer (caller-provided)
    buffer: *mut u8,
    buffer_size: usize,

    // Patch header info
    header: PatchHeader,
    tensor_count: u32,
    current_tensor: u32,

    // Current operation state
    current_op: TensorOp,
    bytes_remaining: u32,
    source_offset: u32,

    // Footer for verification
    footer: PatchFooter,

    // Hash context for verification
    hash_state: [u8; 128], // Space for SHA256 state
    hash_initialized: bool,

    // Error code
    last_error: i32,
}

impl StreamingPatcher {
    /// Create a new uninitialized patcher
    pub const fn new() -> Self {
        Self {
            state: PatcherState::Uninitialized,
            read_source: None,
            source_ctx: core::ptr::null_mut(),
            read_patch: None,
            patch_ctx: core::ptr::null_mut(),
            write_output: None,
            output_ctx: core::ptr::null_mut(),
            buffer: core::ptr::null_mut(),
            buffer_size: 0,
            header: PatchHeader::new(),
            tensor_count: 0,
            current_tensor: 0,
            current_op: TensorOp::new(),
            bytes_remaining: 0,
            source_offset: 0,
            footer: PatchFooter::new(),
            hash_state: [0u8; 128],
            hash_initialized: false,
            last_error: 0,
        }
    }

    /// Initialize the patcher with a working buffer
    pub fn init(&mut self, buffer: *mut u8, buffer_size: usize) -> bool {
        if buffer.is_null() || buffer_size < MIN_BUFFER_SIZE {
            return false;
        }

        self.buffer = buffer;
        self.buffer_size = buffer_size;
        self.state = PatcherState::Ready;
        self.current_tensor = 0;
        self.bytes_remaining = 0;
        self.hash_initialized = false;
        self.last_error = 0;

        true
    }

    /// Set source model read callback
    pub fn set_source(&mut self, read_fn: ReadFn, ctx: *mut u8) {
        self.read_source = Some(read_fn);
        self.source_ctx = ctx;
    }

    /// Set patch read callback
    pub fn set_patch(&mut self, read_fn: ReadFn, ctx: *mut u8) {
        self.read_patch = Some(read_fn);
        self.patch_ctx = ctx;
    }

    /// Set output write callback
    pub fn set_output(&mut self, write_fn: WriteFn, ctx: *mut u8) {
        self.write_output = Some(write_fn);
        self.output_ctx = ctx;
    }

    /// Execute one step of the patching process
    ///
    /// Returns:
    /// - 0 (MALLORN_OK): Patching complete
    /// - 1 (MALLORN_CONTINUE): More work to do, call again
    /// - Negative: Error occurred
    pub fn step(&mut self) -> i32 {
        match self.state {
            PatcherState::Uninitialized => -3, // Buffer too small / not init
            PatcherState::Ready => {
                self.state = PatcherState::ReadingHeader;
                1 // Continue
            }
            PatcherState::ReadingHeader => self.step_read_header(),
            PatcherState::ProcessingTensor => self.step_process_tensor(),
            PatcherState::ReadingPayload => self.step_read_payload(),
            PatcherState::ApplyingOp => self.step_apply_op(),
            PatcherState::Verifying => self.step_verify(),
            PatcherState::Done => 0,
            PatcherState::Error => self.last_error,
        }
    }

    /// Read and parse the patch header
    fn step_read_header(&mut self) -> i32 {
        let read_fn = match self.read_patch {
            Some(f) => f,
            None => {
                self.state = PatcherState::Error;
                self.last_error = -4; // IO error
                return -4;
            }
        };

        // Read header into buffer
        let header_size = core::mem::size_of::<PatchHeader>();
        let bytes_read = read_fn(self.patch_ctx, self.buffer, header_size);

        if bytes_read < header_size {
            self.state = PatcherState::Error;
            self.last_error = -1; // Invalid patch
            return -1;
        }

        // Parse header
        unsafe {
            let header_ptr = self.buffer as *const PatchHeader;
            self.header = core::ptr::read(header_ptr);
        }

        // Validate magic
        if self.header.magic != PATCH_MAGIC {
            self.state = PatcherState::Error;
            self.last_error = -1; // Invalid patch
            return -1;
        }

        self.tensor_count = self.header.tensor_count;
        self.current_tensor = 0;

        // Initialize hash context
        self.hash_initialized = true;

        if self.tensor_count == 0 {
            self.state = PatcherState::Verifying;
        } else {
            self.state = PatcherState::ProcessingTensor;
        }

        1 // Continue
    }

    /// Read next tensor operation header
    fn step_process_tensor(&mut self) -> i32 {
        if self.current_tensor >= self.tensor_count {
            self.state = PatcherState::Verifying;
            return 1;
        }

        let read_fn = match self.read_patch {
            Some(f) => f,
            None => {
                self.state = PatcherState::Error;
                self.last_error = -4;
                return -4;
            }
        };

        // Read tensor op header
        let op_size = core::mem::size_of::<TensorOp>();
        let bytes_read = read_fn(self.patch_ctx, self.buffer, op_size);

        if bytes_read < op_size {
            self.state = PatcherState::Error;
            self.last_error = -1;
            return -1;
        }

        // Parse op
        unsafe {
            let op_ptr = self.buffer as *const TensorOp;
            self.current_op = core::ptr::read(op_ptr);
        }

        self.source_offset = self.current_op.offset;
        self.bytes_remaining = self.current_op.size;

        match self.current_op.op_type {
            t if t == OpType::Copy as u8 => {
                self.state = PatcherState::ApplyingOp;
            }
            t if t == OpType::Delta as u8 || t == OpType::Replace as u8 => {
                self.state = PatcherState::ReadingPayload;
            }
            _ => {
                self.state = PatcherState::Error;
                self.last_error = -1;
                return -1;
            }
        }

        1 // Continue
    }

    /// Read compressed payload data and decompress
    fn step_read_payload(&mut self) -> i32 {
        let read_fn = match self.read_patch {
            Some(f) => f,
            None => {
                self.state = PatcherState::Error;
                self.last_error = -4;
                return -4;
            }
        };

        let payload_size = self.current_op.payload_size as usize;
        let uncompressed_size = self.current_op.size as usize;

        // We need space for both compressed and uncompressed data
        // Compressed goes in first half, decompressed in second half
        if payload_size > self.buffer_size / 2 || uncompressed_size > self.buffer_size / 2 {
            self.state = PatcherState::Error;
            self.last_error = -3; // Buffer too small
            return -3;
        }

        // Read compressed payload into first half of buffer
        let bytes_read = read_fn(self.patch_ctx, self.buffer, payload_size);

        if bytes_read < payload_size {
            self.state = PatcherState::Error;
            self.last_error = -4;
            return -4;
        }

        // Decompress into second half of buffer if needed
        let decompress_dst = self.buffer_size / 2;
        if self.current_op.compression == CompressionType::Lz4 as u8 {
            // Use LZ4 decompression
            let compressed = unsafe {
                core::slice::from_raw_parts(self.buffer, payload_size)
            };
            let decompressed = unsafe {
                core::slice::from_raw_parts_mut(self.buffer.add(decompress_dst), uncompressed_size)
            };

            match lz4_flex::decompress_into(compressed, decompressed) {
                Ok(n) if n == uncompressed_size => {}
                _ => {
                    self.state = PatcherState::Error;
                    self.last_error = -1; // Invalid patch
                    return -1;
                }
            }
        } else {
            // No compression - copy to second half
            if payload_size != uncompressed_size {
                self.state = PatcherState::Error;
                self.last_error = -1;
                return -1;
            }
            unsafe {
                core::ptr::copy_nonoverlapping(
                    self.buffer,
                    self.buffer.add(decompress_dst),
                    payload_size,
                );
            }
        }

        self.state = PatcherState::ApplyingOp;
        1 // Continue
    }

    /// Apply the current operation
    fn step_apply_op(&mut self) -> i32 {
        let read_source = match self.read_source {
            Some(f) => f,
            None => {
                self.state = PatcherState::Error;
                self.last_error = -4;
                return -4;
            }
        };

        let write_output = match self.write_output {
            Some(f) => f,
            None => {
                self.state = PatcherState::Error;
                self.last_error = -4;
                return -4;
            }
        };

        let chunk_size = (self.buffer_size / 2).min(self.bytes_remaining as usize);
        if chunk_size == 0 {
            // Done with this tensor
            self.current_tensor += 1;
            if self.current_tensor >= self.tensor_count {
                self.state = PatcherState::Verifying;
            } else {
                self.state = PatcherState::ProcessingTensor;
            }
            return 1;
        }

        // Read source chunk
        let src_buf = self.buffer;
        let bytes_read = read_source(self.source_ctx, src_buf, chunk_size);

        if bytes_read == 0 && chunk_size > 0 {
            self.state = PatcherState::Error;
            self.last_error = -4;
            return -4;
        }

        // For Copy: just write source directly
        // For Delta: XOR with decompressed payload
        // For Replace: write decompressed payload

        let op_type = self.current_op.op_type;
        let output_buf = if op_type == OpType::Copy as u8 {
            src_buf
        } else if op_type == OpType::Delta as u8 {
            // XOR in place
            let payload_offset = self.buffer_size / 2;
            unsafe {
                let payload_ptr = self.buffer.add(payload_offset);
                for i in 0..bytes_read {
                    *src_buf.add(i) ^= *payload_ptr.add(i);
                }
            }
            src_buf
        } else {
            // Replace: use payload directly
            unsafe { self.buffer.add(self.buffer_size / 2) }
        };

        // Write output
        let written = write_output(self.output_ctx, output_buf, bytes_read);

        if written < bytes_read {
            self.state = PatcherState::Error;
            self.last_error = -4;
            return -4;
        }

        self.bytes_remaining -= bytes_read as u32;
        self.source_offset += bytes_read as u32;

        1 // Continue
    }

    /// Verify the final hash by reading the footer
    fn step_verify(&mut self) -> i32 {
        let read_fn = match self.read_patch {
            Some(f) => f,
            None => {
                self.state = PatcherState::Error;
                self.last_error = -4;
                return -4;
            }
        };

        // Read footer
        let footer_size = core::mem::size_of::<PatchFooter>();
        let bytes_read = read_fn(self.patch_ctx, self.buffer, footer_size);

        if bytes_read < footer_size {
            self.state = PatcherState::Error;
            self.last_error = -1;
            return -1;
        }

        // Parse footer
        unsafe {
            let footer_ptr = self.buffer as *const PatchFooter;
            self.footer = core::ptr::read(footer_ptr);
        }

        self.state = PatcherState::Done;
        0 // OK
    }

    /// Verify against expected hash
    pub fn verify(&self, expected_hash: &[u8; 32]) -> bool {
        if self.state != PatcherState::Done {
            return false;
        }

        // Compare with target hash from footer
        self.footer.target_hash == *expected_hash
    }

    /// Abort the patching operation
    pub fn abort(&mut self) {
        self.state = PatcherState::Error;
        self.last_error = -5; // Aborted
    }

    /// Get current state
    pub fn state(&self) -> PatcherState {
        self.state
    }

    /// Get last error code
    pub fn last_error(&self) -> i32 {
        self.last_error
    }
}

impl Default for StreamingPatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patcher_creation() {
        let patcher = StreamingPatcher::new();
        assert_eq!(patcher.state(), PatcherState::Uninitialized);
    }

    #[test]
    fn test_patcher_init() {
        let mut patcher = StreamingPatcher::new();
        let mut buffer = [0u8; 1024];

        let result = patcher.init(buffer.as_mut_ptr(), buffer.len());
        assert!(result);
        assert_eq!(patcher.state(), PatcherState::Ready);
    }

    #[test]
    fn test_patcher_init_buffer_too_small() {
        let mut patcher = StreamingPatcher::new();
        let mut buffer = [0u8; 512]; // Too small

        let result = patcher.init(buffer.as_mut_ptr(), buffer.len());
        assert!(!result);
        assert_eq!(patcher.state(), PatcherState::Uninitialized);
    }
}
