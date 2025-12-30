//! Integration tests for mallorn-lite streaming patcher
//!
//! These tests verify end-to-end patching with in-memory I/O callbacks.

use mallorn_lite::{
    CompressionType, PatchFooter, PatchHeader, StreamingPatcher, TensorOp,
    PATCH_MAGIC, PATCH_VERSION, MIN_BUFFER_SIZE,
};
use std::cell::RefCell;

// ============================================================================
// Test Helpers
// ============================================================================

/// In-memory reader context
struct MemReader {
    data: Vec<u8>,
    pos: usize,
}

impl MemReader {
    fn new(data: Vec<u8>) -> Self {
        Self { data, pos: 0 }
    }

    fn into_boxed(self) -> Box<RefCell<Self>> {
        Box::new(RefCell::new(self))
    }
}

/// In-memory writer context
struct MemWriter {
    data: Vec<u8>,
}

impl MemWriter {
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    fn into_boxed(self) -> Box<RefCell<Self>> {
        Box::new(RefCell::new(self))
    }
}

/// Read callback for MemReader
extern "C" fn mem_read(ctx: *mut u8, buf: *mut u8, max_len: usize) -> usize {
    let reader = unsafe { &mut *(ctx as *mut RefCell<MemReader>) };
    let mut reader = reader.borrow_mut();

    let remaining = reader.data.len() - reader.pos;
    let to_read = max_len.min(remaining);

    if to_read > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(
                reader.data[reader.pos..].as_ptr(),
                buf,
                to_read,
            );
        }
        reader.pos += to_read;
    }

    to_read
}

/// Write callback for MemWriter
extern "C" fn mem_write(ctx: *mut u8, buf: *const u8, len: usize) -> usize {
    let writer = unsafe { &mut *(ctx as *mut RefCell<MemWriter>) };
    let mut writer = writer.borrow_mut();

    let slice = unsafe { std::slice::from_raw_parts(buf, len) };
    writer.data.extend_from_slice(slice);

    len
}

/// Build a streaming patch from components
fn build_patch(
    source_hash: [u8; 32],
    target_hash: [u8; 32],
    ops: &[(TensorOp, Option<&[u8]>)],
) -> Vec<u8> {
    let mut patch = Vec::new();

    // Calculate total uncompressed size
    let total_size: u32 = ops.iter().map(|(op, _)| { op.size }).sum();

    // Write header
    let header = PatchHeader {
        magic: PATCH_MAGIC,
        version: PATCH_VERSION,
        source_hash,
        tensor_count: ops.len() as u32,
        total_size,
    };

    patch.extend_from_slice(&header.magic.to_le_bytes());
    patch.extend_from_slice(&header.version.to_le_bytes());
    patch.extend_from_slice(&header.source_hash);
    patch.extend_from_slice(&header.tensor_count.to_le_bytes());
    patch.extend_from_slice(&header.total_size.to_le_bytes());

    // Write ops and payloads
    for (op, payload) in ops {
        patch.push(op.op_type);
        patch.push(op.compression);
        patch.extend_from_slice(&op.reserved.to_le_bytes());
        patch.extend_from_slice(&op.offset.to_le_bytes());
        patch.extend_from_slice(&op.size.to_le_bytes());
        patch.extend_from_slice(&op.payload_size.to_le_bytes());

        if let Some(data) = payload {
            patch.extend_from_slice(data);
        }
    }

    // Write footer (without CRC for simplicity in tests)
    let footer = PatchFooter {
        target_hash,
        crc32: 0,
    };

    patch.extend_from_slice(&footer.target_hash);
    patch.extend_from_slice(&footer.crc32.to_le_bytes());

    patch
}

/// Run patcher to completion
fn run_patcher(patcher: &mut StreamingPatcher) -> i32 {
    let mut result = 1;
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 10000;

    while result == 1 && iterations < MAX_ITERATIONS {
        result = patcher.step();
        iterations += 1;
    }

    if iterations >= MAX_ITERATIONS {
        return -99; // Timeout
    }

    result
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn test_buffer_too_small() {
    let mut patcher = StreamingPatcher::new();
    let mut buffer = [0u8; 512]; // Below MIN_BUFFER_SIZE

    let result = patcher.init(buffer.as_mut_ptr(), buffer.len());
    assert!(!result, "Init should fail with small buffer");
}

#[test]
fn test_buffer_minimum_size() {
    let mut patcher = StreamingPatcher::new();
    let mut buffer = [0u8; MIN_BUFFER_SIZE];

    let result = patcher.init(buffer.as_mut_ptr(), buffer.len());
    assert!(result, "Init should succeed with minimum buffer size");
}

#[test]
fn test_null_buffer() {
    let mut patcher = StreamingPatcher::new();

    let result = patcher.init(std::ptr::null_mut(), 1024);
    assert!(!result, "Init should fail with null buffer");
}

#[test]
fn test_empty_patch() {
    let source_hash = [0u8; 32];
    let target_hash = [1u8; 32];

    // Create empty patch (no operations)
    let patch_data = build_patch(source_hash, target_hash, &[]);

    // Set up I/O
    let source = MemReader::new(vec![]).into_boxed();
    let patch = MemReader::new(patch_data).into_boxed();
    let output = MemWriter::new().into_boxed();

    let mut buffer = [0u8; 2048];
    let mut patcher = StreamingPatcher::new();
    patcher.init(buffer.as_mut_ptr(), buffer.len());

    patcher.set_source(mem_read, source.as_ref() as *const _ as *mut u8);
    patcher.set_patch(mem_read, patch.as_ref() as *const _ as *mut u8);
    patcher.set_output(mem_write, output.as_ref() as *const _ as *mut u8);

    let result = run_patcher(&mut patcher);
    assert_eq!(result, 0, "Empty patch should succeed");

    // Verify with correct hash
    assert!(patcher.verify(&target_hash), "Hash verification should pass");

    // Verify with wrong hash
    let wrong_hash = [2u8; 32];
    assert!(!patcher.verify(&wrong_hash), "Wrong hash should fail");
}

#[test]
fn test_copy_operation() {
    let source_data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    let source_hash = [0u8; 32];
    let target_hash = [1u8; 32];

    // Create patch with single copy operation
    let copy_op = TensorOp::copy(0, source_data.len() as u32);
    let patch_data = build_patch(source_hash, target_hash, &[(copy_op, None)]);

    // Set up I/O
    let source = MemReader::new(source_data.clone()).into_boxed();
    let patch = MemReader::new(patch_data).into_boxed();
    let output = MemWriter::new().into_boxed();

    let mut buffer = [0u8; 2048];
    let mut patcher = StreamingPatcher::new();
    patcher.init(buffer.as_mut_ptr(), buffer.len());

    patcher.set_source(mem_read, source.as_ref() as *const _ as *mut u8);
    patcher.set_patch(mem_read, patch.as_ref() as *const _ as *mut u8);
    patcher.set_output(mem_write, output.as_ref() as *const _ as *mut u8);

    let result = run_patcher(&mut patcher);
    assert_eq!(result, 0, "Copy operation should succeed");

    // Verify output matches source (copy = identical)
    let output_data = output.borrow().data.clone();
    assert_eq!(output_data, source_data, "Output should match source for copy");
}

#[test]
fn test_replace_operation_uncompressed() {
    let source_data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    let replacement = vec![10u8, 20, 30, 40, 50, 60, 70, 80];
    let source_hash = [0u8; 32];
    let target_hash = [1u8; 32];

    // Create patch with replace operation (no compression)
    let replace_op = TensorOp::replace(
        0,
        replacement.len() as u32,
        replacement.len() as u32,
        CompressionType::None,
    );
    let patch_data = build_patch(
        source_hash,
        target_hash,
        &[(replace_op, Some(&replacement))],
    );

    // Set up I/O
    let source = MemReader::new(source_data).into_boxed();
    let patch = MemReader::new(patch_data).into_boxed();
    let output = MemWriter::new().into_boxed();

    let mut buffer = [0u8; 2048];
    let mut patcher = StreamingPatcher::new();
    patcher.init(buffer.as_mut_ptr(), buffer.len());

    patcher.set_source(mem_read, source.as_ref() as *const _ as *mut u8);
    patcher.set_patch(mem_read, patch.as_ref() as *const _ as *mut u8);
    patcher.set_output(mem_write, output.as_ref() as *const _ as *mut u8);

    let result = run_patcher(&mut patcher);
    assert_eq!(result, 0, "Replace operation should succeed");

    let output_data = output.borrow().data.clone();
    assert_eq!(output_data, replacement, "Output should match replacement data");
}

#[test]
fn test_delta_operation_uncompressed() {
    let source_data = vec![0xAAu8, 0xBB, 0xCC, 0xDD];
    let delta_data = vec![0x11u8, 0x22, 0x33, 0x44]; // XOR delta
    let expected = vec![0xBBu8, 0x99, 0xFF, 0x99]; // source XOR delta
    let source_hash = [0u8; 32];
    let target_hash = [1u8; 32];

    // Create patch with delta operation (no compression)
    let delta_op = TensorOp::delta(
        0,
        delta_data.len() as u32,
        delta_data.len() as u32,
        CompressionType::None,
    );
    let patch_data = build_patch(
        source_hash,
        target_hash,
        &[(delta_op, Some(&delta_data))],
    );

    // Set up I/O
    let source = MemReader::new(source_data).into_boxed();
    let patch = MemReader::new(patch_data).into_boxed();
    let output = MemWriter::new().into_boxed();

    let mut buffer = [0u8; 2048];
    let mut patcher = StreamingPatcher::new();
    patcher.init(buffer.as_mut_ptr(), buffer.len());

    patcher.set_source(mem_read, source.as_ref() as *const _ as *mut u8);
    patcher.set_patch(mem_read, patch.as_ref() as *const _ as *mut u8);
    patcher.set_output(mem_write, output.as_ref() as *const _ as *mut u8);

    let result = run_patcher(&mut patcher);
    assert_eq!(result, 0, "Delta operation should succeed");

    let output_data = output.borrow().data.clone();
    assert_eq!(output_data, expected, "Output should be source XOR delta");
}

#[test]
fn test_replace_with_lz4_compression() {
    let source_data = vec![0u8; 128];
    // Compressible data (repeated bytes)
    let replacement = vec![0x42u8; 64];
    let source_hash = [0u8; 32];
    let target_hash = [1u8; 32];

    // Compress with LZ4
    let compressed = lz4_flex::compress(&replacement);

    // Create patch with compressed replace
    let replace_op = TensorOp::replace(
        0,
        replacement.len() as u32,
        compressed.len() as u32,
        CompressionType::Lz4,
    );
    let patch_data = build_patch(
        source_hash,
        target_hash,
        &[(replace_op, Some(&compressed))],
    );

    // Set up I/O
    let source = MemReader::new(source_data).into_boxed();
    let patch = MemReader::new(patch_data).into_boxed();
    let output = MemWriter::new().into_boxed();

    let mut buffer = [0u8; 2048];
    let mut patcher = StreamingPatcher::new();
    patcher.init(buffer.as_mut_ptr(), buffer.len());

    patcher.set_source(mem_read, source.as_ref() as *const _ as *mut u8);
    patcher.set_patch(mem_read, patch.as_ref() as *const _ as *mut u8);
    patcher.set_output(mem_write, output.as_ref() as *const _ as *mut u8);

    let result = run_patcher(&mut patcher);
    assert_eq!(result, 0, "LZ4 compressed replace should succeed");

    let output_data = output.borrow().data.clone();
    assert_eq!(output_data, replacement, "Output should match decompressed replacement");
}

#[test]
fn test_multiple_operations() {
    // Source: [AAAA][BBBB][CCCC] (12 bytes, 3 segments)
    let source_data = vec![
        0xAA, 0xAA, 0xAA, 0xAA,
        0xBB, 0xBB, 0xBB, 0xBB,
        0xCC, 0xCC, 0xCC, 0xCC,
    ];
    let source_hash = [0u8; 32];
    let target_hash = [1u8; 32];

    // Op 1: Copy first segment (unchanged)
    let op1 = TensorOp::copy(0, 4);

    // Op 2: Replace second segment
    let replacement = vec![0xDD, 0xDD, 0xDD, 0xDD];
    let op2 = TensorOp::replace(4, 4, 4, CompressionType::None);

    // Op 3: Copy third segment (unchanged)
    let op3 = TensorOp::copy(8, 4);

    let patch_data = build_patch(
        source_hash,
        target_hash,
        &[
            (op1, None),
            (op2, Some(&replacement)),
            (op3, None),
        ],
    );

    // Set up I/O
    let source = MemReader::new(source_data).into_boxed();
    let patch = MemReader::new(patch_data).into_boxed();
    let output = MemWriter::new().into_boxed();

    let mut buffer = [0u8; 2048];
    let mut patcher = StreamingPatcher::new();
    patcher.init(buffer.as_mut_ptr(), buffer.len());

    patcher.set_source(mem_read, source.as_ref() as *const _ as *mut u8);
    patcher.set_patch(mem_read, patch.as_ref() as *const _ as *mut u8);
    patcher.set_output(mem_write, output.as_ref() as *const _ as *mut u8);

    let result = run_patcher(&mut patcher);
    assert_eq!(result, 0, "Multiple operations should succeed");

    let output_data = output.borrow().data.clone();
    let expected = vec![
        0xAA, 0xAA, 0xAA, 0xAA, // Copied
        0xDD, 0xDD, 0xDD, 0xDD, // Replaced
        0xCC, 0xCC, 0xCC, 0xCC, // Copied
    ];
    assert_eq!(output_data, expected, "Output should have middle segment replaced");
}

#[test]
fn test_invalid_magic() {
    // Create patch with wrong magic
    let mut patch_data = build_patch([0u8; 32], [1u8; 32], &[]);
    // Corrupt magic
    patch_data[0] = 0xFF;
    patch_data[1] = 0xFF;

    let source = MemReader::new(vec![]).into_boxed();
    let patch = MemReader::new(patch_data).into_boxed();
    let output = MemWriter::new().into_boxed();

    let mut buffer = [0u8; 2048];
    let mut patcher = StreamingPatcher::new();
    patcher.init(buffer.as_mut_ptr(), buffer.len());

    patcher.set_source(mem_read, source.as_ref() as *const _ as *mut u8);
    patcher.set_patch(mem_read, patch.as_ref() as *const _ as *mut u8);
    patcher.set_output(mem_write, output.as_ref() as *const _ as *mut u8);

    let result = run_patcher(&mut patcher);
    assert!(result < 0, "Invalid magic should return error");
}

#[test]
fn test_hash_verification() {
    let source_hash = [0u8; 32];
    let target_hash = [1u8; 32];
    let patch_data = build_patch(source_hash, target_hash, &[]);

    let source = MemReader::new(vec![]).into_boxed();
    let patch = MemReader::new(patch_data).into_boxed();
    let output = MemWriter::new().into_boxed();

    let mut buffer = [0u8; 2048];
    let mut patcher = StreamingPatcher::new();
    patcher.init(buffer.as_mut_ptr(), buffer.len());

    patcher.set_source(mem_read, source.as_ref() as *const _ as *mut u8);
    patcher.set_patch(mem_read, patch.as_ref() as *const _ as *mut u8);
    patcher.set_output(mem_write, output.as_ref() as *const _ as *mut u8);

    let result = run_patcher(&mut patcher);
    assert_eq!(result, 0, "Patch should complete successfully");

    // Correct hash should verify
    assert!(patcher.verify(&target_hash), "Correct hash should verify");

    // Different hash should fail
    let different_hash = [2u8; 32];
    assert!(!patcher.verify(&different_hash), "Different hash should fail");

    // All zeros should fail
    let zero_hash = [0u8; 32];
    assert!(!patcher.verify(&zero_hash), "Zero hash should fail");
}

#[test]
fn test_abort() {
    let mut patcher = StreamingPatcher::new();
    let mut buffer = [0u8; 2048];
    patcher.init(buffer.as_mut_ptr(), buffer.len());

    // Abort before completion
    patcher.abort();

    assert_eq!(patcher.last_error(), -5, "Abort should set error code -5");
}

#[test]
fn test_patcher_size_for_c() {
    // Verify patcher struct size is reasonable for stack allocation
    let size = std::mem::size_of::<StreamingPatcher>();
    assert!(size < 512, "Patcher struct should be small enough for embedded stack");
    println!("StreamingPatcher size: {} bytes", size);
}
