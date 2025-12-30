//! Fuzz target for patch application
//!
//! Tests patch application robustness against malformed patches.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use mallorn_core::{CompressionMethod, DeltaFormat, Patch, PatchMetadata, PatchOperation};
use mallorn_tflite::TFLitePatcher;

/// Structured input for patch fuzzing
#[derive(Arbitrary, Debug)]
struct FuzzInput {
    /// Source model data
    source_data: Vec<u8>,
    /// Patch operations
    operations: Vec<FuzzOperation>,
    /// Version
    version: u32,
}

#[derive(Arbitrary, Debug)]
enum FuzzOperation {
    Copy { name_len: u8 },
    Replace { name_len: u8, data: Vec<u8> },
    Delta { name_len: u8, delta: Vec<u8> },
}

fn generate_name(len: u8) -> String {
    let actual_len = (len as usize % 64) + 1;
    "t".repeat(actual_len)
}

fuzz_target!(|input: FuzzInput| {
    // Build operations from fuzz input
    let operations: Vec<PatchOperation> = input
        .operations
        .iter()
        .take(16) // Limit operations
        .map(|op| match op {
            FuzzOperation::Copy { name_len } => PatchOperation::CopyTensor {
                name: generate_name(*name_len),
            },
            FuzzOperation::Replace { name_len, data } => PatchOperation::ReplaceTensor {
                name: generate_name(*name_len),
                data: data.clone(),
            },
            FuzzOperation::Delta { name_len, delta } => PatchOperation::DeltaTensor {
                name: generate_name(*name_len),
                delta: delta.clone(),
                delta_format: DeltaFormat::Xor,
            },
        })
        .collect();

    // Create patch structure
    let patch = Patch {
        version: input.version % 10,
        source_hash: [0u8; 32],
        target_hash: [0u8; 32],
        operations,
        compression: CompressionMethod::None,
        metadata: PatchMetadata::default(),
    };

    // Try to apply - should not panic
    let patcher = TFLitePatcher::new();
    let _ = patcher.apply(&input.source_data, &patch);
});
