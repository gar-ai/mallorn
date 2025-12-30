//! Convert standard patches to streaming format for embedded devices

use anyhow::{bail, Context, Result};
use mallorn_core::{CompressionMethod, Patch, PatchOperation};
use mallorn_lite::{
    CompressionType, OpType, PatchFooter, PatchHeader, TensorOp, PATCH_MAGIC, PATCH_VERSION,
};
use std::fs;
use std::path::Path;

/// Convert a standard patch file to streaming format
pub fn run(input: &Path, output: &Path) -> Result<()> {
    println!("Converting patch to streaming format...");

    // Read input patch file
    let data = fs::read(input).context("Failed to read input patch file")?;

    // Detect format and deserialize
    let patch = detect_and_deserialize(&data)?;

    println!("  Source: {}", input.display());
    println!("  Target: {}", output.display());
    println!("  Operations: {}", patch.operations.len());

    // Convert to streaming format
    let streaming_data = convert_to_streaming(&patch)?;

    // Write output
    fs::write(output, &streaming_data).context("Failed to write output file")?;

    let ratio = streaming_data.len() as f64 / data.len() as f64 * 100.0;
    println!("  Input size:  {} bytes", data.len());
    println!(
        "  Output size: {} bytes ({:.1}%)",
        streaming_data.len(),
        ratio
    );
    println!("Conversion complete!");

    Ok(())
}

/// Detect patch format from magic bytes and deserialize
fn detect_and_deserialize(data: &[u8]) -> Result<Patch> {
    if data.len() < 4 {
        bail!("File too small to be a valid patch");
    }

    let magic = &data[0..4];

    if magic == b"TFLP" {
        mallorn_tflite::deserialize_patch(data)
            .map_err(|e| anyhow::anyhow!("Failed to parse TFLite patch: {:?}", e))
    } else if magic == b"GGUP" {
        mallorn_gguf::deserialize_patch(data)
            .map_err(|e| anyhow::anyhow!("Failed to parse GGUF patch: {:?}", e))
    } else {
        bail!(
            "Unknown patch format (magic: {:02x} {:02x} {:02x} {:02x})",
            magic[0],
            magic[1],
            magic[2],
            magic[3]
        )
    }
}

/// Convert a Patch to streaming format bytes
fn convert_to_streaming(patch: &Patch) -> Result<Vec<u8>> {
    let mut output = Vec::new();

    // Calculate total uncompressed size
    let total_size: u32 = patch
        .operations
        .iter()
        .map(|op| match op {
            PatchOperation::ReplaceTensor { data, .. } => data.len() as u32,
            PatchOperation::DeltaTensor { delta, .. } => delta.len() as u32,
            PatchOperation::CopyTensor { .. } => 0,
            PatchOperation::UpdateMetadata { .. } => 0,
        })
        .sum();

    // Write header
    let header = PatchHeader {
        magic: PATCH_MAGIC,
        version: PATCH_VERSION,
        source_hash: patch.source_hash,
        tensor_count: patch.operations.len() as u32,
        total_size,
    };

    output.extend_from_slice(&header.magic.to_le_bytes());
    output.extend_from_slice(&header.version.to_le_bytes());
    output.extend_from_slice(&header.source_hash);
    output.extend_from_slice(&header.tensor_count.to_le_bytes());
    output.extend_from_slice(&header.total_size.to_le_bytes());

    // Track offset for operations
    let mut current_offset: u32 = 0;

    // Write each operation
    for op in &patch.operations {
        let (op_type, uncompressed_data, compression) = match op {
            PatchOperation::CopyTensor { .. } => {
                // Copy operation - no payload
                let tensor_op = TensorOp::copy(current_offset, 0);
                write_tensor_op(&mut output, &tensor_op);
                continue;
            }
            PatchOperation::ReplaceTensor { data, .. } => {
                (OpType::Replace, data.as_slice(), get_compression(patch))
            }
            PatchOperation::DeltaTensor { delta, .. } => {
                (OpType::Delta, delta.as_slice(), get_compression(patch))
            }
            PatchOperation::UpdateMetadata { .. } => {
                // Skip metadata updates in streaming format
                continue;
            }
        };

        // Compress with LZ4 for embedded compatibility
        let compressed = if compression == CompressionType::Lz4 {
            lz4_flex::compress_prepend_size(uncompressed_data)
        } else {
            uncompressed_data.to_vec()
        };

        // Write tensor operation header
        let tensor_op = TensorOp {
            op_type: op_type as u8,
            compression: compression as u8,
            reserved: 0,
            offset: current_offset,
            size: uncompressed_data.len() as u32,
            payload_size: compressed.len() as u32,
        };

        write_tensor_op(&mut output, &tensor_op);

        // Write compressed payload
        output.extend_from_slice(&compressed);

        current_offset += uncompressed_data.len() as u32;
    }

    // Calculate CRC32 of everything so far
    let crc = crc32fast::hash(&output);

    // Write footer
    let footer = PatchFooter {
        target_hash: patch.target_hash,
        crc32: crc,
    };

    output.extend_from_slice(&footer.target_hash);
    output.extend_from_slice(&footer.crc32.to_le_bytes());

    Ok(output)
}

/// Write a TensorOp to the output buffer
fn write_tensor_op(output: &mut Vec<u8>, op: &TensorOp) {
    output.push(op.op_type);
    output.push(op.compression);
    output.extend_from_slice(&op.reserved.to_le_bytes());
    output.extend_from_slice(&op.offset.to_le_bytes());
    output.extend_from_slice(&op.size.to_le_bytes());
    output.extend_from_slice(&op.payload_size.to_le_bytes());
}

/// Get compression type for streaming format
fn get_compression(patch: &Patch) -> CompressionType {
    // Always use LZ4 for streaming format (embedded-friendly)
    match patch.compression {
        CompressionMethod::None => CompressionType::None,
        _ => CompressionType::Lz4, // Convert Zstd/Neural to LZ4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_empty_patch() {
        let patch = Patch {
            version: 1,
            source_hash: [0u8; 32],
            target_hash: [1u8; 32],
            operations: vec![],
            compression: CompressionMethod::None,
            metadata: Default::default(),
        };

        let result = convert_to_streaming(&patch);
        assert!(result.is_ok());

        let data = result.unwrap();
        // Header (48) + Footer (36) = 84 bytes minimum
        assert!(data.len() >= 84);

        // Check magic
        assert_eq!(&data[0..4], &PATCH_MAGIC.to_le_bytes());
    }
}
