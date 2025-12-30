//! `mallorn diff` command implementation

use crate::dict::load_dictionary;
use anyhow::{Context, Result};
use mallorn_core::{CompressionMethod, DiffOptions};
use std::fs;
use std::path::Path;

/// Detect model format from file extension or magic bytes
fn detect_format(path: &Path, data: &[u8]) -> Result<ModelFormat> {
    // Check by magic bytes first
    if data.len() >= 8 {
        // TFLite magic
        let offset = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        if offset < data.len() && data.get(offset..offset + 4) == Some(b"TFL3") {
            return Ok(ModelFormat::TFLite);
        }
        // GGUF magic
        if &data[0..4] == b"GGUF"
            || u32::from_le_bytes([data[0], data[1], data[2], data[3]]) == 0x46554747
        {
            return Ok(ModelFormat::GGUF);
        }
        // SafeTensors: JSON header starting with '{' after 8-byte size prefix
        let header_size = u64::from_le_bytes(data[0..8].try_into().unwrap_or([0; 8])) as usize;
        if header_size > 0 && header_size < data.len() && data.get(8) == Some(&b'{') {
            return Ok(ModelFormat::SafeTensors);
        }
    }

    // Check for XML-based formats (OpenVINO)
    if data.len() >= 5 && &data[0..5] == b"<?xml" {
        return Ok(ModelFormat::OpenVINO);
    }

    // Fall back to extension
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase());

    match ext.as_deref() {
        Some("tflite") => Ok(ModelFormat::TFLite),
        Some("gguf") => Ok(ModelFormat::GGUF),
        Some("onnx") => Ok(ModelFormat::ONNX),
        Some("safetensors") => Ok(ModelFormat::SafeTensors),
        Some("xml") => Ok(ModelFormat::OpenVINO),
        Some("mlpackage") | Some("mlmodelc") => Ok(ModelFormat::CoreML),
        _ => anyhow::bail!(
            "Unknown model format. Supported: .tflite, .gguf, .onnx, .safetensors, .xml (OpenVINO), .mlpackage/.mlmodelc (CoreML)"
        ),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
enum ModelFormat {
    TFLite,
    GGUF,
    ONNX,
    SafeTensors,
    OpenVINO,
    CoreML,
}

pub fn run(
    old: &Path,
    new: &Path,
    output: &Path,
    level: i32,
    neural: bool,
    dict_path: Option<&Path>,
    parallel: bool,
) -> Result<()> {
    println!("Creating patch...");
    println!("  Source: {}", old.display());
    println!("  Target: {}", new.display());
    println!("  Output: {}", output.display());
    if parallel {
        println!("  Parallel: enabled");
    }

    // Read model files
    let old_data = fs::read(old).context("Failed to read source model")?;
    let new_data = fs::read(new).context("Failed to read target model")?;

    // Detect format
    let old_format = detect_format(old, &old_data)?;
    let new_format = detect_format(new, &new_data)?;

    if old_format != new_format {
        anyhow::bail!("Source and target models must be the same format");
    }

    // Load dictionary if provided
    let dictionary = if let Some(path) = dict_path {
        let dict = load_dictionary(path).context("Failed to load dictionary")?;
        println!("  Dictionary: {} (ID {})", path.display(), dict.id);
        Some(dict)
    } else {
        None
    };

    println!("  Format: {:?}", old_format);
    println!("  Compression level: {}", level);
    if neural {
        println!("  Neural compression: enabled");
    }

    // Create diff options with appropriate compression method
    let compression = if let Some(ref dict) = dictionary {
        CompressionMethod::ZstdDict {
            level,
            dict_id: dict.id,
        }
    } else {
        CompressionMethod::Zstd { level }
    };

    let options = DiffOptions {
        compression,
        min_tensor_size: 1024,
        neural_compression: neural,
        target_size_hint: None,
        dictionary,
    };

    // Create patch based on format
    let (patch, patch_bytes) = match old_format {
        ModelFormat::TFLite => {
            let differ =
                mallorn_tflite::TFLiteDiffer::with_options(options).with_parallel(parallel);
            let patch = differ
                .diff_from_bytes(&old_data, &new_data)
                .context("Failed to compute diff")?;
            let bytes =
                mallorn_tflite::serialize_patch(&patch).context("Failed to serialize patch")?;
            (patch, bytes)
        }
        ModelFormat::GGUF => {
            let differ = mallorn_gguf::GGUFDiffer::with_options(options).with_parallel(parallel);
            let patch = differ
                .diff_from_bytes(&old_data, &new_data)
                .context("Failed to compute diff")?;
            let bytes =
                mallorn_gguf::serialize_patch(&patch).context("Failed to serialize patch")?;
            (patch, bytes)
        }
        ModelFormat::SafeTensors => {
            let differ = mallorn_safetensors::SafeTensorsDiffer::with_options(options);
            let patch = differ
                .diff_from_bytes(&old_data, &new_data)
                .context("Failed to compute diff")?;
            let bytes = mallorn_safetensors::serialize_patch(&patch)
                .context("Failed to serialize patch")?;
            (patch, bytes)
        }
        ModelFormat::OpenVINO => {
            let differ = mallorn_openvino::OpenVINODiffer::with_options(options);
            let patch = differ
                .diff_from_bytes(&old_data, &new_data)
                .context("Failed to compute diff")?;
            let bytes =
                mallorn_openvino::serialize_patch(&patch).context("Failed to serialize patch")?;
            (patch, bytes)
        }
        ModelFormat::CoreML => {
            let differ = mallorn_coreml::CoreMLDiffer::with_options(options);
            let patch = differ
                .diff_from_bytes(&old_data, &new_data)
                .context("Failed to compute diff")?;
            let bytes =
                mallorn_coreml::serialize_patch(&patch).context("Failed to serialize patch")?;
            (patch, bytes)
        }
        ModelFormat::ONNX => {
            // Check if output is .trtp (TensorRT workflow)
            let output_ext = output
                .extension()
                .and_then(|e| e.to_str())
                .map(|s| s.to_lowercase());

            if output_ext.as_deref() == Some("trtp") {
                // TensorRT workflow: ONNX diff with TensorRT config
                let differ = mallorn_tensorrt::TensorRTDiffer::with_options(options);
                let config = mallorn_tensorrt::TensorRTConfig::default();
                let patch = differ
                    .diff(&old_data, &new_data, config)
                    .context("Failed to compute diff")?;
                let bytes = mallorn_tensorrt::serialize_patch(&patch)
                    .context("Failed to serialize patch")?;

                // Write patch
                fs::write(output, &bytes).context("Failed to write patch file")?;

                // Print TensorRT-specific info
                let source_size = old_data.len();
                let patch_size = bytes.len();
                let compression_ratio = source_size as f64 / patch_size as f64;
                let savings = 100.0 * (1.0 - patch_size as f64 / source_size as f64);

                println!();
                println!("TensorRT patch created successfully!");
                println!("  Source size:  {:>12} bytes", source_size);
                println!("  Target size:  {:>12} bytes", new_data.len());
                println!("  Patch size:   {:>12} bytes", patch_size);
                println!("  Compression:  {:>12.1}x", compression_ratio);
                println!("  Savings:      {:>12.1}%", savings);
                println!(
                    "  Tensors:      {:>12} modified, {} unchanged",
                    patch.onnx_patch.modified_count(),
                    patch.onnx_patch.unchanged_count()
                );
                println!();
                println!("TensorRT Workflow:");
                println!("  1. Apply patch: mallorn patch model.onnx update.trtp -o new.onnx");
                println!(
                    "  2. Rebuild engine: {}",
                    patch.config.to_trtexec_command("new.onnx", "new.engine")
                );
                return Ok(());
            }

            // Regular ONNX patch
            let differ = mallorn_onnx::ONNXDiffer::with_options(options).with_parallel(parallel);
            let patch = differ
                .diff_from_bytes(&old_data, &new_data)
                .context("Failed to compute diff")?;
            let bytes =
                mallorn_onnx::serialize_patch(&patch).context("Failed to serialize patch")?;
            (patch, bytes)
        }
    };

    // Write patch file
    fs::write(output, &patch_bytes).context("Failed to write patch file")?;

    // Print statistics
    let source_size = old_data.len();
    let target_size = new_data.len();
    let patch_size = patch_bytes.len();
    let compression_ratio = source_size as f64 / patch_size as f64;
    let savings = 100.0 * (1.0 - patch_size as f64 / source_size as f64);

    println!();
    println!("Patch created successfully!");
    println!("  Source size:  {:>12} bytes", source_size);
    println!("  Target size:  {:>12} bytes", target_size);
    println!("  Patch size:   {:>12} bytes", patch_size);
    println!("  Compression:  {:>12.1}x", compression_ratio);
    println!("  Savings:      {:>12.1}%", savings);
    println!(
        "  Tensors:      {:>12} modified, {} unchanged",
        patch.modified_count(),
        patch.unchanged_count()
    );

    Ok(())
}
