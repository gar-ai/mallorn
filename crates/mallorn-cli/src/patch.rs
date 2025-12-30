//! `mallorn patch` command implementation

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use mallorn_core::streaming::{StreamConfig, StreamProgress, StreamingPatcher, TensorIndex, TensorLocation};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Detect patch format from magic bytes
fn detect_patch_format(data: &[u8]) -> Result<PatchFormat> {
    if data.len() < 4 {
        anyhow::bail!("File too small to be a valid patch");
    }

    if mallorn_tflite::is_tflp(data) {
        return Ok(PatchFormat::TFLite);
    }

    if mallorn_gguf::is_ggup(data) {
        return Ok(PatchFormat::GGUF);
    }

    if mallorn_safetensors::is_sftp(data) {
        return Ok(PatchFormat::SafeTensors);
    }

    if mallorn_openvino::is_ovinp(data) {
        return Ok(PatchFormat::OpenVINO);
    }

    if mallorn_coreml::is_cmlp(data) {
        return Ok(PatchFormat::CoreML);
    }

    if mallorn_tensorrt::format::is_trtp(data) {
        return Ok(PatchFormat::TensorRT);
    }

    if mallorn_onnx::is_onxp(data) {
        return Ok(PatchFormat::ONNX);
    }

    anyhow::bail!("Unknown patch format. Expected .tflp, .ggup, .onxp, .sftp, .ovinp, .cmlp, or .trtp file.");
}

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
enum PatchFormat {
    TFLite,
    GGUF,
    ONNX,
    SafeTensors,
    OpenVINO,
    CoreML,
    TensorRT,
}

pub fn run(model: &Path, patch: &Path, output: &Path, streaming: bool, buffer_size: usize) -> Result<()> {
    println!("Applying patch...");
    println!("  Model: {}", model.display());
    println!("  Patch: {}", patch.display());
    println!("  Output: {}", output.display());
    if streaming {
        println!("  Mode: streaming (buffer: {} MB)", buffer_size / (1024 * 1024));
    }

    // Streaming mode for large models
    if streaming {
        return run_streaming(model, patch, output, buffer_size);
    }

    // Read files
    let model_data = fs::read(model).context("Failed to read model file")?;
    let patch_data = fs::read(patch).context("Failed to read patch file")?;

    // Detect patch format
    let format = detect_patch_format(&patch_data)?;
    println!("  Format: {:?}", format);

    // Apply patch based on format
    let (new_model_data, verification) = match format {
        PatchFormat::TFLite => {
            let patch = mallorn_tflite::deserialize_patch(&patch_data)
                .context("Failed to parse TFLite patch")?;
            let patcher = mallorn_tflite::TFLitePatcher::new();
            patcher
                .apply_and_verify(&model_data, &patch)
                .context("Failed to apply TFLite patch")?
        }
        PatchFormat::GGUF => {
            let patch = mallorn_gguf::deserialize_patch(&patch_data)
                .context("Failed to parse GGUF patch")?;
            let patcher = mallorn_gguf::GGUFPatcher::new();
            patcher
                .apply_and_verify(&model_data, &patch)
                .context("Failed to apply GGUF patch")?
        }
        PatchFormat::SafeTensors => {
            let patch = mallorn_safetensors::deserialize_patch(&patch_data)
                .context("Failed to parse SafeTensors patch")?;
            let patcher = mallorn_safetensors::SafeTensorsPatcher::new();
            patcher
                .apply_and_verify(&model_data, &patch)
                .context("Failed to apply SafeTensors patch")?
        }
        PatchFormat::OpenVINO => {
            let patch = mallorn_openvino::deserialize_patch(&patch_data)
                .context("Failed to parse OpenVINO patch")?;
            let patcher = mallorn_openvino::OpenVINOPatcher::new();
            patcher
                .apply_and_verify(&model_data, &patch)
                .context("Failed to apply OpenVINO patch")?
        }
        PatchFormat::CoreML => {
            let patch = mallorn_coreml::deserialize_patch(&patch_data)
                .context("Failed to parse CoreML patch")?;
            let patcher = mallorn_coreml::CoreMLPatcher::new();
            patcher
                .apply_and_verify(&model_data, &patch)
                .context("Failed to apply CoreML patch")?
        }
        PatchFormat::ONNX => {
            let patch = mallorn_onnx::deserialize_patch(&patch_data)
                .context("Failed to parse ONNX patch")?;
            let patcher = mallorn_onnx::ONNXPatcher::new();
            patcher
                .apply_and_verify(&model_data, &patch)
                .context("Failed to apply ONNX patch")?
        }
        PatchFormat::TensorRT => {
            // TensorRT patches produce ONNX output + rebuild instructions
            let patch = mallorn_tensorrt::deserialize_patch(&patch_data)
                .context("Failed to parse TensorRT patch")?;
            let patcher = mallorn_tensorrt::TensorRTPatcher::new();
            let (result, verification) = patcher
                .apply_and_verify(&model_data, &patch)
                .context("Failed to apply TensorRT patch")?;

            // Write output ONNX model
            fs::write(output, &result.onnx_data).context("Failed to write output model")?;

            // Print verification results with TensorRT-specific info
            println!();
            println!("TensorRT patch applied successfully!");
            println!("  Input size:   {:>12} bytes (ONNX)", model_data.len());
            println!("  Output size:  {:>12} bytes (ONNX)", result.onnx_data.len());
            println!(
                "  Tensors:      {:>12} modified, {} unchanged",
                verification.stats.tensors_modified, verification.stats.tensors_unchanged
            );

            if verification.source_valid {
                println!("  Source hash:  verified");
            } else {
                println!("  Source hash:  MISMATCH (WARNING)");
            }

            if let Some(actual) = verification.actual_target {
                if actual == verification.expected_target {
                    println!("  Target hash:  verified");
                } else {
                    println!("  Target hash:  MISMATCH (WARNING)");
                }
            }

            // Print rebuild instructions
            println!();
            println!("Next Steps - Rebuild TensorRT Engine:");
            println!("  {}", result.rebuild_command.replace("model.onnx", &output.display().to_string()).replace("model.engine", &output.with_extension("engine").display().to_string()));
            println!();
            println!("TensorRT Configuration:");
            println!("  Precision:    {:?}", result.config.precision);
            println!("  Workspace:    {} MB", result.config.workspace_size_mb);
            println!("  Max Batch:    {}", result.config.max_batch_size);
            if let Some(dla) = result.config.dla_core {
                println!("  DLA Core:     {}", dla);
            }

            return Ok(());
        }
    };

    // Write output model
    fs::write(output, &new_model_data).context("Failed to write output model")?;

    // Print verification results
    println!();
    println!("Patch applied successfully!");
    println!("  Source size:  {:>12} bytes", verification.stats.source_size);
    println!("  Output size:  {:>12} bytes", verification.stats.target_size);
    println!(
        "  Tensors:      {:>12} modified, {} unchanged",
        verification.stats.tensors_modified, verification.stats.tensors_unchanged
    );

    if verification.source_valid {
        println!("  Source hash:  verified");
    } else {
        println!("  Source hash:  MISMATCH (WARNING)");
    }

    if let Some(actual) = verification.actual_target {
        if actual == verification.expected_target {
            println!("  Target hash:  verified");
        } else {
            println!("  Target hash:  MISMATCH (WARNING)");
        }
    }

    Ok(())
}

/// Progress reporter using indicatif progress bar
struct CliProgress {
    pb: ProgressBar,
    total: u64,
}

impl CliProgress {
    fn new(total: u64) -> Self {
        let pb = ProgressBar::new(total);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({percent}%)")
                .unwrap()
                .progress_chars("#>-"),
        );
        Self { pb, total }
    }
}

impl StreamProgress for CliProgress {
    fn on_tensor_start(&mut self, name: &str, _size: u64) {
        self.pb.set_message(format!("Processing: {}", name));
    }

    fn on_tensor_complete(&mut self, _name: &str, _size: u64) {
        // Progress is updated via on_progress
    }

    fn on_progress(&mut self, processed: u64, _total: u64) {
        self.pb.set_position(processed.min(self.total));
    }
}

/// Apply patch in streaming mode for large models
fn run_streaming(model: &Path, patch: &Path, output: &Path, buffer_size: usize) -> Result<()> {
    // Read patch file to get metadata and operations
    let patch_data = fs::read(patch).context("Failed to read patch file")?;
    let format = detect_patch_format(&patch_data)?;
    println!("  Format: {:?}", format);

    // Deserialize patch to get operations
    let core_patch = match format {
        PatchFormat::TFLite => {
            mallorn_tflite::deserialize_patch(&patch_data)
                .context("Failed to parse TFLite patch")?
        }
        PatchFormat::GGUF => {
            mallorn_gguf::deserialize_patch(&patch_data)
                .context("Failed to parse GGUF patch")?
        }
        PatchFormat::ONNX => {
            mallorn_onnx::deserialize_patch(&patch_data)
                .context("Failed to parse ONNX patch")?
        }
        PatchFormat::SafeTensors => {
            mallorn_safetensors::deserialize_patch(&patch_data)
                .context("Failed to parse SafeTensors patch")?
        }
        PatchFormat::OpenVINO => {
            mallorn_openvino::deserialize_patch(&patch_data)
                .context("Failed to parse OpenVINO patch")?
        }
        PatchFormat::CoreML => {
            mallorn_coreml::deserialize_patch(&patch_data)
                .context("Failed to parse CoreML patch")?
        }
        PatchFormat::TensorRT => {
            // TensorRT doesn't support streaming mode - use non-streaming
            println!();
            println!("TensorRT patches don't support streaming mode.");
            println!("Falling back to standard mode...");
            return run(model, patch, output, false, buffer_size);
        }
    };

    // Get model file size to estimate tensor layout
    let model_metadata = fs::metadata(model).context("Failed to get model file metadata")?;
    let model_size = model_metadata.len();

    // Build tensor index from patch operations
    // For streaming, we need to know where each tensor is in the source file
    // We'll create a simple index assuming tensors are contiguous
    let mut index = TensorIndex::new();
    let mut offset: u64 = 0;

    // Extract tensor names and estimate sizes from patch operations
    let tensor_count = core_patch.modified_count() + core_patch.unchanged_count();
    if tensor_count == 0 {
        // No tensors in patch - treat entire file as one tensor
        index.add(TensorLocation::new("data", 0, model_size));
    } else {
        // Estimate equal-sized tensors (simple heuristic)
        let estimated_tensor_size = model_size / tensor_count as u64;

        for (i, op) in core_patch.operations.iter().enumerate() {
            let name = match op {
                mallorn_core::PatchOperation::ReplaceTensor { name, .. } => name.clone(),
                mallorn_core::PatchOperation::DeltaTensor { name, .. } => name.clone(),
                mallorn_core::PatchOperation::CopyTensor { name } => name.clone(),
                mallorn_core::PatchOperation::UpdateMetadata { .. } => continue,
            };

            let size = if i == tensor_count - 1 {
                // Last tensor gets remaining bytes
                model_size - offset
            } else {
                estimated_tensor_size
            };

            index.add(TensorLocation::new(name, offset, size));
            offset += size;
        }
    }

    println!("  Tensors: {}", index.len());
    println!("  Total size: {} bytes", index.total_size());

    // Open source and output files
    let source_file = File::open(model).context("Failed to open model file")?;
    let source = BufReader::with_capacity(buffer_size, source_file);

    let output_file = File::create(output).context("Failed to create output file")?;
    let output_writer = BufWriter::with_capacity(buffer_size, output_file);

    // Configure streaming
    let config = StreamConfig::default()
        .with_buffer_size(buffer_size);

    // Create streaming patcher
    let mut patcher = StreamingPatcher::new(source, output_writer, core_patch, config)
        .map_err(|e| anyhow::anyhow!("Failed to create streaming patcher: {}", e))?;

    // Apply with progress
    let mut progress = CliProgress::new(index.total_size());
    let verification = patcher
        .apply_with_progress(&index, &mut progress)
        .map_err(|e| anyhow::anyhow!("Failed to apply patch: {}", e))?;

    progress.pb.finish_with_message("Complete");

    // Print verification results
    println!();
    println!("Streaming patch applied successfully!");
    println!("  Source size:  {:>12} bytes", verification.stats.source_size);
    println!("  Output size:  {:>12} bytes", verification.stats.target_size);
    println!(
        "  Tensors:      {:>12} modified, {} unchanged",
        verification.stats.tensors_modified, verification.stats.tensors_unchanged
    );
    println!("  Memory used:  {:>12} MB (buffer)", buffer_size / (1024 * 1024));

    if verification.source_valid {
        println!("  Source hash:  verified");
    } else {
        println!("  Source hash:  MISMATCH (WARNING)");
    }

    if let Some(actual) = verification.actual_target {
        if actual == verification.expected_target {
            println!("  Target hash:  verified");
        } else {
            println!("  Target hash:  MISMATCH (WARNING)");
        }
    }

    Ok(())
}
