//! `mallorn info` command implementation

use anyhow::{Context, Result};
use mallorn_core::{CompressionMethod, DeltaFormat, Patch, PatchOperation};
use std::fs;
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

    anyhow::bail!("Unknown patch format. Expected .tflp, .ggup, .sftp, .ovinp, or .cmlp file.");
}

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
enum PatchFormat {
    TFLite,
    GGUF,
    SafeTensors,
    OpenVINO,
    CoreML,
}

pub fn run(patch_path: &Path, verbose: bool) -> Result<()> {
    // Read patch file
    let patch_data = fs::read(patch_path).context("Failed to read patch file")?;
    let file_size = patch_data.len();

    // Detect format and parse
    let format = detect_patch_format(&patch_data)?;
    let patch: Patch = match format {
        PatchFormat::TFLite => mallorn_tflite::deserialize_patch(&patch_data)
            .context("Failed to parse TFLite patch")?,
        PatchFormat::GGUF => {
            mallorn_gguf::deserialize_patch(&patch_data).context("Failed to parse GGUF patch")?
        }
        PatchFormat::SafeTensors => mallorn_safetensors::deserialize_patch(&patch_data)
            .context("Failed to parse SafeTensors patch")?,
        PatchFormat::OpenVINO => mallorn_openvino::deserialize_patch(&patch_data)
            .context("Failed to parse OpenVINO patch")?,
        PatchFormat::CoreML => mallorn_coreml::deserialize_patch(&patch_data)
            .context("Failed to parse CoreML patch")?,
    };

    // Print basic info
    println!("Patch Information");
    println!("=================");
    println!();
    println!("File:       {}", patch_path.display());
    println!("Size:       {} bytes", file_size);
    println!(
        "Format:     {}",
        match format {
            PatchFormat::TFLite => "TFLite (.tflp)",
            PatchFormat::GGUF => "GGUF (.ggup)",
            PatchFormat::SafeTensors => "SafeTensors (.sftp)",
            PatchFormat::OpenVINO => "OpenVINO (.ovinp)",
            PatchFormat::CoreML => "CoreML (.cmlp)",
        }
    );
    println!("Version:    {}", patch.version);

    // Compression
    println!(
        "Compression: {}",
        match patch.compression {
            CompressionMethod::None => "None".to_string(),
            CompressionMethod::Zstd { level } => format!("Zstd (level {})", level),
            CompressionMethod::Lz4 => "LZ4".to_string(),
            CompressionMethod::Neural { .. } => "Neural".to_string(),
            CompressionMethod::Adaptive { strategy } => format!("Adaptive ({:?})", strategy),
            CompressionMethod::ZstdDict { level, dict_id } =>
                format!("Zstd+Dict (level {}, dict {})", level, dict_id),
        }
    );

    println!();
    println!("Hashes");
    println!("------");
    println!("Source: {}", hex::encode(patch.source_hash));
    println!("Target: {}", hex::encode(patch.target_hash));

    // Metadata
    if patch.metadata.source_version.is_some()
        || patch.metadata.target_version.is_some()
        || patch.metadata.description.is_some()
    {
        println!();
        println!("Metadata");
        println!("--------");
        if let Some(ref v) = patch.metadata.source_version {
            println!("Source version: {}", v);
        }
        if let Some(ref v) = patch.metadata.target_version {
            println!("Target version: {}", v);
        }
        if let Some(ref d) = patch.metadata.description {
            println!("Description:    {}", d);
        }
        if patch.metadata.created_at > 0 {
            // Format timestamp
            let ts = patch.metadata.created_at;
            println!("Created:        {} (Unix timestamp)", ts);
        }
    }

    // Operations summary
    println!();
    println!("Operations");
    println!("----------");

    let mut copy_count = 0;
    let mut replace_count = 0;
    let mut delta_count = 0;
    let mut metadata_count = 0;
    let mut total_delta_size = 0;
    let mut total_replace_size = 0;

    for op in &patch.operations {
        match op {
            PatchOperation::CopyTensor { .. } => copy_count += 1,
            PatchOperation::ReplaceTensor { data, .. } => {
                replace_count += 1;
                total_replace_size += data.len();
            }
            PatchOperation::DeltaTensor { delta, .. } => {
                delta_count += 1;
                total_delta_size += delta.len();
            }
            PatchOperation::UpdateMetadata { .. } => metadata_count += 1,
        }
    }

    println!("Total operations: {}", patch.operations.len());
    println!("  Copy (unchanged):   {}", copy_count);
    println!(
        "  Replace (full):     {} ({} bytes)",
        replace_count, total_replace_size
    );
    println!(
        "  Delta (diff):       {} ({} bytes)",
        delta_count, total_delta_size
    );
    if metadata_count > 0 {
        println!("  Metadata updates:   {}", metadata_count);
    }

    // Verbose: show each operation
    if verbose {
        println!();
        println!("Operation Details");
        println!("-----------------");

        for (i, op) in patch.operations.iter().enumerate() {
            match op {
                PatchOperation::CopyTensor { name } => {
                    println!("[{}] COPY: {}", i, name);
                }
                PatchOperation::ReplaceTensor { name, data, .. } => {
                    println!("[{}] REPLACE: {} ({} bytes)", i, name, data.len());
                }
                PatchOperation::DeltaTensor {
                    name,
                    delta,
                    delta_format,
                    ..
                } => {
                    let format_str = match delta_format {
                        DeltaFormat::Xor => "XOR",
                        DeltaFormat::BsDiff => "BSDiff",
                        DeltaFormat::TensorAware => "TensorAware",
                    };
                    println!(
                        "[{}] DELTA: {} ({} bytes, {})",
                        i,
                        name,
                        delta.len(),
                        format_str
                    );
                }
                PatchOperation::UpdateMetadata { key, value } => {
                    println!("[{}] METADATA: {} = {}", i, key, value);
                }
            }
        }
    }

    Ok(())
}
