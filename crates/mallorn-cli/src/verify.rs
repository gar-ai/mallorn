//! `mallorn verify` command implementation

use anyhow::{Context, Result};
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

pub fn run(model: &Path, patch: &Path) -> Result<()> {
    println!("Verifying patch...");
    println!("  Model: {}", model.display());
    println!("  Patch: {}", patch.display());

    // Read files
    let model_data = fs::read(model).context("Failed to read model file")?;
    let patch_data = fs::read(patch).context("Failed to read patch file")?;

    // Detect patch format
    let format = detect_patch_format(&patch_data)?;
    println!("  Format: {:?}", format);

    // Verify patch based on format
    let verification = match format {
        PatchFormat::TFLite => {
            let patch = mallorn_tflite::deserialize_patch(&patch_data)
                .context("Failed to parse TFLite patch")?;
            let patcher = mallorn_tflite::TFLitePatcher::new();
            patcher
                .verify(&model_data, &patch)
                .context("Failed to verify TFLite patch")?
        }
        PatchFormat::GGUF => {
            let patch = mallorn_gguf::deserialize_patch(&patch_data)
                .context("Failed to parse GGUF patch")?;
            let patcher = mallorn_gguf::GGUFPatcher::new();
            patcher
                .verify(&model_data, &patch)
                .context("Failed to verify GGUF patch")?
        }
        PatchFormat::SafeTensors => {
            let patch = mallorn_safetensors::deserialize_patch(&patch_data)
                .context("Failed to parse SafeTensors patch")?;
            let patcher = mallorn_safetensors::SafeTensorsPatcher::new();
            patcher
                .verify(&model_data, &patch)
                .context("Failed to verify SafeTensors patch")?
        }
        PatchFormat::OpenVINO => {
            let patch = mallorn_openvino::deserialize_patch(&patch_data)
                .context("Failed to parse OpenVINO patch")?;
            let patcher = mallorn_openvino::OpenVINOPatcher::new();
            patcher
                .verify(&model_data, &patch)
                .context("Failed to verify OpenVINO patch")?
        }
        PatchFormat::CoreML => {
            let patch = mallorn_coreml::deserialize_patch(&patch_data)
                .context("Failed to parse CoreML patch")?;
            let patcher = mallorn_coreml::CoreMLPatcher::new();
            patcher
                .verify(&model_data, &patch)
                .context("Failed to verify CoreML patch")?
        }
    };

    println!();

    // Check source hash
    if verification.source_valid {
        println!("Source hash: VALID");
    } else {
        println!("Source hash: INVALID");
        println!("  The model file does not match the patch's source hash.");
        println!("  This patch was created for a different model version.");
        anyhow::bail!("Source hash mismatch");
    }

    // Check patch structure
    if verification.patch_valid {
        println!("Patch structure: VALID");
    } else {
        println!("Patch structure: INVALID");
        println!("  The patch file appears to be corrupted or malformed.");
        anyhow::bail!("Invalid patch structure");
    }

    // Print statistics
    println!();
    println!("Patch statistics:");
    println!("  Source size:      {:>10} bytes", verification.stats.source_size);
    println!("  Patch size:       {:>10} bytes", verification.stats.patch_size);
    println!(
        "  Compression:      {:>10.1}x",
        verification.stats.compression_ratio
    );
    println!(
        "  Tensors modified: {:>10}",
        verification.stats.tensors_modified
    );
    println!(
        "  Tensors unchanged:{:>10}",
        verification.stats.tensors_unchanged
    );

    // Print hashes
    println!();
    println!("Expected target hash: {}", hex::encode(verification.expected_target));

    println!();
    println!("Verification: PASSED");
    println!("  This patch can be safely applied to the model.");

    Ok(())
}
