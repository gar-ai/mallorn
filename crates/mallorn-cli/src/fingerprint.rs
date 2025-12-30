//! `mallorn fingerprint` command implementation

use anyhow::{Context, Result};
use mallorn_core::fingerprint::ModelFingerprint;
use std::fs;
use std::path::Path;

/// Generate a fingerprint for a model file
pub fn run(model: &Path, json_output: bool, _db_path: Option<&Path>) -> Result<()> {
    // Generate fingerprint from file
    let fingerprint =
        ModelFingerprint::from_file(model).context("Failed to generate fingerprint")?;

    if json_output {
        // JSON output for scripting
        let json = serde_json::json!({
            "format": fingerprint.format,
            "header_hash": hex::encode(fingerprint.header_hash),
            "tail_hash": hex::encode(fingerprint.tail_hash),
            "file_size": fingerprint.file_size,
            "tensor_count": fingerprint.tensor_count,
            "metadata_version": fingerprint.metadata_version,
            "compact": fingerprint.to_compact_string(),
        });
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        // Human-readable output
        println!("Model Fingerprint");
        println!("=================");
        println!();
        println!("File:           {}", model.display());
        println!("Format:         {}", fingerprint.format);
        println!("File size:      {} bytes", fingerprint.file_size);
        if let Some(count) = fingerprint.tensor_count {
            println!("Tensor count:   {}", count);
        }
        println!("Header hash:    {}", hex::encode(fingerprint.header_hash));
        println!("Tail hash:      {}", hex::encode(fingerprint.tail_hash));

        if let Some(ref version) = fingerprint.metadata_version {
            println!("Version:        {}", version);
        }

        println!();
        println!("Compact ID:     {}", fingerprint.to_compact_string());
    }

    Ok(())
}

/// Compare two model files by fingerprint
pub fn compare(model1: &Path, model2: &Path) -> Result<()> {
    let fp1 = ModelFingerprint::from_file(model1).context("Failed to fingerprint first model")?;
    let fp2 = ModelFingerprint::from_file(model2).context("Failed to fingerprint second model")?;

    println!("Fingerprint Comparison");
    println!("======================");
    println!();

    let matches = fp1.matches(&fp2);

    println!("Model 1:  {}", model1.display());
    println!("  Compact: {}", fp1.to_compact_string());
    println!();
    println!("Model 2:  {}", model2.display());
    println!("  Compact: {}", fp2.to_compact_string());
    println!();

    if matches {
        println!("Result:   MATCH (models are likely identical)");
    } else {
        println!("Result:   DIFFERENT");

        // Show what differs
        if fp1.format != fp2.format {
            println!("  - Format differs: {} vs {}", fp1.format, fp2.format);
        }
        if fp1.file_size != fp2.file_size {
            println!(
                "  - Size differs: {} vs {} bytes",
                fp1.file_size, fp2.file_size
            );
        }
        if fp1.tensor_count != fp2.tensor_count {
            println!(
                "  - Tensor count differs: {:?} vs {:?}",
                fp1.tensor_count, fp2.tensor_count
            );
        }
        if fp1.header_hash != fp2.header_hash {
            println!("  - Header content differs");
        }
        if fp1.tail_hash != fp2.tail_hash {
            println!("  - Tail content differs");
        }
    }

    Ok(())
}

/// Add a fingerprint to a JSON database file
pub fn add_to_db(model: &Path, db_path: &Path, version: &str) -> Result<()> {
    let fingerprint =
        ModelFingerprint::from_file(model).context("Failed to generate fingerprint")?;

    // Load existing database or create new one
    let mut entries: Vec<serde_json::Value> = if db_path.exists() {
        let data = fs::read_to_string(db_path).context("Failed to read database")?;
        serde_json::from_str(&data).unwrap_or_else(|_| Vec::new())
    } else {
        Vec::new()
    };

    // Add new entry
    entries.push(serde_json::json!({
        "version": version,
        "format": fingerprint.format,
        "file_size": fingerprint.file_size,
        "header_hash": hex::encode(fingerprint.header_hash),
        "tail_hash": hex::encode(fingerprint.tail_hash),
        "tensor_count": fingerprint.tensor_count,
        "compact": fingerprint.to_compact_string(),
    }));

    // Save database
    let data = serde_json::to_string_pretty(&entries).context("Failed to serialize database")?;
    fs::write(db_path, data).context("Failed to write database")?;

    println!("Added fingerprint to database:");
    println!("  Model: {}", model.display());
    println!("  Version: {}", version);
    println!("  Database: {}", db_path.display());
    println!("  Compact: {}", fingerprint.to_compact_string());

    Ok(())
}
