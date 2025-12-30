//! `mallorn dict` command implementation
//!
//! Provides dictionary training and management for improved compression.

use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use mallorn_core::{CompressionDictionary, DictionaryMetadata, DictionaryTrainer};
use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::Path;

/// Train a dictionary from model samples
pub fn train(samples: &[&Path], output: &Path, max_size: usize) -> Result<()> {
    println!("Training compression dictionary...");
    println!("  Samples: {} files", samples.len());
    println!("  Max size: {} bytes", max_size);

    if samples.is_empty() {
        anyhow::bail!("At least one sample file is required");
    }

    // Read all sample files
    let mut sample_data: Vec<Vec<u8>> = Vec::new();
    let mut total_size = 0usize;

    for (i, path) in samples.iter().enumerate() {
        let data = fs::read(path)
            .with_context(|| format!("Failed to read sample file: {}", path.display()))?;
        println!("  [{}] {} ({} bytes)", i + 1, path.display(), data.len());
        total_size += data.len();
        sample_data.push(data);
    }

    println!("\nTotal sample data: {} bytes", total_size);

    // Create trainer and train dictionary
    let trainer = DictionaryTrainer::with_max_size(max_size);
    let sample_refs: Vec<&[u8]> = sample_data.iter().map(|v| v.as_slice()).collect();

    let dictionary = trainer
        .train(&sample_refs)
        .context("Failed to train dictionary")?;

    println!("\nDictionary trained successfully:");
    println!("  ID: {}", dictionary.id);
    println!("  Size: {} bytes", dictionary.data.len());
    println!("  Samples: {}", dictionary.metadata.num_samples);

    // Save dictionary
    save_dictionary(&dictionary, output)?;
    println!("\nSaved to: {}", output.display());

    Ok(())
}

/// Show information about a dictionary file
pub fn info(path: &Path) -> Result<()> {
    let dictionary = load_dictionary(path)?;

    println!("Dictionary Information");
    println!("======================");
    println!();
    println!("File:         {}", path.display());
    println!("Version:      {}", dictionary.version);
    println!("ID:           {}", dictionary.id);
    println!("Size:         {} bytes", dictionary.data.len());
    println!();
    println!("Metadata:");
    println!("  Samples:     {}", dictionary.metadata.num_samples);
    println!("  Total bytes: {}", dictionary.metadata.total_sample_bytes);
    if let Some(ref desc) = dictionary.metadata.description {
        println!("  Description: {}", desc);
    }
    if dictionary.metadata.created_at > 0 {
        println!("  Created at:  {} (Unix timestamp)", dictionary.metadata.created_at);
    }

    Ok(())
}

/// Save dictionary to file with magic header
pub fn save_dictionary(dict: &CompressionDictionary, path: &Path) -> Result<()> {
    let mut file = fs::File::create(path)
        .with_context(|| format!("Failed to create dictionary file: {}", path.display()))?;

    // Magic bytes: "MLRD" (Mallorn Dictionary)
    file.write_all(b"MLRD")?;

    // Version (u32 LE)
    file.write_u32::<LittleEndian>(dict.version)?;

    // Dictionary ID (u32 LE)
    file.write_u32::<LittleEndian>(dict.id)?;

    // Metadata: num_samples (u32 LE)
    file.write_u32::<LittleEndian>(dict.metadata.num_samples as u32)?;

    // Metadata: total_sample_bytes (u64 LE)
    file.write_u64::<LittleEndian>(dict.metadata.total_sample_bytes as u64)?;

    // Metadata: created_at (u64 LE)
    file.write_u64::<LittleEndian>(dict.metadata.created_at)?;

    // Metadata: description length + data
    if let Some(ref desc) = dict.metadata.description {
        let bytes = desc.as_bytes();
        file.write_u32::<LittleEndian>(bytes.len() as u32)?;
        file.write_all(bytes)?;
    } else {
        file.write_u32::<LittleEndian>(0)?;
    }

    // Dictionary data length + data
    file.write_u64::<LittleEndian>(dict.data.len() as u64)?;
    file.write_all(&dict.data)?;

    Ok(())
}

/// Load dictionary from file
pub fn load_dictionary(path: &Path) -> Result<CompressionDictionary> {
    let data = fs::read(path)
        .with_context(|| format!("Failed to read dictionary file: {}", path.display()))?;

    let mut cursor = Cursor::new(&data);

    // Check magic
    let mut magic = [0u8; 4];
    cursor.read_exact(&mut magic)?;
    if &magic != b"MLRD" {
        anyhow::bail!("Invalid dictionary file: bad magic bytes");
    }

    // Read version
    let version = cursor.read_u32::<LittleEndian>()?;

    // Read dictionary ID
    let id = cursor.read_u32::<LittleEndian>()?;

    // Read metadata
    let num_samples = cursor.read_u32::<LittleEndian>()? as usize;
    let total_sample_bytes = cursor.read_u64::<LittleEndian>()? as usize;
    let created_at = cursor.read_u64::<LittleEndian>()?;

    let desc_len = cursor.read_u32::<LittleEndian>()? as usize;
    let description = if desc_len > 0 {
        let mut desc_bytes = vec![0u8; desc_len];
        cursor.read_exact(&mut desc_bytes)?;
        Some(String::from_utf8(desc_bytes)?)
    } else {
        None
    };

    // Read dictionary data
    let dict_len = cursor.read_u64::<LittleEndian>()? as usize;
    let mut dict_data = vec![0u8; dict_len];
    cursor.read_exact(&mut dict_data)?;

    Ok(CompressionDictionary {
        version,
        id,
        data: dict_data,
        metadata: DictionaryMetadata {
            num_samples,
            total_sample_bytes,
            description,
            created_at,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_dictionary_roundtrip() {
        let dict = CompressionDictionary {
            version: 1,
            id: 12345,
            data: vec![1, 2, 3, 4, 5],
            metadata: DictionaryMetadata {
                num_samples: 3,
                total_sample_bytes: 1000,
                description: Some("test dictionary".to_string()),
                created_at: 1234567890,
            },
        };

        let temp = TempDir::new().unwrap();
        let path = temp.path().join("test.dict");

        save_dictionary(&dict, &path).unwrap();
        let loaded = load_dictionary(&path).unwrap();

        assert_eq!(loaded.version, dict.version);
        assert_eq!(loaded.id, dict.id);
        assert_eq!(loaded.data, dict.data);
        assert_eq!(loaded.metadata.num_samples, dict.metadata.num_samples);
        assert_eq!(loaded.metadata.description, dict.metadata.description);
        assert_eq!(loaded.metadata.created_at, dict.metadata.created_at);
    }
}
