//! `mallorn chain` command implementation

use anyhow::{Context, Result};
use mallorn_core::{
    deserialize_chain, is_chain, serialize_chain, ChainLink, ChainMetadata, LinkMetadata,
    PatchChain,
};
use std::fs;
use std::path::Path;

/// Create a new chain from an initial patch
pub fn create(patch: &Path, output: &Path, chain_id: Option<&str>) -> Result<()> {
    println!("Creating new patch chain...");
    println!("  Initial patch: {}", patch.display());
    println!("  Output: {}", output.display());

    // Read the patch file
    let patch_data = fs::read(patch).context("Failed to read patch file")?;

    // Detect patch format from magic bytes
    let format = detect_patch_format(&patch_data)?;
    println!("  Patch format: {}", format);

    // Parse the patch to extract hashes
    let (source_hash, target_hash) = extract_patch_hashes(&patch_data, &format)?;

    // Generate chain ID if not provided
    let chain_id = chain_id
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("chain-{}", hex::encode(&target_hash[..8])));

    // Create the chain with the initial link
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let chain = PatchChain {
        version: 1,
        chain_id: chain_id.clone(),
        links: vec![ChainLink {
            index: 0,
            source_hash,
            target_hash,
            patch_data: patch_data.clone(),
            format: format.clone(),
            link_metadata: LinkMetadata {
                source_version: None,
                target_version: None,
                patch_size: patch_data.len(),
                created_at: now,
            },
        }],
        metadata: ChainMetadata {
            base_version: None,
            head_version: None,
            total_links: 1,
            created_at: now,
            description: Some("Patch chain".into()),
        },
    };

    // Serialize and write the chain
    let chain_bytes = serialize_chain(&chain).context("Failed to serialize chain")?;
    fs::write(output, &chain_bytes).context("Failed to write chain file")?;

    println!();
    println!("Chain created successfully!");
    println!("  Chain ID: {}", chain_id);
    println!("  Links: 1");
    println!("  Size: {} bytes", chain_bytes.len());
    println!("  Base hash: {}", hex::encode(source_hash));
    println!("  Head hash: {}", hex::encode(target_hash));

    Ok(())
}

/// Append a patch to an existing chain
pub fn append(chain_path: &Path, patch: &Path, output: Option<&Path>) -> Result<()> {
    println!("Appending patch to chain...");
    println!("  Chain: {}", chain_path.display());
    println!("  Patch: {}", patch.display());

    // Read existing chain
    let chain_data = fs::read(chain_path).context("Failed to read chain file")?;
    if !is_chain(&chain_data) {
        anyhow::bail!("File is not a valid patch chain");
    }
    let mut chain = deserialize_chain(&chain_data).context("Failed to parse chain")?;

    // Read the new patch
    let patch_data = fs::read(patch).context("Failed to read patch file")?;
    let format = detect_patch_format(&patch_data)?;
    let (source_hash, target_hash) = extract_patch_hashes(&patch_data, &format)?;

    // Verify the new patch connects to the chain head
    let chain_head = chain
        .head_hash()
        .ok_or_else(|| anyhow::anyhow!("Chain is empty"))?;
    if source_hash != chain_head {
        anyhow::bail!(
            "Patch source hash does not match chain head.\n  Chain head: {}\n  Patch source: {}",
            hex::encode(chain_head),
            hex::encode(source_hash)
        );
    }

    // Create the new link
    let new_link = ChainLink {
        index: chain.links.len() as u32,
        source_hash,
        target_hash,
        patch_data: patch_data.clone(),
        format,
        link_metadata: LinkMetadata {
            source_version: None,
            target_version: None,
            patch_size: patch_data.len(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        },
    };

    chain.links.push(new_link);

    // Serialize and write
    let output_path = output.unwrap_or(chain_path);
    let chain_bytes = serialize_chain(&chain).context("Failed to serialize chain")?;
    fs::write(output_path, &chain_bytes).context("Failed to write chain file")?;

    println!();
    println!("Patch appended successfully!");
    println!("  Chain ID: {}", chain.chain_id);
    println!("  Links: {}", chain.links.len());
    println!("  New head: {}", hex::encode(target_hash));

    Ok(())
}

/// Show information about a chain
pub fn info(chain_path: &Path, verbose: bool) -> Result<()> {
    // Read chain file
    let chain_data = fs::read(chain_path).context("Failed to read chain file")?;
    if !is_chain(&chain_data) {
        anyhow::bail!("File is not a valid patch chain");
    }
    let chain = deserialize_chain(&chain_data).context("Failed to parse chain")?;

    println!("Patch Chain Information");
    println!("=======================");
    println!();
    println!("File:     {}", chain_path.display());
    println!("Size:     {} bytes", chain_data.len());
    println!("Version:  {}", chain.version);
    println!("Chain ID: {}", chain.chain_id);
    println!();
    println!("Links: {}", chain.links.len());

    if let Some(base) = chain.base_hash() {
        println!("Base hash: {}", hex::encode(base));
    }
    if let Some(head) = chain.head_hash() {
        println!("Head hash: {}", hex::encode(head));
    }

    let total_size = chain.total_patch_size();
    println!("Total patch data: {} bytes", total_size);

    if let Some(ref desc) = chain.metadata.description {
        println!("Description: {}", desc);
    }

    if verbose && !chain.links.is_empty() {
        println!();
        println!("Link Details");
        println!("------------");

        for link in &chain.links {
            println!(
                "[{}] {} -> {} ({} bytes, {})",
                link.index,
                hex::encode(&link.source_hash[..8]),
                hex::encode(&link.target_hash[..8]),
                link.patch_data.len(),
                link.format
            );
            if let Some(ref src) = link.link_metadata.source_version {
                print!("     v{}", src);
                if let Some(ref tgt) = link.link_metadata.target_version {
                    println!(" -> v{}", tgt);
                } else {
                    println!();
                }
            }
        }
    }

    Ok(())
}

/// Extract patches from a chain for a hash range
pub fn extract(
    chain_path: &Path,
    output_dir: &Path,
    from_hash: Option<&str>,
    to_hash: Option<&str>,
) -> Result<()> {
    println!("Extracting patches from chain...");
    println!("  Chain: {}", chain_path.display());
    println!("  Output directory: {}", output_dir.display());

    // Read chain
    let chain_data = fs::read(chain_path).context("Failed to read chain file")?;
    if !is_chain(&chain_data) {
        anyhow::bail!("File is not a valid patch chain");
    }
    let chain = deserialize_chain(&chain_data).context("Failed to parse chain")?;

    // Create output directory if needed
    fs::create_dir_all(output_dir).context("Failed to create output directory")?;

    // Determine which links to extract
    let links_to_extract: Vec<_> = if from_hash.is_some() || to_hash.is_some() {
        let from = from_hash
            .map(parse_hash)
            .transpose()?
            .or_else(|| chain.base_hash());
        let to = to_hash
            .map(parse_hash)
            .transpose()?
            .or_else(|| chain.head_hash());

        match (from, to) {
            (Some(f), Some(t)) => chain
                .links_between(&f, &t)
                .ok_or_else(|| anyhow::anyhow!("Could not find path between specified hashes"))?,
            _ => chain.links.iter().collect(),
        }
    } else {
        chain.links.iter().collect()
    };

    // Extract each link
    for link in &links_to_extract {
        let ext = get_patch_extension(&link.format);
        let filename = format!("patch_{:04}.{}", link.index, ext);
        let output_path = output_dir.join(&filename);

        fs::write(&output_path, &link.patch_data)
            .with_context(|| format!("Failed to write {}", filename))?;

        println!("  Extracted: {}", filename);
    }

    println!();
    println!("Extracted {} patches", links_to_extract.len());

    Ok(())
}

// Helper functions

fn detect_patch_format(data: &[u8]) -> Result<String> {
    if data.len() < 4 {
        anyhow::bail!("Patch file too small");
    }

    if mallorn_tflite::is_tflp(data) {
        return Ok("tflite".into());
    }
    if mallorn_gguf::is_ggup(data) {
        return Ok("gguf".into());
    }
    if mallorn_safetensors::is_sftp(data) {
        return Ok("safetensors".into());
    }
    if mallorn_openvino::is_ovinp(data) {
        return Ok("openvino".into());
    }
    if mallorn_coreml::is_cmlp(data) {
        return Ok("coreml".into());
    }

    anyhow::bail!("Unknown patch format")
}

fn extract_patch_hashes(data: &[u8], format: &str) -> Result<([u8; 32], [u8; 32])> {
    let patch = match format {
        "tflite" => mallorn_tflite::deserialize_patch(data)
            .map_err(|e| anyhow::anyhow!("Failed to parse TFLite patch: {}", e))?,
        "gguf" => mallorn_gguf::deserialize_patch(data)
            .map_err(|e| anyhow::anyhow!("Failed to parse GGUF patch: {}", e))?,
        "safetensors" => mallorn_safetensors::deserialize_patch(data)
            .map_err(|e| anyhow::anyhow!("Failed to parse SafeTensors patch: {}", e))?,
        "openvino" => mallorn_openvino::deserialize_patch(data)
            .map_err(|e| anyhow::anyhow!("Failed to parse OpenVINO patch: {}", e))?,
        "coreml" => mallorn_coreml::deserialize_patch(data)
            .map_err(|e| anyhow::anyhow!("Failed to parse CoreML patch: {}", e))?,
        _ => anyhow::bail!("Unknown format: {}", format),
    };

    Ok((patch.source_hash, patch.target_hash))
}

fn get_patch_extension(format: &str) -> &'static str {
    match format {
        "tflite" => "tflp",
        "gguf" => "ggup",
        "safetensors" => "sftp",
        "openvino" => "ovinp",
        "coreml" => "cmlp",
        _ => "patch",
    }
}

fn parse_hash(s: &str) -> Result<[u8; 32]> {
    let bytes = hex::decode(s).context("Invalid hex hash")?;
    if bytes.len() != 32 {
        anyhow::bail!("Hash must be 32 bytes (64 hex characters)");
    }
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    Ok(arr)
}

/// Squash patches in a chain into a single patch
pub fn squash(
    chain_path: &Path,
    output: &Path,
    from_hash: Option<&str>,
    to_hash: Option<&str>,
) -> Result<()> {
    println!("Squashing chain patches...");
    println!("  Chain: {}", chain_path.display());
    println!("  Output: {}", output.display());

    // Read chain
    let chain_data = fs::read(chain_path).context("Failed to read chain file")?;
    if !is_chain(&chain_data) {
        anyhow::bail!("File is not a valid patch chain");
    }
    let chain = deserialize_chain(&chain_data).context("Failed to parse chain")?;

    // Determine which links to squash
    let links_to_squash: Vec<_> = if from_hash.is_some() || to_hash.is_some() {
        let from = from_hash
            .map(parse_hash)
            .transpose()?
            .or_else(|| chain.base_hash());
        let to = to_hash
            .map(parse_hash)
            .transpose()?
            .or_else(|| chain.head_hash());

        match (from, to) {
            (Some(f), Some(t)) => chain
                .links_between(&f, &t)
                .ok_or_else(|| anyhow::anyhow!("Could not find path between specified hashes"))?,
            _ => chain.links.iter().collect(),
        }
    } else {
        chain.links.iter().collect()
    };

    if links_to_squash.is_empty() {
        anyhow::bail!("No links to squash");
    }

    if links_to_squash.len() == 1 {
        // Just extract the single patch
        fs::write(output, &links_to_squash[0].patch_data)
            .context("Failed to write patch file")?;
        println!();
        println!("Single patch extracted (no squash needed)");
        return Ok(());
    }

    // Get from/to hashes for subchain
    let first = links_to_squash.first().unwrap();
    let last = links_to_squash.last().unwrap();

    // Create a subchain with the selected links
    let subchain = mallorn_core::subchain(&chain, &first.source_hash, &last.target_hash)
        .ok_or_else(|| anyhow::anyhow!("Failed to create subchain"))?;

    // Serialize the subchain
    let chain_bytes = serialize_chain(&subchain).context("Failed to serialize subchain")?;
    fs::write(output, &chain_bytes).context("Failed to write chain file")?;

    println!();
    println!("Subchain extracted successfully!");
    println!("  Links included: {}", links_to_squash.len());
    println!("  Output size: {} bytes", chain_bytes.len());
    println!("  Source: {}", hex::encode(&first.source_hash[..8]));
    println!("  Target: {}", hex::encode(&last.target_hash[..8]));

    Ok(())
}

/// Apply a chain to a model
pub fn apply(
    model: &Path,
    chain_path: &Path,
    output: &Path,
    target_hash: Option<&str>,
) -> Result<()> {
    println!("Applying chain to model...");
    println!("  Model: {}", model.display());
    println!("  Chain: {}", chain_path.display());
    println!("  Output: {}", output.display());

    // Read model and chain
    let mut model_data = fs::read(model).context("Failed to read model file")?;
    let chain_data = fs::read(chain_path).context("Failed to read chain file")?;

    if !is_chain(&chain_data) {
        anyhow::bail!("File is not a valid patch chain");
    }
    let chain = deserialize_chain(&chain_data).context("Failed to parse chain")?;

    // Determine target
    let target = if let Some(hash_str) = target_hash {
        Some(parse_hash(hash_str)?)
    } else {
        chain.head_hash()
    };

    if let Some(ref t) = target {
        println!("  Target: {}", hex::encode(&t[..8]));
    }

    // Find path to target
    let links_to_apply: Vec<_> = if let Some(ref t) = target {
        // Find path from current model hash to target
        chain
            .links
            .iter()
            .take_while(|link| link.target_hash != *t)
            .chain(chain.links.iter().filter(|link| link.target_hash == *t).take(1))
            .collect()
    } else {
        chain.links.iter().collect()
    };

    if links_to_apply.is_empty() {
        anyhow::bail!("No links to apply");
    }

    // Apply patches sequentially
    let mut links_applied = 0;
    for link in &links_to_apply {
        // Detect format and apply patch
        let format = &link.format;
        let (new_data, _) = apply_patch_by_format(&model_data, &link.patch_data, format)
            .with_context(|| format!("Failed to apply patch {}", link.index))?;
        model_data = new_data;
        links_applied += 1;

        // Check if we've reached the target
        if let Some(ref t) = target {
            if link.target_hash == *t {
                break;
            }
        }
    }

    // Write output
    fs::write(output, &model_data).context("Failed to write output model")?;

    println!();
    println!("Chain applied successfully!");
    println!("  Links applied: {}", links_applied);
    println!("  Output size: {} bytes", model_data.len());

    Ok(())
}

/// Apply a patch based on format
fn apply_patch_by_format(
    model_data: &[u8],
    patch_data: &[u8],
    format: &str,
) -> Result<(Vec<u8>, mallorn_core::PatchVerification)> {
    match format {
        "tflite" => {
            let patch = mallorn_tflite::deserialize_patch(patch_data)
                .map_err(|e| anyhow::anyhow!("Failed to parse TFLite patch: {}", e))?;
            let patcher = mallorn_tflite::TFLitePatcher::new();
            patcher
                .apply_and_verify(model_data, &patch)
                .map_err(|e| anyhow::anyhow!("Failed to apply TFLite patch: {}", e))
        }
        "gguf" => {
            let patch = mallorn_gguf::deserialize_patch(patch_data)
                .map_err(|e| anyhow::anyhow!("Failed to parse GGUF patch: {}", e))?;
            let patcher = mallorn_gguf::GGUFPatcher::new();
            patcher
                .apply_and_verify(model_data, &patch)
                .map_err(|e| anyhow::anyhow!("Failed to apply GGUF patch: {}", e))
        }
        "safetensors" => {
            let patch = mallorn_safetensors::deserialize_patch(patch_data)
                .map_err(|e| anyhow::anyhow!("Failed to parse SafeTensors patch: {}", e))?;
            let patcher = mallorn_safetensors::SafeTensorsPatcher::new();
            patcher
                .apply_and_verify(model_data, &patch)
                .map_err(|e| anyhow::anyhow!("Failed to apply SafeTensors patch: {}", e))
        }
        "openvino" => {
            let patch = mallorn_openvino::deserialize_patch(patch_data)
                .map_err(|e| anyhow::anyhow!("Failed to parse OpenVINO patch: {}", e))?;
            let patcher = mallorn_openvino::OpenVINOPatcher::new();
            patcher
                .apply_and_verify(model_data, &patch)
                .map_err(|e| anyhow::anyhow!("Failed to apply OpenVINO patch: {}", e))
        }
        "coreml" => {
            let patch = mallorn_coreml::deserialize_patch(patch_data)
                .map_err(|e| anyhow::anyhow!("Failed to parse CoreML patch: {}", e))?;
            let patcher = mallorn_coreml::CoreMLPatcher::new();
            patcher
                .apply_and_verify(model_data, &patch)
                .map_err(|e| anyhow::anyhow!("Failed to apply CoreML patch: {}", e))
        }
        _ => anyhow::bail!("Unknown patch format: {}", format),
    }
}
