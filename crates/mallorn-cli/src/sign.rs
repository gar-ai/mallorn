//! Sign and verify patch files with ED25519 signatures

use anyhow::{Context, Result};
use ed25519_dalek::{SigningKey, SECRET_KEY_LENGTH};
use mallorn_core::{generate_keypair, is_signed_patch, SignedPatch};
use std::fs;
use std::path::Path;

/// Generate a new keypair and save to files
pub fn generate(private_key_path: &Path, public_key_path: &Path) -> Result<()> {
    let (signing_key, verifying_key) = generate_keypair();

    // Save private key
    let private_bytes = signing_key.to_bytes();
    fs::write(private_key_path, private_bytes)
        .with_context(|| format!("Failed to write private key to {:?}", private_key_path))?;

    // Save public key
    let public_bytes = verifying_key.to_bytes();
    fs::write(public_key_path, public_bytes)
        .with_context(|| format!("Failed to write public key to {:?}", public_key_path))?;

    println!("Generated keypair:");
    println!("  Private key: {:?}", private_key_path);
    println!("  Public key:  {:?}", public_key_path);
    println!();
    println!("Keep the private key secure. Distribute the public key to devices.");

    Ok(())
}

/// Sign a patch file
pub fn sign(
    patch_path: &Path,
    private_key_path: &Path,
    output_path: &Path,
    version: u64,
) -> Result<()> {
    // Read the patch file
    let patch_data = fs::read(patch_path)
        .with_context(|| format!("Failed to read patch file {:?}", patch_path))?;

    // Check if already signed
    if is_signed_patch(&patch_data) {
        anyhow::bail!("Patch is already signed. Use --force to re-sign.");
    }

    // Read the private key
    let key_bytes = fs::read(private_key_path)
        .with_context(|| format!("Failed to read private key {:?}", private_key_path))?;

    if key_bytes.len() != SECRET_KEY_LENGTH {
        anyhow::bail!(
            "Invalid private key size: expected {} bytes, got {}",
            SECRET_KEY_LENGTH,
            key_bytes.len()
        );
    }

    let key_array: [u8; SECRET_KEY_LENGTH] = key_bytes.try_into().unwrap();
    let signing_key = SigningKey::from_bytes(&key_array);

    // Create signed patch
    let signed = SignedPatch::new(patch_data, &signing_key, version)
        .map_err(|e| anyhow::anyhow!("Failed to sign patch: {:?}", e))?;

    // Write output
    let output_bytes = signed.to_bytes();
    fs::write(output_path, &output_bytes)
        .with_context(|| format!("Failed to write signed patch to {:?}", output_path))?;

    println!("Signed patch created:");
    println!("  Input:   {:?}", patch_path);
    println!("  Output:  {:?}", output_path);
    println!("  Version: {}", version);
    println!("  Size:    {} bytes", output_bytes.len());

    Ok(())
}

/// Verify a signed patch
pub fn verify_signature(
    signed_patch_path: &Path,
    public_key_path: Option<&Path>,
    minimum_version: Option<u64>,
) -> Result<()> {
    // Read the signed patch
    let patch_data = fs::read(signed_patch_path)
        .with_context(|| format!("Failed to read signed patch {:?}", signed_patch_path))?;

    if !is_signed_patch(&patch_data) {
        anyhow::bail!("File is not a signed patch (missing MSIG header)");
    }

    // Parse the signed patch
    let signed = SignedPatch::from_bytes(&patch_data)
        .map_err(|e| anyhow::anyhow!("Failed to parse signed patch: {:?}", e))?;

    // Verify signature
    if let Some(key_path) = public_key_path {
        // Verify against specific trusted key
        let key_bytes = fs::read(key_path)
            .with_context(|| format!("Failed to read public key {:?}", key_path))?;

        if key_bytes.len() != 32 {
            anyhow::bail!("Invalid public key size: expected 32 bytes, got {}", key_bytes.len());
        }

        let key_array: [u8; 32] = key_bytes.try_into().unwrap();
        signed
            .verify_with_key(&key_array)
            .map_err(|e| anyhow::anyhow!("Signature verification failed: {:?}", e))?;

        println!("Signature valid (trusted key)");
    } else {
        // Verify using embedded key
        signed
            .verify()
            .map_err(|e| anyhow::anyhow!("Signature verification failed: {:?}", e))?;

        println!("Signature valid (embedded key)");
    }

    // Check version if specified
    if let Some(min_ver) = minimum_version {
        signed
            .check_version(min_ver)
            .map_err(|e| anyhow::anyhow!("Version check failed: {:?}", e))?;
        println!("Version check passed: {} >= {}", signed.version, min_ver);
    }

    println!();
    println!("Signed patch details:");
    println!("  Version:    {}", signed.version);
    println!("  Patch size: {} bytes", signed.patch_data.len());
    println!("  Public key: {}", hex::encode(&signed.public_key[..8]));

    Ok(())
}

/// Extract the unsigned patch from a signed patch
#[allow(dead_code)]
pub fn extract(signed_patch_path: &Path, output_path: &Path) -> Result<()> {
    // Read the signed patch
    let patch_data = fs::read(signed_patch_path)
        .with_context(|| format!("Failed to read signed patch {:?}", signed_patch_path))?;

    if !is_signed_patch(&patch_data) {
        anyhow::bail!("File is not a signed patch (missing MSIG header)");
    }

    // Parse and extract
    let signed = SignedPatch::from_bytes(&patch_data)
        .map_err(|e| anyhow::anyhow!("Failed to parse signed patch: {:?}", e))?;

    // Write the unsigned patch
    fs::write(output_path, &signed.patch_data)
        .with_context(|| format!("Failed to write patch to {:?}", output_path))?;

    println!("Extracted unsigned patch:");
    println!("  Input:  {:?}", signed_patch_path);
    println!("  Output: {:?}", output_path);
    println!("  Size:   {} bytes", signed.patch_data.len());

    Ok(())
}
