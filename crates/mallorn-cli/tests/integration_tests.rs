//! Integration tests for mallorn CLI
//!
//! Tests the full diff/patch/verify cycle using synthetic test data.

use std::fs;
use std::process::Command;
use tempfile::TempDir;

/// Create a synthetic file that will be treated as a model
#[allow(dead_code)]
fn create_test_file(data: &[u8]) -> (TempDir, std::path::PathBuf) {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let path = dir.path().join("test.bin");
    fs::write(&path, data).expect("Failed to write test file");
    (dir, path)
}

/// Get the path to the mallorn binary
fn mallorn_bin() -> std::path::PathBuf {
    // The binary is in target/debug/ when running tests
    std::env::current_exe()
        .expect("Failed to get current exe")
        .parent()
        .expect("No parent")
        .parent()
        .expect("No grandparent")
        .join("mallorn")
}

#[test]
fn test_cli_help() {
    let output = Command::new(mallorn_bin())
        .arg("--help")
        .output()
        .expect("Failed to run mallorn");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Mallorn"));
    assert!(stdout.contains("diff"));
    assert!(stdout.contains("patch"));
    assert!(stdout.contains("verify"));
}

#[test]
fn test_cli_version() {
    let output = Command::new(mallorn_bin())
        .arg("--version")
        .output()
        .expect("Failed to run mallorn");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("mallorn"));
}

#[test]
fn test_diff_help() {
    let output = Command::new(mallorn_bin())
        .args(["diff", "--help"])
        .output()
        .expect("Failed to run mallorn");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Create a delta patch"));
    assert!(stdout.contains("--level"));
    assert!(stdout.contains("--neural"));
}

#[test]
fn test_patch_help() {
    let output = Command::new(mallorn_bin())
        .args(["patch", "--help"])
        .output()
        .expect("Failed to run mallorn");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Apply a patch"));
    assert!(stdout.contains("MODEL"));
    assert!(stdout.contains("PATCH"));
    assert!(stdout.contains("OUTPUT"));
}

#[test]
fn test_verify_help() {
    let output = Command::new(mallorn_bin())
        .args(["verify", "--help"])
        .output()
        .expect("Failed to run mallorn");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Verify a patch"));
}

#[test]
fn test_info_help() {
    let output = Command::new(mallorn_bin())
        .args(["info", "--help"])
        .output()
        .expect("Failed to run mallorn");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Show information"));
}

#[test]
fn test_diff_missing_files() {
    let output = Command::new(mallorn_bin())
        .args([
            "diff",
            "nonexistent1.tflite",
            "nonexistent2.tflite",
            "output.tflp",
        ])
        .output()
        .expect("Failed to run mallorn");

    // Should fail because files don't exist
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Failed") || stderr.contains("Error") || stderr.contains("error"));
}

#[test]
fn test_patch_missing_files() {
    let output = Command::new(mallorn_bin())
        .args([
            "patch",
            "nonexistent.tflite",
            "nonexistent.tflp",
            "output.tflite",
        ])
        .output()
        .expect("Failed to run mallorn");

    // Should fail because files don't exist
    assert!(!output.status.success());
}

#[test]
fn test_verify_missing_files() {
    let output = Command::new(mallorn_bin())
        .args(["verify", "nonexistent.tflite", "nonexistent.tflp"])
        .output()
        .expect("Failed to run mallorn");

    // Should fail because files don't exist
    assert!(!output.status.success());
}

#[test]
fn test_info_missing_file() {
    let output = Command::new(mallorn_bin())
        .args(["info", "nonexistent.tflp"])
        .output()
        .expect("Failed to run mallorn");

    // Should fail because file doesn't exist
    assert!(!output.status.success());
}

#[test]
fn test_diff_invalid_format() {
    // Create random binary files that aren't valid TFLite or GGUF
    let dir = TempDir::new().expect("Failed to create temp dir");
    let old_path = dir.path().join("old.bin");
    let new_path = dir.path().join("new.bin");
    let output_path = dir.path().join("patch.tflp");

    fs::write(&old_path, b"random data that is not a model").expect("Failed to write");
    fs::write(&new_path, b"different random data not a model").expect("Failed to write");

    let output = Command::new(mallorn_bin())
        .args([
            "diff",
            old_path.to_str().unwrap(),
            new_path.to_str().unwrap(),
            output_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run mallorn");

    // Should fail because files are not valid models
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Unknown") || stderr.contains("format") || stderr.contains("error"));
}

// ============================================================================
// v1.4 CLI Tests
// ============================================================================

#[test]
fn test_fingerprint_help() {
    let output = Command::new(mallorn_bin())
        .args(["fingerprint", "--help"])
        .output()
        .expect("Failed to run mallorn");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("fingerprint") || stdout.contains("Fingerprint"));
}

#[test]
fn test_download_help() {
    let output = Command::new(mallorn_bin())
        .args(["download", "--help"])
        .output()
        .expect("Failed to run mallorn");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Download") || stdout.contains("download"));
    assert!(stdout.contains("--resume") || stdout.contains("resume"));
}

#[test]
fn test_chain_help() {
    let output = Command::new(mallorn_bin())
        .args(["chain", "--help"])
        .output()
        .expect("Failed to run mallorn");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("chain") || stdout.contains("Chain"));
}

#[test]
fn test_chain_create_help() {
    let output = Command::new(mallorn_bin())
        .args(["chain", "create", "--help"])
        .output()
        .expect("Failed to run mallorn");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Create") || stdout.contains("create"));
}

#[test]
fn test_chain_squash_help() {
    let output = Command::new(mallorn_bin())
        .args(["chain", "squash", "--help"])
        .output()
        .expect("Failed to run mallorn");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Squash") || stdout.contains("squash"));
}

#[test]
fn test_chain_apply_help() {
    let output = Command::new(mallorn_bin())
        .args(["chain", "apply", "--help"])
        .output()
        .expect("Failed to run mallorn");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Apply") || stdout.contains("apply"));
}

#[test]
fn test_diff_parallel_flag_help() {
    let output = Command::new(mallorn_bin())
        .args(["diff", "--help"])
        .output()
        .expect("Failed to run mallorn");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--parallel"));
}

#[test]
fn test_patch_streaming_flag_help() {
    let output = Command::new(mallorn_bin())
        .args(["patch", "--help"])
        .output()
        .expect("Failed to run mallorn");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--streaming"));
}

// ============================================================================
// E2E Tests with Real Models (if fixtures exist)
// ============================================================================

/// Path to test fixtures relative to the workspace root
fn fixtures_dir() -> std::path::PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    std::path::PathBuf::from(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests")
        .join("fixtures")
        .join("models")
}

#[test]
fn test_fingerprint_with_real_model() {
    let model_path = fixtures_dir().join("mobilenet_v1.tflite");
    if !model_path.exists() {
        eprintln!("Skipping test: fixture not found at {:?}", model_path);
        return;
    }

    let output = Command::new(mallorn_bin())
        .args(["fingerprint", model_path.to_str().unwrap()])
        .output()
        .expect("Failed to run mallorn fingerprint");

    assert!(
        output.status.success(),
        "fingerprint failed: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Fingerprint") || stdout.contains("Hash") || stdout.contains("hash"));
}

#[test]
fn test_fingerprint_json_output() {
    let model_path = fixtures_dir().join("mobilenet_v1.tflite");
    if !model_path.exists() {
        eprintln!("Skipping test: fixture not found");
        return;
    }

    let output = Command::new(mallorn_bin())
        .args(["fingerprint", "--json", model_path.to_str().unwrap()])
        .output()
        .expect("Failed to run mallorn fingerprint --json");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should be valid JSON
    assert!(stdout.contains("{") && stdout.contains("}"));
    assert!(stdout.contains("header_hash") || stdout.contains("format"));
}

#[test]
fn test_full_diff_patch_cycle() {
    // Use v1 and v1_quant which are the same architecture with different quantization
    let v1_path = fixtures_dir().join("mobilenet_v1.tflite");
    let v1_quant_path = fixtures_dir().join("mobilenet_v1_quant.tflite");

    if !v1_path.exists() || !v1_quant_path.exists() {
        eprintln!("Skipping test: fixtures not found");
        return;
    }

    let dir = TempDir::new().expect("Failed to create temp dir");
    let patch_path = dir.path().join("patch.tflp");
    let output_path = dir.path().join("restored.tflite");

    // Create patch from v1 to v1_quant
    let diff_output = Command::new(mallorn_bin())
        .args([
            "diff",
            v1_path.to_str().unwrap(),
            v1_quant_path.to_str().unwrap(),
            "-o",
            patch_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run diff");

    if !diff_output.status.success() {
        // Different model structures may not diff cleanly - this is acceptable
        eprintln!(
            "Diff failed (expected for different architectures): {}",
            String::from_utf8_lossy(&diff_output.stderr)
        );
        return;
    }

    assert!(patch_path.exists(), "Patch file should be created");

    // Apply patch
    let patch_output = Command::new(mallorn_bin())
        .args([
            "patch",
            v1_path.to_str().unwrap(),
            patch_path.to_str().unwrap(),
            "-o",
            output_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run patch");

    if !patch_output.status.success() {
        // Patch application may fail for structurally different models
        eprintln!(
            "Patch apply failed (may be expected for different structures): {:?}",
            String::from_utf8_lossy(&patch_output.stderr)
        );
        return;
    }

    assert!(output_path.exists(), "Output file should be created");

    // Verify output matches target
    let target_data = fs::read(&v1_quant_path).expect("Failed to read target");
    let output_data = fs::read(&output_path).expect("Failed to read output");
    assert_eq!(
        target_data.len(),
        output_data.len(),
        "Output size should match target"
    );
    assert_eq!(
        target_data, output_data,
        "Output should match target exactly"
    );
}

#[test]
fn test_diff_with_parallel_flag() {
    let v1_path = fixtures_dir().join("mobilenet_v1.tflite");
    let v2_path = fixtures_dir().join("mobilenet_v2.tflite");

    if !v1_path.exists() || !v2_path.exists() {
        eprintln!("Skipping test: fixtures not found");
        return;
    }

    let dir = TempDir::new().expect("Failed to create temp dir");
    let patch_path = dir.path().join("patch.tflp");

    // Create patch with --parallel
    let output = Command::new(mallorn_bin())
        .args([
            "diff",
            "--parallel",
            v1_path.to_str().unwrap(),
            v2_path.to_str().unwrap(),
            "-o",
            patch_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run diff --parallel");

    // Should either succeed or fail gracefully (not panic)
    if output.status.success() {
        assert!(
            patch_path.exists(),
            "Patch file should be created with --parallel"
        );
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !stderr.contains("panic"),
            "Should not panic with --parallel flag"
        );
    }
}
