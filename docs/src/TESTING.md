# Mallorn Testing Strategy

Test-driven development for edge model delta updates, with research adaptations.

## Testing Philosophy

Mallorn straddles known techniques and research optimization:
- **Format parsing**: Well-understood, traditional TDD
- **Tensor diffing**: Some research (optimal alignment)
- **Neural compression**: Research-heavy (ZipNN-style)
- **Embedded patcher**: Well-understood, traditional TDD

We use hybrid approach: **invariant tests** for correctness, **hypothesis tests** for performance, **exploration tests** for research.

---

## Test Categories

### 1. Invariant Tests (Write First, Must Pass)

Non-negotiable correctness properties. Write before implementation.

```rust
// tests/invariants/roundtrip.rs

/// INVARIANT: diff + patch produces exact original
#[test]
fn invariant_roundtrip_exact() {
    let old = load_test_model("mobilenet_v1.tflite");
    let new = load_test_model("mobilenet_v2.tflite");
    
    let patch = diff(&old, &new);
    let reconstructed = apply_patch(&old, &patch);
    
    assert_eq!(reconstructed, new, "Roundtrip must be exact");
}

/// INVARIANT: patch verification detects corruption
#[test]
fn invariant_corruption_detected() {
    let old = load_test_model("test.tflite");
    let new = load_test_model("test_v2.tflite");
    let mut patch = diff(&old, &new);
    
    // Corrupt the patch
    patch.operations[0] = corrupt_operation(&patch.operations[0]);
    
    let result = verify_patch(&old, &patch);
    assert!(!result.patch_valid, "Corruption must be detected");
}

/// INVARIANT: source hash mismatch rejected
#[test]
fn invariant_wrong_source_rejected() {
    let wrong_source = load_test_model("wrong.tflite");
    let patch = load_test_patch("update.tflp");
    
    let result = apply_patch(&wrong_source, &patch);
    assert!(result.is_err(), "Wrong source must be rejected");
}

/// INVARIANT: streaming produces same result as batch
#[test]
fn invariant_streaming_equals_batch() {
    let old = load_test_model("test.tflite");
    let patch = load_test_patch("update.tflp");
    
    let batch_result = apply_patch(&old, &patch).unwrap();
    let stream_result = apply_patch_streaming(&old, &patch, 1024).unwrap();
    
    assert_eq!(batch_result, stream_result);
}

/// INVARIANT: empty diff for identical models
#[test]
fn invariant_identical_models_empty_diff() {
    let model = load_test_model("test.tflite");
    let patch = diff(&model, &model);
    
    assert!(patch.operations.is_empty() || all_copy_operations(&patch));
}
```

### 2. Hypothesis Tests (Validate Assumptions)

Performance expectations. May fail during development — that's learning.

```rust
// tests/hypotheses/compression.rs

/// HYPOTHESIS: Patch size < 10% for minor weight updates
#[test]
fn hypothesis_minor_update_patch_size() {
    let base = load_test_model("base.tflite");
    let fine_tuned = load_test_model("fine_tuned.tflite");  // Same arch
    
    let patch = diff(&base, &fine_tuned);
    let ratio = patch_bytes(&patch).len() as f64 / base.len() as f64;
    
    println!("Patch ratio: {:.1}%", ratio * 100.0);
    assert!(
        ratio <= 0.10,
        "Expected patch ≤10% of model, got {:.1}%",
        ratio * 100.0
    );
}

/// HYPOTHESIS: Neural compression beats zstd by ≥20%
#[test]
fn hypothesis_neural_compression_improvement() {
    let base = load_test_model("base.tflite");
    let updated = load_test_model("updated.tflite");
    
    let zstd_patch = diff_with_options(&base, &updated, DiffOptions {
        compression: CompressionMethod::Zstd { level: 3 },
        neural_compression: false,
        ..Default::default()
    });
    
    let neural_patch = diff_with_options(&base, &updated, DiffOptions {
        compression: CompressionMethod::Zstd { level: 3 },
        neural_compression: true,
        ..Default::default()
    });
    
    let improvement = 1.0 - (neural_patch.len() as f64 / zstd_patch.len() as f64);
    
    println!("Neural vs zstd: {:.1}% smaller", improvement * 100.0);
    assert!(
        improvement >= 0.20,
        "Expected ≥20% improvement, got {:.1}%",
        improvement * 100.0
    );
}

/// HYPOTHESIS: Patch application < 100ms for 10MB model
#[test]
fn hypothesis_patch_speed() {
    let old = load_test_model("10mb_model.tflite");
    let patch = load_test_patch("10mb_update.tflp");
    
    let start = Instant::now();
    let _ = apply_patch(&old, &patch);
    let elapsed = start.elapsed();
    
    println!("Patch application: {:?}", elapsed);
    assert!(
        elapsed < Duration::from_millis(100),
        "Expected <100ms, got {:?}",
        elapsed
    );
}

/// HYPOTHESIS: Streaming patcher works with 1KB buffer
#[test]
fn hypothesis_1kb_streaming() {
    let old = load_test_model("test.tflite");
    let patch = load_test_patch("update.tflp");
    
    let result = apply_patch_streaming(&old, &patch, 1024);  // 1KB
    
    assert!(result.is_ok(), "1KB streaming must work");
}

/// HYPOTHESIS: GGUF quantized tensors compress well
#[test]
fn hypothesis_gguf_quantized_compression() {
    let base = load_test_model("llama-7b-q4.gguf");
    let updated = load_test_model("llama-7b-q4-tuned.gguf");
    
    let patch = diff(&base, &updated);
    let ratio = patch.len() as f64 / base.len() as f64;
    
    println!("GGUF Q4 patch ratio: {:.1}%", ratio * 100.0);
    assert!(
        ratio <= 0.15,
        "Expected ≤15% for quantized, got {:.1}%",
        ratio * 100.0
    );
}
```

### 3. Exploration Tests (Learning)

Not assertions — measurements. Run to understand the problem space.

```rust
// tests/exploration/model_survey.rs

/// EXPLORATION: Patch ratios across model types
#[test]
#[ignore]  // cargo test exploration_ -- --ignored --nocapture
fn exploration_patch_ratios_by_model_type() {
    let model_pairs = vec![
        ("MobileNet V1→V2", "mobilenet_v1.tflite", "mobilenet_v2.tflite"),
        ("YOLO v5s→v5m", "yolov5s.tflite", "yolov5m.tflite"),
        ("Whisper tiny→base", "whisper_tiny.tflite", "whisper_base.tflite"),
        ("Llama 7B Q4 tune", "llama7b_q4.gguf", "llama7b_q4_tuned.gguf"),
    ];
    
    println!("\n=== Patch Ratio Exploration ===\n");
    println!("{:<25} {:>12} {:>12} {:>10}", "Model Pair", "Old Size", "Patch Size", "Ratio");
    println!("{:-<65}", "");
    
    for (name, old_path, new_path) in model_pairs {
        if let (Ok(old), Ok(new)) = (load_test_model(old_path), load_test_model(new_path)) {
            let patch = diff(&old, &new);
            let ratio = patch.len() as f64 / old.len() as f64 * 100.0;
            
            println!(
                "{:<25} {:>12} {:>12} {:>9.1}%",
                name,
                format_bytes(old.len()),
                format_bytes(patch.len()),
                ratio
            );
        }
    }
}

/// EXPLORATION: Compression method comparison
#[test]
#[ignore]
fn exploration_compression_methods() {
    let base = load_test_model("test.tflite");
    let updated = load_test_model("test_v2.tflite");
    
    let methods = vec![
        ("None", CompressionMethod::None),
        ("LZ4", CompressionMethod::Lz4),
        ("Zstd-1", CompressionMethod::Zstd { level: 1 }),
        ("Zstd-3", CompressionMethod::Zstd { level: 3 }),
        ("Zstd-9", CompressionMethod::Zstd { level: 9 }),
        ("Neural", CompressionMethod::Neural { variant: NeuralCompressionVariant::ExponentGrouping }),
    ];
    
    println!("\n=== Compression Method Comparison ===\n");
    println!("{:<15} {:>12} {:>12} {:>12}", "Method", "Patch Size", "Ratio", "Time");
    println!("{:-<55}", "");
    
    for (name, method) in methods {
        let start = Instant::now();
        let patch = diff_with_options(&base, &updated, DiffOptions {
            compression: method,
            ..Default::default()
        });
        let elapsed = start.elapsed();
        let ratio = patch.len() as f64 / base.len() as f64 * 100.0;
        
        println!(
            "{:<15} {:>12} {:>11.1}% {:>12}",
            name,
            format_bytes(patch.len()),
            ratio,
            format_duration(elapsed)
        );
    }
}

/// EXPLORATION: Buffer size vs speed tradeoff
#[test]
#[ignore]
fn exploration_buffer_size_tradeoff() {
    let old = load_test_model("large_model.tflite");
    let patch = load_test_patch("large_update.tflp");
    
    println!("\n=== Buffer Size Tradeoff ===\n");
    println!("{:>10} {:>12} {:>12}", "Buffer", "Time", "Peak RSS");
    println!("{:-<38}", "");
    
    for buffer_kb in [1, 2, 4, 8, 16, 32, 64, 128] {
        let buffer_size = buffer_kb * 1024;
        let (elapsed, peak_mem) = measure_streaming_patch(&old, &patch, buffer_size);
        
        println!(
            "{:>8}KB {:>12} {:>12}",
            buffer_kb,
            format_duration(elapsed),
            format_bytes(peak_mem)
        );
    }
}
```

---

## Test Hardware

### Primary Development
- **M3 Pro 18GB**: All tests must pass
- **RTX 4080S 16GB**: GPU-accelerated tests (if any)

### CI Environment
- **GitHub Actions**: ubuntu-latest, macos-latest, windows-latest
- **Memory limit**: 8GB (GitHub Actions default)

### Embedded Testing
- **ESP32-S3**: 512KB SRAM, 8MB Flash
- **STM32L4**: 256KB SRAM, 1MB Flash
- **QEMU**: For CI when hardware unavailable

---

## Test Data

### Test Models (fixtures/)

| Model | Format | Size | Purpose |
|-------|--------|------|---------|
| `tiny_model.tflite` | TFLite | 50KB | Fast iteration |
| `mobilenet_v1.tflite` | TFLite | 4MB | Standard test |
| `mobilenet_v2.tflite` | TFLite | 4MB | Diff target |
| `yolov5s.tflite` | TFLite | 14MB | Larger model |
| `llama-tiny-q4.gguf` | GGUF | 50MB | GGUF baseline |
| `corrupted.tflite` | TFLite | 1MB | Error handling |

### Test Patches (fixtures/)

| Patch | Source → Target | Size |
|-------|-----------------|------|
| `minor_update.tflp` | mobilenet_v1 → v1_tuned | ~200KB |
| `major_update.tflp` | mobilenet_v1 → v2 | ~2MB |
| `corrupted.tflp` | N/A | N/A |

---

## Test File Organization

```
tests/
├── invariants/
│   ├── mod.rs
│   ├── roundtrip.rs          # Diff + patch = exact
│   ├── verification.rs       # Hash checks
│   ├── corruption.rs         # Error detection
│   └── streaming.rs          # Stream = batch
│
├── hypotheses/
│   ├── mod.rs
│   ├── compression.rs        # Size expectations
│   ├── performance.rs        # Speed expectations
│   └── embedded.rs           # 1KB buffer works
│
├── exploration/
│   ├── mod.rs
│   ├── model_survey.rs       # Patch ratios by type
│   ├── compression_cmp.rs    # Method comparison
│   └── buffer_tradeoff.rs    # Size vs speed
│
├── integration/
│   ├── mod.rs
│   ├── cli.rs                # CLI end-to-end
│   ├── python.rs             # Python bindings
│   └── embedded.rs           # C library
│
└── fixtures/
    ├── models/
    │   ├── tiny_model.tflite
    │   ├── mobilenet_v1.tflite
    │   └── ...
    └── patches/
        ├── minor_update.tflp
        └── ...
```

---

## Development Workflow

### Before Each Module

1. **Write invariant tests first**
```rust
#[test]
fn invariant_parser_roundtrip() { todo!() }
```

2. **Create skeleton**
```rust
pub fn parse(data: &[u8]) -> Result<Model, ParseError> {
    unimplemented!()
}
```

3. **Make invariants pass**

4. **Add hypothesis tests**
```rust
#[test]
fn hypothesis_parse_speed() { todo!() }
```

5. **Improve until hypotheses pass**

6. **Run exploration to validate assumptions**

### CI Pipeline

```yaml
# .github/workflows/test.yml
jobs:
  invariants:
    name: Invariant Tests (MUST pass)
    runs-on: ubuntu-latest
    steps:
      - run: cargo test --test invariants

  hypotheses:
    name: Hypothesis Tests (SHOULD pass)
    runs-on: ubuntu-latest
    continue-on-error: true
    steps:
      - run: cargo test --test hypotheses
      
  exploration:
    name: Exploration (Weekly)
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    steps:
      - run: cargo test exploration_ -- --ignored --nocapture
```

---

## Coverage Targets

| Area | Target | Critical Paths |
|------|--------|----------------|
| mallorn-core | 80% | diff, patch, verify |
| mallorn-tflite | 80% | parser, patcher |
| mallorn-gguf | 80% | parser, patcher |
| mallorn-cli | 70% | commands |
| mallorn-lite | 90% | streaming patcher |

---

## Mocking Strategy

### Format Parsers
```rust
// Mock parsed model for testing differ
fn mock_parsed_model(tensors: Vec<Tensor>) -> ParsedModel {
    ParsedModel {
        format: "mock".into(),
        metadata: ModelMetadata::default(),
        tensors,
        graph: None,
    }
}
```

### IO Operations
```rust
// In-memory reader/writer for streaming tests
struct MemoryIO {
    data: Vec<u8>,
    position: usize,
}
```

---

## Test Commands

```bash
# Run all tests
cargo test

# Run only invariants
cargo test --test invariants

# Run only hypotheses
cargo test --test hypotheses

# Run exploration (manual)
cargo test exploration_ -- --ignored --nocapture

# Run with coverage
cargo tarpaulin --out Html

# Run benchmarks
cargo bench

# Run specific test
cargo test invariant_roundtrip
```
