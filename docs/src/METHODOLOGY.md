# Mallorn Development Methodology

Test-driven development adapted for research contexts. Write tests for hypotheses, validate, iterate.

## The Problem with Pure TDD in Research

Traditional TDD assumes you know the expected behavior:

```rust
#[test]
fn test_add() {
    assert_eq!(add(2, 2), 4);  // We KNOW this is correct
}
```

Research doesn't work that way:

```rust
#[test]
fn test_compression_ratio() {
    assert!(ratio >= 10.0);  // Is 10x even achievable? We don't know yet.
}
```

## Solution: Hypothesis-Driven Testing

### Three Test Categories

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TEST PYRAMID                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│     ┌─────────────────────────────────────────┐                     │
│     │         EXPLORATION TESTS               │  ← Write last       │
│     │  "What compression ratio CAN we get?"   │    (discover)       │
│     └─────────────────────────────────────────┘                     │
│                        │                                             │
│                        ▼                                             │
│     ┌─────────────────────────────────────────┐                     │
│     │         HYPOTHESIS TESTS                │  ← Write second     │
│     │  "IF we group exponents, THEN ratio     │    (validate)       │
│     │   should improve by ~20%"               │                     │
│     └─────────────────────────────────────────┘                     │
│                        │                                             │
│                        ▼                                             │
│     ┌─────────────────────────────────────────┐                     │
│     │         INVARIANT TESTS                 │  ← Write first      │
│     │  "Roundtrip MUST be lossless"           │    (guarantee)      │
│     │  "Patch + old = new (exactly)"          │                     │
│     └─────────────────────────────────────────┘                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1. Invariant Tests (Write First)

These are non-negotiable correctness properties. Write them before any code.

```rust
/// INVARIANT: Diff + patch produces exact original
/// This MUST hold regardless of algorithm choices
#[test]
fn invariant_diff_patch_roundtrip() {
    let old_model = load_test_model("mobilenet_v1.tflite");
    let new_model = load_test_model("mobilenet_v2.tflite");
    
    // Whatever algorithm we use...
    let patch = diff(&old_model, &new_model);
    let reconstructed = apply_patch(&old_model, &patch);
    
    // ...this MUST hold
    assert_eq!(reconstructed, new_model);
}

/// INVARIANT: Patch verification detects corruption
#[test]
fn invariant_corrupted_patch_detected() {
    let patch = create_test_patch();
    let mut corrupted = patch.clone();
    corrupted[100] ^= 0xFF;  // Flip some bits
    
    // MUST detect corruption
    assert!(verify_patch(&corrupted).is_err());
}

/// INVARIANT: Streaming patcher produces same result as batch
#[test]
fn invariant_streaming_equals_batch() {
    let old_model = load_test_model("test.tflite");
    let patch = load_test_patch("test.tflp");
    
    let batch_result = apply_patch_batch(&old_model, &patch);
    let stream_result = apply_patch_streaming(&old_model, &patch, 1024);
    
    assert_eq!(batch_result, stream_result);
}
```

### 2. Hypothesis Tests (Write Second)

These encode our assumptions. They may fail — that's learning.

```rust
/// HYPOTHESIS: Exponent-grouped compression beats naive zstd by ~20%
/// Based on ZipNN paper claims
#[test]
fn hypothesis_exponent_grouping_improves_ratio() {
    let weights = load_float32_weights("resnet50_weights.bin");
    
    let naive_size = zstd_compress(&weights).len();
    let grouped_size = exponent_grouped_compress(&weights).len();
    
    let improvement = 1.0 - (grouped_size as f64 / naive_size as f64);
    
    // Hypothesis: ≥15% improvement (conservative vs paper's 33%)
    // If this fails, we need to investigate why
    println!("Improvement: {:.1}%", improvement * 100.0);
    assert!(
        improvement >= 0.15,
        "Expected ≥15% improvement, got {:.1}%. Hypothesis may be wrong.",
        improvement * 100.0
    );
}

/// HYPOTHESIS: Tensor-aware diff beats bsdiff by ≥30%
#[test]
fn hypothesis_tensor_aware_beats_binary_diff() {
    let old_model = load_test_model("yolo_v1.tflite");
    let new_model = load_test_model("yolo_v2.tflite");
    
    let bsdiff_size = bsdiff(&old_model, &new_model).len();
    let tensor_diff_size = tensor_aware_diff(&old_model, &new_model).len();
    
    let improvement = 1.0 - (tensor_diff_size as f64 / bsdiff_size as f64);
    
    println!("Tensor-aware vs bsdiff: {:.1}% smaller", improvement * 100.0);
    assert!(
        improvement >= 0.30,
        "Expected ≥30% improvement over bsdiff. Got {:.1}%.",
        improvement * 100.0
    );
}

/// HYPOTHESIS: Minor weight updates produce patches <5% of model size
#[test]
fn hypothesis_minor_update_patch_size() {
    let base_model = load_test_model("base.tflite");
    let fine_tuned = load_test_model("fine_tuned.tflite");  // Same arch, different weights
    
    let patch = create_patch(&base_model, &fine_tuned);
    let ratio = patch.len() as f64 / base_model.len() as f64;
    
    println!("Patch size ratio: {:.1}%", ratio * 100.0);
    assert!(
        ratio <= 0.05,
        "Expected patch ≤5% of model, got {:.1}%",
        ratio * 100.0
    );
}
```

### 3. Exploration Tests (Write to Learn)

These aren't assertions — they're measurements. Run them to understand the problem space.

```rust
/// EXPLORATION: What compression ratios do we see across model types?
#[test]
#[ignore]  // Run manually: cargo test exploration_ -- --ignored --nocapture
fn exploration_compression_ratios_by_model_type() {
    let models = vec![
        ("MobileNet V2", "mobilenet_v2.tflite"),
        ("ResNet50", "resnet50.tflite"),
        ("BERT-tiny", "bert_tiny.tflite"),
        ("YOLOv5s", "yolov5s.tflite"),
        ("Whisper-tiny", "whisper_tiny.tflite"),
    ];
    
    println!("\n=== Compression Ratio Exploration ===\n");
    println!("{:<20} {:>12} {:>12} {:>12}", "Model", "Original", "Compressed", "Ratio");
    println!("{:-<60}", "");
    
    for (name, path) in models {
        if let Ok(model) = load_test_model(path) {
            let compressed = compress(&model);
            let ratio = model.len() as f64 / compressed.len() as f64;
            println!(
                "{:<20} {:>12} {:>12} {:>11.1}x",
                name,
                format_bytes(model.len()),
                format_bytes(compressed.len()),
                ratio
            );
        }
    }
}

/// EXPLORATION: How does patch size scale with weight change magnitude?
#[test]
#[ignore]
fn exploration_patch_size_vs_weight_delta() {
    let base = load_test_model("base.tflite");
    
    println!("\n=== Patch Size vs Weight Change ===\n");
    println!("{:>12} {:>12} {:>12}", "Change %", "Patch Size", "Patch %");
    println!("{:-<40}", "");
    
    for change_pct in [1, 5, 10, 25, 50, 100] {
        let modified = perturb_weights(&base, change_pct as f64 / 100.0);
        let patch = create_patch(&base, &modified);
        let patch_pct = patch.len() as f64 / base.len() as f64 * 100.0;
        
        println!(
            "{:>11}% {:>12} {:>11.1}%",
            change_pct,
            format_bytes(patch.len()),
            patch_pct
        );
    }
}

/// EXPLORATION: What's the memory/speed tradeoff for streaming buffer sizes?
#[test]
#[ignore]
fn exploration_streaming_buffer_tradeoff() {
    let model = load_test_model("large_model.tflite");
    let patch = load_test_patch("large_patch.tflp");
    
    println!("\n=== Streaming Buffer Size Tradeoff ===\n");
    println!("{:>12} {:>12} {:>12}", "Buffer", "Time", "Peak RSS");
    println!("{:-<40}", "");
    
    for buffer_kb in [1, 4, 16, 64, 256, 1024] {
        let buffer_size = buffer_kb * 1024;
        let start = std::time::Instant::now();
        let peak_mem = measure_peak_memory(|| {
            apply_patch_streaming(&model, &patch, buffer_size)
        });
        let elapsed = start.elapsed();
        
        println!(
            "{:>10}KB {:>12} {:>12}",
            buffer_kb,
            format_duration(elapsed),
            format_bytes(peak_mem)
        );
    }
}
```

## Development Workflow

### Phase 1: Invariants First

Before writing ANY implementation:

```rust
// tests/invariants.rs

#[test]
fn invariant_roundtrip_lossless() { todo!() }

#[test]  
fn invariant_patch_checksum_verified() { todo!() }

#[test]
fn invariant_streaming_matches_batch() { todo!() }
```

These start as `todo!()` — you're declaring WHAT must be true.

### Phase 2: Skeleton Implementation

Write minimal code that makes invariants compile (but fail):

```rust
pub fn diff(old: &[u8], new: &[u8]) -> Vec<u8> {
    unimplemented!()
}

pub fn patch(old: &[u8], patch: &[u8]) -> Vec<u8> {
    unimplemented!()
}
```

### Phase 3: Make Invariants Pass

Implement the simplest thing that works:

```rust
pub fn diff(old: &[u8], new: &[u8]) -> Vec<u8> {
    // Dumbest possible: just store the new model
    new.to_vec()
}

pub fn patch(old: &[u8], patch: &[u8]) -> Vec<u8> {
    // Dumbest possible: patch IS the new model
    patch.to_vec()
}
```

Tests pass! (Badly, but they pass.)

### Phase 4: Add Hypotheses

Now add performance expectations:

```rust
#[test]
fn hypothesis_diff_smaller_than_new() {
    let old = load_model("v1.tflite");
    let new = load_model("v2.tflite");
    
    let patch = diff(&old, &new);
    
    // This will FAIL with our dumb implementation
    assert!(patch.len() < new.len(), "Patch should be smaller than full model");
}
```

Test fails → improve implementation → test passes → add next hypothesis.

### Phase 5: Explore and Refine

Run exploration tests to understand the landscape:

```bash
cargo test exploration_ -- --ignored --nocapture 2>&1 | tee exploration_results.txt
```

Use findings to:
- Adjust hypothesis thresholds
- Identify new invariants
- Guide algorithm choices

## Test File Organization

```
mallorn/
├── crates/
│   └── mallorn-core/
│       └── src/
│           └── *.rs
└── tests/
    ├── invariants/           # MUST pass, write first
    │   ├── roundtrip.rs
    │   ├── checksum.rs
    │   └── streaming.rs
    ├── hypotheses/           # SHOULD pass, validate assumptions
    │   ├── compression_ratio.rs
    │   ├── patch_size.rs
    │   └── performance.rs
    ├── exploration/          # Learning, run manually
    │   ├── model_survey.rs
    │   ├── algorithm_comparison.rs
    │   └── parameter_sweep.rs
    └── fixtures/             # Test data
        ├── models/
        └── patches/
```

## Handling Hypothesis Failures

When a hypothesis test fails:

### 1. Is the hypothesis wrong?

```rust
// Original hypothesis: "Exponent grouping gives 20% improvement"
// Reality: Only 8% improvement on quantized models

// GOOD: Revise hypothesis with new knowledge
#[test]
fn hypothesis_exponent_grouping_float_models() {
    // Only test on float models where hypothesis holds
}

#[test]
fn hypothesis_exponent_grouping_quantized_models() {
    // Lower expectation for quantized
    assert!(improvement >= 0.05);  // 5% not 20%
}
```

### 2. Is the implementation wrong?

```rust
// Hypothesis is sound, but we have a bug
// Add more specific invariant tests to catch it

#[test]
fn invariant_exponent_extraction_correct() {
    let f: f32 = 3.14159;
    let exp = extract_exponent(f);
    assert_eq!(exp, 128);  // Known value for 3.14159
}
```

### 3. Is the test data unrepresentative?

```rust
// Maybe our test models are weird
// Add exploration to understand data distribution

#[test]
#[ignore]
fn exploration_weight_distributions() {
    for model in all_test_models() {
        let stats = analyze_weight_distribution(&model);
        println!("{}: mean={}, std={}, skew={}", 
            model.name, stats.mean, stats.std, stats.skew);
    }
}
```

## Continuous Integration

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
    continue-on-error: true  # Don't block PR, but report
    steps:
      - run: cargo test --test hypotheses
      
  exploration:
    name: Exploration (Weekly)
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    steps:
      - run: cargo test exploration_ -- --ignored --nocapture > exploration.txt
      - uses: actions/upload-artifact@v3
        with:
          name: exploration-results
          path: exploration.txt
```

## Example: Building the Diff Algorithm

### Iteration 1: Make It Work

```rust
// Invariant (write first)
#[test]
fn invariant_diff_patch_roundtrip() {
    let old = b"hello world";
    let new = b"hello rust";
    let patch = diff(old, new);
    let result = apply_patch(old, &patch);
    assert_eq!(result, new);
}

// Implementation (simplest thing)
fn diff(old: &[u8], new: &[u8]) -> Vec<u8> {
    new.to_vec()  // Just store the whole thing
}

fn apply_patch(old: &[u8], patch: &[u8]) -> Vec<u8> {
    patch.to_vec()
}

// ✅ Invariant passes!
```

### Iteration 2: Make It Smaller

```rust
// Hypothesis (add expectation)
#[test]
fn hypothesis_diff_uses_xor() {
    let old = b"hello world";
    let new = b"hello rust!";
    let patch = diff(old, new);
    
    // XOR of similar data should compress well
    assert!(patch.len() < new.len());
}

// Implementation (XOR-based)
fn diff(old: &[u8], new: &[u8]) -> Vec<u8> {
    let xor: Vec<u8> = old.iter()
        .zip(new.iter())
        .map(|(o, n)| o ^ n)
        .collect();
    
    // Handle length difference
    let mut result = zstd::compress(&xor, 3).unwrap();
    if new.len() > old.len() {
        result.extend(&new[old.len()..]);
    }
    result
}

// ✅ Hypothesis passes!
```

### Iteration 3: Make It Smart

```rust
// Exploration (understand the landscape)
#[test]
#[ignore]
fn exploration_tensor_alignment_benefit() {
    // ... measure improvement from tensor-aware vs byte-level
}

// New hypothesis based on exploration
#[test]
fn hypothesis_tensor_aware_diff_smaller() {
    let old = load_tflite("v1.tflite");
    let new = load_tflite("v2.tflite");
    
    let byte_patch = byte_level_diff(&old, &new);
    let tensor_patch = tensor_aware_diff(&old, &new);
    
    assert!(tensor_patch.len() < byte_patch.len() * 0.7);  // 30% better
}

// Implementation (tensor-aware)
fn tensor_aware_diff(old: &TFLiteModel, new: &TFLiteModel) -> Vec<u8> {
    // Align tensors, diff each separately
    // ...
}

// ✅ Hypothesis passes!
```

## Summary

| Test Type | When to Write | Failure Means |
|-----------|---------------|---------------|
| **Invariant** | Before implementation | Bug in code |
| **Hypothesis** | After basic impl works | Assumption may be wrong |
| **Exploration** | When you need to learn | (Can't fail, just reports) |

The key insight: **research TDD is iterative hypothesis refinement**, not upfront specification. You're encoding your assumptions as tests, then letting reality correct them.
