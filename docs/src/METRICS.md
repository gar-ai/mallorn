# Mallorn Metrics

Success criteria and benchmarks for edge model delta updates.

## Primary Metrics

### Patch Size Ratio

**The core value proposition.** How small can we make patches?

| Metric | Target | Measurement |
|--------|--------|-------------|
| Minor update (same arch, tuned weights) | ≤5% of model | `patch_size / model_size` |
| Major update (architecture change) | ≤50% of model | `patch_size / model_size` |
| Worst case (completely different) | ≤100% of model | Never larger than full model |

**Benchmark models:**

| Model Pair | Expected Ratio |
|------------|----------------|
| MobileNet v1 → v1 fine-tuned | <5% |
| MobileNet v1 → v2 | 30-50% |
| YOLOv5s → YOLOv5m | 40-60% |
| Llama-7B-Q4 → tuned | <10% |

### Application Speed

How fast can we apply patches on target devices?

| Device Class | Target | Measurement |
|--------------|--------|-------------|
| Server (M3 Pro) | >100 MB/s | `model_size / apply_time` |
| Edge (Jetson) | >50 MB/s | `model_size / apply_time` |
| MCU (ESP32) | >1 MB/s | `model_size / apply_time` |
| Tiny MCU (STM32L4) | >500 KB/s | `model_size / apply_time` |

### Memory Usage

Critical for embedded deployment.

| Device Class | Peak RAM Target | Measurement |
|--------------|-----------------|-------------|
| Server | <1GB | `peak_rss` during apply |
| Edge | <100MB | `peak_rss` during apply |
| MCU (streaming) | <4KB | Buffer size + state |
| Tiny MCU (streaming) | <1KB | Buffer size + state |

### Generation Speed

How fast can we create patches (server-side)?

| Model Size | Target | Measurement |
|------------|--------|-------------|
| <10MB | <1s | `diff_time` |
| 10-100MB | <10s | `diff_time` |
| 100MB-1GB | <60s | `diff_time` |
| >1GB | <5min | `diff_time` |

---

## Benchmark Baselines

### Comparison Targets

| Tool | Type | Our Target |
|------|------|------------|
| **bsdiff** | Generic binary diff | Match or beat |
| **xdelta3** | Generic binary diff | Match or beat |
| **zstd** | Compression only | 20%+ smaller patches |
| **Full model** | No diffing | Always smaller |

### Baseline Measurements

Run these to establish baselines before optimization:

```bash
# Generate baseline report
mallorn benchmark --baseline \
    --models fixtures/models/ \
    --output baseline_report.json
```

Expected baseline report:

```json
{
  "mobilenet_v1_to_v2": {
    "model_size": 4200000,
    "bsdiff_patch": 2100000,
    "xdelta_patch": 2300000,
    "zstd_only": 1800000,
    "mallorn_v0.1": 1500000,
    "mallorn_v0.2_neural": 1200000
  }
}
```

---

## Benchmark Suite

### Standard Benchmark Set

```rust
// benches/standard_suite.rs

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

fn bench_diff_speed(c: &mut Criterion) {
    let models = vec![
        ("tiny_1mb", load_model_pair("tiny")),
        ("medium_10mb", load_model_pair("medium")),
        ("large_100mb", load_model_pair("large")),
    ];
    
    for (name, (old, new)) in models {
        let mut group = c.benchmark_group(format!("diff/{}", name));
        group.throughput(Throughput::Bytes(new.len() as u64));
        
        group.bench_function("mallorn", |b| {
            b.iter(|| diff(&old, &new))
        });
        
        group.bench_function("bsdiff", |b| {
            b.iter(|| bsdiff::diff(&old, &new))
        });
        
        group.finish();
    }
}

fn bench_patch_speed(c: &mut Criterion) {
    let cases = vec![
        ("tiny", load_patch_case("tiny")),
        ("medium", load_patch_case("medium")),
        ("large", load_patch_case("large")),
    ];
    
    for (name, (old, patch)) in cases {
        let mut group = c.benchmark_group(format!("patch/{}", name));
        group.throughput(Throughput::Bytes(old.len() as u64));
        
        group.bench_function("batch", |b| {
            b.iter(|| apply_patch(&old, &patch))
        });
        
        group.bench_function("streaming_1kb", |b| {
            b.iter(|| apply_patch_streaming(&old, &patch, 1024))
        });
        
        group.bench_function("streaming_4kb", |b| {
            b.iter(|| apply_patch_streaming(&old, &patch, 4096))
        });
        
        group.finish();
    }
}

fn bench_compression(c: &mut Criterion) {
    let delta = generate_weight_delta(1_000_000);  // 1M floats
    
    let mut group = c.benchmark_group("compression");
    group.throughput(Throughput::Bytes(delta.len() as u64));
    
    group.bench_function("zstd_3", |b| {
        b.iter(|| compress_zstd(&delta, 3))
    });
    
    group.bench_function("neural", |b| {
        b.iter(|| compress_neural(&delta))
    });
    
    group.finish();
}

criterion_group!(benches, bench_diff_speed, bench_patch_speed, bench_compression);
criterion_main!(benches);
```

### Memory Benchmark

```rust
// benches/memory.rs

fn measure_peak_memory<F, R>(f: F) -> (R, usize)
where
    F: FnOnce() -> R,
{
    // Reset allocator stats
    reset_alloc_stats();
    
    let result = f();
    
    let peak = get_peak_allocated();
    (result, peak)
}

#[test]
fn memory_streaming_1kb() {
    let old = load_test_model("large.tflite");
    let patch = load_test_patch("large.tflp");
    
    let (_, peak) = measure_peak_memory(|| {
        apply_patch_streaming(&old, &patch, 1024)
    });
    
    // Should be ~1KB buffer + small overhead
    assert!(peak < 4096, "Peak memory {} exceeds 4KB", peak);
}
```

---

## Real-World Scenarios

### Scenario 1: IoT Sensor Update

```
Device: ESP32 with LoRa
Model: 200KB TFLite classifier
Network: 250 bps effective (LoRa SF12)
```

| Metric | Full Model | Mallorn Patch |
|--------|------------|---------------|
| Transfer size | 200KB | 10KB (5%) |
| Transfer time | 6.4 hours | 19 minutes |
| Apply RAM | N/A | 1KB |
| Apply time | N/A | 200ms |

### Scenario 2: Edge Camera Update

```
Device: Jetson Nano
Model: 50MB YOLO model
Network: 1 Mbps cellular
```

| Metric | Full Model | Mallorn Patch |
|--------|------------|---------------|
| Transfer size | 50MB | 2.5MB (5%) |
| Data cost | $0.50 | $0.025 |
| Transfer time | 7 minutes | 20 seconds |
| Apply RAM | 50MB | 10MB |
| Apply time | N/A | 500ms |

### Scenario 3: Fleet Deployment

```
Devices: 10,000 edge devices
Model: 500MB ONNX model
Network: Mixed (WiFi/cellular)
```

| Metric | Full Model | Mallorn Patch |
|--------|------------|---------------|
| Total transfer | 5 PB | 250 TB |
| CDN cost | $500 | $25 |
| Fleet update time | 2 days | 4 hours |

---

## Performance Regression Testing

### Automated Checks

```yaml
# .github/workflows/perf.yml
name: Performance Regression

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run benchmarks
        run: cargo bench -- --save-baseline pr
        
      - name: Compare to main
        run: |
          git checkout main
          cargo bench -- --save-baseline main
          git checkout -
          cargo bench -- --baseline main --compare pr
          
      - name: Check regression
        run: |
          # Fail if >10% regression
          python scripts/check_regression.py --threshold 0.10
```

### Regression Thresholds

| Metric | Max Regression | Action |
|--------|----------------|--------|
| Patch size | +5% | Warning |
| Patch size | +10% | Block PR |
| Apply speed | -10% | Warning |
| Apply speed | -20% | Block PR |
| Memory | +20% | Warning |
| Memory | +50% | Block PR |

---

## Reporting

### Benchmark Report Format

```json
{
  "version": "0.1.0",
  "timestamp": "2024-12-27T00:00:00Z",
  "hardware": {
    "cpu": "Apple M3 Pro",
    "ram": "18GB",
    "os": "macOS 14.0"
  },
  "results": {
    "diff": {
      "mobilenet_4mb": {
        "time_ms": 120,
        "patch_size": 180000,
        "ratio": 0.043
      }
    },
    "patch": {
      "mobilenet_4mb": {
        "batch_ms": 15,
        "streaming_1kb_ms": 45,
        "peak_memory_kb": 1.2
      }
    },
    "compression": {
      "neural_vs_zstd": {
        "improvement": 0.23
      }
    }
  }
}
```

### Dashboard Metrics

Track over time:
- Patch size ratios by model type
- Application speed by device class
- Memory usage trends
- Compression improvement (neural vs baseline)

---

## Definition of Done

### MVP (v0.1)
- [ ] Patch ratio <10% for minor updates
- [ ] Apply speed >50 MB/s on M3 Pro
- [ ] Streaming works with 1KB buffer
- [ ] All invariant tests pass

### v0.2 (Neural Compression)
- [ ] 20%+ improvement over zstd baseline
- [ ] Decompression >50 MB/s
- [ ] Hypothesis tests pass

### v0.3 (Embedded)
- [ ] ESP32 example works
- [ ] STM32 example works
- [ ] <1KB RAM verified on real hardware

### v1.0 (Production)
- [ ] Beat bsdiff on all test models
- [ ] <5% patch for fine-tuning updates
- [ ] Documentation of all benchmarks
