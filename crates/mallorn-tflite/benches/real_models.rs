//! Real model benchmarks for TFLite
//!
//! Benchmarks using actual MobileNet models to measure real-world performance.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mallorn_tflite::{TFLiteDiffer, TFLitePatcher, TFLiteParser};
use std::fs;
use std::path::PathBuf;

fn get_fixtures_path() -> PathBuf {
    // Get the manifest directory and navigate to fixtures
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(manifest)
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("models")
}

fn mobilenet_v1_path() -> PathBuf {
    get_fixtures_path().join("mobilenet_v1.tflite")
}

fn mobilenet_v1_quant_path() -> PathBuf {
    get_fixtures_path().join("mobilenet_v1_quant.tflite")
}

fn mobilenet_v2_path() -> PathBuf {
    get_fixtures_path().join("mobilenet_v2.tflite")
}

fn bench_parse_real_model(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse_real_model");

    // Load models
    let models = [
        ("mobilenet_v1", mobilenet_v1_path()),
        ("mobilenet_v1_quant", mobilenet_v1_quant_path()),
        ("mobilenet_v2", mobilenet_v2_path()),
    ];

    for (name, path) in models {
        if let Ok(data) = fs::read(&path) {
            let size = data.len();
            group.throughput(Throughput::Bytes(size as u64));

            group.bench_with_input(BenchmarkId::new("parse", name), &data, |b, data| {
                let parser = TFLiteParser::new();
                b.iter(|| parser.parse(data))
            });
        }
    }

    group.finish();
}

fn bench_diff_real_models(c: &mut Criterion) {
    let mut group = c.benchmark_group("diff_real_models");
    group.sample_size(10); // Fewer samples for large models

    // Load model pairs
    let v1 = fs::read(mobilenet_v1_path()).ok();
    let v1_quant = fs::read(mobilenet_v1_quant_path()).ok();
    let v2 = fs::read(mobilenet_v2_path()).ok();

    if let (Some(v1), Some(v1_quant)) = (&v1, &v1_quant) {
        let size = v1.len() + v1_quant.len();
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_function("v1_to_v1_quant", |b| {
            let differ = TFLiteDiffer::new();
            b.iter(|| differ.diff_from_bytes(v1, v1_quant))
        });
    }

    if let (Some(v1), Some(v2)) = (&v1, &v2) {
        let size = v1.len() + v2.len();
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_function("v1_to_v2", |b| {
            let differ = TFLiteDiffer::new();
            b.iter(|| differ.diff_from_bytes(v1, v2))
        });
    }

    group.finish();
}

fn bench_compression_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_ratio");
    group.sample_size(10);

    let v1 = fs::read(mobilenet_v1_path()).ok();
    let v1_quant = fs::read(mobilenet_v1_quant_path()).ok();
    let v2 = fs::read(mobilenet_v2_path()).ok();

    // Report compression ratios
    if let (Some(v1), Some(v1_quant)) = (&v1, &v1_quant) {
        let differ = TFLiteDiffer::new();
        if let Ok(patch) = differ.diff_from_bytes(v1, v1_quant) {
            let patch_bytes = serde_json::to_vec(&patch).unwrap_or_default();
            let ratio = (patch_bytes.len() as f64 / v1_quant.len() as f64) * 100.0;
            println!("\n=== v1 -> v1_quant ===");
            println!("Source: {} bytes", v1.len());
            println!("Target: {} bytes", v1_quant.len());
            println!("Patch:  {} bytes", patch_bytes.len());
            println!("Ratio:  {:.1}%", ratio);
        }
    }

    if let (Some(v1), Some(v2)) = (&v1, &v2) {
        let differ = TFLiteDiffer::new();
        if let Ok(patch) = differ.diff_from_bytes(v1, v2) {
            let patch_bytes = serde_json::to_vec(&patch).unwrap_or_default();
            let ratio = (patch_bytes.len() as f64 / v2.len() as f64) * 100.0;
            println!("\n=== v1 -> v2 ===");
            println!("Source: {} bytes", v1.len());
            println!("Target: {} bytes", v2.len());
            println!("Patch:  {} bytes", patch_bytes.len());
            println!("Ratio:  {:.1}%", ratio);
        }
    }

    // Dummy benchmark to show ratios
    group.bench_function("report_only", |b| b.iter(|| 1 + 1));
    group.finish();
}

fn bench_patch_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("patch_apply");
    group.sample_size(10);

    let v1 = fs::read(mobilenet_v1_path()).ok();
    let v1_quant = fs::read(mobilenet_v1_quant_path()).ok();

    if let (Some(v1), Some(v1_quant)) = (&v1, &v1_quant) {
        let differ = TFLiteDiffer::new();
        if let Ok(patch) = differ.diff_from_bytes(&v1, &v1_quant) {
            let size = v1.len();
            group.throughput(Throughput::Bytes(size as u64));

            group.bench_function("apply_v1_to_v1_quant", |b| {
                let patcher = TFLitePatcher::new();
                b.iter(|| patcher.apply(&v1, &patch))
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_parse_real_model,
    bench_diff_real_models,
    bench_compression_ratio,
    bench_patch_apply
);
criterion_main!(benches);
