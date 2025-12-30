//! Fingerprint benchmarks for Mallorn
//!
//! Benchmarks model fingerprint generation performance.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mallorn_core::ModelFingerprint;
use std::io::Write;
use tempfile::NamedTempFile;

/// Generate synthetic model data
fn generate_model_data(size: usize) -> Vec<u8> {
    let mut data = vec![0u8; size];
    // Add some structure to simulate a real model
    // Magic bytes at start
    data[0..4].copy_from_slice(&[0x54, 0x46, 0x4C, 0x33]); // "TFL3"

    // Fill with pseudo-random but deterministic data
    for i in 4..size {
        data[i] = ((i * 7 + 13) % 256) as u8;
    }
    data
}

fn bench_fingerprint_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("fingerprint");

    // Test different model sizes
    let sizes = [
        (64 * 1024, "64KB"),
        (1024 * 1024, "1MB"),
        (10 * 1024 * 1024, "10MB"),
        (100 * 1024 * 1024, "100MB"),
    ];

    for (size, label) in sizes {
        let data = generate_model_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        // Create temp file for fingerprint testing
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&data).unwrap();
        temp_file.flush().unwrap();
        let path = temp_file.path().to_owned();

        group.bench_with_input(
            BenchmarkId::new("from_file", label),
            &path,
            |b, path| {
                b.iter(|| ModelFingerprint::from_file(black_box(path)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("from_bytes", label),
            &data,
            |b, data| {
                b.iter(|| ModelFingerprint::from_bytes(black_box(data)))
            },
        );
    }

    group.finish();
}

fn bench_fingerprint_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("fingerprint_compare");

    // Create two similar model files
    let size = 10 * 1024 * 1024; // 10MB
    let data1 = generate_model_data(size);
    let mut data2 = data1.clone();
    // Modify a small portion to create a different fingerprint
    for i in 0..1000 {
        data2[size / 2 + i] = ((i * 3) % 256) as u8;
    }

    let fp1 = ModelFingerprint::from_bytes(&data1).unwrap();
    let fp2 = ModelFingerprint::from_bytes(&data2).unwrap();

    group.bench_function("identical_fingerprints", |b| {
        b.iter(|| black_box(&fp1) == black_box(&fp1))
    });

    group.bench_function("different_fingerprints", |b| {
        b.iter(|| black_box(&fp1) == black_box(&fp2))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_fingerprint_generation,
    bench_fingerprint_comparison
);
criterion_main!(benches);
