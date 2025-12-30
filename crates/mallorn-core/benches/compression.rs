//! Compression benchmarks for Mallorn
//!
//! Benchmarks compression and decompression performance across
//! different compressors and data patterns.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mallorn_core::{Compressor, DataType, Lz4Compressor, NeuralCompressor, ZstdCompressor};

/// Generate synthetic tensor data that mimics neural network weights
fn generate_weight_data(size: usize) -> Vec<u8> {
    let mut data = vec![0u8; size];
    // Simulate FP32 weights with small variations
    for i in 0..size / 4 {
        let base_idx = i * 4;
        // Exponent byte (common in neural weights)
        data[base_idx] = 0x3F; // ~1.0 exponent range
        data[base_idx + 1] = ((i % 256) as u8).wrapping_add(0x80);
        data[base_idx + 2] = (i % 128) as u8;
        data[base_idx + 3] = (i % 64) as u8;
    }
    data
}

/// Generate sparse delta data (many zeros)
fn generate_sparse_delta(size: usize, sparsity: f64) -> Vec<u8> {
    let mut data = vec![0u8; size];
    let non_zero_count = ((1.0 - sparsity) * size as f64) as usize;
    for i in 0..non_zero_count {
        let idx = (i * 7) % size; // Pseudo-random distribution
        data[idx] = ((i % 255) + 1) as u8;
    }
    data
}

fn bench_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression");

    // Test different data sizes
    let sizes = [1024, 16 * 1024, 256 * 1024, 1024 * 1024];

    for size in sizes {
        let weight_data = generate_weight_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        // Zstd compression at different levels
        for level in [1, 3, 9] {
            let compressor = ZstdCompressor::new(level);
            group.bench_with_input(
                BenchmarkId::new(format!("zstd_l{}_compress", level), size),
                &weight_data,
                |b, data| {
                    b.iter(|| compressor.compress(black_box(data), DataType::Float32))
                },
            );
        }

        // LZ4 compression
        let lz4 = Lz4Compressor::new();
        group.bench_with_input(
            BenchmarkId::new("lz4_compress", size),
            &weight_data,
            |b, data| {
                b.iter(|| lz4.compress(black_box(data), DataType::Float32))
            },
        );

        // Neural compression
        let neural = NeuralCompressor::new(3);
        group.bench_with_input(
            BenchmarkId::new("neural_compress", size),
            &weight_data,
            |b, data| {
                b.iter(|| neural.compress(black_box(data), DataType::Float32))
            },
        );
    }

    group.finish();
}

fn bench_decompression(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompression");

    let sizes = [1024, 16 * 1024, 256 * 1024, 1024 * 1024];

    for size in sizes {
        let weight_data = generate_weight_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        // Prepare compressed data
        let zstd = ZstdCompressor::new(3);
        let lz4 = Lz4Compressor::new();
        let neural = NeuralCompressor::new(3);

        let zstd_compressed = zstd.compress(&weight_data, DataType::Float32).unwrap();
        let lz4_compressed = lz4.compress(&weight_data, DataType::Float32).unwrap();
        let neural_compressed = neural.compress(&weight_data, DataType::Float32).unwrap();

        // Zstd decompression
        group.bench_with_input(
            BenchmarkId::new("zstd_decompress", size),
            &zstd_compressed,
            |b, data| {
                b.iter(|| zstd.decompress(black_box(data), DataType::Float32))
            },
        );

        // LZ4 decompression
        group.bench_with_input(
            BenchmarkId::new("lz4_decompress", size),
            &lz4_compressed,
            |b, data| {
                b.iter(|| lz4.decompress(black_box(data), DataType::Float32))
            },
        );

        // Neural decompression
        group.bench_with_input(
            BenchmarkId::new("neural_decompress", size),
            &neural_compressed,
            |b, data| {
                b.iter(|| neural.decompress(black_box(data), DataType::Float32))
            },
        );
    }

    group.finish();
}

fn bench_sparse_delta_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_delta");

    let size = 256 * 1024; // 256KB
    let sparsities = [0.5, 0.8, 0.95, 0.99];

    for sparsity in sparsities {
        let delta = generate_sparse_delta(size, sparsity);
        let sparsity_pct = (sparsity * 100.0) as u32;
        group.throughput(Throughput::Bytes(size as u64));

        let zstd = ZstdCompressor::new(3);
        group.bench_with_input(
            BenchmarkId::new(format!("zstd_{}pct_sparse", sparsity_pct), size),
            &delta,
            |b, data| {
                b.iter(|| zstd.compress(black_box(data), DataType::Float32))
            },
        );

        let neural = NeuralCompressor::new(3);
        group.bench_with_input(
            BenchmarkId::new(format!("neural_{}pct_sparse", sparsity_pct), size),
            &delta,
            |b, data| {
                b.iter(|| neural.compress(black_box(data), DataType::Float32))
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_compression,
    bench_decompression,
    bench_sparse_delta_compression
);
criterion_main!(benches);
