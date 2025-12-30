//! Sparse tensor encoding/decoding benchmarks
//!
//! Benchmarks CSR sparse encoding performance at various sparsity levels.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mallorn_core::{Compressor, DataType, SparseCompressor, SparseEncoder, ZstdCompressor};

/// Generate sparse float32 data with specified sparsity ratio
fn generate_sparse_data(rows: usize, cols: usize, sparsity: f64) -> Vec<u8> {
    let num_elements = rows * cols;
    let mut data = vec![0u8; num_elements * 4];

    // Calculate how many elements should be non-zero
    let non_zero_count = ((1.0 - sparsity) * num_elements as f64) as usize;

    // Distribute non-zero values
    for i in 0..non_zero_count {
        let idx = (i * 7 + 13) % num_elements; // Pseudo-random distribution
        let value = (i as f32 + 1.0) * 0.01;
        let bytes = value.to_le_bytes();
        data[idx * 4..idx * 4 + 4].copy_from_slice(&bytes);
    }

    data
}

fn bench_sparse_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_encode");

    let shapes = [(100, 100), (256, 256), (512, 512), (1024, 1024)];
    let sparsities = [0.5, 0.7, 0.9, 0.95];

    for (rows, cols) in shapes {
        let size = rows * cols * 4;
        group.throughput(Throughput::Bytes(size as u64));

        for sparsity in sparsities {
            let data = generate_sparse_data(rows, cols, sparsity);
            let sparsity_pct = (sparsity * 100.0) as u32;

            group.bench_with_input(
                BenchmarkId::new(
                    format!("csr_{}x{}_{}_sparse", rows, cols, sparsity_pct),
                    size,
                ),
                &(&data, rows, cols),
                |b, (data, rows, cols)| {
                    b.iter(|| SparseEncoder::encode_csr(black_box(data), 4, &[*rows, *cols]))
                },
            );
        }
    }

    group.finish();
}

fn bench_sparse_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_decode");

    let shapes = [(100, 100), (256, 256), (512, 512)];
    let sparsities = [0.7, 0.9, 0.95];

    for (rows, cols) in shapes {
        let size = rows * cols * 4;
        group.throughput(Throughput::Bytes(size as u64));

        for sparsity in sparsities {
            let data = generate_sparse_data(rows, cols, sparsity);
            let sparsity_pct = (sparsity * 100.0) as u32;

            // Pre-encode for decode benchmark
            let sparse = SparseEncoder::encode_csr(&data, 4, &[rows, cols]).unwrap();

            group.bench_with_input(
                BenchmarkId::new(
                    format!("csr_{}x{}_{}_sparse", rows, cols, sparsity_pct),
                    size,
                ),
                &sparse,
                |b, sparse| b.iter(|| SparseEncoder::decode_csr(black_box(sparse))),
            );
        }
    }

    group.finish();
}

fn bench_sparse_vs_dense_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_vs_dense");

    // Compare compression ratios and speeds
    let (rows, cols) = (256, 256);
    let size = rows * cols * 4;
    group.throughput(Throughput::Bytes(size as u64));

    for sparsity in [0.5, 0.7, 0.9, 0.95] {
        let data = generate_sparse_data(rows, cols, sparsity);
        let sparsity_pct = (sparsity * 100.0) as u32;

        // Dense (standard zstd)
        let dense_compressor = ZstdCompressor::new(3);
        group.bench_with_input(
            BenchmarkId::new(format!("dense_{}pct", sparsity_pct), size),
            &data,
            |b, data| b.iter(|| dense_compressor.compress(black_box(data), DataType::Float32)),
        );

        // Sparse-aware compression
        let sparse_compressor = SparseCompressor::new(3);
        group.bench_with_input(
            BenchmarkId::new(format!("sparse_{}pct", sparsity_pct), size),
            &data,
            |b, data| b.iter(|| sparse_compressor.compress(black_box(data), DataType::Float32)),
        );

        // Sparse-aware with shape (CSR path)
        group.bench_with_input(
            BenchmarkId::new(format!("sparse_csr_{}pct", sparsity_pct), size),
            &(&data, rows, cols),
            |b, (data, rows, cols)| {
                b.iter(|| {
                    sparse_compressor.compress_with_shape(
                        black_box(data),
                        DataType::Float32,
                        &[*rows, *cols],
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_sparsity_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparsity_detection");

    let sizes = [1024 * 4, 64 * 1024, 256 * 1024, 1024 * 1024];

    for size in sizes {
        let data = generate_sparse_data(size / 16, 4, 0.7);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("calculate_sparsity", size),
            &data,
            |b, data| b.iter(|| SparseEncoder::calculate_sparsity(black_box(data), 4)),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_sparse_encode,
    bench_sparse_decode,
    bench_sparse_vs_dense_compression,
    bench_sparsity_detection
);
criterion_main!(benches);
