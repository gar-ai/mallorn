//! Parallel compression benchmarks for Mallorn
//!
//! Benchmarks parallel vs sequential tensor compression.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mallorn_core::{Compressor, DataType, NeuralCompressor, ZstdCompressor};
use rayon::prelude::*;
use std::sync::Arc;

/// Generate synthetic tensor data that mimics neural network weights
fn generate_tensor_data(size: usize) -> Vec<u8> {
    let mut data = vec![0u8; size];
    for i in 0..size / 4 {
        let base_idx = i * 4;
        data[base_idx] = 0x3F;
        data[base_idx + 1] = ((i % 256) as u8).wrapping_add(0x80);
        data[base_idx + 2] = (i % 128) as u8;
        data[base_idx + 3] = (i % 64) as u8;
    }
    data
}

/// Simulate multiple tensors from a model
fn generate_model_tensors(num_tensors: usize, tensor_size: usize) -> Vec<Vec<u8>> {
    (0..num_tensors)
        .map(|i| {
            let mut data = generate_tensor_data(tensor_size);
            // Add variation between tensors
            for j in 0..std::cmp::min(100, data.len()) {
                data[j] = data[j].wrapping_add((i % 256) as u8);
            }
            data
        })
        .collect()
}

fn bench_parallel_vs_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_compression");

    // Test with different numbers of tensors
    let configs = [
        (4, 256 * 1024),  // 4 tensors x 256KB = 1MB
        (16, 256 * 1024), // 16 tensors x 256KB = 4MB
        (32, 256 * 1024), // 32 tensors x 256KB = 8MB
        (64, 128 * 1024), // 64 tensors x 128KB = 8MB
    ];

    for (num_tensors, tensor_size) in configs {
        let tensors = generate_model_tensors(num_tensors, tensor_size);
        let total_bytes = num_tensors * tensor_size;
        group.throughput(Throughput::Bytes(total_bytes as u64));

        let label = format!("{}x{}KB", num_tensors, tensor_size / 1024);

        // Sequential compression with Zstd
        let zstd = ZstdCompressor::new(3);
        group.bench_with_input(
            BenchmarkId::new("zstd_sequential", &label),
            &tensors,
            |b, tensors| {
                b.iter(|| {
                    tensors
                        .iter()
                        .map(|t| zstd.compress(black_box(t), DataType::Float32))
                        .collect::<Vec<_>>()
                })
            },
        );

        // Parallel compression with Zstd
        let zstd_arc: Arc<dyn Compressor + Send + Sync> = Arc::new(ZstdCompressor::new(3));
        group.bench_with_input(
            BenchmarkId::new("zstd_parallel", &label),
            &tensors,
            |b, tensors| {
                b.iter(|| {
                    tensors
                        .par_iter()
                        .map(|t| zstd_arc.compress(black_box(t), DataType::Float32))
                        .collect::<Vec<_>>()
                })
            },
        );

        // Sequential with Neural compression
        let neural = NeuralCompressor::new(3);
        group.bench_with_input(
            BenchmarkId::new("neural_sequential", &label),
            &tensors,
            |b, tensors| {
                b.iter(|| {
                    tensors
                        .iter()
                        .map(|t| neural.compress(black_box(t), DataType::Float32))
                        .collect::<Vec<_>>()
                })
            },
        );

        // Parallel with Neural compression
        let neural_arc: Arc<dyn Compressor + Send + Sync> = Arc::new(NeuralCompressor::new(3));
        group.bench_with_input(
            BenchmarkId::new("neural_parallel", &label),
            &tensors,
            |b, tensors| {
                b.iter(|| {
                    tensors
                        .par_iter()
                        .map(|t| neural_arc.compress(black_box(t), DataType::Float32))
                        .collect::<Vec<_>>()
                })
            },
        );
    }

    group.finish();
}

fn bench_parallel_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_scaling");

    // Fixed total work, varying parallelism
    let total_size = 8 * 1024 * 1024; // 8MB total
    let tensor_counts = [1, 2, 4, 8, 16, 32];

    for num_tensors in tensor_counts {
        let tensor_size = total_size / num_tensors;
        let tensors = generate_model_tensors(num_tensors, tensor_size);
        group.throughput(Throughput::Bytes(total_size as u64));

        let zstd: Arc<dyn Compressor + Send + Sync> = Arc::new(ZstdCompressor::new(3));

        group.bench_with_input(
            BenchmarkId::new("parallel_tensors", num_tensors),
            &tensors,
            |b, tensors| {
                b.iter(|| {
                    tensors
                        .par_iter()
                        .map(|t| zstd.compress(black_box(t), DataType::Float32))
                        .collect::<Vec<_>>()
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_parallel_vs_sequential,
    bench_parallel_scaling
);
criterion_main!(benches);
