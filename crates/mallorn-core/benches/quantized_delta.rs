//! Quantization-aware delta benchmarks
//!
//! Benchmarks block-aligned delta computation for quantized tensor data.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mallorn_core::{apply_xor_delta, xor_delta, DataType, QuantizationBlockInfo, QuantizedDelta};

/// Generate Q4_0-like data (2 bytes scale + 16 bytes data per block)
fn generate_q4_0_data(num_blocks: usize) -> Vec<u8> {
    let block_info = QuantizationBlockInfo::for_dtype(DataType::Q4_0).unwrap();
    let size = num_blocks * block_info.bytes_per_block;

    (0..size)
        .map(|i| {
            let block = i / block_info.bytes_per_block;
            let offset = i % block_info.bytes_per_block;
            if offset < 2 {
                // Scale factor bytes - simulate fp16 scale
                ((block + offset * 128) % 256) as u8
            } else {
                // Quantized data bytes
                ((i * 7 + 13) % 256) as u8
            }
        })
        .collect()
}

/// Generate Q4K-like data (K-quant 256-value superblocks)
fn generate_q4k_data(num_blocks: usize) -> Vec<u8> {
    let block_info = QuantizationBlockInfo::for_dtype(DataType::Q4K).unwrap();
    let size = num_blocks * block_info.bytes_per_block;

    (0..size)
        .map(|i| {
            let block = i / block_info.bytes_per_block;
            let offset = i % block_info.bytes_per_block;
            // First 12 bytes are scale/min factors, rest is data
            if offset < 12 {
                ((block + offset * 64) % 256) as u8
            } else {
                ((i * 11 + 7) % 256) as u8
            }
        })
        .collect()
}

/// Create a modified version of data (simulating weight updates)
fn modify_data(data: &[u8], modification_rate: f64) -> Vec<u8> {
    let mut new_data = data.to_vec();
    let num_modifications = (data.len() as f64 * modification_rate) as usize;

    for i in 0..num_modifications {
        let idx = (i * 17 + 3) % data.len();
        new_data[idx] = new_data[idx].wrapping_add(((i % 16) + 1) as u8);
    }

    new_data
}

fn bench_delta_compute(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_compute");

    // Test Q4_0 format
    let q4_0_blocks = [100, 1000, 10000];
    for num_blocks in q4_0_blocks {
        let block_info = QuantizationBlockInfo::for_dtype(DataType::Q4_0).unwrap();
        let size = num_blocks * block_info.bytes_per_block;
        group.throughput(Throughput::Bytes(size as u64));

        let old = generate_q4_0_data(num_blocks);
        let new = modify_data(&old, 0.1); // 10% modification

        // Regular XOR delta
        group.bench_with_input(
            BenchmarkId::new(format!("q4_0_regular_{}_blocks", num_blocks), size),
            &(&old, &new),
            |b, (old, new)| {
                b.iter(|| xor_delta(black_box(old), black_box(new)))
            },
        );

        // Block-aligned delta
        group.bench_with_input(
            BenchmarkId::new(format!("q4_0_aligned_{}_blocks", num_blocks), size),
            &(&old, &new),
            |b, (old, new)| {
                b.iter(|| {
                    QuantizedDelta::compute_block_aligned(black_box(old), black_box(new), &block_info)
                })
            },
        );

        // Smart delta (auto-detect)
        group.bench_with_input(
            BenchmarkId::new(format!("q4_0_smart_{}_blocks", num_blocks), size),
            &(&old, &new),
            |b, (old, new)| {
                b.iter(|| QuantizedDelta::compute(black_box(old), black_box(new), DataType::Q4_0))
            },
        );
    }

    // Test Q4K format (K-quant)
    let q4k_blocks = [10, 100, 1000];
    for num_blocks in q4k_blocks {
        let block_info = QuantizationBlockInfo::for_dtype(DataType::Q4K).unwrap();
        let size = num_blocks * block_info.bytes_per_block;
        group.throughput(Throughput::Bytes(size as u64));

        let old = generate_q4k_data(num_blocks);
        let new = modify_data(&old, 0.1);

        group.bench_with_input(
            BenchmarkId::new(format!("q4k_regular_{}_blocks", num_blocks), size),
            &(&old, &new),
            |b, (old, new)| {
                b.iter(|| xor_delta(black_box(old), black_box(new)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("q4k_aligned_{}_blocks", num_blocks), size),
            &(&old, &new),
            |b, (old, new)| {
                b.iter(|| {
                    QuantizedDelta::compute_block_aligned(black_box(old), black_box(new), &block_info)
                })
            },
        );
    }

    group.finish();
}

fn bench_delta_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_apply");

    let q4_0_blocks = [100, 1000, 10000];
    for num_blocks in q4_0_blocks {
        let block_info = QuantizationBlockInfo::for_dtype(DataType::Q4_0).unwrap();
        let size = num_blocks * block_info.bytes_per_block;
        group.throughput(Throughput::Bytes(size as u64));

        let old = generate_q4_0_data(num_blocks);
        let new = modify_data(&old, 0.1);
        let delta = xor_delta(&old, &new);

        // Regular XOR apply
        group.bench_with_input(
            BenchmarkId::new(format!("q4_0_regular_{}_blocks", num_blocks), size),
            &(&old, &delta),
            |b, (old, delta)| {
                b.iter(|| apply_xor_delta(black_box(old), black_box(delta)))
            },
        );

        // Block-aligned apply
        group.bench_with_input(
            BenchmarkId::new(format!("q4_0_aligned_{}_blocks", num_blocks), size),
            &(&old, &delta),
            |b, (old, delta)| {
                b.iter(|| {
                    QuantizedDelta::apply_block_aligned(black_box(old), black_box(delta), &block_info)
                })
            },
        );
    }

    group.finish();
}

fn bench_separated_delta(c: &mut Criterion) {
    let mut group = c.benchmark_group("separated_delta");

    // Benchmark the separated scale/data delta computation
    let q4_0_blocks = [100, 1000, 10000];
    let scale_bytes = 2; // Q4_0 has 2-byte scale

    for num_blocks in q4_0_blocks {
        let block_info = QuantizationBlockInfo::for_dtype(DataType::Q4_0).unwrap();
        let size = num_blocks * block_info.bytes_per_block;
        group.throughput(Throughput::Bytes(size as u64));

        let old = generate_q4_0_data(num_blocks);
        let new = modify_data(&old, 0.1);

        // Separated compute
        group.bench_with_input(
            BenchmarkId::new(format!("q4_0_separated_compute_{}_blocks", num_blocks), size),
            &(&old, &new),
            |b, (old, new)| {
                b.iter(|| {
                    QuantizedDelta::compute_separated(
                        black_box(old),
                        black_box(new),
                        &block_info,
                        scale_bytes,
                    )
                })
            },
        );

        // Pre-compute for apply benchmark
        let (scale_delta, data_delta) =
            QuantizedDelta::compute_separated(&old, &new, &block_info, scale_bytes);

        // Separated apply
        group.bench_with_input(
            BenchmarkId::new(format!("q4_0_separated_apply_{}_blocks", num_blocks), size),
            &(&old, &scale_delta, &data_delta),
            |b, (old, scale_delta, data_delta)| {
                b.iter(|| {
                    QuantizedDelta::apply_separated(
                        black_box(old),
                        black_box(scale_delta),
                        black_box(data_delta),
                        &block_info,
                        scale_bytes,
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_quantization_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization_detection");

    // Benchmark block info lookup
    let dtypes = [
        DataType::Q4_0,
        DataType::Q4_1,
        DataType::Q8_0,
        DataType::Q4K,
        DataType::Q6K,
        DataType::Float32,
    ];

    for dtype in dtypes {
        group.bench_with_input(
            BenchmarkId::new(format!("{:?}", dtype), 0),
            &dtype,
            |b, dtype| {
                b.iter(|| QuantizationBlockInfo::for_dtype(black_box(*dtype)))
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_delta_compute,
    bench_delta_apply,
    bench_separated_delta,
    bench_quantization_detection
);
criterion_main!(benches);
