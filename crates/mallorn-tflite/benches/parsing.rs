//! TFLite parsing benchmarks
//!
//! Benchmarks parsing and serialization performance for TFLite models.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mallorn_tflite::{TFLiteDiffer, TFLiteParser, TFLitePatcher};

/// Generate synthetic TFLite-like data for benchmarking
/// Real TFLite files would be used in production benchmarks
fn generate_synthetic_model(num_tensors: usize, tensor_size: usize) -> Vec<u8> {
    // Generate data that exercises the parser
    // This is synthetic - real benchmarks should use actual model files
    let mut data = Vec::with_capacity(num_tensors * tensor_size + 1024);

    // TFLite header (simplified for benchmark)
    data.extend_from_slice(&[0u8; 4]); // Identifier offset placeholder
    data.extend_from_slice(&(8u32).to_le_bytes()); // Root table offset

    // Padding to reach identifier
    data.resize(8, 0);
    data.extend_from_slice(b"TFL3"); // TFLite magic at offset

    // Add tensor data
    for _ in 0..num_tensors {
        data.extend(vec![0u8; tensor_size]);
    }

    data
}

fn bench_parser_creation(c: &mut Criterion) {
    c.bench_function("parser_creation", |b| b.iter(TFLiteParser::new));
}

fn bench_differ_creation(c: &mut Criterion) {
    c.bench_function("differ_creation", |b| b.iter(TFLiteDiffer::new));
}

fn bench_patcher_creation(c: &mut Criterion) {
    c.bench_function("patcher_creation", |b| b.iter(TFLitePatcher::new));
}

fn bench_synthetic_diff(c: &mut Criterion) {
    let mut group = c.benchmark_group("synthetic_diff");

    // Different model sizes
    let configs = [
        (10, 1024, "10_tensors_1kb"),
        (50, 4096, "50_tensors_4kb"),
        (100, 16384, "100_tensors_16kb"),
    ];

    for (num_tensors, tensor_size, name) in configs {
        let old_model = generate_synthetic_model(num_tensors, tensor_size);
        let mut new_model = old_model.clone();

        // Modify some bytes to simulate weight changes
        for i in 0..new_model.len().min(1000) {
            if i % 10 == 0 {
                new_model[i] = new_model[i].wrapping_add(1);
            }
        }

        let total_size = old_model.len();
        group.throughput(Throughput::Bytes(total_size as u64));

        group.bench_with_input(
            BenchmarkId::new("diff", name),
            &(&old_model, &new_model),
            |b, (old, new)| {
                let differ = TFLiteDiffer::new();
                b.iter(|| {
                    // Note: This may error on synthetic data, which is expected
                    let _ = differ.diff_from_bytes(black_box(old), black_box(new));
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_parser_creation,
    bench_differ_creation,
    bench_patcher_creation,
    bench_synthetic_diff
);
criterion_main!(benches);
