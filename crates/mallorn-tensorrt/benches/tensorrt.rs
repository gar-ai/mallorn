//! TensorRT patch benchmarks
//!
//! Benchmarks for patch creation and application performance.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mallorn_core::{CompressionMethod, DiffOptions, Patch, PatchMetadata, PatchOperation};
use mallorn_tensorrt::{
    deserialize_patch, serialize_patch, Precision, TensorRTConfig, TensorRTDiffer,
    TensorRTPatcher, TensorRTPatch,
};

/// Create a synthetic test patch for benchmarking
fn create_test_patch(num_ops: usize, data_size: usize) -> TensorRTPatch {
    let mut operations = Vec::with_capacity(num_ops);

    for i in 0..num_ops {
        if i % 3 == 0 {
            operations.push(PatchOperation::CopyTensor {
                name: format!("tensor_{}", i),
            });
        } else if i % 3 == 1 {
            operations.push(PatchOperation::ReplaceTensor {
                name: format!("tensor_{}", i),
                data: vec![i as u8; data_size],
                compression: None,
            });
        } else {
            operations.push(PatchOperation::DeltaTensor {
                name: format!("tensor_{}", i),
                delta: vec![0u8; data_size], // Sparse delta
                delta_format: mallorn_core::DeltaFormat::Xor,
                compression: None,
            });
        }
    }

    TensorRTPatch {
        onnx_patch: Patch {
            version: 1,
            source_hash: [0xAA; 32],
            target_hash: [0xBB; 32],
            operations,
            compression: CompressionMethod::Zstd { level: 3 },
            metadata: PatchMetadata {
                source_version: Some("v1.0".into()),
                target_version: Some("v1.1".into()),
                created_at: 0,
                description: Some("Benchmark patch".into()),
            },
        },
        config: TensorRTConfig::new()
            .with_precision(Precision::FP16)
            .with_workspace_mb(2048)
            .with_max_batch_size(8),
        source_onnx_hash: [0xCC; 32],
        target_onnx_hash: [0xDD; 32],
    }
}

fn bench_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensorrt_serialization");

    // Different patch sizes
    let configs = [
        (10, 1024, "10_ops_1kb"),
        (50, 4096, "50_ops_4kb"),
        (100, 16384, "100_ops_16kb"),
    ];

    for (num_ops, data_size, name) in configs {
        let patch = create_test_patch(num_ops, data_size);

        // Estimate total size for throughput
        let estimated_size = num_ops * data_size;
        group.throughput(Throughput::Bytes(estimated_size as u64));

        group.bench_with_input(
            BenchmarkId::new("serialize", name),
            &patch,
            |b, patch| {
                b.iter(|| {
                    let _ = serialize_patch(black_box(patch));
                })
            },
        );
    }

    group.finish();
}

fn bench_deserialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensorrt_deserialization");

    let configs = [
        (10, 1024, "10_ops_1kb"),
        (50, 4096, "50_ops_4kb"),
        (100, 16384, "100_ops_16kb"),
    ];

    for (num_ops, data_size, name) in configs {
        let patch = create_test_patch(num_ops, data_size);
        let serialized = serialize_patch(&patch).unwrap();

        group.throughput(Throughput::Bytes(serialized.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("deserialize", name),
            &serialized,
            |b, data| {
                b.iter(|| {
                    let _ = deserialize_patch(black_box(data));
                })
            },
        );
    }

    group.finish();
}

fn bench_config_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensorrt_config");

    group.bench_function("config_creation", |b| {
        b.iter(|| {
            TensorRTConfig::new()
                .with_precision(black_box(Precision::FP16))
                .with_workspace_mb(black_box(2048))
                .with_max_batch_size(black_box(8))
                .with_dla_core(black_box(0))
                .with_strict_types(black_box(true))
        })
    });

    let config = TensorRTConfig::new()
        .with_precision(Precision::FP16)
        .with_workspace_mb(2048)
        .with_max_batch_size(8)
        .with_dla_core(0);

    group.bench_function("trtexec_command_generation", |b| {
        b.iter(|| config.to_trtexec_command(black_box("model.onnx"), black_box("model.engine")))
    });

    group.bench_function("config_json_serialization", |b| {
        b.iter(|| serde_json::to_string(black_box(&config)).unwrap())
    });

    let config_json = serde_json::to_string(&config).unwrap();
    group.bench_function("config_json_deserialization", |b| {
        b.iter(|| serde_json::from_str::<TensorRTConfig>(black_box(&config_json)).unwrap())
    });

    group.finish();
}

fn bench_differ_creation(c: &mut Criterion) {
    c.bench_function("differ_creation", |b| {
        b.iter(|| TensorRTDiffer::new())
    });

    c.bench_function("differ_with_options", |b| {
        b.iter(|| {
            let options = DiffOptions {
                compression: CompressionMethod::Zstd { level: 5 },
                min_tensor_size: 512,
                neural_compression: false,
                target_size_hint: None,
                dictionary: None,
            };
            TensorRTDiffer::with_options(options)
        })
    });
}

fn bench_patcher_creation(c: &mut Criterion) {
    c.bench_function("patcher_creation", |b| {
        b.iter(|| TensorRTPatcher::new())
    });
}

fn bench_rebuild_instructions(c: &mut Criterion) {
    let patcher = TensorRTPatcher::new();
    let patch = create_test_patch(10, 1024);

    c.bench_function("get_rebuild_instructions", |b| {
        b.iter(|| patcher.get_rebuild_instructions(black_box(&patch)))
    });
}

criterion_group!(
    benches,
    bench_serialization,
    bench_deserialization,
    bench_config_operations,
    bench_differ_creation,
    bench_patcher_creation,
    bench_rebuild_instructions
);
criterion_main!(benches);
