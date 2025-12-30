//! Diff and patch benchmarks for Mallorn
//!
//! Benchmarks XOR delta computation and application performance.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mallorn_core::{apply_xor_delta, xor_delta};

/// Generate old tensor data
fn generate_old_data(size: usize) -> Vec<u8> {
    (0..size).map(|i| ((i * 17) % 256) as u8).collect()
}

/// Generate new tensor data with specified change ratio
fn generate_new_data(old: &[u8], change_ratio: f64) -> Vec<u8> {
    let mut new_data = old.to_vec();
    let changes = (old.len() as f64 * change_ratio) as usize;

    for i in 0..changes {
        let idx = (i * 13) % old.len();
        new_data[idx] = new_data[idx].wrapping_add(((i % 128) + 1) as u8);
    }

    new_data
}

fn bench_xor_delta(c: &mut Criterion) {
    let mut group = c.benchmark_group("xor_delta");

    let sizes = [1024, 16 * 1024, 256 * 1024, 1024 * 1024];

    for size in sizes {
        let old_data = generate_old_data(size);
        let new_data = generate_new_data(&old_data, 0.1); // 10% changes
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("compute_delta", size),
            &(&old_data, &new_data),
            |b, (old, new)| b.iter(|| xor_delta(black_box(old), black_box(new))),
        );
    }

    group.finish();
}

fn bench_apply_delta(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_delta");

    let sizes = [1024, 16 * 1024, 256 * 1024, 1024 * 1024];

    for size in sizes {
        let old_data = generate_old_data(size);
        let new_data = generate_new_data(&old_data, 0.1);
        let delta = xor_delta(&old_data, &new_data);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("apply_delta", size),
            &(&old_data, &delta),
            |b, (old, delta)| b.iter(|| apply_xor_delta(black_box(old), black_box(delta))),
        );
    }

    group.finish();
}

fn bench_delta_by_change_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_change_ratio");

    let size = 256 * 1024; // 256KB
    let change_ratios = [0.01, 0.05, 0.10, 0.25, 0.50];

    for ratio in change_ratios {
        let old_data = generate_old_data(size);
        let new_data = generate_new_data(&old_data, ratio);
        let ratio_pct = (ratio * 100.0) as u32;
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new(format!("delta_{}pct_changed", ratio_pct), size),
            &(&old_data, &new_data),
            |b, (old, new)| b.iter(|| xor_delta(black_box(old), black_box(new))),
        );
    }

    group.finish();
}

fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip");

    let sizes = [1024, 16 * 1024, 256 * 1024];

    for size in sizes {
        let old_data = generate_old_data(size);
        let new_data = generate_new_data(&old_data, 0.1);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("delta_roundtrip", size),
            &(&old_data, &new_data),
            |b, (old, new)| {
                b.iter(|| {
                    let delta = xor_delta(black_box(old), black_box(new));
                    apply_xor_delta(black_box(old), &delta)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_xor_delta,
    bench_apply_delta,
    bench_delta_by_change_ratio,
    bench_roundtrip
);
criterion_main!(benches);
