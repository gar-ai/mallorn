# Mallorn Fuzzing

Security fuzzing for Mallorn parsers and patch application using cargo-fuzz.

## Prerequisites

```bash
# Install cargo-fuzz (requires nightly Rust)
cargo install cargo-fuzz
```

## Fuzz Targets

| Target | Description |
|--------|-------------|
| `fuzz_tflite_parser` | TFLite file parser |
| `fuzz_gguf_parser` | GGUF file parser |
| `fuzz_patch_apply` | Patch application logic |
| `fuzz_compression` | Compression roundtrip |

## Running Fuzzing

```bash
cd fuzz

# Fuzz TFLite parser
cargo +nightly fuzz run fuzz_tflite_parser

# Fuzz GGUF parser
cargo +nightly fuzz run fuzz_gguf_parser

# Fuzz patch application
cargo +nightly fuzz run fuzz_patch_apply

# Fuzz compression
cargo +nightly fuzz run fuzz_compression
```

## Options

```bash
# Run for specific duration (seconds)
cargo +nightly fuzz run fuzz_tflite_parser -- -max_total_time=300

# Limit input size
cargo +nightly fuzz run fuzz_tflite_parser -- -max_len=65536

# Use multiple cores
cargo +nightly fuzz run fuzz_tflite_parser -- -jobs=4 -workers=4
```

## Corpus

Seed corpus files are in `corpus/<target>/`:

```bash
# Add seed files
mkdir -p corpus/fuzz_tflite_parser
cp tests/fixtures/models/*.tflite corpus/fuzz_tflite_parser/
```

## Reproducing Crashes

```bash
# Crashes are saved to artifacts/<target>/
cargo +nightly fuzz run fuzz_tflite_parser artifacts/fuzz_tflite_parser/crash-*
```

## Coverage

```bash
cargo +nightly fuzz coverage fuzz_tflite_parser
```

## CI Integration

Add to GitHub Actions:

```yaml
- name: Fuzz Tests
  run: |
    cargo install cargo-fuzz
    cd fuzz
    cargo +nightly fuzz run fuzz_tflite_parser -- -max_total_time=60
    cargo +nightly fuzz run fuzz_gguf_parser -- -max_total_time=60
```

## Security Notes

- All parsers should handle malformed input gracefully
- No panics should occur on arbitrary input
- Memory safety is verified through ASAN/MSAN

## License

MIT OR Apache-2.0
