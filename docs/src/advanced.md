# Advanced Compression

Mallorn offers several advanced compression techniques for maximizing patch efficiency.

## Compression Levels

The `--level` flag controls the zstd compression level:

| Level | Speed | Ratio | Use Case |
|-------|-------|-------|----------|
| 1-3   | Fast  | Good  | Development, frequent updates |
| 4-9   | Medium| Better| Production, balanced |
| 10-19 | Slow  | Best  | Infrequent updates, bandwidth-critical |
| 20-22 | Very slow | Maximum | Archival, one-time patches |

```bash
# Fast compression for testing
mallorn diff old.tflite new.tflite -o update.tflp --level 1

# Maximum compression for deployment
mallorn diff old.tflite new.tflite -o update.tflp --level 19
```

## Neural-Aware Compression

The `--neural` flag enables compression optimized for neural network weight patterns:

```bash
mallorn diff base.tflite finetuned.tflite -o update.tflp --neural
```

### How It Works

1. **Byte-plane separation**: Splits 32-bit floats into 4 byte streams
2. **Exponent clustering**: Groups similar magnitude values
3. **Delta encoding**: Exploits locality in weight updates

### When to Use

Neural compression works best for:
- Fine-tuned models (small weight changes)
- Quantization updates (INT8 ↔ INT8)
- Same-architecture model variants

Less effective for:
- Completely different models
- Major architectural changes
- Non-neural binary data

### Compression Comparison

| Scenario | Standard | Neural | Improvement |
|----------|----------|--------|-------------|
| Fine-tuning | 15x | 22x | +47% |
| Quantization change | 8x | 12x | +50% |
| Architecture change | 5x | 4x | -20% |

## Dictionary Compression

Pre-trained dictionaries capture common patterns across model versions:

### Training a Dictionary

```bash
# Collect sample models (same architecture, different versions)
mallorn dict train \
  model_v1.tflite \
  model_v2.tflite \
  model_v3.tflite \
  model_v4.tflite \
  -o tflite.dict --max-size 114688
```

### Using the Dictionary

```bash
mallorn diff old.tflite new.tflite -o update.tflp --dict tflite.dict
```

### Dictionary Best Practices

1. **Train on representative samples**: Use 5-10 model versions
2. **Match architectures**: Train separate dicts for different model types
3. **Include the dictionary**: Devices need the same dict to decompress
4. **Size tradeoffs**: Larger dicts (up to 112KB) give better ratios

### Dictionary Info

```bash
mallorn dict info tflite.dict
```

Output:
```
Dictionary: tflite.dict
=======================
Size:       114688 bytes
Trained on: 4 samples
Created:    2024-12-29T10:30:00Z
```

## Parallel Compression

The `--parallel` flag enables multi-threaded tensor compression:

```bash
mallorn diff old.gguf new.gguf -o update.ggup --parallel
```

### Performance

| Cores | Speedup | Use Case |
|-------|---------|----------|
| 1     | 1x      | Baseline |
| 4     | 3.2x    | Laptop |
| 8     | 5.8x    | Workstation |
| 16    | 9.5x    | Server |

Parallel mode processes each tensor independently, making it ideal for models with many tensors.

## Streaming Mode

For memory-constrained devices, streaming mode processes patches in chunks:

```bash
mallorn patch model.tflite update.tflp -o new.tflite --streaming
```

### Memory Usage

| Mode | Peak Memory | Use Case |
|------|-------------|----------|
| Standard | 2x model size | Workstations |
| Streaming (64MB buffer) | 64MB | Embedded Linux |
| Streaming (1MB buffer) | 1MB | Microcontrollers |

### Custom Buffer Size

```bash
# 16MB buffer for constrained devices
mallorn patch model.tflite update.tflp -o new.tflite \
  --streaming --buffer-size 16777216

# 1MB buffer for very constrained devices
mallorn patch model.tflite update.tflp -o new.tflite \
  --streaming --buffer-size 1048576
```

## Combining Options

Options can be combined for maximum efficiency:

```bash
# Maximum compression with neural + dictionary + high level
mallorn diff old.tflite new.tflite -o update.tflp \
  --neural \
  --dict models.dict \
  --level 19 \
  --parallel
```

## Format-Specific Optimizations

### TFLite

- Flatbuffer-aware diffing
- Tensor alignment preservation
- Metadata delta compression

### GGUF

- Quantization-aware compression
- K-quant block alignment
- Vocabulary delta handling

### ONNX

- Protobuf structural diff
- Graph topology preservation
- External data support

### SafeTensors

- Header-first streaming
- Tensor offset optimization
- Minimal metadata overhead

## Benchmarking

Compare compression strategies:

```bash
# Create patches with different settings
mallorn diff old.tflite new.tflite -o standard.tflp
mallorn diff old.tflite new.tflite -o neural.tflp --neural
mallorn diff old.tflite new.tflite -o dict.tflp --dict models.dict
mallorn diff old.tflite new.tflite -o max.tflp --neural --dict models.dict --level 19

# Compare sizes
ls -la *.tflp
```

Example output:
```
-rw-r--r--  1 user  842301 standard.tflp
-rw-r--r--  1 user  614523 neural.tflp
-rw-r--r--  1 user  578234 dict.tflp
-rw-r--r--  1 user  412156 max.tflp
```
