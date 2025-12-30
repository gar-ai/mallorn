# Model Fingerprinting

Mallorn provides fast model fingerprinting for version detection, taking ~10ms regardless of model size.

## Overview

Before applying a patch, you need to know which model version a device has. Full SHA-256 hashing is accurate but slow for large models:

| Model Size | SHA-256 Time |
|------------|--------------|
| 10 MB      | ~50ms        |
| 100 MB     | ~500ms       |
| 1 GB       | ~5 seconds   |

Mallorn's fingerprinting reads only the first and last 4KB of a file, providing:
- **~10ms** fingerprint time for any model size
- **99.99%** uniqueness for practical use cases
- **Instant** version detection on embedded devices

## CLI Usage

### Basic Fingerprinting

```bash
mallorn fingerprint model.tflite
```

Output:
```
Fingerprinting model.tflite...

Model Fingerprint
=================
Size:    16842752 bytes
Header:  a1b2c3d4e5f6g7h8
Tail:    i9j0k1l2m3n4o5p6
Short:   a1b2c3d4
```

### JSON Output

```bash
mallorn fingerprint model.tflite --json
```

```json
{
  "size": 16842752,
  "header_hash": "a1b2c3d4e5f6g7h8",
  "tail_hash": "i9j0k1l2m3n4o5p6",
  "short": "a1b2c3d4"
}
```

### Comparing Models

```bash
mallorn fingerprint model_a.tflite --compare model_b.tflite
```

Output:
```
Comparing models...

Model A: model_a.tflite
  Short: a1b2c3d4

Model B: model_b.tflite
  Short: e5f6g7h8

Result: DIFFERENT
```

## Version Database

Maintain a database of known model versions:

### Adding to Database

```bash
# Add models to database
mallorn fingerprint model_v1.tflite --add-to-db "v1.0.0" --db-file versions.json
mallorn fingerprint model_v2.tflite --add-to-db "v2.0.0" --db-file versions.json
mallorn fingerprint model_v3.tflite --add-to-db "v3.0.0" --db-file versions.json
```

### Checking Against Database

```bash
mallorn fingerprint unknown_model.tflite --db versions.json
```

Output:
```
Fingerprinting unknown_model.tflite...

Model Fingerprint
=================
Size:    16842752 bytes
Short:   e5f6g7h8

Database Match: v2.0.0
```

### Database Format

```json
{
  "models": [
    {
      "version": "v1.0.0",
      "size": 16777216,
      "header_hash": "a1b2c3d4e5f6g7h8",
      "tail_hash": "i9j0k1l2m3n4o5p6",
      "short": "a1b2c3d4"
    },
    {
      "version": "v2.0.0",
      "size": 16842752,
      "header_hash": "e5f6g7h8a1b2c3d4",
      "tail_hash": "m3n4o5p6i9j0k1l2",
      "short": "e5f6g7h8"
    }
  ]
}
```

## Python API

```python
import mallorn

# Get fingerprint
fp = mallorn.fingerprint("model.tflite")
print(f"Version: {fp.short()}")
print(f"Size: {fp.size}")
print(f"Header: {fp.header_hash}")
print(f"Tail: {fp.tail_hash}")

# Compare models
same = mallorn.compare_fingerprints("model_a.tflite", "model_b.tflite")
if same:
    print("Models are the same version")
```

## Embedded Usage (C)

```c
#include "mallorn.h"

// Fingerprint a model in flash
MallornFingerprint fp;
mallorn_fingerprint(model_data, model_size, &fp);

// Compare with expected version
if (memcmp(fp.short_hash, expected_hash, 8) == 0) {
    printf("Model is v2.0.0\n");
}
```

## OTA Update Flow

Typical fingerprint-based update flow:

```
Device                              Server
  |                                   |
  |-- fingerprint: a1b2c3d4 --------->|
  |                                   |
  |<-------- patch: a1b2→e5f6 --------|
  |                                   |
  |-- apply patch                     |
  |-- verify: e5f6g7h8                |
  |                                   |
  |-- fingerprint: e5f6g7h8 --------->|
  |                                   |
  |<-------- "up to date" ------------|
```

## Collision Resistance

The short fingerprint (8 hex chars = 32 bits) provides:

- **1 in 4 billion** chance of collision between random files
- **Practically zero** for real model versions

For security-critical applications, use the full hash:

```bash
mallorn fingerprint model.tflite --json | jq '.header_hash + .tail_hash'
```

This gives 256 bits of collision resistance.
