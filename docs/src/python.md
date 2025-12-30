# Python API

The `mallorn` Python package provides bindings for creating and applying model patches.

## Installation

```bash
pip install mallorn
```

Or build from source:

```bash
cd crates/mallorn-python
maturin develop --release
```

## Quick Start

```python
import mallorn

# Create a patch
stats = mallorn.create_patch(
    "model_v1.tflite",
    "model_v2.tflite",
    "update.tflp"
)
print(f"Patch size: {stats.patch_size} bytes ({stats.compression_ratio:.1f}x compression)")

# Apply a patch
result = mallorn.apply_patch(
    "model_v1.tflite",
    "update.tflp",
    "model_v2_restored.tflite"
)
print(f"Success: {result.is_valid()}")
```

## API Reference

### create_patch

Create a delta patch between two model versions.

```python
def create_patch(
    source: str | Path | bytes,
    target: str | Path | bytes,
    output: str | Path,
    compression_level: int = 3,
    neural: bool = False,
    private_key: bytes | None = None,
) -> PatchStats
```

**Parameters:**
- `source` - Path to original model, or model bytes
- `target` - Path to updated model, or model bytes
- `output` - Output patch file path
- `compression_level` - Zstd level 1-19 (default: 3)
- `neural` - Enable neural-optimized compression
- `private_key` - Ed25519 private key for signing (64 bytes)

**Returns:** `PatchStats` object

**Example:**

```python
import mallorn

# Basic usage
stats = mallorn.create_patch("v1.tflite", "v2.tflite", "patch.tflp")

# With options
stats = mallorn.create_patch(
    "v1.gguf",
    "v2.gguf",
    "patch.ggup",
    compression_level=9,
    neural=True
)

# From bytes (useful for in-memory models)
old_model = open("v1.tflite", "rb").read()
new_model = open("v2.tflite", "rb").read()
stats = mallorn.create_patch(old_model, new_model, "patch.tflp")

# With signing
private_key = open("mallorn.key", "rb").read()
stats = mallorn.create_patch(
    "v1.tflite",
    "v2.tflite",
    "patch.tflp",
    private_key=private_key
)
```

---

### apply_patch

Apply a patch to reconstruct the target model.

```python
def apply_patch(
    source: str | Path | bytes,
    patch: str | Path | bytes,
    output: str | Path,
    public_key: bytes | None = None,
) -> PatchVerification
```

**Parameters:**
- `source` - Path to original model, or model bytes
- `patch` - Path to patch file, or patch bytes
- `output` - Output model file path
- `public_key` - Ed25519 public key for verification (32 bytes)

**Returns:** `PatchVerification` object

**Example:**

```python
import mallorn

# Basic usage
result = mallorn.apply_patch("v1.tflite", "patch.tflp", "v2.tflite")

if result.is_valid():
    print("Patch applied successfully!")
else:
    print(f"Error: {result.error}")

# With signature verification
public_key = open("mallorn.pub", "rb").read()
result = mallorn.apply_patch(
    "v1.tflite",
    "patch.tflp",
    "v2.tflite",
    public_key=public_key
)

if not result.signature_valid:
    raise ValueError("Signature verification failed!")
```

---

### verify_patch

Verify a patch without applying it.

```python
def verify_patch(
    source: str | Path | bytes,
    patch: str | Path | bytes,
    public_key: bytes | None = None,
) -> PatchVerification
```

**Parameters:**
- `source` - Path to original model, or model bytes
- `patch` - Path to patch file, or patch bytes
- `public_key` - Ed25519 public key for verification

**Returns:** `PatchVerification` object

**Example:**

```python
import mallorn

result = mallorn.verify_patch("v1.tflite", "patch.tflp")

print(f"Source valid: {result.source_valid}")
print(f"Patch valid: {result.patch_valid}")
print(f"Expected target: {result.expected_target.hex()}")
```

---

### patch_info

Get information about a patch file.

```python
def patch_info(patch: str | Path | bytes) -> PatchInfo
```

**Parameters:**
- `patch` - Path to patch file, or patch bytes

**Returns:** `PatchInfo` object

**Example:**

```python
import mallorn

info = mallorn.patch_info("update.tflp")

print(f"Format: {info.format}")
print(f"Source hash: {info.source_hash.hex()}")
print(f"Target hash: {info.target_hash.hex()}")
print(f"Patch size: {info.patch_size} bytes")
print(f"Tensors modified: {info.tensors_modified}")
print(f"Tensors unchanged: {info.tensors_unchanged}")
print(f"Compression: {info.compression}")
print(f"Signed: {info.signed}")
```

---

### generate_keypair

Generate an Ed25519 keypair for signing.

```python
def generate_keypair() -> tuple[bytes, bytes]
```

**Returns:** Tuple of (public_key, private_key)

**Example:**

```python
import mallorn

public_key, private_key = mallorn.generate_keypair()

# Save keys
with open("mallorn.pub", "wb") as f:
    f.write(public_key)  # 32 bytes

with open("mallorn.key", "wb") as f:
    f.write(private_key)  # 64 bytes
```

---

### sign_patch

Sign an existing patch file.

```python
def sign_patch(
    patch: str | Path | bytes,
    private_key: bytes,
    output: str | Path,
) -> None
```

**Parameters:**
- `patch` - Path to unsigned patch, or patch bytes
- `private_key` - Ed25519 private key (64 bytes)
- `output` - Output signed patch file path

**Example:**

```python
import mallorn

private_key = open("mallorn.key", "rb").read()
mallorn.sign_patch("update.tflp", private_key, "update.tflp.signed")
```

---

### fingerprint

Generate a fingerprint for a model file (~10ms for any size).

```python
def fingerprint(model_path: str) -> Fingerprint
```

**Parameters:**
- `model_path` - Path to the model file

**Returns:** `Fingerprint` object

**Example:**

```python
import mallorn

fp = mallorn.fingerprint("model.tflite")

print(f"Size: {fp.file_size} bytes")
print(f"Header hash: {fp.header_hash}")
print(f"Tail hash: {fp.tail_hash}")
print(f"Short ID: {fp.short()}")

# Check if two fingerprints match
fp2 = mallorn.fingerprint("other_model.tflite")
if fp.matches(fp2):
    print("Models are the same version")
```

---

### compare_fingerprints

Compare two models by fingerprint (faster than full hash comparison).

```python
def compare_fingerprints(model1: str, model2: str) -> bool
```

**Parameters:**
- `model1` - Path to the first model file
- `model2` - Path to the second model file

**Returns:** `True` if fingerprints match (models are likely identical)

**Example:**

```python
import mallorn

# Quick version comparison
if mallorn.compare_fingerprints("model_a.tflite", "model_b.tflite"):
    print("Same model version")
else:
    print("Different versions - update needed")
```

---

## Data Classes

### Fingerprint

Model fingerprint for quick version detection.

```python
@dataclass
class Fingerprint:
    format: str        # Model format (tflite, gguf, etc.)
    file_size: int     # File size in bytes
    header_hash: str   # MD5 of first 64KB (hex)
    tail_hash: str     # MD5 of last 4KB (hex)
    combined_hash: str # XOR of header and tail (hex)

    def short(self) -> str:
        """Returns first 16 chars of combined hash."""

    def matches(self, other: Fingerprint) -> bool:
        """Returns True if combined hashes match."""
```

### PatchStats

Statistics from patch creation.

```python
@dataclass
class PatchStats:
    source_size: int       # Original model size in bytes
    target_size: int       # Updated model size in bytes
    patch_size: int        # Patch file size in bytes
    compression_ratio: float  # target_size / patch_size
    tensors_modified: int  # Number of changed tensors
    tensors_unchanged: int # Number of copied tensors
```

### PatchVerification

Result of patch verification or application.

```python
@dataclass
class PatchVerification:
    source_valid: bool       # Source hash matches
    patch_valid: bool        # Patch integrity OK
    signature_valid: bool    # Signature verified (if signed)
    expected_target: bytes   # Expected output hash (32 bytes)
    actual_target: bytes | None  # Actual output hash (after apply)

    def is_valid(self) -> bool:
        """Returns True if all checks passed."""
```

### PatchInfo

Information about a patch file.

```python
@dataclass
class PatchInfo:
    format: str            # "TFLite", "GGUF", or "ONNX"
    version: int           # Patch format version
    source_hash: bytes     # SHA256 of source model
    target_hash: bytes     # SHA256 of target model
    patch_size: int        # Patch file size
    compression: str       # Compression method
    neural: bool           # Neural compression enabled
    tensors_modified: int
    tensors_unchanged: int
    signed: bool           # Has signature
    created_at: int | None # Unix timestamp
```

---

## Error Handling

```python
import mallorn
from mallorn import MallornError

try:
    mallorn.apply_patch("wrong_model.tflite", "patch.tflp", "output.tflite")
except MallornError as e:
    print(f"Patch failed: {e}")
    # MallornError subtypes:
    # - SourceHashMismatch
    # - TargetHashMismatch
    # - InvalidPatch
    # - SignatureError
    # - IoError
```

---

## Integration Examples

### TensorFlow Lite Workflow

```python
import mallorn
import tensorflow as tf

# Load and update model
interpreter = tf.lite.Interpreter(model_path="model_v1.tflite")

# After retraining, save new model
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
new_model = converter.convert()
with open("model_v2.tflite", "wb") as f:
    f.write(new_model)

# Create patch for OTA
stats = mallorn.create_patch(
    "model_v1.tflite",
    "model_v2.tflite",
    "update.tflp",
    neural=True
)
print(f"Update size: {stats.patch_size / 1024:.1f} KB")
print(f"Savings: {(1 - stats.patch_size / stats.target_size) * 100:.1f}%")
```

### GGUF/llama.cpp Workflow

```python
import mallorn

# Create patch for quantized LLM update
stats = mallorn.create_patch(
    "llama-7b-q4.gguf",
    "llama-7b-q4-finetuned.gguf",
    "update.ggup",
    compression_level=9
)

print(f"Original: {stats.target_size / 1e9:.1f} GB")
print(f"Patch: {stats.patch_size / 1e6:.1f} MB")
```

### Batch Processing

```python
import mallorn
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def process_model_pair(old_path, new_path, output_dir):
    name = old_path.stem
    output = output_dir / f"{name}.tflp"
    return mallorn.create_patch(str(old_path), str(new_path), str(output))

# Process multiple model pairs in parallel
old_models = Path("models/v1").glob("*.tflite")
new_models = Path("models/v2")
output_dir = Path("patches")

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for old in old_models:
        new = new_models / old.name
        if new.exists():
            futures.append(executor.submit(
                process_model_pair, old, new, output_dir
            ))

    for future in futures:
        stats = future.result()
        print(f"Created patch: {stats.patch_size} bytes")
```
