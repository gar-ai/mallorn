# CLI Reference

The `mallorn` command-line tool provides commands for creating, applying, and verifying model patches.

## Installation

```bash
cargo install mallorn-cli
```

Or build from source:

```bash
git clone https://github.com/gabedavis/mallorn
cd mallorn
cargo build --release -p mallorn-cli
```

## Commands

### mallorn diff

Create a patch between two model versions.

```bash
mallorn diff <SOURCE> <TARGET> -o <OUTPUT> [OPTIONS]
```

**Arguments:**
- `SOURCE` - Path to the original model file
- `TARGET` - Path to the updated model file
- `-o, --output` - Output patch file path

**Options:**
- `-l, --level <LEVEL>` - Compression level (1-22 for zstd, default: 3)
- `--neural` - Enable neural-aware compression (better for fine-tuned models)
- `--dict <FILE>` - Pre-trained dictionary for improved compression
- `--parallel` - Enable parallel tensor compression (faster on multi-core)

**Examples:**

```bash
# Basic diff
mallorn diff model_v1.tflite model_v2.tflite -o update.tflp

# Maximum compression
mallorn diff old.gguf new.gguf -o update.ggup --level 19

# With neural compression (best for fine-tuned models)
mallorn diff base.tflite finetuned.tflite -o update.tflp --neural

# With pre-trained dictionary (best compression)
mallorn diff old.tflite new.tflite -o update.tflp --dict models.dict

# Parallel compression (multi-core speedup)
mallorn diff old.gguf new.gguf -o update.ggup --parallel
```

**Output:**
```
Creating patch...
  Source: model_v1.tflite
  Target: model_v2.tflite
  Output: update.tflp
  Format: TFLite
  Compression level: 3

Patch created successfully!
  Source size:    16777216 bytes
  Target size:    16842752 bytes
  Patch size:       842301 bytes
  Compression:        19.9x
  Savings:            95.0%
  Tensors:        42 modified, 8 unchanged
```

---

### mallorn patch

Apply a patch to reconstruct the target model.

```bash
mallorn patch <SOURCE> <PATCH> -o <OUTPUT> [OPTIONS]
```

**Arguments:**
- `SOURCE` - Path to the original model file
- `PATCH` - Path to the patch file
- `-o, --output` - Output model file path

**Options:**
- `--streaming` - Use streaming mode (low memory, for large models)
- `--buffer-size <BYTES>` - Buffer size for streaming mode (default: 64MB)

**Examples:**

```bash
# Apply patch
mallorn patch model_v1.tflite update.tflp -o model_v2.tflite

# Streaming mode for large models (low memory usage)
mallorn patch large_model.gguf update.ggup -o updated.gguf --streaming

# Custom buffer size for streaming
mallorn patch model.onnx update.onxp -o new.onnx --streaming --buffer-size 16777216
```

**Output:**
```
Applying patch...
  Source: model_v1.tflite
  Patch:  update.tflp
  Output: model_v2.tflite

Patch applied successfully!
  Output size: 16842752 bytes
  Hash verified: OK
```

---

### mallorn verify

Verify a patch without applying it.

```bash
mallorn verify <SOURCE> <PATCH> [OPTIONS]
```

**Arguments:**
- `SOURCE` - Path to the original model file
- `PATCH` - Path to the patch file

**Options:**
- `--pubkey <KEY>` - Public key for signature verification

**Examples:**

```bash
# Verify patch can be applied
mallorn verify model_v1.tflite update.tflp

# Verify with signature check
mallorn verify model_v1.tflite update.tflp --pubkey mallorn.pub
```

**Output:**
```
Verifying patch...
  Source hash: matches
  Patch format: valid
  Signature: VALID (if signed)

Verification passed!
```

---

### mallorn info

Display information about a patch file.

```bash
mallorn info <PATCH> [OPTIONS]
```

**Arguments:**
- `PATCH` - Path to the patch file

**Options:**
- `--verbose` - Show detailed tensor-level information

**Examples:**

```bash
# Basic info
mallorn info update.tflp

# Detailed tensor information
mallorn info update.tflp --verbose
```

**Output:**
```
Patch: update.tflp
  Format: TFLite Patch v1
  Source hash: sha256:a1b2c3d4...
  Target hash: sha256:e5f6g7h8...
  Patch size: 842,301 bytes
  Compression: zstd (level 3)
  Neural: enabled
  Tensors modified: 42
  Tensors unchanged: 8
  Signed: yes
  Created: 2024-12-27T10:30:00Z
```

---

### mallorn keygen

Generate a new Ed25519 keypair for signing.

```bash
mallorn keygen -o <OUTPUT_DIR>
```

**Arguments:**
- `-o, --output` - Directory to write keys to

**Examples:**

```bash
mallorn keygen -o keys/
```

**Output:**
```
Generating Ed25519 keypair...
  Public key:  keys/mallorn.pub (32 bytes)
  Private key: keys/mallorn.key (64 bytes)

Keep the private key secure! Embed the public key in firmware.
```

---

### mallorn sign

Sign an existing patch file.

```bash
mallorn sign <PATCH> --key <KEY> -o <OUTPUT>
```

**Arguments:**
- `PATCH` - Path to the unsigned patch file
- `--key` - Path to private key
- `-o, --output` - Output signed patch file

**Examples:**

```bash
mallorn sign update.tflp --key mallorn.key -o update.tflp.signed
```

---

### mallorn convert

Convert a patch to streaming format for embedded devices.

```bash
mallorn convert <INPUT> -o <OUTPUT>
```

**Arguments:**
- `INPUT` - Input patch file (.tflp or .ggup)
- `-o, --output` - Output streaming patch file (.mllp)

---

### mallorn fingerprint

Generate a quick fingerprint for model version detection (~10ms for any size).

```bash
mallorn fingerprint <MODEL> [OPTIONS]
```

**Arguments:**
- `MODEL` - Model file to fingerprint

**Options:**
- `--json` - Output as JSON
- `--db <FILE>` - Check against fingerprint database
- `--compare <FILE>` - Compare with another model
- `--add-to-db <VERSION>` - Add to database with version string
- `--db-file <FILE>` - Database file for --add-to-db

**Examples:**

```bash
# Get fingerprint
mallorn fingerprint model.tflite

# JSON output
mallorn fingerprint model.tflite --json

# Compare two models
mallorn fingerprint model_a.tflite --compare model_b.tflite

# Check against version database
mallorn fingerprint model.tflite --db versions.json

# Add to database
mallorn fingerprint model.tflite --add-to-db "v2.1.0" --db-file versions.json
```

**Output:**
```
Fingerprinting model.tflite...

Model Fingerprint
=================
Size:    16842752 bytes
Header:  a1b2c3d4e5f6g7h8
Tail:    i9j0k1l2m3n4o5p6
Short:   a1b2c3d4
```

---

### mallorn download

Download patches from a URL with resume support.

```bash
mallorn download <URL> -o <OUTPUT> [OPTIONS]
```

**Arguments:**
- `URL` - URL to download from
- `-o, --output` - Output file path

**Options:**
- `--resume` - Enable resume for interrupted downloads
- `--verify <HASH>` - Expected SHA256 hash (hex) for verification
- `--cache <DIR>` - Cache directory for partial downloads
- `--header-only` - Fetch header only (for inspection)
- `--manifest` - Fetch manifest from URL

**Examples:**

```bash
# Download with resume support
mallorn download https://models.example.com/update.tflp -o update.tflp --resume

# Download with hash verification
mallorn download https://example.com/patch.tflp -o patch.tflp \
  --verify a1b2c3d4e5f6...

# Inspect server support for range requests
mallorn download https://example.com/patch.tflp -o /dev/null --header-only

# Fetch patch manifest
mallorn download https://example.com/manifest.json -o /dev/null --manifest
```

**Output:**
```
Downloading patch...
  URL: https://models.example.com/update.tflp
  Output: update.tflp
  Resume: enabled

⠋ [00:00:05] [████████████████████████████████████████] 842KB/842KB (168KB/s, 0s)
Download complete!
  Size: 842301 bytes
  Hash: verified
```

---

### mallorn chain

Manage patch chains for incremental updates (v1 → v2 → v3).

#### mallorn chain create

Create a new patch chain from an initial patch.

```bash
mallorn chain create <PATCH> -o <OUTPUT> [--chain-id <ID>]
```

#### mallorn chain append

Append a patch to an existing chain.

```bash
mallorn chain append <CHAIN> <PATCH> [-o <OUTPUT>]
```

#### mallorn chain info

Show information about a patch chain.

```bash
mallorn chain info <CHAIN> [--verbose]
```

#### mallorn chain extract

Extract patches from a chain.

```bash
mallorn chain extract <CHAIN> -o <OUTPUT> [--from <HASH>] [--to <HASH>]
```

#### mallorn chain squash

Squash multiple patches into a single patch.

```bash
mallorn chain squash <CHAIN> -o <OUTPUT> [--from <HASH>] [--to <HASH>]
```

#### mallorn chain apply

Apply a chain to update a model through multiple versions.

```bash
mallorn chain apply <MODEL> <CHAIN> -o <OUTPUT> [--target <HASH>]
```

**Examples:**

```bash
# Create chain from initial patch
mallorn chain create v1_to_v2.tflp -o updates.mchn --chain-id "model-updates"

# Append new version
mallorn chain append updates.mchn v2_to_v3.tflp

# View chain info
mallorn chain info updates.mchn --verbose

# Apply chain to update v1 → v3
mallorn chain apply model_v1.tflite updates.mchn -o model_v3.tflite

# Squash v1→v3 into single patch
mallorn chain squash updates.mchn -o v1_to_v3.tflp
```

---

### mallorn dict

Manage compression dictionaries for improved compression ratios.

#### mallorn dict train

Train a dictionary from model samples.

```bash
mallorn dict train <SAMPLES...> -o <OUTPUT> [--max-size <BYTES>]
```

**Arguments:**
- `SAMPLES` - Sample model files to train from (multiple)
- `-o, --output` - Output dictionary file (.dict)
- `--max-size <BYTES>` - Maximum dictionary size (default: 112KB)

#### mallorn dict info

Show information about a dictionary file.

```bash
mallorn dict info <DICT>
```

**Examples:**

```bash
# Train dictionary from model samples
mallorn dict train model_v1.tflite model_v2.tflite model_v3.tflite \
  -o tflite_models.dict

# Use with diff
mallorn diff old.tflite new.tflite -o update.tflp --dict tflite_models.dict

# View dictionary info
mallorn dict info tflite_models.dict
```

**Output:**
```
Training dictionary...
  Samples: 3 files
  Total size: 50331648 bytes

Dictionary trained successfully!
  Output: tflite_models.dict
  Size: 114688 bytes
  Trained on: 3 samples
```

---

### mallorn verify-signature

Verify a signed patch file.

```bash
mallorn verify-signature <PATCH> [-k <KEY>] [--min-version <VERSION>]
```

**Arguments:**
- `PATCH` - Signed patch file

**Options:**
- `-k, --key <KEY>` - Public key file (uses embedded key if not provided)
- `--min-version <VERSION>` - Minimum version to accept (downgrade protection)

**Examples:**

```bash
# Verify signature
mallorn verify-signature update.tflp.signed -k mallorn.pub

# With downgrade protection
mallorn verify-signature update.tflp.signed --min-version 5
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MALLORN_KEY` | Default private key path | none |
| `MALLORN_PUBKEY` | Default public key path | none |
| `MALLORN_COMPRESSION` | Default compression level | 3 |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Source hash mismatch |
| 4 | Target hash mismatch |
| 5 | Signature verification failed |
| 6 | I/O error |

## Shell Completion

Generate shell completions:

```bash
# Bash
mallorn completions bash > /etc/bash_completion.d/mallorn

# Zsh
mallorn completions zsh > ~/.zfunc/_mallorn

# Fish
mallorn completions fish > ~/.config/fish/completions/mallorn.fish
```
