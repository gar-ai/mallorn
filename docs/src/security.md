# Security & Signing

Mallorn supports Ed25519 digital signatures to ensure patch authenticity and integrity. This prevents malicious patches from being applied to devices.

## Overview

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   Server    │         │   Patch     │         │   Device    │
│             │         │   File      │         │             │
│ Private Key │────────▶│ + Signature │────────▶│ Public Key  │
│             │  sign   │             │ verify  │             │
└─────────────┘         └─────────────┘         └─────────────┘
```

- **Private key**: Kept secure on build server, never distributed
- **Public key**: Embedded in device firmware
- **Signature**: Ed25519 signature appended to patch file

## Generating Keys

### CLI

```bash
# Generate a new keypair
mallorn keygen -o keys/

# Output:
# keys/mallorn.pub  (32 bytes, embed in firmware)
# keys/mallorn.key  (64 bytes, keep secret!)
```

### Python

```python
import mallorn

# Generate keypair
public_key, private_key = mallorn.generate_keypair()

# Save keys
with open("mallorn.pub", "wb") as f:
    f.write(public_key)
with open("mallorn.key", "wb") as f:
    f.write(private_key)
```

### Rust

```rust
use mallorn_core::sign::{generate_keypair, save_keypair};

let (public_key, private_key) = generate_keypair();
save_keypair("keys/", &public_key, &private_key)?;
```

## Signing Patches

### CLI

```bash
# Create and sign a patch in one step
mallorn diff model_v1.tflite model_v2.tflite -o update.tflp \
    --sign --key keys/mallorn.key

# Or sign an existing patch
mallorn sign update.tflp --key keys/mallorn.key -o update.tflp.signed
```

### Python

```python
import mallorn

# Create signed patch
mallorn.create_patch(
    "model_v1.tflite",
    "model_v2.tflite",
    "update.tflp",
    private_key=open("mallorn.key", "rb").read()
)

# Sign existing patch
mallorn.sign_patch("update.tflp", "mallorn.key", "update.tflp.signed")
```

### Rust

```rust
use mallorn_core::sign::sign_patch;

let private_key = std::fs::read("mallorn.key")?;
let patch_data = std::fs::read("update.tflp")?;

let signed = sign_patch(&patch_data, &private_key)?;
std::fs::write("update.tflp.signed", signed)?;
```

## Verifying Signatures

### CLI

```bash
# Verify a signed patch
mallorn verify model_v1.tflite update.tflp.signed --pubkey keys/mallorn.pub

# Output:
# Signature: VALID
# Source hash: matches
# Patch integrity: OK
```

### Python

```python
import mallorn

public_key = open("mallorn.pub", "rb").read()

result = mallorn.verify_patch(
    "model_v1.tflite",
    "update.tflp.signed",
    public_key=public_key
)

if result.signature_valid and result.source_valid:
    print("Patch is authentic and applicable")
```

### Embedded (C)

```c
#include "mallorn.h"

// Embed public key in firmware (32 bytes)
static const uint8_t PUBLIC_KEY[32] = {
    0x12, 0x34, 0x56, /* ... */
};

// Verify before applying
if (mallorn_verify_signature(&patcher, PUBLIC_KEY) != OK) {
    // Reject untrusted patch!
    return;
}

// Safe to apply
while (mallorn_step(&patcher) == CONTINUE) { }
```

## Signed Patch Format

```
┌─────────────────────────────────┐
│ Original Patch Data             │
│ (variable length)               │
├─────────────────────────────────┤
│ Signature Magic: "MSIG" (4B)    │
├─────────────────────────────────┤
│ Ed25519 Signature (64 bytes)    │
└─────────────────────────────────┘
```

The signature covers the entire patch data (header, operations, checksums).

## Key Management Best Practices

### DO

- Store private keys in a hardware security module (HSM) or secure enclave
- Use separate keys for development and production
- Rotate keys periodically
- Embed public keys at firmware build time
- Verify signatures before applying any patch

### DON'T

- Commit private keys to version control
- Share private keys between projects
- Skip signature verification "just for testing"
- Use the same key for signing and encryption

## Key Rotation

When rotating keys:

1. Generate new keypair
2. Distribute new public key in firmware update
3. Sign patches with both old and new keys during transition
4. After all devices updated, retire old key

```bash
# Sign with multiple keys for transition
mallorn sign update.tflp --key old.key -o update.signed
mallorn sign update.signed --key new.key -o update.dual-signed
```

## Threat Model

Mallorn signing protects against:

| Threat | Protection |
|--------|------------|
| Patch tampering in transit | Ed25519 signature |
| Malicious patch injection | Public key verification |
| Replay attacks | Source hash binding |
| Downgrade attacks | Target hash verification |

Not protected (out of scope):

- Key compromise (use HSM)
- Physical device access (use secure boot)
- Side-channel attacks (use constant-time crypto)

## Compliance Notes

- Ed25519 uses Curve25519, approved by NIST SP 800-186
- 128-bit security level (equivalent to RSA-3072)
- Deterministic signatures (no RNG required for signing)
- Suitable for FIPS 140-2 environments when using validated implementations
