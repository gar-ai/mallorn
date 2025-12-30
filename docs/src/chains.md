# Patch Chains

Patch chains enable incremental updates across multiple model versions, allowing devices to update from any version to the latest without storing every intermediate patch.

## Overview

When you release frequent model updates (v1 → v2 → v3 → v4), you have two choices:

1. **Direct patches**: Store a separate patch for each version pair (O(n²) storage)
2. **Patch chains**: Store sequential patches and apply them in order (O(n) storage)

Mallorn's patch chains provide the best of both worlds:
- Sequential storage efficiency
- Ability to "squash" ranges into single patches
- Skip unnecessary intermediate versions

## Creating a Chain

Start with your first version update:

```bash
# Create patch from v1 to v2
mallorn diff model_v1.tflite model_v2.tflite -o v1_to_v2.tflp

# Initialize chain
mallorn chain create v1_to_v2.tflp -o updates.mchn --chain-id "my-model"
```

## Adding Updates

As you release new versions, append patches to the chain:

```bash
# Create v2 → v3 patch
mallorn diff model_v2.tflite model_v3.tflite -o v2_to_v3.tflp

# Append to chain
mallorn chain append updates.mchn v2_to_v3.tflp

# Create v3 → v4 patch
mallorn diff model_v3.tflite model_v4.tflite -o v3_to_v4.tflp
mallorn chain append updates.mchn v3_to_v4.tflp
```

## Viewing Chain Info

```bash
mallorn chain info updates.mchn --verbose
```

Output:
```
Patch Chain: my-model
=====================
Links: 3
Total size: 2,456,789 bytes

Chain History:
  [0] a1b2c3d4 → e5f6g7h8 (842,301 bytes) - v1→v2
  [1] e5f6g7h8 → i9j0k1l2 (814,256 bytes) - v2→v3
  [2] i9j0k1l2 → m3n4o5p6 (800,232 bytes) - v3→v4
```

## Applying Chains

Update a device from any version to the latest:

```bash
# Device has v1, update to v4 (applies all 3 patches)
mallorn chain apply model_v1.tflite updates.mchn -o model_v4.tflite

# Device has v2, only needs 2 patches
mallorn chain apply model_v2.tflite updates.mchn -o model_v4.tflite

# Update to specific version (not head)
mallorn chain apply model_v1.tflite updates.mchn -o model_v3.tflite \
  --target i9j0k1l2
```

## Squashing Patches

For devices far behind, squash multiple patches into one:

```bash
# Squash entire chain into single v1→v4 patch
mallorn chain squash updates.mchn -o v1_to_v4.tflp

# Squash specific range
mallorn chain squash updates.mchn -o v2_to_v4.tflp \
  --from e5f6g7h8 --to m3n4o5p6
```

## Extracting Patches

Extract individual patches from a chain:

```bash
# Extract all patches
mallorn chain extract updates.mchn -o patches/

# Extract specific range
mallorn chain extract updates.mchn -o patches/ \
  --from e5f6g7h8 --to i9j0k1l2
```

## Best Practices

### Chain Management

1. **Use meaningful chain IDs**: `--chain-id "mobilenet-v2-quantized"`
2. **Keep chains focused**: One chain per model variant
3. **Periodically squash old versions**: Devices rarely need to update from v1

### Server-Side Strategy

```
models/
├── latest.tflite           # Current model
├── updates.mchn            # Full chain history
├── v1_to_latest.tflp       # Squashed for legacy devices
└── manifest.json           # Version metadata
```

### Manifest Example

```json
{
  "model_family": "mobilenet-v2",
  "current_version": "v4",
  "current_hash": "m3n4o5p6...",
  "chain_url": "https://models.example.com/updates.mchn",
  "squashed_patches": {
    "a1b2c3d4": "https://models.example.com/v1_to_latest.tflp"
  }
}
```

## Memory-Efficient Chain Application

For embedded devices with limited RAM, chains can be applied in streaming mode:

```bash
mallorn chain apply model.tflite updates.mchn -o updated.tflite --streaming
```

This applies each patch sequentially without loading the entire chain into memory.
