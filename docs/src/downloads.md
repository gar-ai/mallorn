# HTTP Downloads

Mallorn includes robust HTTP download support with resume capability, hash verification, and manifest handling.

## Basic Downloads

Download a patch file:

```bash
mallorn download https://models.example.com/update.tflp -o update.tflp
```

Output:
```
Downloading patch...
  URL: https://models.example.com/update.tflp
  Output: update.tflp

⠋ [00:00:05] [████████████████████████] 842KB/842KB (168KB/s)

Download complete!
  Size: 842301 bytes
```

## Resume Support

For unreliable connections, enable resume:

```bash
mallorn download https://models.example.com/large_update.tflp \
  -o large_update.tflp --resume
```

If interrupted, re-run the same command to continue from where it stopped:

```
Downloading patch...
  URL: https://models.example.com/large_update.tflp
  Output: large_update.tflp
  Resume: enabled
  Resuming from 524288 bytes

⠋ [00:00:03] [████████████████████████] 318KB/318KB (106KB/s)

Download complete!
  Size: 842301 bytes
```

### How Resume Works

1. Mallorn sends a `HEAD` request to check `Accept-Ranges: bytes`
2. If the server supports range requests, partial downloads are resumed
3. Download state is cached in `~/.cache/mallorn/` (or `--cache` directory)
4. State includes URL, bytes downloaded, ETag for validation

## Hash Verification

Verify downloaded files match expected hash:

```bash
mallorn download https://models.example.com/update.tflp -o update.tflp \
  --verify a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6
```

Output on success:
```
Download complete!
  Size: 842301 bytes
  Hash: verified
```

Output on mismatch:
```
error: Hash mismatch!
  Expected: a1b2c3d4...
  Got:      x9y8z7w6...
```

## Inspecting Servers

Check server capabilities before downloading:

```bash
mallorn download https://models.example.com/update.tflp -o /dev/null --header-only
```

Output:
```
Fetching patch header...
  URL: https://models.example.com/update.tflp

Patch Header
============
Content-Length: 842301 bytes
Accepts-Ranges: true
Content-Type:   application/octet-stream
ETag:           "abc123"
Last-Modified:  Sun, 29 Dec 2024 10:30:00 GMT
```

## Patch Manifests

Manifests describe available patches and update chains:

```bash
mallorn download https://models.example.com/manifest.json -o /dev/null --manifest
```

Output:
```
Fetching patch manifest...
  URL: https://models.example.com/manifest.json

Patch Manifest: mobilenet-v2
================
Version:    3
Patches:    5
Chains:     1

Available Patches:
  a1b2c3d4... → e5f6g7h8... (842301 bytes, zstd)
  e5f6g7h8... → i9j0k1l2... (814256 bytes, zstd)
  i9j0k1l2... → m3n4o5p6... (800232 bytes, zstd)

Available Chains:
  mobilenet-updates (3 patches)
```

### Manifest Format

```json
{
  "model_family": "mobilenet-v2",
  "version": 3,
  "patches": [
    {
      "source_hash": "a1b2c3d4...",
      "target_hash": "e5f6g7h8...",
      "url": "https://models.example.com/v1_to_v2.tflp",
      "size": 842301,
      "format": "zstd"
    }
  ],
  "chains": [
    {
      "chain_id": "mobilenet-updates",
      "url": "https://models.example.com/updates.mchn",
      "num_patches": 3
    }
  ]
}
```

## Cache Management

Specify a custom cache directory:

```bash
mallorn download https://example.com/update.tflp -o update.tflp \
  --resume --cache /tmp/mallorn-cache
```

Cache structure:
```
/tmp/mallorn-cache/
├── download_a1b2c3d4.state    # Resume state (JSON)
└── download_e5f6g7h8.state
```

State files are automatically cleaned up after successful downloads.

## Embedded Device Integration

For devices with limited connectivity:

### 1. Check Current Version

```bash
# On device: get current model fingerprint
mallorn fingerprint /data/model.tflite --json > /tmp/version.json
# Upload version.json to server
```

### 2. Server Determines Patch

```python
# Server-side logic
current = request.json['short']
patches = manifest['patches']

for patch in patches:
    if patch['source_hash'].startswith(current):
        return {'patch_url': patch['url'], 'expected_hash': patch['target_hash']}

return {'status': 'up_to_date'}
```

### 3. Download and Apply

```bash
# On device
mallorn download "$PATCH_URL" -o /tmp/update.tflp --resume \
  --verify "$EXPECTED_HASH"

mallorn patch /data/model.tflite /tmp/update.tflp -o /data/model_new.tflite

# Atomic swap
mv /data/model_new.tflite /data/model.tflite
```

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| HTTP 416 | Range not satisfiable | Delete partial file, retry |
| Hash mismatch | Corrupted download | Delete and retry |
| Connection timeout | Network issues | Use `--resume` and retry |
| No range support | Server limitation | Cannot resume; download fully |

## Python Integration

```python
import subprocess
import json

def download_patch(url, output, verify_hash=None):
    cmd = ['mallorn', 'download', url, '-o', output, '--resume']
    if verify_hash:
        cmd.extend(['--verify', verify_hash])

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

def get_manifest(url):
    result = subprocess.run(
        ['mallorn', 'download', url, '-o', '/dev/null', '--manifest'],
        capture_output=True, text=True
    )
    # Parse output...
```
