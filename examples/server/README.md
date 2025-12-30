# Mallorn Patch Server

A FastAPI-based server for serving model patches to edge devices.

## Features

- Fingerprint-based version detection
- Automatic patch selection based on device version
- Range request support for resumable downloads
- REST API for patch management

## Setup

```bash
pip install fastapi uvicorn mallorn
```

## Usage

```bash
# Start the server
python patch_server.py

# Or with uvicorn directly
uvicorn patch_server:app --reload --host 0.0.0.0 --port 8000
```

## Directory Structure

```
./models/          # Model files (*.tflite)
./patches/         # Generated patches (*.tflp)
```

## API Endpoints

### GET /manifest

Get manifest of available patches.

```bash
curl http://localhost:8000/manifest
```

### POST /check-update

Check if update is available for a device.

```bash
curl -X POST http://localhost:8000/check-update \
  -H "Content-Type: application/json" \
  -d '{"short_id": "a1b2c3d4e5f6g7h8"}'
```

### GET /patch/{patch_id}

Download a patch file (supports range requests for resume).

```bash
# Full download
curl -O http://localhost:8000/patch/v1_to_v2

# Resume from byte 1000
curl -H "Range: bytes=1000-" -O http://localhost:8000/patch/v1_to_v2
```

### GET /fingerprint/{model_name}

Get fingerprint of a model.

```bash
curl http://localhost:8000/fingerprint/model_v1
```

### POST /create-patch

Create a new patch (admin endpoint).

```bash
curl -X POST "http://localhost:8000/create-patch?source_version=v1&target_version=v2"
```

## Device Integration

Example device update flow:

```python
import requests

# 1. Get current model fingerprint (on device)
# fp = mallorn.fingerprint("model.tflite")

# 2. Check for updates
response = requests.post(
    "http://server:8000/check-update",
    json={"short_id": "a1b2c3d4e5f6g7h8"}
)
update = response.json()

if update["update_available"]:
    # 3. Download patch with resume support
    patch_url = f"http://server:8000{update['patch_url']}"
    # Use mallorn download or requests with Range header

    # 4. Apply patch
    # mallorn.apply_patch("model.tflite", "update.tflp", "model_new.tflite")
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| MODELS_DIR | Directory containing model files | ./models |
| PATCHES_DIR | Directory for storing patches | ./patches |
