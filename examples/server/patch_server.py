#!/usr/bin/env python3
"""
Mallorn Patch Server

A FastAPI server for serving model patches to edge devices.

Features:
- Fingerprint-based version detection
- Automatic patch selection
- Range request support for resumable downloads
- Patch chain management

Usage:
    pip install fastapi uvicorn
    python patch_server.py

API Endpoints:
    GET  /manifest           - Get available patches
    POST /check-update       - Check if update needed (send fingerprint)
    GET  /patch/{patch_id}   - Download a specific patch
    GET  /fingerprint/{model} - Get fingerprint of a model
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# Try to import mallorn, fall back to mock if not available
try:
    import mallorn
    MALLORN_AVAILABLE = True
except ImportError:
    MALLORN_AVAILABLE = False
    print("Warning: mallorn not installed, using mock functions")

app = FastAPI(
    title="Mallorn Patch Server",
    description="Edge model delta update server",
    version="0.1.0"
)

# Configuration
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "./models"))
PATCHES_DIR = Path(os.environ.get("PATCHES_DIR", "./patches"))

# In-memory version database (in production, use a proper database)
VERSION_DB: dict[str, dict] = {}


class FingerprintRequest(BaseModel):
    """Request to check for updates."""
    short_id: str
    file_size: Optional[int] = None


class FingerprintResponse(BaseModel):
    """Response with fingerprint info."""
    format: str
    file_size: int
    header_hash: str
    tail_hash: str
    short_id: str


class UpdateCheckResponse(BaseModel):
    """Response indicating if update is available."""
    update_available: bool
    current_version: Optional[str] = None
    latest_version: Optional[str] = None
    patch_url: Optional[str] = None
    patch_size: Optional[int] = None
    expected_hash: Optional[str] = None


class PatchManifest(BaseModel):
    """Manifest of available patches."""
    model_family: str
    latest_version: str
    patches: list[dict]


def get_file_hash(path: Path) -> str:
    """Get SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def fingerprint_model(path: Path) -> dict:
    """Generate fingerprint for a model file."""
    if MALLORN_AVAILABLE:
        fp = mallorn.fingerprint(str(path))
        return {
            "format": fp.format,
            "file_size": fp.file_size,
            "header_hash": fp.header_hash,
            "tail_hash": fp.tail_hash,
            "short_id": fp.short()
        }
    else:
        # Mock fingerprint
        file_size = path.stat().st_size
        file_hash = get_file_hash(path)
        return {
            "format": "unknown",
            "file_size": file_size,
            "header_hash": file_hash[:32],
            "tail_hash": file_hash[32:64],
            "short_id": file_hash[:16]
        }


@app.on_event("startup")
async def startup():
    """Initialize version database from models directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)

    # Scan models directory
    for model_path in MODELS_DIR.glob("*.tflite"):
        fp = fingerprint_model(model_path)
        VERSION_DB[fp["short_id"]] = {
            "path": str(model_path),
            "version": model_path.stem,
            "fingerprint": fp
        }

    print(f"Loaded {len(VERSION_DB)} model versions")


@app.get("/manifest", response_model=PatchManifest)
async def get_manifest():
    """Get manifest of available patches."""
    patches = []

    for patch_path in PATCHES_DIR.glob("*.tflp"):
        patches.append({
            "id": patch_path.stem,
            "url": f"/patch/{patch_path.stem}",
            "size": patch_path.stat().st_size
        })

    return PatchManifest(
        model_family="default",
        latest_version=max(VERSION_DB.values(), key=lambda x: x["version"])["version"] if VERSION_DB else "unknown",
        patches=patches
    )


@app.post("/check-update", response_model=UpdateCheckResponse)
async def check_update(request: FingerprintRequest):
    """Check if an update is available for a device."""
    # Look up current version by fingerprint
    current = VERSION_DB.get(request.short_id)

    if not current:
        return UpdateCheckResponse(
            update_available=False,
            current_version="unknown"
        )

    # Find latest version
    if not VERSION_DB:
        return UpdateCheckResponse(
            update_available=False,
            current_version=current["version"]
        )

    latest = max(VERSION_DB.values(), key=lambda x: x["version"])

    if current["version"] == latest["version"]:
        return UpdateCheckResponse(
            update_available=False,
            current_version=current["version"],
            latest_version=latest["version"]
        )

    # Find patch from current to latest
    patch_name = f"{current['version']}_to_{latest['version']}"
    patch_path = PATCHES_DIR / f"{patch_name}.tflp"

    if not patch_path.exists():
        return UpdateCheckResponse(
            update_available=True,
            current_version=current["version"],
            latest_version=latest["version"],
            patch_url=None  # Patch not pre-generated
        )

    return UpdateCheckResponse(
        update_available=True,
        current_version=current["version"],
        latest_version=latest["version"],
        patch_url=f"/patch/{patch_name}",
        patch_size=patch_path.stat().st_size,
        expected_hash=get_file_hash(Path(latest["path"]))
    )


@app.get("/patch/{patch_id}")
async def download_patch(patch_id: str, request: Request):
    """Download a patch file with range request support."""
    patch_path = PATCHES_DIR / f"{patch_id}.tflp"

    if not patch_path.exists():
        raise HTTPException(status_code=404, detail="Patch not found")

    file_size = patch_path.stat().st_size

    # Check for range request (resume support)
    range_header = request.headers.get("range")

    if range_header:
        # Parse range header
        try:
            range_spec = range_header.replace("bytes=", "")
            start, end = range_spec.split("-")
            start = int(start) if start else 0
            end = int(end) if end else file_size - 1
        except ValueError:
            raise HTTPException(status_code=416, detail="Invalid range")

        if start >= file_size:
            raise HTTPException(status_code=416, detail="Range not satisfiable")

        # Return partial content
        def iter_file():
            with open(patch_path, 'rb') as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk_size = min(8192, remaining)
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(
            iter_file(),
            status_code=206,
            media_type="application/octet-stream",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Content-Length": str(end - start + 1),
                "Accept-Ranges": "bytes"
            }
        )

    # Return full file
    return FileResponse(
        patch_path,
        media_type="application/octet-stream",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size)
        }
    )


@app.get("/fingerprint/{model_name}", response_model=FingerprintResponse)
async def get_fingerprint(model_name: str):
    """Get fingerprint of a model by name."""
    model_path = MODELS_DIR / f"{model_name}.tflite"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    fp = fingerprint_model(model_path)
    return FingerprintResponse(**fp)


@app.post("/create-patch")
async def create_patch(source_version: str, target_version: str):
    """Create a patch between two model versions (admin endpoint)."""
    if not MALLORN_AVAILABLE:
        raise HTTPException(status_code=501, detail="mallorn not installed")

    source_path = MODELS_DIR / f"{source_version}.tflite"
    target_path = MODELS_DIR / f"{target_version}.tflite"

    if not source_path.exists() or not target_path.exists():
        raise HTTPException(status_code=404, detail="Model version not found")

    patch_name = f"{source_version}_to_{target_version}"
    patch_path = PATCHES_DIR / f"{patch_name}.tflp"

    try:
        stats = mallorn.create_patch(
            str(source_path),
            str(target_path),
            str(patch_path),
            compression_level=9,
            neural=True
        )

        return {
            "patch_id": patch_name,
            "patch_size": stats.patch_size,
            "compression_ratio": stats.compression_ratio,
            "url": f"/patch/{patch_name}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "mallorn_available": MALLORN_AVAILABLE,
        "models_count": len(VERSION_DB),
        "patches_count": len(list(PATCHES_DIR.glob("*.tflp")))
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
