# Mallorn Python Bindings

Edge model delta updates for TFLite, GGUF, and ONNX models.

## Installation

```bash
pip install mallorn
```

## Quick Start

```python
import mallorn

# Create a delta patch between two model versions
patch = mallorn.diff("model_v1.tflite", "model_v2.tflite")
patch.save("model.patch")

# Apply a patch to update a model
mallorn.patch("model_v1.tflite", "model.patch", "model_v2.tflite")
```

## Supported Formats

- **TFLite** - TensorFlow Lite models (.tflite)
- **GGUF** - GGML Unified Format models (.gguf)
- **ONNX** - Open Neural Network Exchange models (.onnx)

## Features

- 70-95% smaller than full model downloads
- Streaming patch application for embedded systems
- Cryptographic signatures for secure OTA updates
- Cross-platform: Linux, macOS, Windows

## License

MIT OR Apache-2.0
