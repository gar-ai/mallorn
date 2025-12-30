#!/bin/bash
# Download test models for Mallorn testing
set -e

FIXTURE_DIR="tests/fixtures/models"
mkdir -p "$FIXTURE_DIR"

echo "Downloading TFLite test models..."

# MobileNet V1 Float16 (~4MB) - from TensorFlow official models
if [ ! -f "$FIXTURE_DIR/mobilenet_v1.tflite" ]; then
    echo "  Downloading MobileNet V1..."
    curl -L -o "$FIXTURE_DIR/mobilenet_v1.tflite" \
        "https://www.kaggle.com/api/v1/models/tensorflow/mobilenet-v1/tfLite/100-224-fp32/1/download" \
        || curl -L -o "$FIXTURE_DIR/mobilenet_v1.tflite" \
        "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_1.0_224/1/default/1?lite-format=tflite"
fi

# MobileNet V2 (~4MB)
if [ ! -f "$FIXTURE_DIR/mobilenet_v2.tflite" ]; then
    echo "  Downloading MobileNet V2..."
    curl -L -o "$FIXTURE_DIR/mobilenet_v2.tflite" \
        "https://www.kaggle.com/api/v1/models/tensorflow/mobilenet-v2/tfLite/100-224-fp32/1/download" \
        || curl -L -o "$FIXTURE_DIR/mobilenet_v2.tflite" \
        "https://tfhub.dev/tensorflow/lite-model/mobilenet_v2_1.0_224/1/default/1?lite-format=tflite"
fi

# MobileNet V1 Quantized (~1.4MB)
if [ ! -f "$FIXTURE_DIR/mobilenet_v1_quant.tflite" ]; then
    echo "  Downloading MobileNet V1 Quantized..."
    curl -L -o "$FIXTURE_DIR/mobilenet_v1_quant.tflite" \
        "https://www.kaggle.com/api/v1/models/tensorflow/mobilenet-v1/tfLite/100-224-int8/1/download" \
        || curl -L -o "$FIXTURE_DIR/mobilenet_v1_quant.tflite" \
        "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_1.0_224_quantized/1/default/1?lite-format=tflite"
fi

echo ""
echo "TFLite models downloaded to $FIXTURE_DIR"
echo ""
echo "Note: GGUF models are large. Download manually from HuggingFace if needed:"
echo "  - TinyLlama: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
echo ""
ls -lh "$FIXTURE_DIR"
