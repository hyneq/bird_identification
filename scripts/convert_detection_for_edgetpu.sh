#!/bin/bash

# Converts a Keras Model to EdgeTPU-compatible TF Lite model

source "$(dirname "$0")/../activate"

model_name="$1"
model_path="$project_root/models/$model_name"
model_dataset="$project_root/images/$model_name/images"

# Make batch size fixed
input_shape="640 640 3"
"$project_root/training/keras_fix_input_shape.py" "$model_path/model.keras" "$model_path/model_fixedinputshape.keras" -s $input_shape

# Convert from Keras to quantized TF Lite model
quantization="float_fallback_quant"
"$project_root/training/keras_to_tflite.py" "$model_path/model_fixedinputshape.keras" "$model_path/model.tflite" -o "$quantization" -d "$model_dataset"

# Compile the TF Lite model for Edge TPU
"$project_root/edgetpu_compiler/edgetpu_compiler" -o "$model_path" "$model_path/model.tflite"
