#!/bin/bash
# Initializes keras-YOLOv3-model-set submodule and sets up its own Python venv
cd "$(dirname "$0")"

git submodule update --init keras-YOLOv3-model-set || exit

test -z "$KERAS_YOLOV3_NO_CREATE_VENV" && keras-YOLOv3-model-set/create_venv.sh
