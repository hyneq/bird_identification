#!/bin/bash
# Evaluates the model using ultralytics

source "$(dirname "$0")/../activate"

"$project_root/scripts/yolo" val model=yolov8s_int8.tflite data="$project_root/images/yolo-eval.yml" classes=[14]
