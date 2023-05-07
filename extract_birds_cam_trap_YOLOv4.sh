#!/bin/bash

batch="$1"

project_root="$(dirname $0)"
images=$project_root/images
log=$project_root/logs/extract_birds_cam_trap-$batch-$(date -Iminutes)
models=$project_root/models

python3 -u $project_root/cli/detection_file_extractor.py  --model-type="darknet-YOLOv4" --class=bird $images/cam_trap/$batch $images/cam_trap_extracted_YOLOv4/$batch | tee $log