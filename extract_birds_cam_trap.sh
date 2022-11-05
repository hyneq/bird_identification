#!/bin/bash

project_root="$(dirname $0)"
images=$project_root/images
log=$project_root/logs/extract_birds_cam_trap-$(date -Iminutes)

python3 -u $project_root/detection_extract.py $images/'cam_trap/*/*.JPG' 'bird' $images/'cam_trap_extracted/$i.jpg' | tee $log