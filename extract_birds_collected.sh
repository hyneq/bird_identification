#!/bin/bash

batch="$1"

project_root="$(dirname $0)"
images=$project_root/images
log=$project_root/logs/extract_birds_collected-$batch-$(date -Iminutes)

python3 -u $project_root/cli/detection_file_extractor.py --class=bird $images/collected/$batch $images/collected_extracted/$batch | tee $log
