#!/bin/bash

batch="$1"

project_root="$(dirname $0)"
images=$project_root/images
classes_path=$project_root/models/czbirds/classes.csv
log=$project_root/logs/extract_birds_cam_trap-$batch-$(date -Iminutes)

source_path=$images/'cam_trap/'$batch'/*/*.JPG'
target_path_base=$images/'cam_trap_extracted_classes'
target_path=$target_path_base'/$c/$i.jpg'

mkdir -p $target_path
$project_root/create_class_dirs.sh $classes_path $target_path_base
mkdir $target_path/__not_recognized__

python3 -u $project_root/detection_extract.py $images/'cam_trap/'$batch'/*/*.JPG' 'bird' $target_path | tee $log