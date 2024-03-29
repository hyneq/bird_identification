#!/bin/bash

source "$(dirname "$0")/../activate"

model_name="$1"

model_dir="$project_root/models/$model_name"
class_names_path="$model_dir/classes.txt"
dataset_path="$project_root/images/$model_name"
dataset_annots_path="$dataset_path/annotations"
dataset_imgs_path="$dataset_path/images"
save_path="$model_dir/model.keras"
log="$project_root/logs/train_$model_name-$(date -Iminutes).log"

python3 -u "$project_root/training/train_keras_yolo8.py" -d "$class_names_path" "$dataset_annots_path" "$dataset_imgs_path" "$save_path" | tee "$log"
