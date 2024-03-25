#!/bin/bash

source "$(dirname "$0")/../activate"

model_name="$1"

model_dir="$project_root/models/$model_name"
class_names_path="$model_dir/classes.txt"
dataset_path="$project_root/images/$model_name"
save_path="$model_dir/$model_name.h5"
log="$project_root/logs/train_jizbirds-$(date -Iminutes).log"

python3 -u "$project_root/training/train_keras_classification.py" -d -a -t "$model_name" "$class_names_path" "$dataset_path" "$save_path" | tee "$log"
