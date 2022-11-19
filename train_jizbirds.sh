#!/bin/bash

project_root="$(dirname $0)"
model_name=jizbirds
model_dir=$project_root/models/$model_name
class_names_path=$model_dir/classes.csv
dataset_path=$project_root/images/jizbirds
save_path=$model_dir/$model_name.h5
log=$project_root/logs/train_jizbirds-$(date -Iminutes)

python3 -u $project_root/train.py -vv -d -a -t $model_name $class_names_path $dataset_path $save_path | tee $log