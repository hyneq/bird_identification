#!/bin/bash

model_name=jizbirds

project_root="$(dirname $0)"
model_dir=$project_root/models/$model_name
class_names_path=$model_dir/classes.csv
dataset_path=$project_root/images/jizbirds
save_path=$model_dir/$model_name.h5
log=$project_root/logs/train_jizbirds-$(date -Iminutes).log

python -u $project_root/train.py -vv -d -a -t $model_name $class_names_path $dataset_path $save_path | tee $log