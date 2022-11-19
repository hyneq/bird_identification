#!/bin/bash

project_root="$(dirname $0)"
model_name=jizbirds
dataset_path=$project_root/images/jizbirds
save_path=$project_root/models/jizbirds.h5
log=$project_root/logs/train_jizbirds-$(date -Iminutes)

python3 -u $project_root/train.py -vv -d -a -t $model_name $dataset_path $save_path | tee $log