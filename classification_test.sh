#!/bin/bash

project_root="$(dirname $0)"

test_dir=$project_root/images/classification_test

log=$project_root/logs/test_classification-$(date -Iminutes).log

python3 $project_root/classification_test.py $test_dir | grep --line-buffered -v "\[\=" | tee $log