#!/bin/bash

# Compresses model's dataset and class names in order to be able to extract in a different environment for training.

source "$(dirname "$0")/../activate"

model_name="$1"
archive="archives/$model_name.zip"
model_class_names="models/$model_name/classes.txt"
model_dataset="images/$model_name"

cd $project_root
zip -r "$archive" "$model_class_names" "$model_dataset"
