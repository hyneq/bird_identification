#!/bin/bash

classes="$1"
base_dir="$2"

mkdir -p $base_dir

while read -r class_name; do
    mkdir $base_dir/"$class_name"
done < $classes