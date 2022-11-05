#!/bin/bash

project_root="$(dirname $0)"
images=$project_root/images

$project_root/detection_extract.py $images/'cam_trap/*/*.JPG' 'bird' $images/'cam_trap_extracted/$i.jpg'