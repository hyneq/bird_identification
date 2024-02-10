#!/bin/bash

birds_file="$(dirname $0)/models/BIRDS_450/birds.csv"
classes_file="$(dirname $0)/models/BIRDS_450/classes.csv"

cat ${birds_file} | tail -n +2 | cut -d, -f1,3 | sort -n | uniq > ${classes_file}
