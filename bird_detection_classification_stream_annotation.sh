#!/bin/bash

workdir="$(realpath $(dirname $0))"
file_in="$1"
file_out="$2"

if [ -n "PYTHONPATH" ]; then
    export PYTHONPATH=$workdir:$PYTHONPATH
else
    export PYTHONPATH=$workdir
fi

exec python3 $workdir/cli/detection_classification_annotation.py --detection-class="bird" --detection-min-confidence=60 --classification-min-confidence=60 -i $file_in -o $file_out
