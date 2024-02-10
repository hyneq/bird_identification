#!/bin/bash

workdir="$(dirname $0)"

in_dir=$workdir/images/stream_extractions
out_dir=$workdir/images/stream_extractions_annotated

for in_path in $in_dir/*.mp4; do
    $workdir/bird_detection_classification_stream_annotation.sh $in_path $out_dir/$(basename $in_path)
done
