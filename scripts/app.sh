#!/bin/bash

source "$(dirname "$0")/../activate"

source "$project_root/config/vars"

test -z "$rtsp_path" && rtsp_path="rtsp://localhost:$rtsp_port/prediction"


python3 cli/app.py \
    --width "$camera_width" --height "$camera_height" \
    --out-type ffmpeg_rtsp -o "$rtsp_path" \
    --logger-type sqlalchemy --log-path "$db_path" \
    @"$app_args_file"
