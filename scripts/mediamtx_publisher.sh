#!/bin/bash

source "$(dirname "$0")/../vars"

export MTX_RTMP=no
export MTX_HLS=no
export MTX_SRT=no

export MTX_PATHS_PREDICTION_SOURCE=publisher

exec $project_root/mediamtx
