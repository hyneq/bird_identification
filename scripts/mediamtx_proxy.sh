#!/bin/bash

source "$(dirname "$0")/../vars"

export MTX_PATHS_PREDICTION_SOURCE="$REMOTE_STREAM_PATH"
export MTX_PATHS_PREDICTION_SOURCEONDEMAND=yes

exec $project_root/mediamtx
