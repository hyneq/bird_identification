#!/bin/bash

source "$(dirname "$0")/../vars"

export MTX_PATHS_PREDICTION_SOURCE=publisher

exec $project_root/mediamtx
