#!/bin/bash

source "$(dirname "$0")/../activate"

source "$project_root/config/vars"

cd "$project_root"

exec python3 cli/logged_objects_upload.py \
    "$db_path" \
    "$remote_api_url"
    @"$remote_api_credentials"
