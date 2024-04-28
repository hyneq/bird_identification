#!/bin/bash

source "$(dirname "$0")/../vars"

mediamtx_version="v1.8.0"
mediamtx_system="$(uname -s | tr '[:upper:]' '[:lower:]')"
mediamtx_arch="$(uname -m | sed s/aarch64/arm64v8/ | sed s/x86_64/amd64/)"

mediamtx_download_url="https://github.com/bluenviron/mediamtx/releases/download/$mediamtx_version/mediamtx_${mediamtx_version}_${mediamtx_system}_${mediamtx_arch}.tar.gz"

curl -L "$mediamtx_download_url" | tar -xvzf - mediamtx -C "$project_root"
