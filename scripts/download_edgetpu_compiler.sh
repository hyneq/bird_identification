#!/bin/bash

# This script downloads edgetpu_compiler version 15.0
# to overcome bugs in version 16.0
# see https://github.com/google-coral/edgetpu/issues/480

source "$(dirname "$0")/../vars"
scripts_dir="$workdir/scripts"

download_url=https://github.com/google-coral/edgetpu/files/7278734/compiler.zip
archive_location=/tmp/edgetpu-compiler.zip

wget -O "$archive_location" "$download_url" &&
unzip "$archive_location" 'compiler/*' -d "$project_root"
mv "$project_root/compiler" "$project_root/edgetpu_compiler"
