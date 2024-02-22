#!/bin/bash

source "$(dirname "$0")/vars"

# Create the Python virtual environment and activate
test -z "$SETUP_NO_CREATE_VENV" && python3 -m venv "$venv" && source "$project_root/activate" || exit $?

# Install Python packages
if [[ -z "$SETUP_NO_INSTALL_PACKAGES" ]]; then

    for mode in common "$SETUP_MODE"; do

        for platform in "" "-$SETUP_PLATFORM"; do

            req_path="$req_dir/$mode$platform.txt"

            test -f "$req_path" && pip3 install -r "$req_path"

        done

    done

fi
