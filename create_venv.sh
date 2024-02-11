#!/bin/bash

source "$(dirname "$0")/vars"

# Create the Python virtual environment and activate
python3 -m venv "$venv" && source "$project_root/activate" || exit $?

# Install common Python packages
pip install -r "$project_root/requirements.txt"

# Install Python packages for deployment, if enabled
test -n "$INSTALL_DEPLOY_DEPENDENCIES" && pip3 install -r "$workdir/requirements_deploy.txt"

# Install Python packages for testing, if enabled
test -n "$INSTALL_TEST_DEPENDENCIES" && pip3 install -r "$workdir/requirements_test.txt"
