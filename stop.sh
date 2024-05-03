#!/bin/bash

# based on https://github.com/hyneq/STINBankSystem/blob/main/stop_server.sh

# Stops application if running

source "$(dirname "$0")/activate"

source "$project_root/config/vars"

# Exit if no pid_file is present
if [[ ! -f "$pid_file" ]]; then
    echo "Server for $app_name not running, cannot stop" >&2
    exit 1
fi

# Kill the app's process
# based on https://stackoverflow.com/a/5789674
pid=$(cat "$pid_file")
kill -SIGINT $pid
