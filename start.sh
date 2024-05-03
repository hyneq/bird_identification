#!/bin/bash

# Starts the application along with mediamtx

# Load variables and activate venv
source "$(dirname "$0")"/activate

# Load config vars
source "$project_root/config/vars"

# Create log directory
mkdir -p "$logs_dir"

# Enable job control
set -m

# Run with a strategy dependent on mode
mode="$1"
test -z "$mode" && mode="normal"
case "$mode" in
    "nohup")
        # Run start_server.sh out of session
        export APP_STORE_PID=1
        nohup "$0" normal &>"$logs_dir/app.out" &
        ;;

    "normal")
        # Start mediamtx, if not explicitly disabled
        if [[ -z "$NO_MEDIAMTX" ]]; then
            "$project_root/scripts/mediamtx_publisher.sh" &
            mediamtx_pid=$!
        fi

        if [[ -n "$APP_STORE_PID" ]]; then
            # Start app in the background
            "$project_root/scripts/app.sh" &
            app_pid=$!

            # Store the PID
            echo $app_pid > "$pid_file"

            # Bring app to the foreground (blocking call)
            fg

            # Remove the PID
            rm -f "$pid_file"
        else
            # Start app in the foreground
            "$project_root/scripts/app.sh"
        fi
        

        # Wait for mediamtx, if running
        if [[ -n "$mediamtx_pid" ]]; then
            kill $mediamtx_pid 2>/dev/null &&
            wait $mediamtx_pid || true
        fi

        ;;

    *)
        echo "$0: The first argument must be 'normal', 'nohup' or omitted for default ('apache')" > /dev/stderr
        exit 1

esac
