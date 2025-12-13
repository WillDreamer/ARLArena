#!/bin/bash

# List the PIDs you want to wait for
PIDS=(1632301 1632302 1632303 1632304)

echo "Waiting for PIDs to terminate: ${PIDS[@]}"

while true; do
    alive=()

    # Check which PIDs are still running
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            alive+=("$pid")
        fi
    done

    # If none left → break
    if [ ${#alive[@]} -eq 0 ]; then
        echo "All PIDs have terminated."
        break
    fi

    echo "Still running: ${alive[@]} ... checking again in 5s"
    sleep 100
done

# ----------------------------
# START NEXT COMMAND HERE
# ----------------------------
echo "Starting next job..."
bash /data1/xw27/agent/ARLArena/examples/math_trainer/train_grpo_s_cispo.sh
