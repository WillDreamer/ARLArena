#!/bin/bash

# PID=4026931
# CHECK_INTERVAL=30

# echo "Waiting for process $PID to finish..."

# while kill -0 $PID 2>/dev/null; do
#     sleep $CHECK_INTERVAL
# done

# echo "Process finished. Sleeping 5 minutes..."
# sleep 5m

sleep 11h


echo "Running next job..."
bash ./examples/shop_agent_trainer/Qwen3_4B/train_sapo.sh