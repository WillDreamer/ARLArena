#!/bin/bash

PID=1024441
CHECK_INTERVAL=30

echo "Waiting for process $PID to finish..."

while kill -0 $PID 2>/dev/null; do
    sleep $CHECK_INTERVAL
done

echo "Process finished. Sleeping 5 minutes..."
sleep 5m

echo "Stopping Ray..."
ray stop

echo "Sleeping 1 minute before starting next job..."
sleep 1m

echo "Running next job..."
bash ./examples/embody_agent_trainer/train_alfworld_cispo.sh