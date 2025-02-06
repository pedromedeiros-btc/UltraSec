#!/bin/bash

# Directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Wait for tunnel to be ready
echo "Waiting for tunnel to door controller..."
for i in {1..30}; do
    if curl -s http://localhost:8082 > /dev/null; then
        echo "Tunnel is ready!"
        break
    fi
    echo "Waiting for tunnel... ($i/30)"
    sleep 1
done

# Start door proxy
echo "Starting door proxy..."
python3 $DIR/door_proxy.py > door_proxy.log 2>&1 &

# Wait a bit for door proxy to start
sleep 5

# Start face recognition system
echo "Starting face recognition system..."
python3 $DIR/web_app.py > web_app.log 2>&1 