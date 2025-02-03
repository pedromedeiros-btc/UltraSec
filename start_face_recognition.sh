#!/bin/bash

# Change to the project directory
cd /home/cloudwalkoffice/face_detection

# Kill any existing ngrok processes
pkill -f ngrok || true

# Start ngrok in the background
ngrok http 5000 > ngrok.log 2>&1 &

# Wait for ngrok to start (try multiple times)
max_attempts=12
attempt=1
while [ $attempt -le $max_attempts ]; do
    echo "Attempt $attempt to get ngrok URL..."
    sleep 5
    if curl -s http://localhost:4040/api/tunnels | grep -q "public_url"; then
        NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o '"public_url":"[^"]*' | grep -o '[^"]*$')
        echo "Ngrok URL: $NGROK_URL"
        echo $NGROK_URL > current_ngrok_url.txt
        break
    fi
    attempt=$((attempt + 1))
done

if [ -z "$NGROK_URL" ]; then
    echo "Failed to get ngrok URL after $max_attempts attempts"
    cat ngrok.log
fi

# Start the web app
source face_detection_env/bin/activate
python3 web_app.py 