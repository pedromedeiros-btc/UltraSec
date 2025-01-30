#!/bin/bash

# Change to the project directory
cd /home/cloudwalkoffice/face_detection

# Start ngrok in the background
/usr/local/bin/ngrok http 5000 > /dev/null 2>&1 &

# Start the web app
python3 web_app.py 